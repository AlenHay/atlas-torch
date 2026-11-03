"""XLINOP (dfsynthe.for): NLTE-list lines (TAPE19) added to the opacity buffer, REAL*4 exact.

Line types: normal (TYPE 0, 3 (PRD), < -2), merged continua (TYPE = NLAST > 3), autoionizing (1),
hydrogen (-1, -2; HPROF4 family, every 20th point computed and log-interpolated), coronal (2, skipped).
All layers are processed together.  The wavelength chains (WAVE=WAVE*RATIO: REAL*8 sequential products,
identical for all layers of a line) are built on CPU with numpy ufunc.accumulate (sequential, exact).
Contributions are generated per chunk of TAPE19 lines, stably ordered by line (= per-bin Fortran order) and
added with a sequential 1-D index_add_.

SOFBET cache state (see hydrogen.py) flows through all hydrogen-line calls in Fortran order (layer J outer,
lines, wing points, HPROF4 before HPROF4m/p).  A dry run over the hydrogen lines determines, for every layer,
the state left by the previous layers (iterated to self-consistency); the real pass uses those carry-ins.
"""
from __future__ import annotations

import numpy as np
import torch

from ...tables import table
from .f32 import F32, F64, I64, Flavor, lit, rdiv
from .hydrogen import Hydrogen, hfnm
from .tables import IXWLBEG, LENBUFF, LENGTH, RATIO, RATIOLG, WBEGIN, WLBEG, WLEND

SRC = "dfsynthe.for"
L = lambda s: float(lit(s))   # noqa: E731
f4 = np.float32
C17 = float(lit("2.99792458E17"))
E7 = float(lit("1.E7"))


def ehyd_tables():
    e = np.zeros(100)
    for i, s in enumerate(("0.", "82259.105", "97492.302", "102823.893", "105291.651", "106632.160",
                           "107440.444", "107965.051")):
        e[i] = float(lit(s))
    n = np.arange(9, 101)
    e[8:] = 109678.764 - 109677.576 / (n * n).astype(np.float64)
    alpha = 1.0e7 / (e[1:] - e[:-1])
    return e, alpha


def voigt(v: torch.Tensor, a: torch.Tensor, tabs: dict) -> torch.Tensor:
    """FUNCTION VOIGT(V,A) (REAL*4), elementwise."""
    h0, h1, h2, atab = tabs["h0tab"], tabs["h1tab"], tabs["h2tab"], tabs["atab"]
    small = a < L(".2")
    iv = torch.trunc(v * 200.0 + 1.5).to(I64) - 1
    ivs = iv.clamp(0, h0.numel() - 1)
    r_small = torch.where(v > 10.0, (L(".5642") * a) / (v * v), (h2[ivs] * a + h1[ivs]) * a + h0[ivs])
    asym = (a > L("1.4")) | ((a + v) > L("3.2"))
    aa = a * a
    vv = v * v
    u = (aa + vv) * L("1.4142")
    r0 = (a * L(".79788")) / u
    aau = aa / u
    vvu = vv / u
    r1 = r0 * (((((aau - 10.0 * vvu) * aau + 5.0 * vvu * vvu) / u + vvu - aau / 3.0) * 3.0) / u + 1.0)
    r_asym = torch.where(a > 100.0, r0, r1)
    ia = (torch.trunc(a * 200.0 + 1.5).to(I64) - 1).clamp(0, atab.shape[0] - 1)
    r_tab = atab[ia, iv.clamp(0, atab.shape[1] - 1)]
    return torch.where(small, r_small, torch.where(asym, r_asym, r_tab))


def chain(w0: np.ndarray, n: int, up: bool) -> np.ndarray:
    """[len(w0), n] = w0, w0*R, (w0*R)*R, ... (or /R), REAL*8 sequential (ufunc.accumulate is sequential)."""
    m = np.empty((len(w0), n))
    m[:, 0] = w0
    m[:, 1:] = RATIO
    return (np.multiply if up else np.divide).accumulate(m, axis=1)


class XLinop:
    def __init__(self, tabs: dict, lay: dict, cont: torch.Tensor, fl: Flavor, block: int = 256):
        self.tabs, self.fl, self.cont = tabs, fl, cont
        self.dev = cont.device
        self.nl = cont.shape[0]
        st = lay["st"]
        self.xne, self.txnxn = st["xne"], lay["txnxn"]
        self.xnfpel, self.xnfdop, self.dopple = lay["xnfpel"], lay["xnfdop"], lay["dopple"]
        inglis = rdiv(1600.0, fl.powf(self.xne, torch.tensor(float(f4(f4(2.0) / f4(15.0))), dtype=F32)))
        nmerge = inglis - 1.5
        self.emerge = rdiv(L("109737.312"), nmerge * nmerge).to(F64)
        self.emergeh = rdiv(L("109677.576"), nmerge * nmerge).to(F64)
        self.ehyd, self.alphahyd = ehyd_tables()
        self.contx = np.asarray(table(SRC, "xlinop", "contx"), dtype=np.float64)     # (26,17)
        self.conth = np.asarray(table(SRC, "xlinop", "conth"), dtype=np.float64)
        self.hyd = Hydrogen(st, lay["xnfph"], tabs["extab"], tabs["h0tab"], fl)
        self.block = block
        self.stats = {"normal_items": 0, "h_samples": 0, "contribs": 0, "h_reruns": 0}

    def wave_start(self, nb) -> np.ndarray:
        """WBEGIN*RATIO**(NB-1) as compiled."""
        return WBEGIN * self.fl.powi8_np(RATIO, np.asarray(nb, np.int64) - 1)

    # ================================================================== driver
    def run(self, buffer_flat: torch.Tensor, tape19: np.ndarray, chunk_lines: int = 200):
        """Add all TAPE19 lines to buffer_flat (which must be zero: XLINOP is the first contributor)."""
        typ = tape19["type"]
        hidx = np.nonzero((typ == -1) | (typ == -2))[0]
        if not len(hidx):
            return self._pass(buffer_flat, tape19, None, chunk_lines)
        # SOFBET carry-ins: a dry run of the hydrogen lines with fresh caches gives the per-layer call sequences,
        # hence the carry-ins; the real pass records its own sequences and must reproduce them.
        nl, dev = self.nl, self.dev
        dry = self._h_state(torch.zeros(nl, dtype=I64, device=dev), torch.full((nl,), float("nan"), device=dev,
                                                                                dtype=F32))
        for k in hidx:
            self.h_line(tape19[k], int(k), dry, emit=False)
        t_idx, t_p = self._true_carry(dry)
        for _ in range(nl + 1):
            st = self._h_state(t_idx, t_p)
            buffer_flat.zero_()
            self._pass(buffer_flat, tape19, st, chunk_lines)
            n_idx, n_p = self._true_carry(st)
            f = st["first"]
            dep_used = (f != 0) & (f == t_idx)
            dep_true = (f != 0) & (f == n_idx)
            same_p = (n_p == t_p) | (torch.isnan(n_p) & torch.isnan(t_p))
            if not bool(((dep_true != dep_used) | (dep_true & ~same_p)).any()):
                return self.stats
            self.stats["h_reruns"] += 1
            t_idx, t_p = n_idx, n_p
        raise RuntimeError("SOFBET carry-in iteration did not converge")

    def _h_state(self, idx, p):
        nl, dev = self.nl, self.dev
        return {"idx": idx.clone(), "p": p.clone(), "first": torch.zeros(nl, dtype=I64, device=dev),
                "change": torch.zeros(nl, dtype=torch.bool, device=dev), "layers": torch.arange(nl, device=dev),
                "idx_in": idx.clone(), "p_in": p.clone()}

    def _true_carry(self, res):
        """Carry-ins implied by per-layer hydrogen call sequences (first INDEX, internal INDEX change, last
        INDEX): layer 1 starts fresh, layer j+1 starts with layer j's end state."""
        nl, dev, pp = self.nl, self.dev, self.hyd.pp
        t_idx = torch.zeros(nl, dtype=I64, device=dev)
        t_p = torch.full((nl,), float("nan"), dtype=F32, device=dev)
        first, change, last = res["first"].tolist(), res["change"].tolist(), res["idx"].tolist()
        for j in range(1, nl):                      # scalar scan over the 25 layers
            q = j - 1
            if first[q] == 0:
                t_idx[j], t_p[j] = t_idx[q], t_p[q]
                continue
            event = (first[q] != int(t_idx[q])) or change[q]
            t_idx[j] = last[q]
            t_p[j] = pp[q] if event else t_p[q]
        return t_idx, t_p

    def _pass(self, buffer_flat, tape19, state, chunk_lines):
        typ = tape19["type"]
        n = len(tape19)
        for s in range(0, n, chunk_lines):
            e = min(n, s + chunk_lines)
            ids = np.arange(s, e)
            t = typ[s:e]
            parts = []
            norm = (t == 0) | (t == 3) | (t < -2)
            if norm.any():
                parts += self.normal_lines(tape19, ids[norm])
            if (t > 3).any():
                parts += self.merged_lines(tape19, ids[t > 3])
            if (t == 1).any():
                parts += self.autoion_lines(tape19, ids[t == 1])
            for k in ids[(t == -1) | (t == -2)]:
                parts += self.h_line(tape19[k], int(k), state, emit=True)
            if not parts:
                continue
            li = torch.cat([p[0].to(torch.int32) for p in parts])
            order = torch.sort(li, stable=True).indices
            del li
            fi = torch.cat([p[1] for p in parts])[order]
            va = torch.cat([p[2] for p in parts])[order]
            del parts, order
            buffer_flat.index_add_(0, fi, va)
            self.stats["contribs"] += int(va.numel())
            del fi, va
        return self.stats

    # ================================================================== normal lines
    def normal_lines(self, tape19, ids):
        dev = self.dev
        r = tape19[ids]
        nb = torch.from_numpy(r["nbuff"].astype(np.int64)).to(dev)
        nel = torch.from_numpy(r["nelion"].astype(np.int64)).to(dev)
        gf = torch.from_numpy(r["gf"].copy()).to(dev)
        wl = torch.from_numpy(r["wl"].copy()).to(dev)
        kappa0 = gf[None, :] * self.xnfdop[:, nel - 1]
        cnb = self.cont[:, nb - 1]
        ok = ~(kappa0 < cnb)
        strong = kappa0 > 500.0 * cnb
        ncon = r["ncon"].astype(np.int64)
        nelx = r["nelionx"].astype(np.int64)
        cxv = np.where(ncon > 0, self.contx[np.clip(ncon, 1, 26) - 1, np.clip(nelx, 1, 17) - 1], 0.0)
        cx = torch.from_numpy(cxv).to(dev)[None, :]
        has_con = torch.from_numpy(ncon > 0).to(dev)[None, :]
        zero = torch.zeros(1, dtype=F64, device=dev)
        wcon = torch.where(has_con, rdiv(E7, cx - self.emerge[:, None]), zero)
        wtail = torch.where(has_con, rdiv(E7, (cx - self.emerge[:, None]) - 500.0), zero)
        gr = torch.from_numpy(r["gammar"].copy()).to(dev)
        gs = torch.from_numpy(r["gammas"].copy()).to(dev)
        gw = torch.from_numpy(r["gammaw"].copy()).to(dev)
        dop = self.dopple[:, nel - 1]
        adamp = ((gr[None, :] + gs[None, :] * self.xne[:, None]) + gw[None, :] * self.txnxn[:, None]) / dop
        dopwl = (dop.to(F64) * wl[None, :]).to(F32)
        jj, ii = torch.nonzero(ok, as_tuple=True)
        self.stats["normal_items"] += int(jj.numel())
        if jj.numel() == 0:
            return []
        it = dict(j=jj, i=ii, k0=kappa0[jj, ii], strong=strong[jj, ii], wcon=wcon[jj, ii], wtail=wtail[jj, ii],
                  a=adamp[jj, ii], dopwl=dopwl[jj, ii], wl=wl[ii], nb=nb[ii])
        lid = torch.from_numpy(ids.astype(np.int64)).to(dev)
        nb_np = r["nbuff"].astype(np.int64)
        out = []
        red_ok = ~(it["wl"] > WLEND)
        minred = torch.clamp(it["nb"], min=1)
        maxred = torch.where(it["strong"], torch.full_like(minred, LENGTH), torch.clamp(it["nb"] + 1000, max=LENGTH))
        cnt = torch.where(red_ok, (maxred - minred + 1).clamp(min=0), torch.zeros_like(minred))
        out += self._walk(it, lid, minred, cnt, +1, self.wave_start(np.maximum(nb_np, 1)))
        # 212: IF(MINRED.EQ.1)GO TO 900; IF(WL.LT.WLBEG)GO TO 900  (only after a red wing)
        blue_ok = (~red_ok) | ~((minred == 1) | (it["wl"] < WLBEG))
        ib0 = torch.clamp(it["nb"], max=LENGTH + 1)
        maxblue = torch.where(it["strong"], ib0 - 1, torch.clamp(ib0 - 1, max=1000))
        cnt = torch.where(blue_ok, maxblue.clamp(min=0), torch.zeros_like(maxblue))
        out += self._walk(it, lid, ib0 - 1, cnt, -1, self.wave_start(np.minimum(nb_np, LENGTH + 1)))
        return out

    def _walk(self, it, lid, start, count, step, w0_line):
        """Voigt wings of normal-line items.  All items advance in lock-step (offset k); the WAVE chain of a
        line is shared by its items.  red: chain value at offset k is WAVE(start+k); blue: WAVE(IBUFF0)/R**(k+1)."""
        dev = self.dev
        out = []
        S = 8
        ii = it["i"]
        ii_np = ii.cpu().numpy()
        act = torch.nonzero(count > 0)[:, 0]
        wcur = w0_line.copy()
        k = 0
        while act.numel():
            S = min(2 * S, self.block * 16)                 # adaptive block: 16, 32, ... samples
            offs = torch.arange(S, device=dev)
            lines = np.unique(ii_np[act.cpu().numpy()])
            ch = chain(wcur[lines], S + 1, step > 0)
            wcur[lines] = ch[:, S]
            chv = ch[:, :S] if step > 0 else ch[:, 1:]
            lmap = np.full(len(w0_line), -1, np.int64)
            lmap[lines] = np.arange(len(lines))
            wave = torch.from_numpy(np.ascontiguousarray(chv)).to(dev)[torch.from_numpy(lmap).to(dev)[ii[act]]]
            bins = start[act][:, None] + step * (k + offs)[None, :]
            valid = (k + offs)[None, :] < count[act][:, None]
            wcon = it["wcon"][act][:, None]
            skip = wave < wcon
            need = valid & ~skip
            jl = it["j"][act]
            n_i, n_k = torch.nonzero(need, as_tuple=True)
            kap = torch.full(need.shape, float("inf"), dtype=F32, device=dev)
            if n_i.numel():
                ai = act[n_i]
                wv = wave[n_i, n_k]
                vv = (torch.abs(wv - it["wl"][ai]) / it["dopwl"][ai].to(F64)).to(F32)
                kv = it["k0"][ai] * voigt(vv, it["a"][ai], self.tabs)
                wc, wt = it["wcon"][ai], it["wtail"][ai]
                kv = torch.where(wv < wt, ((kv.to(F64) * (wv - wc)) / (wt - wc)).to(F32), kv)
                kap[n_i, n_k] = kv
            cb = self.cont[jl[:, None], (bins - 1).clamp(0, LENGTH - 1)]
            term = need & (kap < cb)
            hit = term.any(dim=1)
            tpos = torch.where(hit, term.to(torch.int8).argmax(dim=1), torch.full_like(jl, S))
            use = need & (offs[None, :] <= tpos[:, None])
            e_i, e_k = torch.nonzero(use, as_tuple=True)
            if e_i.numel():
                out.append((lid[ii[act][e_i]], jl[e_i] * LENBUFF + (bins[e_i, e_k] - 1), kap[e_i, e_k]))
            more = (~hit) & ((k + S) < count[act])
            if step < 0:                                    # WAVE decreasing: once below WCON, all skipped
                more &= ~skip[:, -1]
            act = act[more]
            k += S
        return out

    # ================================================================== merged continua
    def merged_lines(self, tape19, ids):
        dev, fl = self.dev, self.fl
        out = []
        for lidx in ids:
            r = tape19[lidx]
            wl = float(r["wl"])
            nlast = int(r["type"])
            nel = int(r["nelion"])
            if nel == 1:
                wshift = 1.0e7 / (1.0e7 / wl - 109677.576 / float(nlast * nlast))
                wmerge = rdiv(1.0e7, 1.0e7 / wl - self.emergeh)
            else:
                wshift = 1.0e7 / (1.0e7 / wl - 109737.312 / float(nlast * nlast))
                wmerge = rdiv(1.0e7, 1.0e7 / wl - self.emerge)
            wmerge = torch.where(wmerge < 0, torch.full_like(wmerge, wshift + wshift), wmerge)
            wmerge = torch.clamp(wmerge, min=wshift)
            wmerge = torch.clamp(wmerge, max=wshift + wshift)
            wtail = rdiv(1.0e7, rdiv(1.0e7, wmerge) - 500.0)
            wtail = torch.where(wtail < 0, wmerge + wmerge, wtail)
            wtail = torch.minimum(wmerge + wmerge, wtail)
            lw = float(fl.log(torch.tensor([wl], dtype=F64))[0])
            ixwl = int(lw / RATIOLG)
            edgeblue = np.float32(float(fl.exp(torch.tensor([ixwl * RATIOLG], dtype=F64))[0]))
            if float(edgeblue) > wl:
                ixwl -= 1
            nb1 = ixwl + 1 - IXWLBEG + 1
            nb2 = torch.trunc(fl.log(wmerge) / RATIOLG + 0.5).to(I64) - IXWLBEG + 1
            nb3 = torch.trunc(fl.log(wtail) / RATIOLG + 0.5).to(I64) - IXWLBEG + 1
            keep = ~((nb1 > LENGTH) | (nb3 < 1))
            dn = (nb3 - nb2).to(F32)
            nb1c = max(nb1, 1)
            kap = torch.tensor(np.float32(r["gf"]), dtype=F32, device=dev) * self.xnfpel[:, nel - 1]
            hi = torch.clamp(nb3, max=LENGTH)
            cnt = torch.where(keep, (hi - nb1c + 1).clamp(min=0), torch.zeros_like(hi))
            tot = int(cnt.sum())
            if tot == 0:
                continue
            j = torch.repeat_interleave(torch.arange(self.nl, device=dev), cnt)
            first = torch.cumsum(cnt, 0) - cnt
            ib = nb1c + torch.arange(tot, device=dev) - torch.repeat_interleave(first, cnt)
            tail = torch.where(ib > nb2[j], (nb3[j] - ib).to(F32) / dn[j], torch.ones(1, dtype=F32, device=dev))
            out.append((torch.full((tot,), int(lidx), dtype=I64, device=dev), j * LENBUFF + (ib - 1), kap[j] * tail))
        return out

    # ================================================================== autoionizing
    def autoion_lines(self, tape19, ids):
        dev = self.dev
        out = []
        for lidx in ids:
            r = tape19[lidx]
            nel, nb = int(r["nelion"]), int(r["nbuff"])
            ashore, bshore, gammar = np.float32(r["gammas"]), np.float32(r["gammaw"]), np.float32(r["gammar"])
            g = np.float32(r["gf"])
            wl = float(r["wl"])
            kappa0 = (torch.tensor(f4(bshore * g), device=dev) * self.xnfpel[:, nel - 1])
            layers = torch.nonzero(~(kappa0 < self.cont[:, nb - 1]))[:, 0]
            if layers.numel() == 0:
                continue
            frelin = f4(C17 / wl)
            k0 = kappa0[layers]
            red = not (wl > WLEND)
            if red:
                minred = max(1, nb)
                f0 = f4(C17 / float(self.wave_start([minred])[0]))
                out += self._auto_walk(lidx, layers, k0, frelin, ashore, bshore, gammar, minred, LENGTH - minred + 1,
                                       +1, f0)
                if nb == 1:
                    continue
            ib0 = min(LENGTH + 1, nb)
            f0 = f4(C17 / float(self.wave_start([ib0])[0]))
            out += self._auto_walk(lidx, layers, k0, frelin, ashore, bshore, gammar, ib0 - 1, ib0 - 1, -1, f0)
        return out

    def _auto_walk(self, lidx, layers, k0, frelin, ashore, bshore, gammar, start, count, step, f0):
        dev = self.dev
        out = []
        S = 4096
        act = torch.arange(layers.numel(), device=dev)
        fcur = np.float32(f0)
        k = 0
        while act.numel() and k < count:
            n = min(S, count - k)
            fr = np.empty(n, np.float32)
            for q in range(n):                     # REAL*4 sequential chain FREQ=FREQ/RATIO (or *RATIO)
                if step > 0:
                    fr[q] = fcur
                    fcur = np.float32(float(fcur) / RATIO)
                else:
                    fcur = np.float32(float(fcur) * RATIO)
                    fr[q] = fcur
            frt = torch.from_numpy(fr).to(dev)[None, :]
            eps = (2.0 * (frt - float(frelin))) / float(gammar)
            kap = ((k0[act][:, None] * (float(ashore) * eps + float(bshore))) / (eps * eps + 1.0)) / float(bshore)
            bins = start + step * (k + torch.arange(n, device=dev))
            jl = layers[act]
            cb = self.cont[jl[:, None], (bins - 1)[None, :]]
            term = kap < cb
            hit = term.any(dim=1)
            tpos = torch.where(hit, term.to(torch.int8).argmax(dim=1), torch.full_like(jl, n))
            use = torch.arange(n, device=dev)[None, :] <= tpos[:, None]
            e_i, e_k = torch.nonzero(use, as_tuple=True)
            out.append((torch.full((e_i.numel(),), int(lidx), dtype=I64, device=dev),
                        jl[e_i] * LENBUFF + (bins[e_k] - 1), kap[e_i, e_k]))
            act = act[~hit]
            k += n
        return out

    # ================================================================== hydrogen
    def h_line(self, r, lidx, state, emit):
        """One hydrogen line for the layers in state['layers'] (all, in the real pass).  Updates the SOFBET
        state in place; returns contributions if emit."""
        dev, fl, hyd = self.dev, self.fl, self.hyd
        n, m = int(r["nblo"]), int(r["nbup"])
        wl = float(r["wl"])
        ncon, typ, nb = int(r["ncon"]), int(r["type"]), int(r["nbuff"])
        cgf = torch.tensor(np.float32(r["gf"]), device=dev)
        layers = state["layers"]
        kappa0 = cgf * self.xnfdop[layers, 0]
        keep = ~(kappa0 < self.cont[layers, nb - 1])
        layers, kappa0 = layers[keep], kappa0[keep]
        if layers.numel() == 0:
            return []
        dopph = self.dopple[layers, 0]
        if typ == -2:
            dopph = dopph / L("1.4142")
        path = "alpha" if (ncon == 0 or m == n + 1) else ("beta" if m == n + 2 else "general")
        E = self.ehyd
        g = dict(layers=layers, k0=kappa0, dopph=dopph, wl=wl, n=n, m=m, path=path, lidx=lidx)
        if path == "general":
            conth = self.conth[ncon - 1]
            wshift = 1.0e7 / (conth - 109677.576 / 81.0 ** 2)
            wmerge = rdiv(1.0e7, conth - self.emergeh[layers])
            wmerge = torch.where(wmerge < 0, torch.full_like(wmerge, wshift + wshift), wmerge)
            wcon = torch.clamp(wmerge, min=wshift)
            wtail = rdiv(1.0e7, rdiv(1.0e7, wcon) - 500.0)
            wcon = torch.clamp(wcon, max=wshift + wshift)
            wtail = torch.where(wtail < 0, wcon + wcon, wtail)
            wtail = torch.minimum(wcon + wcon, wtail)
            g.update(wcon=wcon, wtail=wtail)
            g["redcut"] = 1.0e7 / (109678.764 - 109677.576 / (m - 0.8) ** 2 - E[n - 1])
            g["wlminus1"] = 1.0e7 / (E[m - 2] - E[n - 1])
            g["wlminus2"] = 1.0e7 / (E[m - 3] - E[n - 1])
            g["k0red"] = ((((kappa0 * float(hfnm(n, m - 2, fl))) / float(hfnm(n, m, fl))).to(F64)
                           / (E[m - 3] - E[n - 1])) * (E[m - 1] - E[n - 1])).to(F32)
            g["bluecut"] = 1.0e7 / (109678.764 - 109677.576 / (m + 0.8) ** 2 - E[n - 1])
            g["wlplus1"] = 1.0e7 / (E[m] - E[n - 1])
            g["wlplus2"] = 1.0e7 / (E[m + 1] - E[n - 1])
            g["k0blue"] = ((((kappa0 * float(hfnm(n, m + 2, fl))) / float(hfnm(n, m, fl))).to(F64)
                            / (E[m + 1] - E[n - 1])) * (E[m - 1] - E[n - 1])).to(F32)
        out = []
        do_red = not (wl > WLEND)
        if path == "general" and do_red and WLBEG > self.alphahyd[n - 1]:
            do_red = False                                      # GO TO 613 directly
            skip_checks = True
        else:
            skip_checks = False
        minred = max(1, nb)
        if do_red:
            out += self._h_wing(g, state, emit, +1, minred, LENGTH - minred + 1)
            if minred == 1 or wl < WLBEG:          # label 612/622
                return out
        ib0 = min(LENGTH + 1, nb)
        out += self._h_wing(g, state, emit, -1, ib0, ib0 - 1)
        return out

    def _h_wing(self, g, state, emit, step, start, count):
        """Red (step +1: bins start.., IB counted over non-skipped bins) or blue (step -1: bins start-1..,
        I counted over all) wing of one hydrogen line for all its layers, in blocks of samples."""
        dev, hyd, fl = self.dev, self.hyd, self.fl
        layers = g["layers"]
        nlay = layers.numel()
        path, n, m, wl = g["path"], g["n"], g["m"], g["wl"]
        general = path == "general"
        w0 = float(self.wave_start([start])[0])        # WAVE(MINRED) (red) / WAVE(IBUFF0) (blue)
        out = []
        Bblk = 8                                    # samples per block (doubled every block)
        alive = torch.ones(nlay, dtype=torch.bool, device=dev)
        old = torch.zeros(nlay, dtype=F32, device=dev)
        # Precompute waves lazily in chunks of bins
        wave_cache = {"w": np.empty(0), "next": w0, "n": 0}

        def waves_upto(nbins):
            c = wave_cache
            if c["n"] >= nbins:
                return c["w"]
            need = max(nbins - c["n"], 4096)
            if step > 0:
                ch = chain(np.array([c["next"]]), need + 1, True)[0]
                new, c["next"] = ch[:need], ch[need]
            else:
                ch = chain(np.array([c["next"]]), need + 1, False)[0]
                new, c["next"] = ch[1:], ch[need]
            c["w"] = np.concatenate([c["w"], new]) if c["n"] else new
            c["n"] += need
            return c["w"]

        # skip prefix (red, general): number of bins with WAVE < WCON per layer
        if general and step > 0:
            wmax = float(g["wcon"].max())
            nb_ = 4096
            while True:
                w = waves_upto(min(nb_, max(count, 1)))
                if w[-1] >= wmax or len(w) >= count:
                    break
                nb_ *= 4
            s0 = torch.from_numpy(np.minimum(np.searchsorted(w, g["wcon"].cpu().numpy(), side="left"),
                                             count)).to(dev)
        else:
            s0 = torch.zeros(nlay, dtype=I64, device=dev)
        sample = 0                                  # sample counter (IB-1 / I-1)
        pp = hyd.pp[layers]
        sidx = state["idx"]; sp = state["p"]
        main_s = hyd.setup(n, m)
        sec_s = hyd.setup(n, m - 2) if (general and step > 0) else (hyd.setup(n, m + 2) if general else None)
        while bool(alive.any()):
            Bblk = min(2 * Bblk, 4096)
            # sample ordinals q = sample..sample+Bblk-1 -> counter value IB/I = q+1 -> bin offset
            q = torch.arange(sample, sample + Bblk, device=dev)
            cntr = q + 1
            off_c = torch.where(cntr <= 21, cntr, 21 + (cntr - 21) * 20)          # IB/I: 1..21, 41, 61, ...
            boff = s0[:, None] + (off_c[None, :] - 1)                              # bin offset from start
            nb_needed = int(boff.max()) + 2
            in_range = boff < count
            w_np = waves_upto(min(nb_needed, count) if count > 0 else 1)
            wave = torch.from_numpy(w_np[np.clip(boff.cpu().numpy(), 0, len(w_np) - 1)]).to(dev)
            # positional stop conditions (before the sample is evaluated)
            stop_pos = ~in_range
            if general and step > 0:
                stop_pos |= wave > g["wlminus1"]
            if general and step < 0:
                stop_pos |= (wave < g["wcon"][:, None]) | (wave < g["wlplus1"])
            if path == "beta" and step > 0:
                stop_pos |= wave > float(self.alphahyd[n - 1])
            # a stop at sample t kills t and everything after
            stop_pos = torch.cummax(stop_pos.to(torch.int8), dim=1).values.bool()
            nsmp = Bblk
            jl = layers[:, None].expand(nlay, nsmp)
            dl = (wave - wl)
            pm = hyd.prep(main_s, jl.reshape(-1), torch.full((nlay * nsmp,), wl, dtype=F64, device=dev),
                          dl.reshape(-1), g["dopph"][:, None].expand(nlay, nsmp).reshape(-1))
            if general:
                if step > 0:
                    use_sec = wave > g["redcut"]
                    dsec = wave - g["wlminus2"]
                else:
                    use_sec = wave < g["bluecut"]
                    dsec = wave - g["wlplus2"]
                ps = hyd.prep(sec_s, jl.reshape(-1), torch.full((nlay * nsmp,), wl, dtype=F64, device=dev),
                              dsec.reshape(-1), g["dopph"][:, None].expand(nlay, nsmp).reshape(-1))
            # ---- SOFBET call sequence: slots [main51, main4000, sec51, sec4000] per sample
            def calls(p, ok):
                c51 = (p["call51"].reshape(nlay, nsmp) & ok)
                c4 = (p["call4000"].reshape(nlay, nsmp) & ok)
                v51 = c51 & (p["beta"].reshape(nlay, nsmp) < 500.0)
                v4 = c4 & (p["beta4000"].reshape(nlay, nsmp) < 500.0)
                return v51, v4
            live = alive[:, None] & ~stop_pos
            v1, v2 = calls(pm, live)
            if general:
                v3, v4 = calls(ps, live & use_sec)
            else:
                v3 = v4 = torch.zeros_like(v1)
            valid = torch.stack([v1, v2, v3, v4], dim=2).reshape(nlay, nsmp * 4)
            ix_main, ix_sec = main_s["index"], (sec_s["index"] if sec_s is not None else 0)
            ixs = torch.tensor([ix_main, ix_main, ix_sec, ix_sec], device=dev).repeat(nsmp)[None, :].expand(nlay, -1)
            pos = torch.where(valid, torch.arange(nsmp * 4, device=dev)[None, :], -1)
            pos = torch.cummax(pos, dim=1).values
            prev_pos = torch.cat([torch.full((nlay, 1), -1, device=dev, dtype=I64), pos[:, :-1]], dim=1)
            idx0 = sidx[layers][:, None]
            prev_ix = torch.where(prev_pos >= 0, ixs.gather(1, prev_pos.clamp(min=0)), idx0.expand_as(prev_pos))
            event = valid & (ixs != prev_ix)
            cum_ev = torch.cummax(event.to(torch.int8), dim=1).values.bool()
            p_call = torch.where(cum_ev, pp[:, None], sp[layers][:, None])          # [nlay, nsmp*4]
            p_call = p_call.reshape(nlay, nsmp, 4)
            kap = g["k0"][:, None] * hyd.finish(pm, p_call[:, :, 0].reshape(-1), p_call[:, :, 1].reshape(-1)).reshape(nlay, nsmp)
            if general:
                tl = wave < g["wtail"][:, None]
                fac = lambda x: ((x.to(F64) * (wave - g["wcon"][:, None])) / (g["wtail"][:, None] - g["wcon"][:, None])).to(F32)  # noqa: E731
                kap = torch.where(tl, fac(kap), kap)
                kr = (g["k0red"] if step > 0 else g["k0blue"])[:, None] * \
                    hyd.finish(ps, p_call[:, :, 2].reshape(-1), p_call[:, :, 3].reshape(-1),
                               "m" if step > 0 else "p").reshape(nlay, nsmp)
                kr = torch.where(tl, fac(kr), kr)
                cut = use_sec & ((kr >= kap) if step > 0 else (kr > kap))
            else:
                cut = torch.zeros_like(live)
            # termination: cut (no add) or KAPPA < CONTINUUM(bin) (after add)
            binidx = self._bin_of(start, step, boff)
            cb = self.cont[layers[:, None], (binidx - 1).clamp(0, LENGTH - 1)]
            stop_cut = live & cut
            stop_cont = live & ~cut & (kap < cb)
            first_cut = torch.where(stop_cut.any(1), stop_cut.to(torch.int8).argmax(1), torch.full_like(layers, nsmp))
            first_cont = torch.where(stop_cont.any(1), stop_cont.to(torch.int8).argmax(1), torch.full_like(layers, nsmp))
            first_pos = torch.where(stop_pos.any(1), stop_pos.to(torch.int8).argmax(1), torch.full_like(layers, nsmp))
            ends = torch.minimum(torch.minimum(first_cut, first_cont), first_pos)
            # samples whose calls happened: < ends, plus the terminating sample if it is cut/cont
            sm = torch.arange(nsmp, device=dev)[None, :]
            called = live & ((sm < ends[:, None]) | ((sm == ends[:, None]) & ((first_cut == ends) | (first_cont == ends))[:, None]))
            added = live & ((sm < ends[:, None]) | ((sm == ends[:, None]) & (first_cont == ends)[:, None] & (first_cut > ends)[:, None]))
            # state update with the calls that happened
            vc = valid.reshape(nlay, nsmp, 4) & called[:, :, None]
            vflat = vc.reshape(nlay, -1)
            anyc = vflat.any(1)
            lastp = torch.where(vflat, torch.arange(nsmp * 4, device=dev)[None, :], -1).amax(1)
            ev_c = (event & vflat).any(1)
            if "first" in state:
                firstp = torch.where(vflat.any(1), vflat.to(torch.int8).argmax(1), 0)
                fresh = state["first"][layers] == 0
                setf = fresh & anyc
                ff = state["first"].clone()
                ff[layers[setf]] = ixs[setf, firstp[setf]] if setf.any() else ff[layers[setf]]
                # internal change: an event that is not the first call of the layer
                internal = event & vflat
                if bool(setf.any()):
                    internal[setf, firstp[setf]] = False
                state["change"][layers] |= internal.any(1)
                state["first"] = ff
            newidx = torch.where(anyc, ixs.gather(1, lastp.clamp(min=0)[:, None])[:, 0], sidx[layers])
            sidx[layers] = newidx
            sp[layers] = torch.where(ev_c, pp, sp[layers])
            self.stats["h_samples"] += int(called.sum())
            # contributions: sample points + log interpolation of the 19 skipped bins (counter > 21)
            if emit:
                newk = fl.logf(torch.clamp(kap, min=L("1.E-20")))
                prevk = torch.cat([old[:, None], newk[:, :-1]], dim=1)
                ai, ak = torch.nonzero(added, as_tuple=True)
                if ai.numel():
                    lid = torch.full((ai.numel(),), g["lidx"], dtype=I64, device=dev)
                    out.append((lid, layers[ai] * LENBUFF + (binidx[ai, ak] - 1), kap[ai, ak]))
                    interp = cntr[ak] > 21
                    if bool(interp.any()):
                        bi, bk = ai[interp], ak[interp]
                        dk = (newk[bi, bk] - prevk[bi, bk]) / 20.0
                        kk = torch.arange(1, 20, device=dev, dtype=F32)
                        vals = fl.expf(newk[bi, bk][:, None] - kk[None, :] * dk[:, None])
                        tb = binidx[bi, bk][:, None] - step * torch.arange(1, 20, device=dev)[None, :]
                        out.append((torch.full((vals.numel(),), g["lidx"], dtype=I64, device=dev),
                                    (layers[bi][:, None] * LENBUFF + (tb - 1)).reshape(-1), vals.reshape(-1)))
                # OLDKAPPA = NEWKAPPA of the last added sample
                la = torch.where(added.any(1), torch.where(added, sm, -1).amax(1), -1)
                old = torch.where(la >= 0, newk.gather(1, la.clamp(min=0)[:, None])[:, 0], old)
            alive = alive & (ends >= nsmp)
            sample += nsmp
        return out

    @staticmethod
    def _bin_of(start, step, boff):
        """red: bin = start + offset; blue: bin = start - 1 - offset (I = offset+1)."""
        return start + boff if step > 0 else start - 1 - boff
