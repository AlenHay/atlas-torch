"""Line-opacity accumulation of the selected TAPE12 lines (dfsynthe.for MAIN, labels 301-359), REAL*4 exact.

Vectorised over (layer J, line) items, streamed over chunks of lines, with element (profile point) budgets
so memory stays bounded.  Every profile value is the same REAL*4 expression as the Fortran; the contributions
of a chunk are added to the flattened [NL*LENBUFF] buffer with a 1-D `index_add_` in (layer, line, point)
order, which on CPU is a sequential float32 sum in the Fortran's per-bin order (line order).  On GPU the
atomics make the per-bin order non-deterministic (rounding-level differences only).

Fortran quirks reproduced:
* 0.2 < ADAMP <= 1.4 far wings use AA of the most recent line (any layer before, same run) that took the
  ADAMP > 1.4 or > 100 branch (AA is never set in that branch) -> forward fill over (layer, line) order.
* ADAMP <= 0.2 core: TABI accumulates TABSTEP in REAL*4 step by step (sequential loop over steps).
* first-crossing termination `IF(PROFILE(NSTEP).LT.CONTMIN)` (inclusive), MAXSTEP truncations (weak lines,
  molecules: MOD(NELION,6)=0), MAXPROF=10000.
"""
from __future__ import annotations

import torch

from .f32 import F32, F64, I64, lit, rdiv, sqrtf
from .tables import LENBUFF, LENGTH, MAXPROF, RESOLU, VSTEPS

_C1128 = float(lit("1.128"))
_C79788 = float(lit(".79788"))
_C14142 = float(lit("1.4142"))
_C75 = float(lit(".75"))
_C1EM4 = float(lit(".0001"))


def _t(x, dev):
    return torch.tensor(x, dtype=F32, device=dev)


class LineAccumulator:
    """Holds the per-temperature layer quantities and accumulates line profiles into `buffer`."""

    def __init__(self, tabs: dict, xne, txnxn, dopple, xnfdop, cont, *, aa_init: float = 0.0,
                 max_elements: int = 16_000_000):
        self.tabs = tabs
        self.dev = cont.device
        self.xne, self.txnxn, self.dopple, self.xnfdop, self.cont = xne, txnxn, dopple, xnfdop, cont
        self.nl = cont.shape[0]
        self.aa_carry = torch.full((self.nl,), float(aa_init), dtype=F32, device=self.dev)
        self.max_elements = max_elements
        self.c = {k: _t(v, self.dev) for k, v in dict(c1128=_C1128, c79=_C79788, c14=_C14142, c75=_C75,
                                                        one=1.0, half=0.5, two=2.0, three=3.0, five=5.0,
                                                        ten=10.0, c200=200.0, c15=1.5, c1em4=_C1EM4).items()}

    # ------------------------------------------------------------------ dense per-chunk quantities
    def _dense(self, nbuff, congf, nelion, gamr, gams, gamw):
        nel0 = nelion - 1
        kappa0 = congf[None, :] * self.xnfdop[:, nel0]
        contmin = self.cont[:, nbuff - 1]
        passed = ~(kappa0 < contmin)
        adamp = ((gamr[None, :] + gams[None, :] * self.xne[:, None]) + gamw[None, :] * self.txnxn[:, None]) \
            / self.dopple[:, nel0]
        return kappa0, contmin, passed, adamp

    def aa_prepass(self, t12, chunk: int = 200_000):
        """AA entering each layer = AA of the last ADAMP>1.4 line of the previous layers (Fortran order is
        layer-major, so the state crosses layers); self.aa_carry[0] is the AA left by earlier temperatures."""
        last = torch.full((self.nl,), float("nan"), dtype=F32, device=self.dev)
        n = t12["nbuff"].shape[0]
        ar = None
        for s in range(0, n, chunk):
            sl = slice(s, min(n, s + chunk))
            _, _, passed, adamp = self._dense(t12["nbuff"][sl], t12["congf"][sl], t12["nelion"][sl],
                                              t12["gamr"][sl], t12["gams"][sl], t12["gamw"][sl])
            big = passed & (adamp > float(lit("1.4")))
            ar = torch.arange(big.shape[1], device=self.dev)
            idx = torch.where(big, ar[None, :], -1).amax(dim=1)
            v = adamp.gather(1, idx.clamp(min=0)[:, None])[:, 0]
            last = torch.where(idx >= 0, v * v, last)
        aa_in = torch.empty_like(last)
        cur = self.aa_carry[0]
        for j in range(self.nl):            # scalar scan over the 25 layers (inter-layer state dependency)
            aa_in[j] = cur
            cur = torch.where(torch.isnan(last[j]), cur, last[j])
        self.aa_carry = aa_in
        self.aa_after = cur                 # state left for a following temperature
        return aa_in

    # ------------------------------------------------------------------ main entry
    def accumulate(self, buffer_flat: torch.Tensor, t12: dict, chunk: int = 50_000):
        """Add all TAPE12 lines.  buffer_flat: [NL*LENBUFF] float32 (index J*LENBUFF + NBUFF-1)."""
        self.aa_prepass(t12)
        n = t12["nbuff"].shape[0]
        stats = {"items": 0, "elements": 0}
        for s in range(0, n, chunk):
            sl = slice(s, min(n, s + chunk))
            self._chunk(buffer_flat, {k: v[sl] for k, v in t12.items()}, stats)
        return stats

    def _chunk(self, buf, t, stats):
        dev, c = self.dev, self.c
        nbuff, congf, nelion = t["nbuff"], t["congf"], t["nelion"]
        kappa0, contmin, passed, adamp = self._dense(nbuff, congf, nelion, t["gamr"], t["gams"], t["gamw"])
        C = nbuff.shape[0]
        # AA forward fill (state of the 0.2-1.4 branch)
        big = passed & (adamp > float(lit("1.4")))
        pos = torch.where(big, torch.arange(C, device=dev)[None, :], -1)
        pos = torch.cummax(pos, dim=1).values
        aa_eff = torch.where(pos >= 0, (adamp * adamp).gather(1, pos.clamp(min=0)), self.aa_carry[:, None])
        self.aa_carry = aa_eff[:, -1].clone()
        jj, ii = torch.nonzero(passed, as_tuple=True)          # (layer, line) order
        if jj.numel() == 0:
            return
        k0 = kappa0[jj, ii]; cm = contmin[jj, ii]; a = adamp[jj, ii]; aa_st = aa_eff[jj, ii]
        nb = nbuff[ii]; nel = nelion[ii]; cg = congf[ii]
        d = self.dopple[jj, nel - 1]
        ni = jj.numel()
        brA = a <= float(lit(".2"))
        brB = (~brA) & (a > 100.0)
        brC = (~brA) & (~brB) & (a > float(lit("1.4")))
        brD = ~(brA | brB | brC)
        center = torch.empty(ni, dtype=F32, device=dev)
        nstep = torch.zeros(ni, dtype=I64, device=dev)
        dv = ((c["one"] / d).to(F64) / RESOLU).to(F32)
        # per-branch parameters kept for value generation
        P = {}
        # ---------------- branch A : ADAMP <= .2
        iA = torch.nonzero(brA)[:, 0]
        if iA.numel():
            aA, kA, cA, dA = a[iA], k0[iA], cm[iA], d[iA]
            center[iA] = kA * (c["one"] - c["c1128"] * aA)
            tabstep = rdiv(float(VSTEPS), dA.to(F64) * RESOLU).to(F32)
            n10 = torch.trunc((c["ten"] * dA).to(F64) * RESOLU).to(I64)
            nmax = int(n10.max())
            core = torch.zeros((iA.numel(), max(nmax, 1)), dtype=F32, device=dev)
            tabi = torch.full_like(tabstep, 1.5)
            h0, h1 = self.tabs["h0tab"], self.tabs["h1tab"]
            for k in range(nmax):
                tabi = tabi + tabstep
                it = torch.trunc(tabi).to(I64).clamp(1, h0.numel()) - 1
                core[:, k] = kA * (h0[it] + aA * h1[it])
            kk = torch.arange(1, core.shape[1] + 1, device=dev)[None, :]
            below = (core < cA[:, None]) & (kk <= n10[:, None])
            crossed = below.any(dim=1)
            kc = torch.where(crossed, below.to(torch.int8).argmax(dim=1) + 1, n10)
            pn = core.gather(1, (n10 - 1).clamp(min=0)[:, None])[:, 0]
            pn = torch.where(n10 >= 1, pn, torch.zeros_like(pn))
            fn10 = n10.to(F32)
            X = pn * (fn10 * fn10)
            maxstep = torch.trunc(sqrtf(X / cA) + c["one"]).to(I64).clamp(max=MAXPROF)
            weak = (cg[iA] * self.tabs["freq1000"][nb[iA] // 1000]) < c["c1em4"]
            mol = (nel[iA] % 6) == 0
            maxstep = torch.where(weak | mol, torch.minimum(maxstep, n10 * 10), maxstep)
            nstep[iA] = torch.where(crossed, kc, maxstep)
            P["A"] = (iA, core, n10, X)
        # ---------------- branch B : ADAMP > 100
        iB = torch.nonzero(brB)[:, 0]
        if iB.numel():
            aB, kB = a[iB], k0[iB]
            pc = ((kB / aB) * c["c79"]) / c["c14"]
            center[iB] = pc
            aaB = aB * aB
            pcaa = pc * aaB
            dvB = dv[iB]
            fB = lambda k, sel: pcaa[sel] / (aaB[sel] + (k.to(F32) * dvB[sel]) * (k.to(F32) * dvB[sel]))
            nstep[iB] = self._first_cross(fB, cm[iB], 1)
            P["B"] = (iB, pcaa, aaB)
        # ---------------- branch C : 1.4 < ADAMP <= 100
        iC = torch.nonzero(brC)[:, 0]
        if iC.numel():
            aC, kC = a[iC], k0[iC]
            aaC = aC * aC
            profc = (kC * aC) * c["c79"]
            center[iC] = (((kC / aC) * c["c79"]) / c["c14"]) * (c["one"] + ((c["c75"] / aaC) - c["half"]) / aaC)
            dvC = dv[iC]
            fC = lambda k, sel: self._asym(profc[sel], aaC[sel], k.to(F32) * dvC[sel])
            nstep[iC] = self._first_cross(fC, cm[iC], 1)
            P["C"] = (iC, profc, aaC)
        # ---------------- branch D : .2 < ADAMP <= 1.4
        iD = torch.nonzero(brD)[:, 0]
        if iD.numel():
            aD, kD, cD, dD, dvD = a[iD], k0[iD], cm[iD], d[iD], dv[iD]
            atab = self.tabs["atab"]
            ia = torch.trunc(aD * c["c200"] + c["c15"]).to(I64) - 1
            center[iD] = kD * atab[ia, 0]
            n5 = torch.trunc((c["five"] * dD).to(F64) * RESOLU).to(I64)
            nmax = int(n5.max())
            kk = torch.arange(1, nmax + 1, device=dev)
            iv = torch.trunc((kk[None, :].to(F32) * dvD[:, None]) * c["c200"] + c["c15"]).to(I64) - 1
            core = kD[:, None] * atab[ia[:, None], iv.clamp(0, atab.shape[1] - 1)]
            below = (core < cD[:, None]) & (kk[None, :] <= n5[:, None])
            crossed = below.any(dim=1)
            kc = below.to(torch.int8).argmax(dim=1) + 1
            profc = (kD * aD) * c["c79"]
            aaD = aa_st[iD]
            fD = lambda k, sel: self._asym(profc[sel], aaD[sel], k.to(F32) * dvD[sel])
            far = self._first_cross(fD, cD, n5 + 1, active=~crossed)
            nstep[iD] = torch.where(crossed, kc, far)
            P["D"] = (iD, core, n5, profc, aaD)
        self._emit(buf, jj, nb, center, nstep, dv, brA, brB, brC, brD, P, stats)

    # ------------------------------------------------------------------ helpers
    def _asym(self, profc, aa, kdv):
        c = self.c
        vv = kdv * kdv
        u = (aa + vv) * c["c14"]
        aau = aa / u
        vvu = vv / u
        return (profc / u) * (((((aau - c["ten"] * vvu) * aau + c["five"] * vvu * vvu) / u + vvu - aau / c["three"])
                               * c["three"]) / u + c["one"])

    def _first_cross(self, f, cmin, start, active=None):
        """First k in [start, MAXPROF] with f(k) < cmin (else MAXPROF): exact scan of the first 64 steps,
        then bisection (the profiles are strictly decreasing there)."""
        n = cmin.numel()
        dev = cmin.device
        start = torch.as_tensor(start, device=dev).expand(n).to(I64)
        res = torch.full((n,), MAXPROF, dtype=I64, device=dev)
        act = torch.ones(n, dtype=torch.bool, device=dev) if active is None else active.clone()
        act = act & (start <= MAXPROF)
        S0 = 64
        sel = torch.nonzero(act)[:, 0]
        if sel.numel():
            ks = start[sel][:, None] + torch.arange(S0, device=dev)[None, :]
            ok = ks <= MAXPROF
            vals = f(ks, sel[:, None])
            below = (vals < cmin[sel][:, None]) & ok
            hit = below.any(dim=1)
            res[sel[hit]] = ks[hit].gather(1, below[hit].to(torch.int8).argmax(dim=1, keepdim=True))[:, 0]
            rest = sel[~hit]
            lo = start[rest] + S0 - 1            # f(lo) >= cmin
            hi = torch.full_like(lo, MAXPROF + 1)
            todo = lo < MAXPROF
            rest, lo, hi = rest[todo], lo[todo], hi[todo]
            while rest.numel() and bool((hi - lo > 1).any()):
                mid = (lo + hi) // 2
                fm = f(mid, rest)
                b = fm < cmin[rest]
                hi = torch.where(b, mid, hi)
                lo = torch.where(b, lo, mid)
            res[rest] = hi.clamp(max=MAXPROF)
        return res

    def _emit(self, buf, jj, nb, center, nstep, dv, brA, brB, brC, brD, P, stats):
        """Generate the contributions item by item (center, red k=1..r, blue k=1..b) and add them."""
        dev = self.dev
        r = torch.where(nb < LENGTH, torch.minimum(LENGTH - nb, nstep), torch.zeros_like(nstep))
        b = torch.where(nb > 1, torch.minimum(nb - 1, nstep), torch.zeros_like(nstep))
        r = r.clamp(min=0); b = b.clamp(min=0)
        cnt = 1 + r + b
        ni = cnt.numel()
        stats["items"] += ni
        stats["elements"] += int(cnt.sum())
        # branch-local index of every item
        loc = torch.full((ni,), -1, dtype=I64, device=dev)
        for key in P:
            idx = P[key][0]
            loc[idx] = torch.arange(idx.numel(), device=dev)
        br = torch.zeros(ni, dtype=torch.int8, device=dev)
        br[brB] = 1; br[brC] = 2; br[brD] = 3
        csum = torch.cumsum(cnt, 0)
        g0 = 0
        while g0 < ni:
            base = int(csum[g0 - 1]) if g0 else 0
            g1 = int(torch.searchsorted(csum, torch.tensor(base + self.max_elements, device=dev), right=True))
            g1 = max(g1, g0 + 1)
            sl = slice(g0, g1)
            cn = cnt[sl]
            tot = int(cn.sum())
            item = torch.repeat_interleave(torch.arange(g0, g1, device=dev), cn)
            first = torch.cumsum(cn, 0) - cn
            p = torch.arange(tot, device=dev) - torch.repeat_interleave(first, cn)
            ri = r[item]
            is_red = (p >= 1) & (p <= ri)
            k = torch.where(is_red, p, p - ri)
            tgt = torch.where(p == 0, nb[item], torch.where(is_red, nb[item] + k, nb[item] - k))
            val = torch.empty(tot, dtype=F32, device=dev)
            m0 = p == 0
            val[m0] = center[item[m0]]
            w = ~m0
            bi = br[item]
            for code, key in ((0, "A"), (1, "B"), (2, "C"), (3, "D")):
                m = w & (bi == code)
                if not bool(m.any()) or key not in P:
                    continue
                e_item = item[m]; e_k = k[m]; lj = loc[e_item]
                if key == "A":
                    _, core, n10, X = P["A"]
                    in_core = e_k <= n10[lj]
                    fk = e_k.to(F32)
                    v = X[lj] / (fk * fk)
                    cv = core[lj, (e_k - 1).clamp(max=core.shape[1] - 1)]
                    val[m] = torch.where(in_core, cv, v)
                elif key == "B":
                    _, pcaa, aaB = P["B"]
                    kd = e_k.to(F32) * dv[e_item]
                    val[m] = pcaa[lj] / (aaB[lj] + kd * kd)
                elif key == "C":
                    _, profc, aaC = P["C"]
                    val[m] = self._asym(profc[lj], aaC[lj], e_k.to(F32) * dv[e_item])
                else:
                    _, core, n5, profc, aaD = P["D"]
                    in_core = e_k <= n5[lj]
                    cv = core[lj, (e_k - 1).clamp(max=core.shape[1] - 1)]
                    fv = self._asym(profc[lj], aaD[lj], e_k.to(F32) * dv[e_item])
                    val[m] = torch.where(in_core, cv, fv)
            flat = jj[item] * LENBUFF + (tgt - 1)
            buf.index_add_(0, flat, val)
            g0 = g1
