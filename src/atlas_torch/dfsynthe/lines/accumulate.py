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

import numpy as np
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
                 max_elements: int = 16_000_000, fastmath: bool = False):
        self.tabs = tabs
        # fastmath: the stock binary's (ifx -fp-model fast) evaluation of the ADAMP>.2 branches, see _asym_fast
        self.fastmath = fastmath
        self.dev = cont.device
        self.xne, self.txnxn, self.dopple, self.xnfdop, self.cont = xne, txnxn, dopple, xnfdop, cont
        self.nl = cont.shape[0]
        self.aa_carry = torch.full((self.nl,), float(aa_init), dtype=F32, device=self.dev)
        self.max_elements = max_elements
        self.c = {k: _t(v, self.dev) for k, v in dict(c1128=_C1128, c79=_C79788, c14=_C14142, c75=_C75,
                                                        one=1.0, half=0.5, two=2.0, three=3.0, five=5.0,
                                                        ten=10.0, c200=200.0, c15=1.5, c1em4=_C1EM4).items()}
        cb = np.float32(np.float32(_C79788) / np.float32(_C14142))        # .79788/1.4142 folded (0x3F106EDF)
        assert cb.view(np.uint32) == 0x3F106EDF
        self.c.update({k: _t(v, self.dev) for k, v in dict(cb=float(cb), m10=-10.0, m3rd=float(np.float32(-1.0 / 3.0)),
                                                           mhalf=-0.5).items()})

    # ------------------------------------------------------------------ dense per-chunk quantities
    def _dense(self, nbuff, congf, nelion, gamr, gams, gamw):
        nel0 = nelion - 1
        kappa0 = congf[None, :] * self.xnfdop[:, nel0]
        contmin = self.cont[:, nbuff - 1]
        passed = ~(kappa0 < contmin)
        adamp = ((gamr[None, :] + gams[None, :] * self.xne[:, None]) + gamw[None, :] * self.txnxn[:, None]) \
            / self.dopple[:, nel0]
        return kappa0, contmin, passed, adamp

    @staticmethod
    def _slice(t12, sl):
        """Chunk of the TAPE12 columns, integer columns widened to int64."""
        return {k: (v[sl].to(I64) if v.dtype in (torch.int32, torch.int16) else v[sl]) for k, v in t12.items()}

    def _chunks(self, t12, chunk: int):
        """Chunks (int64-widened) of TAPE12: t12 is a dict of columns, or a callable returning an iterable of
        such dicts (streamed selection, called once per pass)."""
        parts = t12() if callable(t12) else (t12,)
        for d in parts:
            n = d["nbuff"].shape[0]
            for s in range(0, n, chunk):
                yield self._slice(d, slice(s, min(n, s + chunk)))

    def aa_prepass(self, t12, chunk: int = 200_000):
        """AA entering each layer = AA of the last ADAMP>1.4 line of the previous layers (Fortran order is
        layer-major, so the state crosses layers); self.aa_carry[0] is the AA left by earlier temperatures."""
        last = torch.full((self.nl,), float("nan"), dtype=F32, device=self.dev)
        native = self._native() if self.use_native else None
        if native is not None:
            import ctypes
            P = ctypes.c_void_p
            c32 = lambda x: x.to(F32).contiguous()  # noqa: E731
            arrs = [c32(self.xnfdop), c32(self.dopple), c32(self.xne), c32(self.txnxn), c32(self.cont)]
            last = last.contiguous()
            for c in self._chunks(t12, 1_000_000):
                cols = [(c[k].to(torch.int32) if k in ("nbuff", "nelion") else c[k].to(F32)).contiguous()
                        for k in ("nbuff", "congf", "nelion", "gamr", "gams", "gamw")]
                native.aa_scan(ctypes.c_int(self.nl), ctypes.c_int(cols[0].numel()), *(P(x.data_ptr()) for x in cols),
                               *(P(x.data_ptr()) for x in arrs), P(last.data_ptr()))
        for c in (self._chunks(t12, chunk) if native is None else ()):
            _, _, passed, adamp = self._dense(c["nbuff"], c["congf"], c["nelion"], c["gamr"], c["gams"], c["gamw"])
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
    PAD = 16384                       # > max padded profile length (<= 11264): masked terms land here

    def accumulate(self, buffer_flat: torch.Tensor, t12: dict, chunk: int = 50_000, padded: torch.Tensor = None):
        """Add all TAPE12 lines.  buffer_flat: [NL*LENBUFF] float32 (index J*LENBUFF + NBUFF-1).  t12: dict of
        TAPE12 columns or a callable returning an iterable of such dicts (streamed; called twice).
        padded: optional [PAD + NL*LENBUFF + PAD] tensor of which buffer_flat is the middle view (avoids a
        copy); otherwise a padded working copy is made."""
        self.aa_prepass(t12)
        stats = {"items": 0, "elements": 0}
        nf = buffer_flat.numel()
        own = padded is None
        if own:
            padded = torch.zeros(nf + 2 * self.PAD, dtype=F32, device=self.dev)
            padded[self.PAD:self.PAD + nf] = buffer_flat
        native = self._native() if self.use_native else None
        if native is not None:
            self._run_native(native, padded, t12, stats)
        else:
            for c in self._chunks(t12, chunk):
                self._chunk(padded, c, stats)
        if own:
            buffer_flat.copy_(padded[self.PAD:self.PAD + nf])
        return stats

    use_native = True

    def _native(self):
        if self.dev.type != "cpu":
            return None
        from .native import load
        return load()

    def _run_native(self, fn, padded, t12, stats, chunk: int = 500_000):
        """The C transcription of the line loop (native/linesum.c), chunk by chunk, layer-major inside a chunk;
        the AA state per layer is carried across chunks (self.aa_carry from aa_prepass)."""
        import ctypes
        from .rcpps import _table_np
        P = ctypes.c_void_p
        c32 = lambda x: x.to(F32).contiguous()  # noqa: E731
        keep = dict(xnfdop=c32(self.xnfdop), dopple=c32(self.dopple), xne=c32(self.xne), txnxn=c32(self.txnxn),
                    cont=c32(self.cont), h0=c32(self.tabs["h0tab"]), h1=c32(self.tabs["h1tab"]),
                    atab=c32(self.tabs["atab"]), f1000=c32(self.tabs["freq1000"]),
                    rcp=torch.from_numpy(_table_np().astype(np.uint32).view(np.int32).copy()),
                    prof=torch.zeros(MAXPROF + 2, dtype=F32), st=torch.zeros(2, dtype=torch.int64),
                    aa=self.aa_carry.to(F32).contiguous().clone())
        assert keep["cont"].shape[1] == LENGTH and padded.is_contiguous()
        bufp = padded.data_ptr() + 4 * self.PAD
        for c in self._chunks(t12, chunk):
            cols = {k: (c[k].to(torch.int32) if k in ("nbuff", "nelion") else c[k].to(F32)).contiguous()
                    for k in ("nbuff", "congf", "nelion", "gamr", "gams", "gamw")}
            n = cols["nbuff"].numel()
            if n == 0:
                continue
            fn.linesum(ctypes.c_int(self.nl), ctypes.c_int(n), *(P(cols[k].data_ptr()) for k in
               ("nbuff", "congf", "nelion", "gamr", "gams", "gamw")),
               P(keep["xnfdop"].data_ptr()), P(keep["dopple"].data_ptr()), P(keep["xne"].data_ptr()),
               P(keep["txnxn"].data_ptr()), P(keep["cont"].data_ptr()), P(bufp), P(keep["aa"].data_ptr()),
               P(keep["h0"].data_ptr()), P(keep["h1"].data_ptr()), P(keep["atab"].data_ptr()),
               P(keep["f1000"].data_ptr()), P(keep["rcp"].data_ptr()), ctypes.c_int(1 if self.fastmath else 0),
               P(keep["prof"].data_ptr()), P(keep["st"].data_ptr()))
        stats["items"] += int(keep["st"][0])
        stats["elements"] += int(keep["st"][1])
        self.aa_carry = keep["aa"]

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
            h0, h1 = self.tabs["h0tab"], self.tabs["h1tab"]
            # TABI=TABI+TABSTEP (REAL*4, sequential) depends only on TABSTEP, i.e. on (layer, ion): one table
            # of H0TAB/H1TAB indices per distinct TABSTEP, gathered for every item
            ust, inv = torch.unique(tabstep, return_inverse=True)
            itab = torch.zeros((ust.numel(), max(nmax, 1)), dtype=I64, device=dev)
            tabi = torch.full_like(ust, 1.5)
            for k in range(nmax):
                tabi = tabi + ust
                itab[:, k] = torch.trunc(tabi).to(I64).clamp(1, h0.numel()) - 1
            itk = itab[inv]
            core = kA[:, None] * (h0[itk] + aA[:, None] * h1[itk])
            del itk
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
            # stock: DO 322 NSTEP=N1,MAXSTEP is vectorised (rcpps + Newton step for X/NSTEP**2) over the first
            # (MAXSTEP-N10DOP)&~3 steps, the remaining ones use an exact division
            vend = n10 + ((maxstep - n10).clamp(min=0) & ~3) if self.fastmath else n10
            P["A"] = (iA, core, n10, X, vend)
        # ---------------- branch B : ADAMP > 100
        iB = torch.nonzero(brB)[:, 0]
        if iB.numel():
            aB, kB = a[iB], k0[iB]
            if self.fastmath:          # stock: (KAPPA0*(.79788/1.4142))/ADAMP
                pc = (kB * c["cb"]) / aB
            else:
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
            dvC = dv[iC]
            if self.fastmath:          # stock: PROFILEC=(KAPPA0*.79788)*ADAMP; centre with R=1/AA (see _asym_fast)
                profc = (kC * c["c79"]) * aC
                r = c["one"] / aaC
                center[iC] = ((((r * c["c75"]) + c["mhalf"]) * r + c["one"]) * (kC * c["cb"])) / aC
                fC = lambda k, sel: self._asym_fast(profc[sel], aaC[sel], k.to(F32) * dvC[sel])
            else:
                profc = (kC * aC) * c["c79"]
                center[iC] = (((kC / aC) * c["c79"]) / c["c14"]) * (c["one"] + ((c["c75"] / aaC) - c["half"]) / aaC)
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
            if self.fastmath:          # stock: IV = NSTEP*(DVOIGT*200.)+1.5
                iv = torch.trunc(kk[None, :].to(F32) * (dvD * c["c200"])[:, None] + c["c15"]).to(I64) - 1
            else:
                iv = torch.trunc((kk[None, :].to(F32) * dvD[:, None]) * c["c200"] + c["c15"]).to(I64) - 1
            core = kD[:, None] * atab[ia[:, None], iv.clamp(0, atab.shape[1] - 1)]
            below = (core < cD[:, None]) & (kk[None, :] <= n5[:, None])
            crossed = below.any(dim=1)
            kc = below.to(torch.int8).argmax(dim=1) + 1
            aaD = aa_st[iD]
            if self.fastmath:
                profc = (kD * c["c79"]) * aD
                fD = lambda k, sel: self._asym_fast(profc[sel], aaD[sel], k.to(F32) * dvD[sel])
            else:
                profc = (kD * aD) * c["c79"]
                fD = lambda k, sel: self._asym(profc[sel], aaD[sel], k.to(F32) * dvD[sel])
            far = self._first_cross(fD, cD, n5 + 1, active=~crossed)
            nstep[iD] = torch.where(crossed, kc, far)
            P["D"] = (iD, core, n5, profc, aaD)
        self._emit(buf, jj, nb, center, nstep, dv, brA, brB, brC, brD, P, stats)

    # ------------------------------------------------------------------ helpers
    def _asym(self, profc, aa, kdv):
        """PROFILEC/U*((((AAU-10.*VVU)*AAU+5.*VVU*VVU)/U+VVU-AAU/3.)*3./U+1.), U=(AA+VV)*1.4142, VV=(K*DVOIGT)**2:
        the same REAL*4 operations in the same order as the Fortran, evaluated in place (fewer temporaries;
        additions/multiplications only swap their operands, which IEEE leaves unchanged)."""
        c = self.c
        vv = kdv * kdv
        u = vv + aa
        u.mul_(c["c14"])
        aau = aa / u
        vvu = vv.div_(u)
        t = vvu * c["ten"]
        torch.sub(aau, t, out=t)
        t.mul_(aau)
        w = vvu * c["five"]
        w.mul_(vvu)
        t.add_(w)
        del w
        t.div_(u)
        t.add_(vvu)
        aau.div_(c["three"])
        t.sub_(aau)
        t.mul_(c["three"])
        t.div_(u)
        t.add_(c["one"])
        return (profc / u).mul_(t)

    def _prof(self, profc, aa, kdv):
        return self._asym_fast(profc, aa, kdv) if self.fastmath else self._asym(profc, aa, kdv)

    def _asym_fast(self, profc, aa, kdv):
        """The far-wing profile as the stock binary evaluates it (dfsynthe.exe 0x407220..0x4072ad, label 1323, and
        0x407490.., label 1325): one reciprocal R=1/U instead of the four divisions, AAU/3 as AAU*(-1/3) added
        to VVU, 5.*VVU*VVU as (VVU*VVU)*5:
            AAU=AA*R  VVU=VV*R  T=(AAU+VVU*(-10))*AAU+(VVU*VVU)*5  S=T*R+(VVU+AAU*(-1/3))
            PROFILE=(R*PROFILEC)*((S*3)*R+1)"""
        c = self.c
        vv = kdv * kdv
        u = vv + aa
        u.mul_(c["c14"])
        r = c["one"] / u
        del u
        aau = aa * r
        vvu = vv.mul_(r)
        t = vvu * c["m10"]
        t.add_(aau)
        t.mul_(aau)
        w = vvu * vvu
        w.mul_(c["five"])
        t.add_(w)
        t.mul_(r)
        aau.mul_(c["m3rd"])
        aau.add_(vvu)
        t.add_(aau)
        t.mul_(c["three"])
        t.mul_(r)
        t.add_(c["one"])
        return (r * profc).mul_(t)

    def _first_cross(self, f, cmin, start, active=None):
        """First k in [start, MAXPROF] with f(k) < cmin (else MAXPROF): exact scan of the first 64 steps (in
        growing windows 4, 8, 16, 36: items that cross early are not evaluated further), then bisection (the
        profiles are strictly decreasing there)."""
        n = cmin.numel()
        dev = cmin.device
        start = torch.as_tensor(start, device=dev).expand(n).to(I64)
        res = torch.full((n,), MAXPROF, dtype=I64, device=dev)
        act = torch.ones(n, dtype=torch.bool, device=dev) if active is None else active.clone()
        act = act & (start <= MAXPROF)
        S0 = 64
        rest = torch.nonzero(act)[:, 0]
        w0 = 0
        for w1 in (4, 12, 28, S0):
            if rest.numel() == 0:
                break
            ks = start[rest][:, None] + torch.arange(w0, w1, device=dev)[None, :]
            ok = ks <= MAXPROF
            vals = f(ks, rest[:, None])
            below = (vals < cmin[rest][:, None]) & ok
            hit = below.any(dim=1)
            if bool(hit.any()):
                res[rest[hit]] = ks[hit].gather(1, below[hit].to(torch.int8).argmax(dim=1, keepdim=True))[:, 0]
            rest = rest[~hit]
            w0 = w1
        if rest.numel():
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

    def _values(self, key, P, lj, kk, dv_i):
        """Profile values PROFILE(k) of branch `key` for branch-local items lj [n] at steps kk [K] -> [n, K]."""
        fk = kk.to(F32)[None, :]
        if key == "A":
            _, core, n10, X, vend = P["A"]
            y = fk * fk
            xl = X[lj][:, None]
            far = xl / y
            if self.fastmath:
                vec = kk[None, :] <= vend[lj][:, None]
                if bool(vec.any()):
                    from .rcpps import rcpps
                    r = rcpps(y)                         # depends on the step only: one row for all items
                    q0 = xl * r                          # rcpps.div_nr, with the reciprocal shared
                    nr = q0 * y
                    torch.sub(xl, nr, out=nr)
                    nr.mul_(r)
                    nr.add_(q0)
                    far = torch.where(vec, nr, far)
            nc = core.shape[1]
            if nc == 0 or int(kk[0]) > nc:
                return far
            cv = core[lj[:, None], (kk - 1).clamp(max=nc - 1)[None, :]]
            return torch.where(kk[None, :] <= n10[lj][:, None], cv, far)
        if key == "B":
            _, pcaa, aaB = P["B"]
            kd = fk * dv_i[:, None]
            return pcaa[lj][:, None] / (aaB[lj][:, None] + kd * kd)
        if key == "C":
            _, profc, aaC = P["C"]
            return self._prof(profc[lj][:, None], aaC[lj][:, None], fk * dv_i[:, None])
        _, core, n5, profc, aaD = P["D"]
        fv = self._prof(profc[lj][:, None], aaD[lj][:, None], fk * dv_i[:, None])
        nc = core.shape[1]
        if nc == 0 or int(kk[0]) > nc:
            return fv
        cv = core[lj[:, None], (kk - 1).clamp(max=nc - 1)[None, :]]
        return torch.where(kk[None, :] <= n5[lj][:, None], cv, fv)

    _KGRID: dict = {}

    @classmethod
    def _kgrid(cls, dev):
        """Padded profile lengths: 0, 1..8, then 8 per octave (<= 12.5 % padding) up to >= MAXPROF."""
        if dev not in cls._KGRID:
            g = list(range(0, 9))
            b = 8
            while g[-1] < MAXPROF:
                g += [b + b * i // 8 for i in range(1, 9)]
                b *= 2
            cls._KGRID[dev] = torch.tensor(sorted(set(g)), dtype=I64, device=dev)
        return cls._KGRID[dev]

    def _emit(self, buf, jj, nb, center, nstep, dv, brA, brB, brC, brD, P, stats):
        """Add the contributions of the items (centre, red k=1..r, blue k=1..b) to the padded buffer.

        Items are grouped by branch and padded length K >= max(r, b) (grid of 8 lengths per octave);
        each item gets a fixed row [centre | red 1..K | blue 1..K] in a flat (target, value) list at an
        item-major offset.  PROFILE(1..K) is computed once per item (dense blocks); entries beyond r (b) are
        +0.0 (x + 0.0 == x for the non-negative buffer) and may point into the PAD margins or a neighbouring
        layer.  One sequential index_add_ of the list then gives every bin its terms in (layer, line) order,
        i.e. the Fortran's REAL*4 summation order (independent of the order of the lines in the chunk)."""
        dev = self.dev
        PAD = self.PAD
        r = torch.where(nb < LENGTH, torch.minimum(LENGTH - nb, nstep), torch.zeros_like(nstep)).clamp(min=0)
        b = torch.where(nb > 1, torch.minimum(nb - 1, nstep), torch.zeros_like(nstep)).clamp(min=0)
        ni = r.numel()
        stats["items"] += ni
        stats["elements"] += int((1 + r + b).sum())
        loc = torch.full((ni,), -1, dtype=I64, device=dev)
        for key in P:
            idx = P[key][0]
            loc[idx] = torch.arange(idx.numel(), device=dev)
        br = torch.zeros(ni, dtype=torch.int8, device=dev)
        br[brB] = 1; br[brC] = 2; br[brD] = 3
        base = jj * LENBUFF + (nb - 1) + PAD                                  # index in the padded buffer
        m = torch.maximum(r, b)
        grid = self._kgrid(dev)
        gidx = torch.searchsorted(grid, m)                                  # padded length K = grid[gidx] >= m
        Kp = grid[gidx]
        cnt = 1 + 2 * Kp
        csum = torch.cumsum(cnt, 0)
        budget = self.max_elements
        zero = torch.zeros((), dtype=F32, device=dev)
        g0 = 0
        while g0 < ni:
            e0 = int(csum[g0 - 1]) if g0 else 0
            g1 = int(torch.searchsorted(csum, torch.tensor(e0 + budget, device=dev), right=True))
            g1 = max(g1, g0 + 1)
            tot = int(csum[g1 - 1]) - e0
            o = csum[g0:g1] - cnt[g0:g1] - e0                                 # piece-local offset of each item
            val = torch.empty(tot, dtype=F32, device=dev)
            tgt = torch.empty(tot, dtype=I64, device=dev)
            val.scatter_(0, o, center[g0:g1])
            tgt.scatter_(0, o, base[g0:g1])
            bp, kp = br[g0:g1], gidx[g0:g1]
            gkey = kp * 4 + bp.to(I64)
            for gk in torch.unique(gkey).tolist():
                gi_, code = gk // 4, gk % 4
                key = "ABCD"[code]
                K = int(grid[gi_])
                if K == 0 or key not in P:
                    continue
                kk = torch.arange(1, K + 1, device=dev)
                k0 = kk - 1
                rows = max(1, (1 << 22) // K)
                sel = torch.nonzero(gkey == gk)[:, 0]
                for s0 in range(0, sel.numel(), rows):
                    it = sel[s0:s0 + rows]                                   # piece-local item indices
                    gi = it + g0
                    V = self._values(key, P, loc[gi], kk, dv[gi])
                    ri, bi = r[gi], b[gi]
                    vr = torch.where(kk <= ri[:, None], V, zero)
                    vb = vr if bool((ri == bi).all()) else torch.where(kk <= bi[:, None], V, zero)
                    pr = (o[it] + 1)[:, None] + k0
                    ba = base[gi][:, None]
                    val.scatter_(0, pr.view(-1), vr.view(-1))
                    tgt.scatter_(0, pr.view(-1), (ba + kk).view(-1))
                    pr += K
                    val.scatter_(0, pr.view(-1), vb.view(-1))
                    tgt.scatter_(0, pr.view(-1), (ba - kk).view(-1))
            buf.index_add_(0, tgt, val)
            g0 = g1
