"""Hydrogen line profiles of synthe.for: HPROF4 (+ HFNM, VCSE1F, SOFBET), vectorised over (layer, point).

Single precision follows the Fortran (implicit REAL*4; DELW and wavelength constants REAL*8).  The
depth-vector setup (loop 10 of HPROF4) is a gfortran-vectorised loop (libmvec powf on groups of 4 layers)
in the stock binary and scalar in the -O0 oracle: pass mode="stock"|"scalar".
"""
from __future__ import annotations

import numpy as np
import torch

from ...tables import table
from . import libm32 as L
from .tabs import fastex

SRC = "synthe.patched.for"
F32 = np.float32
F = torch.float32
D = torch.float64


def f(x) -> np.float32:
    return F32(x)


def sqrtf(t: torch.Tensor) -> torch.Tensor:
    """Correctly rounded REAL*4 SQRT (torch's float32/float64 sqrt kernels are not correctly rounded;
    the double result rounded once to float32 is)."""
    return torch.sqrt(t.to(D)).to(F)


def rdiv(c: float, t: torch.Tensor) -> torch.Tensor:
    """c / t with a true IEEE division (torch's `scalar / tensor` is reciprocal(t) * c)."""
    return torch.div(torch.tensor(c, dtype=t.dtype, device=t.device), t)


def _t(u, v):
    return table(SRC, u, v)


def _folded_pow10(x: float) -> np.float32:
    """10.**x for a REAL*4 constant expression: gfortran folds it with MPFR (correctly rounded)."""
    import mpmath
    with mpmath.workprec(200):
        return F32(float(mpmath.power(10, mpmath.mpf(x))))


# ------------------------------------------------------------------ HFNM
def hfnm(n: int, m: int) -> np.float32:
    if m <= n:
        return F32(0)
    xn = F32(n)
    ginf = f(.2027) / F32(L.spowf1(xn, f(.71)))
    gca = f(.124) / xn
    fkn = xn * f(1.9603)
    wtc = f(.45) - f(2.4) / (xn * xn * xn) * (xn - F32(1))
    xm = F32(m)
    xmn = F32(m - n)
    t = xm / (xmn * (xm + xn))
    fk = fkn * (t * t * t)
    xmn12 = F32(L.spowf1(xmn, f(1.2)))
    wt = (xmn12 - F32(1)) / (xmn12 + wtc)
    return F32(fk * (F32(1) - wt * ginf - (f(.222) + gca / xm) * (F32(1) - wt)))


# ------------------------------------------------------------------ VCSE1F
def vcse1f(x: torch.Tensor, extab: torch.Tensor) -> torch.Tensor:
    """VCSE1F(X) (REAL*4); each branch evaluated only where taken (logf is a scalar libm call)."""
    x = x.to(F)
    shape = x.shape
    xf = x.reshape(-1)
    out = torch.zeros_like(xf)
    i1 = torch.nonzero((xf > 0) & (xf <= 1.0)).squeeze(1)
    if i1.numel():
        xs = xf[i1]
        lg = L.slogf(xs)
        r1 = -lg - .577215 + xs
        p = xs * .00107857
        p = -.00976004 + p
        p = xs * p
        p = .05519968 + p
        p = xs * p
        p = -.24991055 + p
        p = xs * p
        p = .99999193 + p
        r2 = -lg - .57721566 + xs * p
        out[i1] = torch.where(xs <= 0.01, r1, r2)
    i3 = torch.nonzero((xf > 1.0) & (xf <= 30.0)).squeeze(1)
    if i3.numel():
        xc = xf[i3]
        out[i3] = (xc * (xc + 2.334733) + .25062) / (xc * (xc + 3.330657) + 1.681534) / xc * fastex(xc, extab)
    return out.reshape(shape)


# ------------------------------------------------------------------ SOFBET
class Sofbet:
    def __init__(self, device):
        prob = np.concatenate([_t("sofbet", f"prob{i}") for i in range(1, 8)]).astype(F32)   # PROPBMvec(525)
        cv = np.concatenate([_t("sofbet", f"c{i}") for i in range(1, 8)]).astype(F32)
        dv = np.concatenate([_t("sofbet", f"d{i}") for i in range(1, 8)]).astype(F32)
        self.prob = torch.as_tensor(prob, device=device)
        self.c = torch.as_tensor(cv, device=device)
        self.d = torch.as_tensor(dv, device=device)
        self.pp = torch.as_tensor(_t("sofbet", "pp").astype(F32), device=device)
        self.beta = torch.as_tensor(_t("sofbet", "beta").astype(F32), device=device)
        self.device = device

    def __call__(self, b: torch.Tensor, p: torch.Tensor, n: int, m: int) -> torch.Tensor:
        """b: BETA (any shape), p: PP(J) broadcastable to b (per-layer quantities stay small)."""
        b, p = b.to(F), p.to(F)
        b2 = b * b
        sb = sqrtf(b)
        indx = 7
        mmn = m - n
        if n <= 3 and mmn <= 2:
            indx = 2 * (n - 1) + mmn
        im = torch.clamp(torch.trunc(5.0 * p).to(torch.int64) + 1, max=4)
        ip = im + 1
        wtpp = 5.0 * (p - self.pp[im - 1])
        wtpm = 1.0 - wtpp
        # beta interval: first J in 2..15 with B <= BETA(J)
        bc = torch.clamp(b, max=float(self.beta[14]))
        jj = torch.searchsorted(self.beta[1:].contiguous(), bc.contiguous(), right=False) + 2  # J (1-based)
        jj = torch.clamp(jj, 2, 15)
        jm = jj - 1
        jp = jj
        bt = self.beta
        wtbp = (b - bt[jm - 1]) / (bt[jp - 1] - bt[jm - 1])
        wtbm = 1.0 - wtbp
        k0 = 75 * (indx - 1)

        def pr(i, j):
            return self.prob[(i - 1) + 5 * (j - 1) + k0]
        cbp = pr(ip, jp) * wtpp + pr(im, jp) * wtpm
        cbm = pr(ip, jm) * wtpp + pr(im, jm) * wtpm
        corr = 1.0 + cbp * wtbp + cbm * wtbm
        wt = torch.clamp(torch.clamp(0.5 * (10.0 - b), max=1.0), min=0.0)
        pr1 = torch.where(b <= 10.0, rdiv(8.0, 83.0 + (2.0 + f(.95).item() * b2) * b), torch.zeros_like(b))
        asym = (rdiv(1.5, sb) + rdiv(27.0, b2)) / b2
        pr2 = torch.where(b >= 8.0, asym, torch.zeros_like(b))
        inner = (pr1 * wt + pr2 * (1.0 - wt)) * corr
        ci = (indx - 1) * 5
        cc = self.c[ci + ip - 1] * wtpp + self.c[ci + im - 1] * wtpm
        dd = self.d[ci + ip - 1] * wtpp + self.d[ci + im - 1] * wtpm
        corr30 = 1.0 + dd / (cc + b * sb)
        r30 = asym * corr30
        r40 = asym * 1.0
        return torch.where(b > 500.0, r40, torch.where(b > f(25.12).item(), r30, inner))


# ------------------------------------------------------------------ HPROF4
class HProf4:
    """HPROF4(N,M,J,DELW,DOPPH) for all layers of the run (depth vectors computed once, ITEMP fixed)."""

    def __init__(self, st: dict, extab: torch.Tensor, mode: str = "stock"):
        """st: float32 tensors t, xne, xnfhe1, xnfh2, xnfph1, xnfph2 (layers 1..NRHOX)."""
        self.extab = extab
        dev = st["t"].device
        self.device = dev
        t = st["t"].to(F)
        xne = st["xne"].to(F)
        powf = (lambda x, y: L.loopvec(L.vpowf, L.spowf, x, y)) if mode == "stock" else L.spowf
        xne16 = powf(xne, torch.full_like(xne, f(.1666667).item()))
        self.pp = xne16 * f(.08989).item() / sqrtf(t)
        x2 = xne16 * xne16
        self.fo = (x2 * x2) * f(1.25e-9).item()
        self.y1b = rdiv(2.0, 1.0 + rdiv(.012, t) * sqrtf(xne / t))
        t4 = t / 10000.0
        t43 = powf(t4, torch.full_like(t4, f(.3).item()))
        self.y1s = t43 / xne16
        self.t3nhe = t43 * st["xnfhe1"].to(F)
        self.t3nh2 = t43 * st["xnfh2"].to(F)
        self.c1d = self.fo * 78940.0 / t
        self.c2d = self.fo * self.fo / f(5.96e-23).item() / xne
        self.gcon1 = .2 + .09 * sqrtf(t4) / (1.0 + xne / 1e13)   # python floats act as REAL*4 literals
        self.gcon2 = rdiv(.2, 1.0 + xne / 1e15)
        self.xne = xne
        self.xnfph1 = st["xnfph1"].to(F)
        self.xnfph2 = st["xnfph2"].to(F)
        self.sofbet = Sofbet(dev)
        self._line = {}

    # per-line constants (labels 20-23)
    _LAYER = ("pp", "fo", "y1b", "y1s", "t3nhe", "t3nh2", "c1d", "c2d", "gcon1", "gcon2", "xne", "xnfph1", "xnfph2")

    def sliced(self, j0: int, j1: int) -> "HProf4":
        """HPROF4 of layers j0..j1-1 with the depth vectors of the all-layer object (computed by
        gfortran-vectorised DO J loops, whose libmvec/scalar split depends on NRHOX: not recomputed)."""
        h = object.__new__(HProf4)
        h.__dict__.update(self.__dict__)
        for k in self._LAYER:
            setattr(h, k, getattr(self, k)[j0:j1])
        h._line = {}
        return h

    def line(self, n: int, m: int) -> dict:
        key = (n, m)
        if key in self._line:
            return self._line[key]
        mmn = m - n
        xn = F32(n)
        xn2 = xn * xn
        xm = F32(m)
        xm2 = xm * xm
        xmn2 = xm2 * xn2
        xm2mn2 = xm2 - xn2
        gnm = xm2mn2 / xmn2
        if mmn <= 3 and n <= 4:
            xknm = F32(_t("hprof4", "xknmtb")[n - 1, mmn - 1])
        else:
            xknm = f(5.5e-5) / gnm * xmn2 / (F32(1) + f(.13) / F32(mmn))
        y1num = F32(320.)
        if m == 2:
            y1num = F32(550.)
        if m == 3:
            y1num = F32(380.)
        y1wht = f(1e13)
        if mmn <= 3:
            y1wht = f(1e14)
        if mmn <= 2 and n <= 2:
            y1wht = F32(_t("hprof4", "y1wtm")[n - 1, mmn - 1])
        rydh = F32(_t("hprof4", "rydh"))
        freqnm = rydh * gnm
        dbeta = F32(2.99792458e18 / float(freqnm * freqnm) / float(xknm))
        wavenm = F32(2.99792458e18 / float(freqnm))
        c1con = xknm / wavenm * gnm * xm2mn2
        q = xknm / wavenm
        c2con = q * q
        asum = _t("hprof4", "asum").astype(F32)
        asuml = _t("hprof4", "asumlyman").astype(F32)
        radamp = asum[n - 1] + asum[m - 1]
        if n == 1:
            radamp = asuml[m - 1]
        radamp = radamp / f(12.5664)
        radamp = radamp / freqnm
        resont = hfnm(1, m) / xm / (F32(1) - F32(1) / xm2)
        if n != 1:
            resont = resont + hfnm(1, n) / xn / (F32(1) - F32(1) / xn2)
        resont = resont * f(3.579e-24) / gnm
        vdw = f(4.45e-26) / gnm * F32(L.spowf1(xm2 * (F32(7) * xm2 + F32(5)), f(.4)))
        stark = f(1.6678e-18) * freqnm * xknm
        if n > 4 or m > 10:
            finest = np.array([0.], F32)
            finswt = np.array([1.], F32)
        elif mmn != 1:
            ifins = int(_t("hprof4", "lncomp")[n - 1])
            stc = _t("hprof4", "stcomp").astype(F32)
            stw = _t("hprof4", "stcpwt").astype(F32)
            finest = np.array([stc[i, n - 1] * f(1e7) for i in range(ifins)], F32)
            finswt = np.array([stw[i, n - 1] / xn2 for i in range(ifins)], F32)
        else:
            ifins = int(_t("hprof4", "lnghal")[n - 1])
            ipos = int(_t("hprof4", "istal")[n - 1])
            sta = _t("hprof4", "stalph").astype(F32)
            stw = _t("hprof4", "stwtal").astype(F32)
            finest = np.array([sta[ipos - 1 + i] * f(1e7) for i in range(ifins)], F32)
            finswt = np.array([stw[ipos - 1 + i] / xn2 / F32(3) for i in range(ifins)], F32)
        d = dict(n=n, m=m, xknm=xknm, y1num=y1num, y1wht=y1wht, freqnm=freqnm, dbeta=dbeta, wavenm=wavenm,
                 c1con=c1con, c2con=c2con, radamp=radamp, resont=resont, vdw=vdw, stark=stark, finest=finest,
                 finswt=finswt)
        # layer quantities of the Stark part that do not depend on DELW (recomputed per call in Fortran,
        # identical values)
        xne = self.xne
        wty1 = rdiv(1.0, 1.0 + xne / float(y1wht))
        y1scal = float(y1num) * self.y1s * wty1 + self.y1b * (1.0 - wty1)
        c1 = self.c1d * float(c1con) * y1scal
        c2 = self.c2d * float(c2con)
        g1 = 6.77 * sqrtf(c1)
        lg = L.slogf(sqrtf(c2) / c1)
        gnot = g1 * torch.clamp(f(.2114).item() + lg, min=0.0) * (1.0 - self.gcon1 - self.gcon2)
        d.update(c1=c1, c2=c2, g1=g1, gnot=gnot)
        self._line[key] = d
        return d

    def __call__(self, n: int, m: int, j: torch.Tensor, delw: torch.Tensor, dopph: torch.Tensor) -> torch.Tensor:
        """j: 0-based layer index tensor broadcastable with delw (float64); dopph: float32 [nrhox]."""
        c = self.line(n, m)
        ext = self.extab
        delw = delw.to(D)    # j: layer indices broadcastable to delw (e.g. [nl, 1])
        fq = float(c["freqnm"])
        wn = float(c["wavenm"])
        delstark = (((-10.0 * delw) / wn) * fq).to(F)
        wl = (wn + delw * 10.0).to(F)
        freq = rdiv(2.99792458e18, wl.to(D)).to(F)
        fqt = torch.tensor(c["freqnm"], dtype=F, device=delw.device)
        dl = torch.abs(freq - fqt)
        fo = self.fo[j]
        hwstk = float(c["stark"]) * fo
        vdw = float(c["vdw"])
        hwvdw = vdw * self.t3nhe[j] + float(F32(2) * c["vdw"]) * self.t3nh2[j]
        hwrad = float(c["radamp"])
        hwres = float(c["resont"]) * self.xnfph1[j] * 2.0
        hwlor = hwres + hwvdw + hwrad
        hwdop = dopph[j].to(F)
        nwid = torch.where((hwdop >= hwstk) & (hwdop >= hwlor), 1, torch.where(hwlor >= hwstk, 2, 3))
        hfwid = fq * torch.maximum(torch.maximum(hwdop, hwlor), hwstk)
        ifcore = dl <= hfwid
        dop = fq * hwdop
        zero = torch.zeros_like(dl)
        # Doppler core with fine structure (label 32)
        hd = zero
        df = freq - fqt
        # D = |FREQ-FREQNM-FINEST|/DOP <= 7 needs |FREQ-FREQNM| <= 7*DOP + max|FINEST| (safe superset)
        near = torch.abs(df) <= (7.0 * 1.001) * dop + float(np.abs(c["finest"]).max()) * 1.001 + 1.0
        if bool(near.any()):
            dop_b = dop.expand_as(df)
            idx = torch.nonzero(near.reshape(-1)).squeeze(1)
            dfn = df.reshape(-1)[idx]
            dpn = dop_b.reshape(-1)[idx]
            hn = torch.zeros_like(dfn)
            for fe, fw in zip(c["finest"], c["finswt"]):
                dd = torch.abs(dfn - float(fe)) / dpn
                hn = torch.where(dd <= 7.0, hn + fastex(dd * dd, ext) * float(fw), hn)
            hd = torch.zeros(df.shape, dtype=F, device=df.device)
            hd.view(-1)[idx] = hn
        # Lorentz (labels 40-49)
        if n == 1 and m == 2:
            hwres4 = hwres * 4.0
            hwlor4 = hwres4 + hwvdw + hwrad
            hhw = fq * hwlor4
            fd = freq.to(D)
            near = fd > (82259.105 - 4000.0) * 2.99792458e10
            hres_near = hwres4 * fq / f(3.14159).item() / (dl * dl + hhw * hhw) * f(1.77245).item() * dop
            cut = self._lyman_h2(fd, j)
            hres_far = cut * f(1.77245).item() * dop
            hres = torch.where(near, hres_near, hres_far)
            hrad = torch.where((fd > 2.4190611e15) & (fd < float(f(.77)) * 3.288051e15),
                               rdiv(float(c["radamp"] * c["freqnm"] / f(3.14159)), dl * dl + hhw * hhw)
                               * f(1.77245).item() * dop, zero)
            hvdw = hwvdw * fq / f(3.14159).item() / (dl * dl + hhw * hhw) * f(1.77245).item() * dop
            hvdw = torch.where(fd < 1.8e15, zero, hvdw)
            hlor = hres + hrad + hvdw
        else:
            hhw = fq * hwlor
            top = hhw
            if n == 1 and m <= 5:
                lo, hi = {3: (.885, .890), 4: (.936, .938), 5: (.959, .961)}.get(m, (None, None))
                if lo is not None:
                    fd = freq.to(D)
                    win = (fd > float(f(lo)) * 3.288051e15) & (fd < float(f(hi)) * 3.288051e15)
                    top = torch.where(win, hhw - float(c["freqnm"] * c["radamp"]), hhw)
            hlor = top / f(3.14159).item() / (dl * dl + hhw * hhw) * f(1.77245).item() * dop
        # Stark (label 50)
        extra, main = self._stark(c, j, delstark, dop, freq)
        # assemble according to the GO TO structure (HPROF4 accumulates left to right)
        res_core1 = hd                         # NWID=1 core: Doppler only
        res_core2 = zero + hlor                # NWID=2 core: Lorentz only (HPROF4 starts at 0)
        res_core3 = (zero + extra) + main      # NWID=3 core: Stark only
        res_wing = ((hd + hlor) + extra) + main  # not in core: Doppler + Lorentz + Stark
        res = torch.where(ifcore, torch.where(nwid == 1, res_core1, torch.where(nwid == 2, res_core2, res_core3)),
                          res_wing)
        return res

    def _lyman_h2(self, fd, j):
        """Lyman alpha far red wing quasi-H2 cutoff (label 40..43), returns CUTOFF (REAL*4)."""
        ch2 = torch.as_tensor(_t("hprof4", "cutoffh2").astype(F32), device=fd.device)
        spacing = F32(200. * 2.99792458e10)
        f22 = F32((82259.105 - 22000.0) * 2.99792458e10)
        ff = fd.to(F)
        lo = (ch2[1] - ch2[0]) / float(spacing) * (ff - float(f22)) + ch2[0]
        icut = torch.trunc((ff - float(f22)) / float(spacing)).to(torch.int64).clamp(0, 89)
        cutfreq = icut.to(F) * float(spacing) + float(f22)
        hi = (ch2[icut + 1] - ch2[icut]) / float(spacing) * (ff - cutfreq) + ch2[icut]
        cut = torch.where(ff < float(f22), lo, hi)
        p = L.spowf(torch.full_like(cut, 10.0), cut - 14.0)
        cut = (p * self.xnfph1[j] * 2.0).to(D) / 2.99792458e10
        cut = cut.to(F)
        return torch.where(fd < 50000.0 * 2.99792458e10, torch.zeros_like(cut), cut)

    def _stark(self, c, j, delstark, dop, freq):
        ext = self.extab
        n, m = c["n"], c["m"]
        fo = self.fo[j]
        c1 = c["c1"][j]
        c2 = c["c2"][j]
        g1 = c["g1"][j]
        gnot = c["gnot"][j]
        gcon1 = self.gcon1[j]
        gcon2 = self.gcon2[j]
        beta = torch.abs(delstark) / fo * float(c["dbeta"])
        y1 = c1 * beta
        y2 = c2 * (beta * beta)
        skip = (y2 <= f(1e-4).item()) & (y1 <= f(1e-5).item())
        t90 = 90.0 * y1
        gam = g1 * (0.5 * fastex(torch.clamp(y1, max=80.0), ext) + vcse1f(y1, ext) - 0.5 * vcse1f(y2, ext)) * \
            (1.0 - gcon1 / (1.0 + t90 * t90 * t90) - gcon2 / (1.0 + 2000.0 * y1))
        gam = torch.where(gam <= f(1e-20).item(), torch.zeros_like(gam), gam)
        gam = torch.where(skip, gnot, gam)
        prqs = self.sofbet(beta, self.pp[j], n, m)
        dbeta = float(c["dbeta"])
        extra = torch.zeros_like(beta)
        if m <= 2:
            prqs = prqs * 0.5
            fd = freq.to(D)
            lo = fd < (float(f(82259.105)) - 20000.0) * 2.99792458e10
            hi = fd > (float(f(82259.105)) - 4000.0) * 2.99792458e10
            ch2p = torch.as_tensor(_t("hprof4", "cutoffh2plus").astype(F32), device=fd.device)
            f15 = F32((float(f(82259.105)) - 15000.0) * 2.99792458e10)
            spacing = F32(100. * 2.99792458e10)
            ff = freq
            a = (ch2p[1] - ch2p[0]) / float(spacing) * (ff - float(f15)) + ch2p[0]
            icut = torch.trunc((ff - float(f15)) / float(spacing)).to(torch.int64).clamp(0, 109)
            cutfreq = icut.to(F) * float(spacing) + float(f15)
            b = (ch2p[icut + 1] - ch2p[icut]) / float(spacing) * (ff - cutfreq) + ch2p[icut]
            cut = torch.where(ff < float(f15), a, b)
            p = L.spowf(torch.full_like(cut, 10.0), cut - 14.0)
            cutv = ((p.to(D) / 2.99792458e10) * self.xnfph2[j].to(D)).to(F)
            mid = cutv * f(1.77245).item() * dop
            # label 52
            beta4000 = (rdiv(4000.0 * 2.99792458e10, fo.to(D)) * dbeta).to(F)
            prqsp4000 = self.sofbet(beta4000, self.pp[j], n, m) * 0.5 / fo * dbeta
            c4000 = _folded_pow10(float(f(-11.07) - F32(14.)))  # constant expression folded by gfortran
            cut4000 = ((float(c4000) / 2.99792458e10) * self.xnfph2[j].to(D)).to(F)
            high = cut4000 / prqsp4000 * prqs / fo * dbeta * f(1.77245).item() * dop
            extra = torch.where(lo, extra, torch.where(hi, high, mid))
        ff_ = torch.where(gam > 0, gam / f(3.14159).item() / (gam * gam + beta * beta), torch.zeros_like(gam))
        p1 = (f(.9).item() * y1)
        p1 = p1 * p1
        fns = (p1 + f(.03).item() * sqrtf(y1)) / (p1 + 1.0)
        return extra, (prqs * (1.0 + fns) + ff_) / fo * dbeta * f(1.77245).item() * dop
