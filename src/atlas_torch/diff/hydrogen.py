"""Differentiable hydrogen lines of fort.19 (XLINOP TYPE -1/-2 with HPROF4) in float64.

Same structure as atlas_torch.synthe.lineopac.xlinop.XLinop._hydrogen and hydrogen.HProf4 (the exact REAL*4
port), evaluated in float64 with torch ops on the differentiable layer quantities (T, XNE, N(He I), N(H2),
N(H I)/U, N(H II), Doppler width with TURBV, HCKT). Literal constants are the REAL*4 values the Fortran uses;
FASTEX goes through the exact table with the derivative of exp (straight_through, or exp itself for
profile="smooth"). The wing walks (which points each line reaches in each layer: the cutoff KAPMIN, the
ALPHAHYD / WLMINUS1 / WLPLUS1 limits and the comparison with the neighbouring series member) are decided on
detached values; contributions are then summed in float64.
"""
from __future__ import annotations

import numpy as np
import torch

from ..synthe.lineopac.hydrogen import Sofbet, _folded_pow10, _t, hfnm
from ..synthe.lineopac.xlinop import vacair
from ..synthe.lineopac import tabs as T32
from .native import straight_through

D = torch.float64
F = torch.float32
F32 = np.float32


def f(x) -> float:
    """A REAL*4 literal as a float64 number."""
    return float(np.float32(x))


def _st(tab, smooth, mode):
    return smooth if mode == "smooth" else straight_through(tab, smooth)


class Ext:
    """FASTEX with straight-through derivative, and VCSE1F, in float64."""

    def __init__(self, tabs, mode):
        self.tabs, self.mode = tabs, mode

    def fastex(self, x):
        return _st(T32.fastex(x.detach().to(F), self.tabs.extab32).to(D), torch.exp(-x), self.mode)

    def vcse1f(self, x):
        out = torch.zeros_like(x)
        x1 = torch.where((x > 0) & (x <= 1.0), x, torch.full_like(x, 0.5))
        lg = torch.log(x1)
        r1 = -lg - f(.577215) + x1
        p = (((f(.00107857) * x1 + f(-.00976004)) * x1 + f(.05519968)) * x1 + f(-.24991055)) * x1 + f(.99999193)
        r2 = -lg - f(.57721566) + x1 * p
        v1 = torch.where(x1 <= 0.01, r1, r2)
        x3 = torch.where((x > 1.0) & (x <= 30.0), x, torch.full_like(x, 2.0))
        v3 = (x3 * (x3 + f(2.334733)) + f(.25062)) / (x3 * (x3 + f(3.330657)) + f(1.681534)) / x3 * self.fastex(x3)
        out = torch.where((x > 0) & (x <= 1.0), v1, out)
        return torch.where((x > 1.0) & (x <= 30.0), v3, out)


class SofbetD:
    """SOFBET (Stark profile of the quasi-static ions) in float64, tables of the exact port."""

    def __init__(self, device):
        s = Sofbet(device)
        self.prob, self.c, self.d = s.prob.to(D), s.c.to(D), s.d.to(D)
        self.pp, self.beta = s.pp.to(D), s.beta.to(D)
        self.beta32 = s.beta
        self.pp32 = s.pp

    def __call__(self, b, p, n, m):
        b2 = b * b
        bpos = torch.where(b > 0, b, torch.ones_like(b))
        sb = torch.sqrt(bpos)
        indx = 7
        mmn = m - n
        if n <= 3 and mmn <= 2:
            indx = 2 * (n - 1) + mmn
        with torch.no_grad():
            im = torch.clamp(torch.trunc(5.0 * p.detach().to(F)).to(torch.int64) + 1, max=4)
            bc = torch.clamp(b.detach().to(F), max=float(self.beta32[14]))
            jj = torch.searchsorted(self.beta32[1:].contiguous(), bc.contiguous(), right=False) + 2
            jj = torch.clamp(jj, 2, 15)
        ip = im + 1
        wtpp = 5.0 * (p - self.pp[im - 1])
        wtpm = 1.0 - wtpp
        jm, jp = jj - 1, jj
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
        pr1 = torch.where(b <= 10.0, 8.0 / (83.0 + (2.0 + f(.95) * b2) * b), torch.zeros_like(b))
        b2s = torch.where(b > 0, b2, torch.ones_like(b2))
        asym = (1.5 / sb + 27.0 / b2s) / b2s
        pr2 = torch.where(b >= 8.0, asym, torch.zeros_like(b))
        inner = (pr1 * wt + pr2 * (1.0 - wt)) * corr
        ci = (indx - 1) * 5
        cc = self.c[ci + ip - 1] * wtpp + self.c[ci + im - 1] * wtpm
        dd = self.d[ci + ip - 1] * wtpp + self.d[ci + im - 1] * wtpm
        r30 = asym * (1.0 + dd / (cc + b * sb))
        return torch.where(b > 500.0, asym, torch.where(b > f(25.12), r30, inner))


class HProf4D:
    """HPROF4(N,M,J,DELW,DOPPH) in float64 for all layers (depth vectors from attached tensors)."""

    def __init__(self, st: dict, ext: Ext):
        self.ext = ext
        t, xne = st["t"], st["xne"]
        dev = t.device
        xne16 = torch.pow(xne, f(.1666667))
        self.pp = xne16 * f(.08989) / torch.sqrt(t)
        x2 = xne16 * xne16
        self.fo = (x2 * x2) * f(1.25e-9)
        self.y1b = 2.0 / (1.0 + f(.012) / t * torch.sqrt(xne / t))
        t4 = t / 10000.0
        t43 = torch.pow(t4, f(.3))
        self.y1s = t43 / xne16
        self.t3nhe = t43 * st["xnfhe1"]
        self.t3nh2 = t43 * st["xnfh2"]
        self.c1d = self.fo * 78940.0 / t
        self.c2d = self.fo * self.fo / f(5.96e-23) / xne
        self.gcon1 = f(.2) + f(.09) * torch.sqrt(t4) / (1.0 + xne / 1e13)
        self.gcon2 = f(.2) / (1.0 + xne / 1e15)
        self.xne = xne
        self.xnfph1 = st["xnfph1"]
        self.xnfph2 = st["xnfph2"]
        self.sofbet = SofbetD(dev)
        self._line = {}

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
            xknm = F32(5.5e-5) / gnm * xmn2 / (F32(1) + F32(.13) / F32(mmn))
        y1num = F32(320.)
        if m == 2:
            y1num = F32(550.)
        if m == 3:
            y1num = F32(380.)
        y1wht = F32(1e13)
        if mmn <= 3:
            y1wht = F32(1e14)
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
        radamp = radamp / F32(12.5664)
        radamp = radamp / freqnm
        resont = hfnm(1, m) / xm / (F32(1) - F32(1) / xm2)
        if n != 1:
            resont = resont + hfnm(1, n) / xn / (F32(1) - F32(1) / xn2)
        resont = resont * F32(3.579e-24) / gnm
        from ..synthe.lineopac import libm32 as L
        vdw = F32(4.45e-26) / gnm * F32(L.spowf1(xm2 * (F32(7) * xm2 + F32(5)), F32(.4)))
        stark = F32(1.6678e-18) * freqnm * xknm
        if n > 4 or m > 10:
            finest = np.array([0.], F32)
            finswt = np.array([1.], F32)
        elif mmn != 1:
            ifins = int(_t("hprof4", "lncomp")[n - 1])
            stc = _t("hprof4", "stcomp").astype(F32)
            stw = _t("hprof4", "stcpwt").astype(F32)
            finest = np.array([stc[i, n - 1] * F32(1e7) for i in range(ifins)], F32)
            finswt = np.array([stw[i, n - 1] / xn2 for i in range(ifins)], F32)
        else:
            ifins = int(_t("hprof4", "lnghal")[n - 1])
            ipos = int(_t("hprof4", "istal")[n - 1])
            sta = _t("hprof4", "stalph").astype(F32)
            stw = _t("hprof4", "stwtal").astype(F32)
            finest = np.array([sta[ipos - 1 + i] * F32(1e7) for i in range(ifins)], F32)
            finswt = np.array([stw[ipos - 1 + i] / xn2 / F32(3) for i in range(ifins)], F32)
        d = dict(n=n, m=m, freqnm=float(freqnm), dbeta=float(dbeta), wavenm=float(wavenm), radamp=float(radamp),
                 resont=float(resont), vdw=float(vdw), stark=float(stark), finest=finest.astype(np.float64),
                 finswt=finswt.astype(np.float64))
        xne = self.xne
        wty1 = 1.0 / (1.0 + xne / float(y1wht))
        y1scal = float(y1num) * self.y1s * wty1 + self.y1b * (1.0 - wty1)
        c1 = self.c1d * float(c1con) * y1scal
        c2 = self.c2d * float(c2con)
        g1 = f(6.77) * torch.sqrt(c1)
        lg = torch.log(torch.sqrt(c2) / c1)
        gnot = g1 * torch.clamp(f(.2114) + lg, min=0.0) * (1.0 - self.gcon1 - self.gcon2)
        d.update(c1=c1, c2=c2, g1=g1, gnot=gnot)
        self._line[key] = d
        return d

    def __call__(self, n, m, j, delw, dopph):
        """j: layer index tensor (nl, 1); delw: (nl, np) float64 [nm]; dopph: (n,) attached."""
        c = self.line(n, m)
        ext = self.ext
        fq = c["freqnm"]
        wn = c["wavenm"]
        # wavelength-only quantities (constants): computed with the exact path's REAL*4 roundings, since
        # DL = |FREQ - FREQNM| cancels (a float64 FREQ would change far-wing values by ~1e-5)
        with torch.no_grad():
            dw = delw.detach().to(D)
            delstark = (((-10.0 * dw) / wn) * fq).to(F).to(D)
            wl32 = (wn + dw * 10.0).to(F)
            freq32 = torch.div(torch.tensor(2.99792458e18, dtype=D, device=dw.device), wl32.to(D)).to(F)
            fqt = torch.tensor(fq, dtype=F, device=dw.device)
            dl = torch.abs(freq32 - fqt).to(D)
            df_ = (freq32 - fqt).to(D)
            freq = freq32.to(D)
        fo = self.fo[j]
        hwstk = c["stark"] * fo
        hwvdw = c["vdw"] * self.t3nhe[j] + (2.0 * c["vdw"]) * self.t3nh2[j]
        hwrad = c["radamp"]
        hwres = c["resont"] * self.xnfph1[j] * 2.0
        hwlor = hwres + hwvdw + hwrad
        hwdop = dopph[j]
        with torch.no_grad():
            nwid = torch.where((hwdop >= hwstk) & (hwdop >= hwlor), 1, torch.where(hwlor >= hwstk, 2, 3))
            hfwid = fq * torch.maximum(torch.maximum(hwdop, hwlor), hwstk)
            ifcore = dl <= hfwid
        dop = fq * hwdop
        zero = torch.zeros_like(dl)
        df = df_
        hd = zero
        for fe, fw in zip(c["finest"], c["finswt"]):
            dd = torch.abs(df - fe) / dop
            ddc = torch.clamp(dd, max=7.0)
            hd = torch.where(dd <= 7.0, hd + ext.fastex(ddc * ddc) * fw, hd)
        pi4, sqpi = f(3.14159), f(1.77245)
        if n == 1 and m == 2:
            hwres4 = hwres * 4.0
            hwlor4 = hwres4 + hwvdw + hwrad
            hhw = fq * hwlor4
            near = freq > (82259.105 - 4000.0) * 2.99792458e10
            hres_near = hwres4 * fq / pi4 / (dl * dl + hhw * hhw) * sqpi * dop
            hres_far = self._lyman_h2(freq, j) * sqpi * dop
            hres = torch.where(near, hres_near, hres_far)
            hrad = torch.where((freq > 2.4190611e15) & (freq < f(.77) * 3.288051e15),
                               c["radamp"] * c["freqnm"] / pi4 / (dl * dl + hhw * hhw) * sqpi * dop, zero)
            hvdw = hwvdw * fq / pi4 / (dl * dl + hhw * hhw) * sqpi * dop
            hvdw = torch.where(freq < 1.8e15, zero, hvdw)
            hlor = hres + hrad + hvdw
        else:
            hhw = fq * hwlor
            top = hhw
            if n == 1 and m <= 5:
                lo, hi = {3: (.885, .890), 4: (.936, .938), 5: (.959, .961)}.get(m, (None, None))
                if lo is not None:
                    win = (freq > f(lo) * 3.288051e15) & (freq < f(hi) * 3.288051e15)
                    top = torch.where(win, hhw - c["freqnm"] * c["radamp"], hhw)
            hlor = top / pi4 / (dl * dl + hhw * hhw) * sqpi * dop
        extra, main = self._stark(c, j, delstark, dop, freq)
        res_core1 = hd
        res_core2 = zero + hlor
        res_core3 = (zero + extra) + main
        res_wing = ((hd + hlor) + extra) + main
        return torch.where(ifcore, torch.where(nwid == 1, res_core1, torch.where(nwid == 2, res_core2, res_core3)),
                           res_wing)

    def _lyman_h2(self, fd, j):
        ch2 = torch.as_tensor(_t("hprof4", "cutoffh2").astype(np.float64), device=fd.device)
        spacing = f(200. * 2.99792458e10)
        f22 = f((82259.105 - 22000.0) * 2.99792458e10)
        lo = (ch2[1] - ch2[0]) / spacing * (fd - f22) + ch2[0]
        with torch.no_grad():
            icut = torch.trunc((fd - f22) / spacing).to(torch.int64).clamp(0, 89)
        cutfreq = icut.to(D) * spacing + f22
        hi = (ch2[icut + 1] - ch2[icut]) / spacing * (fd - cutfreq) + ch2[icut]
        cut = torch.where(fd < f22, lo, hi)
        cut = torch.pow(10.0, cut - 14.0) * self.xnfph1[j] * 2.0 / 2.99792458e10
        return torch.where(fd < 50000.0 * 2.99792458e10, torch.zeros_like(cut), cut)

    def _stark(self, c, j, delstark, dop, freq):
        ext = self.ext
        n, m = c["n"], c["m"]
        fo = self.fo[j]
        c1, c2, g1, gnot = c["c1"][j], c["c2"][j], c["g1"][j], c["gnot"][j]
        gcon1, gcon2 = self.gcon1[j], self.gcon2[j]
        beta = torch.abs(delstark) / fo * c["dbeta"]
        y1 = c1 * beta
        y2 = c2 * (beta * beta)
        skip = (y2 <= f(1e-4)) & (y1 <= f(1e-5))
        t90 = 90.0 * y1
        gam = g1 * (0.5 * ext.fastex(torch.clamp(y1, max=80.0)) + ext.vcse1f(y1) - 0.5 * ext.vcse1f(y2)) * \
            (1.0 - gcon1 / (1.0 + t90 * t90 * t90) - gcon2 / (1.0 + 2000.0 * y1))
        gam = torch.where(gam <= f(1e-20), torch.zeros_like(gam), gam)
        gam = torch.where(skip, gnot.expand_as(gam), gam)
        prqs = self.sofbet(beta, self.pp[j], n, m)
        dbeta = c["dbeta"]
        sqpi, pi4 = f(1.77245), f(3.14159)
        extra = torch.zeros_like(beta)
        if m <= 2:
            prqs = prqs * 0.5
            lo = freq < (f(82259.105) - 20000.0) * 2.99792458e10
            hi = freq > (f(82259.105) - 4000.0) * 2.99792458e10
            ch2p = torch.as_tensor(_t("hprof4", "cutoffh2plus").astype(np.float64), device=freq.device)
            f15 = f((f(82259.105) - 15000.0) * 2.99792458e10)
            spacing = f(100. * 2.99792458e10)
            a = (ch2p[1] - ch2p[0]) / spacing * (freq - f15) + ch2p[0]
            with torch.no_grad():
                icut = torch.trunc((freq - f15) / spacing).to(torch.int64).clamp(0, 109)
            cutfreq = icut.to(D) * spacing + f15
            b = (ch2p[icut + 1] - ch2p[icut]) / spacing * (freq - cutfreq) + ch2p[icut]
            cut = torch.where(freq < f15, a, b)
            cutv = torch.pow(10.0, cut - 14.0) / 2.99792458e10 * self.xnfph2[j]
            mid = cutv * sqpi * dop
            beta4000 = 4000.0 * 2.99792458e10 / fo * dbeta
            prqsp4000 = self.sofbet(beta4000, self.pp[j], n, m) * 0.5 / fo * dbeta
            c4000 = float(_folded_pow10(float(F32(-11.07) - F32(14.))))
            cut4000 = c4000 / 2.99792458e10 * self.xnfph2[j]
            high = cut4000 / prqsp4000 * prqs / fo * dbeta * sqpi * dop
            extra = torch.where(lo, extra, torch.where(hi, high, mid))
        gpos = gam > 0
        gs = torch.where(gpos, gam, torch.ones_like(gam))
        ff_ = torch.where(gpos, gs / pi4 / (gs * gs + beta * beta), torch.zeros_like(gam))
        p1 = (f(.9) * y1)
        p1 = p1 * p1
        fns = (p1 + f(.03) * torch.sqrt(torch.clamp(y1, min=0.0))) / (p1 + 1.0)
        return extra, (prqs * (1.0 + fns) + ff_) / fo * dbeta * sqpi * dop


# -------------------------------------------------------------------------------------------- the walks

def red_waves(win, start: int, count: int) -> np.ndarray:
    a = np.empty(count, np.float64)
    from .lines import powi_np
    a[0] = win.wbegin * float(powi_np(win.ratio, start - 1))
    a[1:] = win.ratio
    return np.multiply.accumulate(a)


def blue_waves(win, ib0: int, count: int) -> np.ndarray:
    from .lines import powi_np
    a = np.empty(count + 1, np.float64)
    a[0] = win.wbegin * float(powi_np(win.ratio, ib0 - 1))
    a[1:] = win.ratio
    return np.divide.accumulate(a)[1:]


class HydrogenLines:
    """The TYPE -1/-2 lines of fort.19 (others raise NotImplementedError)."""

    def __init__(self, lines19: np.ndarray, win, tabs, mode="exact"):
        self.lines = lines19
        self.win = win
        self.tabs = tabs
        self.mode = mode
        for ln in lines19:
            typ = int(ln["type"])
            if typ in (2, -5, -6):
                continue
            if typ not in (-1, -2):
                raise NotImplementedError(f"fort.19 line type {typ} is not supported by the differentiable path")
        ehyd = np.zeros(100)
        ehyd[1:8] = [82259.105, 97492.302, 102823.893, 105291.651, 106632.160, 107440.444, 107965.051]
        for k in range(9, 101):
            ehyd[k - 1] = 109678.764 - 109677.576 / (k * k)
        self.ehyd = ehyd
        self.alphahyd = 1.e7 / (ehyd[1:] - ehyd[:-1])
        # walk decisions: recorded on the first call (or after reset()), replayed afterwards
        self._rec: dict | None = None
        self._recording: dict | None = None

    def reset(self):
        """Forget the recorded walk decisions (the next buffer() call re-takes them)."""
        self._rec = None

    def buffer(self, x, q, cont32: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Contribution to BUFFER [n, LENGTH] (attached). q: LineSet.layer_quantities; cont32: decision
        continuum [n, LENGTH] (float32, detached)."""
        win = self.win
        n, L = x.n, win.length
        dev = x.t.device
        ext = Ext(self.tabs, self.mode)
        xn = x.xnfpel
        st = dict(t=x.t, xne=x.xne, xnfhe1=x.xnfhe[:, 0], xnfh2=x.xnfh2, xnfph1=xn[:, 0, 0], xnfph2=xn[:, 1, 0])
        hp = HProf4D(st, ext)
        self.cutcont = cont32.to(D) * win.cutoff
        # Inglis-Teller merging (XLINOP first call), REAL*4 formula
        xne32 = x.xne.to(F)
        inglis = f(1600.) / torch.pow(x.xne, f(F32(2.) / F32(15.)))
        nmerge = inglis - f(1.5)
        self.emergeh = 109677.576 / (nmerge * nmerge)
        del xne32
        buf = torch.zeros(n * L, dtype=D, device=dev)
        dopple0 = q["dop"][:, 0]
        xnfdop0 = q["qxnfdop"][:, 0]
        record = self._rec is None
        if record:
            self._recording = {}
        for il, ln in enumerate(self.lines):
            typ = int(ln["type"])
            if typ not in (-1, -2):
                continue
            self._il, self._iw = il, 0
            buf = self._line(buf, ln, x, q, hp, ext, dopple0, xnfdop0, chunk)
        if record:
            self._rec, self._recording = self._recording, None
        return buf.reshape(n, L)

    # --------------------------------------------------------------------------------- one line
    def _line(self, buf, ln, x, q, hp, ext, dopple0, xnfdop0, chunk):
        win = self.win
        n, L = x.n, win.length
        dev = x.t.device
        if np.any(win.velshift[:n] != 0):
            raise NotImplementedError("velocity fields with fort.19 hydrogen lines")
        nbuff = int(ln["nbuff"])
        nblo, nbup = int(ln["nblo"]), int(ln["nbup"])
        typ = int(ln["type"])
        wl = float(ln["wl"])
        gf = f(ln["gf"])
        elo = f(ln["elo"])
        kappa0 = gf * xnfdop0
        k = min(max(nbuff, 1), L) - 1
        kapmin = self.cutcont[:, L - 1 if nbup == 2 else k]
        kap32 = kappa0.detach().to(F)
        ok = ~(kap32 < kapmin.to(F))
        bolth = ext.fastex(elo * q["hckt"])
        dp = dopple0 / 1.4142 if typ == -2 else dopple0
        kappa0 = kappa0 * bolth
        ok = ok & ~((kappa0.detach().to(F)) < kapmin.to(F))
        lay = torch.nonzero(ok).squeeze(1)
        key = (self._il, "lay")
        if self._rec is not None:
            lay = self._rec[key]
        else:
            self._recording[key] = lay
        if not lay.numel():
            return buf
        ncon = int(ln["ncon"])
        k0 = kappa0

        def prof(nn, mm, lj, w, wlc, kk):
            dl = torch.as_tensor(w, device=dev)[None, :] - wlc
            return kk[lj][:, None] * hp(nn, mm, lj[:, None], dl.expand(lj.numel(), -1), dp)

        ehyd, alphahyd = self.ehyd, self.alphahyd
        cc = lambda lj, ib: self.cutcont[lj[:, None], ib[None, :]]     # noqa: E731
        if ncon == 0 or nbup == nblo + 1 or nbup == nblo + 2:
            beta = (ncon != 0 and nbup == nblo + 2)

            def evr(lj, ib, w):
                kv = prof(nblo, nbup, lj, w, wl, k0)
                z = torch.zeros(kv.shape, dtype=torch.bool, device=dev)
                stop = kv.detach() < cc(lj, ib)
                if beta:
                    stop = stop | torch.as_tensor(w > alphahyd[nblo - 1], device=dev)[None, :]
                return kv, z, stop, z

            def evb(lj, ib, w):
                kv = prof(nblo, nbup, lj, w, wl, k0)
                z = torch.zeros(kv.shape, dtype=torch.bool, device=dev)
                return kv, z, z, kv.detach() < cc(lj, ib)
            if wl > win.wlend:
                return self._walk_blue(buf, lay, min(L + 1, nbuff), evb, n, L, chunk)
            minred = max(1, nbuff)
            if not beta:
                buf, _ = self._walk_red(buf, lay, minred, evr, n, L, chunk)
                if minred == 1 or wl < win.wbegin:
                    return buf
                return self._walk_blue(buf, lay, min(L + 1, nbuff), evb, n, L, chunk)
            alpha_lim = alphahyd[nblo - 1]
            buf, ended_alpha = self._walk_red(buf, lay, minred, evr, n, L, chunk,
                                              track=lambda w: w > alpha_lim)
            l633, l622 = lay[ended_alpha], lay[~ended_alpha]
            if l633.numel() and nbuff >= 1:
                buf = self._walk_blue(buf, l633, nbuff, evb, n, L, chunk)
            if l622.numel() and not (minred == 1 or wl < win.wbegin):
                buf = self._walk_blue(buf, l622, min(L + 1, nbuff), evb, n, L, chunk)
            return buf
        # general case: series with merging into the continuum
        from ..tables import table
        conth = table("synthe.patched.for", "xlinop", "conth")
        wshift = 1.e7 / (conth[ncon - 1] - 109677.576 / 81.0 ** 2)
        emh = self.emergeh
        wmerge = 1.e7 / (conth[ncon - 1] - emh)
        wmerge = torch.where(wmerge < 0., torch.full_like(wmerge, wshift + wshift), wmerge)
        wcon = torch.clamp(wmerge, min=wshift)
        wtail = 1.e7 / (1.e7 / wcon - 500.)
        wcon = torch.clamp(wcon, max=wshift + wshift)
        wtail = torch.where(wtail < 0., wcon + wcon, wtail)
        wtail = torch.minimum(wcon + wcon, wtail)
        if win.ifvac == 0:
            raise NotImplementedError("air wavelengths with fort.19 hydrogen series (VACAIR of WCON)")

        def tail(kv, w, lj):
            wt = torch.as_tensor(w, device=dev)[None, :]
            tl = wt < wtail[lj][:, None]
            kt = kv * (wt - wcon[lj][:, None]) / (wtail[lj][:, None] - wcon[lj][:, None])
            return torch.where(tl, kt, kv)

        go_blue = True
        if not wl > win.wlend and not win.wbegin > float(alphahyd[nblo - 1]):
            redcut = 1.e7 / (109678.764 - 109677.576 / (nbup - 0.8) ** 2 - ehyd[nblo - 1])
            wlm1 = 1.e7 / (ehyd[nbup - 2] - ehyd[nblo - 1])
            wlm2 = 1.e7 / (ehyd[nbup - 3] - ehyd[nblo - 1])
            k0red = (k0 * f(hfnm(nblo, nbup - 2)) / f(hfnm(nblo, nbup)) /
                     (ehyd[nbup - 3] - ehyd[nblo - 1]) * (ehyd[nbup - 1] - ehyd[nblo - 1]))

            def evr(lj, ib, w):
                kv = tail(prof(nblo, nbup, lj, w, wl, k0), w, lj)
                wt = torch.as_tensor(w, device=dev)[None, :]
                below = (wt < wcon[lj][:, None]).detach()
                stop = (wt > wlm1).expand_as(kv) | (kv.detach() < cc(lj, ib))
                rc = wt > redcut
                if bool(rc.any()):
                    kr = tail(prof(nblo, nbup - 2, lj, w, wlm2, k0red), w, lj)
                    stop = stop | (rc & (kr.detach() >= kv.detach()))
                return kv, below.expand_as(kv), stop, torch.zeros_like(stop)
            minred = max(1, nbuff)
            buf, _ = self._walk_red(buf, lay, minred, evr, n, L, chunk)
            if minred == 1 or wl < win.wbegin:
                go_blue = False
        if not go_blue:
            return buf
        bluecut = 1.e7 / (109678.764 - 109677.576 / (nbup + 0.8) ** 2 - ehyd[nblo - 1])
        wlp1 = 1.e7 / (ehyd[nbup] - ehyd[nblo - 1])
        wlp2 = 1.e7 / (ehyd[nbup + 1] - ehyd[nblo - 1])
        k0blue = (k0 * f(hfnm(nblo, nbup + 2)) / f(hfnm(nblo, nbup)) /
                  (ehyd[nbup + 1] - ehyd[nblo - 1]) * (ehyd[nbup - 1] - ehyd[nblo - 1]))

        def evb2(lj, ib, w):
            kv = tail(prof(nblo, nbup, lj, w, wl, k0), w, lj)
            wt = torch.as_tensor(w, device=dev)[None, :]
            stop_b = (wt < wcon[lj][:, None]).detach() | (wt < wlp1)
            bc = wt < bluecut
            if bool(bc.any()):
                kb = tail(prof(nblo, nbup + 2, lj, w, wlp2, k0blue), w, lj)
                stop_b = stop_b | (bc & (kb.detach() > kv.detach()))
            stop_b = stop_b.expand_as(kv)
            return kv, torch.zeros_like(stop_b), stop_b, kv.detach() < cc(lj, ib)
        return self._walk_blue(buf, lay, min(L + 1, nbuff), evb2, n, L, chunk)

    # --------------------------------------------------------------------------------- walks
    def _walk(self, buf, lay, ibuf, waves, ev, L, chunk, track=None):
        """Generic sequential walk (XLinop.walk): per layer, skip -> no add, stop_before -> stop without adding,
        else add, stop_after -> stop after adding. ibuf 1-based point numbers in walk order."""
        dev = buf.device
        key = (self._il, self._iw)
        self._iw += 1
        if self._rec is not None:                       # replay the recorded decisions
            steps, ended_track = self._rec[key]
            for alive, ib, w, add in steps:
                kap = ev(alive, ib, w)[0]
                shape = kap.shape
                rows = alive.unsqueeze(1).expand(shape)[add]
                cols = ib.unsqueeze(0).expand(shape)[add]
                buf = buf.index_add(0, rows * L + cols, kap[add])
            return buf, ended_track
        steps = []
        alive = lay.clone()
        ended_track = torch.zeros(lay.numel(), dtype=torch.bool, device=dev)
        pos = 0
        npts = len(ibuf)
        while alive.numel() and pos < npts:
            end_ = min(pos + chunk, npts)
            ib = torch.as_tensor(ibuf[pos:end_] - 1, device=dev)
            w = waves[pos:end_]
            kap, skip, sb, sa = ev(alive, ib, w)
            shape = kap.shape
            skip = skip.expand(shape)
            sb = sb.expand(shape) & ~skip
            sa = sa.expand(shape) & ~skip
            m = ib.numel()
            ar = torch.arange(m, device=dev)
            big = torch.full(shape, m, device=dev, dtype=torch.long)
            fsb = torch.where(sb, ar, big).min(dim=1).values
            fsa = torch.where(sa, ar + 1, big).min(dim=1).values
            end = torch.minimum(fsb, fsa)
            if track is not None:
                tw = torch.as_tensor(track(w), device=dev)
                hit = (fsb < m) & (fsb <= fsa) & tw[fsb.clamp(max=m - 1)]
                idx = torch.searchsorted(lay, alive[hit])
                ended_track[idx] = True
            add = (ar.unsqueeze(0) < end.unsqueeze(1)) & ~skip
            rows = alive.unsqueeze(1).expand(shape)[add]
            cols = ib.unsqueeze(0).expand(shape)[add]
            buf = buf.index_add(0, rows * L + cols, kap[add])
            steps.append((alive, ib, w, add))
            alive = alive[end >= m]
            pos += m
        self._recording[key] = (steps, ended_track)
        return buf, ended_track

    def _walk_red(self, buf, lay, minred, ev, n, L, chunk, track=None):
        if minred > L:
            return buf, torch.zeros(lay.numel(), dtype=torch.bool, device=buf.device)
        cnt = L - minred + 1
        ibuf = minred + np.arange(cnt, dtype=np.int64)
        return self._walk(buf, lay, ibuf, red_waves(self.win, minred, cnt), ev, L, chunk, track)

    def _walk_blue(self, buf, lay, ib0, ev, n, L, chunk):
        if ib0 - 1 < 1:
            return buf
        cnt = ib0 - 1
        ibuf = (ib0 - 1) - np.arange(cnt, dtype=np.int64)
        return self._walk(buf, lay, ibuf, blue_waves(self.win, ib0, cnt), ev, L, chunk)[0]
