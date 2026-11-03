"""Hydrogen line profiles of DFSYNTHE (HPROF4 / HPROF4m / HPROF4p, SOFBET, VCSE1F, HFNM), REAL*4 exact,
vectorised over arbitrary batches of (layer, wavelength) points.

HPROF4m/HPROF4p are textual copies of HPROF4 with their own SAVE caches; the caches (per-ITEMP depth vectors,
per-(N,M) line set-up, HFNM) are value-transparent, so one implementation serves all three.  SOFBET's cache is
NOT transparent: when INDEX does not change it keeps the P-interpolated tables of the last recompute even if
P changed.  The caller therefore passes the effective P (`p_eff`) of every SOFBET call; `hprof4_prep` exposes
which SOFBET calls happen (and their B, INDEX) so the caller can run the state machine first.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ...tables import table
from .f32 import F32, F64, I64, Flavor, lit, rdiv, sqrtf

SRC = "dfsynthe.for"
f4 = np.float32
L = lambda s: float(lit(s))    # noqa: E731  REAL*4 literal as python float (exact in f32)
C10 = L("2.99792458E10")


def _tab(unit, var):
    return np.asarray(table(SRC, unit, var))


@lru_cache(maxsize=1)
def sofbet_tables():
    propbm = np.concatenate([_tab("sofbet", f"prob{k}") for k in range(1, 8)]).astype(f4)
    propbm = propbm.reshape(7, 15, 5).transpose(2, 1, 0)                 # (ip, j, index)
    c = np.stack([_tab("sofbet", f"c{k}") for k in range(1, 8)], axis=1).astype(f4)   # (ip, index)
    d = np.stack([_tab("sofbet", f"d{k}") for k in range(1, 8)], axis=1).astype(f4)
    return dict(propbm=propbm, c=c, d=d, pp=_tab("sofbet", "pp").astype(f4), beta=_tab("sofbet", "beta").astype(f4))


def sofbet_index(n: int, m: int) -> int:
    return n + m - 2 if (n <= 3 and m - n <= 2) else 7


def sofbet(b: torch.Tensor, p_eff: torch.Tensor, index: int) -> torch.Tensor:
    """SOFBET(B,P,N,M) given the effective P of its cached tables (see module doc)."""
    t = sofbet_tables()
    dev = b.device
    one = torch.tensor(1.0, dtype=F32, device=dev)
    big = ~(b < 500.0)
    sb = sqrtf(b)
    b2 = b * b
    far = (rdiv(L("1.5"), sb) + rdiv(27.0, b2)) / b2
    # P interpolation (recompute part)
    p_eff = torch.nan_to_num(p_eff, nan=0.0)   # NaN = no recompute yet (only for B >= 500 calls)
    im = torch.clamp(torch.trunc(5.0 * p_eff).to(I64) + 1, max=4)           # 1-based
    ip = im + 1
    pp = torch.from_numpy(t["pp"]).to(dev)
    wtpp = 5.0 * (p_eff - pp[im - 1])
    wtpm = one - wtpp
    ci = torch.from_numpy(t["c"][:, index - 1].copy()).to(dev)
    di = torch.from_numpy(t["d"][:, index - 1].copy()).to(dev)
    cc = ci[ip - 1] * wtpp + ci[im - 1] * wtpm
    dd = di[ip - 1] * wtpp + di[im - 1] * wtpm
    corr_hi = one + dd / (cc + b * sb)
    v_hi = (rdiv(L("1.5"), sb) + rdiv(27.0, b2)) / b2 * corr_hi
    beta = torch.from_numpy(t["beta"]).to(dev)
    # first J in 2..15 with B <= BETA(J)
    jj = torch.searchsorted(beta[1:].contiguous(), b.contiguous(), right=False) + 2   # 1-based J
    jj = jj.clamp(max=15)
    jm, jp = jj - 1, jj
    pb = torch.from_numpy(t["propbm"][:, :, index - 1].copy()).to(dev)     # (ip, j)
    prob_jp = pb[ip - 1, jp - 1] * wtpp + pb[im - 1, jp - 1] * wtpm
    prob_jm = pb[ip - 1, jm - 1] * wtpp + pb[im - 1, jm - 1] * wtpm
    wtbp = (b - beta[jm - 1]) / (beta[jp - 1] - beta[jm - 1])
    wtbm = one - wtbp
    corr = (one + prob_jp * wtbp) + prob_jm * wtbm
    wt = torch.clamp(torch.clamp(L(".5") * (10.0 - b), max=1.0), min=0.0)
    pr1 = torch.where(b <= 10.0, rdiv(8.0, 83.0 + (2.0 + L(".95") * b2) * b), torch.zeros_like(b))
    pr2 = torch.where(b >= 8.0, (rdiv(L("1.5"), sb) + rdiv(27.0, b2)) / b2, torch.zeros_like(b))
    v_lo = (pr1 * wt + pr2 * (one - wt)) * corr
    return torch.where(big, far, torch.where(b < L("25.12"), v_lo, v_hi))


def vcse1f(x: torch.Tensor, fl: Flavor) -> torch.Tensor:
    one = torch.ones_like(x)
    xs = torch.where(x > 0, x, one)
    lg = fl.logf(xs)
    r1 = (-lg - L(".577215")) + x
    r2 = (-lg - L(".57721566")) + x * (L(".99999193") + x * (L("-.24991055") + x * (L(".05519968") +
                                                                                 x * (L("-.00976004") + x * L(".00107857")))))
    r3 = (((x * (x + L("2.334733")) + L(".25062")) / (x * (x + L("3.330657")) + L("1.681534"))) / x) * \
        fl.expf(-torch.where(x <= 30.0, x, one))
    z = torch.zeros_like(x)
    return torch.where(x <= 0, z, torch.where(x <= L(".01"), r1, torch.where(x <= 1.0, r2,
                                                                                torch.where(x <= 30.0, r3, z))))


def hfnm(n: int, m: int, fl: Flavor) -> np.float32:
    """HFNM (REAL*4).  XN**3 and (..)**3 are integer powers (flavour dependent)."""
    if m <= n:
        return f4(0)
    p3 = lambda x: f4(fl.powi4(torch.tensor([x], dtype=F32), 3)[0])   # noqa: E731
    pw = lambda a, b: f4(fl.powf(torch.tensor([a], dtype=F32), torch.tensor([b], dtype=F32))[0])  # noqa: E731
    xn = f4(n)
    ginf = f4(lit(".2027") / pw(xn, lit(".71")))
    gca = f4(lit(".124") / xn)
    fkn = f4(xn * lit("1.9603"))
    wtc = f4(lit(".45") - f4(f4(lit("2.4") / p3(xn)) * f4(xn - f4(1))))
    xm = f4(m)
    xmn = f4(m - n)
    fk = f4(fkn * p3(f4(xm / f4(xmn * f4(xm + xn)))))
    xmn12 = pw(xmn, lit("1.2"))
    wt = f4(f4(xmn12 - f4(1)) / f4(xmn12 + wtc))
    return f4(fk * f4(f4(f4(1) - f4(wt * ginf)) - f4(f4(lit(".222") + f4(gca / xm)) * f4(f4(1) - wt))))


class Hydrogen:
    """Per-temperature depth vectors and per-(N,M) set-up of HPROF4, plus the vectorised profile."""

    def __init__(self, st: dict, xnfph: torch.Tensor, extab: torch.Tensor, h0tab: torch.Tensor, fl: Flavor):
        self.fl = fl
        self.dev = st["t"].device
        self.extab, self.h0tab = extab, h0tab
        t, xne = st["t"], st["xne"]
        xne16 = fl.powf(xne, torch.tensor(L(".1666667"), dtype=F32))
        self.pp = (xne16 * L(".08989")) / sqrtf(t)
        self.fo = fl.powi4(xne16, 4) * L("1.25E-9")
        self.y1b = rdiv(2.0, 1.0 + rdiv(L(".012"), t) * sqrtf(xne / t))
        t4 = t / 10000.0
        t43 = fl.powf(t4, torch.tensor(L(".3"), dtype=F32))
        self.y1s = t43 / xne16
        self.t3nhe = t43 * st["xnfhe"][:, 0]
        self.t3nh2 = t43 * st["xnfh2"]
        self.c1d = (self.fo * 78940.0) / t
        self.c2d = ((self.fo * self.fo) / L("5.96E-23")) / xne
        self.gcon1 = L(".2") + (L(".09") * sqrtf(t4)) / (1.0 + xne / L("1.E13"))
        self.gcon2 = rdiv(L(".2"), 1.0 + xne / L("1.E15"))
        self.xne = xne
        self.xnfph = xnfph
        self._setup = {}
        self.extab_oob = 0

    # ------------------------------------------------------------------ per line set-up
    def setup(self, n: int, m: int) -> dict:
        if (n, m) in self._setup:
            return self._setup[(n, m)]
        fl = self.fl
        S = lambda x: f4(x)  # noqa: E731
        mmn = m - n
        xn = f4(n); xn2 = S(xn * xn); xm = f4(m); xm2 = S(xm * xm); xmn2 = S(xm2 * xn2)
        xm2mn2 = S(xm2 - xn2); gnm = S(xm2mn2 / xmn2)
        if mmn <= 3 and n <= 4:
            xknm = f4(_tab("hprof4", "xknmtb")[n - 1, mmn - 1])
        if mmn > 3 or n > 4:
            xknm = S(S(S(lit("5.5E-5") / gnm) * xmn2) / S(f4(1) + S(lit(".13") / f4(mmn))))
        y1num = f4(320.0)
        if m == 2:
            y1num = f4(550.0)
        if m == 3:
            y1num = f4(380.0)
        y1wht = f4(1e13)
        if mmn <= 3:
            y1wht = f4(1e14)
        if mmn <= 2 and n <= 2:
            y1wht = f4(_tab("hprof4", "y1wtm")[n - 1, mmn - 1])
        rydh = f4(_tab("hprof4", "rydh"))
        freqnm = S(rydh * gnm)
        c18 = lit("2.99792458E18")
        dbeta = S(S(c18 / S(freqnm * freqnm)) / xknm)
        wavenm = S(c18 / freqnm)
        c1con = S(S(S(xknm / wavenm) * gnm) * xm2mn2)
        q = S(xknm / wavenm)
        c2con = S(q * q)
        asum, asuml = _tab("hprof4", "asum").astype(f4), _tab("hprof4", "asumlyman").astype(f4)
        radamp = S(asum[n - 1] + asum[m - 1])
        if n == 1:
            radamp = asuml[m - 1]
        radamp = S(radamp / lit("12.5664"))
        radamp = S(radamp / freqnm)
        resont = S(hfnm(1, m, fl) / S(f4(1) - S(f4(1) / xm2)))
        if n != 1:
            resont = S(resont + S(hfnm(1, n, fl) / S(f4(1) - S(f4(1) / xn2))))
        resont = S(S(S(resont * lit("3.579E-24")) * lit("0.5773")) / gnm)
        base = S(xm2 * S(S(f4(7) * xm2) + f4(5)))
        vdw = S(S(lit("4.45E-26") / gnm) * f4(fl.powf(torch.tensor([base]), torch.tensor([lit(".4")]))[0]))
        stark = S(S(lit("1.6678E-18") * freqnm) * xknm)
        if n > 4 or m > 10:
            finest = np.array([0.0], f4); finswt = np.array([1.0], f4)
        elif mmn != 1:
            ifins = int(_tab("hprof4", "lncomp")[n - 1])
            finest = (_tab("hprof4", "stcomp")[:ifins, n - 1].astype(f4) * f4(1e7)).astype(f4)
            finswt = (_tab("hprof4", "stcpwt")[:ifins, n - 1].astype(f4) / xn2).astype(f4)
        else:
            ifins = int(_tab("hprof4", "lnghal")[n - 1])
            ipos = int(_tab("hprof4", "istal")[n - 1])
            k = np.arange(ipos - 1, ipos - 1 + ifins)
            finest = (_tab("hprof4", "stalph")[k].astype(f4) * f4(1e7)).astype(f4)
            finswt = ((_tab("hprof4", "stwtal")[k].astype(f4) / xn2).astype(f4) / f4(3)).astype(f4)
        s = dict(n=n, m=m, y1num=y1num, y1wht=y1wht, freqnm=freqnm, dbeta=dbeta, c1con=c1con, c2con=c2con,
                 radamp=radamp, resont=resont, vdw=vdw, stark=stark, finest=finest, finswt=finswt,
                 index=sofbet_index(n, m))
        self._setup[(n, m)] = s
        return s

    # ------------------------------------------------------------------ profile, stage 1
    def prep(self, s: dict, j: torch.Tensor, wl0: torch.Tensor, delw: torch.Tensor, dopph: torch.Tensor) -> dict:
        """Everything of HPROF4 that does not depend on SOFBET.  j: 0-based layer per point; wl0, delw REAL*8;
        dopph REAL*4 per point.  Returns intermediates + SOFBET call descriptors (call51, b, call4000, b4000)."""
        fl = self.fl
        n, m = s["n"], s["m"]
        f = lambda x: torch.tensor(float(x), dtype=F32, device=self.dev)  # noqa: E731
        freqnm = f(s["freqnm"])
        wl = (wl0 * 10.0 + delw * 10.0).to(F32)
        freq = f(lit("2.99792458E18")) / wl
        freq0 = rdiv(float(lit("2.99792458E17")), wl0).to(F32)
        dl = torch.abs(freq - freq0)
        fo = self.fo[j]
        hwstk = f(s["stark"]) * fo
        hwvdw = f(s["vdw"]) * self.t3nhe[j] + (2.0 * f(s["vdw"])) * self.t3nh2[j]
        hwrad = f(s["radamp"]).expand_as(hwstk)
        hwres = (f(s["resont"]) * self.xnfph[j, 0]) * 2.0
        hwlor = (hwres + hwvdw) + hwrad
        hwdop = dopph
        nwid = torch.where((hwdop >= hwstk) & (hwdop >= hwlor), 1, torch.where(hwlor >= hwstk, 2, 3))
        hfwid = freqnm * torch.maximum(torch.maximum(hwdop, hwlor), hwstk)
        ifcore = torch.abs(dl) <= hfwid
        dop = freq0 * hwdop
        # Stark part, up to the SOFBET call
        wty1 = rdiv(1.0, 1.0 + self.xne[j] / f(s["y1wht"]))
        y1scal = (f(s["y1num"]) * self.y1s[j]) * wty1 + self.y1b[j] * (1.0 - wty1)
        c1 = (self.c1d[j] * f(s["c1con"])) * y1scal
        c2 = self.c2d[j] * f(s["c2con"])
        beta = (torch.abs(dl) / fo) * f(s["dbeta"])
        call51 = (~ifcore) | (nwid == 3)
        lyman_a = (m <= 2)
        call4000 = call51 & lyman_a & (freq > f(f4(lit("82259.105") - f4(4000.0)) * lit("2.99792458E10")))
        beta4000 = ((4000.0 * f(C10)) / fo) * f(s["dbeta"])
        return dict(s=s, j=j, freq=freq, freq0=freq0, dl=dl, hwvdw=hwvdw, hwrad=hwrad, hwres=hwres,
                    hwlor=hwlor, nwid=nwid, ifcore=ifcore, dop=dop, c1=c1, c2=c2, beta=beta, call51=call51,
                    call4000=call4000, beta4000=beta4000, fo=fo)

    # ------------------------------------------------------------------ profile, stage 2
    def finish(self, q: dict, p51: torch.Tensor, p4000: torch.Tensor, variant: str = "4") -> torch.Tensor:
        """variant '4' (HPROF4), 'm' (HPROF4m), 'p' (HPROF4p).  HPROF4m differs: its statement
        `GAM=G1*(.5*EXTAB(...` (dfsynthe.for line 1786) starts in column 6, i.e. it is a continuation of the
        preceding `IF(Y2.LE.1.E-4.AND.Y1.LE.1.E-5)GO TO 51`, so ifx never assigns it and GAM stays GNOT."""
        fl = self.fl
        s, j = q["s"], q["j"]
        n, m = s["n"], s["m"]
        dev = self.dev
        f = lambda x: torch.tensor(float(x), dtype=F32, device=dev)  # noqa: E731
        freqnm = f(s["freqnm"])
        freq, freq0, dl, dop = q["freq"], q["freq0"], q["dl"], q["dop"]
        ifcore, nwid = q["ifcore"], q["nwid"]
        zero = torch.zeros_like(freq)
        # ---- Doppler core with fine structure
        hd = zero.clone()
        for fe, fw in zip(s["finest"], s["finswt"]):
            d = torch.abs((freq - freq0) - f(fe)) / dop
            idx = torch.trunc(200.0 * d + 1.5).to(I64).clamp(1, self.h0tab.numel()) - 1
            hd = torch.where(d <= 7.0, hd + self.h0tab[idx] * f(fw), hd)
        # ---- Lorentz
        if n == 1 and m == 2:
            hwres = q["hwres"] * 4.0
            hwlor = (hwres + q["hwvdw"]) + q["hwrad"]
            hhw = freqnm * hwlor
            den = dl * dl + hhw * hhw
            near = freq > f(f4(lit("82259.105") - f4(4000.0)) * lit("2.99792458E10"))
            res_near = (((hwres * freqnm) / L("3.14159")) / den * L("1.77245")) * dop
            spacing = f(f4(200.0) * lit("2.99792458E10"))
            f22 = f(f4(lit("82259.105") - f4(22000.0)) * lit("2.99792458E10"))
            ch2 = torch.from_numpy(_tab("hprof4", "cutoffh2").astype(f4)).to(dev)
            lo = (ch2[1] - ch2[0]) / spacing * (freq - f22) + ch2[0]
            icut = torch.clamp(torch.trunc((freq - f22) / spacing).to(I64), max=89).clamp(min=0)
            cutfreq = icut.to(F32) * spacing + f22
            hi = (ch2[icut + 1] - ch2[icut]) / spacing * (freq - cutfreq) + ch2[icut]
            cut = torch.where(freq < f22, lo, hi)
            cut = ((fl.powf(torch.full_like(cut, 10.0), cut - 14.0) * self.xnfph[j, 0]) * 2.0) / f(C10)
            cut = torch.where(freq < f(f4(50000.0) * lit("2.99792458E10")), zero, cut)
            res_far = (cut * L("1.77245")) * dop
            hprofres = torch.where(near, res_near, res_far)
            radwin = (freq > L("2.4190611E15")) & (freq < f(f4(lit(".77") * lit("3.28805E15"))))
            hprofrad = torch.where(radwin, (((q["hwrad"] * freqnm) / L("3.14159")) / den * L("1.77245")) * dop, zero)
            hprofvdw = (((q["hwvdw"] * freqnm) / L("3.14159")) / den * L("1.77245")) * dop
            hprofvdw = torch.where(freq < L("1.8E15"), zero, hprofvdw)
            hl = (hprofres + hprofrad) + hprofvdw
        else:
            hhw = freqnm * q["hwlor"]
            top = hhw
            if n == 1 and m <= 5:
                win = {3: (".885", ".890"), 4: (".936", ".938"), 5: (".959", ".961")}.get(m)
                if win is not None:
                    a1 = f(f4(lit(win[0]) * lit("3.288051E15")))
                    a2 = f(f4(lit(win[1]) * lit("3.288051E15")))
                    top = torch.where((freq > a1) & (freq < a2), hhw - freqnm * q["hwrad"], hhw)
            hl = ((top / L("3.14159")) / (dl * dl + hhw * hhw) * L("1.77245")) * dop
        # ---- Stark
        c1, c2, beta, fo = q["c1"], q["c2"], q["beta"], q["fo"]
        dbeta = f(s["dbeta"])
        g1 = 6.77 * sqrtf(c1)
        gnot = (g1 * torch.clamp(L(".2114") + fl.logf(sqrtf(c2) / c1), min=0.0)) * \
            ((1.0 - self.gcon1[j]) - self.gcon2[j])
        y1 = c1 * beta
        y2 = c2 * (beta * beta)
        skip = (y2 <= L("1.E-4")) & (y1 <= L("1.E-5"))
        x100 = (100.0 * torch.clamp(y1, max=80.0)).to(F64)
        ni = (torch.sign(x100) * torch.floor(torch.abs(x100) + 0.5)).to(I64)            # NINT
        # NINT(100*Y1)=0 reads EXTAB(0), the word before COMMON /EXTAB/: alignment padding (0.0) in every ifx
        # build of dfsynthe.for (checked with nm: /RHOX/ ends 24 bytes before extab_)
        self.extab_oob += int(((ni < 1) & ~skip & q["call51"]).sum())
        ex = torch.where(ni >= 1, self.extab[(ni - 1).clamp(0, self.extab.numel() - 1)], torch.zeros_like(y1))
        gam = g1 * ((0.5 * ex + vcse1f(y1, fl)) - 0.5 * vcse1f(y2, fl)) * \
            ((1.0 - self.gcon1[j] / (1.0 + fl.powi4(90.0 * y1, 3))) - self.gcon2[j] / (1.0 + 2000.0 * y1))
        if variant == "m":
            gam = gnot
        gam = torch.where(gam <= L("1.E-20"), zero, gam)
        gam = torch.where(skip, gnot, gam)
        idx = s["index"]
        prqs = sofbet(beta, p51, idx)
        hs = zero.clone()
        if m <= 2:
            prqs = prqs * 0.5
            f20 = f(f4(lit("82259.105") - f4(20000.0)) * lit("2.99792458E10"))
            f4000 = f(f4(lit("82259.105") - f4(4000.0)) * lit("2.99792458E10"))
            f15 = f(f4(lit("82259.105") - f4(15000.0)) * lit("2.99792458E10"))
            spacing = f(f4(100.0) * lit("2.99792458E10"))
            ch = torch.from_numpy(_tab("hprof4", "cutoffh2plus").astype(f4)).to(dev)
            lo = (ch[1] - ch[0]) / spacing * (freq - f15) + ch[0]
            icut = torch.clamp(torch.trunc((freq - f15) / spacing).to(I64), max=109).clamp(min=0)
            cutfreq = icut.to(F32) * spacing + f15
            hi = (ch[icut + 1] - ch[icut]) / spacing * (freq - cutfreq) + ch[icut]
            cut = torch.where(freq < f15, lo, hi)
            cut = (fl.powf(torch.full_like(cut, 10.0), cut - 14.0) / f(C10)) * self.xnfph[j, 1]
            mid = (cut * L("1.77245")) * dop
            prqsp4000 = (((sofbet(q["beta4000"], p4000, idx) * 0.5) / fo) * dbeta)
            cut4000 = (fl.powf(torch.full_like(cut, 10.0), torch.full_like(cut, f4(lit("-11.07") - f4(14.0))))
                       / f(C10)) * self.xnfph[j, 1]
            hi4000 = (((((cut4000 / prqsp4000) * prqs) / fo) * dbeta) * L("1.77245")) * dop
            hs = torch.where(freq < f20, zero, torch.where(freq > f4000, hi4000, mid))
        F = torch.where(gam > 0, (gam / L("3.14159")) / (gam * gam + beta * beta), zero)
        p1 = (L(".9") * y1) * (L(".9") * y1)
        fns = (p1 + L(".03") * sqrtf(y1)) / (p1 + 1.0)
        hst = ((((prqs * (1.0 + fns) + F) / fo) * dbeta) * L("1.77245")) * dop
        # ---- assemble in Fortran order (HPROF4=0; +Doppler; +Lorentz; +[H2+ cutoff]; +Stark)
        out_noncore = (((zero + hd) + hl) + hs) + hst
        out_core = torch.where(nwid == 1, zero + hd, torch.where(nwid == 2, zero + hl, (zero + hs) + hst))
        return torch.where(ifcore, out_core, out_noncore)
