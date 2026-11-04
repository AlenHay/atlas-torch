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


def _nr(a, b):
    from .rcpps import div_nr
    return div_nr(a if isinstance(a, torch.Tensor) else torch.full_like(b, float(a)), b)


def sofbet(b: torch.Tensor, p_eff: torch.Tensor, index: int, fast: bool = False) -> torch.Tensor:
    """SOFBET(B,P,N,M) given the effective P of its cached tables (see module doc).
    fast: the stock dfsynthe.exe forms (sofbet_, same in SOFBETM/SOFBETP): 1.5/SB and 27./B2 by rcpps + Newton
    (rcpps.div_nr), B>=500: (27/B2+1.5/SB)/B2; B<25.12: ((PR1-PR2)*WT+PR2)*CORR; otherwise
    ((27*(1/B2)+Q15)*((1+QD)+CD))*(1/B2) with DD/(CC+B*SB) = QD+CD and 1.5/SB = Q15 split as the Newton step's
    q0 and correction (added to 1 and to 27*(1/B2) term by term)."""
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
    if not fast:
        v_lo = (pr1 * wt + pr2 * (one - wt)) * corr
        return torch.where(big, far, torch.where(b < L("25.12"), v_lo, v_hi))
    from .rcpps import rcpps
    nrsum = (_nr(27.0, b2) + _nr(L("1.5"), sb)) / b2          # lanes [1.5/SB, 27/B2] summed lane1+lane0
    far = nrsum
    pr2 = torch.where(b >= 8.0, nrsum, torch.zeros_like(b))
    v_lo = ((pr1 - pr2) * wt + pr2) * corr
    x = b * sb + cc
    rx, rs = rcpps(x), rcpps(sb)
    qd, q15 = dd * rx, L("1.5") * rs
    cd, c15 = (dd - x * qd) * rx, (L("1.5") - sb * q15) * rs
    rb2 = rdiv(1.0, b2)
    v_hi = (((27.0 * rb2) + q15) + c15) * ((1.0 + qd) + cd) * rb2
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


def vcse1f_fast(x: torch.Tensor, logf, expf) -> torch.Tensor:
    """VCSE1F as inlined in the stock HPROF4 (X<=.01: (X-.577215)-LOG(X); X<=30: (EXP(-X)*NUM)/(X*DEN));
    logf/expf: the stock binary's libimf functions."""
    one = torch.ones_like(x)
    xs = torch.where(x > 0, x, one)
    lg = logf(xs)
    r1 = (x + L("-.577215")) - lg
    r2 = (((((x * L(".00107857") + L("-.00976004")) * x + L(".05519968")) * x + L("-.24991055")) * x
            + L(".99999193")) * x) + (L("-.57721566") - lg)
    xe = torch.where(x <= 30.0, x, one)
    num = (x + L("2.334733")) * x + L(".25062")
    den = (x + L("3.330657")) * x + L("1.681534")
    r3 = (expf(-xe) * num) / (x * den)
    z = torch.zeros_like(x)
    return torch.where(x <= 0, z, torch.where(x <= L(".01"), r1, torch.where(x <= 1.0, r2,
                                                                                torch.where(x <= 30.0, r3, z))))


@lru_cache(maxsize=None)
def _hfnm_fast(n: int, m: int, fl: Flavor) -> np.float32:
    """HFNM as compiled into the stock dfsynthe.exe (hfnm_ 0x40fb60): GINF=XN**(-.71)*.2027 (libimf powf),
    WTC=(XN-1)*((-1/XN**3)*2.4)+.45, FK=X**3*FKN with X=XM/((XN+XM)*XMN), and
    FNM=((.778-GCA/XM)+((1-XMN12)*((-.222-GCA/XM)+GINF))/(WTC+XMN12))*FK."""
    S = lambda x: f4(x)  # noqa: E731

    def pw(a, b):
        from . import imf
        y = imf.powf(np.array([a], f4), np.array([b], f4)) if fl.libm is None else None
        return f4(y[0]) if y is not None else \
            f4(fl.powf(torch.tensor([a], dtype=F32), torch.tensor([b], dtype=F32))[0])
    xn = f4(n)
    ginf = S(pw(xn, lit("-.71")) * lit(".2027"))
    gca = S(lit(".124") / xn)
    fkn = S(xn * lit("1.9603"))
    wtc = S(S(S(xn + f4(-1)) * S(S(f4(-1) / S(S(xn * xn) * xn)) * lit("2.4"))) + lit(".45"))
    xm = f4(m)
    xmn = f4(m - n)
    x = S(xm / S(S(xn + xm) * xmn))
    fk = S(S(S(x * x) * x) * fkn)
    x12 = pw(xmn, lit("1.2"))
    g = S(gca / xm)
    t2 = S(S(S(f4(1) - x12) * S(S(lit("-.222") - g) + ginf)) / S(wtc + x12))
    return S(S(S(lit(".778") - g) + t2) * fk)


def hfnm(n: int, m: int, fl: Flavor) -> np.float32:
    """HFNM (REAL*4).  XN**3 and (..)**3 are integer powers (flavour dependent)."""
    if m <= n:
        return f4(0)
    if fl.fastmath:
        return _hfnm_fast(n, m, fl)
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
        if fl.fastmath:
            self._depth_fast(st)

    # ------------------------------------------------------------------ ifx -fp-model fast: constants, libimf
    @staticmethod
    @lru_cache(maxsize=None)
    def _fast_consts() -> dict:
        """Constants ifx folds in the stock HPROF4 (single precision; bit patterns from .rodata 0x4890c0..0x489198)."""
        S = lambda x: f4(x)  # noqa: E731
        c10, c18, rydh = lit("2.99792458E10"), lit("2.99792458E18"), f4(_tab("hprof4", "rydh"))
        kl = S(lit("1.77245") / lit("3.14159"))
        k = dict(w=S(c18 / rydh), r=S(f4(1) / lit("12.5664")), s=S(lit("3.579E-24") * lit("0.5773")),
                 t=S(lit("1.6678E-18") * rydh), l=kl, res=S(kl * f4(8)), pi=S(f4(1) / lit("3.14159")),
                 c2=S(f4(2) / c10))
        for nm, d, sp in (("22", 22000.0, 200.0), ("15", 15000.0, 100.0)):
            fr = S(S(lit("82259.105") - f4(d)) * c10)
            spc = S(f4(sp) * c10)
            ch = _tab("hprof4", "cutoffh2" if nm == "22" else "cutoffh2plus").astype(f4)
            a = S(S(ch[1] - ch[0]) / spc)
            k.update({"nf" + nm: S(-fr), "nsp" + nm: S(-spc), "r" + nm: S(f4(1) / spc), "c" + nm: S(-S(fr * S(f4(1) / spc))),
                      "a" + nm: a, "b" + nm: S(ch[0] - S(fr * a)), "f" + nm: fr})
        want = dict(w=0x4463F0DA, r=0x3DA2F96B, s=0x181FDC73, t=0x3BB3B18D, l=0x3F106EAD, res=0x40906EAD, pi=0x3EA2F98C,
                    c2=0x2E92B3FA, nf22=0xD8CD60B9, nsp22=0xD4AE808F, r22=0x2A3BC7A7, c22=0xC396A5D3, a22=0x28B444BE,
                    b22=0xC2472E84, nf15=0xD8E53C4D, nsp15=0xD42E808F, r15=0x2ABBC7A7, c15=0xC42825D4,
                    a15=0x28F05BA8, b15=0xC289E4FC)
        for key, bits in want.items():
            assert int(f4(k[key]).view(np.uint32)) == bits, key
        return k

    def _imf(self, fn: str, *a):
        """Scalar libimf function of the stock binary on numpy float32 arrays (lines/imf.py); the flavour's own
        function without the Intel runtime or outside the stock flavour."""
        if self.fl.fastmath and self.fl.libm is None:
            from . import imf
            y = imf.map1(fn, *a)
            if y is not None:
                return y
        t = [torch.from_numpy(np.ascontiguousarray(x)) for x in a]
        return getattr(self.fl, fn)(*t).numpy()

    def _imf_t(self, fn: str, *a: torch.Tensor) -> torch.Tensor:
        dev = a[0].device
        return torch.from_numpy(np.ascontiguousarray(self._imf(fn, *(x.detach().cpu().numpy() for x in a)))).to(dev)

    def _cut_interp(self, freq, ch, nm: str, imax: int):
        """The H2/H2+ CUTOFF interpolation as compiled (FREQ*(1/SPACING) forms, folded constants)."""
        K = self._fast_consts()
        f = lambda x: torch.tensor(float(x), dtype=F32, device=freq.device)  # noqa: E731
        lo = freq * f(K["a" + nm]) + f(K["b" + nm])
        icut = torch.clamp(torch.trunc(freq * f(K["r" + nm]) + f(K["c" + nm])).to(I64), max=imax).clamp(min=0)
        hi = ((ch[icut + 1] - ch[icut]) * f(K["r" + nm])) * ((freq + f(K["nf" + nm])) + icut.to(F32) * f(K["nsp" + nm])) \
            + ch[icut]
        return torch.where(freq < f(K["f" + nm]), lo, hi)

    # ------------------------------------------------------------------ ifx -fp-model fast: depth vectors
    @staticmethod
    @lru_cache(maxsize=None)
    def _depth_consts():
        """Constants folded by ifx in the stock HPROF4 depth loop (.rodata 0x489098 / 0x48909c)."""
        k1 = f4(lit("1.25E-9") * lit("78940."))
        k2 = f4(lit("1.25E-9") / lit("5.96E-23"))
        assert k1.view(np.uint32) == 0x38CEEFBD and k2.view(np.uint32) == 0x5598998B
        return k1, k2

    def _depth_fast(self, st):
        """HPROF4's per-temperature depth vectors (DO 10) as the stock dfsynthe.exe computes them (hprof4_
        0x40b0c0..0x40b33a, same in HPROF4m/p): layers 1..NRHOX&~3 four at a time with XNE**.1666667 and T4**.3
        by __svml_powf4, the divisions by rcpps + one Newton step (rcpps.py div_nr: Y1S=T43/XNE16, C2D=(..)/XNE,
        GCON2=.2/(..); 1/T as the reciprocal form (1-T*r)*r+r; Y1B=2./D as (2-D*(2r))*r+2r; GCON1 as
        (q0+.2)+(A-D*q0)*r) and the square roots by rsqrtps + Newton (rcpps.sqrt_rsqrt; PP=XNE16*.08989/SQRT(T)
        as ((y*(-.5))*(XNE16*.08989))*((T*y)*y-3), y=rsqrtps(T)); the remaining layers scalar with __svml_powf1,
        divss and sqrtss (PP with rsqrtss as above).  FO**2/5.96E-23 and FO*78940. use the folded constants of
        _depth_consts with XNE16**4.  Needs the Intel runtime for SVML (otherwise the source forms stay)."""
        from . import svml
        from .rcpps import div_nr, rcpps, rsqrtps, sqrt_rsqrt
        t, xne = st["t"], st["xne"]
        nl = t.shape[0]
        nv = (nl // 4) * 4
        e16, e3 = f4(lit(".1666667")), f4(lit(".3"))
        xne_np = xne.cpu().numpy().astype(f4)
        t4 = t * L("1.E-4")
        t4_np = t4.cpu().numpy().astype(f4)
        p16 = (svml.powf4(xne_np[:nv], e16), svml.powf_xy(xne_np[nv:], e16))
        p3 = (svml.powf4(t4_np[:nv], e3), svml.powf_xy(t4_np[nv:], e3))
        if any(x is None for x in p16 + p3):
            return
        dev = t.device
        xne16 = torch.from_numpy(np.concatenate(p16)).to(dev)
        t43 = torch.from_numpy(np.concatenate(p3)).to(dev)
        k1, k2 = (float(k) for k in self._depth_consts())
        one = torch.ones_like(t)
        x4 = (xne16 * xne16) * (xne16 * xne16)
        fo = x4 * L("1.25E-9")
        pq = xne16 * L(".08989")
        y = rsqrtps(t)
        pp = ((y * (-0.5)) * pq) * (((t * y) * y) + (-3.0))
        # vector layers
        rt = (one - t * rcpps(t)) * rcpps(t) + rcpps(t)
        sq = sqrt_rsqrt(xne * rt)
        d = ((sq * L(".012")) * rt) + 1.0
        r = rcpps(d)
        y1b = ((2.0 - d * (r + r)) * r) + (r + r)
        a = sqrt_rsqrt(t4) * L(".09")
        r13 = float(f4(f4(1) / lit("1.E13")))           # XNE/1.E13 folded to XNE*(1/1.E13) (.rodata 0x29E12E14)
        d1 = xne * r13 + 1.0
        r1 = rcpps(d1)
        q0 = a * r1
        gcon1 = (q0 + L(".2")) + (a - d1 * q0) * r1
        vec = dict(pp=pp, fo=fo, y1b=y1b, y1s=div_nr(t43, xne16), c1d=(x4 * k1) * rt,
                   c2d=div_nr((x4 * k2) * fo, xne), gcon1=gcon1,
                   gcon2=div_nr(torch.full_like(t, L(".2")), xne * L("1.E-15") + 1.0))
        # scalar layers
        rts = rdiv(1.0, t)
        sca = dict(pp=pp, fo=fo, y1b=rdiv(2.0, ((sqrtf(xne * rts) * L(".012")) * rts) + 1.0), y1s=t43 / xne16,
                   c1d=(x4 * k1) * rts, c2d=((x4 * k2) * fo) / xne,
                   gcon1=((sqrtf(t4) * L(".09")) / (xne * r13 + 1.0)) + L(".2"),
                   gcon2=rdiv(L(".2"), xne * L("1.E-15") + 1.0))
        isv = torch.arange(nl, device=dev) < nv
        for k in vec:
            setattr(self, k, torch.where(isv, vec[k], sca[k]))
        self.t3nhe = st["xnfhe"][:, 0] * t43
        self.t3nh2 = t43 * st["xnfh2"]

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
        if fl.fastmath:
            K = self._fast_consts()
            if mmn > 3 or n > 4:
                xknm = S(S(xmn2 * lit("5.5E-5")) / S(S(S(lit(".13") / f4(mmn)) + f4(1)) * gnm))
            dbeta = S(c18 / S(S(freqnm * freqnm) * xknm))
            rw = S(f4(1) / S(K["w"] / gnm))
            c1con = S(S(S(xm2mn2 * gnm) * xknm) * rw)
            c2con = S(S(rw * xknm) * S(rw * xknm))
            radamp = S(asum[n - 1] + asum[m - 1]) if n != 1 else asuml[m - 1]
            radamp = S(S(radamp * K["r"]) / freqnm)
            rg = S(f4(1) / gnm)
            resont = S(hfnm(1, m, fl) / S(f4(1) - S(f4(1) / xm2)))
            if n != 1:
                resont = S(resont + S(hfnm(1, n, fl) / S(f4(1) - S(f4(1) / xn2))))
            resont = S(S(resont * K["s"]) * rg)
            vdw = S(S(self._imf("powf", np.array([base], f4), np.array([lit(".4")], f4))[0] * lit("4.45E-26")) * rg)
            stark = S(S(gnm * K["t"]) * xknm)
            rxn2 = S(f4(1) / xn2)
            if n > 4 or m > 10:
                pass
            elif mmn != 1:
                finswt = (_tab("hprof4", "stcpwt")[:ifins, n - 1].astype(f4) * rxn2).astype(f4)
            else:
                finswt = (_tab("hprof4", "stwtal")[k].astype(f4) * S(rxn2 * S(f4(1) / f4(3)))).astype(f4)
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
        fast = fl.fastmath
        wl = ((wl0 + delw) * 10.0).to(F32) if fast else (wl0 * 10.0 + delw * 10.0).to(F32)
        freq = f(lit("2.99792458E18")) / wl
        freq0 = rdiv(float(lit("2.99792458E17")), wl0).to(F32)
        dl = torch.abs(freq - freq0)
        fo = self.fo[j]
        hwstk = f(s["stark"]) * fo
        if fast:    # stock: VDW*((T3NH2+T3NH2)+T3NHE), HWRES+(HWVDW+HWRAD)
            hwvdw = f(s["vdw"]) * ((self.t3nh2[j] + self.t3nh2[j]) + self.t3nhe[j])
        else:
            hwvdw = f(s["vdw"]) * self.t3nhe[j] + (2.0 * f(s["vdw"])) * self.t3nh2[j]
        hwrad = f(s["radamp"]).expand_as(hwstk)
        hwres = (f(s["resont"]) * self.xnfph[j, 0]) * 2.0
        hwlor = hwres + (hwvdw + hwrad) if fast else (hwres + hwvdw) + hwrad
        hwdop = dopph
        nwid = torch.where((hwdop >= hwstk) & (hwdop >= hwlor), 1, torch.where(hwlor >= hwstk, 2, 3))
        hfwid = freqnm * torch.maximum(torch.maximum(hwdop, hwlor), hwstk)
        ifcore = torch.abs(dl) <= hfwid
        dop = freq0 * hwdop
        # Stark part, up to the SOFBET call
        if fast:    # stock: (Y1S*Y1NUM-Y1B)/(XNE/Y1WHT+1)+Y1B
            y1scal = (self.y1s[j] * f(s["y1num"]) - self.y1b[j]) / (self.xne[j] / f(s["y1wht"]) + 1.0) + self.y1b[j]
        else:
            wty1 = rdiv(1.0, 1.0 + self.xne[j] / f(s["y1wht"]))
            y1scal = (f(s["y1num"]) * self.y1s[j]) * wty1 + self.y1b[j] * (1.0 - wty1)
        c1 = (self.c1d[j] * f(s["c1con"])) * y1scal
        c2 = self.c2d[j] * f(s["c2con"])
        beta = (f(s["dbeta"]) * torch.abs(dl)) / fo if fast else (torch.abs(dl) / fo) * f(s["dbeta"])
        call51 = (~ifcore) | (nwid == 3)
        lyman_a = (m <= 2)
        call4000 = call51 & lyman_a & (freq > f(f4(lit("82259.105") - f4(4000.0)) * lit("2.99792458E10")))
        if fl.fastmath:      # stock code: BETA4000 = (4000*2.99792458E10 [folded] * DBETA) / FO
            beta4000 = (f(f4(f4(4000.0) * f4(C10))) * f(s["dbeta"])) / fo
        else:
            beta4000 = ((4000.0 * f(C10)) / fo) * f(s["dbeta"])
        return dict(s=s, j=j, freq=freq, freq0=freq0, dl=dl, hwvdw=hwvdw, hwrad=hwrad, hwres=hwres,
                    hwlor=hwlor, nwid=nwid, ifcore=ifcore, dop=dop, c1=c1, c2=c2, beta=beta, call51=call51,
                    call4000=call4000, beta4000=beta4000, fo=fo)

    # ------------------------------------------------------------------ ifx -fp-model fast: label 52
    @staticmethod
    @lru_cache(maxsize=None)
    def _c52() -> np.float32:
        """Folded constant (10.**(-11.07-14.))/2.99792458E10*1.77245 of the stock binary (.rodata 0x04D60B42)."""
        c = f4(f4(f4(10.0 ** np.float64(f4(lit("-11.07") - f4(14.0)))) / f4(C10)) * lit("1.77245"))
        assert c.view(np.uint32) == 0x04D60B42
        return c

    def _lya_h2plus_fast(self, q, p4000, prqs, fo, dbeta, dop, j, idx):
        """HPROF4 label 52 (Lyman alpha within 4000 cm-1 of the centre, outside the core) as the stock ifx
        build computes it.  With -fp-model fast ifx inlines PRQSP4000 = SOFBET(BETA4000)*.5/FO*DBETA into
        CUTOFF4000/PRQSP4000*PRQS/FO*DBETA*1.77245*DOP and evaluates ONE numerator/denominator pair
        (SIMD lanes, dfsynthe.exe 0x40c8ab..0x40c900, same in HPROF4m/HPROF4p):
            num = (FO*((C52*DOP)*DBETA)) * (XNFPH(J,2)*PRQS)      den = (FO*DBETA) * (SOFBET(BETA4000)*.5)
        At low electron density (small FO, SOFBET(BETA4000) ~ BETA4000**-2.5) both products fall below
        FLT_MIN and flush to zero (FTZ) -> 0/0 = NaN.  The precise/-O0 builds evaluate left to right and get 0
        (CUTOFF4000 flushes) or a small positive number.  The NaN makes HPROF4 NaN for every such sample: the
        wing never terminates (NaN .LT. x is false), MAX(KAPPA,1E-20) gives 1E-20 (maxss), and the final
        MAX(BUFFER,CONTINUUM) (maxps) replaces every NaN bin by the continuum -> the stock ODFs lose all line
        opacity in those bins (cool, low-pressure layers: 3.6-dex row shifts vs the precise builds)."""
        f = lambda x: torch.tensor(float(x), dtype=F32, device=self.dev)  # noqa: E731
        sb = sofbet(q["beta4000"], p4000, idx, fast=self.fl.fastmath)
        a = (f(self._c52()) * dop) * dbeta
        num = (fo * a) * (self.xnfph[j, 1] * prqs)
        den = (fo * dbeta) * (sb * 0.5)
        return num / den

    @staticmethod
    @lru_cache(maxsize=None)
    def _kh2plus() -> np.float32:
        """Folded constant (1./2.99792458E10)*1.77245 of the stock binary (.rodata 0x489198 = 0x2E82030B)."""
        c = f4(f4(f4(1.0) / f4(C10)) * lit("1.77245"))
        assert c.view(np.uint32) == 0x2E82030B
        return c

    def _lya_h2plus_cut_fast(self, cut, dop, j):
        """HPROF4 Lyman alpha quasi-H2+ cutoff, 4000-20000 cm-1 red of the centre, as the stock ifx build computes
        `CUTOFF=(10.**(CUTOFF-14.))/2.99792458E10*XNFPH(J,2); HPROF4=HPROF4+CUTOFF*1.77245*DOP` (dfsynthe.exe
        0x40c9a6..0x40c9cb, same in HPROF4m 0x40e0bf and HPROF4p 0x40fa86):
            ((XNFPH(J,2) * K) * DOP) * 10.**(CUTOFF-14.),   K = (1./2.99792458E10)*1.77245 folded.
        The source order forms 10.**(CUTOFF-14.)/2.99792458E10 first, which is subnormal (down to 2e-40) for the
        tabulated CUTOFF < -13.45, i.e. 13100-15000 cm-1 and beyond from the centre: under FTZ it flushes to 0 and
        the H2+ wing is lost (up to 5% of the opacity of dense hot layers, 15-step ODF jumps at 12589 K); the
        stock binary never forms that intermediate."""
        f = lambda x: torch.tensor(float(x), dtype=F32, device=self.dev)  # noqa: E731
        pw = self._imf_t("powf", torch.full_like(cut, 10.0), cut - 14.0)
        return ((self.xnfph[j, 1] * f(self._kh2plus())) * dop) * pw

    def _doppler_fast(self, s, freq, freq0, dop, zero):
        """Doppler core with fine structure (labels 32-33) as the stock binary computes it (hprof4_ 0x40bac5..
        0x40bf03, same in HPROF4m/p): D=|FREQ-(FINEST(I)+FREQ0)|*(1./DOP) (FINEST+FREQ0 is rounded at the
        2e15 Hz scale, unlike the source's (FREQ-FREQ0)-FINEST); for IFINS >= 4 the first IFINS&~3 components
        are summed in 4 SIMD lanes (lane l: I = l mod 4), reduced as (s1+s3)+(s0+s2), the rest added in order."""
        dev = self.dev
        f = lambda x: torch.tensor(float(x), dtype=F32, device=dev)  # noqa: E731
        rdop = f(1.0) / dop
        terms = []
        for fe, fw in zip(s["finest"], s["finswt"]):
            d = torch.abs(freq - (f(fe) + freq0)) * rdop
            idx = torch.trunc(d * 200.0 + 1.5).to(I64).clamp(1, self.h0tab.numel()) - 1
            terms.append((d <= 7.0, self.h0tab[idx] * f(fw)))
        nv = len(terms) & ~3
        if nv:
            lanes = [zero.clone() for _ in range(4)]
            for i in range(nv):
                ok, t = terms[i]
                lanes[i % 4] = torch.where(ok, lanes[i % 4] + t, lanes[i % 4])
            hd = (lanes[1] + lanes[3]) + (lanes[0] + lanes[2])
        else:
            hd = zero.clone()
        for ok, t in terms[nv:]:
            hd = torch.where(ok, hd + t, hd)
        return hd

    # ------------------------------------------------------------------ profile, stage 2, stock forms
    def _finish_fast(self, q: dict, p51, p4000, variant: str, hd: torch.Tensor) -> torch.Tensor:
        """HPROF4 after the Doppler core as the stock dfsynthe.exe computes it (hprof4_ 0x40bde9..0x40ca65, same
        in HPROF4m/p; constants _fast_consts, libimf logf/expf/powf):
          Lyman alpha: HWLOR=(HWVDW+HWRAD)+HWRES*4, near: ((DOP*FREQNM)*((HWRES/2)*K_RES))/(DEL**2+HHW**2),
            far: (1.77245*DOP)*((XNFPH(J,1)*(2/C))*10**(CUTOFF-14)) with the folded CUTOFF interpolation,
            RAD/VDW: ((HW*FREQNM)*(K_L*DOP))/(DEL**2+HHW**2), sum ((HD+VDW)+RES)+RAD;
          other lines: ((K_L*DOP)*TOP)/(DEL**2+HHW**2)+HD;
          Stark: Y1SCAL=(Y1S*Y1NUM-Y1B)/(XNE/Y1WHT+1)+Y1B, GNOT=((1-(GCON1+GCON2))*G1)*MAX(0,.2114+LOG(..)),
            GAM=(G1*((EXTAB-V2)*.5+V1))*(1-(GCON1/((90Y1)**3+1)+GCON2/(2000Y1+1))) (both divisions rcpps+NR),
            F=(GAM*(1/3.14159))/(GAM**2+BETA**2), HST=(((DOP*1.77245)*DBETA)*(PRQS*(1+FNS)+F))/FO; SOFBET fast."""
        from .rcpps import div_nr
        s, j = q["s"], q["j"]
        n, m = s["n"], s["m"]
        dev = self.dev
        K = self._fast_consts()
        f = lambda x: torch.tensor(float(x), dtype=F32, device=dev)  # noqa: E731
        freqnm = f(s["freqnm"])
        freq, dl, dop = q["freq"], q["dl"], q["dop"]
        ifcore, nwid = q["ifcore"], q["nwid"]
        zero = torch.zeros_like(freq)
        kdop = f(K["l"]) * dop
        hd = torch.where(ifcore & (nwid == 2), zero, hd)     # core, NWID=2: Lorentz only (GO TO 40)
        if n == 1 and m == 2:
            hwres_h = f(s["resont"]) * self.xnfph[j, 0]
            hwlor = (q["hwvdw"] + q["hwrad"]) + hwres_h * 8.0
            hhw = hwlor * freqnm
            den = dl * dl + hhw * hhw
            near = freq > f(f4(lit("82259.105") - f4(4000.0)) * lit("2.99792458E10"))
            res_near = ((dop * freqnm) * (hwres_h * f(K["res"]))) / den
            ch2 = torch.from_numpy(_tab("hprof4", "cutoffh2").astype(f4)).to(dev)
            cut = self._cut_interp(freq, ch2, "22", 89)
            pw = self._imf_t("powf", torch.full_like(cut, 10.0), cut + (-14.0))
            cut = (self.xnfph[j, 0] * f(K["c2"])) * pw
            cut = torch.where(freq < f(f4(50000.0) * lit("2.99792458E10")), zero, cut)
            res_far = (L("1.77245") * dop) * cut
            hprofres = torch.where(near, res_near, res_far)
            radwin = (freq > L("2.4190611E15")) & (freq < f(f4(lit(".77") * lit("3.28805E15"))))
            hprofrad = torch.where(radwin, ((q["hwrad"] * freqnm) * kdop) / den, zero)
            hprofvdw = torch.where(freq < L("1.8E15"), zero, ((q["hwvdw"] * freqnm) * kdop) / den)
            lor = ((hd + hprofvdw) + hprofres) + hprofrad
        else:
            hhw = q["hwlor"] * freqnm
            top = hhw
            if n == 1 and m <= 5:
                win = {3: (".885", ".890"), 4: (".936", ".938"), 5: (".959", ".961")}.get(m)
                if win is not None:
                    a1 = f(f4(lit(win[0]) * lit("3.288051E15")))
                    a2 = f(f4(lit(win[1]) * lit("3.288051E15")))
                    top = torch.where((freq > a1) & (freq < a2), hhw - freqnm * q["hwrad"], hhw)
            lor = ((kdop * top) / (dl * dl + hhw * hhw)) + hd
        # ---- Stark
        c1, c2, beta, fo = q["c1"], q["c2"], q["beta"], q["fo"]
        dbeta = f(s["dbeta"])
        g1 = sqrtf(c1) * 6.77
        lg = self._imf_t("logf", sqrtf(c2) / c1)
        gnot = ((1.0 - (self.gcon1[j] + self.gcon2[j])) * g1) * torch.clamp(lg + L(".2114"), min=0.0)
        y1 = c1 * beta
        y2 = c2 * (beta * beta)
        skip = (y2 <= L("1.E-4")) & (y1 <= L("1.E-5"))
        x100 = (100.0 * torch.clamp(y1, max=80.0)).to(F64)
        ni = (torch.sign(x100) * torch.floor(torch.abs(x100) + 0.5)).to(I64)
        self.extab_oob += int(((ni < 1) & ~skip & q["call51"]).sum())
        ex = torch.where(ni >= 1, self.extab[(ni - 1).clamp(0, self.extab.numel() - 1)], torch.zeros_like(y1))
        lgf = lambda x: self._imf_t("logf", x)  # noqa: E731
        exf = lambda x: self._imf_t("expf", x)  # noqa: E731
        inner = (ex - vcse1f_fast(y2, lgf, exf)) * 0.5 + vcse1f_fast(y1, lgf, exf)
        t90 = 90.0 * y1
        fg = 1.0 - (div_nr(self.gcon1[j], (t90 * t90) * t90 + 1.0) + div_nr(self.gcon2[j], 2000.0 * y1 + 1.0))
        gam = (g1 * inner) * fg
        if variant == "m":
            gam = gnot
        gam = torch.where(gam <= L("1.E-20"), zero, gam)
        gam = torch.where(skip, gnot, gam)
        idx = s["index"]
        prqs = sofbet(beta, p51, idx, fast=True)
        hs = None
        if m <= 2:
            prqs = prqs * 0.5
            f20 = f(f4(lit("82259.105") - f4(20000.0)) * lit("2.99792458E10"))
            f4000 = f(f4(lit("82259.105") - f4(4000.0)) * lit("2.99792458E10"))
            ch = torch.from_numpy(_tab("hprof4", "cutoffh2plus").astype(f4)).to(dev)
            cut = self._cut_interp(freq, ch, "15", 109)
            mid = self._lya_h2plus_cut_fast(cut, dop, j)
            hi4000 = self._lya_h2plus_fast(q, p4000, prqs, fo, dbeta, dop, j, idx)
            hs = torch.where(freq < f20, zero, torch.where(freq > f4000, hi4000, mid))
        F = torch.where(gam > 0, (gam * f(K["pi"])) / (gam * gam + beta * beta), zero)
        p1 = (L(".9") * y1) * (L(".9") * y1)
        fns = (p1 + L(".03") * sqrtf(y1)) / (p1 + 1.0)
        hst = (((dop * L("1.77245")) * dbeta) * ((prqs * (1.0 + fns)) + F)) / fo
        base = lor if hs is None else lor + hs
        out_noncore = base + hst
        hs0 = zero if hs is None else hs
        out_core = torch.where(nwid == 1, zero + hd, torch.where(nwid == 2, lor, (zero + hs0) + hst))
        return torch.where(ifcore, out_core, out_noncore)

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
        if fl.fastmath:
            hd = self._doppler_fast(s, freq, freq0, dop, zero)
        else:
            hd = zero.clone()
            for fe, fw in zip(s["finest"], s["finswt"]):
                d = torch.abs((freq - freq0) - f(fe)) / dop
                idx = torch.trunc(200.0 * d + 1.5).to(I64).clamp(1, self.h0tab.numel()) - 1
                hd = torch.where(d <= 7.0, hd + self.h0tab[idx] * f(fw), hd)
        if fl.fastmath:
            return self._finish_fast(q, p51, p4000, variant, hd)
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
            if fl.fastmath:
                mid = self._lya_h2plus_cut_fast(cut, dop, j)
            else:
                cut = (fl.powf(torch.full_like(cut, 10.0), cut - 14.0) / f(C10)) * self.xnfph[j, 1]
                mid = (cut * L("1.77245")) * dop
            prqsp4000 = (((sofbet(q["beta4000"], p4000, idx) * 0.5) / fo) * dbeta)
            cut4000 = (fl.powf(torch.full_like(cut, 10.0), torch.full_like(cut, f4(lit("-11.07") - f4(14.0))))
                       / f(C10)) * self.xnfph[j, 1]
            if fl.fastmath:
                hi4000 = self._lya_h2plus_fast(q, p4000, prqs, fo, dbeta, dop, j, idx)
            else:
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
