"""Continuous opacities of the ATLAS9/KAPPA9 library: KAPP (continuum part) and every *OP routine it reaches
with the IFOP switches (HOP, H2PLOP, HMINOP, HRAYOP, HE1OP, HE2OP, HEMIOP, HERAOP, COOLOP, LUKEOP, HOTOP,
ELECOP, H2RAOP) and their helpers (XKARSAS, COULFF, LINTER, MAP1, CROSSHE, HE12*, SEATON, C1OP, MG1OP,
AL1OP, SI1OP + si1seaton*, FE1OP, CHOP, OHOP, H2COLLOP, N1OP, O1OP, MG2OP, SI2OP, CA2OP, COULX/COULBF1S).

Conventions: every routine takes (s, fq, n, v) -- State, frequency context (freq/freqlg python floats,
ehvkt/stim/bnu tensors [n]), number of layers NRHOX, variant -- and is vectorised over layers. FREQ is a
scalar in the Fortran, so frequency branches are Python branches. Literal kinds are exact: r4('13.595') is a
REAL*4 literal, f4(...) a REAL*4 product folded in binary32 (e.g. .77*3.289841E15).

The ITEMP/FREQ1 caches of the Fortran (BOLT arrays recomputed only when ITEMP changes, cross-sections only
when FREQ changes) are pure memoisation of functions of the current State: the port recomputes them.
Variant differences (results/fmap/variants.md, atlas9mem vs kappa9): HRAYOP (D vs E literals), HE2OP
(3.6919E8, 2.815E29, 1.31522E14 single in kappa9), C1OP (RYD single, 2.0761E15 threshold), HOTOP (kappa9
also sums XNFSI(J,2)+XNFS(J,2) into the singly ionised free-free term). Others are identical.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from . import libm
from ._data import equivalenced, r4, table
from .libm import ipow_const, ipow_var, rdiv
from .pfsaha import c4

F32 = np.float32


def f4(*ops) -> float:
    """REAL*4 constant expression folded in binary32, e.g. f4('.77','*','3.289841E15')."""
    return c4(*ops)


def _T(v, unit, var, dtype=torch.float64):
    return torch.from_numpy(np.ascontiguousarray(table(v, unit, var))).to(dtype)


@lru_cache(maxsize=None)
def _tab(v, unit, var):
    return table(v, unit, var)


def _eq(v, unit, var):
    return equivalenced(v, unit, var)


class Freq:
    """/FREQ/ for one frequency: freq, freqlg (floats), ehvkt, stim, bnu (tensors [n])."""

    def __init__(self, freq, freqlg, ehvkt, stim, bnu):
        self.freq, self.freqlg, self.ehvkt, self.stim, self.bnu = float(freq), float(freqlg), ehvkt, stim, bnu

    @classmethod
    def from_state(cls, s, n):
        if "freq.freq" not in s:            # routines that do not use /FREQ/ (ELECOP)
            z = torch.zeros(n, dtype=torch.float64)
            return cls(float("nan"), float("nan"), z, z, z)
        return cls(s.scalar("freq.freq"), s.scalar("freq.freqlg"), s["freq.ehvkt"][:n], s["freq.stim"][:n],
                   s["freq.bnu"][:n])

    @classmethod
    def derive(cls, s, n, freq):
        """The main program's lines 22-20: FREQLG=DLOG(FREQ), EHVKT=EXP(-FREQ*HKT), STIM=1-EHVKT,
        BNU=1.47439D-2*(FREQ/1.D15)**3*EHVKT/STIM."""
        hkt = s["temp.hkt"][:n]
        ehvkt = libm.exp(-freq * hkt)
        stim = 1.0 - ehvkt
        freq15 = freq / 1e15
        bnu = 1.47439e-2 * ipow_const(freq15, 3) * ehvkt / stim
        return cls(freq, math.log(freq), ehvkt, stim, bnu)


def _g(s, k, n):
    return s[k][:n]


# --------------------------------------------------------------------------------------- scalar helpers

@lru_cache(maxsize=None)
def _xk(v):
    return (_eq(v, "xkarsas", "freqn").reshape(-1, order="F"), _eq(v, "xkarsas", "xn").reshape(-1, order="F"),
            _eq(v, "xkarsas", "xl").reshape(-1, order="F"), _tab(v, "xkarsas", "ekarsas"))


_XK_LN10_D = 2.30258509299405
_XK_LN10_E = F32(r4("2.30258509299405"))
_XK_R = F32(r4("109677.576"))


def xkarsas(freq: float, zeff2: float, n: int, l: int, v: str = "atlas9", _fn15=None) -> float:
    """FUNCTION XKARSAS(FREQ,ZEFF2,N,L) (REAL*4 routine): Karzas & Latter hydrogenic bound-free cross-section."""
    fn, xn, xl, ek = _xk(v)
    freqlg = libm.log10(F32(freq / zeff2))
    if l >= n or n > 6:
        if n > 15:
            fn15 = np.zeros(29, np.float32) if _fn15 is None else _fn15
            fn15[28] = F32(math.log10(float(_XK_R) * 2.99792458e10 / ipow_var(n, 2)))
            if freqlg < fn15[28]:
                return 0.0
            i = 29
            for ii in range(2, 29):
                fn15[ii - 1] = F32(math.log10(float((ek[ii - 1] + F32(1.0) / F32(ipow_var(n, 2))) * _XK_R) * 2.99792458e10))
                if freqlg > fn15[ii - 1]:
                    i = ii
                    break
            x = (freqlg - fn15[i - 1]) / (fn15[i - 2] - fn15[i - 1]) * (xn[i - 2 + 14 * 29] - xn[i - 1 + 14 * 29]) + xn[i - 1 + 14 * 29]
            return math.exp(float(x) * _XK_LN10_D) / zeff2
        b = (n - 1) * 29
        if freqlg < fn[b + 28]:
            return 0.0
        i = 30
        for ii in range(2, 30):
            if freqlg > fn[b + ii - 1]:
                i = ii
                break
        x = (freqlg - fn[b + i - 1]) / (fn[b + i - 2] - fn[b + i - 1]) * (xn[b + i - 2] - xn[b + i - 1]) + xn[b + i - 1]
        return math.exp(float(x) * _XK_LN10_D) / zeff2
    b = (n - 1) * 29
    if freqlg < fn[b + 28]:
        return 0.0
    i = 30
    for ii in range(2, 30):
        if freqlg > fn[b + ii - 1]:
            i = ii
            break
    c = l * 29 + (n - 1) * 29 * 6
    x = (freqlg - fn[b + i - 1]) / (fn[b + i - 2] - fn[b + i - 1]) * (xl[c + i - 2] - xl[c + i - 1]) + xl[c + i - 1]
    return float(libm.exp(F32(x * _XK_LN10_E))) / zeff2


def coulbf1s(freq: float, z: float, v: str = "atlas9") -> float:
    """FUNCTION COULBF1S(FREQ,Z): Gaunt factor of hydrogenic 1s photoionisation."""
    g = _tab(v, "coulbf1s", "gaunt1s")
    if freq / ipow_const(z, 2) < 3.28805e15:
        return 0.0
    elog = math.log10(freq / ipow_const(z, 2) / r4("3.28805E15"))
    i = int(elog / r4(".02"))
    i = max(min(i + 1, 150), 1)
    return float(g[i - 1]) + (float(g[i]) - float(g[i - 1])) / r4(".02") * (elog - float(F32(i - 1) * F32(r4(".02"))))


def coulx(n: int, freq: float, z: float, v: str = "atlas9") -> float:
    """FUNCTION COULX(N,FREQ,Z) (not reached with BasicATLAS' IFOP; kept for completeness)."""
    a, b, c = _tab(v, "coulx", "a"), _tab(v, "coulx", "b"), _tab(v, "coulx", "c")
    if freq < z * z * 3.28805e15 / float(n * n):
        return 0.0
    cx = 2.815e29 / freq / freq / freq / float(ipow_var(n, 5)) * ipow_const(z, 4)
    if n > 6:
        return cx
    if n == 1:
        return cx * coulbf1s(freq, z, v)
    return cx * (float(a[n - 1]) + (float(b[n - 1]) + float(c[n - 1]) * (z * z / freq)) * (z * z / freq))


def seaton(freq: float, freq0: float, xsect: float, power: float, a: float) -> float:
    """FUNCTION SEATON(FREQ0,XSECT,POWER,A)."""
    return xsect * (a + (1.0 - a) * (freq0 / freq)) * math.sqrt(ipow_var(freq0 / freq, int(2.0 * power + r4(".01"))))


def coulff(s, fq: Freq, n: int, nz: int, v: str = "atlas9") -> torch.Tensor:
    """FUNCTION COULFF(J,NZ): hydrogenic free-free Gaunt factor, vectorised over J."""
    A = torch.from_numpy(_tab(v, "coulff", "a"))
    z4 = float(_tab(v, "coulff", "z4log")[nz - 1])
    tlog = _g(s, "temp.tlog", n)
    gamlog = r4("10.39638") - tlog / r4("1.15129") + z4
    igam = torch.clamp(torch.trunc(gamlog + 7.0).to(torch.int64), 1, 10)
    hvktlg = (fq.freqlg - tlog) / r4("1.15129") - r4("20.63764")
    ihv = torch.clamp(torch.trunc(hvktlg + 9.0).to(torch.int64), 1, 11)
    p = gamlog - (igam - 7).double()
    q = hvktlg - (ihv - 9).double()
    a = lambda i, k: A[i - 1, k - 1]
    return (1.0 - p) * ((1.0 - q) * a(igam, ihv) + q * a(igam, ihv + 1)) + p * ((1.0 - q) * a(igam + 1, ihv) + q * a(igam + 1, ihv + 1))


def linter(xold: torch.Tensor, yold: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
    """SUBROUTINE LINTER (NNEW points, independently: IOLD restarts at 2 for each call with NNEW=1)."""
    nold = xold.shape[-1]
    # first IOLD>=2 with XNEW < XOLD(IOLD), else NOLD
    iold = torch.searchsorted(xold.contiguous(), xnew.contiguous(), right=True) + 1     # 1-based index of first xold > xnew
    iold = torch.clamp(iold, 2, nold)
    x1, x0 = xold[iold - 1], xold[iold - 2]
    y1, y0 = yold[..., iold - 1], yold[..., iold - 2]
    return y0 + (y1 - y0) / (x1 - x0) * (xnew - x0)


# ------------------------------------------------------------------------------------------------- HOP

def hop(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE HOP: H I bound-free (n=1..8 + higher-level integral) and free-free."""
    tkev, rho, T = _g(s, "temp.tkev", n), _g(s, "state.rho", n), _g(s, "temp.t", n)
    xnfph, bhyd = s["ions.xnfph"][:n], s["depart.bhyd"][:n]
    bolt = []
    for N in range(1, 9):
        nn = float(N * N)
        b = libm.exp(rdiv(-(r4("13.595") - r4("13.595") / nn), tkev)) * 2.0 * nn * xnfph[:, 0] / rho
        if N <= 6:
            b = b * bhyd[:, N - 1]
        bolt.append(b)
    freet = _g(s, "state.xne", n) * s["ions.xnfh"][:n, 1] / rho / libm.sqrt(T)
    xr = xnfph[:, 0] * f4("2.", "/", "2.", "/", "13.595") * tkev / rho
    boltex = libm.exp(rdiv(-r4("13.427"), tkev)) * xr
    exlim = libm.exp(rdiv(-r4("13.595"), tkev)) * xr
    freq = fq.freq
    cont = [xkarsas(freq, 1.0, N, N, v) for N in range(1, 9)]
    freq3 = ipow_const(freq, 3)
    cfree = 3.6919e8 / freq3
    c = 2.815e29 / freq3
    ex = exlim / fq.ehvkt if freq < 4.05933e13 else boltex
    h = (cont[6] * bolt[6] + cont[7] * bolt[7] + (ex - exlim) * c + coulff(s, fq, n, 1, v) * freet * cfree) * fq.stim
    sh = h * fq.bnu
    for N in range(1, 7):
        h = h + cont[N - 1] * bolt[N - 1] * (1.0 - fq.ehvkt / bhyd[:, N - 1])
        sh = sh + cont[N - 1] * bolt[N - 1] * fq.bnu * fq.stim / bhyd[:, N - 1]
    return {"ahyd": h, "shyd": sh / h}


def h2plop(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE H2PLOP: H2+ bound-free and free-free (Mihalas fit)."""
    freq, fl = fq.freq, fq.freqlg
    if freq > 3.28805e15:
        return {}
    fr = -3.0233e3 + (3.7797e2 + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * fl) * fl) * fl) * fl
    es = -7.342e-3 + (-2.409e-15 + (1.028e-30 + (-4.230e-46 + (1.224e-61 - 1.351e-77 * freq) * freq) * freq) * freq) * freq
    xnfph = s["ions.xnfph"][:n]
    a = libm.exp(rdiv(-es, _g(s, "temp.tkev", n)) + fr) * xnfph[:, 0] * 2.0 * s["depart.bhyd"][:n, 0] * xnfph[:, 1] \
        / _g(s, "state.rho", n) * fq.stim
    return {"ah2p": a}


@lru_cache(maxsize=None)
def _hmin_tables(v):
    wavek = _tab(v, "hminop", "wavek")
    ff = _eq(v, "hminop", "ff")                       # FF(11,22)
    wfflog = np.array([math.log(91.134 / float(w)) for w in wavek])
    fflog = np.array([[math.log(float(ff[it, iw]) * 1e-26) for it in range(11)] for iw in range(22)])  # (22, 11)
    return (torch.from_numpy(wfflog), torch.from_numpy(fflog), torch.from_numpy(_tab(v, "hminop", "thetaff")),
            torch.from_numpy(_tab(v, "hminop", "wbf")), torch.from_numpy(_tab(v, "hminop", "bf")))


def hminop(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE HMINOP: H- bound-free (Wishart/Mathisen table, MAP1) and free-free (Bell & Berrington)."""
    from ...interp import map1
    wfflog, fflog, thetaff, wbf, bf = _hmin_tables(v)
    T, tkev, rho, xne = _g(s, "temp.t", n), _g(s, "temp.tkev", n), _g(s, "state.rho", n), _g(s, "state.xne", n)
    bmin, b1, xnfph1 = s["depart.bmin"][:n], s["depart.bhyd"][:n, 0], s["ions.xnfph"][:n, 0]
    theta = rdiv(5040.0, T)
    xhmin = libm.exp(rdiv(r4(".754209"), tkev)) / (2.0 * 2.4148e15 * T * libm.sqrt(T)) * bmin * b1 * xnfph1 * xne
    freq = fq.freq
    wave = 2.99792458e17 / freq
    wavelog = math.log(wave)
    fftlog = linter(wfflog, fflog.T, torch.tensor([wavelog], dtype=torch.float64))[:, 0]     # (11,)
    fftt = libm.exp(fftlog) / thetaff * 5040.0 * 1.380658e-16
    hminbf = 0.0
    if freq > 1.82365e14:
        hminbf = float(map1(wbf, bf, torch.tensor([wave], dtype=torch.float64))[0][0])
    fftheta = linter(thetaff, fftt, theta)
    hminff = fftheta * xnfph1 * 2.0 * b1 * xne / rho
    h = hminbf * 1e-18 * (1.0 - fq.ehvkt / bmin) * xhmin / rho
    ahmin = h + hminff
    shmin = (h * fq.bnu * fq.stim / (bmin - fq.ehvkt) + hminff * fq.bnu) / ahmin
    return {"ahmin": ahmin, "shmin": shmin}


def hrayop_xsect(freq: float, v="atlas9") -> float:
    """HRAYOP's cross-section per H atom (Gavrila 1967) at FREQ."""
    from ...interp import map1
    gm, gab, gbc, gcd = (_tab(v, "hrayop", k) for k in ("gavrilam", "gavrilamab", "gavrilambc", "gavrilamcd"))
    nu13 = r4("3.289841E13")
    sig = r4("6.65246E-25")
    if freq < nu13:
        return sig * ipow_const(float(gm[0]), 2) * ipow_const(freq / nu13, 4)
    if freq <= r4("2.4190611e15"):
        i = min(int(freq / nu13) + 1, 74)
        g = float(gm[i - 2]) + (float(gm[i - 1]) - float(gm[i - 2])) / nu13 * (freq - float(F32(i - 1) * F32(nu13)))
        return sig * ipow_const(g, 2)

    def band(tab, lo, step, step2, imax):
        base = f4(lo, "*", "3.289841E15")
        i = int((freq - base) / r4(step)) + 1
        i = min(i + 1, imax)
        off = float(F32(base) + F32(i - 2) * F32(r4(step2)))
        g = float(tab[i - 2]) + (float(tab[i - 1]) - float(tab[i - 2])) / r4(step) * (freq - off)
        return sig * ipow_const(g, 2)
    if freq < f4(".77", "*", "3.289841E15"):
        return 0.0
    if freq <= f4(".885", "*", "3.289841E15"):
        return band(gab, ".755", "1.644921E13", "1.6644921E13", 27)
    if freq < f4(".890", "*", "3.289841E15"):
        return 0.0
    if freq <= f4(".936", "*", "3.289841E15"):
        return band(gbc, ".890", "0.657968E13", "0.657968E13", 24)
    if freq < f4(".938", "*", "3.289841E15"):
        return 0.0
    if freq <= f4(".959", "*", "3.289841E15"):
        return band(gcd, ".938", "0.3289841E13", "0.3289841E13", 22)
    lyc, flyc = _tab(v, "hrayop", "gavrilalymancont"), _tab(v, "hrayop", "fgavrilalymancont")
    if v == "atlas9":
        if freq < r4(".961") * 3.289841e15:
            return 0.0
        if freq <= r4("1.000") * 3.289841e15:
            return sig * float(lyc[0])
        g = float(map1(torch.from_numpy(flyc), torch.from_numpy(lyc), torch.tensor([freq / 3.2898041e15], dtype=torch.float64))[0][0])
        return 6.65246e-25 * g
    if freq < f4(".961", "*", "3.289841E15"):
        return 0.0
    if freq <= f4("1.000", "*", "3.289841E15"):
        return sig * float(lyc[0])
    g = float(map1(torch.from_numpy(flyc), torch.from_numpy(lyc), torch.tensor([freq / r4("3.2898041E15")], dtype=torch.float64))[0][0])
    return sig * g


def hrayop(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE HRAYOP: Rayleigh scattering by H I."""
    x = hrayop_xsect(fq.freq, v)
    return {"sigh": x * s["ions.xnfph"][:n, 0] * 2.0 * s["depart.bhyd"][:n, 0] / _g(s, "state.rho", n)}


# ---------------------------------------------------------------------------------------------- helium

def crosshe(freq: float, v="atlas9") -> float:
    """FUNCTION CROSSHE: He I 1s photoionisation (Marr & West)."""
    if freq < 5.945209e15:
        return 0.0
    wave = 2.99792458e18 / freq
    if wave > 50.0:
        x = _tab(v, "crosshe", "x505")
        i = int(93.0 - (wave - 50.0) / 5.0)
        i = min(92, max(2, i))
        return ((wave - float((92 - i) * 5) - 50.0) / 5.0 * (x[i - 2] - x[i - 1]) + x[i - 1]) * 1e-18
    if wave > 20.0:
        x = _tab(v, "crosshe", "x50")
        i = int(17.0 - (wave - 20.0) / 2.0)
        i = min(16, max(2, i))
        return ((wave - float((16 - i) * 2) - 20.0) / 2.0 * (x[i - 2] - x[i - 1]) + x[i - 1]) * 1e-18
    if wave > 10.0:
        x = _tab(v, "crosshe", "x20")
        i = int(12.0 - (wave - 10.0) / 1.0)
        i = min(11, max(2, i))
        return ((wave - float((11 - i) * 1) - 10.0) / 1.0 * (x[i - 2] - x[i - 1]) + x[i - 1]) * 1e-18
    x = _tab(v, "crosshe", "x10")
    i = int(22.0 - wave / 0.5)
    i = min(21, max(2, i))
    return ((wave - float(F32(21 - i) * F32(0.5))) / 0.5 * (x[i - 2] - x[i - 1]) + x[i - 1]) * 1e-18


def _he12(freq, v, unit, fname, xname, e0, wn_ref, power, fit, has_res=True):
    if freq < r4(e0) * 2.99792458e10:
        return 0.0
    if has_res and freq > f4("2.4", "*", "109722.267") * 2.99792458e10:
        waveno = freq / 2.99792458e10
        ek = (waveno - r4(e0)) / r4("109722.267")
        return fit(waveno, ek)
    fr, xs = _tab(v, unit, fname), _tab(v, unit, xname)
    fl = math.log10(freq)
    i = 17
    for ii in range(2, 17):
        if fl > fr[ii - 1]:
            i = ii
            break
    if i == 17:
        x = (fl - fr[15]) / (fr[14] - fr[15]) * (xs[14] - xs[15]) + xs[15]   # loop fall-through: I=17 reads past
        raise RuntimeError("HE12 table fall-through")
    x = (fl - fr[i - 1]) / (fr[i - 2] - fr[i - 1]) * (xs[i - 2] - xs[i - 1]) + xs[i - 1]
    return math.exp(x * 2.30258509299405)


def he12s1s(freq, v="atlas9"):
    def fit(wn, ek):
        eps = 2.0 * (ek - r4("2.612316")) / r4(".00322")
        return r4(".008175") * libm.pow(484940.0 / wn, r4("2.71")) * 8.067e-18 * ipow_const(eps + r4("76.21"), 2) / (1.0 + ipow_const(eps, 2))
    return _he12(freq, v, "he12s1s", "freq1s", "x1s", "32033.214", None, None, fit)


def he12s3s(freq, v="atlas9"):
    def fit(wn, ek):
        eps = 2.0 * (ek - r4("2.47898")) / r4(".000780")
        return r4(".01521") * libm.pow(470310.0 / wn, r4("3.12")) * 8.067e-18 * ipow_const(eps - r4("122.4"), 2) / (1.0 + ipow_const(eps, 2))
    return _he12(freq, v, "he12s3s", "freq3s", "x3s", "38454.691", None, None, fit)


def he12p1p(freq, v="atlas9"):
    def fit(wn, ek):
        e1s = 2.0 * (ek - r4("2.446534")) / r4(".01037")
        e1d = 2.0 * (ek - r4("2.59427")) / r4(".00538")
        return r4(".0009487") * libm.pow(466750.0 / wn, r4("3.69")) * 8.067e-18 * (
            ipow_const(e1s - r4("29.30"), 2) / (1.0 + ipow_const(e1s, 2)) + ipow_const(e1d + r4("172.4"), 2) / (1.0 + ipow_const(e1d, 2)))
    return _he12(freq, v, "he12p1p", "freq1p", "x1p", "27175.76", None, None, fit)


def he12p3p(freq, v="atlas9"):
    return _he12(freq, v, "he12p3p", "freq3p", "x3p", "29223.753", None, None, None, has_res=False)


def he1op(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE HE1OP: He I bound-free (10 levels + n=4..27 hydrogenic) and free-free."""
    tkev, rho, T = _g(s, "temp.tkev", n), _g(s, "state.rho", n), _g(s, "temp.t", n)
    xnfphe1 = s["ions.xnfphe"][:n, 0]
    G, chi, hefreq = _tab(v, "he1op", "g"), _tab(v, "he1op", "chi"), _tab(v, "he1op", "hefreq")
    ryd = r4("109722.273") * 2.99792458e10
    bolt = [libm.exp(rdiv(-float(chi[N]), tkev)) * float(G[N]) * xnfphe1 / rho for N in range(10)]
    boltn = {}
    for N in range(4, 28):
        num = float(F32(r4("24.587")) * (F32(1.0) - F32(1.0) / F32(N * N)))       # REAL*4 product
        boltn[N] = libm.exp(rdiv(-num, tkev)) * 4.0 * float(N * N) * xnfphe1 / rho
    freet = _g(s, "state.xne", n) * s["ions.xnfhe"][:n, 1] / rho / libm.sqrt(T)
    xr = xnfphe1 * f4("4.", "/", "2.", "/", "13.595") * tkev / rho
    boltex = libm.exp(rdiv(-r4("23.730"), tkev)) * xr
    exlim = libm.exp(rdiv(-r4("24.587"), tkev)) * xr
    freq = fq.freq
    freq3 = ipow_const(freq, 3)
    cfree = 3.6919e8 / freq3
    c = 2.815e29 / freq3
    imin = 0
    for k in range(1, 11):
        if float(hefreq[k - 1]) <= freq:
            imin = k
            break
    trans = [0.0] * 11
    transn = {}
    if imin:
        fns = [None, lambda: crosshe(freq, v), lambda: he12s3s(freq, v), lambda: he12s1s(freq, v),
               lambda: he12p3p(freq, v), lambda: he12p1p(freq, v),
               lambda: xkarsas(freq, 1.236439, 3, 0, v), lambda: xkarsas(freq, 1.102898, 3, 0, v),
               lambda: xkarsas(freq, 1.045499, 3, 1, v), lambda: xkarsas(freq, 1.001427, 3, 2, v),
               lambda: xkarsas(freq, 0.9926, 3, 1, v)]
        for k in range(imin, 11):
            trans[k] = fns[k]()
        elim = r4("527490.06")
        cont = True
        for lev, e in ((5, "171135.000"), (4, "169087."), (3, "166277.546"), (2, "159856.069")):
            fhe = (elim - r4(e)) * (r4("2.99792458E10") if lev == 5 else 2.99792458e10)
            if freq < fhe:
                cont = False
                break
            trans[lev] = trans[lev] + xkarsas(freq, fhe / ryd, 1, 0, v)
        elim = r4("588451.59")
        for lev, e in ((10, "186209.471"), (9, "186101."), (8, "185564."), (7, "184864."), (6, "183236.")):
            fhe = (elim - r4(e)) * 2.99792458e10
            if freq < fhe:
                break
            trans[lev] = trans[lev] + xkarsas(freq, fhe / ryd, 1, 0, v)
        else:
            if freq >= 1.25408e16:
                for N in range(4, 28):
                    transn[N] = xkarsas(freq, float(F32(4.0) - F32(3.0) / F32(N * N)), 1, 0, v)
    ex = exlim / fq.ehvkt if freq < 2.055e14 else boltex
    he1 = (ex - exlim) * c
    if imin:
        for N in range(imin, 11):
            he1 = he1 + trans[N] * bolt[N - 1]
        if freq >= 1.25408e16:
            for N in range(4, 28):
                he1 = he1 + transn.get(N, 0.0) * boltn[N]
    a = (he1 + coulff(s, fq, n, 1, v) * freet * cfree) * fq.stim
    return {"ahe1": a}


def he2op(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE HE2OP: He II bound-free (n=1..9) and free-free."""
    tkev, rho, T = _g(s, "temp.tkev", n), _g(s, "state.rho", n), _g(s, "temp.t", n)
    xnfphe2 = s["ions.xnfphe"][:n, 1]
    bolt = [libm.exp(rdiv(-(r4("54.403") - r4("54.403") / float(N * N)), tkev)) * 2.0 * float(N * N) * xnfphe2 / rho
            for N in range(1, 10)]
    freet = _g(s, "state.xne", n) * s["ions.xnfhe"][:n, 2] / rho / libm.sqrt(T)
    xr = xnfphe2 * f4("2.", "/", "2.", "/", "13.595") * tkev / rho
    boltex = libm.exp(rdiv(-r4("53.859"), tkev)) * xr
    exlim = libm.exp(rdiv(-r4("54.403"), tkev)) * xr
    freq = fq.freq
    cont = [xkarsas(freq, 4.0, N, N, v) for N in range(1, 10)]
    freq3 = ipow_const(freq, 3)
    if v == "kappa9":
        cfree = r4("3.6919E8") / freq3 * 4.0
        c = f4("2.815E29", "*", "2.", "*", "2.") / freq3
        exth = r4("1.31522E14")
    else:
        cfree = 3.6919e8 / freq3 * 4.0
        c = 2.815e29 * 2.0 * 2.0 / freq3
        exth = 1.31522e14
    ex = exlim / fq.ehvkt if freq < exth else boltex
    he2 = (ex - exlim) * c
    for N in range(9):
        he2 = he2 + cont[N] * bolt[N]
    return {"ahe2": (he2 + coulff(s, fq, n, 2, v) * cfree * freet) * fq.stim}


def hemiop(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE HEMIOP: He- free-free (John 1968 fit)."""
    f = fq.freq
    a = 3.397e-46 + (-5.216e-31 + 7.039e-15 / f) / f
    b = -4.116e-42 + (1.067e-26 + 8.135e-11 / f) / f
    c = 5.081e-37 + (-8.724e-23 - 5.659e-8 / f) / f
    T = _g(s, "temp.t", n)
    return {"ahemin": (a * T + b + rdiv(c, T)) * _g(s, "state.xne", n) * s["ions.xnfphe"][:n, 0] / _g(s, "state.rho", n)}


def heraop(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE HERAOP: Rayleigh scattering by He I."""
    wave = 2.997925e18 / min(fq.freq, 5.15e15)
    ww = ipow_const(wave, 2)
    sig = 5.484e-14 / ww / ww * ipow_const(1.0 + (2.44e5 + 5.94e10 / (ww - 2.90e5)) / ww, 2)
    return {"sighe": sig * s["ions.xnfphe"][:n, 0] / _g(s, "state.rho", n)}


# ------------------------------------------------------------------------------------------ metals (cool)

def c1op(s, fq: Freq, n, v="atlas9"):
    """FUNCTION C1OP(J): C I bound-free (1100, 1240, 1444 A edges), vectorised over J."""
    tkev = _g(s, "temp.tkev", n)
    ryd = 109732.298 if v == "atlas9" else r4("109732.298")
    c1240 = 5.0 * libm.exp(rdiv(-r4("1.264"), tkev))
    c1444 = libm.exp(rdiv(-r4("2.683"), tkev))
    freq = fq.freq
    waveno = freq / 2.99792458e10
    x1100 = x1240 = x1444 = 0.0
    if freq >= 2.7254e15:
        x1100 = libm.pow(10.0, -r4("16.80") - (waveno - 90777.000) / 3.0 / ryd)
    if freq >= r4("2.4196E15"):
        xd0 = libm.pow(10.0, -r4("16.80") - (waveno - r4("80627.760")) / 3.0 / ryd)
        eps = (waveno - 93917.0) * 2.0 / 9230.0
        xd1 = (22e-18 * eps + 26e-18) / (ipow_const(eps, 2) + 1.0)
        eps = (waveno - 111130.0) * 2.0 / 2743.0
        xd2 = (-10.5e-18 * eps + 46e-18) / (ipow_const(eps, 2) + 1.0)
        x1240 = xd0 + xd1 + xd2
    if freq >= (2.0761e15 if v == "atlas9" else r4("2.0761E15")):
        xs0 = libm.pow(10.0, -r4("16.80") - (waveno - r4("69172.400")) / 3.0 / ryd)
        eps = (waveno - 97700.0) * 2.0 / 2743.0
        xs1 = (68e-18 * eps + 118e-18) / (ipow_const(eps, 2) + 1.0)
        x1444 = xs0 + xs1
    return x1100 * 9.0 + x1240 * c1240 + x1444 * c1444


def _peach_op(s, fq, n, v, unit, nfreq, tdiv, toff, nmax, fix14):
    """Shared body of MG1OP / SI2OP: Peach's bound-free tables interpolated in log T and log nu."""
    T, tlog = _g(s, "temp.t", n), _g(s, "temp.tlog", n)
    tlg = torch.from_numpy(_tab(v, unit, "tlg"))
    nt = torch.clamp(torch.clamp(torch.trunc(T / tdiv).to(torch.int64) - toff, max=nmax), min=1)
    dt = (tlog - tlg[nt - 1]) / (tlg[nt] - tlg[nt - 1])
    freqs = _tab(v, unit, "freqmg" if unit == "mg1op" else "freqsi")
    flog, peach = _tab(v, unit, "flog"), _tab(v, unit, "peach")
    N = 8
    for k in range(1, 8):
        if fq.freq > freqs[k - 1]:
            N = k
            break
    d = (fq.freqlg - flog[N - 1]) / (flog[N] - flog[N - 1])
    if N > 2:
        N = 2 * N - 2
    if fix14 and N == 14:
        N = 13
    d1 = 1.0 - d
    x = torch.from_numpy(np.array([peach[it, N] * d + peach[it, N - 1] * d1 for it in range(peach.shape[0])]))
    return libm.exp(x[nt - 1] * (1.0 - dt) + x[nt] * dt)


def mg1op(s, fq, n, v="atlas9"):
    """FUNCTION MG1OP(J): Mg I bound-free (Peach)."""
    return _peach_op(s, fq, n, v, "mg1op", 7, 1000.0, 3, 6, False)


def si2op(s, fq, n, v="atlas9"):
    """FUNCTION SI2OP(J): Si II bound-free (Peach), times 6."""
    return _peach_op(s, fq, n, v, "si2op", 7, 2000.0, 4, 5, True) * 6.0


def al1op(s, fq, n, v="atlas9"):
    """FUNCTION AL1OP(J)."""
    if fq.freq >= 1.443e15:
        return 6.5e-17 * libm.pow(1.443e15 / fq.freq, r4("4.4")) * 6.0
    return 0.0


def _si1seaton(freq, v, unit, lam, xs, c):
    fl, x = _tab(v, unit, lam), _tab(v, unit, xs)
    freqs = [c / float(w) for w in fl]
    if freq > freqs[0]:
        return 0.0
    i = len(freqs) + 1
    for ii in range(2, len(freqs) + 1):
        if freq > freqs[ii - 1]:
            i = ii
            break
    if i > len(freqs):
        raise RuntimeError("si1seaton table fall-through")
    xx = (freq - freqs[i - 1]) / (freqs[i - 2] - freqs[i - 1]) * (x[i - 2] - x[i - 1]) + x[i - 1]
    return xx * 1e-18


def si1op(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE SI1OP: Si I bound-free (33 levels, Karzas-Latter + Seaton/Opacity-Project fits) -> /BSI/ASI1."""
    hckt = _g(s, "temp.hckt", n)
    elev, glev = _tab(v, "si1op", "elev"), _tab(v, "si1op", "glev")
    bolt = [float(glev[i]) * libm.exp(-float(elev[i]) * hckt) for i in range(33)]
    z = 1.0
    freq = fq.freq
    freq3 = 2.815e29 / freq / freq / freq * ipow_const(z, 4)
    waveno = freq / 2.99792458e10
    X = [0.0] * 34
    E = lambda i: float(elev[i - 1])
    ryd = r4("109732.298")
    goto = None
    elim = r4("65939.18")
    xk = lambda q, nn, ll, i: xkarsas(freq, q / ryd * (elim - E(i)), nn, ll, v)
    plan = [(1, 16., 4, 2), (2, 16., 4, 3), (3, 16., 4, 2), (4, 16., 4, 2), (5, 16., 4, 2), (6, 16., 4, 3),
            (7, 16., 4, 2), (8, 16., 4, 2), (9, 9., 3, 2), (10, 9., 3, 2), (11, 9., 3, 2), (12, 16., 4, 1)]
    for i, q, nn, ll in plan:
        if waveno < elim - E(i):
            goto = 30
            break
        X[i] = xk(q, nn, ll, i)
    if goto is None:
        for i in range(13, 24):
            if waveno < elim - E(i):
                goto = 60 if i == 21 else 30
                break
            if i == 13:
                X[13] = _si1seaton(freq, v, "si1seaton3p3d3p", "flam3p3d3p", "x3p3d3p", 2.99792458e18)
            elif i == 14:
                X[14] = _si1seaton(freq, v, "si1seaton3p4p1d", "flam3p4p1d", "x3p4p1d", 2.99792458e18)
            elif i == 15:
                X[15] = 42.68e-18 * libm.pow(r4("15973.206") / waveno, r4("2.4"))
            elif i == 16:
                X[16] = _si1seaton(freq, v, "si1seaton3p4p3s", "flam3p4p3s", "x3p4p3s", 2.9979245e18)
            elif i == 17:
                if waveno >= r4("16810.969"):
                    X[17] = 15.67e-18 * libm.pow(r4("16810.969") / waveno, r4("0.7"))
                if waveno > 23803.0:
                    X[17] = 11.90e-18 * libm.pow(23803.0 / waveno, r4("3.7"))
            elif i == 18:
                X[18] = 18.7e-18 * libm.pow(r4("17777.641") / waveno, r4("2.5"))
            elif i == 19:
                X[19] = 37e-18 * ipow_const(r4("18587.564") / waveno, 2)
            elif i == 20:
                X[20] = 12.41e-18 * libm.pow(r4("18655.039") / waveno, r4("2.3"))
            elif i == 21:
                X[21] = 16.64e-18 * libm.pow(r4("20636.37") / waveno, r4("1.4"))
            elif i == 22:
                X[22] = 4.09e-18 * ipow_const(r4("24947.216") / waveno, 2)
            elif i == 23:
                X[23] = 1.25e-18 * ipow_const(r4("26079.180") / waveno, 2)
        else:
            goto = 30
    if goto == 30:
        # label 30: 3s3p3 levels relative to the 3s2 3p 2P1/2 limit
        elim = r4("65747.55")
        stop = False
        for k in (24, 25, 26, 27, 28):
            if waveno < elim - E(k):
                stop = True
                break
            if k == 24:
                eps = (waveno - 70000.0) * 2.0 / 6500.0
                res = (97e-18 * eps + 94e-18) / (ipow_const(eps, 2) + 1.0)
                X[24] = (37e-18 * libm.pow(r4("50353.180") / waveno, r4("2.40")) + res) / 3.0
            elif k == 25:
                eps = (waveno - 78600.0) * 2.0 / 13000.0
                res = (-r4("10.E-18") * eps + 77e-18) / (ipow_const(eps, 2) + 1.0)
                X[25] = (24.5e-18 * libm.pow(r4("59448.700") / waveno, r4("1.85")) + res) / 3.0
            else:
                if waveno <= 74000.0:
                    p_ = 72e-18 * libm.pow(r4("65524.393") / waveno, r4("1.90"))
                    X[k] = p_ * 2.0 / 3.0 if k == 27 else p_ / 3.0
                if waveno > 74000.0:
                    p_ = 93e-18 * libm.pow(r4("65524.393") / waveno, r4("4.00"))
                    X[k] = p_ * 2.0 / 3.0 if k == 27 else p_ / 3.0
        # label 40: 2P3/2 limit
        elim = f4("65747.55", "+", "287.45")
        for k in (24, 25, 26, 27, 28):
            if waveno < elim - E(k):
                break
            if k == 24:
                eps = (waveno - 70000.0) * 2.0 / 6500.0
                res = (r4("97.E-18") * eps + 94e-18) / (ipow_const(eps, 2) + 1.0)
                X[24] = X[24] + (37e-18 * libm.pow(r4("50640.630") / waveno, r4("2.40")) + res) * 2.0 / 3.0
            elif k == 25:
                eps = (waveno - 78600.0) * 2.0 / 13000.0
                res = (-10e-18 * eps + 77e-18) / (ipow_const(eps, 2) + 1.0)
                X[25] = X[25] + (24.5e-18 * libm.pow(r4("59448.700") / waveno, r4("1.85")) + res) * 2.0 / 3.0
            else:
                if waveno <= 74000.0:
                    X[k] = X[k] + 72e-18 * libm.pow(r4("65524.393") / waveno, r4("1.90")) * 2.0 / 3.0
                if waveno > 74000.0:
                    X[k] = X[k] + 93e-18 * libm.pow(r4("65524.393") / waveno, r4("4.00")) * 2.0 / 3.0
        # label 50: 3s2 3p 4s... limit
        elim = f4("65747.5", "+", "42824.35")
        degen = 3.0
        for k in (29, 30, 31, 32, 33):
            if waveno < elim - E(k):
                break
            if k == 33:
                X[33] = 52.4e-18 * libm.pow(r4("75423.767") / waveno, r4("3.5"))
            else:
                X[k] = xkarsas(freq, 9.0 / ryd * (elim - E(k)), 3, 1, v) * degen
    # label 60
    elim = r4("65747.55")
    gfactor = 6.0
    rz2 = ryd * ipow_const(z, 2)
    m = max(elim - rz2 / 25.0, elim - waveno)
    h = rdiv(freq3 * gfactor * 2.0 / 2.0, rz2 * hckt) * (libm.exp(-m * hckt) - libm.exp(-elim * hckt))
    for i in range(1, 34):
        h = h + X[i] * bolt[i - 1]
    return h * fq.stim


def fe1op(s, fq: Freq, n, v="atlas9"):
    """FUNCTION FE1OP(J): Fe I bound-free (48 levels, schematic)."""
    hkt = _g(s, "temp.hkt", n)
    G, E, wno = _tab(v, "fe1op", "g"), _tab(v, "fe1op", "e"), _tab(v, "fe1op", "wno")
    waveno = fq.freq / 2.99792458e10
    if waveno < 21000.0:
        return torch.zeros_like(hkt)
    out = torch.zeros_like(hkt)
    for i in range(48):
        xs = 0.0
        if wno[i] < waveno:
            xs = 3e-18 / (1.0 + ipow_const((float(wno[i]) + 3000.0 - waveno) / float(wno[i]) / r4(".1"), 4))
        out = out + xs * (float(G[i]) * libm.exp(-float(E[i]) * 2.99792458e10 * hkt))
    return out


def _molop(s, fq, n, v, unit, cross_name, part_name, nshift, nmax, ln10, chop):
    T = _g(s, "temp.t", n)
    cross, part = _eq(v, unit, cross_name), _tab(v, unit, part_name)
    waveno = fq.freq / 2.99792458e10
    evolt = waveno / r4("8065.479")
    if chop:
        N = int(evolt * 10.0)
        en = float(F32(N) * F32(r4(".1")))
        if N < 20 or N >= nmax:
            return torch.zeros_like(T)
    else:
        N = int(evolt * 10.0 - 20.0)
        en = float(F32(N) * F32(r4(".1")) + F32(2.0))
        if N <= 0 or N >= nmax:
            return torch.zeros_like(T)
    crt = torch.from_numpy(np.array([cross[it, N - 1] + (cross[it, N] - cross[it, N - 1]) * (evolt - en) / r4(".1")
                                     for it in range(15)]))
    pt = torch.from_numpy(part)
    it = torch.clamp(torch.trunc((T - 1000.0) / 200.0 + 1.0).to(torch.int64), min=1, max=len(part) - 1)
    tn = (it.to(torch.float32) * 200.0 + 800.0).double()
    prt = pt[it - 1] + (pt[it] - pt[it - 1]) * (T - tn) / 200.0
    it2 = torch.clamp(torch.trunc((T - 2000.0) / 500.0 + 1.0).to(torch.int64), min=1, max=14)
    tn2 = (it2.to(torch.float32) * 500.0 + 1500.0).double()
    val = libm.exp((crt[it2 - 1] + (crt[it2] - crt[it2 - 1]) * (T - tn2) / 500.0) * ln10) * prt
    return torch.where(T >= 9000.0, torch.zeros_like(T), val)


def chop(s, fq, n, v="atlas9"):
    """FUNCTION CHOP(J): CH photodissociation (Kurucz, van Dishoeck & Tarafdar)."""
    return _molop(s, fq, n, v, "chop", "crossch", "partch", 0, 105, r4("2.30258509299405E0"), True)


def ohop(s, fq, n, v="atlas9"):
    """FUNCTION OHOP(J): OH photodissociation."""
    return _molop(s, fq, n, v, "ohop", "crossoh", "partoh", 20, 130, 2.30258509299405, False)


def _xnh2(s, n):
    """(XNFPH(J,1)*2.*BHYD(J,1))**2*EXP(4.477/TKEV-4.6628D1+(1.8031D-3+...)*T-1.5*TLOG) (H2COLLOP/H2RAOP)."""
    T, tkev, tlog = _g(s, "temp.t", n), _g(s, "temp.tkev", n), _g(s, "temp.tlog", n)
    b = s["ions.xnfph"][:n, 0] * 2.0 * s["depart.bhyd"][:n, 0]
    return ipow_const(b, 2) * libm.exp(rdiv(r4("4.477"), tkev) - 4.6628e1 + (1.8031e-3 + (-5.0239e-7 + (
        8.1424e-11 - 5.0501e-15 * T) * T) * T) * T - 1.5 * tlog)


def h2collop(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE H2COLLOP(AH2COLL): H2-H2 and H2-He collision-induced absorption (Borysow)."""
    T, rho = _g(s, "temp.t", n), _g(s, "state.rho", n)
    xnh2 = _xnh2(s, n)
    waveno = fq.freq / r4("2.99792458E10")
    if waveno > 20000.0:
        return torch.zeros_like(T)
    h2he, h2h2 = _eq(v, "h2collop", "h2he"), _eq(v, "h2collop", "h2h2")
    nu = min(80, int(waveno / 250.0) + 1)
    delnu = (waveno - float(F32(250.0) * F32(nu - 1))) / 250.0
    h2h2nu = torch.from_numpy(np.array([h2h2[it, nu - 1] * (1.0 - delnu) + h2h2[it, nu] * delnu for it in range(7)]))
    h2henu = torch.from_numpy(np.array([h2he[it, nu - 1] * (1.0 - delnu) + h2he[it, nu] * delnu for it in range(7)]))
    it = torch.clamp(torch.trunc(T / 1000.0).to(torch.int64), 1, 6)
    delt = (T - (it.to(torch.float32) * 1000.0).double()) / 1000.0
    delt = torch.clamp(torch.clamp(delt, max=1.0), min=0.0)
    xh2h2 = h2h2nu[it - 1] * (1.0 - delt) + h2h2nu[it] * delt
    xh2he = h2henu[it - 1] * (1.0 - delt) + h2henu[it] * delt
    return (libm.pow(10.0, xh2he) * s["ions.xnfhe"][:n, 0] + libm.pow(10.0, xh2h2) * xnh2) * xnh2 / rho * fq.stim


def coolop(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE COOLOP: C I, Mg I, Al I, Si I, Fe I, CH, OH bound-free + H2 CIA."""
    if fq.freq > 3.28805e15:
        return {}
    ah2coll = h2collop(s, fq, n, v)
    asi1 = si1op(s, fq, n, v)
    xc, xmg = s["xnfp.xnfpc"][:n, 0], s["xnfp.xnfpmg"][:n, 0]
    xal, xfe = s["xnfp.xnfpal"][:n].reshape(n, -1)[:, 0], s["xnfp.xnfpfe"][:n].reshape(n, -1)[:, 0]
    a = (c1op(s, fq, n, v) * xc + mg1op(s, fq, n, v) * xmg + al1op(s, fq, n, v) * xal + fe1op(s, fq, n, v) * xfe
         + chop(s, fq, n, v) * s["xnfp.xnfpch"][:n] + ohop(s, fq, n, v) * s["xnfp.xnfpoh"][:n]) * fq.stim \
        / _g(s, "state.rho", n) + asi1 * s["xnfp.xnfpsi"][:n, 0] / _g(s, "state.rho", n) + ah2coll
    return {"acool": a, "_asi1": asi1}


# ------------------------------------------------------------------------------------------ metals (luke)

def n1op(s, fq, n, v="atlas9"):
    tkev = _g(s, "temp.tkev", n)
    c1130 = 6.0 * libm.exp(rdiv(-r4("3.575"), tkev))
    c1020 = 10.0 * libm.exp(rdiv(-r4("2.384"), tkev))
    f = fq.freq
    x853 = seaton(f, 3.517915e15, 1.142e-17, 2.0, 4.29) if f >= 3.517915e15 else 0.0
    x1020 = seaton(f, 2.941534e15, 4.41e-18, 1.5, 3.85) if f >= 2.941534e15 else 0.0
    x1130 = seaton(f, 2.653317e15, 4.2e-18, 1.5, 4.34) if f >= 2.653317e15 else 0.0
    return x853 * 4.0 + x1020 * c1020 + x1130 * c1130


def o1op(s, fq, n, v="atlas9"):
    f = fq.freq
    x911 = seaton(f, 3.28805e15, 2.94e-18, 1.0, 2.66) if f >= 3.28805e15 else 0.0
    return x911 * 9.0


def mg2op(s, fq, n, v="atlas9"):
    tkev = _g(s, "temp.tkev", n)
    c1169 = 6.0 * libm.exp(rdiv(-r4("4.43"), tkev))
    f = fq.freq
    x824 = seaton(f, 3.635492e15, 1.40e-19, 4.0, 6.7) if f >= 3.635492e15 else 0.0
    x1169 = r4("5.11E-19") * ipow_const(r4("2.564306E15") / f, 3) if f >= 2.564306e15 else 0.0
    return x824 * 2.0 + x1169 * c1169


def ca2op(s, fq, n, v="atlas9"):
    tkev = _g(s, "temp.tkev", n)
    c1218 = 10.0 * libm.exp(rdiv(-r4("1.697"), tkev))
    c1420 = 6.0 * libm.exp(rdiv(-r4("3.142"), tkev))
    f = fq.freq
    x1044 = 5.4e-20 * ipow_const(2.870454e15 / f, 3) if f >= r4("2.870454E15") else 0.0
    x1218 = 1.64e-17 * math.sqrt(2.460127e15 / f) if f >= r4("2.460127E15") else 0.0
    x1420 = seaton(f, 2.110779e15, 4.13e-18, 3.0, 0.69) if f >= r4("2.110779E15") else 0.0
    return x1044 * 2.0 + x1218 * c1218 + x1420 * c1420


def lukeop(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE LUKEOP: N I, O I, Mg II, Si II, Ca II bound-free."""
    a = (n1op(s, fq, n, v) * s["xnfp.xnfpn"][:n, 0] + o1op(s, fq, n, v) * s["xnfp.xnfpo"][:n, 0]
         + mg2op(s, fq, n, v) * s["xnfp.xnfpmg"][:n, 1] + si2op(s, fq, n, v) * s["xnfp.xnfpsi"][:n, 1]
         + ca2op(s, fq, n, v) * s["xnfp.xnfpca"][:n, 1]) * fq.stim / _g(s, "state.rho", n)
    return {"aluke": a}


# ------------------------------------------------------------------------------------------------ hot

_XNFP_CANON = [("xnfpc", 4), ("xnfpn", 5), ("xnfpo", 6), ("xnfpne", 6), ("xnfpal", 1), ("xnfpmg", 2),
               ("xnfpsi", 2), ("xnfpca", 2), ("xnfpfe", 1), ("xnfpch", 1), ("xnfpoh", 1)]


def _xnfp_flat(s, n):
    cols = []
    for name, nc in _XNFP_CANON:
        a = s["xnfp." + name][:n]
        cols.append(a.reshape(n, -1)[:, :nc])
    return torch.cat(cols, 1)                       # (n, 30): XNFP(J, 1..30) by storage association


def hotop(s, fq: Freq, n, v="atlas9"):
    """SUBROUTINE HOTOP: free-free of C,N,O,Ne,Mg,Si,S,Fe ions + 60 hot bound-free edges."""
    xnf = {k: s["xnf." + k][:n] for k in ("xnfc", "xnfn", "xnfo", "xnfne", "xnfmg", "xnfsi", "xnfs", "xnffe")}
    T, tkev, rho, xne = _g(s, "temp.t", n), _g(s, "temp.tkev", n), _g(s, "state.rho", n), _g(s, "state.xne", n)
    X = lambda k, i: xnf[k][:, i - 1]
    g = [None] + [coulff(s, fq, n, z, v) for z in range(1, 6)]
    if v == "kappa9":
        s1 = X("xnfc", 2) + X("xnfn", 2) + X("xnfo", 2) + X("xnfne", 2) + X("xnfmg", 2) + X("xnfsi", 2) + X("xnfs", 2) + X("xnffe", 2)
    else:
        s1 = X("xnfc", 2) + X("xnfn", 2) + X("xnfo", 2) + X("xnfne", 2) + X("xnfmg", 2) + X("xnffe", 2)
    free = g[1] * 1.0 * s1
    for z in (2, 3, 4):
        sz = X("xnfc", z + 1) + X("xnfn", z + 1) + X("xnfo", z + 1) + X("xnfne", z + 1) + X("xnfmg", z + 1) \
            + X("xnfsi", z + 1) + X("xnfs", z + 1) + X("xnffe", z + 1)
        free = free + g[z] * float(z * z) * sz
    s5 = X("xnfc", 6) + X("xnfn", 6) + X("xnfo", 6) + X("xnfne", 6) + X("xnfmg", 6) + X("xnfsi", 6) + X("xnfs", 6)
    free = free + g[5] * 25.0 * s5
    freq = fq.freq
    ahot = free * r4("3.6919E8") / ipow_const(freq, 3) * xne / libm.sqrt(T)
    for _ in range(12):                       # + AC2OP + ... + AO6OP (all zero)
        ahot = ahot + 0.0
    A = _eq(v, "hotop", "a")
    xnfp = _xnfp_flat(s, n)
    for i in range(60):
        L = 7 * i
        if freq < A[L]:
            continue
        r = A[L] / freq
        xs = A[L + 1] * (A[L + 2] + r - A[L + 2] * r) * math.sqrt(ipow_var(r, int(A[L + 3])))
        idd = int(A[L + 6])
        xx = xs * xnfp[:, idd - 1] * float(A[L + 4])
        ahot = torch.where(xx > ahot / 100.0, ahot + xx / libm.exp(rdiv(float(A[L + 5]), tkev)), ahot)
    return {"ahot": ahot * fq.stim / rho}


def elecop(s, fq, n, v="atlas9"):
    """SUBROUTINE ELECOP: Thomson scattering."""
    return {"sigel": 0.6653e-24 * _g(s, "state.xne", n) / _g(s, "state.rho", n)}


def h2raop(s, fq, n, v="atlas9"):
    """SUBROUTINE H2RAOP: Rayleigh scattering by H2 (Dalgarno & Williams)."""
    xnh2 = _xnh2(s, n) / _g(s, "state.rho", n)
    wave = 2.997925e18 / min(fq.freq, 2.922e15)
    ww = ipow_const(wave, 2)
    sig = (8.14e-13 + 1.28e-6 / ww + r4("1.61") / (ww * ww)) / (ww * ww)
    return {"sigh2": sig * xnh2}


# ------------------------------------------------------------------------------------------------ KAPP

OPS = ("ahyd ah2p ahmin sigh ahe1 ahe2 ahemin sighe acool aluke ahot sigel sigh2 ahline alines siglin axline "
       "sigxl axcont sigx shyd shmin shline sxline sxcont").split()

IFOP_ROUTINES = {1: hop, 2: h2plop, 3: hminop, 4: hrayop, 5: he1op, 6: he2op, 7: hemiop, 8: heraop, 9: coolop,
                 10: lukeop, 11: hotop, 12: elecop, 13: h2raop}


def kapp(s, fq: Freq | None = None, variant: str = "atlas9", n_step: int = 0) -> dict:
    """SUBROUTINE KAPP(N,NSTEPS,STEPWT) for N<=1 (continuum): zero /OPS/ (25 arrays, J<=NRHOX), call the
    IFOP-enabled continuum sources, combine into /OPTOT/. Line sources (IFOP 14-18) and XCONOP/XSOP (19, 20)
    are not part of this library: they must be off (BasicATLAS: IFOP = 1x13, 0, 1, 0x5 with N=0)."""
    n = s.iscalar("rhox.nrhox")
    fq = Freq.from_state(s, n) if fq is None else fq
    ifop = [int(x) for x in s["ifop.ifop"].reshape(-1)]
    for k in (14, 15, 16, 17, 18):
        if ifop[k - 1] == 1 and n_step > 0:
            raise NotImplementedError(f"IFOP({k}) line opacity with N>0 is not part of the physics library")
    for k in (19, 20):
        if ifop[k - 1] == 1:
            raise NotImplementedError(f"IFOP({k}) (XCONOP/XSOP) is not implemented")
    def arr(key):
        return s[key].clone() if key in s else torch.zeros(max(99, n), dtype=torch.float64, device=s["temp.t"].device)
    ops = {}
    for name in OPS:
        a = arr("ops." + name)
        a[:n] = 0.0
        ops[name] = a
    upd = {}
    for k in range(1, 14):
        if ifop[k - 1] != 1:
            continue
        r = IFOP_ROUTINES[k](s, fq, n, variant)
        for name, val in r.items():
            if name == "_asi1":
                a = arr("bsi.asi1")
                a[:n] = val
                upd["bsi.asi1"] = a
                continue
            ops[name][:n] = val
    o = {k: ops[k][:n] for k in OPS}
    bnu = fq.bnu
    a = o["ah2p"] + o["ahe1"] + o["ahe2"] + o["ahemin"] + o["acool"] + o["aluke"] + o["ahot"]
    acont = a + o["ahyd"] + o["ahmin"] + o["axcont"]
    scont = torch.where(acont > 0, (a * bnu + o["ahyd"] * o["shyd"] + o["ahmin"] * o["shmin"]
                                    + o["axcont"] * o["sxcont"]) / acont, bnu)
    aline = o["ahline"] + o["alines"] + o["axline"]
    sline = torch.where(aline > 0, (o["ahline"] * o["shline"] + o["alines"] * bnu + o["axline"] * o["sxline"]) / aline, bnu)
    sigmac = o["sigh"] + o["sighe"] + o["sigel"] + o["sigh2"] + o["sigx"]
    sigmal = o["siglin"] + o["sigxl"]
    for name, val in (("acont", acont), ("scont", scont), ("aline", aline), ("sline", sline), ("sigmac", sigmac),
                      ("sigmal", sigmal)):
        t = arr("optot." + name)
        t[:n] = val
        upd["optot." + name] = t
    upd.update({"ops." + k: v for k, v in ops.items()})
    return upd
