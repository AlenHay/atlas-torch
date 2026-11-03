"""KAPP and the continuum opacity sources of xnfdf.for (N=1, IFOP(1..13)), vectorised over layers.

Only what PROGRAM XNFDF uses is computed: ACONT and SIGMAC (log10(ACONT+SIGMAC) is the tabulated
continuum). The source functions (SCONT, SHYD, ...) never reach xnfpdf.dat and are not computed.
In this variant KAPP sums AC1+AMG1+AAL1+ASI1+AFE1 directly and never ACOOL, so CHOP, OHOP and
H2COLLOP (called by COOLOP) have no effect on any output; their only side effect, WAVENO, is reset
by the main program before the next KAPP call. They are not ported.

Frequency-only quantities are Python floats (glibc libm; ifx's single-precision literals via F()),
per-layer quantities torch tensors of the batch shape S. `python_float / tensor` is avoided (torch
evaluates it as reciprocal-multiply); rdiv() is used instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from ...interp import map1
from .fdata import T as tab, equiv_assemble, hminop_ff, karzas
from .fmath import F, fsqrt, powi, rdiv

DT = torch.float64


# ---------------------------------------------------------------------------------------------------
# scalar helpers (frequency-only)
# ---------------------------------------------------------------------------------------------------
def _f32(x):
    return np.float32(x)


def xkarsas(freq: float, zeff2: float, n: int, l: int) -> float:
    """FUNCTION XKARSAS(FREQ,ZEFF2,N,L): Karzas & Latter hydrogenic cross-sections (REAL*4 body)."""
    frn, xn, xl, ek = karzas()
    frf = frn.reshape(-1, order="F")          # flat views for the Fortran's out-of-range I=30 access
    xnf = xn.reshape(-1, order="F")
    freqlg = np.float32(math.log10(freq / zeff2))
    if l >= n or n > 6:
        if n > 15:
            fn15 = np.zeros(29, np.float32)
            fn15[28] = np.float32(math.log10(F(109677.576) * 2.99792458e10 / (n * n)))
            if freqlg < fn15[28]:
                return 0.0
            i = 29
            for ii in range(2, 29):
                fn15[ii - 1] = np.float32(math.log10(float((ek[ii - 1] + np.float32(1.) / np.float32(n * n))
                                                           * np.float32(109677.576)) * 2.99792458e10))
                if freqlg > fn15[ii - 1]:
                    i = ii
                    break
            x = (freqlg - fn15[i - 1]) / (fn15[i - 2] - fn15[i - 1]) * (xn[i - 2, 14] - xn[i - 1, 14]) + xn[i - 1, 14]
            return math.exp(float(x) * 2.30258509299405) / zeff2
        if freqlg < frn[28, n - 1]:
            return 0.0
        i = next((ii for ii in range(2, 30) if freqlg > frn[ii - 1, n - 1]), 30)
        o = (n - 1) * 29
        x = (freqlg - frf[o + i - 1]) / (frf[o + i - 2] - frf[o + i - 1]) * (xnf[o + i - 2] - xnf[o + i - 1]) \
            + xnf[o + i - 1]
        return math.exp(float(x) * 2.30258509299405) / zeff2
    if freqlg < frn[28, n - 1]:
        return 0.0
    i = next((ii for ii in range(2, 30) if freqlg > frn[ii - 1, n - 1]), 30)
    x = (freqlg - frn[i - 1, n - 1]) / (frn[i - 2, n - 1] - frn[i - 1, n - 1]) * \
        (xl[i - 2, l, n - 1] - xl[i - 1, l, n - 1]) + xl[i - 1, l, n - 1]
    return math.exp(float(x) * 2.30258509299405) / zeff2


def seaton(freq0, xsect, power, a, freq):
    return xsect * (a + (1. - a) * (freq0 / freq)) * math.sqrt(powi(freq0 / freq, int(2. * power + F(.01))))


@lru_cache(maxsize=None)
def _t(unit, var):
    return tab(unit, var)


@lru_cache(maxsize=None)
def _hotop_a():
    return equiv_assemble("hotop", "a", (420,), np.float64)


@lru_cache(maxsize=None)
def _hminop_tables():
    wavek = _t("hminop", "wavek")
    thetaff = _t("hminop", "thetaff")
    ff = hminop_ff()
    wfflog = np.array([math.log(91.134 / w) for w in wavek])
    fflog = np.empty((22, 11))
    for iw in range(22):
        for it in range(11):
            fflog[iw, it] = math.log(ff[it, iw] / thetaff[it] * 5040. * 1.380658e-16)
    return wfflog, fflog, thetaff


def linter_scalar(xold, yold, xnew):
    nold = len(xold)
    iold = 2
    while not (xnew < xold[iold - 1]) and iold != nold:
        iold += 1
    return yold[iold - 2] + (yold[iold - 1] - yold[iold - 2]) / (xold[iold - 1] - xold[iold - 2]) * (xnew - xold[iold - 2])


def linter_t(xold: np.ndarray, yold: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
    """LINTER for one table and many XNEW (each XNEW scanned from IOLD=2, NNEW=1 calls)."""
    xo = torch.as_tensor(xold, dtype=DT)
    nold = len(xold)
    cnt = torch.searchsorted(xo, xnew.contiguous(), right=True)    # number of XOLD <= XNEW
    iold = (cnt + 1).clamp(2, nold)                                # 1-based
    i1 = iold - 2
    i2 = iold - 1
    yo = torch.as_tensor(yold, dtype=DT)
    return yo[i1] + (yo[i2] - yo[i1]) / (xo[i2] - xo[i1]) * (xnew - xo[i1])


def coulff(lay, freqlg: float, nz: int) -> torch.Tensor:
    a = torch.as_tensor(_t("coulff", "a"), dtype=DT)          # (11, 12)
    z4log = _t("coulff", "z4log")
    gamlog = F(10.39638) - lay.tlog / F(1.15129) + float(z4log[nz - 1])
    igam = torch.trunc(gamlog + 7.).to(torch.int64).clamp(max=10).clamp(min=1)
    hvktlg = (freqlg - lay.tlog) / F(1.15129) - F(20.63764)
    ihvkt = torch.trunc(hvktlg + 9.).to(torch.int64).clamp(max=11).clamp(min=1)
    p = gamlog - (igam - 7).to(DT)
    q = hvktlg - (ihvkt - 9).to(DT)
    g = igam - 1
    h = ihvkt - 1
    return (1. - p) * ((1. - q) * a[g, h] + q * a[g, h + 1]) + p * ((1. - q) * a[g + 1, h] + q * a[g + 1, h + 1])


# ---------------------------------------------------------------------------------------------------
# per-model state ("ITEMP" caches of HMINOP/LUKEOP/N1OP/MG2OP/SI2OP/CA2OP/HOTOP/H2RAOP)
# ---------------------------------------------------------------------------------------------------
@dataclass
class KappInput:
    t: torch.Tensor
    tkev: torch.Tensor
    hkt: torch.Tensor
    hckt: torch.Tensor
    tlog: torch.Tensor
    xne: torch.Tensor
    rho: torch.Tensor
    b: dict                 # departure coefficients (S+(nlev,)), bmin (S)
    pops: dict              # "xnfph" S+(2,), "xnfhe" S+(2,), "xnfphe" S+(3,), "xnfpc","xnfpmg","xnfpal",
                            # "xnfpsi","xnfpfe" S+(k,), luke: "lk_n","lk_o","lk_mg","lk_si","lk_ca", "hot" S+(21,)
    ifop: tuple


def prepare(ki: KappInput) -> dict:
    """Frequency-independent quantities (computed once per model by the Fortran's ITEMP caches)."""
    c = {}
    t, tkev = ki.t, ki.tkev
    xnfph = ki.pops["xnfph"]
    bh1 = ki.b["bhyd"][..., 0]
    # HMINOP
    c["theta"] = rdiv(5040., t)
    c["xhmin"] = torch.exp(rdiv(F(.754209), tkev)) / (2. * 2.4148e15 * t * fsqrt(t)) * ki.b["bmin"] * bh1 * \
        xnfph[..., 0] * ki.xne
    # N1OP / MG2OP / CA2OP
    c["c1130"] = 6. * torch.exp(rdiv(-F(3.575), tkev))
    c["c1020"] = 10. * torch.exp(rdiv(-F(2.384), tkev))
    c["c1169"] = 6. * torch.exp(rdiv(-F(4.43), tkev))
    c["c1218"] = 10. * torch.exp(rdiv(-F(1.697), tkev))
    c["c1420"] = 6. * torch.exp(rdiv(-F(3.142), tkev))
    # SI2OP
    tlg = torch.as_tensor(_t("si2op", "tlg"), dtype=DT)
    n = (torch.trunc(t / 2000.).to(torch.int64) - 4).clamp(max=5).clamp(min=1)
    c["si2_nt"] = n
    c["si2_dt"] = (ki.tlog - tlg[n - 1]) / (tlg[n] - tlg[n - 1])
    # H2RAOP
    c["xnh2"] = powi(xnfph[..., 0] * 2. * bh1, 2) * torch.exp(
        rdiv(F(4.478), tkev) - 4.64584e1 + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (
            1.06206e-18 - 3.08720e-23 * t) * t) * t) * t) * t) * t - 1.5 * ki.tlog) / ki.rho
    return c


# ---------------------------------------------------------------------------------------------------
# opacity routines (one frequency)
# ---------------------------------------------------------------------------------------------------
class Freq:
    def __init__(self, freq: float, ki: KappInput):
        self.freq = freq
        self.freqlg = math.log(freq)
        wave = 2.99792458e17 / freq
        self.waveno = 1e7 / wave
        self.ehvkt = torch.exp(-freq * ki.hkt)
        self.stim = 1. - self.ehvkt


def hop(f: Freq, ki: KappInput) -> torch.Tensor:
    freq, wn, hckt, stim, ehvkt = f.freq, f.waveno, ki.hckt, f.stim, f.ehvkt
    bh = ki.b["bhyd"]
    freq3 = 2.815e29 / freq / freq / freq
    h = rdiv(freq3 * 2. / 2., F(109677.576) * hckt) * (
        torch.exp(-max(109250.336, 109678.764 - wn) * hckt) - torch.exp(-F(109678.764) * hckt)) * stim
    hi = [(15, 487.456, 450., 109191.313), (14, 559.579, 392., 109119.188), (13, 648.980, 338., 109029.789),
          (12, 761.649, 288., 108917.117), (11, 906.426, 242., 108772.336), (10, 1096.776, 200., 108581.992),
          (9, 1354.044, 162., 108324.719), (8, 1713.713, 128., 107965.051), (7, 2238.320, 98., 107440.444)]
    lo = [(6, 3046.604, 72., 106632.160), (5, 4387.113, 50., 105291.651), (4, 6854.871, 32., 102823.893),
          (3, 12186.462, 18., 97492.302), (2, 27419.659, 8., 82259.105), (1, 109678.764, 2., None)]
    done = False
    for n, thr, g, e in hi:
        if wn < F(thr):
            done = True
            break
        x = xkarsas(freq, 1.0, n, n)
        h = h + x * g * torch.exp(-F(e) * hckt) * stim
    if not done:
        for n, thr, g, e in lo:
            if wn < F(thr):
                break
            x = xkarsas(freq, 1.0, n, n)
            if e is None:
                a = x * g * 1. * (bh[..., 0] - ehvkt)
            else:
                a = x * g * torch.exp(-F(e) * hckt) * (bh[..., n - 1] - ehvkt)
            h = h + a
    xnfph = ki.pops["xnfph"]
    h = h * xnfph[..., 0] / ki.rho
    a = rdiv(3.6919e8, fsqrt(ki.t)) * coulff(ki, f.freqlg, 1) / freq * ki.xne / freq * xnfph[..., 1] / freq * stim / ki.rho
    return h + a


def h2plop(f: Freq, ki: KappInput):
    freq, fl = f.freq, f.freqlg
    if freq > 3.28805e15:
        return None
    fr = -3.0233e3 + (3.7797e2 + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * fl) * fl) * fl) * fl
    f15 = freq / 1e15
    es = -7.342e-3 + (-2.409e-00 + (1.028e-00 + (-4.230e-01 + (1.224e-01 - 1.351e-02 * f15) * f15) * f15) * f15) * f15
    xnfph = ki.pops["xnfph"]
    return torch.exp(rdiv(-es, ki.tkev) + fr + torch.log(xnfph[..., 0])) * 2. * ki.b["bhyd"][..., 0] * \
        xnfph[..., 1] / ki.rho * f.stim


def hminop(f: Freq, ki: KappInput, c: dict) -> torch.Tensor:
    wfflog, fflog, thetaff = _hminop_tables()
    freq = f.freq
    wave = 2.99792458e17 / freq
    wavelog = math.log(wave)
    fftt = np.array([math.exp(linter_scalar(wfflog, fflog[:, it], wavelog)) for it in range(11)])
    hminbf = 0.
    if freq > 1.82365e14:
        wbf = torch.as_tensor(_t("hminop", "wbf"), dtype=DT)
        bf = torch.as_tensor(_t("hminop", "bf"), dtype=DT)
        hminbf = float(map1(wbf, bf, torch.tensor([wave], dtype=DT))[0][0])
    fftheta = linter_t(thetaff, fftt, c["theta"])
    xnfph = ki.pops["xnfph"]
    bh1 = ki.b["bhyd"][..., 0]
    hminff = fftheta * xnfph[..., 0] * 2. * bh1 * ki.xne / ki.rho * 1e-26
    h = hminbf * 1e-18 * (1. - f.ehvkt / ki.b["bmin"]) * c["xhmin"] / ki.rho
    return h + hminff


@lru_cache(maxsize=None)
def _gavrila():
    return {k: _t("hrayop", k).astype(np.float64) for k in (
        "gavrilam", "gavrilamab", "gavrilambc", "gavrilamcd", "gavrilalymancont", "fgavrilalymancont")}


def hrayop_xsect(freq: float) -> float:
    g_ = _gavrila()
    nu = 3.288051e15
    if freq < 3.288051e13:
        return 6.65e-25 * powi(float(g_["gavrilam"][0]), 2) * powi(freq / 3.288051e13, 4)
    if freq <= 2.4190611e15:
        i = min(int(freq / 3.288051e13) + 1, 74)
        gm = g_["gavrilam"]
        g = gm[i - 2] + (gm[i - 1] - gm[i - 2]) / 3.288051e13 * (freq - (i - 1) * 3.288051e13)
        return 6.65e-25 * powi(float(g), 2)
    if freq < F(.77) * nu:
        return 0.
    if freq <= F(.885) * nu:
        i = int((freq - F(.755) * nu) / 1.644026e13) + 1
        i = min(i + 1, 27)
        t = g_["gavrilamab"]
        g = t[i - 2] + (t[i - 1] - t[i - 2]) / 1.644026e13 * (freq - (F(.755) * nu + ((i - 1) - 1) * 1.664026e13))
        return 6.65e-25 * powi(float(g), 2)
    if freq < F(.890) * nu:
        return 0.
    if freq <= F(.936) * nu:
        i = int((freq - F(.890) * 3.28851e15) / 0.657610e13) + 1
        i = min(i + 1, 24)
        t = g_["gavrilambc"]
        g = t[i - 2] + (t[i - 1] - t[i - 2]) / 0.657610e13 * (freq - (F(.890) * nu + ((i - 1) - 1) * 0.657610e13))
        return 6.65e-25 * powi(float(g), 2)
    if freq < F(.938) * nu:
        return 0.
    if freq <= F(.959) * nu:
        i = int((freq - F(.938) * nu) / 0.3288051e13) + 1
        i = min(i + 1, 22)
        t = g_["gavrilamcd"]
        g = t[i - 2] + (t[i - 1] - t[i - 2]) / 0.3288051e13 * (freq - (F(.938) * nu + ((i - 1) - 1) * 0.3288051e13))
        return 6.65e-25 * powi(float(g), 2)
    if freq < F(.961) * nu:
        return 0.
    xs = 6.65e-25 * float(g_["gavrilalymancont"][0])
    if freq <= F(1.000) * nu:
        return xs
    g, _ = map1(torch.as_tensor(g_["fgavrilalymancont"]), torch.as_tensor(g_["gavrilalymancont"]),
                torch.tensor([freq / nu], dtype=DT))
    return 6.65e-25 * float(g[0])


def hrayop(f: Freq, ki: KappInput) -> torch.Tensor:
    return hrayop_xsect(f.freq) * ki.pops["xnfph"][..., 0] * 2. * ki.b["bhyd"][..., 0] / ki.rho


_HE1_HI = [  # (threshold, divisor of FREQ3, g, E, level of BHE1)
    (4368.190, 3125., 3., 193942.57, 29), (4388.260, 3125., 9., 193922.5, 28), (4388.260, 3125., 27., 193922.5, 27),
    (4389.390, 3125., 7., 193921.37, 26), (4389.450, 3125., 15., 193921.31, 25), (4392.369, 3125., 5., 193918.391, 24),
    (4393.515, 3125., 15., 193917.245, 23), (4509.980, 3125., 9., 193800.78, 22), (4647.133, 3125., 1., 193663.627, 21),
    (4963.671, 3125., 3., 193347.089, 20), (6817.943, 1024., 3., 191492.817, 19), (6858.680, 1024., 7., 191452.08, 18),
    (6858.960, 1024., 21., 191451.80, 17), (6864.201, 1024., 5., 191446.559, 16), (6866.172, 1024., 15., 191444.588, 15),
    (7093.620, 1024., 9., 191217.14, 14), (7370.429, 1024., 1., 190940.331, 13), (8012.550, 1024., 3., 190298.210, 12)]
_HE1_LO = [  # (threshold, X=EXP(a-b*FREQLG) coefficients, g, E, level)
    (12101.289, (58.81, 2.89), 3., 186209.471, 11), (12205.695, (85.20, 3.69), 5., 186105.065, 10),
    (12209.106, (85.20, 3.69), 15., 186101.654, 9), (12746.066, (49.30, 2.60), 9., 185564.694, 8),
    (13445.824, (23.85, 1.86), 1., 184864.936, 7), (15073.868, (12.69, 1.54), 3., 183236.892, 6),
    (27175.760, (81.35, 3.5), 3., 171135.000, 5), (29223.753, (61.21, 2.9), 9., 169087.007, 4),
    (32033.214, (26.83, 1.91), 1., 166277.546, 3)]


def he1op(f: Freq, ki: KappInput) -> torch.Tensor:
    freq, wn, hckt, stim, ehvkt, fl = f.freq, f.waveno, ki.hckt, f.stim, f.ehvkt, f.freqlg
    bhe1, bhe2 = ki.b["bhe1"], ki.b["bhe2"]
    b21 = bhe2[..., 0]
    freq3 = 2.815e29 / freq / freq / freq
    h = rdiv(freq3 * 4. / 2., F(109722.267) * hckt) * (
        torch.exp(-max(195262.919, 198310.76 - wn) * hckt) - torch.exp(-F(198310.76) * hckt)) * stim * b21

    def term(x, g, e, lev):
        return x * g * torch.exp(-F(e) * hckt) * (bhe1[..., lev - 1] - b21 * ehvkt)
    stop = False
    for thr, div, g, e, lev in _HE1_HI:
        if wn < F(thr):
            stop = True
            break
        h = h + term(freq3 / div, g, e, lev)
    if not stop:
        for thr, (ca, cb), g, e, lev in _HE1_LO:
            if wn < F(thr):
                stop = True
                break
            h = h + term(math.exp(F(ca) - F(cb) * fl), g, e, lev)
    if not stop and wn >= F(38454.691):
        x = math.exp(-F(390.026) + (F(21.035) - F(.318) * fl) * fl)
        h = h + term(x, 3., 159856.069, 2)
        if wn >= F(198310.760):
            x = math.exp(F(33.32) - 2. * fl)
            h = h + x * 1. * 1. * (bhe1[..., 0] - b21 * ehvkt)
    h = h * ki.pops["xnfphe"][..., 0] / ki.rho
    a = rdiv(3.619e8, fsqrt(ki.t)) * coulff(ki, fl, 1) / freq * ki.xne / freq * ki.pops["xnfhe"][..., 1] / freq * \
        stim / ki.rho
    return h + a


def he2op(f: Freq, ki: KappInput) -> torch.Tensor:
    freq, wn, hckt, stim, ehvkt = f.freq, f.waveno, ki.hckt, f.stim, f.ehvkt
    bhe2 = ki.b["bhe2"]
    xnfphe = ki.pops["xnfphe"]
    xr = xnfphe[..., 1] / ki.rho
    freq3 = 2.815e29 / freq / freq / freq
    h = rdiv(freq3 * 16. * 2. / 2., F(438889.068) * hckt) * (
        torch.exp(-max(434519.959, 438908.85 - wn) * hckt) - torch.exp(-F(438908.85) * hckt)) * stim * xr
    steps = [
        (5418.390, lambda: freq3 / 59049. * 16., 162., 433490.46, None),
        (6857.660, lambda: freq3 * 16. / 32768., 128., 432051.19, None),
        (8956.950, lambda: freq3 * 16. / 16807., 98., 429951.90, None),
        (12191.437, lambda: freq3 * 16. / 7776. * (F(1.0986) + (-2.704e13 + 1.229e27 / freq) / freq), 72., 426717.413, 6),
        (17555.715, lambda: freq3 * 16. / 3125. * (F(1.102) + (-3.909e13 + 2.371e27 / freq) / freq), 50., 421353.135, 5),
        (27430.925, lambda: freq3 * 16. / 1024. * (F(1.101) + (-5.765e13 + 4.593e27 / freq) / freq), 32., 411477.925, 4),
        (48766.491, lambda: freq3 * 16. / 243. * (F(1.101) + (-9.863e13 + 1.035e28 / freq) / freq), 18., 390142.359, 3),
        (109726.529, lambda: freq3 * 16. / 32. * (F(1.105) + (-2.375e14 + 4.077e28 / freq) / freq), 8., 329182.321, 2),
        (438908.850, lambda: freq3 * 16. / 1. * (F(.9916) + (2.719e13 - 2.268e30 / freq) / freq), 2., None, 1)]
    for thr, xf, g, e, lev in steps:
        if wn < F(thr):
            break
        x = xf()
        if lev is None:
            a = x * g * torch.exp(-F(e) * hckt) * stim * xr
        elif e is None:
            a = x * g * 1. * (bhe2[..., 0] - ehvkt) * xr
        else:
            a = x * g * torch.exp(-F(e) * hckt) * (bhe2[..., lev - 1] - ehvkt) * xr
        h = h + a
    a = rdiv(3.6919e8 * 4., fsqrt(ki.t)) * coulff(ki, f.freqlg, 2) / freq * ki.xne / freq * xnfphe[..., 2] / freq * \
        stim / ki.rho
    return h + a


def hemiop(f: Freq, ki: KappInput) -> torch.Tensor:
    freq = f.freq
    a = 3.397e-01 + (-5.216e+14 + 7.039e+30 / freq) / freq
    b = -4.116e+03 + (1.067e+19 + 8.135e+34 / freq) / freq
    c = 5.081e+08 + (-8.724e+22 - 5.659e+37 / freq) / freq
    return (a * ki.t + b + rdiv(c, ki.t)) / 1e15 * ki.xne / 1e15 * ki.pops["xnfphe"][..., 0] / 1e15 / ki.rho


def heraop(f: Freq, ki: KappInput) -> torch.Tensor:
    wave = 2.99792458e18 / min(f.freq, 5.15e15)
    ww = powi(wave, 2)
    sig = 5.484e-14 / ww / ww * powi(1. + (2.44e5 + 5.94e10 / (ww - 2.90e5)) / ww, 2)
    return sig * ki.pops["xnfphe"][..., 0] / ki.rho * ki.b["bhe1"][..., 0]


def _bterm(x, g, e, b, lev, b2, ehvkt, hckt):
    """X*g*EXP(-E*HCKT)*(B1(J,lev)-B2(J,1)*EHVKT(J)); e=None for the X*g*1.* ground-level form."""
    bb = b[..., lev - 1] - b2 * ehvkt
    if e is None:
        return x * g * 1. * bb
    return x * g * torch.exp(-F(e) * hckt) * bb


def c1op(f: Freq, ki: KappInput) -> torch.Tensor:
    wn, hckt, ehvkt = f.waveno, ki.hckt, f.ehvkt
    b1, b2 = ki.b["bc1"], ki.b["bc2"][..., 0]
    ryd = F(109732.298)
    h = torch.full_like(hckt, 1e-30)

    def T_(x, g, e, lev):
        return _bterm(x, g, e, b1, lev, b2, ehvkt, hckt)

    def go():
        nonlocal h
        # X=0 terms (WAVENO >= 16886.790 .. 21140.700, 58601.270) add exact zeros and are skipped
        if wn < F(22006.370):
            return
        h = h + T_(2.1e-18 * math.pow(F(22006.370) / wn, F(1.5)), 3., 68856.33, 8)
        if wn < F(28880.880):
            return
        h = h + T_(1.54e-18 * math.pow(F(28880.880) / wn, F(1.2)), 3., 61981.82, 6)
        if wn < F(30489.700):
            return
        h = h + T_(0.2e-18 * math.pow(F(30489.700) / wn, F(1.2)), 9., 60373.00, 5)
        if wn < F(58601.270) or wn < F(69172.400):
            return
        x = math.pow(10., -F(16.80) - (wn - F(69172.400)) / 3. / ryd)
        eps = (wn - 97700.) * 2. / 2743.
        x = x + (68.e-18 * eps + 118.e-18) / (powi(eps, 2) + 1.)
        x = x / 3.
        a = T_(x, 1., 21648.02, 3)
        h = h + a
        if wn < F(69235.820):
            return
        a = a * 2.
        h = h + a
        if wn < F(69767.350):
            return
        h = h + T_(16.e-18 * powi(F(69767.350) / wn, 3), 15., 64088.85, 7)
        if wn < F(80627.760):
            return
        x = math.pow(10., -F(16.80) - (wn - F(80627.760)) / 3. / ryd)
        eps = (wn - 93917.) * 2. / 9230.
        x = x + (22.e-18 * eps + 26.e-18) / (powi(eps, 2) + 1.)
        eps = (wn - 111130.) * 2. / 2743.
        x = x + (-10.5e-18 * eps + 46.e-18) / (powi(eps, 2) + 1.)
        x = x / 3.
        a = T_(x, 5., 10192.66, 2)
        h = h + a
        if wn < F(80691.180):
            return
        a = a * 2.
        h = h + a
        if wn < F(90777.000):
            return
        x = math.pow(10., -F(16.80) - (wn - F(90777.000)) / 3. / ryd)
        x = x / 3.
        h = h + T_(x, 5., 43.42, 1)
        if wn < F(90804.000):
            return
        h = h + T_(x, 3., 16.42, 1)
        if wn < F(90820.420):
            return
        h = h + T_(x, 1., None, 1)
        if wn < F(90840.420):
            return
        x = x * 2.
        h = h + T_(x, 5., 43.42, 1)
        if wn < F(90867.420):
            return
        h = h + T_(x, 3., 16.42, 1)
        if wn < F(90883.840):
            return
        h = h + T_(x, 1., None, 1)
        if wn < F(100121.000):
            return
        h = h + T_(1.e-18 * powi(F(100121.000) / wn, 3), 5., 33735.20, 4)
    go()
    return h * ki.pops["xnfpc"][..., 0] / ki.rho


_MG1 = [  # (threshold, X(wn) , g, E, level)
    (13713.986, lambda w: 25.e-18 * math.pow(F(13713.986) / w, F(2.7)), 15., 47957.034, 8),
    (13823.223, lambda w: 33.8e-18 * math.pow(F(13823.223) / w, F(2.8)), 9., 47847.797, 7),
    (15267.955, lambda w: 45.e-18 * math.pow(F(15267.955) / w, F(2.7)), 5., 46403.065, 6),
    (18167.687, lambda w: .43e-18 * math.pow(F(18167.687) / w, F(2.6)), 1., 43503.333, 5),
    (20473.617, lambda w: 2.1e-18 * math.pow(F(20473.617) / w, F(2.6)), 3., 41197.043, 4),
    (26619.756, lambda w: 16.e-18 * math.pow(F(26619.756) / w, F(2.1)) - 7.8e-18 * math.pow(F(26619.756) / w, F(9.5)),
     3., 35051.264, 3)]


def mg1op(f: Freq, ki: KappInput) -> torch.Tensor:
    wn, hckt, ehvkt = f.waveno, ki.hckt, f.ehvkt
    b1, b2 = ki.b["bmg1"], ki.b["bmg2"][..., 0]
    h = torch.full_like(hckt, 1e-30)

    def go():
        nonlocal h
        for thr, xf, g, e, lev in _MG1:
            if wn < F(thr):
                return
            h = h + _bterm(xf(wn), g, e, b1, lev, b2, ehvkt, hckt)
        if wn < F(39759.842):
            return
        x = 20.e-18 * math.pow(F(39759.842) / wn, F(2.7))
        x = max(x, 40.e-18 * powi(F(39759.842) / wn, 14))
        h = h + _bterm(x, 5., 21911.178, b1, 2, b2, ehvkt, hckt)
        if wn < F(39800.556):
            return
        h = h + _bterm(x, 3., 21870.464, b1, 2, b2, ehvkt, hckt)
        if wn < F(39820.615):
            return
        h = h + _bterm(x, 1., 21850.405, b1, 2, b2, ehvkt, hckt)
        if wn < F(61671.020):
            return
        h = h + _bterm(1.1e-18 * powi(F(61671.020) / wn, 10), 1., None, b1, 1, b2, ehvkt, hckt)
    go()
    return h * ki.pops["xnfpmg"][..., 0] / ki.rho


_AL1 = [
    (8002.467, lambda w: 50.e-18 * powi(F(8002.467) / w, 3), 6., 40275.903, 8),
    (9346.231, lambda w: 50.e-18 * powi(F(9346.231) / w, 3), 10., 38932.139, 7),
    (10588.957, lambda w: 56.7e-18 * math.pow(F(10588.957) / w, F(1.9)), 2., 37689.413, 6),
    (15318.007, lambda w: 14.5e-18 * F(15318.007) / w, 6., 32960.363, 5),
    (15842.129, lambda w: 47.e-18 * math.pow(F(15842.129) / w, F(1.83)), 10., 32436.241, 4),
    (22930.614, lambda w: 10.e-18 * powi(F(22930.614) / w, 2), 2., 25347.756, 2)]


def al1op(f: Freq, ki: KappInput) -> torch.Tensor:
    wn, hckt, ehvkt = f.waveno, ki.hckt, f.ehvkt
    b1, b2 = ki.b["bal1"], ki.b["bal2"][..., 0]
    h = torch.full_like(hckt, 1e-30)

    def go():
        nonlocal h
        if wn < F(6958.993):          # (X=0 term above 6958.993)
            return
        for thr, xf, g, e, lev in _AL1:
            if wn < F(thr):
                return
            h = h + _bterm(xf(wn), g, e, b1, lev, b2, ehvkt, hckt)
        if wn < F(48166.309):
            return
        x = 65.e-18 * powi(F(48166.309) / wn, 5)
        h = h + _bterm(x, 4., 112.061, b1, 1, b2, ehvkt, hckt)
        if wn < F(48278.370):
            return
        h = h + _bterm(x, 2., None, b1, 1, b2, ehvkt, hckt)
        if wn < F(55903.260):
            return
        h = h + _bterm(10.e-18 * powi(F(55903.260) / wn, 2), 12., 29097.11, b1, 3, b2, ehvkt, hckt)
    go()
    return h * ki.pops["xnfpal"][..., 0] / ki.rho


def si1op(f: Freq, ki: KappInput) -> torch.Tensor:
    wn, hckt, ehvkt = f.waveno, ki.hckt, f.ehvkt
    b1, b2 = ki.b["bsi1"], ki.b["bsi2"][..., 0]
    h = torch.full_like(hckt, 1e-30)

    def B(x, g, e, lev):
        return _bterm(x, g, e, b1, lev, b2, ehvkt, hckt)

    def go():
        nonlocal h
        if wn < F(16810.969) or wn < F(17777.641):
            return
        h = h + B(F(18.e-18) * powi(F(17777.641) / wn, 3), 15., 48161.459, 10)
        if wn < F(18587.546) or wn < F(18655.039) or wn < F(24947.216):
            return
        h = h + B(4.09e-18 * powi(F(24947.216) / wn, 2), 3., 40991.884, 6)
        if wn < F(26079.180):
            return
        h = h + B(1.25e-18 * powi(F(26079.180) / wn, 2), 9., 39859.920, 5)
        if wn < F(50353.180):
            return
        eps = (wn - 70000.) * 2. / 6500.
        reson1 = (97.e-18 * eps + 94.e-18) / (powi(eps, 2) + 1.)
        x = 37.e-18 * math.pow(F(50353.180) / wn, F(2.40)) + reson1
        bolt = 1. * torch.exp(-F(15394.370) * hckt) * (b1[..., 2] - b2 * ehvkt)
        h = h + x * bolt / 3.
        if wn < F(50640.630):
            return
        x = 37.e-18 * math.pow(F(50640.630) / wn, F(2.40)) + reson1
        h = h + x * bolt * 2. / 3.
        if wn < F(59448.700):
            return
        eps = (wn - 78600.) * 2. / 13000.
        reson1 = (-10.e-18 * eps + 77.e-18) / (powi(eps, 2) + 1.)
        x = 24.5e-18 * math.pow(F(59448.700) / wn, F(1.85)) + reson1
        bolt = 5. * torch.exp(-F(6298.850) * hckt) * (b1[..., 1] - b2 * ehvkt)
        h = h + x * bolt / 3.
        if wn < F(59736.150):
            return
        x = 24.5e-18 * math.pow(F(59736.150) / wn, F(1.85)) + reson1
        h = h + x * bolt * 2. / 3.
        if wn < F(63446.510):
            return
        h = h + B(18.e-18 * powi(F(63446.510) / wn, 3), 15., 45303.310, 7)
        if wn < F(65524.393):
            return
        x = 72.e-18 * math.pow(F(65524.393) / wn, F(1.90))
        if wn > 74000.:
            x = 93.e-18 * math.pow(F(65524.393) / wn, 4.00)
        x = x / 3.
        h = h + B(x, 5., 223.157, 1)
        if wn < F(65670.435):
            return
        h = h + B(x, 3., 77.115, 1)
        if wn < F(65747.550):
            return
        h = h + B(x, 1., None, 1)
        if wn < F(65811.843):
            return
        x = x * 2.
        h = h + B(x, 5., 223.157, 1)
        if wn < F(65957.885):
            return
        h = h + B(x, 3., 77.115, 1)
        if wn < F(66035.000):
            return
        h = h + B(x, 1., None, 1)
        if wn < F(75423.767):
            return
        h = h + B(15.e-18 * powi(F(75423.767) / wn, 3), 5., 33326.053, 4)
    go()
    return h * ki.pops["xnfpsi"][..., 0] / ki.rho


def fe1op(f: Freq, ki: KappInput) -> torch.Tensor:
    wn, hckt = f.waveno, ki.hckt
    afe1 = torch.zeros_like(hckt)
    if wn < 21000.:
        return afe1
    g, e, wno = _t("fe1op", "g"), _t("fe1op", "e"), _t("fe1op", "wno")
    for i in range(48):
        if wno[i] > wn:
            continue
        xsect = 3.e-18 / (1. + powi((float(wno[i]) + 3000. - wn) / float(wno[i]) / F(.1), 4))
        afe1 = afe1 + xsect * float(g[i]) * torch.exp(-float(e[i]) * hckt)
    return afe1 * f.stim * ki.pops["xnfpfe"][..., 0] / ki.rho


def lukeop(f: Freq, ki: KappInput, c: dict) -> torch.Tensor:
    freq = f.freq
    # N1OP
    x853 = seaton(3.517915e15, 1.142e-17, 2.0, 4.29, freq) if freq >= 3.517915e15 else 0.
    x1020 = seaton(2.941534e15, 4.41e-18, 1.5, 3.85, freq) if freq >= 2.941534e15 else 0.
    x1130 = seaton(2.653317e15, 4.2e-18, 1.5, 4.34, freq) if freq >= 2.653317e15 else 0.
    n1 = x853 * 4. + x1020 * c["c1020"] + x1130 * c["c1130"]
    # O1OP
    x911 = seaton(3.28805e15, 2.94e-18, 1.0, 2.66, freq) if freq >= 3.28805e15 else 0.
    o1 = x911 * 9.
    # MG2OP
    x824 = seaton(3.635492e15, 1.40e-19, 4.0, 6.7, freq) if freq >= 3.635492e15 else 0.
    x1169 = F(5.11e-19) * powi(2.564306e15 / freq, 3) if freq >= 2.564306e15 else 0.
    mg2 = x824 * 2. + x1169 * c["c1169"]
    # SI2OP
    peach = _t("si2op", "peach")
    freqsi = _t("si2op", "freqsi")
    flog = _t("si2op", "flog")
    n = next((k for k in range(1, 8) if freq > freqsi[k - 1]), 8)
    d = (f.freqlg - flog[n - 1]) / (flog[n] - flog[n - 1])
    if n > 2:
        n = 2 * n - 2
    if n == 14:
        n = 13
    d1 = 1. - d
    xs = torch.as_tensor(np.array([peach[it, n] * d + peach[it, n - 1] * d1 for it in range(6)]), dtype=DT)
    nt = c["si2_nt"]
    dts = c["si2_dt"]
    si2 = torch.exp(xs[nt - 1] * (1. - dts) + xs[nt] * dts) * 6.
    # CA2OP
    x1044 = F(5.4e-20) * powi(2.870454e15 / freq, 3) if freq >= 2.870454e15 else 0.
    x1218 = F(1.64e-17) * math.sqrt(2.460127e15 / freq) if freq >= 2.460127e15 else 0.
    x1420 = seaton(2.110779e15, 4.13e-18, 3.0, .69, freq) if freq >= 2.110779e15 else 0.
    ca2 = x1044 * 2. + x1218 * c["c1218"] + x1420 * c["c1420"]
    p = ki.pops
    return (n1 * p["lk_n"] + o1 * p["lk_o"] + mg2 * p["lk_mg"] + si2 * p["lk_si"] + ca2 * p["lk_ca"]) * f.stim / ki.rho


def hotop(f: Freq, ki: KappInput) -> torch.Tensor:
    a = _hotop_a()
    num = int(_t("hotop", "num"))
    freq = f.freq
    xnfp = ki.pops["hot"]
    ahot = torch.zeros_like(ki.hckt)
    for i in range(num):
        l = 7 * i
        if freq < a[l]:
            continue
        r = a[l] / freq
        xsect = a[l + 1] * (a[l + 2] + r - a[l + 2] * r) * math.sqrt(powi(r, int(a[l + 3])))
        idx = int(a[l + 6])
        xx = xsect * xnfp[..., idx - 1] * a[l + 4]
        ahot = torch.where(xx > ahot / 100., ahot + xx * torch.exp(rdiv(-float(a[l + 5]), ki.tkev)), ahot)
    return ahot * f.stim / ki.rho


def elecop(ki: KappInput) -> torch.Tensor:
    return .6653e-24 * ki.xne / ki.rho


def h2raop(f: Freq, ki: KappInput, c: dict) -> torch.Tensor:
    wave = 2.99792458e18 / min(f.freq, 2.922e15)
    ww = powi(wave, 2)
    sig = (8.14e-13 + 1.28e-6 / ww + F(1.61) / (ww * ww)) / (ww * ww)
    return sig * c["xnh2"]


def kapp(freq: float, ki: KappInput, c: dict, parts: bool = False):
    """KAPP(N=1): returns (ACONT, SIGMAC) at frequency FREQ for all layers (and, with parts=True, a dict
    of every opacity component under its COMMON name)."""
    ifop = ki.ifop
    for k in range(13, 20):
        if ifop[k] == 1 and k not in (13, 14, 15, 16):
            raise NotImplementedError(f"IFOP({k + 1}) opacity not ported")
    f = Freq(freq, ki)
    z = torch.zeros_like(ki.hckt)
    on = lambda k: ifop[k - 1] == 1
    ahyd = hop(f, ki) if on(1) else z
    ah2p = (h2plop(f, ki) if on(2) else None)
    ah2p = z if ah2p is None else ah2p
    ahmin = hminop(f, ki, c) if on(3) else z
    sigh = hrayop(f, ki) if on(4) else z
    ahe1 = he1op(f, ki) if on(5) else z
    ahe2 = he2op(f, ki) if on(6) else z
    ahemin = hemiop(f, ki) if on(7) else z
    sighe = heraop(f, ki) if on(8) else z
    if on(9):
        ac1, amg1, aal1, asi1, afe1 = c1op(f, ki), mg1op(f, ki), al1op(f, ki), si1op(f, ki), fe1op(f, ki)
    else:
        ac1 = amg1 = aal1 = asi1 = afe1 = z
    aluke = lukeop(f, ki, c) if on(10) else z
    ahot = hotop(f, ki) if on(11) else z
    sigel = elecop(ki) if on(12) else z
    sigh2 = h2raop(f, ki, c) if on(13) else z
    a = ah2p + ahemin + aluke + ahot
    acont = a + ahyd + ahmin + z + ahe1 + ahe2 + ac1 + amg1 + aal1 + asi1 + afe1
    sigmac = sigh + sighe + sigel + sigh2 + z
    if parts:
        return acont, sigmac, dict(ahyd=ahyd, ah2p=ah2p, ahmin=ahmin, sigh=sigh, ahe1=ahe1, ahe2=ahe2, ahemin=ahemin,
                                   sighe=sighe, ac1=ac1, amg1=amg1, aal1=aal1, asi1=asi1, afe1=afe1, aluke=aluke,
                                   ahot=ahot, sigel=sigel, sigh2=sigh2)
    return acont, sigmac
