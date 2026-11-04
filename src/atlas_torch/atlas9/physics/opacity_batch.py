"""KAPP's continuum sources for many frequencies at once: every /OPS/ array as [F, n].

The per-frequency routines of :mod:`.opacity` (the reference, tested against the Fortran) take the scalar FREQ of
the Fortran and branch on it in Python. Here the same arithmetic is arranged as

  * frequency-only factors (cross-sections, Gaunt factors, table interpolations in nu): the scalar helpers of
    :mod:`.opacity` (``hop_coef``, ``he1op_coef``, ... -- the very code the per-frequency routines run, memoised
    per frequency, so they are computed once per run for the fixed frequency set), gathered into [F, 1] columns;
  * temperature/population factors (Boltzmann factors, free-free terms ...): [n] rows, computed once per call
    (the Fortran's ITEMP caches);
  * their combination with the frequency-dependent /FREQ/ arrays (EHVKT, STIM, BNU [F, n]) by broadcasting,
    operation by operation in the order of the per-frequency routine (IEEE + - * / and the glibc exp/pow of
    :mod:`.libm` are elementwise, so each element is bit-identical to the per-frequency result);
  * Python branches on FREQ become torch.where on a per-frequency mask (early RETURNs give 0, as KAPP's zeroed
    /OPS/), loops whose range depends on FREQ become masked accumulations that skip (not add 0 to) the inactive
    frequencies.

Traps avoided: ``python_float / tensor`` (torch multiplies by the reciprocal) never occurs -- scalar numerators are
[F, 1] tensors or go through rdiv.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import libm
from . import opacity as O
from ._data import r4
from .libm import ipow_const, rdiv
from .opacity import F32, _eq, _g, _tab, _xnfp_flat, _xnh2, coulff, f4, linter


class FreqBatch:
    """/FREQ/ for F frequencies: python lists `freqs`, `freqlgs`; [F, 1] tensors `freq`, `freqlg` (used by
    COULFF); ehvkt, stim, bnu [F, n]."""

    def __init__(self, freqs, freqlgs, ehvkt, stim, bnu):
        self.freqs = [float(f) for f in freqs]
        self.freqlgs = [float(f) for f in freqlgs]
        self.F = len(self.freqs)
        self.ehvkt, self.stim, self.bnu = ehvkt, stim, bnu
        self.dev = ehvkt.device
        self.freq = self.col(self.freqs)
        self.freqlg = self.col(self.freqlgs)

    def col(self, vals) -> torch.Tensor:
        return torch.tensor([float(v) for v in vals], dtype=torch.float64, device=self.dev).reshape(-1, 1)

    def mask(self, flags) -> torch.Tensor:
        return torch.tensor([bool(f) for f in flags], dtype=torch.bool, device=self.dev).reshape(-1, 1)


def _zero_unless(mask, v):
    return torch.where(mask, v, torch.zeros_like(v))


# ------------------------------------------------------------------------------------------------- hydrogen

def hop(s, fb: FreqBatch, n, v="atlas9"):
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
    co = [O.hop_coef(f, v) for f in fb.freqs]
    cont = [fb.col([c[0][N] for c in co]) for N in range(8)]
    cfree, c = fb.col([x[1] for x in co]), fb.col([x[2] for x in co])
    ex = torch.where(fb.mask([f < 4.05933e13 for f in fb.freqs]), exlim / fb.ehvkt, boltex)
    h = (cont[6] * bolt[6] + cont[7] * bolt[7] + (ex - exlim) * c + coulff(s, fb, n, 1, v) * freet * cfree) * fb.stim
    sh = h * fb.bnu
    for N in range(1, 7):
        h = h + cont[N - 1] * bolt[N - 1] * (1.0 - fb.ehvkt / bhyd[:, N - 1])
        sh = sh + cont[N - 1] * bolt[N - 1] * fb.bnu * fb.stim / bhyd[:, N - 1]
    return {"ahyd": h, "shyd": sh / h}


def h2plop(s, fb: FreqBatch, n, v="atlas9"):
    fr, es = [], []
    for freq, fl in zip(fb.freqs, fb.freqlgs):
        fr.append(-3.0233e3 + (3.7797e2 + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * fl) * fl) * fl) * fl)
        es.append(-(-7.342e-3 + (-2.409e-15 + (1.028e-30 + (-4.230e-46 + (1.224e-61 - 1.351e-77 * freq) * freq)
                                                * freq) * freq) * freq))
    xnfph = s["ions.xnfph"][:n]
    a = libm.exp(rdiv(fb.col(es), _g(s, "temp.tkev", n)) + fb.col(fr)) * xnfph[:, 0] * 2.0 * s["depart.bhyd"][:n, 0] \
        * xnfph[:, 1] / _g(s, "state.rho", n) * fb.stim
    return {"ah2p": _zero_unless(fb.mask([f <= 3.28805e15 for f in fb.freqs]), a)}


def hminop(s, fb: FreqBatch, n, v="atlas9"):
    wfflog, fflog, thetaff, wbf, bf = O._hmin_tables(v)
    T, tkev, rho, xne = _g(s, "temp.t", n), _g(s, "temp.tkev", n), _g(s, "state.rho", n), _g(s, "state.xne", n)
    bmin, b1, xnfph1 = s["depart.bmin"][:n], s["depart.bhyd"][:n, 0], s["ions.xnfph"][:n, 0]
    theta = rdiv(5040.0, T)
    xhmin = libm.exp(rdiv(r4(".754209"), tkev)) / (2.0 * 2.4148e15 * T * libm.sqrt(T)) * bmin * b1 * xnfph1 * xne
    co = [O.hminop_coef(f, v) for f in fb.freqs]
    fftlog = linter(wfflog, fflog.T, torch.tensor([c[0] for c in co], dtype=torch.float64)).T     # (F, 11)
    fftt = libm.exp(fftlog) / thetaff * 5040.0 * 1.380658e-16
    fftheta = linter(thetaff, fftt, theta)                                                         # (F, n)
    hminff = fftheta * xnfph1 * 2.0 * b1 * xne / rho
    h = fb.col([c[1] * 1e-18 for c in co]) * (1.0 - fb.ehvkt / bmin) * xhmin / rho
    ahmin = h + hminff
    shmin = (h * fb.bnu * fb.stim / (bmin - fb.ehvkt) + hminff * fb.bnu) / ahmin
    return {"ahmin": ahmin, "shmin": shmin}


def hrayop(s, fb: FreqBatch, n, v="atlas9"):
    x = fb.col([O.hrayop_xsect(f, v) for f in fb.freqs])
    return {"sigh": x * s["ions.xnfph"][:n, 0] * 2.0 * s["depart.bhyd"][:n, 0] / _g(s, "state.rho", n)}


# --------------------------------------------------------------------------------------------------- helium

def he1op(s, fb: FreqBatch, n, v="atlas9"):
    tkev, rho, T = _g(s, "temp.tkev", n), _g(s, "state.rho", n), _g(s, "temp.t", n)
    xnfphe1 = s["ions.xnfphe"][:n, 0]
    G, chi = _tab(v, "he1op", "g"), _tab(v, "he1op", "chi")
    bolt = [libm.exp(rdiv(-float(chi[N]), tkev)) * float(G[N]) * xnfphe1 / rho for N in range(10)]
    boltn = {}
    for N in range(4, 28):
        num = float(F32(r4("24.587")) * (F32(1.0) - F32(1.0) / F32(N * N)))
        boltn[N] = libm.exp(rdiv(-num, tkev)) * 4.0 * float(N * N) * xnfphe1 / rho
    freet = _g(s, "state.xne", n) * s["ions.xnfhe"][:n, 1] / rho / libm.sqrt(T)
    xr = xnfphe1 * f4("4.", "/", "2.", "/", "13.595") * tkev / rho
    boltex = libm.exp(rdiv(-r4("23.730"), tkev)) * xr
    exlim = libm.exp(rdiv(-r4("24.587"), tkev)) * xr
    co = [O.he1op_coef(f, v) for f in fb.freqs]
    imin = [c[0] for c in co]
    cfree, c = fb.col([x[3] for x in co]), fb.col([x[4] for x in co])
    ex = torch.where(fb.mask([f < 2.055e14 for f in fb.freqs]), exlim / fb.ehvkt, boltex)
    he1 = (ex - exlim) * c
    for N in range(1, 11):
        on = [im != 0 and N >= im for im in imin]
        if any(on):
            he1 = torch.where(fb.mask(on), he1 + fb.col([x[1][N] for x in co]) * bolt[N - 1], he1)
    hi = [im != 0 and f >= 1.25408e16 for im, f in zip(imin, fb.freqs)]
    if any(hi):
        m = fb.mask(hi)
        for N in range(4, 28):
            he1 = torch.where(m, he1 + fb.col([x[2].get(N, 0.0) for x in co]) * boltn[N], he1)
    return {"ahe1": (he1 + coulff(s, fb, n, 1, v) * freet * cfree) * fb.stim}


def he2op(s, fb: FreqBatch, n, v="atlas9"):
    tkev, rho, T = _g(s, "temp.tkev", n), _g(s, "state.rho", n), _g(s, "temp.t", n)
    xnfphe2 = s["ions.xnfphe"][:n, 1]
    bolt = [libm.exp(rdiv(-(r4("54.403") - r4("54.403") / float(N * N)), tkev)) * 2.0 * float(N * N) * xnfphe2 / rho
            for N in range(1, 10)]
    freet = _g(s, "state.xne", n) * s["ions.xnfhe"][:n, 2] / rho / libm.sqrt(T)
    xr = xnfphe2 * f4("2.", "/", "2.", "/", "13.595") * tkev / rho
    boltex = libm.exp(rdiv(-r4("53.859"), tkev)) * xr
    exlim = libm.exp(rdiv(-r4("54.403"), tkev)) * xr
    co = [O.he2op_coef(f, v) for f in fb.freqs]
    cfree, c = fb.col([x[1] for x in co]), fb.col([x[2] for x in co])
    ex = torch.where(fb.mask([f < x[3] for f, x in zip(fb.freqs, co)]), exlim / fb.ehvkt, boltex)
    he2 = (ex - exlim) * c
    for N in range(9):
        he2 = he2 + fb.col([x[0][N] for x in co]) * bolt[N]
    return {"ahe2": (he2 + coulff(s, fb, n, 2, v) * cfree * freet) * fb.stim}


def hemiop(s, fb: FreqBatch, n, v="atlas9"):
    co = [O.hemiop_coef(f) for f in fb.freqs]
    a, b, c = (fb.col([x[i] for x in co]) for i in range(3))
    T = _g(s, "temp.t", n)
    return {"ahemin": (a * T + b + rdiv(c, T)) * _g(s, "state.xne", n) * s["ions.xnfphe"][:n, 0] / _g(s, "state.rho", n)}


def heraop(s, fb: FreqBatch, n, v="atlas9"):
    sig = fb.col([O.heraop_sig(f) for f in fb.freqs])
    return {"sighe": sig * s["ions.xnfphe"][:n, 0] / _g(s, "state.rho", n)}


# ------------------------------------------------------------------------------------------- metals (cool)

def c1op(s, fb, n, v="atlas9"):
    tkev = _g(s, "temp.tkev", n)
    c1240 = 5.0 * libm.exp(rdiv(-r4("1.264"), tkev))
    c1444 = libm.exp(rdiv(-r4("2.683"), tkev))
    co = [O.c1op_coef(f, v) for f in fb.freqs]
    return fb.col([x[0] * 9.0 for x in co]) + fb.col([x[1] for x in co]) * c1240 + fb.col([x[2] for x in co]) * c1444


def _peach_op(s, fb, n, v, unit, tdiv, toff, nmax, fix14):
    T, tlog = _g(s, "temp.t", n), _g(s, "temp.tlog", n)
    tlg = torch.from_numpy(_tab(v, unit, "tlg"))
    nt = torch.clamp(torch.clamp(torch.trunc(T / tdiv).to(torch.int64) - toff, max=nmax), min=1)
    dt = (tlog - tlg[nt - 1]) / (tlg[nt] - tlg[nt - 1])
    x = torch.from_numpy(np.stack([O.peach_x(f, fl, v, unit, fix14) for f, fl in zip(fb.freqs, fb.freqlgs)]))
    return libm.exp(x[:, nt - 1] * (1.0 - dt) + x[:, nt] * dt)


def mg1op(s, fb, n, v="atlas9"):
    return _peach_op(s, fb, n, v, "mg1op", 1000.0, 3, 6, False)


def si2op(s, fb, n, v="atlas9"):
    return _peach_op(s, fb, n, v, "si2op", 2000.0, 4, 5, True) * 6.0


def al1op(s, fb, n, v="atlas9"):
    return fb.col([O.al1op_coef(f) for f in fb.freqs])


def si1op(s, fb: FreqBatch, n, v="atlas9"):
    hckt = _g(s, "temp.hckt", n)
    elev, glev = _tab(v, "si1op", "elev"), _tab(v, "si1op", "glev")
    bolt = [float(glev[i]) * libm.exp(-float(elev[i]) * hckt) for i in range(33)]
    co = [O.si1op_coef(f, v) for f in fb.freqs]
    gfactor = 6.0
    num = fb.col([x[1] * gfactor * 2.0 / 2.0 for x in co])
    rz2 = co[0][3]
    elim = co[0][4]
    h = rdiv(num, rz2 * hckt) * (libm.exp(fb.col([-x[2] for x in co]) * hckt) - libm.exp(-elim * hckt))
    for i in range(1, 34):
        h = h + fb.col([x[0][i] for x in co]) * bolt[i - 1]
    return h * fb.stim


def fe1op(s, fb, n, v="atlas9"):
    hkt = _g(s, "temp.hkt", n)
    G, E = _tab(v, "fe1op", "g"), _tab(v, "fe1op", "e")
    co = [O.fe1op_coef(f, v) for f in fb.freqs]
    on = [x is not None for x in co]
    out = torch.zeros((fb.F, n), dtype=torch.float64, device=hkt.device)
    if not any(on):
        return out
    co = [x if x is not None else [0.0] * 48 for x in co]
    for i in range(48):
        out = out + fb.col([x[i] for x in co]) * (float(G[i]) * libm.exp(-float(E[i]) * 2.99792458e10 * hkt))
    return _zero_unless(fb.mask(on), out)


def _molop(s, fb, n, v, unit, cross_name, part_name, nmax, ln10, chop):
    T = _g(s, "temp.t", n)
    crts = [O.molop_crt(f, v, unit, cross_name, nmax, chop) for f in fb.freqs]
    on = [x is not None for x in crts]
    if not any(on):
        return torch.zeros((fb.F, n), dtype=torch.float64, device=T.device)
    crt = torch.from_numpy(np.stack([x if x is not None else np.zeros(15) for x in crts]))       # (F, 15)
    pt = torch.from_numpy(_tab(v, unit, part_name))
    it = torch.clamp(torch.trunc((T - 1000.0) / 200.0 + 1.0).to(torch.int64), min=1, max=len(pt) - 1)
    tn = (it.to(torch.float32) * 200.0 + 800.0).double()
    prt = pt[it - 1] + (pt[it] - pt[it - 1]) * (T - tn) / 200.0
    it2 = torch.clamp(torch.trunc((T - 2000.0) / 500.0 + 1.0).to(torch.int64), min=1, max=14)
    tn2 = (it2.to(torch.float32) * 500.0 + 1500.0).double()
    val = libm.exp((crt[:, it2 - 1] + (crt[:, it2] - crt[:, it2 - 1]) * (T - tn2) / 500.0) * ln10) * prt
    val = torch.where(T >= 9000.0, torch.zeros_like(val), val)
    return _zero_unless(fb.mask(on), val)


def chop(s, fb, n, v="atlas9"):
    return _molop(s, fb, n, v, "chop", "crossch", "partch", 105, r4("2.30258509299405E0"), True)


def ohop(s, fb, n, v="atlas9"):
    return _molop(s, fb, n, v, "ohop", "crossoh", "partoh", 130, 2.30258509299405, False)


def h2collop(s, fb: FreqBatch, n, v="atlas9"):
    T, rho = _g(s, "temp.t", n), _g(s, "state.rho", n)
    xnh2 = _xnh2(s, n)
    nus = [O.h2collop_nu(f, v) for f in fb.freqs]
    on = [x is not None for x in nus]
    if not any(on):
        return torch.zeros((fb.F, n), dtype=torch.float64, device=T.device)
    h2h2nu = torch.from_numpy(np.stack([x[0] if x is not None else np.zeros(7) for x in nus]))     # (F, 7)
    h2henu = torch.from_numpy(np.stack([x[1] if x is not None else np.zeros(7) for x in nus]))
    it = torch.clamp(torch.trunc(T / 1000.0).to(torch.int64), 1, 6)
    delt = (T - (it.to(torch.float32) * 1000.0).double()) / 1000.0
    delt = torch.clamp(torch.clamp(delt, max=1.0), min=0.0)
    xh2h2 = h2h2nu[:, it - 1] * (1.0 - delt) + h2h2nu[:, it] * delt
    xh2he = h2henu[:, it - 1] * (1.0 - delt) + h2henu[:, it] * delt
    a = (libm.pow(10.0, xh2he) * s["ions.xnfhe"][:n, 0] + libm.pow(10.0, xh2h2) * xnh2) * xnh2 / rho * fb.stim
    return _zero_unless(fb.mask(on), a)


def coolop(s, fb: FreqBatch, n, v="atlas9"):
    ah2coll = h2collop(s, fb, n, v)
    asi1 = si1op(s, fb, n, v)
    xc, xmg = s["xnfp.xnfpc"][:n, 0], s["xnfp.xnfpmg"][:n, 0]
    xal, xfe = s["xnfp.xnfpal"][:n].reshape(n, -1)[:, 0], s["xnfp.xnfpfe"][:n].reshape(n, -1)[:, 0]
    a = (c1op(s, fb, n, v) * xc + mg1op(s, fb, n, v) * xmg + al1op(s, fb, n, v) * xal + fe1op(s, fb, n, v) * xfe
         + chop(s, fb, n, v) * s["xnfp.xnfpch"][:n] + ohop(s, fb, n, v) * s["xnfp.xnfpoh"][:n]) * fb.stim \
        / _g(s, "state.rho", n) + asi1 * s["xnfp.xnfpsi"][:n, 0] / _g(s, "state.rho", n) + ah2coll
    on = fb.mask([f <= 3.28805e15 for f in fb.freqs])
    return {"acool": _zero_unless(on, a), "_asi1": asi1, "_cool_on": on[:, 0]}


# ------------------------------------------------------------------------------------------- metals (luke)

def lukeop(s, fb: FreqBatch, n, v="atlas9"):
    tkev = _g(s, "temp.tkev", n)
    c1130 = 6.0 * libm.exp(rdiv(-r4("3.575"), tkev))
    c1020 = 10.0 * libm.exp(rdiv(-r4("2.384"), tkev))
    c1169 = 6.0 * libm.exp(rdiv(-r4("4.43"), tkev))
    c1218 = 10.0 * libm.exp(rdiv(-r4("1.697"), tkev))
    c1420 = 6.0 * libm.exp(rdiv(-r4("3.142"), tkev))
    n1 = [O.n1op_coef(f) for f in fb.freqs]
    mg2 = [O.mg2op_coef(f) for f in fb.freqs]
    ca2 = [O.ca2op_coef(f) for f in fb.freqs]
    col = fb.col
    n1op = col([x[0] * 4.0 for x in n1]) + col([x[1] for x in n1]) * c1020 + col([x[2] for x in n1]) * c1130
    o1op = col([O.o1op_coef(f) for f in fb.freqs])
    mg2op = col([x[0] * 2.0 for x in mg2]) + col([x[1] for x in mg2]) * c1169
    ca2op = col([x[0] * 2.0 for x in ca2]) + col([x[1] for x in ca2]) * c1218 + col([x[2] for x in ca2]) * c1420
    a = (n1op * s["xnfp.xnfpn"][:n, 0] + o1op * s["xnfp.xnfpo"][:n, 0] + mg2op * s["xnfp.xnfpmg"][:n, 1]
         + si2op(s, fb, n, v) * s["xnfp.xnfpsi"][:n, 1] + ca2op * s["xnfp.xnfpca"][:n, 1]) * fb.stim \
        / _g(s, "state.rho", n)
    return {"aluke": a}


# ---------------------------------------------------------------------------------------------------- hot

def hotop(s, fb: FreqBatch, n, v="atlas9"):
    xnf = {k: s["xnf." + k][:n] for k in ("xnfc", "xnfn", "xnfo", "xnfne", "xnfmg", "xnfsi", "xnfs", "xnffe")}
    T, tkev, rho, xne = _g(s, "temp.t", n), _g(s, "temp.tkev", n), _g(s, "state.rho", n), _g(s, "state.xne", n)
    X = lambda k, i: xnf[k][:, i - 1]
    g = [None] + [coulff(s, fb, n, z, v) for z in range(1, 6)]
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
    ahot = free * r4("3.6919E8") / fb.col([ipow_const(f, 3) for f in fb.freqs]) * xne / libm.sqrt(T)
    for _ in range(12):                       # + AC2OP + ... + AO6OP (all zero)
        ahot = ahot + 0.0
    A = _eq(v, "hotop", "a")
    xnfp = _xnfp_flat(s, n)
    xss = [O.hotop_xs(f, v) for f in fb.freqs]
    for i in range(60):
        L = 7 * i
        on = [x[i] is not None for x in xss]
        if not any(on):
            continue
        xs = fb.col([x[i] if x[i] is not None else 0.0 for x in xss])
        idd = int(A[L + 6])
        xx = xs * xnfp[:, idd - 1] * float(A[L + 4])
        upd = (xx > ahot / 100.0) & fb.mask(on)
        ahot = torch.where(upd, ahot + xx / libm.exp(rdiv(float(A[L + 5]), tkev)), ahot)
    return {"ahot": ahot * fb.stim / rho}


def elecop(s, fb, n, v="atlas9"):
    return {"sigel": (0.6653e-24 * _g(s, "state.xne", n) / _g(s, "state.rho", n)).expand(fb.F, n)}


def h2raop(s, fb, n, v="atlas9"):
    xnh2 = _xnh2(s, n) / _g(s, "state.rho", n)
    return {"sigh2": fb.col([O.h2raop_sig(f) for f in fb.freqs]) * xnh2}


IFOP_ROUTINES = {1: hop, 2: h2plop, 3: hminop, 4: hrayop, 5: he1op, 6: he2op, 7: hemiop, 8: heraop, 9: coolop,
                 10: lukeop, 11: hotop, 12: elecop, 13: h2raop}


def continuum(s, fb: FreqBatch, variant: str = "atlas9") -> dict:
    """KAPP(N=0)'s source calls for all frequencies of `fb`: {ops member: [F, n]} for the members the enabled
    sources write (the others are KAPP's zeros), plus '_asi1' [F, n] and '_cool_on' [F] (COOLOP's early RETURN
    leaves /BSI/ASI1 untouched) when COOLOP is enabled."""
    n = s.iscalar("rhox.nrhox")
    ifop = [int(x) for x in s["ifop.ifop"].reshape(-1)]
    for k in (19, 20):
        if ifop[k - 1] == 1:
            raise NotImplementedError(f"IFOP({k}) (XCONOP/XSOP) is not implemented")
    out = {}
    for k in range(1, 14):
        if ifop[k - 1] == 1:
            out.update(IFOP_ROUTINES[k](s, fb, n, variant))
    return out
