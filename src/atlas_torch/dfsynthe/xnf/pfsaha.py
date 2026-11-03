"""PFSAHA / PFIRON / PFGROUND of xnfdf.for, vectorised over layers.

`Layers` carries the per-layer COMMON inputs (any common batch shape S, e.g. (models, NRHOX) or
(models,)): T, TKEV, TK, HCKT, TLOG (/TEMP/), P, XNE (/STATE/), the departure-coefficient arrays
(shape S+(nlev,)) and NLTEON. pfsaha(L, iz, nion, mode) returns ANSWER(J,1..k) as a tensor S+(k,)
with k = NION for MODE>10 and k = 1 otherwise.

Fortran literal kinds are reproduced: default REAL constants (and constant subexpressions such as
109677.576/6.5/6.5 or 1./3.) are single precision; PFIRON runs entirely in REAL*4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from .fdata import T as tab, nnn, pfground, potion, pftab
from .fmath import F, fsqrt, powi, rdiv

DT = torch.float64
F32 = torch.float32


def c32(*ops) -> float:
    """Evaluate a constant expression left to right in REAL*4: c32(a, '/', b, '/', c) etc."""
    v = np.float32(ops[0])
    for k in range(1, len(ops), 2):
        o, b = ops[k], np.float32(ops[k + 1])
        v = {"/": v / b, "*": v * b, "+": v + b, "-": v - b}[o]
    return float(v)


@dataclass
class Layers:
    t: torch.Tensor
    tkev: torch.Tensor
    tk: torch.Tensor
    hckt: torch.Tensor
    tlog: torch.Tensor
    p: torch.Tensor
    xne: torch.Tensor
    b: dict = field(default_factory=dict)    # "bhyd" -> S+(8,), ...
    nlteon: int = 0

    def with_(self, **kw) -> "Layers":
        d = dict(self.__dict__)
        d.update(kw)
        return Layers(**d)

    def B(self, name, i):
        """Departure coefficient B(J,I) (I 1-based), or 1 when NLTEON = -1."""
        if self.nlteon == -1 or name not in self.b:
            return None
        return self.b[name][..., i - 1]


@lru_cache(maxsize=None)
def _tabs():
    d = {k: torch.as_tensor(tab("pfsaha", k), dtype=DT) for k in (
        "ehyd", "ghyd", "ehe1", "ghe1", "ehe2", "ghe2", "eb1", "gb1", "ec1", "gc1", "ec2", "gc2", "eo1", "go1",
        "ena1", "gna1", "emg1", "gmg1", "emg2", "gmg2", "eal1", "gal1", "esi1", "gsi1", "esi2", "gsi2", "ek1",
        "gk1", "eca1", "gca1", "eca2", "gca2", "scale")}
    d["locz"] = [int(v) for v in tab("pfsaha", "locz")]
    d["nnn"] = torch.as_tensor(nnn(), dtype=torch.int64)
    d["potion"] = potion()
    return d


@lru_cache(maxsize=None)
def _pfiron_tabs():
    return (torch.as_tensor(pftab(), dtype=F32),
            torch.as_tensor(tab("pfiron", "potlo"), dtype=F32),
            torch.as_tensor(tab("pfiron", "potlolog"), dtype=F32))


def pfiron(nelem: int, ion: int, tlog8: torch.Tensor, potlow8: torch.Tensor) -> torch.Tensor:
    """SUBROUTINE PFIRON(NELEM,ION,TLOG8,POTLOW8,PF): REAL*4 table interpolation (Fe group 20..28)."""
    pft, potlo, potlolog = _pfiron_tabs()
    tl = tlog8.to(F32)
    pl = potlow8.to(F32)
    f32 = lambda v: torch.tensor(v, dtype=F32)
    hi = tl > f32(4.)
    lo = tl < f32(3.7)
    it_hi = torch.trunc((tl - f32(4.0)) / f32(.05) + f32(31.)).to(torch.int64).clamp(max=56)
    it_lo = torch.trunc((tl - f32(3.32)) / f32(.02) + f32(2.)).to(torch.int64).clamp(min=2)
    it_md = torch.trunc((tl - f32(3.7)) / f32(.03) + f32(21.)).to(torch.int64)
    it = torch.where(hi, it_hi, torch.where(lo, it_lo, it_md))
    f_hi = (tl - (it - 31).to(F32) * f32(.05) - f32(4.)) / f32(.05)
    f_lo = (tl - (it - 2).to(F32) * f32(.02) - f32(3.32)) / f32(.02)
    f_md = (tl - (it - 21).to(F32) * f32(.03) - f32(3.7)) / f32(.03)
    f = torch.where(hi, f_hi, torch.where(lo, f_lo, f_md))
    # LOW: first LOW in 1..7 with POTLOW < POTLO(LOW); none -> 7 (label 32)
    less = pl.unsqueeze(-1) < potlo                       # S+(7,)
    anyl = less.any(-1)
    first = torch.argmax(less.to(torch.int8), dim=-1) + 1  # 1-based
    low = torch.where(anyl, first, torch.full_like(first, 7))
    interp35 = anyl & (first > 1)
    tb = pft[:, :, ion - 1, nelem - 20]                    # (7, 56)
    itc = it.clamp(1, 56)

    def g(l, i):
        return tb[(l - 1).clamp(0, 6), (i - 1).clamp(0, 55)]
    one = f32(1.)
    pf32 = f * g(low, itc) + (one - f) * g(low, itc - 1)
    p = (torch.log10(pl.to(DT)).to(F32) - potlolog[(low - 2).clamp(0, 6)]) / f32(.30103)
    pf35 = p * (f * g(low, itc) + (one - f) * g(low, itc - 1)) + \
        (one - p) * (f * g(low - 1, itc) + (one - f) * g(low - 1, itc - 1))
    return torch.where(interp35, pf35, pf32).to(DT)


def _sum_levels(L: Layers, bname, gs, es, i0, i1, base):
    """PART = base + sum_{I=i0..i1} G(I)*B(J,I)*EXP(-E(I)*HCKT(J)) (sequential)."""
    part = base
    for i in range(i0, i1 + 1):
        b = L.B(bname, i)
        term = gs[i - 1] * b if b is not None else gs[i - 1] * 1.
        part = part + term * torch.exp(-es[i - 1] * L.hckt)
    return part


def _b1(L: Layers, bname):
    b = L.B(bname, 1)
    return b if b is not None else torch.ones_like(L.t)


def _ex(c, L):
    return torch.exp(-F(c) * L.hckt)


def _special(n: int, L: Layers, d):
    """Special-case partition functions (labels 1100..1210). Returns (part, d1 or None, g or None,
    mask_for_label14 or None) where part is the value at label 18 or before label 14."""
    t = L.t
    if n == 1:      # H I
        part = 2. * _b1(L, "bhyd")
        hot = _sum_levels(L, "bhyd", d["ghyd"], d["ehyd"], 2, 6, part)
        part = torch.where(t < 9000., part, hot)
        return part, c32(109677.576, "/", 6.5, "/", 6.5) * L.hckt, None, ~(t < 9000.)
    if n == 3:      # He I
        part = _b1(L, "bhe1")
        hot = _sum_levels(L, "bhe1", d["ghe1"], d["ehe1"], 2, 29, part)
        part = torch.where(t < 15000., part, hot)
        return part, c32(109677.576, "/", 5.5, "/", 5.5) * L.hckt, None, ~(t < 15000.)
    if n == 4:      # He II (PART(2))
        part = 2. * _b1(L, "bhe2")
        hot = _sum_levels(L, "bhe2", d["ghe2"], d["ehe2"], 2, 6, part)
        part = torch.where(t < 30000., part, hot)
        return part, c32(4., "*", 109722.267, "/", 6.5, "/", 6.5) * L.hckt, None, ~(t < 30000.)
    h = L.hckt
    if n == 354:    # C I
        part = _b1(L, "bc1") * (1. + 3. * _ex(16.42, L) + 5. * _ex(43.42, L))
        part = _sum_levels(L, "bc1", d["gc1"], d["ec1"], 2, 14, part)
        part = part + 108. * _ex(80000., L) + F(189) * _ex(84000., L) + 247. * _ex(87000., L) + \
            231. * _ex(88000., L) + 190. * _ex(89000., L) + 300. * _ex(90000., L)
        return part, None, None, None
    if n == 355:    # C II
        part = _b1(L, "bc2") * (2. + 4. * _ex(63.42, L))
        part = _sum_levels(L, "bc2", d["gc2"], d["ec2"], 2, 6, part)
        part = part + 6. * _ex(131731.80, L) + 4. * _ex(142027.1, L) + 10. * _ex(145550.13, L) + \
            10. * _ex(150463.62, L) + 2. * _ex(157234.07, L) + 6. * _ex(162500., L) + 42. * _ex(168000., L) + \
            56. * _ex(178000., L) + 102. * _ex(183000., L) + 400. * _ex(188000., L)
        return part, None, None, None
    if n == 51:     # Mg I
        part = _sum_levels(L, "bmg1", d["gmg1"], d["emg1"], 2, 11, _b1(L, "bmg1"))
        part = part + 5. * _ex(53134., L) + 15. * _ex(54192., L) + 28. * _ex(54676., L) + 9. * _ex(57853., L)
        return part, c32(109734.83, "/", 4.5, "/", 4.5) * h, 4., True
    if n == 52:     # Mg II
        part = _sum_levels(L, "bmg2", d["gmg2"], d["emg2"], 2, 6, _b1(L, "bmg2") * 2.)
        part = part + 10. * _ex(93310.80, L) + 14. * _ex(93799.70, L) + 6. * _ex(97464.32, L) + \
            10. * _ex(103419.82, L) + 14. * _ex(103689.89, L) + 18. * _ex(103705.66, L)
        return part, c32(4., "*", 109734.83, "/", 5.5, "/", 5.5) * h, 2., True
    if n == 57:     # Al I
        part = _b1(L, "bal1") * (2. + 4. * _ex(112.061, L))
        part = _sum_levels(L, "bal1", d["gal1"], d["eal1"], 2, 9, part)
        part = part + 10. * _ex(42235., L) + 14. * _ex(43831., L)
        return part, c32(109735.08, "/", 5.5, "/", 5.5) * h, 2., True
    if n == 63:     # Si I
        part = _b1(L, "bsi1") * (1. + 3. * _ex(77.115, L) + 5. * _ex(223.157, L))
        part = _sum_levels(L, "bsi1", d["gsi1"], d["esi1"], 2, 11, part)
        part = part + 76. * _ex(53000., L) + 71. * _ex(57000., L) + 191. * _ex(60000., L) + \
            240. * _ex(62000., L) + 251. * _ex(63000., L) + 300. * _ex(65000., L)
        return part, None, None, None
    if n == 64:     # Si II
        part = _b1(L, "bsi2") * (2. + 4. * _ex(287.32, L))
        part = _sum_levels(L, "bsi2", d["gsi2"], d["esi2"], 2, 6, part)
        part = part + 6. * _ex(81231.59, L) + 6. * _ex(83937.08, L) + 10. * _ex(101024.09, L) + \
            14. * _ex(103556.35, L) + 10. * _ex(108800., L) + 42. * _ex(115000., L) + 6. * _ex(121000., L) + \
            38. * _ex(125000., L) + 34. * _ex(132000., L)
        return part, c32(4., "*", 109734.83, "/", 4.5, "/", 4.5) * h, 2., True
    if n == 96:     # Ca I (unreachable: Ca goes to PFIRON)
        part = _sum_levels(L, "bca1", d["gca1"], d["eca1"], 2, 8, _b1(L, "bca1"))
        part = part + 28. * _ex(37000., L) + 67. * _ex(40000., L) + 21. * _ex(43000., L) + 34. * _ex(48000., L)
        return part, c32(109734.82, "/", 4.5, "/", 4.5) * h, 4., True
    if n == 97:     # Ca II (unreachable)
        part = _sum_levels(L, "bca2", d["gca2"], d["eca2"], 2, 5, _b1(L, "bca2") * 2.)
        part = part + 12. * _ex(68000., L)
        return part, c32(109734.83, "/", 4.5, "/", 4.5) * h, 2., True
    if n == 367:    # O I (unreachable in this variant: N never reaches 367)
        part = _b1(L, "bo1") * (5. + 3. * _ex(158.265, L) + _ex(226.977, L))
        part = _sum_levels(L, "bo1", d["go1"], d["eo1"], 2, 13, part)
        part = part + 15. * _ex(101140., L) + 131. * _ex(103000., L) + 128. * _ex(105000., L) + \
            600. * _ex(107000., L)
        return part, None, None, None
    if n == 45:     # Na I
        part = _sum_levels(L, "bna1", d["gna1"], d["ena1"], 2, 8, _b1(L, "bna1") * 2.)
        part = part + 10. * _ex(34548.745, L) + 14. * _ex(34586.96, L)
        return part, c32(109734.83, "/", 4.5, "/", 4.5) * h, 2., True
    if n == 14:     # B I
        part = _b1(L, "bb1") * (2. + 4. * _ex(15.25, L))
        part = _sum_levels(L, "bb1", d["gb1"], d["eb1"], 2, 7, part)
        part = part + 6. * _ex(57786.80, L) + 10. * _ex(59989., L) + 14. * _ex(60031.03, L) + 2. * _ex(63561., L)
        return part, c32(109734.83, "/", 4.5, "/", 4.5) * h, 2., True
    if n == 91:     # K I
        part = _sum_levels(L, "bk1", d["gk1"], d["ek1"], 2, 8, _b1(L, "bk1") * 2.)
        part = part + 10. * _ex(27397.077, L) + 14. * _ex(28127.85, L)
        return part, c32(109734.83, "/", 5.5, "/", 5.5) * h, 2., True
    return None


THIRD = c32(1., "/", 3.)
EIGHTEENTH = c32(1., "/", 18.)


def _label14(part, g, ip, tv, potlo, d1, z):
    d2 = potlo / tv
    c = rdiv(F(13.595) * z * z, tv)

    def poly(dd):
        return THIRD + (1. - (.5 + (EIGHTEENTH + dd / 120.) * dd) * dd) * dd
    return part + g * torch.exp(rdiv(-ip, tv)) * (powi(fsqrt(c / d2), 3) * poly(d2) - powi(fsqrt(c / d1), 3) * poly(d1))


def pfsaha(L: Layers, iz: int, nion: int, mode: int) -> torch.Tensor:
    d = _tabs()
    mode1 = mode - 10 if mode > 10 else mode
    t, tk, xne = L.t, L.tk, L.xne
    charge = xne * 2.
    excess = 2. * xne - L.p / tk
    charge = torch.where(excess > 0., charge + 2. * excess, charge)
    charge = torch.where(charge == 0., torch.ones_like(charge), charge)
    debye = fsqrt(tk / 2.8965e-18 / charge)
    potlow = torch.clamp(rdiv(1.44e-7, debye), max=1.0)
    tkev = L.tkev
    locz = d["locz"]
    if iz <= 28:
        n = locz[iz - 1]
        nions = locz[iz] - n
    else:
        n = 3 * iz + 54
        nions = 3
    if iz == 6:
        n, nions = 354, 6
    if iz == 7:
        n, nions = 360, 6
    if 20 <= iz < 29:
        nions = 10
    nion2 = min(nion + 2, nions)
    n -= 1
    nn = d["nnn"]
    scale = d["scale"]
    pot = d["potion"]
    part, ip, potlo = {}, {}, {}
    for ion in range(1, nion2 + 1):
        z = float(ion)
        potlo[ion] = potlow * z
        index = iz * (iz + 1) // 2 + ion - 1 if iz <= 30 else iz * 5 + 341 + ion - 1
        ipv = pot[index - 1] / F(8065.479)
        if ipv == 0:
            ipv = pot[index - 2] / F(8065.479)
        ip[ion] = ipv
        if 20 <= iz < 29:
            part[ion] = pfiron(iz, ion, L.tlog / F(2.30258509299405), potlo[ion] * F(8065.479))
            continue
        n += 1
        g = float(int(nn[5, n - 1]) - (int(nn[5, n - 1]) // 100) * 100)
        sp = _special(n, L, d)
        if sp is not None:
            p0, d1, gsp, m14 = sp
            if d1 is None:
                part[ion] = p0
            else:
                gg = g if gsp is None else gsp
                p14 = _label14(p0, gg, ipv, tkev, potlo[ion], d1, z)
                part[ion] = p14 if m14 is True else torch.where(m14, p14, p0)
            continue
        # generic: NNN interpolation table
        t2000 = ipv * 2000. / 11.
        it = torch.trunc(t / t2000 - .5).to(torch.int64).clamp(1, 9)
        dt = t / t2000 - it.to(DT) - .5
        i = (it + 1) // 2
        col = nn[:, n - 1]
        v = col[i - 1]
        k1 = v // 100000
        k2 = v - k1 * 100000
        k3 = k2 // 10
        ksc = k2 - k3 * 10
        sc = scale[(ksc - 1).clamp(0, 3)]
        odd = (it % 2) == 1
        p1o = k1.to(DT) * sc
        p2o = k3.to(DT) * sc
        kp1 = torch.trunc(p1o)
        pmin_o = torch.where((dt < 0.) & (ksc <= 1) & (kp1 == torch.trunc(p2o + .5)), kp1, torch.ones_like(dt))
        v2 = col[i.clamp(max=5)]
        k1e = v2 // 100000
        p2e = k1e.to(DT) * scale[((v2 % 10) - 1).clamp(0, 3)]
        p1e = k3.to(DT) * sc
        p1 = torch.where(odd, p1o, p1e)
        p2 = torch.where(odd, p2o, p2e)
        pmin = torch.where(odd, pmin_o, torch.ones_like(dt))
        pt = torch.maximum(pmin, p1 + (p2 - p1) * dt)
        cold = t < t2000 * 2.0
        nelion = (iz - 1) * 6 + ion
        pt_cold = torch.maximum(pfground(nelion, t), pt)
        skip14 = (g == 0.) | (potlo[ion] < F(.1)) | (t < t2000 * 4.)
        if g != 0.:
            tv = torch.where(t > t2000 * 11., torch.full_like(t, t2000 * 11. * 8.6171e-5), tkev)
            pt14 = _label14(pt, g, ipv, tv, potlo[ion], rdiv(F(.1), tv), z)
            pt = torch.where(skip14, pt, pt14)
        part[ion] = torch.where(cold, pt_cold, pt)
    k = nion if mode > 10 else 1
    if mode1 == 3:
        if mode > 10:
            return torch.stack([part[i] for i in range(1, nion + 1)], -1)
        return part[nion].unsqueeze(-1)
    cf = 2. * 2.4148e15 * t * fsqrt(t) / xne
    f = {1: torch.ones_like(t)}
    for ion in range(2, nion2 + 1):
        f[ion] = cf * part[ion] / part[ion - 1] * torch.exp(-torch.clamp((ip[ion - 1] - potlo[ion - 1]) / tkev, max=75.))
    f1 = torch.ones_like(t)
    l = nion2 + 1
    for ion in range(2, nion2 + 1):
        l -= 1
        f1 = 1. + f[l] * f1
    f[1] = rdiv(1., f1)
    for ion in range(2, nion2 + 1):
        f[ion] = f[ion - 1] * f[ion]
    if mode1 == 4:
        a = torch.zeros_like(t)
        for ion in range(2, nion2 + 1):
            a = a + f[ion] * float(ion - 1)
        return a.unsqueeze(-1)
    ions = range(1, nion + 1) if mode > 10 else [nion]
    if mode1 == 1:
        return torch.stack([f[i] / part[i] for i in ions], -1)
    return torch.stack([f[i] for i in ions], -1)
