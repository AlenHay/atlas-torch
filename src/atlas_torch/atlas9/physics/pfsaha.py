"""PFSAHA, PFIRON, PFGROUND (atlas9mem.for / kappa9.for): partition functions and Saha ionisation.

PFSAHA(J,IZ,NION,MODE,ANSWER) is vectorised over layers: `pfsaha(L, iz, nion, mode)` evaluates it for
every layer in the Layers bundle `L` at once (the Fortran calls it once per J). Branches that depend on
IZ/ION are Python branches (they are the same for all layers); branches on T(J) are torch.where.

Literal kinds are reproduced exactly: r4('13.595') is the REAL*4 literal widened to double; constant
sub-expressions made only of REAL*4 literals (e.g. 109677.576/6.5/6.5, 1./3.) are folded in binary32 as
gfortran/ifx do (np.float32 arithmetic). PFIRON is a REAL*4 routine and runs in float32.

The routine is identical in both variants apart from table assembly (EQUIVALENCE through NNNvec in
atlas9mem) and the ionisation potentials in /POTION/ (IONPOTS DATA differ: 2017 NIST values in atlas9mem,
older compilation in kappa9), which are passed in through the State (potion.potion).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from . import libm
from .libm import rdiv
from ._data import equivalenced, pfground_expr, r4, table
from ._fexpr import evaluate

F = np.float32


def c4(*ops) -> float:
    """Fold a constant REAL*4 expression left to right: c4('109677.576', '/', '6.5', '/', '6.5')."""
    v = F(r4(ops[0]))
    for k in range(1, len(ops), 2):
        op, b = ops[k], F(r4(ops[k + 1]))
        v = {"+": v + b, "-": v - b, "*": v * b, "/": v / b}[op]
    return float(v)


@dataclass
class Layers:
    """The per-layer COMMON data PFSAHA reads (/TEMP/, /STATE/, /DEPART/), any 1-D batch of layers."""
    t: torch.Tensor
    tk: torch.Tensor
    tkev: torch.Tensor
    hckt: torch.Tensor
    tlog: torch.Tensor
    xne: torch.Tensor
    p: torch.Tensor
    bhyd: torch.Tensor       # (n, 6)
    nlteon: int = 0

    @classmethod
    def from_state(cls, s, n: int | None = None) -> "Layers":
        n = s.iscalar("rhox.nrhox") if n is None else n
        g = lambda k: s[k][:n]
        return cls(g("temp.t"), g("temp.tk"), g("temp.tkev"), g("temp.hckt"), g("temp.tlog"), g("state.xne"),
                   g("state.p"), s["depart.bhyd"][:n], int(s.iscalar("depart.nlteon")))

    def index(self, idx) -> "Layers":
        return Layers(self.t[idx], self.tk[idx], self.tkev[idx], self.hckt[idx], self.tlog[idx], self.xne[idx],
                      self.p[idx], self.bhyd[idx], self.nlteon)

    def replace(self, **kw) -> "Layers":
        d = dict(self.__dict__)
        d.update(kw)
        return Layers(**d)


# ------------------------------------------------------------------------------------------------ tables

@lru_cache(maxsize=None)
def _pfsaha_tables(variant: str):
    t = {k: table(variant, "pfsaha", k) for k in (
        "ehyd", "ghyd", "ehe1", "ghe1", "ehe2", "ghe2", "eb1", "gb1", "ec1", "gc1", "ec2", "gc2", "eo1", "go1",
        "ena1", "gna1", "emg1", "gmg1", "emg2", "gmg2", "eal1", "gal1", "esi1", "gsi1", "esi2", "gsi2",
        "ek1", "gk1", "eca1", "gca1", "eca2", "gca2", "locz", "scale")}
    t["nnn"] = equivalenced(variant, "pfsaha", "nnn")
    return t


@lru_cache(maxsize=None)
def _pfiron_tables(variant: str):
    return (equivalenced(variant, "pfiron", "pftab"),
            table(variant, "pfiron", "potlo").astype(np.float32),
            table(variant, "pfiron", "potlolog").astype(np.float32))


# ------------------------------------------------------------------------------------------------ PFIRON

_PFI = {k: F(r4(k)) for k in ("4.", "3.7", "3.32", ".02", "2.", ".03", "21.", "4.0", ".05", "31.", "1.", ".30103")}


def pfiron_stage1(nelem: int, ion: int, tlog8: torch.Tensor, variant: str = "atlas9") -> torch.Tensor:
    """PFIRON's temperature interpolation for all 7 lowering rows: A(LOW) = F*PFTAB(LOW,IT)+(1-F)*PFTAB(LOW,IT-1)
    (REAL*4), shape (n, 7)."""
    pftab, _, _ = _pfiron_tables(variant)
    c = {k: torch.tensor(v, dtype=torch.float32, device=tlog8.device) for k, v in _PFI.items()}
    tl = tlog8.to(torch.float32)
    it_hi = torch.clamp(torch.trunc((tl - c["4.0"]) / c[".05"] + c["31."]).to(torch.int64), max=56)
    f_hi = (tl - (it_hi - 31).to(torch.float32) * c[".05"] - c["4."]) / c[".05"]
    it_lo = torch.clamp(torch.trunc((tl - c["3.32"]) / c[".02"] + c["2."]).to(torch.int64), min=2)
    f_lo = (tl - (it_lo - 2).to(torch.float32) * c[".02"] - c["3.32"]) / c[".02"]
    it_mid = torch.trunc((tl - c["3.7"]) / c[".03"] + c["21."]).to(torch.int64)
    f_mid = (tl - (it_mid - 21).to(torch.float32) * c[".03"] - c["3.7"]) / c[".03"]
    hi, lo = tl > c["4."], tl < c["3.7"]
    it = torch.where(hi, it_hi, torch.where(lo, it_lo, it_mid))
    f = torch.where(hi, f_hi, torch.where(lo, f_lo, f_mid))
    tab = torch.from_numpy(np.ascontiguousarray(pftab[:, :, ion - 1, nelem - 20])).to(tl.device)   # (7,56)
    one = c["1."]
    a = tab[:, it - 1].T * f[:, None] + (one - f)[:, None] * tab[:, it - 2].T
    return a                                                     # (n, 7) float32


def pfiron_stage2(a: torch.Tensor, potlow8: torch.Tensor, variant: str = "atlas9") -> torch.Tensor:
    """PFIRON's lowering interpolation (labels 30-35) given stage-1 rows `a` (..., 7) and POTLOW8 (...)."""
    _, potlo, potlolog = _pfiron_tables(variant)
    pl = potlow8.to(torch.float32)
    potlo_t = torch.from_numpy(potlo).to(pl.device)
    less = pl[..., None] < potlo_t
    anyless = less.any(-1)
    first = torch.where(anyless, less.to(torch.int64).argmax(-1) + 1, torch.zeros_like(pl, dtype=torch.int64))
    direct = (first == 1) | (first == 0)          # label 32 (LOW=1, or LOW=7 after the loop)
    low = torch.where(first == 0, torch.full_like(first, 7), first)
    lowm = torch.clamp(low - 1, min=1)
    a_low = torch.gather(a, -1, (low - 1)[..., None])[..., 0]
    a_lowm = torch.gather(a, -1, (lowm - 1)[..., None])[..., 0]
    pp = (libm.log10(pl) - torch.from_numpy(potlolog).to(pl.device)[lowm - 1]) / torch.tensor(_PFI[".30103"], dtype=torch.float32)
    one = torch.tensor(_PFI["1."], dtype=torch.float32)
    pf = torch.where(direct, a_low, pp * a_low + (one - pp) * a_lowm)
    return pf.to(torch.float64)


def _pfiron_stage2_np(a: np.ndarray, potlow8: np.ndarray, variant: str) -> np.ndarray:
    """pfiron_stage2 on numpy arrays (same REAL*4 operations)."""
    _, potlo, potlolog = _pfiron_tables(variant)
    pl = np.asarray(potlow8).astype(np.float32)
    less = pl[..., None] < potlo
    anyless = less.any(-1)
    first = np.where(anyless, less.astype(np.int64).argmax(-1) + 1, 0)
    direct = (first == 1) | (first == 0)
    low = np.where(first == 0, 7, first)
    lowm = np.maximum(low - 1, 1)
    a_low = np.take_along_axis(a, (low - 1)[..., None], -1)[..., 0]
    a_lowm = np.take_along_axis(a, (lowm - 1)[..., None], -1)[..., 0]
    lg = libm.log10(torch.from_numpy(np.ascontiguousarray(pl))).numpy()
    pp = (lg - potlolog[lowm - 1]) / np.float32(_PFI[".30103"])
    one = np.float32(_PFI["1."])
    pf = np.where(direct, a_low, pp * a_low + (one - pp) * a_lowm)
    return pf.astype(np.float64)


def pfiron(nelem: int, ion: int, tlog8: torch.Tensor, potlow8: torch.Tensor, variant: str = "atlas9") -> torch.Tensor:
    """SUBROUTINE PFIRON(NELEM,ION,TLOG8,POTLOW8,PF) (REAL*4 routine), vectorised over the layer batch."""
    return pfiron_stage2(pfiron_stage1(nelem, ion, tlog8, variant), potlow8, variant)


# ---------------------------------------------------------------------------------------------- PFGROUND

HCK = 6.6256e-27 * 2.99792458e10 / 1.38054e-16


def pfground(nelion: int, t: torch.Tensor, variant: str = "atlas9"):
    """FUNCTION PFGROUND(NELION,T), elementwise over a tensor of temperatures."""
    v = evaluate(pfground_expr(variant, nelion), {"HCK": HCK, "T": t})
    if not isinstance(v, torch.Tensor):
        v = torch.full_like(t, float(v))
    return v.to(torch.float64)


# ------------------------------------------------------------------------------------------------ PFSAHA

# REAL*4 literals / folded constants of PFSAHA
_C = {
    "8065.479": r4("8065.479"), "2.30258509299405": r4("2.30258509299405"), ".1": r4(".1"),
    "13.595": r4("13.595"), "1/3": c4("1.", "/", "3."), "1/18": c4("1.", "/", "18."),
    "D1H": c4("109677.576", "/", "6.5", "/", "6.5"), "D1HE1": c4("109677.576", "/", "5.5", "/", "5.5"),
    "D1HE2": c4("4.", "*", "109722.267", "/", "6.5", "/", "6.5"),
    "D1MG1": c4("109734.83", "/", "4.5", "/", "4.5"), "D1MG2": c4("4.", "*", "109734.83", "/", "5.5", "/", "5.5"),
    "D1AL1": c4("109735.08", "/", "5.5", "/", "5.5"), "D1SI2": c4("4.", "*", "109734.83", "/", "4.5", "/", "4.5"),
    "D1CA1": c4("109734.82", "/", "4.5", "/", "4.5"), "D1CA2": c4("109734.83", "/", "4.5", "/", "4.5"),
    "D1K1": c4("109734.83", "/", "5.5", "/", "5.5"),
}


def _bsum(base, g, e, hckt, i0=1):
    """PART = base; DO I=i0+1..n: PART=PART+G(I)*B*EXP(-E(I)*HCKT) with B=1."""
    for i in range(i0, len(g)):
        base = base + float(g[i]) * 1.0 * libm.exp(-float(e[i]) * hckt)
    return base


def _terms(part, pairs, hckt):
    """PART=PART+c1*EXP(-e1*HCKT)+c2*EXP(-e2*HCKT)+... (literal pairs, evaluated left to right)."""
    acc = part
    for cst, en in pairs:
        acc = acc + cst * libm.exp(-en * hckt)
    return acc


def _l(*vals):
    return [r4(v) if isinstance(v, str) else float(v) for v in vals]


def _special(n: int, L: Layers, tb):
    """Labels 1100..1210: closed-form partition functions. Returns (part, ion_slot, g, d1) — d1 None means
    GO TO 18 (no high-level tail)."""
    h = L.hckt
    ex = lambda c: libm.exp(-r4(c) * h)
    if n == 1:                     # H I with departure coefficients
        b = L.bhyd[:, 0] if L.nlteon != -1 else torch.ones_like(h)
        part = 2.0 * b
        for i in range(1, 6):
            b = L.bhyd[:, i] if L.nlteon != -1 else torch.ones_like(h)
            part = part + float(tb["ghyd"][i]) * b * libm.exp(-float(tb["ehyd"][i]) * h)
        return part, 1, None, _C["D1H"] * h
    if n == 3:                     # He I
        return _bsum(torch.ones_like(h), tb["ghe1"], tb["ehe1"], h), 1, None, _C["D1HE1"] * h
    if n == 4:                     # He II
        return _bsum(2.0 * torch.ones_like(h), tb["ghe2"], tb["ehe2"], h), 2, None, _C["D1HE2"] * h
    if n == 354:                   # C I
        part = 1.0 * (1.0 + 3.0 * ex("16.42") + 5.0 * ex("43.42"))
        part = _bsum(part, tb["gc1"], tb["ec1"], h)
        part = _terms(part, zip(_l("108.", 189, "247.", "231.", "190.", "300."),
                                _l("80000.", "84000.", "87000.", "88000.", "89000.", "90000.")), h)
        return part, 1, None, None
    if n == 355:                   # C II
        part = 1.0 * (2.0 + 4.0 * ex("63.42"))
        part = _bsum(part, tb["gc2"], tb["ec2"], h)
        part = _terms(part, zip(_l("6.", "4.", "10.", "10.", "2.", "6.", "42.", "56.", "102.", "400."),
                                _l("131731.80", "142027.1", "145550.13", "150463.62", "157234.07", "162500.",
                                   "168000.", "178000.", "183000.", "188000.")), h)
        return part, 2, None, None
    if n == 51:                    # Mg I
        part = _bsum(torch.ones_like(h), tb["gmg1"], tb["emg1"], h)
        part = _terms(part, zip(_l("5.", "15.", "28.", "9."), _l("53134.", "54192.", "54676.", "57853.")), h)
        return part, 1, 4.0, _C["D1MG1"] * h
    if n == 52:                    # Mg II
        part = _bsum(torch.ones_like(h) * 2.0, tb["gmg2"], tb["emg2"], h)
        part = _terms(part, zip(_l("10.", "14.", "6.", "10.", "14.", "18."),
                                _l("93310.80", "93799.70", "97464.32", "103419.82", "103689.89", "103705.66")), h)
        return part, 2, 2.0, _C["D1MG2"] * h
    if n == 57:                    # Al I
        part = 1.0 * (2.0 + 4.0 * ex("112.061"))
        part = _bsum(part, tb["gal1"], tb["eal1"], h)
        part = _terms(part, zip(_l("10.", "14."), _l("42235.", "43831.")), h)
        return part, 1, 2.0, _C["D1AL1"] * h
    if n == 63:                    # Si I
        part = 1.0 * (1.0 + 3.0 * ex("77.115") + 5.0 * ex("223.157"))
        part = _bsum(part, tb["gsi1"], tb["esi1"], h)
        part = _terms(part, zip(_l("76.", "71.", "191.", "240.", "251.", "300."),
                                _l("53000.", "57000.", "60000.", "62000.", "63000.", "65000.")), h)
        return part, 1, None, None
    if n == 64:                    # Si II
        part = 1.0 * (2.0 + 4.0 * ex("287.32"))
        part = _bsum(part, tb["gsi2"], tb["esi2"], h)
        part = _terms(part, zip(_l("6.", "6.", "10.", "14.", "10.", "42.", "6.", "38.", "34."),
                                _l("81231.59", "83937.08", "101024.09", "103556.35", "108800.", "115000.",
                                   "121000.", "125000.", "132000.")), h)
        return part, 2, 2.0, _C["D1SI2"] * h
    if n == 367:                   # O I
        part = 1.0 * (5.0 + 3.0 * ex("158.265") + ex("226.977"))
        part = _bsum(part, tb["go1"], tb["eo1"], h)
        part = _terms(part, zip(_l("15.", "131.", "128.", "600."),
                                _l("101140.", "103000.", "105000.", "107000.")), h)
        return part, 1, None, None
    if n == 45:                    # Na I
        part = _bsum(torch.ones_like(h) * 2.0, tb["gna1"], tb["ena1"], h)
        part = _terms(part, zip(_l("10.", "14."), _l("34548.745", "34586.96")), h)
        return part, 1, 2.0, _C["D1CA2"] * h
    if n == 14:                    # B I
        part = 1.0 * (2.0 + 4.0 * ex("15.25"))
        part = _bsum(part, tb["gb1"], tb["eb1"], h)
        part = _terms(part, zip(_l("6.", "10.", "14.", "2."), _l("57786.80", "59989.", "60031.03", "63561.")), h)
        return part, 1, 2.0, _C["D1CA2"] * h
    if n == 91:                    # K I
        part = _bsum(torch.ones_like(h) * 2.0, tb["gk1"], tb["ek1"], h)
        part = _terms(part, zip(_l("10.", "14."), _l("27397.077", "28127.85")), h)
        return part, 1, 2.0, _C["D1K1"] * h
    return None




_SPECIAL = (1, 3, 4, 354, 51, 57, 63, 355, 52, 64, 367, 45, 14, 91)

# item kinds
DEFAULT, SPEC_TAIL, SPEC, IRON = 0, 1, 2, 3


def _blk(c, d):
    """SQRT(13.595*Z*Z/TV/D)**3*(1./3.+(1.-(.5+(1./18.+D/120.)*D)*D)*D) with c = 13.595*Z*Z/TV."""
    return libm.ipow_const(libm.sqrt(c / d), 3) * (_C["1/3"] + (1.0 - (0.5 + (_C["1/18"] + d / 120.0) * d) * d) * d)


def _nions(iz: int, variant: str) -> int:
    locz = _pfsaha_tables(variant)["locz"]
    nions = int(locz[iz] - locz[iz - 1]) if iz <= 28 else 3
    if iz == 6 or iz == 7:
        nions = 6
    if 20 <= iz < 29:
        nions = 10
    return nions


def _item_static(iz: int, ion: int, potion: np.ndarray, variant: str):
    """Layer-independent data of PFSAHA's loop 18 for (IZ, ION): (kind, ip, z, g, n)."""
    tb = _pfsaha_tables(variant)
    index = iz * (iz + 1) // 2 + ion - 1 if iz <= 30 else iz * 5 + 341 + ion - 1
    ipv = float(potion[index - 1]) / _C["8065.479"]
    if ipv == 0:
        ipv = float(potion[index - 2]) / _C["8065.479"]
    if 20 <= iz < 29:
        return IRON, ipv, float(ion), 0.0, None
    locz = tb["locz"]
    n = int(locz[iz - 1]) if iz <= 28 else 3 * iz + 54
    if iz == 6:
        n = 354
    if iz == 7:
        n = 360
    n = n - 1 + ion
    nnn = tb["nnn"]
    g = float(int(nnn[5, n - 1]) - (int(nnn[5, n - 1]) // 100) * 100)
    if n in _SPECIAL:
        return SPEC, ipv, float(ion), g, n
    return DEFAULT, ipv, float(ion), g, n


def item_stage1(L: Layers, iz: int, ion: int, potion: np.ndarray, variant: str) -> dict:
    """The T-only part of PFSAHA's partition function for (IZ, ION) on all layers of L.

    Everything that depends on XNE (through the Debye lowering POTLOW) is left to item_stage2; the split
    keeps the Fortran's operation order (tail = PART + (G*EXP(-IP/TV))*(blk(D2)-blk(D1)))."""
    kind, ipv, z, g, n = _item_static(iz, ion, potion, variant)
    tv = L.tkev
    d = {"kind": kind, "ip": ipv, "z": z, "g": g}
    if kind == IRON:
        d["a"] = pfiron_stage1(iz, ion, L.tlog / _C["2.30258509299405"], variant)
        return d
    if kind == SPEC:
        part, _, gset, d1 = _special(n, L, _pfsaha_tables(variant))
        d["part0"] = part
        if d1 is not None:
            gg = gset if gset is not None else g
            d["kind"] = SPEC_TAIL
            d["tvx"] = tv
            d["c"] = rdiv(_C["13.595"] * z * z, tv)
            d["blk1"] = _blk(d["c"], d1)
            d["ge"] = gg * libm.exp(rdiv(-ipv, tv))
        return d
    # default: interpolate the NNN table in T/T2000
    tb = _pfsaha_tables(variant)
    dev = L.t.device
    t2000 = ipv * 2000.0 / 11.0
    x = L.t / t2000
    it = torch.clamp(torch.trunc(x - 0.5).to(torch.int64), 1, 9)
    dt = x - it.double() - 0.5
    i = (it + 1) // 2
    row = torch.from_numpy(tb["nnn"][:, n - 1].astype(np.int64)).to(dev)
    scale = torch.from_numpy(tb["scale"].astype(np.float64)).to(dev)
    ni = row[i - 1]
    k1 = ni // 100000
    k2 = ni - k1 * 100000
    k3 = k2 // 10
    ksc = k2 - k3 * 10
    odd = (it % 2) == 1
    p1_odd = k1.double() * scale[ksc - 1]
    p2_odd = k3.double() * scale[ksc - 1]
    kp1 = torch.trunc(p1_odd)
    pmin_set = odd & (dt < 0) & (ksc <= 1) & (kp1 == torch.trunc(p2_odd + 0.5))
    pmin = torch.where(pmin_set, kp1, torch.ones_like(dt))
    ni1 = row[torch.clamp(i, max=5)]                      # NNN(I+1,N), only used for even IT (I<=4)
    k1e = ni1 // 100000
    ksce = torch.remainder(ni1, 10)
    p1_even = k3.double() * scale[ksc - 1]
    p2_even = k1e.double() * scale[torch.clamp(ksce, 1, 4) - 1]
    p1 = torch.where(odd, p1_odd, p1_even)
    p2 = torch.where(odd, p2_odd, p2_even)
    part = torch.maximum(pmin, p1 + (p2 - p1) * dt)
    d["part0"] = part
    d["cold"] = L.t < t2000 * 2.0
    d["part_cold"] = torch.maximum(pfground((iz - 1) * 6 + ion, L.t, variant), part)
    d["tcool"] = L.t < t2000 * 4.0
    tv2 = torch.where(L.t > t2000 * 11.0, torch.full_like(tv, (t2000 * 11.0) * 8.6171e-5), tv)
    d["tvx"] = tv2
    d["c"] = rdiv(_C["13.595"] * z * z, tv2)
    d["blk1"] = _blk(d["c"], rdiv(_C[".1"], tv2))
    d["ge"] = g * libm.exp(rdiv(-ipv, tv2))
    return d


def potlow_of(L: Layers) -> torch.Tensor:
    """Debye lowering POTLOW (eV) of PFSAHA."""
    charge = L.xne * 2.0
    excess = 2.0 * L.xne - L.p / L.tk
    charge = torch.where(excess > 0, charge + 2.0 * excess, charge)
    charge = torch.where(charge == 0, torch.ones_like(charge), charge)
    debye = libm.sqrt(L.tk / 2.8965e-18 / charge)
    return torch.clamp(rdiv(1.44e-7, debye), max=1.0)


# ---- stage-1 memo: item_stage1 is a pure function of (IZ, ION, the layers' T-side arrays, POTION, variant).
# NMOLEC, its partition-function pass and the POPS calls of one iteration -- and CONVEC's two pressure-perturbed
# NMOLEC calls -- evaluate it on byte-identical layers; the memo (keyed on the exact bytes) returns the stored
# result of the same computation. Bounded by entry bytes (oldest evicted first).
_STAGE1: OrderedDict = OrderedDict()
_SIZES: dict = {}
_STAGE1_BYTES = [0]
STAGE1_BUDGET = 96 * 2**20


def _layers_key(L: Layers, potion: np.ndarray, variant: str):
    """Memo key of the layer bundle, or None (no memo: autograd inputs)."""
    arrs = (L.t, L.tk, L.tkev, L.hckt, L.tlog, L.bhyd)
    if any(a.requires_grad for a in arrs):
        return None
    h = hashlib.blake2b(digest_size=20)
    for a in arrs:
        h.update(np.ascontiguousarray(a.detach().cpu().numpy()).tobytes())
    h.update(np.ascontiguousarray(potion).tobytes())
    return (variant, int(L.nlteon), str(L.t.device), str(L.t.dtype), bool(libm.EXACT), h.digest())


def _stage1_cached(L, iz, ion, potion, variant, lkey):
    if lkey is None:
        return item_stage1(L, iz, ion, potion, variant)
    key = (lkey, iz, ion)
    d = _STAGE1.get(key)
    if d is not None:
        _STAGE1.move_to_end(key)
        return d
    d = item_stage1(L, iz, ion, potion, variant)
    nb = sum(v.element_size() * v.nelement() for v in d.values() if isinstance(v, torch.Tensor))
    _STAGE1[key] = d
    _STAGE1_BYTES[0] += nb
    _SIZES[key] = nb
    while _STAGE1_BYTES[0] > STAGE1_BUDGET and _STAGE1:
        k, _ = _STAGE1.popitem(last=False)
        _STAGE1_BYTES[0] -= _SIZES.pop(k)
    return d


class SahaPlan:
    """PFSAHA for several (IZ, NION2) rows at once: stage 1 (T only) is computed once for all layers,
    stage 2 (XNE dependent) + the Saha chain are vectorised over rows x ions, for all layers or any subset
    (NMOLEC calls it one layer at a time inside its layer recurrence)."""

    def __init__(self, L: Layers, rows: list[tuple[int, int]], potion, variant: str = "atlas9"):
        self.rows = rows
        self.variant = variant
        potion = potion if isinstance(potion, np.ndarray) else potion.detach().cpu().numpy()
        self.L = L
        n = L.t.shape[0]
        R = len(rows)
        M = max(nion2 for _, nion2 in rows)
        self.M = M
        dev = L.t.device
        cache = {}
        lkey = _layers_key(L, potion, variant)
        f64 = dict(dtype=torch.float64, device=dev)
        one, zero = torch.ones(n, **f64), torch.zeros(n, **f64)
        false = torch.zeros(n, dtype=torch.bool, device=dev)
        one7 = torch.ones((n, 7), dtype=torch.float32, device=dev)
        names = ("part0", "part_cold", "cold", "tcool", "tvx", "c", "blk1", "ge", "a")
        defaults = (one, one, false, false, one, one, zero, zero, one7)
        cols = {k: [] for k in names}
        z, ip, g0, kind, valid = [], [], [], [], []
        for r, (iz, nion2) in enumerate(rows):
            for m in range(M):
                ion = m + 1
                if ion > nion2:
                    d = None
                else:
                    key = (iz, ion)
                    if key not in cache:
                        cache[key] = _stage1_cached(L, iz, ion, potion, variant, lkey)
                    d = cache[key]
                if d is None:                                   # padding slot (ION > NION2 of the row)
                    z.append(1.0); ip.append(0.0); g0.append(False); kind.append(SPEC); valid.append(False)
                    use = ()
                else:
                    z.append(d["z"]); ip.append(d["ip"]); g0.append(d["g"] == 0); kind.append(d["kind"])
                    valid.append(True)
                    k_ = d["kind"]
                    use = ("a",) if k_ == IRON else ("part0",) + (
                        ("tvx", "c", "blk1", "ge") if k_ in (DEFAULT, SPEC_TAIL) else ()) + (
                        ("part_cold", "cold", "tcool") if k_ == DEFAULT else ())
                for k, dv in zip(names, defaults):
                    cols[k].append(d[k] if k in use else dv)
        stk = lambda k: torch.stack(cols[k], 1).reshape((n, R, M) + cols[k][0].shape[1:])
        part0, part_cold, cold, tcool = stk("part0"), stk("part_cold"), stk("cold"), stk("tcool")
        tvx, c, blk1, ge, a = stk("tvx"), stk("c"), stk("blk1"), stk("ge"), stk("a")
        z = torch.tensor(z, **f64).reshape(R, M)
        ip = torch.tensor(ip, **f64).reshape(R, M)
        g0 = torch.tensor(g0, dtype=torch.bool, device=dev).reshape(R, M)
        kind = torch.tensor(kind, dtype=torch.int64, device=dev).reshape(R, M)
        valid = torch.tensor(valid, dtype=torch.bool, device=dev).reshape(R, M)
        self.z, self.ip, self.g0, self.kind, self.valid = z, ip, g0, kind, valid
        self.part0, self.part_cold, self.cold, self.tcool = part0, part_cold, cold, tcool
        self.tvx, self.c, self.blk1, self.ge, self.a = tvx, c, blk1, ge, a
        self.nion2 = torch.tensor([n2 for _, n2 in rows], device=dev)
        self.has_iron = bool((kind == IRON).any())
        self._np_cache = None

    def parts(self, idx, Lsub: Layers):
        """PART(ION) and POTLO(ION) for layers `idx` (slice/index of the plan's layers) with the XNE etc.
        of Lsub (same layers). Returns (part (n, R, M), potlo (n, R, M))."""
        potlow = potlow_of(Lsub)
        potlo = potlow[:, None, None] * self.z
        tvx = self.tvx[idx]
        d2 = potlo / tvx
        tail = self.part0[idx] + self.ge[idx] * (_blk(self.c[idx], d2) - self.blk1[idx])
        skip = self.g0 | (potlo < _C[".1"]) | self.tcool[idx]
        dflt = torch.where(self.cold[idx], self.part_cold[idx], torch.where(skip, self.part0[idx], tail))
        part = torch.where(self.kind == DEFAULT, dflt, torch.where(self.kind == SPEC_TAIL, tail, self.part0[idx]))
        if self.has_iron:
            pf = pfiron_stage2(self.a[idx], potlo * _C["8065.479"], self.variant)
            part = torch.where(self.kind == IRON, pf, part)
        part = torch.where(self.valid, part, torch.ones_like(part))
        return part, potlo

    def fractions_layer(self, j: int, xne: float) -> np.ndarray:
        """fractions(slice(j, j+1), L.index(j) with XNE=xne)[0][0] as a numpy (R, M) array: the same operations
        on one layer in numpy (NMOLEC calls this once per layer inside its sequential layer recurrence, where
        torch's per-operation overhead dominated)."""
        c = self._np_cache
        if c is None:
            g = lambda a: a.detach().cpu().numpy()
            L = self.L
            c = self._np_cache = {
                "t": g(L.t), "tk": g(L.tk), "tkev": g(L.tkev), "p": g(L.p), "z": g(self.z), "ip": g(self.ip),
                "g0": g(self.g0), "kind": g(self.kind), "valid": g(self.valid), "part0": g(self.part0),
                "part_cold": g(self.part_cold), "cold": g(self.cold), "tcool": g(self.tcool), "tvx": g(self.tvx),
                "c": g(self.c), "blk1": g(self.blk1), "ge": g(self.ge), "a": g(self.a), "nion2": g(self.nion2)}
        t, tk, tkev, p = c["t"][j], c["tk"][j], c["tkev"][j], c["p"][j]
        with np.errstate(all="ignore"):
            # potlow_of
            charge = xne * 2.0
            excess = 2.0 * xne - p / tk
            if excess > 0:
                charge = charge + 2.0 * excess
            if charge == 0:
                charge = 1.0
            debye = np.sqrt(np.float64(tk / 2.8965e-18 / charge))
            potlow = np.minimum(np.float64(1.44e-7) / debye, 1.0)
            # parts
            potlo = potlow * c["z"]
            d2 = potlo / c["tvx"][j]
            cj = c["c"][j]
            blk2 = libm.ipow_const(np.sqrt(cj / d2), 3) * (_C["1/3"] + (1.0 - (0.5 + (_C["1/18"] + d2 / 120.0) * d2) * d2) * d2)
            part0 = c["part0"][j]
            tail = part0 + c["ge"][j] * (blk2 - c["blk1"][j])
            skip = c["g0"] | (potlo < _C[".1"]) | c["tcool"][j]
            dflt = np.where(c["cold"][j], c["part_cold"][j], np.where(skip, part0, tail))
            kind = c["kind"]
            part = np.where(kind == DEFAULT, dflt, np.where(kind == SPEC_TAIL, tail, part0))
            if self.has_iron:
                pf = _pfiron_stage2_np(c["a"][j], potlo * _C["8065.479"], self.variant)
                part = np.where(kind == IRON, pf, part)
            part = np.where(c["valid"], part, 1.0)
            # fractions
            cf = 2.0 * 2.4148e15 * t * np.sqrt(np.float64(t)) / xne
            ex = libm.exp(torch.from_numpy(-np.minimum((c["ip"][:, :-1] - potlo[:, :-1]) / tkev, 100.0))).numpy()
            ratio = cf * part[:, 1:] / part[:, :-1] * ex
        M = self.M
        cols = [None, None] + [ratio[:, k - 2] for k in range(2, M + 1)]
        f1 = np.ones(part.shape[0])
        for l in range(M, 1, -1):
            f1 = np.where(c["nion2"] >= l, 1.0 + cols[l] * f1, f1)
        f = [None, 1.0 / f1]
        for k in range(2, M + 1):
            f.append(f[k - 1] * cols[k])
        return np.stack(f[1:], -1)

    def fractions(self, idx, Lsub: Layers):
        """Ionisation fractions F(ION) (n, R, M) of each row (PFSAHA label 19-22), plus part, potlo."""
        part, potlo = self.parts(idx, Lsub)
        tv = Lsub.tkev[:, None, None]
        cf = (2.0 * 2.4148e15 * Lsub.t * libm.sqrt(Lsub.t) / Lsub.xne)[:, None, None]
        # F(ION)=CF*PART(ION)/PART(ION-1)*EXP(-MIN((IP(ION-1)-POTLO(ION-1))/TV,100)), ION>=2
        ex = libm.exp(-torch.clamp((self.ip[:, :-1] - potlo[..., :-1]) / tv, max=100.0))
        ratio = cf * part[..., 1:] / part[..., :-1] * ex
        M = self.M
        cols = [None] * (M + 1)
        for k in range(2, M + 1):
            cols[k] = ratio[..., k - 2]
        f1 = torch.ones_like(part[..., 0])
        for l in range(M, 1, -1):
            use = (self.nion2 >= l)
            f1 = torch.where(use, 1.0 + cols[l] * f1, f1)
        f = [None] * (M + 1)
        f[1] = rdiv(1.0, f1)
        for k in range(2, M + 1):
            f[k] = f[k - 1] * cols[k]
        return torch.stack(f[1:], -1), part, potlo


def pfsaha(L: Layers, iz: int, nion: int, mode: int, potion, variant: str = "atlas9"):
    """SUBROUTINE PFSAHA for all layers of L.

    Returns, as the Fortran stores into ANSWER(J,*):
      MODE 1/2/3 (+10): tensor (n, nion) for MODE>10, (n,) for MODE<10 (ion NION only);
      MODE 4: (n,) electrons per atom; MODE 5: (pf (n, nion), eion (n, nion+1)): pf = ANSWER(1..NION),
      eion = ANSWER(7..7+NION), i.e. EION(ION) (EQUIVALENCEd to ANSWER(6+ION) by the callers) is
      eion[:, ION-1] = IP(1)+...+IP(ION-1), the energy (eV) to form ion stage ION from the neutral.
    """
    mode1 = mode - 10 if mode > 10 else mode
    nion2 = min(nion + 2, _nions(iz, variant))
    plan = SahaPlan(L, [(iz, nion2)], potion, variant)
    sl = slice(None)
    if mode1 == 5:
        part, _ = plan.parts(sl, L)
        pf = part[:, 0, :nion]
        eion, acc = [0.0], 0.0          # ANSWER(7)=0, ANSWER(ION+7)=IP(ION)+ANSWER(ION+6)
        for k in range(nion):
            acc = float(plan.ip[0, k]) + acc
            eion.append(acc)
        return pf, torch.tensor(eion, dtype=torch.float64, device=L.t.device).expand(L.t.shape[0], nion + 1)
    if mode1 == 3:
        part, _ = plan.parts(sl, L)
        part = part[:, 0]
        return part[:, :nion] if mode >= 10 else part[:, nion - 1]
    f, part, _ = plan.fractions(sl, L)
    f, part = f[:, 0], part[:, 0]
    if mode1 == 4:
        a = torch.zeros_like(L.t)
        for ion in range(2, nion2 + 1):
            a = a + f[:, ion - 1] * float(ion - 1)
        return a
    if mode >= 10:
        return f[:, :nion] / part[:, :nion] if mode1 == 1 else f[:, :nion]
    return f[:, nion - 1] / part[:, nion - 1] if mode1 == 1 else f[:, nion - 1]
