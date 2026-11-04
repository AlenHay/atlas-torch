"""PFSAHA, PFIRON, PFGROUND, IONPOTS of atlas7v.for (SYNTHE variant): partition functions and Saha
ionisation fractions.

PFSAHA(J,IZ,NION,MODE,ANSWER) is vectorised over layers: `pfsaha(L, iz, nion, mode)` evaluates it for every
layer of the `Layers` bundle at once (the Fortran calls it once per J). Branches on IZ/ION are Python
branches; branches on T(J)/XNE(J) are torch.where.

The computation is split the same way as the Fortran's data flow, so that NMOLEC (which calls PFSAHA one
layer at a time inside its layer recurrence, with the XNE of the current Newton start) can reuse the
temperature-only work:
  stage 1 (`SahaPlan.__init__`): everything that depends on T only (NNN-table interpolation, PFGROUND,
          the closed-form partition functions of labels 1100-1210, the D1 part of the high-level tail,
          G*EXP(-IP/TV), PFIRON's temperature interpolation);
  stage 2 (`SahaPlan.parts/fractions`): the Debye lowering POTLOW(XNE), the D2 part of the tail, PFIRON's
          lowering interpolation, the Saha chain F(ION).
The split keeps the Fortran's operation order (PART=PART+(G*EXP(-IP/TV))*(blk(D2)-blk(D1))).

Literal kinds follow atlas7v.for exactly (this variant writes most constants with D0; the remaining
default-REAL literals are widened from binary32 with f32(), and constant REAL*4 sub-expressions such as
1./3. are folded in binary32 as gfortran does). Elementary functions go through atlas9.physics.libm
(glibc exp/sqrt, gfortran's powi chains for constant integer powers).

Shared code (identical Fortran per results/fmap/variants.md): PFIRON and IONPOTS are identical in
atlas7v.for and atlas9mem.for, so PFIRON's tensor port is imported from atlas_torch.atlas9.physics.pfsaha.
PFGROUND differs from the atlas9 copy in literal kinds (D0 here), so its case table is read from
atlas7v.for. NB: in this PFGROUND the `IF(NELION.EQ.431/491/501)GO TO ...` statements sit inside the
`IF(NELION.LT.114)THEN` block after an always-taken computed GO TO, so they are dead code: NELION 431
(O I), 491 (F I), 501 (F II) fall through to the final PFGROUND=1. -- reproduced here.

FSAVE/F/PART are SAVEd in the Fortran; a call with NION > NION2 would return stale values from a previous
call. No caller in xnfpelsyn/NMOLEC does that (NION <= NIONS for every code used); it raises here.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ..atlas9.physics import libm
from ..atlas9.physics._fexpr import evaluate, parse
from ..atlas9.physics.pfsaha import pfiron_stage1, pfiron_stage2
from ..fnum import f32
from ..tables import table

SRC = "atlas7v.for"
DT = torch.float64

# ------------------------------------------------------------------------------------------------ tables


@lru_cache(maxsize=None)
def _tb() -> dict:
    names = ("ehyd", "ghyd", "ehe1", "ghe1", "ehe2", "ghe2", "eb1", "gb1", "ec1", "gc1", "ec2", "gc2", "eo1", "go1",
             "ena1", "gna1", "emg1", "gmg1", "emg2", "gmg2", "eal1", "gal1", "esi1", "gsi1", "esi2", "gsi2",
             "ek1", "gk1", "eca1", "gca1", "eca2", "gca2", "locz", "scale", "nnn")
    return {k: np.asarray(table(SRC, "pfsaha", k)) for k in names}


@lru_cache(maxsize=None)
def potion() -> np.ndarray:
    """COMMON /POTION/POTION(999) as initialised by IONPOTS' DATA statements (cm^-1)."""
    return np.asarray(table(SRC, "ionpots", "potion"), dtype=np.float64)


def pfiron(nelem: int, ion: int, tlog8: torch.Tensor, potlow8: torch.Tensor) -> torch.Tensor:
    """SUBROUTINE PFIRON(NELEM,ION,TLOG8,POTLOW8,PF) (REAL*4 routine), elementwise over the batch."""
    return pfiron_stage2(pfiron_stage1(nelem, ion, tlog8, "atlas9"), potlow8, "atlas9")


# ---------------------------------------------------------------------------------------------- PFGROUND

HCK = 6.6256e-27 * 2.99792458e10 / 1.38054e-16      # PARAMETER (HCK=6.6256D-27*2.99792458D10/1.38054D-16)


def _source() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "builds" / "src" / SRC


@lru_cache(maxsize=None)
def _labelled_unit(unit: str) -> list:
    """(label, statement) list of one program unit of atlas7v.for (fixed form, continuations joined)."""
    lines = _source().read_text(encoding="latin-1").split("\n")
    hdr = re.compile(r"^[ \t]+(?:REAL\*8\s+|REAL\s+|INTEGER\s+)?(?:SUBROUTINE|FUNCTION)\s+(\w+)", re.I)
    out, inside = [], False
    for raw in lines:
        if not raw.strip() or raw[0] in "cC*!":
            continue
        m = hdr.match(raw)
        if m:
            inside = m.group(1).lower() == unit.lower()
        if not inside:
            continue
        lab, cont, text = raw[:5], len(raw) > 5 and raw[5] not in " 0", raw[6:72]
        if cont and out:
            out[-1] = (out[-1][0], out[-1][1] + text)
        else:
            out.append((int(lab) if lab.strip() else None, text.strip()))
        if re.match(r"^\s*END\s*$", text, re.I):
            break
    return out


@lru_cache(maxsize=None)
def pfground_cases() -> dict:
    """NELION -> Fortran expression assigned to PFGROUND (control flow of atlas7v's PFGROUND)."""
    stmts = _labelled_unit("pfground")
    labels = {lab: i for i, (lab, _) in enumerate(stmts) if lab is not None}
    gotos = []
    for _, s in stmts:
        m = re.match(r"^GO\s*TO\s*\(([\d,\s]+)\)\s*,?\s*(.*)$", s, re.I)
        if m:
            gotos.append([int(x) for x in m.group(1).split(",")])

    def run_from(i):
        val = None
        while i < len(stmts):
            s = stmts[i][1]
            m = re.match(r"^PFGROUND\s*=(.*)$", s, re.I)
            if m:
                val = m.group(1).strip()
            elif re.match(r"^RETURN\s*$", s, re.I):
                return val
            i += 1
        raise ValueError("PFGROUND case without RETURN")

    first, second = gotos[0], gotos[1]
    cases = {}
    for k, lab in enumerate(first, start=1):           # IF(NELION.LT.114) GO TO (...), NELION
        if k < 114:
            cases[k] = run_from(labels[lab])
    for k, lab in enumerate(second, start=169):         # IF(NELION.GT.168.AND.NELION.LT.337) GO TO (...)
        if k < 337:
            cases[k] = run_from(labels[lab])
    for v in cases.values():
        parse(v)
    return cases                                        # anything else: PFGROUND=1.


def pfground(nelion: int, t):
    """FUNCTION PFGROUND(NELION,T), elementwise over a tensor (or float) of temperatures."""
    expr = pfground_cases().get(int(nelion), "1.")
    v = evaluate(expr, {"HCK": HCK, "T": t})
    if isinstance(t, torch.Tensor):
        if not isinstance(v, torch.Tensor):
            v = torch.full_like(t, float(v))
        return v.to(DT)
    return float(v)


# ------------------------------------------------------------------------------------------------ layers

# departure-coefficient arrays read by PFSAHA's closed-form partition functions: name -> State key
B_KEYS = {"bhyd": "bhyd.bhyd", "bhe1": "bhe.bhe1", "bhe2": "bhe.bhe2", "bc1": "bc.bc1", "bc2": "bc.bc2",
          "bmg1": "bmg.bmg1", "bmg2": "bmg.bmg2", "bal1": "bal.bal1", "bsi1": "bsi.bsi1", "bsi2": "bsi.bsi2",
          "bca1": "bca.bca1", "bca2": "bca.bca2", "bo1": "bo.bo1", "bna1": "bna.bna1", "bb1": "bb.bb1",
          "bk1": "bk.bk1"}


@dataclass
class Layers:
    """Per-layer COMMON data PFSAHA reads (/TEMP/, /STATE/ P,XNE, the departure coefficients), any 1-D batch
    of layers, plus NLTEON of /IF/."""
    t: torch.Tensor
    tk: torch.Tensor
    tkev: torch.Tensor
    hckt: torch.Tensor
    tlog: torch.Tensor
    p: torch.Tensor
    xne: torch.Tensor
    b: dict = field(default_factory=dict)        # name -> (n, levels)
    nlteon: int = 0

    @classmethod
    def from_state(cls, s, n: int | None = None) -> "Layers":
        n = s.iscalar("rhox.nrhox") if n is None else n
        g = lambda k: s[k][:n].to(DT)
        b = {nm: s[k][:n].to(DT) for nm, k in B_KEYS.items() if k in s}
        return cls(g("temp.t"), g("temp.tk"), g("temp.tkev"), g("temp.hckt"), g("temp.tlog"), g("state.p"),
                   g("state.xne"), b, int(s.iscalar("if.nlteon")))

    def index(self, idx) -> "Layers":
        return Layers(self.t[idx], self.tk[idx], self.tkev[idx], self.hckt[idx], self.tlog[idx], self.p[idx],
                      self.xne[idx], {k: v[idx] for k, v in self.b.items()}, self.nlteon)

    def with_(self, **kw) -> "Layers":
        return replace(self, **kw)

    def bcol(self, name: str, i: int) -> torch.Tensor:
        """B=Bxx(J,i) (1-based level i); IF(NLTEON.EQ.-1)B=1."""
        if self.nlteon == -1 or name not in self.b:
            return torch.ones_like(self.t)
        return self.b[name][:, i - 1]


# ------------------------------------------------------------------------------------------- constants

_F13 = float(np.float32(1.0) / np.float32(3.0))       # 1./3.  folded in REAL*4
_F118 = float(np.float32(1.0) / np.float32(18.0))     # 1./18. folded in REAL*4
_P1 = f32(.1)


def _ex(c: float, hckt):
    """EXP(-c*HCKT(J)) for a literal c (already of the right kind)."""
    return libm.exp(-c * hckt)


def _blk(c, d):
    """SQRT(c/D)**3*(1./3.+(1.-(.5+(1./18.+D/120.)*D)*D)*D) with c = 13.595D0*Z*Z/TV."""
    return libm.ipow_const(libm.sqrt(c / d), 3) * (_F13 + (1.0 - (0.5 + (_F118 + d / 120.0) * d) * d) * d)


def _gsum(part, L: Layers, bname: str, g, e, n: int):
    """DO I=2,n: B=Bxx(J,I) (1 if NLTEON=-1); PART=PART+G(I)*B*EXP(-E(I)*HCKT(J))."""
    h = L.hckt
    for i in range(2, n + 1):
        part = part + float(g[i - 1]) * L.bcol(bname, i) * libm.exp(-float(e[i - 1]) * h)
    return part


def _terms(part, pairs, hckt):
    """PART=PART+c1*EXP(-e1*HCKT)+c2*EXP(-e2*HCKT)+... (evaluated left to right)."""
    for c, e in pairs:
        part = part + c * libm.exp(-e * hckt)
    return part


def _special(n: int, L: Layers):
    """Labels 1100..1210. Returns (part, g_override or None, d1 or None); d1 None = GO TO 18 (no tail)."""
    tb = _tb()
    h = L.hckt
    if n == 1:                                                    # 1100 H I
        part = 2.0 * L.bcol("bhyd", 1)
        part = _gsum(part, L, "bhyd", tb["ghyd"], tb["ehyd"], 6)
        return part, None, 109677.576 / 6.5 / 6.5 * h
    if n == 3:                                                    # 1110 He I
        part = _gsum(L.bcol("bhe1", 1), L, "bhe1", tb["ghe1"], tb["ehe1"], 29)
        return part, None, 109677.576 / 5.5 / 5.5 * h
    if n == 4:                                                    # 1120 He II
        part = _gsum(2.0 * L.bcol("bhe2", 1), L, "bhe2", tb["ghe2"], tb["ehe2"], 6)
        return part, None, 4.0 * 109722.267 / 6.5 / 6.5 * h
    if n == 354:                                                  # 1130 C I (16.42, 43.42 are REAL*4)
        part = L.bcol("bc1", 1) * (1.0 + 3.0 * _ex(f32(16.42), h) + 5.0 * _ex(f32(43.42), h))
        part = _gsum(part, L, "bc1", tb["gc1"], tb["ec1"], 14)
        part = _terms(part, ((108.0, 80000.0), (189.0, 84000.0), (247.0, 87000.0), (231.0, 88000.0),
                             (190.0, 89000.0), (300.0, 90000.0)), h)
        return part, None, None
    if n == 355:                                                  # 1132 C II
        part = L.bcol("bc2", 1) * (2.0 + 4.0 * _ex(63.42, h))
        part = _gsum(part, L, "bc2", tb["gc2"], tb["ec2"], 6)
        part = _terms(part, ((6.0, 131731.80), (4.0, 142027.1), (10.0, 145550.13), (10.0, 150463.62),
                             (2.0, 157234.07), (6.0, 162500.0), (42.0, 168000.0), (56.0, 178000.0),
                             (102.0, 183000.0), (400.0, 188000.0)), h)
        return part, None, None
    if n == 51:                                                   # 1140 Mg I
        part = _gsum(L.bcol("bmg1", 1), L, "bmg1", tb["gmg1"], tb["emg1"], 11)
        part = _terms(part, ((5.0, 53134.0), (15.0, 54192.0), (28.0, 54676.0), (9.0, 57853.0)), h)
        return part, 4.0, 109734.83 / 4.5 / 4.5 * h
    if n == 52:                                                   # 1142 Mg II
        part = _gsum(L.bcol("bmg2", 1) * 2.0, L, "bmg2", tb["gmg2"], tb["emg2"], 6)
        part = _terms(part, ((10.0, 93310.80), (14.0, 93799.70), (6.0, 97464.32), (10.0, 103419.82),
                             (14.0, 103689.89), (18.0, 103705.66)), h)
        return part, 2.0, 4.0 * 109734.83 / 5.5 / 5.5 * h
    if n == 57:                                                   # 1150 Al I
        part = L.bcol("bal1", 1) * (2.0 + 4.0 * _ex(112.061, h))
        part = _gsum(part, L, "bal1", tb["gal1"], tb["eal1"], 9)
        part = _terms(part, ((10.0, 42235.0), (14.0, 43831.0)), h)
        return part, 2.0, 109735.08 / 5.5 / 5.5 * h
    if n == 63:                                                   # 1160 Si I
        part = L.bcol("bsi1", 1) * (1.0 + 3.0 * _ex(77.115, h) + 5.0 * _ex(223.157, h))
        part = _gsum(part, L, "bsi1", tb["gsi1"], tb["esi1"], 11)
        part = _terms(part, ((76.0, 53000.0), (71.0, 57000.0), (191.0, 60000.0), (240.0, 62000.0),
                             (251.0, 63000.0), (300.0, 65000.0)), h)
        return part, None, None
    if n == 64:                                                   # 1162 Si II
        part = L.bcol("bsi2", 1) * (2.0 + 4.0 * _ex(287.32, h))
        part = _gsum(part, L, "bsi2", tb["gsi2"], tb["esi2"], 6)
        part = _terms(part, ((6.0, 81231.59), (6.0, 83937.08), (10.0, 101024.09), (14.0, 103556.35),
                             (10.0, 108800.0), (42.0, 115000.0), (6.0, 121000.0), (38.0, 125000.0),
                             (34.0, 132000.0)), h)
        return part, 2.0, 4.0 * 109734.83 / 4.5 / 4.5 * h
    if n == 96:                                                   # 1170 Ca I
        part = _gsum(L.bcol("bca1", 1), L, "bca1", tb["gca1"], tb["eca1"], 8)
        part = _terms(part, ((28.0, 37000.0), (67.0, 40000.0), (21.0, 43000.0), (34.0, 48000.0)), h)
        return part, 4.0, 109734.82 / 4.5 / 4.5 * h
    if n == 97:                                                   # 1172 Ca II
        part = _gsum(L.bcol("bca2", 1) * 2.0, L, "bca2", tb["gca2"], tb["eca2"], 5)
        part = _terms(part, ((12.0, 68000.0),), h)
        return part, 2.0, 109734.83 / 4.5 / 4.5 * h
    if n == 367:                                                  # 1180 O I
        part = L.bcol("bo1", 1) * (5.0 + 3.0 * _ex(158.265, h) + _ex(226.977, h))
        part = _gsum(part, L, "bo1", tb["go1"], tb["eo1"], 13)
        part = _terms(part, ((15.0, 101140.0), (131.0, 103000.0), (128.0, 105000.0), (600.0, 107000.0)), h)
        return part, None, None
    if n == 45:                                                   # 1190 Na I
        part = _gsum(L.bcol("bna1", 1) * 2.0, L, "bna1", tb["gna1"], tb["ena1"], 8)
        part = _terms(part, ((10.0, 34548.745), (14.0, 34586.96)), h)
        return part, 2.0, 109734.83 / 4.5 / 4.5 * h
    if n == 14:                                                   # 1200 B I
        part = L.bcol("bb1", 1) * (2.0 + 4.0 * _ex(15.25, h))
        part = _gsum(part, L, "bb1", tb["gb1"], tb["eb1"], 7)
        part = _terms(part, ((6.0, 57786.80), (10.0, 59989.0), (14.0, 60031.03), (2.0, 63561.0)), h)
        return part, 2.0, 109734.83 / 4.5 / 4.5 * h
    if n == 91:                                                   # 1210 K I
        part = _gsum(L.bcol("bk1", 1) * 2.0, L, "bk1", tb["gk1"], tb["ek1"], 8)
        part = _terms(part, ((10.0, 27397.077), (14.0, 28127.85)), h)
        return part, 2.0, 109734.83 / 5.5 / 5.5 * h
    return None


_SPECIAL = (1, 3, 4, 354, 51, 57, 63, 355, 52, 64, 96, 97, 367, 45, 14, 91)
DEFAULT, SPEC_TAIL, SPEC, IRON = 0, 1, 2, 3


def nions_of(iz: int) -> int:
    """NIONS of PFSAHA for element IZ."""
    locz = _tb()["locz"]
    nions = int(locz[iz] - locz[iz - 1]) if iz <= 28 else 3
    if iz == 6:
        nions = 6
    if iz == 7:
        nions = 7
    if iz == 8:
        nions = 8
    if 20 <= iz < 29:
        nions = 10
    return nions


def _item_static(iz: int, ion: int):
    """Layer-independent data of PFSAHA's loop 18 for (IZ, ION): (kind, ip, z, g, n)."""
    tb = _tb()
    pot = potion()
    index = iz * (iz + 1) // 2 + ion - 1 if iz <= 30 else iz * 5 + 341 + ion - 1
    ipv = float(pot[index - 1]) / 8065.479
    if ipv == 0:
        ipv = float(pot[index - 2]) / 8065.479
    if 20 <= iz < 29:
        return IRON, ipv, float(ion), 0.0, None
    locz = tb["locz"]
    n = int(locz[iz - 1]) if iz <= 28 else 3 * iz + 54
    if iz == 6:
        n = 354
    if iz == 7:
        n = 360
    if iz == 8:
        n = 367
    n = n - 1 + ion
    v = int(tb["nnn"][5, n - 1])
    g = float(v - (v // 100) * 100)
    return (SPEC if n in _SPECIAL else DEFAULT), ipv, float(ion), g, n


def _item_stage1(L: Layers, iz: int, ion: int) -> dict:
    """T-only part of PART(ION) for (IZ, ION) on all layers of L (see module docstring)."""
    kind, ipv, z, g, n = _item_static(iz, ion)
    tv = L.tkev
    d = {"kind": kind, "ip": ipv, "z": z, "g": g}
    if kind == IRON:
        d["a"] = pfiron_stage1(iz, ion, L.tlog / 2.30258509299405, "atlas9")
        return d
    if kind == SPEC:
        part, gset, d1 = _special(n, L)
        d["part0"] = part
        if d1 is not None:
            gg = gset if gset is not None else g
            d["kind"] = SPEC_TAIL
            d["tvx"] = tv
            d["c"] = libm.rdiv(13.595 * z * z, tv)
            d["blk1"] = _blk(d["c"], d1)
            d["ge"] = gg * libm.exp(libm.rdiv(-ipv, tv))
        return d
    # generic: interpolate the NNN table in T/T2000
    tb = _tb()
    dev = L.t.device
    t2000 = ipv * 2000.0 / 11.0
    x = L.t / t2000
    it = torch.clamp(torch.trunc(x - 0.5).to(torch.int64), 1, 9)
    dt = x - it.to(DT) - 0.5
    i = (it + 1) // 2
    row = torch.from_numpy(tb["nnn"][:, n - 1].astype(np.int64)).to(dev)
    scale = torch.from_numpy(tb["scale"].astype(np.float64)).to(dev)
    ni = row[i - 1]
    k1 = ni // 100000
    k2 = ni - k1 * 100000
    k3 = k2 // 10
    ksc = k2 - k3 * 10
    odd = (it % 2) == 1
    p1_odd = k1.to(DT) * scale[ksc - 1]
    p2_odd = k3.to(DT) * scale[ksc - 1]
    kp1 = torch.trunc(p1_odd)
    pmin_set = odd & ~(dt >= 0) & ~(ksc > 1) & (kp1 == torch.trunc(p2_odd + 0.5))
    pmin = torch.where(pmin_set, kp1, torch.ones_like(dt))
    ni1 = row[torch.clamp(i, max=5)]                       # NNN(I+1,N): only used for even IT (I<=4)
    k1e = ni1 // 100000
    ksce = torch.remainder(ni1, 10)
    p1_even = k3.to(DT) * scale[ksc - 1]
    p2_even = k1e.to(DT) * scale[torch.clamp(ksce, 1, 4) - 1]
    p1 = torch.where(odd, p1_odd, p1_even)
    p2 = torch.where(odd, p2_odd, p2_even)
    part = torch.maximum(pmin, p1 + (p2 - p1) * dt)
    nelion = (iz - 1) * 6 + ion
    if iz == 8 and ion == 1:
        nelion = 431
    if iz == 9 and ion == 1:
        nelion = 491
    if iz == 9 and ion == 2:
        nelion = 501
    d["part0"] = part
    d["cold"] = L.t < t2000 * 2.0
    d["part_cold"] = torch.maximum(pfground(nelion, L.t), part)
    d["tcool"] = L.t < t2000 * 4.0
    tv2 = torch.where(L.t > t2000 * 11.0, torch.full_like(tv, (t2000 * 11.0) * 8.6171e-5), tv)
    d["tvx"] = tv2
    d["c"] = libm.rdiv(13.595 * z * z, tv2)
    d["blk1"] = _blk(d["c"], libm.rdiv(_P1, tv2))
    d["ge"] = g * libm.exp(libm.rdiv(-ipv, tv2))
    return d


def potlow_of(L: Layers) -> torch.Tensor:
    """Debye lowering POTLOW (eV for unit Z) of PFSAHA, from XNE(J), P(J), TK(J)."""
    charge = L.xne * 2.0
    excess = 2.0 * L.xne - L.p / L.tk
    charge = torch.where(excess > 0, charge + 2.0 * excess, charge)
    charge = torch.where(charge == 0, torch.ones_like(charge), charge)
    debye = libm.sqrt(L.tk / 2.8965e-18 / charge)
    return torch.clamp(libm.rdiv(1.44e-7, debye), max=1.0)


class SahaPlan:
    """PFSAHA for several (IZ, NION2) rows at once. Stage 1 (T only) is evaluated once for all layers of L;
    stage 2 (XNE dependent) + the Saha chain are vectorised over rows x ions for any subset of the layers."""

    def __init__(self, L: Layers, rows: list):
        self.rows = [(int(iz), int(n2)) for iz, n2 in rows]
        self.L = L
        n = L.t.shape[0]
        R = len(self.rows)
        M = max(n2 for _, n2 in self.rows)
        self.M = M
        dev = L.t.device
        f64 = dict(dtype=DT, device=dev)
        z = torch.ones((R, M), **f64)
        ip = torch.zeros((R, M), **f64)
        g0 = torch.zeros((R, M), dtype=torch.bool, device=dev)
        kind = torch.full((R, M), SPEC, dtype=torch.int64, device=dev)
        part0 = torch.ones((n, R, M), **f64)
        part_cold = torch.ones((n, R, M), **f64)
        cold = torch.zeros((n, R, M), dtype=torch.bool, device=dev)
        tcool = torch.zeros((n, R, M), dtype=torch.bool, device=dev)
        tvx = torch.ones((n, R, M), **f64)
        c = torch.ones((n, R, M), **f64)
        blk1 = torch.zeros((n, R, M), **f64)
        ge = torch.zeros((n, R, M), **f64)
        a = torch.ones((n, R, M, 7), dtype=torch.float32, device=dev)
        valid = torch.zeros((R, M), dtype=torch.bool, device=dev)
        cache = {}
        for r, (iz, nion2) in enumerate(self.rows):
            for ion in range(1, nion2 + 1):
                if (iz, ion) not in cache:
                    cache[(iz, ion)] = _item_stage1(L, iz, ion)
                d = cache[(iz, ion)]
                m = ion - 1
                valid[r, m] = True
                kind[r, m] = d["kind"]
                z[r, m] = d["z"]
                ip[r, m] = d["ip"]
                g0[r, m] = d["g"] == 0
                if d["kind"] == IRON:
                    a[:, r, m] = d["a"]
                    continue
                part0[:, r, m] = d["part0"]
                if d["kind"] in (DEFAULT, SPEC_TAIL):
                    tvx[:, r, m], c[:, r, m], blk1[:, r, m], ge[:, r, m] = d["tvx"], d["c"], d["blk1"], d["ge"]
                if d["kind"] == DEFAULT:
                    part_cold[:, r, m], cold[:, r, m], tcool[:, r, m] = d["part_cold"], d["cold"], d["tcool"]
        self.z, self.ip, self.g0, self.kind, self.valid = z, ip, g0, kind, valid
        self.part0, self.part_cold, self.cold, self.tcool = part0, part_cold, cold, tcool
        self.tvx, self.c, self.blk1, self.ge, self.a = tvx, c, blk1, ge, a
        self.nion2 = torch.tensor([n2 for _, n2 in self.rows], device=dev)
        self.has_iron = bool((kind == IRON).any())

    def parts(self, idx, Lsub: Layers):
        """PART(ION) and POTLO(ION) (n, R, M) for layers `idx` of the plan, with XNE etc. of Lsub."""
        potlow = potlow_of(Lsub)
        potlo = potlow[:, None, None] * self.z
        tvx = self.tvx[idx]
        d2 = potlo / tvx
        tail = self.part0[idx] + self.ge[idx] * (_blk(self.c[idx], d2) - self.blk1[idx])
        skip = self.g0 | (potlo < _P1) | self.tcool[idx]
        dflt = torch.where(self.cold[idx], self.part_cold[idx], torch.where(skip, self.part0[idx], tail))
        part = torch.where(self.kind == DEFAULT, dflt, torch.where(self.kind == SPEC_TAIL, tail, self.part0[idx]))
        if self.has_iron:
            pf = pfiron_stage2(self.a[idx], potlo * 8065.479, "atlas9")
            part = torch.where(self.kind == IRON, pf, part)
        part = torch.where(self.valid, part, torch.ones_like(part))
        return part, potlo

    def fractions(self, idx, Lsub: Layers):
        """Ionisation fractions F(ION) (n, R, M) of each row (labels 19-22), plus part, potlo."""
        part, potlo = self.parts(idx, Lsub)
        tv = Lsub.tkev[:, None, None]
        cf = (2.0 * 2.4148e15 * Lsub.t * libm.sqrt(Lsub.t) / Lsub.xne)[:, None, None]
        # F(ION)=CF*PART(ION)/PART(ION-1)*EXP(-(IP(ION-1)-POTLO(ION-1))/TV), ION>=2
        ex = libm.exp(-(self.ip[:, :-1] - potlo[..., :-1]) / tv)
        ratio = cf * part[..., 1:] / part[..., :-1] * ex
        M = self.M
        f1 = torch.ones_like(part[..., 0])
        for l in range(M, 1, -1):                       # F(1)=1.+F(L)*F(1), L=NION2..2
            use = self.nion2 >= l
            f1 = torch.where(use, 1.0 + ratio[..., l - 2] * f1, f1)
        f = [libm.rdiv(1.0, f1)]
        for k in range(2, M + 1):
            f.append(f[-1] * ratio[..., k - 2])
        return torch.stack(f, -1), part, potlo


def pfsaha(L: Layers, iz: int, nion: int, mode: int) -> torch.Tensor:
    """SUBROUTINE PFSAHA for all layers of L. Returns what the Fortran stores into ANSWER(J,*):
    MODE 1/2/3 + 10: (n, NION) (ANSWER(J,1..NION)); MODE 1/2/3: (n,) (ANSWER(J,1), ion NION only);
    MODE 4 / 14: (n,) electrons per atom (ANSWER(J,1))."""
    mode1 = mode - 10 if mode > 10 else mode
    nion2 = min(nion + 2, nions_of(iz))
    if nion > nion2 and mode1 != 4:
        raise NotImplementedError(f"PFSAHA IZ={iz} NION={nion} > NION2={nion2}: Fortran returns stale SAVEd values")
    plan = SahaPlan(L, [(iz, nion2)])
    sl = slice(None)
    if mode1 == 3:
        part = plan.parts(sl, L)[0][:, 0]
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
