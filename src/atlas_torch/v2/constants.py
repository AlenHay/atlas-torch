"""Physical constants (cgs) and the element data the EOS uses.

Two constant sets:
  CODATA2018 -- the v2 default (CODATA 2018 recommended values; h, k, e, c exact in SI).
  KURUCZ     -- the values hard-coded in atlas7v.for / atlas9mem.for, kept to separate the effect of the
                constants from algorithmic differences when comparing with the exact port.

Where they differ (Kurucz/CODATA - 1):
  k        1.38054e-16 erg/K       READIN: TK=1.38054D-16*T                    -7.9e-5
  h        6.6256e-27 erg s        READIN: HKT=6.6256D-27/TK                   -7.1e-5
  k/eV     8.6171e-5 eV/K          READIN: TKEV=8.6171D-5*T                    -2.7e-5
  eV/hc    8065.479 cm^-1/eV       PFSAHA: IP=POTION/8065.479D0                -8.1e-6
  amu      1.660e-24 g             READIN/NMOLEC: RHO=XNATOM*WTMOLE*1.660D-24  -3.3e-4
  Ry       13.595 eV               PFSAHA, hydrogenic level tail               -7.9e-4
  e^2      1.44e-7 eV cm           PFSAHA, Debye lowering                      +2.5e-5
  4 pi e^2 2.8965e-18 erg cm       PFSAHA, Debye length                        -9.2e-4
  (2 pi m_e k/h^2)^1.5   2.4148e15 PFSAHA, Saha                                +4.8e-5
  (2 pi amu k/h^2)^1.5   1.8786e20 NMOLEC, molecular n/U                       -3.8e-4
  hc/k     (from h, c, k above)    PFSAHA/PFGROUND: HCKT, HCK                  +8.0e-6
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from ..tables import table

N_ELEMENTS = 99
MAX_STAGES = 10


@dataclass(frozen=True)
class PhysicalConstants:
    """cgs constants; derived combinations are stored explicitly because Kurucz's are not mutually consistent."""
    name: str
    c: float            # cm s^-1
    h: float            # erg s
    k: float            # erg K^-1
    amu: float          # g
    k_ev: float         # eV K^-1
    ev_cm: float        # cm^-1 per eV (E/hc)
    hc_k: float         # cm K
    ry_ev: float        # Rydberg energy, eV
    e2_ev_cm: float     # e^2 in eV cm (Coulomb energy at 1 cm)
    four_pi_e2: float   # 4 pi e^2, erg cm
    saha_e: float       # (2 pi m_e k / h^2)^1.5, cm^-3 K^-1.5
    saha_amu: float     # (2 pi amu k / h^2)^1.5, cm^-3 (amu K)^-1.5


def _codata2018() -> PhysicalConstants:
    c, h, k = 2.99792458e10, 6.62607015e-27, 1.380649e-16
    ev = 1.602176634e-12
    e_esu = 4.803204712570263e-10
    m_e, amu = 9.1093837015e-28, 1.66053906660e-24
    return PhysicalConstants(
        name="CODATA2018", c=c, h=h, k=k, amu=amu, k_ev=k / ev, ev_cm=ev / (h * c), hc_k=h * c / k,
        ry_ev=13.605693122994, e2_ev_cm=e_esu ** 2 / ev, four_pi_e2=4.0 * math.pi * e_esu ** 2,
        saha_e=(2.0 * math.pi * m_e * k / h ** 2) ** 1.5, saha_amu=(2.0 * math.pi * amu * k / h ** 2) ** 1.5)


CODATA2018 = _codata2018()
KURUCZ = PhysicalConstants(
    name="KURUCZ", c=2.99792458e10, h=6.6256e-27, k=1.38054e-16, amu=1.660e-24, k_ev=8.6171e-5,
    ev_cm=8065.479, hc_k=6.6256e-27 * 2.99792458e10 / 1.38054e-16, ry_ev=13.595, e2_ev_cm=1.44e-7,
    four_pi_e2=2.8965e-18, saha_e=2.4148e15, saha_amu=1.8786e20)

# eV/K with which the molecules.dat equilibrium-constant fits (Kurucz 1970 eq. 4.35) are evaluated
# (NMOLEC: EXP(EQUIL(1)/TKEV(J)-...)); part of the fit's definition, not a physical constant choice.
FIT_K_EV = 8.6171e-5


def decimal32(a) -> np.ndarray:
    """REAL*4 DATA values as the decimal numbers written in the source (1.00800002 -> 1.008)."""
    a = np.asarray(a)
    return np.array([float(np.format_float_positional(np.float32(x), unique=True)) for x in a.ravel()]
                    ).reshape(a.shape)


@lru_cache(maxsize=None)
def _element_table() -> tuple[tuple[str, ...], np.ndarray]:
    sym = tuple(s.decode().strip().capitalize() for s in np.asarray(table("atlas7v.for", "blocke", "elem")))
    return sym, decimal32(table("atlas7v.for", "blocke", "atmass"))


def element_symbols() -> tuple[str, ...]:
    """Element symbols Z=1..99 (atlas7v.for BLOCKE, DATA ELEM)."""
    return _element_table()[0]


def atomic_masses(device=None) -> torch.Tensor:
    """Atomic masses Z=1..99 in amu (atlas7v.for BLOCKE, DATA ATMASS), shape (99,)."""
    return torch.as_tensor(_element_table()[1], dtype=torch.float64, device=device)


ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")


def species_name(z: int, stage: int) -> str:
    """'Fe II' for z=26, stage=2 (stage 1 = neutral)."""
    return f"{element_symbols()[z - 1]} {ROMAN[stage - 1]}"


def parse_species(name: str) -> tuple[int, int]:
    """'Fe II' -> (26, 2); Kurucz code '26.01' -> (26, 2)."""
    s = name.strip()
    try:
        code = float(s)
        return int(code), int(round((code - int(code)) * 100)) + 1
    except ValueError:
        pass
    sym, roman = s.split()
    return element_symbols().index(sym.capitalize()) + 1, ROMAN.index(roman.upper()) + 1
