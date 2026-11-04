"""HIGH (atlas9mem.for): geometric height scale in km, HEIGHT = INTEG(RHOX, 1.E-5/RHO) (1.E-5 is REAL*4)."""
from __future__ import annotations

from ..fnum import f32
from ..state import State
from . import fmath
from .numerics import integ


def high(s: State) -> dict:
    n = s.iscalar("rhox.nrhox")
    rhoinv = fmath.rdiv(f32(1.0e-5), s["state.rho"][:n])
    h = s["height.height"].clone()
    h[:n] = integ(s["rhox.rhox"][:n], rhoinv, 0.0)
    return {"height.height": h}
