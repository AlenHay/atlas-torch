"""TURB (atlas9mem.for): turbulent velocity law VTURB=(TRBFDG*RHO**TRBPOW+TRBSND*VELSND/1E5+TRBCON)*1E5."""
from __future__ import annotations

from ..state import State
from . import fmath


def turb(s: State) -> dict:
    n = s.iscalar("rhox.nrhox")
    rho = s["state.rho"][:n]
    v = (s.scalar("turbpr.trbfdg") * fmath.pow(rho, s.scalar("turbpr.trbpow"))
         + s.scalar("turbpr.trbsnd") * s["conv.velsnd"][:n] / 1.0e5 + s.scalar("turbpr.trbcon")) * 1.0e5
    vt = s["turbpr.vturb"].clone(); vt[:n] = v
    pt = s["turbpr.pturb"].clone(); pt[:n] = rho * (v * v) * 0.5
    return {"turbpr.vturb": vt, "turbpr.pturb": pt}
