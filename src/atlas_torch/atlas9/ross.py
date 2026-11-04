"""ROSS (atlas9mem.for): Rosseland mean opacity and optical depth.

MODE 1 zeroes ABROSS; MODE 2 contributions DBDT/ABTOT*RCOWT are accumulated by the caller in Fortran order;
MODE 3: ABROSS=(4*sigma/pi)*T**3/ABROSS (3.14159 is a REAL*4 literal; T**3=(T*T)*T), TAUROS=INTEG.
"""
from __future__ import annotations

import torch

from ..fnum import f32
from ..state import State
from .numerics import integ

C_ROSS = 4.0 * 5.6697e-5 / f32(3.14159)


def dbdt(bnu, freq, hkt, t, stim):
    """DBDT = BNU*FREQ*HKT/T/STIM ([..., n]; freq broadcast over layers)."""
    return bnu * freq.unsqueeze(-1) * hkt / t / stim


def ross_terms(bnu, freq, hkt, t, stim, abtot, rcowt, numnu: int):
    if numnu == 1:
        d = C_ROSS * (t * t * t)
        d = d.expand_as(abtot)
    else:
        d = dbdt(bnu, freq, hkt, t, stim)
    return d / abtot * rcowt.unsqueeze(-1)


def ross3(s: State) -> dict:
    n = s.iscalar("rhox.nrhox")
    t = s["temp.t"][:n]
    ab = s["abross.abross"].clone()
    ab[:n] = C_ROSS * (t * t * t) / ab[:n]
    rx = s["rhox.rhox"][:n]
    tr = s["abross.tauros"].clone()
    tr[:n] = integ(rx, ab[:n], ab[0] * rx[0])
    return {"abross.abross": ab, "abross.tauros": tr}
