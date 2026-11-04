"""TTAUP (atlas9mem.for): integrate hydrostatic equilibrium dP/dlog tau = g tau / kappa_Ross on a log-spaced
tau grid with a 4th-order predictor-corrector, iterating each depth to |dlog P| < 5e-5.

The recurrence is sequential in depth (each point needs the previous four), so the depth loop stays a Python
loop; the per-depth iteration is scalar. Batching over models is the natural extension (masks per model).

Quirks reproduced:
  * ABSTD(1)=.1 is a default-REAL literal (0.10000000149011612) unless PRAD(1)>0 (then DMIN1(.1D0, ...));
  * the convergence test ERROR.GT..00005 compares with the single-precision literal;
  * P(J).LE.0 -> P(J)=P(J-1);  N>1000 -> 'HYDROFAIL' and CALL EXIT (raised as FortranExit).
"""
from __future__ import annotations

import math

import torch

from ..fnum import f32
from .errors import FortranExit

_TOL = f32(0.00005)
_P1 = f32(0.1)


def ttaup(t, tau, abstd, prad, pturb, vturb, grav: float, numtau: int, iross: int, rosstab):
    """Returns (ABSTD, PTOTAL, P) for J=1..NUMTAU (tensors of length NUMTAU).

    `abstd` is only read at J=1 when iross==1. `rosstab(T,P,V)` is the ROSSTAB callable."""
    T = [float(v) for v in t[:numtau]]
    TAU = [float(v) for v in tau[:numtau]]
    PRAD = [float(v) for v in prad[:numtau]]
    PTURB = [float(v) for v in pturb[:numtau]]
    VT = [float(v) for v in vturb[:numtau]]
    AB = [float(v) for v in abstd[:numtau]] if abstd is not None else [0.0] * numtau
    PT = [0.0] * numtau
    P = [0.0] * numtau
    dlgtau = math.log(TAU[1] / TAU[0])
    plog3 = plog2 = plog1 = dplog2 = dplog1 = 0.0
    plog4 = dplog3 = 0.0
    if iross != 1:
        AB[0] = _P1
        if PRAD[0] > 0.0:
            AB[0] = min(0.1, grav * TAU[0] / PRAD[0] / 2.0)
    for j in range(numtau):
        J = j + 1
        if J == 1:
            plog = math.log(grav / AB[0] * TAU[0])
        elif J <= 4:
            plog = plog1 + dplog1
        else:
            plog = (3.0 * plog4 + 8.0 * dplog1 - 4.0 * dplog2 + 8.0 * dplog3) / 3.0
        error = 1.0
        n = 1
        first = True
        while True:
            if not first:
                if J == 1:
                    pnew = math.log(grav / AB[0] * TAU[0])
                elif J <= 4:
                    pnew = (plog + 2.0 * plog1 + dplog + dplog1) / 3.0
                else:
                    pnew = (126.0 * plog1 - 14.0 * plog3 + 9.0 * plog4 + 42.0 * dplog + 108.0 * dplog1
                            - 54.0 * dplog2 + 24.0 * dplog3) / 121.0
                error = abs(pnew - plog)
                plog = (pnew + plog) / 2.0
            first = False
            PT[j] = math.exp(plog)
            P[j] = PT[j] + (PRAD[0] - PRAD[j]) - PTURB[j]
            if P[j] <= 0.0:
                P[j] = P[j - 1]
            AB[j] = float(rosstab(T[j], P[j], VT[j]))
            dplog = grav / AB[j] * TAU[j] / PT[j] * dlgtau
            n += 1
            if n > 1000:
                raise FortranExit("HYDROFAIL", 0)
            if not (error > _TOL):
                break
        plog4, plog3, plog2, plog1 = plog3, plog2, plog1, plog
        dplog3, dplog2, dplog1 = dplog2, dplog1, dplog
    f = lambda a: torch.tensor(a, dtype=torch.float64)
    return f(AB), f(PT), f(P)
