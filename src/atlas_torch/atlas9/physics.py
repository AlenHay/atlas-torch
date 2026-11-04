"""Interface between the ATLAS9 iteration machinery and the atlas9-variant physics (populations and
continuous opacities), which is ported separately.

The iteration calls exactly three things, matching the Fortran call sites:
  electrons(s)        POPS(0.D0,1,XNE): electron density / gas state for the current T, P (NELECT, or NMOLEC
                      when IFMOL=1), recomputed only when ITEMP changed. Returns updates of /STATE/ (XNE,
                      XNATOM, RHO), /EDENS/ (EDENS) and whatever else those routines write -- notably, with
                      IFEDNS=1 (inside CONVEC) NMOLEC/NELECT perturb T, TK, TKEV by *1.001 and undo it by
                      division for every molecule/ion, leaving ulp-level changes in /TEMP/ that the Fortran
                      then uses: those must be returned too.
  populations(s)      the main program's block after it: POPS(...,11|12,...) for H, He, C, ..., Fe (+ CH, OH)
                      and the PFSAHA products for the molecule case -> /IONS/, /XNF/, /XNFP/ updates.
  continuum(s, fq)    KAPP(0,...) up to (excluding) its combination loop: zero /OPS/ and call every enabled
                      continuum source (HOP ... XSOP per IFOP) for each frequency in `fq` (dict with 'freq' [F],
                      'freqlg' [F], 'ehvkt','stim','bnu' [F,n]). Returns {ops member: [F, n] tensor}.
A physics object may keep internal SAVEd state between calls (the Fortran does: NMOLEC's starting guesses,
ITEMP caches); calls arrive in the Fortran's order.
"""
from __future__ import annotations

from typing import Protocol

import torch

from ..state import State


class Physics(Protocol):
    def electrons(self, s: State) -> dict: ...

    def populations(self, s: State) -> dict: ...

    def continuum(self, s: State, fq: dict) -> dict[str, torch.Tensor]: ...
