"""ATLAS9 / KAPPA9 physics library: populations and continuous opacities (atlas9mem.for, kappa9.for).

Entry points (pure functions of a State; they return the entries they write):

  populations(s, variant)   the main program's per-iteration population block: (IFPRES=1) IFEDNS=0 and
                            POPS(0.D0,1,XNE) [NELECT or NMOLEC], then POPS(1.01D0,11,XNFPH) ... POPS(26.00D0,11,
                            XNFPFE), the IFMOL=0/1 branches (XNF*, XNFPCH/OH) and the PFSAHA products (loop 444).
                            Updates /STATE/, /XNMOL/, /XNSAVE/, /IONS/, /XNFP/, /XNF/, save.pops.itemp1.
  electrons(s, variant)     POPS(0.D0,1,XNE) alone (also what CONVEC calls with IFEDNS=1).
  continuum(s, freq, variant)  KAPP(N=0): /FREQ/ for `freq` (scalar) must be in the State or is derived
                            (derive_freq=True); zero /OPS/, call every IFOP-enabled continuum source and combine
                            into /OPTOT/ ACONT, SCONT, SIGMAC (ALINE/SLINE/SIGMAL from the zeroed line arrays).
  pops(s, code, mode, variant)  SUBROUTINE POPS for one species code.

`variant` is 'atlas9' (atlas9mem.patched.for, BasicATLAS build) or 'kappa9' (kappa9.for); the routines
that differ between them (results/fmap/variants.md: HRAYOP, HE2OP, C1OP, HOTOP, NMOLEC's CHEMFAIL cap,
IONPOTS data) switch on it.

`Atlas9Physics(variant)` implements the iteration machinery's `Physics` protocol (atlas9/physics.py,
which this package shadows by name; the protocol is re-exported here unchanged).
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

from .api import Atlas9Physics, continuum, electrons, populations, pops  # noqa: F401


def _load_protocol():
    p = _Path(__file__).resolve().parent.parent / "physics.py"
    if not p.exists():
        return None
    spec = _ilu.spec_from_file_location("atlas_torch.atlas9._physics_protocol", p)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_proto = _load_protocol()
if _proto is not None:
    Physics = _proto.Physics
