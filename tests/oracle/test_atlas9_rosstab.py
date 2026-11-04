"""ROSSTAB (atlas9mem.for): golden traced calls + random (T, P, VTURB) vs the Fortran function."""
import ctypes
from pathlib import Path

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9.commons import initial_state
from atlas_torch.atlas9.rosstab import LN10_F32, Rosstab, read_kappa_ros
from atlas_torch.state import State

STAGE = "sun_atlas9/atlas9"


def _state(ext):
    _, tab = read_kappa_ros(Path(ext) / "BasicATLAS/data/solar_ODF/kappa.ros")
    s = initial_state()
    s["tabtp.tabkap"] = tab
    s["tabtp.iftabk"] = torch.tensor([1])
    return s


def test_rosstab_golden(golden):
    for hit in range(3):
        e, x = golden(STAGE, "rosstab", hit, "entry"), golden(STAGE, "rosstab", hit, "exit")
        s = initial_state()
        s.update({k: v for k, v in State.from_trace(e).items() if k.startswith("tabtp.")})
        r = Rosstab(s)(float(e["arg:t"][0]), float(e["arg:p"][0]), float(e["arg:v"][0]))
        assert float(r) == float(np.exp(x["static:ablog"][0] * LN10_F32))


@settings(max_examples=200, deadline=None)
@given(lt=st.floats(3.25, 5.4), lp=st.floats(-6.0, 9.0), v=st.sampled_from([0.0, 1e5, 1.5e5, 2e5, 3e5, 8e5, 9e5]))
def test_rosstab_vs_fortran(oracle, ext, lt, lp, v):
    o = oracle("atlas9")
    s = _state(ext)
    o.set_common("tabtp", tabkap=s["tabtp.tabkap"].numpy(), iftabk=1)
    t, p = 10.0 ** lt, 10.0 ** lp
    r, _ = o.call("rosstab", t, p, v, restype=ctypes.c_double)
    assert float(Rosstab(s)(t, p, v)) == r


def test_rosstab_builtin_table(oracle):
    """Without READ KAPPA the KTAB DATA tables are used (IFTABK=0 path)."""
    o = oracle("atlas9")
    s = initial_state()
    R = Rosstab(s)
    for t, p in ((4000.0, 1e3), (6000.0, 1e5), (12000.0, 10.0)):
        r, _ = o.call("rosstab", t, p, 2e5, restype=ctypes.c_double)
        assert float(R(t, p, 2e5)) == r
