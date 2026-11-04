"""Oracle tests for FUNCTION SEATON(FREQ0,XSECT,POWER,A) (atlas7v.for): Seaton photoionisation fit at the
COMMON FREQ. Golden: the traced arguments; property: random edges, powers and frequencies."""
import ctypes

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.state import State
from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity._core import Ctx
from atlas_torch.synthe.opacity.luke import seaton
from atlas_torch.testing.compare import assert_close


def _port(s, freq0, xsect, power, a):
    return float(seaton(Ctx.from_state(s), freq0, xsect, power, a)[0, 0])


def test_seaton_golden(golden, oracle):
    for hit in range(3):
        d = golden(T.STAGE, "seaton", hit, "entry")
        args = [float(d[f"arg:{k}"][0]) for k in ("freq0", "xsect", "power", "a")]
        s = T.base_state(golden).updated({"freq.freq": torch.from_numpy(d["freq.freq"].astype(np.float64))})
        o = T.run_oracle(oracle, s)
        ref, _ = o.call("seaton", *args, restype=ctypes.c_double)
        assert_close(np.array([_port(s, *args)]), np.array([ref]), rtol=T.RTOL, name=f"seaton[{hit}]")


@settings(max_examples=100, deadline=None)
@given(lf=st.floats(14.5, 16.5), lf0=st.floats(14.5, 16.0), power=st.sampled_from([1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
       a=st.floats(0.1, 7.0), lx=st.floats(-20.0, -16.0))
def test_seaton_vs_fortran(oracle, golden, lf, lf0, power, a, lx):
    s = T.base_state(golden).updated({"freq.freq": torch.tensor([10.0**lf])})
    o = T.run_oracle(oracle, s)
    args = (10.0**lf0, 10.0**lx, power, a)
    ref, _ = o.call("seaton", *args, restype=ctypes.c_double)
    assert_close(np.array([_port(s, *args)]), np.array([ref]), rtol=T.RTOL, name="seaton")
