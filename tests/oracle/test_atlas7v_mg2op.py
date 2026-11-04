"""Oracle tests for FUNCTION MG2OP(J) (atlas7v.for; called by LUKEOP for J=1..NRHOX): golden SAVE state and
the port vs the Fortran function on perturbed states, on the ITEMP refresh path and with a stale cache."""
import ctypes

import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.luke import mg2op
from atlas_torch.testing.compare import assert_close

CACHE = {'c1169': ((99,), 0.01)}


def _fortran(o, n):
    return np.array([o.call("mg2op", j, restype=ctypes.c_double)[0] for j in range(1, n + 1)])


def test_mg2op_golden(golden, oracle):
    for hit in range(3):
        s, e, _ = T.golden_state(golden, "mg2op", hit)
        val, out = mg2op(s)
        T.check(out, e, list(out), name=f"mg2op[{hit}] ")
        o = T.run_oracle(oracle, s, saves=["mg2op"])
        assert_close(val, _fortran(o, s.iscalar("rhox.nrhox")), rtol=T.RTOL, name=f"mg2op[{hit}]")


@settings(max_examples=60, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1), cached=st.booleans())
def test_mg2op_vs_fortran(oracle, golden, lf, seed, cached):
    s = T.random_case(golden, seed, 10.0**lf)
    if cached:
        s = T.stale_saves(s, "mg2op", CACHE, np.random.default_rng(seed + 1))
    o = T.run_oracle(oracle, s, saves=["mg2op"] if cached else [])
    ref = _fortran(o, s.iscalar("rhox.nrhox"))
    val, out = mg2op(s)
    assert_close(val, ref, rtol=T.RTOL, name=f"mg2op@{10**lf:.6e} cached={cached}")
    T.check(out, T.oracle_ref(o, "mg2op"), list(out), name=f"mg2op@{10**lf:.6e} ")
