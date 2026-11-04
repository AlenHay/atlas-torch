"""Oracle tests for FUNCTION O1OP(J) (atlas7v.for): O I (Peterson after Peach), J-independent. No golden trace
exists for O1OP (it is exercised in the LUKEOP golden test); port vs the Fortran function."""
import ctypes

import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.luke import o1op
from atlas_torch.testing.compare import assert_close


@settings(max_examples=60, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1))
def test_o1op_vs_fortran(oracle, golden, lf, seed):
    s = T.random_case(golden, seed, 10.0**lf)
    o = T.run_oracle(oracle, s)
    n = s.iscalar("rhox.nrhox")
    ref = np.array([o.call("o1op", j, restype=ctypes.c_double)[0] for j in range(1, n + 1)])
    val, _ = o1op(s)
    assert_close(val, ref, rtol=T.RTOL, name=f"o1op@{10**lf:.6e}")
