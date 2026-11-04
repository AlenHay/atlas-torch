"""LINTER (HMINOP's linear interpolation) and MAP1 (atlas_torch.interp, reused: the atlas9mem copy differs
from atlas7v only by MIN0 vs AMIN0) vs the atlas9 library."""
import ctypes

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics.opacity import linter
from atlas_torch.interp import map1


@settings(max_examples=150, deadline=None)
@given(n=st.integers(3, 30), seed=st.integers(0, 2**31 - 1), x=st.floats(-2.0, 12.0))
def test_linter_map1_vs_fortran(n, seed, x):
    rng = np.random.default_rng(seed)
    xo = np.asfortranarray(np.sort(rng.uniform(0, 10, n)))
    if np.any(np.diff(xo) == 0):
        return
    yo = np.asfortranarray(rng.normal(size=n))
    o = T.make_oracle("atlas9")
    xn, yn = np.array([x], order="F"), np.zeros(1, order="F")
    o.call("linter", xo, yo, n, xn, yn, 1)
    assert linter(torch.from_numpy(xo), torch.from_numpy(yo), torch.from_numpy(xn)).item() == yn[0]
    fn = np.zeros(1, order="F")
    o.call("map1", xo, yo, n, xn, fn, 1, restype=ctypes.c_int)
    assert map1(torch.from_numpy(xo), torch.from_numpy(yo), torch.from_numpy(xn))[0].item() == fn[0]
