"""EXPI (atlas9mem.for) vs the Fortran function, and the traced (N, X) -> EX1 golden values."""
import ctypes

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9 import numerics as N

STAGE = "sun_atlas9/atlas9"


def _arrays(seed, n, scale):
    rng = np.random.default_rng(seed)
    x = np.asfortranarray(np.sort(rng.uniform(1e-6, 10, n)) * scale)
    f = np.asfortranarray(rng.normal(size=n) * np.exp(rng.normal(size=n)))
    return x, f


@settings(max_examples=300, deadline=None)
@given(x=st.one_of(st.floats(1e-12, 0.02), st.floats(0.02, 30.0)), n=st.integers(1, 4))
def test_expi_vs_fortran(oracle, x, n):
    o = oracle("atlas9")
    r, _ = o.call("expi", n, float(x), restype=ctypes.c_double)
    assert float(N.expi(n, torch.tensor([x]))[0]) == r


def test_expi_golden(golden):
    for hit in range(3):
        e = golden(STAGE, "expi", hit, "exit")
        assert float(N.expi(1, torch.tensor([float(e["arg:x"][0])]))[0]) == float(e["static:ex1"][0])
