"""DERIV (atlas9mem.for) vs the Fortran, single and batched rows (bit-for-bit).
(Golden traces record only the first element of array arguments of these leaf routines; their golden coverage
is through the callers' tests: JOSH, TCORR, ROSS, RADIAP, HIGH, READIN.)"""
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


@settings(max_examples=150, deadline=None)
@given(n=st.integers(2, 99), seed=st.integers(0, 2**31 - 1), scale=st.sampled_from([1e-7, 1e-3, 1.0, 1e3]))
def test_deriv_vs_fortran(oracle, n, seed, scale):
    x, f = _arrays(seed, n, scale)
    if np.any(np.diff(x) == 0):
        return
    d = np.zeros(n, order="F")
    oracle("atlas9").call("deriv", x, f, d, n)
    assert np.array_equal(N.deriv(torch.from_numpy(x), torch.from_numpy(f)).numpy(), d)
    xb = torch.from_numpy(np.stack([x, x * 1.5])); fb = torch.from_numpy(np.stack([f, -f]))
    assert np.array_equal(N.deriv(xb, fb)[0].numpy(), d)
