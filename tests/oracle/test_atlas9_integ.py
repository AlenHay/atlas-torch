"""INTEG and PARCOE (atlas9mem.for) vs the Fortran (bit-for-bit).
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
@given(n=st.integers(4, 99), seed=st.integers(0, 2**31 - 1), scale=st.sampled_from([1e-7, 1e-3, 1.0, 1e3]))
def test_integ_parcoe_vs_fortran(oracle, n, seed, scale):
    o = oracle("atlas9")
    x, f = _arrays(seed, n, scale)
    if np.any(np.diff(x) == 0):
        return
    start = float(np.random.default_rng(seed).normal())
    fi = np.zeros(n, order="F"); o.call("integ", x, f, fi, n, start)
    assert np.array_equal(N.integ(torch.from_numpy(x), torch.from_numpy(f), start).numpy(), fi)
    a, b, c = (np.zeros(n, order="F") for _ in range(3)); o.call("parcoe", f, x, a, b, c, n)
    pa, pb, pc = N.parcoe(torch.from_numpy(f), torch.from_numpy(x))
    assert np.array_equal(pa.numpy(), a) and np.array_equal(pb.numpy(), b) and np.array_equal(pc.numpy(), c)
    # batched rows with per-row START
    fb = torch.from_numpy(np.stack([f, 2 * f]))
    r = N.integ(torch.from_numpy(x), fb, torch.tensor([start, 0.0]))
    assert np.array_equal(r[0].numpy(), fi)
