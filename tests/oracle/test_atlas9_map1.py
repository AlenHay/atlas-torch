"""MAP1 (atlas9mem.for) as batched (map1b) and shared-abscissa (map1_shared) vectorisations vs the Fortran.
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
@given(n=st.integers(4, 72), m=st.integers(1, 60), seed=st.integers(0, 2**31 - 1))
def test_map1_batched_vs_fortran(oracle, n, m, seed):
    rng = np.random.default_rng(seed)
    xold = np.asfortranarray(np.sort(rng.uniform(0, 10, n)))
    if np.any(np.diff(xold) == 0):
        return
    fold = np.asfortranarray(rng.normal(size=n))
    xnew = np.asfortranarray(np.sort(rng.uniform(-1, 11, m)))
    fnew = np.zeros(m, order="F")
    r, _ = oracle("atlas9").call("map1", xold, fold, n, xnew, fnew, m, restype=ctypes.c_int)
    got, ll = N.map1b(torch.from_numpy(xold).unsqueeze(0), torch.from_numpy(fold).unsqueeze(0),
                      torch.from_numpy(xnew))
    assert np.array_equal(got[0].numpy(), fnew) and int(ll[0]) == r
    # one point at a time on a shared abscissa (ROSSTAB's use)
    pts = N.map1_shared(torch.from_numpy(xold), torch.from_numpy(fold).unsqueeze(0), torch.from_numpy(xnew))
    for k in range(m):
        f1 = np.zeros(1, order="F")
        oracle("atlas9").call("map1", xold, fold, n, np.asfortranarray(xnew[k:k + 1]), f1, 1, restype=ctypes.c_int)
        assert float(pts[k, 0]) == f1[0]
