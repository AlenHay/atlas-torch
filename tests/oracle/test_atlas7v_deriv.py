"""DERIV (atlas7v.for). DERIV is only reached on JOSH's optically thick branch (TAUNU(1) > 20), which the
traced 500-510 nm run never takes, so there is no golden trace: the golden-style check is the thick-branch
JOSH test in test_atlas7v_josh.py (Ca II K core inputs vs the Fortran). Here: property vs the Fortran."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.josh import deriv
from atlas_torch.testing.compare import assert_close


def _fortran(o, x, f):
    d = np.zeros(len(x), order="F")
    o.call("deriv", np.asfortranarray(x), np.asfortranarray(f), d, len(x))
    return d


@settings(max_examples=100, deadline=None)
@given(n=st.integers(2, 99), seed=st.integers(0, 2**31 - 1), decreasing=st.booleans(),
       zeros=st.booleans())
def test_deriv_vs_fortran(oracle, n, seed, decreasing, zeros):
    rng = np.random.default_rng(seed)
    o = oracle("atlas7v")
    xs, fs = [], []
    for _ in range(4):
        x = 20.0 + np.cumsum(10.0 ** rng.uniform(-3, 1, n))
        if decreasing:
            x = x[::-1].copy()
        f = rng.normal(size=n) * 10.0 ** rng.uniform(-12, 0, n)
        if zeros:
            f[rng.random(n) < 0.4] = 0.0             # SCALE = 0 -> 1 branch
        xs.append(x); fs.append(f)
    got = deriv(torch.as_tensor(np.stack(xs)), torch.as_tensor(np.stack(fs)))
    for i in range(4):
        assert_close(got[i], _fortran(o, xs[i], fs[i]), rtol=0, atol=0, name=f"deriv n={n}")
