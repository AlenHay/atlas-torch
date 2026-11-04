"""Oracle tests for H2PLOP (atlas7v.for, SYNTHE variant): golden trace entry -> exit, and the port vs the
Fortran routine on perturbed solar states (T, XNE, RHO, populations, departure coefficients) at
FREQ = 10**U(12.5, 16.5) Hz."""
import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.hydrogen import h2plop

KEYS = ['bhyd.ah2p']


def test_h2plop_golden(golden):
    T.golden_check(golden, "h2plop", h2plop, KEYS)


@settings(max_examples=60, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1))
def test_h2plop_vs_fortran(oracle, golden, lf, seed):
    s = T.random_case(golden, seed, 10.0**lf)
    o = T.run_oracle(oracle, s, "h2plop")
    T.check(h2plop(s), T.oracle_ref(o, "h2plop"), KEYS, name=f"h2plop@{10**lf:.6e} ")
