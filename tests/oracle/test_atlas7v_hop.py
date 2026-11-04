"""Oracle tests for HOP (atlas7v.for, SYNTHE variant): golden trace entry -> exit, and the port vs the
Fortran routine on perturbed solar states (T, XNE, RHO, populations, departure coefficients) at
FREQ = 10**U(12.5, 16.5) Hz."""
import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.hydrogen import hop

KEYS = ['bhyd.ahyd', 'bhyd.shyd']


def test_hop_golden(golden):
    T.golden_check(golden, "hop", hop, KEYS)


@settings(max_examples=80, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1))
def test_hop_vs_fortran(oracle, golden, lf, seed):
    s = T.random_case(golden, seed, 10.0**lf)
    o = T.run_oracle(oracle, s, "hop")
    T.check(hop(s), T.oracle_ref(o, "hop"), KEYS, name=f"hop@{10**lf:.6e} ")
