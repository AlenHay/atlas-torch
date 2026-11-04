"""Oracle tests for HEMIOP (atlas7v.for, SYNTHE variant): golden trace entry -> exit, and the port vs the
Fortran routine on perturbed solar states (T, XNE, RHO, populations, departure coefficients) at
FREQ = 10**U(12.5, 16.5) Hz."""
import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.helium import hemiop

KEYS = ['bhe.ahemin']


def test_hemiop_golden(golden):
    T.golden_check(golden, "hemiop", hemiop, KEYS)


@settings(max_examples=40, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1))
def test_hemiop_vs_fortran(oracle, golden, lf, seed):
    s = T.random_case(golden, seed, 10.0**lf)
    o = T.run_oracle(oracle, s, "hemiop")
    T.check(hemiop(s), T.oracle_ref(o, "hemiop"), KEYS, name=f"hemiop@{10**lf:.6e} ")
