"""Oracle tests for FE1OP (atlas7v.for, SYNTHE variant): golden trace entry -> exit, and the port vs the
Fortran routine on perturbed solar states (T, XNE, RHO, populations, departure coefficients) at
FREQ = 10**U(12.5, 16.5) Hz."""
import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.cool import fe1op

KEYS = ['bfe.afe1', 'bfe.sfe1', 'flux.flxrad']


def test_fe1op_golden(golden):
    T.golden_check(golden, "fe1op", fe1op, KEYS)


@settings(max_examples=80, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1))
def test_fe1op_vs_fortran(oracle, golden, lf, seed):
    s = T.random_case(golden, seed, 10.0**lf)
    o = T.run_oracle(oracle, s, "fe1op")
    T.check(fe1op(s), T.oracle_ref(o, "fe1op"), KEYS, name=f"fe1op@{10**lf:.6e} ")
