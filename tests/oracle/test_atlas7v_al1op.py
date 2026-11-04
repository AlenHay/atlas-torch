"""Oracle tests for AL1OP (atlas7v.for, SYNTHE variant): golden trace entry -> exit, and the port vs the
Fortran routine on perturbed solar states (T, XNE, RHO, populations, departure coefficients) at
FREQ = 10**U(12.5, 16.5) Hz."""
import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.cool import al1op

KEYS = ['bal.aal1', 'bal.sal1']


def test_al1op_golden(golden):
    T.golden_check(golden, "al1op", al1op, KEYS)


@settings(max_examples=80, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1))
def test_al1op_vs_fortran(oracle, golden, lf, seed):
    s = T.random_case(golden, seed, 10.0**lf)
    o = T.run_oracle(oracle, s, "al1op")
    T.check(al1op(s), T.oracle_ref(o, "al1op"), KEYS, name=f"al1op@{10**lf:.6e} ")
