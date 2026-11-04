"""Oracle tests for HMINOP (atlas7v.for, SYNTHE variant): golden trace entry -> exit, and the port vs the
Fortran routine on perturbed solar states (T, XNE, RHO, populations, departure coefficients) at
FREQ = 10**U(12.5, 16.5) Hz."""
import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.hydrogen import hminop

KEYS = ['bhyd.ahmin', 'bhyd.shmin', 'save.hminop.itemp1', 'save.hminop.xhmin', 'save.hminop.theta']
CACHE = {'xhmin': ((99,), 1e9), 'theta': ((99,), 1.0)}


def test_hminop_golden(golden):
    T.golden_check(golden, "hminop", hminop, KEYS)


@settings(max_examples=80, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1), cached=st.booleans())
def test_hminop_vs_fortran(oracle, golden, lf, seed, cached):
    """cached=False: ITEMP1=0, the ITEMP block recomputes; cached=True: ITEMP1=ITEMP with arbitrary SAVEd
    arrays, which must be used as they are."""
    s = T.random_case(golden, seed, 10.0**lf)
    if cached:
        s = T.stale_saves(s, "hminop", CACHE, np.random.default_rng(seed + 1))
    o = T.run_oracle(oracle, s, "hminop", saves=["hminop"] if cached else [])
    T.check(hminop(s), T.oracle_ref(o, "hminop"), KEYS, name=f"hminop@{10**lf:.6e} cached={cached} ")
