"""Oracle tests for H2RAOP (atlas7v.for, SYNTHE variant): golden trace entry -> exit, and the port vs the
Fortran routine on perturbed solar states (T, XNE, RHO, populations, departure coefficients) at
FREQ = 10**U(12.5, 16.5) Hz."""
import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.hydrogen import h2raop

KEYS = ['bhyd.sigh2', 'save.h2raop.itemp1', 'save.h2raop.xnh2']
CACHE = {'xnh2': ((99,), 1e18)}


def test_h2raop_golden(golden):
    T.golden_check(golden, "h2raop", h2raop, KEYS)


@settings(max_examples=60, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1), cached=st.booleans())
def test_h2raop_vs_fortran(oracle, golden, lf, seed, cached):
    """cached=False: ITEMP1=0, the ITEMP block recomputes; cached=True: ITEMP1=ITEMP with arbitrary SAVEd
    arrays, which must be used as they are."""
    s = T.random_case(golden, seed, 10.0**lf)
    if cached:
        s = T.stale_saves(s, "h2raop", CACHE, np.random.default_rng(seed + 1))
    o = T.run_oracle(oracle, s, "h2raop", saves=["h2raop"] if cached else [])
    T.check(h2raop(s), T.oracle_ref(o, "h2raop"), KEYS, name=f"h2raop@{10**lf:.6e} cached={cached} ")
