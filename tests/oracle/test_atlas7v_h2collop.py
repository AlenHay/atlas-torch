"""Oracle tests for H2COLLOP (atlas7v.for): H2-H2 / H2-He collision-induced absorption. Golden entry -> exit
and port vs Fortran on perturbed states, both on the ITEMP refresh path and with a stale XNH2 cache.
Also checks the side effect WAVENO = FREQ/2.99792458D10 on COMMON /FREQ/."""
import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.cool import h2collop

KEYS = ["arg:ah2coll", "freq.waveno", "save.h2collop.itemp1", "save.h2collop.xnh2"]


def test_h2collop_golden(golden):
    T.golden_check(golden, "h2collop", h2collop, KEYS)


@settings(max_examples=80, deadline=None)
@given(lf=st.floats(12.5, 14.5), seed=st.integers(0, 2**32 - 1), cached=st.booleans(),
       tscale=st.sampled_from([(0.8, 1.25), (0.3, 0.6)]))
def test_h2collop_vs_fortran(oracle, golden, lf, seed, cached, tscale):
    """FREQ up to 10**14.5 (the CIA is zero above WAVENO=20000); tscale (0.3,0.6) reaches T < 2000 K."""
    s = T.random_case(golden, seed, 10.0**lf, tscale=tscale)
    if cached:
        s = T.stale_saves(s, "h2collop", {"xnh2": ((99,), 1e11)}, np.random.default_rng(seed + 1))
    ah2coll = np.zeros(99, order="F")
    o = T.run_oracle(oracle, s, "h2collop", saves=["h2collop"] if cached else [], args=(ah2coll,))
    ref = T.oracle_ref(o, "h2collop")
    T.check(h2collop(s), lambda k: ah2coll if k == "arg:ah2coll" else ref(k), KEYS, name=f"h2collop@{10**lf:.4e} ")


def test_h2collop_high_frequency_zero(oracle, golden):
    s = T.random_case(golden, 1, 1e15)
    out = h2collop(s)
    assert float(out["arg:ah2coll"].abs().max()) == 0.0
