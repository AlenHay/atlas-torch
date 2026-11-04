"""Oracle tests for COOLOP (atlas7v.for): C1OP+MG1OP+AL1OP+SI1OP+FE1OP + (CH+OH)*STIM/RHO + H2 CIA.
Checks every COMMON/SAVE entry COOLOP and its callees write, including FE1OP's FLXRAD (EQUIVALENCE) and
H2COLLOP's WAVENO, on the golden trace and against the Fortran on perturbed states."""
import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.cool import coolop

KEYS = ["bc.ac1", "bc.sc1", "bmg.amg1", "bmg.smg1", "bal.aal1", "bal.sal1", "bsi.asi1", "bsi.ssi1", "bfe.afe1",
        "bfe.sfe1", "ops.acool", "flux.flxrad", "freq.waveno", "save.coolop.ah2coll", "save.h2collop.itemp1",
        "save.h2collop.xnh2", "save.chop.freq1", "save.ohop.freq1"]


def test_coolop_golden(golden):
    T.golden_check(golden, "coolop", coolop, KEYS)


@settings(max_examples=80, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1),
       tscale=st.sampled_from([(0.8, 1.25), (0.3, 0.6), (1.2, 1.6)]))
def test_coolop_vs_fortran(oracle, golden, lf, seed, tscale):
    s = T.random_case(golden, seed, 10.0**lf, tscale=tscale)
    o = T.run_oracle(oracle, s, "coolop")
    T.check(coolop(s), T.oracle_ref(o, "coolop"), KEYS, name=f"coolop@{10**lf:.6e} ")
