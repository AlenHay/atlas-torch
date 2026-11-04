"""Oracle tests for COOLOP: C I, Mg I, Al I, Si I, Fe I, CH, OH, H2 CIA (atlas9mem.patched.for, ATLAS9 variant).
golden: solar ATLAS9 trace entry -> port -> exit (bitwise under the trace's glibc, else 1e-13);
property: perturbed layers (T 3000..50000 K, XNE, RHO) and FREQ = 10**U(12.5,16.5) vs the Fortran (bitwise).
"""
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T


def test_coolop_golden(golden):
    T.golden_opacity_check(golden, "coolop")


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 2**31 - 1), lf=st.floats(12.5, 16.5))
def test_coolop_vs_fortran(golden, seed, lf):
    T.oracle_opacity_check(golden, "coolop", "atlas9", seed, lf, rtol=0.0)
