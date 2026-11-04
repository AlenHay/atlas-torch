"""KAPPA9 variant (kappa9.for, ifx oracle): HE2OP (3.6919E8, 2.815E29, 1.31522E14 REAL*4).
The ifx build uses Intel libimf exp/log/pow: agreement to a few ulps (T.KAPPA9_RTOL), not bitwise.
"""
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T


@pytest.fixture(scope="module", autouse=True)
def _intel():
    if not T.preload_intel():
        pytest.skip("Intel runtime (libifport etc.) not found; set INTEL_LIB")


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 2**31 - 1), lf=st.floats(12.5, 16.5))
def test_he2op_kappa9_vs_fortran(golden, seed, lf):
    T.oracle_opacity_check(golden, "he2op", "kappa9", seed, lf, rtol=T.KAPPA9_RTOL)
