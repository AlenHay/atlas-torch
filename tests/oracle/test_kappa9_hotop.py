"""KAPPA9 variant (kappa9.for, ifx oracle): HOTOP (Si II and S II in the Z=1 free-free sum).
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
def test_hotop_kappa9_vs_fortran(golden, seed, lf):
    T.oracle_opacity_check(golden, "hotop", "kappa9", seed, lf, rtol=T.KAPPA9_RTOL)


def test_hotop_variant_switch_matters(golden):
    """The kappa9 free-free sum includes XNFSI(J,2)+XNFS(J,2): the atlas9 formula must not match it."""
    base = T.opacity_state(golden, "hotop", 0)
    s = T.perturbed_state(base, 7, 15.0)
    a9 = T.port_opacity(s, "hotop", "atlas9")["ahot"]
    k9 = T.port_opacity(s, "hotop", "kappa9")["ahot"]
    exp = T.oracle_opacity(T.make_oracle("kappa9"), s, "hotop")["ahot"]
    assert T.max_rel(k9, exp) <= T.KAPPA9_RTOL
    assert T.max_rel(a9, exp) > 1e-6
