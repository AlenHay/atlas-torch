"""KAPPA9 variant of PFSAHA: same code, but /POTION/ from kappa9.for's IONPOTS DATA (older ionisation
potentials) and the ifx oracle (libimf exp: a few ulps)."""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics._data import potion

from test_atlas9_pfsaha import ELEMENTS, check_vs_fortran


@pytest.fixture(scope="module", autouse=True)
def _intel():
    if not T.preload_intel():
        pytest.skip("Intel runtime not found; set INTEL_LIB")


def test_kappa9_potion_differs():
    """IONPOTS differs between the variants (e.g. POTH 109678.772D0 vs 109678.764)."""
    a, k = potion("atlas9"), potion("kappa9")
    assert a[0] == 109678.772 and k[0] == np.float32(109678.764)
    assert np.array_equal(k, T.make_oracle("kappa9").common("potion")["potion"])


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 2**31 - 1), iz=st.sampled_from(ELEMENTS), nf=st.floats(0, 0.999),
       mode=st.sampled_from([1, 2, 3, 4, 5, 11, 12, 13]))
def test_pfsaha_kappa9_vs_fortran(seed, iz, nf, mode):
    check_vs_fortran("kappa9", seed, iz, nf, mode, rtol=T.KAPPA9_RTOL)
