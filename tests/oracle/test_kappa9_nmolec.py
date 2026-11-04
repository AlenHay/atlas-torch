"""KAPPA9 NMOLEC: no NCHEMITER/CHEMFAIL cap (the BasicATLAS patch is ATLAS9-only); ifx/libimf oracle.
libimf ulps propagate through the Newton solve (and may flip the 1E-4 convergence test), hence 1e-6."""
import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics.molec import nmolec


@pytest.fixture(scope="module", autouse=True)
def _intel():
    if not T.preload_intel():
        pytest.skip("Intel runtime not found; set INTEL_LIB")


@pytest.mark.stage   # ~3 s (4 whole-model NMOLEC solves); fast tier: test_chemfail_cap_is_atlas9_only, atlas9/atlas7v NMOLEC tests
@settings(max_examples=4, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(hit=st.sampled_from([0, 1]), ft=st.floats(0.9, 1.1), fp=st.floats(0.5, 2.0))
def test_nmolec_kappa9_vs_fortran(golden, hit, ft, fp):
    s = T.nmolec_state(golden, hit, "kappa9")
    T_ = s["temp.t"] * ft
    TK = 1.38054e-16 * T_
    s.update({"temp.t": T_, "temp.tk": TK, "temp.tkev": 8.6171e-5 * T_, "temp.hckt": 6.6256e-27 / TK * 2.99792458e10,
              "temp.hkt": 6.6256e-27 / TK, "temp.tlog": torch.log(T_), "state.p": s["state.p"] * fp})
    out = nmolec(s, 1, "kappa9")
    exp = T.oracle_nmolec(T.make_oracle("kappa9"), s, 1)
    T.compare_nmolec(out, exp, s.iscalar("rhox.nrhox"), 1e-6, "nmolec[kappa9]: ")


def test_chemfail_cap_is_atlas9_only(golden):
    """With an iteration cap of 1 the atlas9 path stops (CHEMFAIL); kappa9 has no cap."""
    s = T.nmolec_state(golden, 0)
    a = nmolec(s, 1, "atlas9", max_iter=1)
    assert a["_nmolec.chemfail"]
    b = nmolec(s, 1, "kappa9")
    assert not b["_nmolec.chemfail"]
