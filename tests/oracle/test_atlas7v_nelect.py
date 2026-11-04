"""Oracle tests for SUBROUTINE NELECT (atlas7v.for): electron density by fixed-point iteration (the
IFMOL=0 path of POPS), layer recurrence with the Fortran's masking of minor donors.
  golden:   NELECT is not called in the traced solar xnfpelsyn run (MOLECULES ON), so there is no traced
            hit; the solar model itself is checked against the Fortran instead.
  property: perturbed (T*U(0.8,1.25), P*10^U(-0.5,0.5)), cool (~3500 K) and hot (~20000 K) models, NLTEON
            0/1 with random departure coefficients, port vs the Fortran NELECT, bitwise. Cases where the
            Fortran would CALL EXIT ("XNE DOES NOT CONVERGE") are detected on the port first (an EXIT would
            kill the test process).
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.freeform import FortranExit
from atlas_torch.synthe.pops import nelect
from atlas_torch.testing.compare import assert_close
from test_atlas7v_pfsaha import SETTINGS, load_oracle, model, note

OUT = ("state.xne", "state.xnatom", "state.rho")


def _check(oracle, s, name):
    try:
        out = nelect(s)
    except FortranExit:
        return False
    o = oracle("atlas7v")
    load_oracle(o, s)
    o.call("nelect")
    n = s.iscalar("rhox.nrhox")
    st_ = o.common("state")
    for k in OUT:
        e = st_[k.split(".")[1]][:n]
        note("nelect", out[k][:n], e)
        assert_close(out[k][:n], e, rtol=0, name=f"{name} {k}")
    return True


def test_nelect_solar(oracle, golden):
    for nlteon in (0, 1):
        s = model(golden, "solar", 0)
        s["if.ifmol"][0] = 0
        s["if.nlteon"][0] = nlteon
        assert _check(oracle, s, f"solar nlteon={nlteon}")


@pytest.mark.stage   # ~2.5 s; fast tier: test_nelect_solar
@settings(max_examples=10, **SETTINGS)
@given(kind=st.sampled_from(["perturbed", "cool", "hot"]), seed=st.integers(0, 2**31 - 1),
       nlteon=st.sampled_from([0, 1]))
def test_nelect_vs_fortran(oracle, golden, kind, seed, nlteon):
    s = model(golden, kind, seed, nlte=nlteon == 1)
    s["if.ifmol"][0] = 0
    s["if.nlteon"][0] = nlteon
    _check(oracle, s, f"{kind} seed={seed} nlteon={nlteon}")
