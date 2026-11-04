"""Oracle tests for NELECT (molecules off, IFMOL=0): electron-density fixed point over 10 donors, layer
recurrence, IFEDNS=1 internal energy with the in-place T perturbations. No golden trace (BasicATLAS runs
with molecules on): property tests vs the Fortran on perturbed solar layers, bitwise."""
import numpy as np
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics.pops import nelect


@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ft=st.floats(0.8, 1.3), fp=st.floats(0.3, 3.0), ifedns=st.sampled_from([0, 1]), last=st.booleans())
def test_nelect_vs_fortran(golden, ft, fp, ifedns, last):
    # IFEDNS=1 starts every layer from the current XNE: take a State with converged XNE (the second
    # traced NMOLEC call); the first call's XNE is still undefined (NaN) and NELECT would CALL EXIT.
    s = T.nmolec_state(golden, ifedns)
    T_ = s["temp.t"] * ft
    TK = 1.38054e-16 * T_
    s.update({"temp.t": T_, "temp.tk": TK, "temp.tkev": 8.6171e-5 * T_, "temp.hckt": 6.6256e-27 / TK * 2.99792458e10,
              "temp.hkt": 6.6256e-27 / TK, "temp.tlog": torch.log(T_), "state.p": s["state.p"] * fp,
              "if.ifmol": torch.tensor([0]), "edens.ifedns": torch.tensor([ifedns])})
    if last:
        s["iter.iter"] = s["iter.numits"].clone()
    out = nelect(s)
    o = T.make_oracle("atlas9")
    T.load_state(o, s)
    o.call("nelect")
    n = s.iscalar("rhox.nrhox")
    for k, v in out.items():
        b, m = k.split(".")
        assert T.max_rel(v[:n].numpy(), o.common(b)[m][:n]) == 0.0, k
