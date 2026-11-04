"""Oracle tests for SUBROUTINE NMOLEC (atlas7v.for): molecular + ionisation equilibrium (Newton iteration per
layer, layer recurrence), XNMOL / XNFPMOL / XNE / XNATOM / RHO.
  golden:   the traced NMOLEC call of the solar xnfpelsyn run (entry State -> port -> exit COMMONs), bitwise
  property: perturbed models (T*U(0.8,1.25), P*10^U(-0.5,0.5) per layer, XNE*10^U(-0.5,0.5) -- the input XNE
            enters the NLTE corrections CPF), random departure coefficients, a cool model (~3500 K at the top,
            molecules dominant) and a hot one (~20000 K at the bottom, T>10000 K molecules switched off),
            port vs the Fortran NMOLEC. The port follows the Fortran iteration exactly (same start, same
            update/damping/step-limiter, same convergence test), so agreement is bitwise.
"""
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.state import State
from atlas_torch.synthe.pops import nmolec
from atlas_torch.testing.compare import assert_close
from test_atlas7v_pfsaha import SETTINGS, STAGE, load_oracle, model, note

OUT = ("xnmol.xnmol", "xnmol.xnfpmol", "state.xne", "state.xnatom", "state.rho")


def oracle_nmolec(o, s):
    load_oracle(o, s)
    o.call("nmolec")
    return {"xnmol.xnmol": o.common("xnmol")["xnmol"].copy(), "xnmol.xnfpmol": o.common("xnmol")["xnfpmol"].copy(),
            "state.xne": o.common("state")["xne"].copy(), "state.xnatom": o.common("state")["xnatom"].copy(),
            "state.rho": o.common("state")["rho"].copy(), "if.nlteon": o.common("if")["nlteon"].copy()}


def compare(out, exp, n, name, rtol=0.0):
    for k in OUT:
        g, e = out[k][:n], np.asarray(exp[k])[:n]
        note("nmolec", g, e)
        assert_close(g, e, rtol=rtol, name=f"{name} {k}")
    assert int(out["if.nlteon"].reshape(-1)[0]) == int(np.asarray(exp["if.nlteon"]).reshape(-1)[0])


def test_nmolec_golden(golden):
    s = State.from_trace(golden(STAGE, "nmolec", 0, "entry"))
    x = golden(STAGE, "nmolec", 0, "exit")
    out = nmolec(s)
    compare(out, x, s.iscalar("rhox.nrhox"), "golden")
    assert torch.equal(out["xnmol.xnmol"], torch.from_numpy(x["xnmol.xnmol"]))   # incl. untouched columns


@pytest.mark.stage   # ~4 s; fast tier: test_nmolec_golden, test_nmolec_cool_and_hot
@settings(max_examples=6, **SETTINGS)
@given(kind=st.sampled_from(["perturbed", "cool", "hot"]), seed=st.integers(0, 2**31 - 1), nlte=st.booleans())
def test_nmolec_vs_fortran(oracle, golden, kind, seed, nlte):
    s = model(golden, kind, seed, nlte=nlte)
    exp = oracle_nmolec(oracle("atlas7v"), s)
    out = nmolec(s)
    compare(out, exp, s.iscalar("rhox.nrhox"), f"{kind} seed={seed} nlte={nlte}")


@pytest.mark.stage   # ~2 s; fast tier: test_nmolec_golden
def test_nmolec_cool_and_hot(oracle, golden):
    """Deterministic cool / hot / solar-NLTE cases (always run, independent of hypothesis' draws)."""
    for kind, seed, nlte in (("cool", 1, False), ("hot", 2, False), ("solar", 3, True)):
        s = model(golden, kind, seed, nlte=nlte)
        exp = oracle_nmolec(oracle("atlas7v"), s)
        out = nmolec(s)
        compare(out, exp, s.iscalar("rhox.nrhox"), f"{kind}")
        if kind == "cool":   # molecules matter: most carbon is in CO at the top
            code = s["xnmol.code"].numpy()
            jco, jc = int(np.nonzero(code == 608.0)[0][0]), int(np.nonzero(code == 6.0)[0][0])
            assert float(out["xnmol.xnmol"][0, jco] / out["xnmol.xnmol"][0, jc]) > 1.0
