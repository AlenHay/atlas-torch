"""Oracle tests for SUBROUTINE POPS(CODE,MODE,NUMBER) (atlas7v.for): number densities / partition-function
ratios of atoms, ions and molecules, with the ITEMP caching of the equilibrium solve.
  golden:   the three traced POPS hits of the solar xnfpelsyn run (hit 0 runs NMOLEC for all layers; hits 1,2
            reuse the stored solution): NUMBER, the COMMONs NMOLEC updates and the SAVEd ITEMP1, bitwise.
  property: the complete xnfpelsyn POPS call sequence (Fortran order and arguments, incl. the REAL*4 molecule
            codes IDMOL that never match CODEMOL exactly) on perturbed / cool / hot models, every NUMBER
            compared with the Fortran POPS; IFMOL=0 (NELECT path) sequences; random caller NUMBER contents
            (POPS rescales columns PFSAHA did not write in the MODE=2/3 path); ITEMP changes re-running the
            solve. All bitwise.
"""
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.state import State
from atlas_torch.synthe.pops import pops
from atlas_torch.tables import table
from atlas_torch.testing.compare import assert_close
from test_atlas7v_pfsaha import SETTINGS, STAGE, load_oracle, model, note, solar_state


def xnfpelsyn_calls():
    """(CODE, MODE) of every CALL POPS in xnfpelsyn.for, in program order."""
    first = [(1.00, 12), (2.01, 12), (1.01, 11), (2.02, 11), (5.00, 11), (6.01, 11), (8.00, 11), (11.00, 11),
             (12.01, 11), (13.01, 11), (14.01, 11), (19.00, 11), (20.01, 11), (26.00, 11), (106.00, 1),
             (108.00, 1)]
    second = [(1.01, 11), (2.02, 11), (3.03, 11), (4.03, 11), (5.03, 11)] + \
             [(float(z) + 0.05, 11) for z in range(6, 17)] + [(17.04, 11), (18.04, 11), (19.04, 11)] + \
             [(float(z) + 0.09, 11) for z in range(20, 29)] + \
             [(float(np.float32(z)) + 0.02, 11) for z in range(29, 100)]
    idmol = np.asarray(table("xnfpelsyn.for", "xnfpelsyn", "idmol"), dtype=np.float64)
    third = [(float(c), 1) for c in idmol]
    return first + second + third


def _merge_trace(golden, hit):
    e = golden(STAGE, "pops", hit, "entry")
    s = solar_state(golden).updated(State.from_trace(e))
    s["save.pops.itemp1"] = torch.tensor([int(e["static:itemp1"][0])])
    return s, e


def test_pops_golden(golden):
    for hit in range(3):
        s, e = _merge_trace(golden, hit)
        x = golden(STAGE, "pops", hit, "exit")
        number = torch.from_numpy(e["arg:number"].reshape(99, 10, order="F").copy())
        got, upd = pops(s, float(e["arg:code"][0]), int(e["arg:mode"][0]), number)
        exp = x["arg:number"].reshape(99, 10, order="F")
        note("pops", got, exp)
        assert_close(got, exp, rtol=0, name=f"pops hit {hit} NUMBER")
        assert int(upd["save.pops.itemp1"][0]) == int(x["static:itemp1"][0])
        s2 = s.updated(upd)
        for k in ("state.xne", "state.xnatom", "state.rho", "xnmol.xnmol", "xnmol.xnfpmol", "if.nlteon"):
            assert_close(s2[k], x[k], rtol=0, name=f"pops hit {hit} {k}")
        assert (hit == 0) == ("xnmol.xnmol" in upd)          # only the first call solves the equilibrium


def run_sequence(o, s, calls, init=None):
    """Run CALL POPS for each (code, mode) on the oracle and the port; compare every NUMBER."""
    load_oracle(o, s)
    for i, (code, mode) in enumerate(calls):
        num0 = np.zeros((99, 10)) if init is None else init(i)
        fnum = np.asfortranarray(num0.copy())
        o.call("pops", float(code), int(mode), fnum)
        got, upd = pops(s, code, mode, torch.from_numpy(num0.copy()))
        s = s.updated(upd)
        note("pops", got, fnum)
        assert_close(got, fnum, rtol=0, name=f"call {i} POPS({code!r},{mode})")
    n = s.iscalar("rhox.nrhox")
    st_ = o.common("state")
    for k in ("xne", "xnatom", "rho"):
        assert_close(s["state." + k][:n], st_[k][:n], rtol=0, name=k)
    return s


@pytest.mark.stage   # ~5 s; fast tier: test_pops_golden, test_pops_solar_sequence_and_itemp
@settings(max_examples=5, **SETTINGS)
@given(kind=st.sampled_from(["perturbed", "cool", "hot"]), seed=st.integers(0, 2**31 - 1))
def test_pops_xnfpelsyn_sequence(oracle, golden, kind, seed):
    s = model(golden, kind, seed)
    run_sequence(oracle("atlas7v"), s, xnfpelsyn_calls())


def test_pops_solar_sequence_and_itemp(oracle, golden):
    """Solar model, full sequence; then ITEMP=2 forces a new solve on a perturbed T (cache semantics)."""
    o = oracle("atlas7v")
    s = run_sequence(o, model(golden, "solar", 0), xnfpelsyn_calls())
    t = s["temp.t"].clone()
    n = s.iscalar("rhox.nrhox")
    t[:n] = t[:n] * 1.05
    from test_atlas7v_pfsaha import with_temperature
    s = with_temperature(s, t[:n])
    s["temp.itemp"] = torch.tensor([2])
    load_oracle(o, s, ("temp",))
    calls = [(26.09, 11), (1.0, 12), (106.0, 1), (606.0, 2), (608.0, 3)]
    for code, mode in calls:
        fnum = np.zeros((99, 10), order="F")
        o.call("pops", code, mode, fnum)
        got, upd = pops(s, code, mode)
        s = s.updated(upd)
        assert_close(got, fnum, rtol=0, name=f"itemp=2 POPS({code},{mode})")
    assert int(s["save.pops.itemp1"][0]) == 2


@pytest.mark.stage   # ~3.5 s; fast tier: test_pops_golden, test_pops_solar_sequence_and_itemp
@settings(max_examples=6, **SETTINGS)
@given(kind=st.sampled_from(["solar", "perturbed", "cool", "hot"]), seed=st.integers(0, 2**31 - 1),
       ifmol=st.sampled_from([0, 1]))
def test_pops_modes_and_caller_number(oracle, golden, kind, seed, ifmol):
    """Random codes / modes (1,2,3,4,11,12,13,14) with random caller NUMBER contents; IFMOL 0 and 1."""
    s = model(golden, kind, seed)
    s["if.ifmol"][0] = ifmol
    rng = np.random.default_rng(seed)
    codes = [1.0, 1.01, 2.02, 6.03, 8.0, 12.01, 26.04, 26.09, 30.02, 56.01, 92.02, 3.03, 14.05]
    if ifmol == 1:
        codes += [100.0, 101.0, 608.0, 10108.0, 106.01, 999.0]
    calls = [(c, int(rng.choice([1, 2, 3, 4, 11, 12, 13, 14]))) for c in codes]
    if ifmol == 0:
        from atlas_torch.synthe.freeform import FortranExit
        from atlas_torch.synthe.pops import nelect
        try:
            nelect(s)
        except FortranExit:          # the Fortran would CALL EXIT
            return
    run_sequence(oracle("atlas7v"), s, calls, init=lambda i: rng.uniform(-1.0, 1.0, (99, 10)))
