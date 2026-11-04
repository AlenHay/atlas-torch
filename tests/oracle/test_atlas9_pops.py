"""Oracle tests for POPS / MOLEC and the ATLAS9 main program's population block (populations()).
golden: POPS hits (NMOLEC on the first call, MOLEC look-ups after), and the whole iteration-1 block against the
populations the opacity routines see (HOP/HOTOP/COOLOP/LUKEOP/KAPP entries); property: POPS(code, mode) vs the
Fortran for every species code the main program uses (+ ones that take MOLEC's PFSAHA path), after NMOLEC.
"""
import numpy as np
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics import populations
from atlas_torch.atlas9.physics.molec import nmolec
from atlas_torch.atlas9.physics.pops import pops


def _start_state(golden):
    s = T.nmolec_state(golden, 0)
    p, _ = T.golden_state(golden, T.TRACE_STAGE, "pops", 0, "entry")
    s = s.updated(p)
    ref = T.merged_golden(golden, ("hop", "hotop", "coolop", "lukeop", "kapp"), 0)
    for k, v in ref.items():
        if k.split(".")[0] in ("xnf", "xnfp", "ions") and k not in s:
            s[k] = torch.zeros_like(v)
    s["save.pops.itemp1"] = torch.tensor([0])
    return s, ref


def test_populations_golden(golden):
    s, ref = _start_state(golden)
    upd = populations(s)
    n = s.iscalar("rhox.nrhox")
    tol = T.golden_tol()
    checked = 0
    for k, v in upd.items():
        if k in ref and v.dtype == torch.float64:
            r = T.max_rel(v[:n].numpy(), ref[k][:n].numpy())
            assert r <= tol, f"{k}: {r:.3e}"
            checked += 1
    assert checked >= 25


def test_pops_golden(golden):
    s, _ = _start_state(golden)
    s = s.updated({k: v for k, v in nmolec(s, 1).items() if not k.startswith("_")})
    s["save.pops.itemp1"] = s["temp.itemp"].clone()
    for hit in (1, 2):
        e = golden(T.TRACE_STAGE, "pops", hit, "entry")
        x = golden(T.TRACE_STAGE, "pops", hit, "exit")
        num, _ = pops(s, float(e["arg:code"][0]), int(e["arg:mode"][0]))
        n = num.shape[0]
        exp = x["arg:number"].reshape(-1)
        got = num.T.reshape(-1).numpy()          # NUMBER(J,ION) column-major
        exp = np.concatenate([exp[:n]] + [exp[99 * c: 99 * c + n] for c in range(1, num.shape[1])]) \
            if exp.size >= 99 * num.shape[1] else exp[:n]
        assert T.max_rel(got[:exp.size], exp) <= T.golden_tol(), hit


CODES = [(1.01, 11), (2.02, 11), (6.03, 11), (7.04, 11), (8.05, 11), (10.05, 11), (12.01, 11), (13.00, 11),
         (14.01, 11), (20.01, 11), (26.00, 11), (106.00, 11), (108.00, 11), (6.05, 11), (16.05, 11), (26.04, 11),
         (10.02, 1), (18.01, 11), (28.03, 11), (5.01, 12), (101.0, 11), (8.0, 1)]


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(cm=st.sampled_from(CODES), ft=st.floats(0.9, 1.1))
def test_pops_vs_fortran(golden, cm, ft):
    code, mode = cm
    s, _ = _start_state(golden)
    T_ = s["temp.t"] * ft
    TK = 1.38054e-16 * T_
    s.update({"temp.t": T_, "temp.tk": TK, "temp.tkev": 8.6171e-5 * T_, "temp.hckt": 6.6256e-27 / TK * 2.99792458e10,
              "temp.hkt": 6.6256e-27 / TK, "temp.tlog": torch.log(T_)})
    s = s.updated({k: v for k, v in nmolec(s, 1).items() if not k.startswith("_")})
    o = T.make_oracle("atlas9")
    T.load_state(o, s)
    o.saved("pops")["itemp1"][...] = s.iscalar("temp.itemp")      # NMOLEC already done for this ITEMP
    s["save.pops.itemp1"] = s["temp.itemp"].clone()
    num, _ = pops(s, code, mode)
    n = num.shape[0]
    arr = np.zeros((99, 31), order="F")
    o.call("pops", float(code), mode, arr)
    exp = arr[:n, :num.shape[1]]
    assert T.max_rel(num.numpy(), exp) == 0.0, (code, mode)
