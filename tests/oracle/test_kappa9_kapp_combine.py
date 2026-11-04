"""Oracle tests for the KAPPA9 program's KAPP combination step (atlas_torch.kappa9.kapp: kapp_first/kapp_next).

(tests/oracle/test_kappa9_kapp.py belongs to the physics library port: KAPP with all continuum sources.)
1. golden: the stock run's first KAPP call (N = 1): /OPS/ at exit -> port ACONT == trace ACONT.
2. hypothesis N = 1: a pressure column of the solar deck (populations through the test-only oracle adapter),
   random frequency -> Fortran KAPP(1) (LINOP off) vs the adapter's sources + port: all of /OPTOT/.
3. hypothesis N > 1: random /OPS/, BNU and previous SLINE -> Fortran KAPP(2..12) (LINOP off) vs port.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.kappa9 import kappa9_prog as kp
from atlas_torch.kappa9.kapp import kapp_first, kapp_next
from atlas_torch.kappa9.readin import readin
from atlas_torch.testing import kappa9_oracle_physics as k9o

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
STAGE = "odf_solar/kappa9"
RUN = EXT / "traces" / "odf_solar" / "kw"
OPTOT = ("acont", "scont", "aline", "sline", "sigmac", "sigmal")

pytestmark = pytest.mark.skipif(not k9o.available(), reason="libkappa9.so or the Intel Fortran runtime missing")


def test_kapp_combine_golden(golden):
    ex = golden(STAGE, "kapp", 0, "exit")
    assert int(ex["arg:n"][0]) == 1
    d = torch.from_numpy(ex["ops.dummy"])
    ops = {k: d[:, i] for i, k in enumerate(k9o.OPS)}
    out = kapp_first(ops, torch.ones(99))
    assert np.array_equal(out["acont"][:57].numpy(), ex["optot.acont"][:57])


@pytest.fixture(scope="module")
def col():
    rd = readin((RUN / "stdin.txt").read_text(), RUN / "fort.2")
    ph = k9o.OracleKappa9Physics()
    ph.load(rd)
    return rd, ph, {"k": 0}


def _kapp_fortran(o, n):
    ifop = o.common("ifop")["ifop"]
    save = ifop.copy()
    ifop[14] = ifop[15] = 0          # LINOP / LINSOP: the program's own (tested in test_kappa9_linop.py)
    try:
        o.call("kapp", n, 0, 0.0)
    finally:
        ifop[:] = save


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(k=st.integers(1, 42), inu=st.integers(0, 336))
def test_kapp_first_vs_fortran(col, k, inu):
    rd, ph, last = col
    g = kp.grid_temperatures()
    freq, _ = kp.frequency_set(rd)
    fa = kp.freq_arrays(torch.from_numpy(freq), g)
    if last["k"] != k:
        gk = dict(g)
        gk["p"] = kp.grid_pressures()[k - 1].expand(57).clone()
        ph.populations(k, gk)
        last["k"] = k
    ops = ph.continuum(float(freq[inu]), float(fa["freqlg"][inu]), fa["ehvkt"][inu], fa["stim"][inu], fa["bnu"][inu])
    port = kapp_first(ops, fa["bnu"][inu])
    _kapp_fortran(ph.o, 1)
    ot = ph.o.common("optot")
    for v in OPTOT:
        assert np.array_equal(port[v].numpy(), ot[v][:57]), v


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 2**31 - 1), n=st.integers(2, 12), nl=st.integers(1, 99))
def test_kapp_next_vs_fortran(col, seed, n, nl):
    _, ph, last = col
    o = ph.o
    rng = np.random.default_rng(seed)
    o.set_common("rhox", nrhox=nl)
    vals = rng.lognormal(0, 3, (99, 25)) * (rng.random((99, 25)) < 0.7)
    o.common("ops")["dummy"][:] = vals
    bnu = rng.lognormal(0, 2, 99)
    sline0 = rng.lognormal(0, 2, 99)
    o.set_common("freq", bnu=bnu)
    o.set_common("optot", sline=sline0)
    _kapp_fortran(o, n)
    last["k"] = 0     # the column's COMMONs are clobbered: re-run populations next time
    ops = {k: torch.from_numpy(vals[:nl, i].copy()) for i, k in enumerate(k9o.OPS)}
    port = kapp_next(ops, torch.from_numpy(bnu[:nl]), torch.from_numpy(sline0[:nl]))
    ot = o.common("optot")
    for v in ("aline", "sline", "sigmal"):
        assert np.array_equal(port[v].numpy(), ot[v][:nl]), v
