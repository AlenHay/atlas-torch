"""Oracle tests for KAPPA9's LINOP (in-memory ODF line opacity, 12 sub-steps).

1. golden: the stock solar run's LINOP hits 0..2 (first pressure, first frequency, N = 1..3): interpolation
   setup (CO1..CO4, IPJ, ITJ), step weights, WLBEG/WLEND, IWL, NSTEPS/STEPWT, the ODF image KAPWL and the
   last layer's A against the port.
2. hypothesis: random temperature / pressure columns and increasing frequency sequences -> the Fortran LINOP
   (unit 9 = the solar vturb-2 ODF) vs the port: ALINES, SIGLIN, NSTEPS, STEPWT, IWL, the setup statics.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.kappa9 import linop
from atlas_torch.kappa9.kappa9_prog import grid_pressures, grid_temperatures
from atlas_torch.kappa9.readin import wavedf
from atlas_torch.testing import kappa9_oracle_physics as k9o

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
RUN = EXT / "traces" / "odf_solar" / "kw"
STAGE = "odf_solar/kappa9"

pytestmark = pytest.mark.skipif(not k9o.available(), reason="libkappa9.so or the Intel Fortran runtime missing")


@pytest.fixture(scope="module")
def kap():
    p = RUN / "fort.9"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return linop.odf_kap(p)


def test_linop_golden(golden, kap):
    g = grid_temperatures()
    p = grid_pressures()[0].expand(57)
    co = linop.setup(g["tlog"], p)
    wlbeg, wlend = linop.wl_bounds(wavedf())
    for hit in range(3):
        ent = golden(STAGE, "linop", hit, "entry")
        ex = golden(STAGE, "linop", hit, "exit")
        n = int(ent["arg:n"][0])
        assert n == hit + 1
        # the program's grid = the trace's layers
        assert np.array_equal(ent["temp.t"][:57], g["t"].numpy())
        assert np.array_equal(ent["state.p"][:57], p.numpy())
        assert np.array_equal(ent["biglit.wavedf"], wavedf())          # BLOCKBIG DATA image
        for c in ("co1", "co2", "co3", "co4"):
            assert np.array_equal(ex[f"static:{c}"][:57], co[c].numpy()), c
        assert np.array_equal(ex["static:ipj"][:57], co["ip"].numpy())
        assert np.array_equal(ex["static:itj"][:57], co["it"].numpy())
        assert np.array_equal(ex["static:wt"], linop.step_weights().numpy())
        assert np.array_equal(ex["static:wlbeg"], wlbeg) and np.array_equal(ex["static:wlend"], wlend)
        kapwl = ex["static:kapwl"].reshape((25, 57, 12, 328), order="F")
        assert np.array_equal(kapwl.transpose(3, 1, 0, 2), kap.numpy())
        freq = float(ent["freq.freq"][0])
        iwl = linop.iwl_sequence([freq], wlbeg, wlend)[0]
        assert iwl == int(ex["static:iwl"][0])
        assert int(ex["arg:nsteps"][0]) == 12 and float(ex["arg:stepwt"][0]) == float(linop.step_weights()[n - 1])
        a = linop.line_opacity(kap, torch.tensor([iwl]), co, expf=linop.glibc_expf)[0, n - 1]
        assert np.float32(ex["static:a"][0]) == a[56].item()


@pytest.fixture(scope="module")
def flin(tmp_path_factory, kap):
    """Fortran LINOP with unit 9 read once (implicit OPEN of fort.9 in the cwd of the first call)."""
    k9o.preload_ifx_runtime()
    from atlas_torch.testing.fortran_oracle import Oracle
    o = Oracle(k9o.LIB)
    d = tmp_path_factory.mktemp("k9linop")
    os.symlink(RUN / "fort.9", d / "fort.9")
    g = grid_temperatures()
    o.set_common("rhox", nrhox=57)
    o.set_common("temp", t=g["t"].numpy(), tlog=g["tlog"].numpy(), itemp=1)
    o.set_common("state", p=grid_pressures()[0].expand(57).numpy())
    o.set_common("if", tauscat=0.0)
    o.set_common("biglit", wavedf=wavedf())
    o.set_common("freq", freq=3.0e15)
    cwd = os.getcwd()
    try:
        os.chdir(d)
        o.call("linop", 1, 0, 0.0)
    finally:
        os.chdir(cwd)
    o._itemp = 1
    return o


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 2**31 - 1), nl=st.integers(1, 99), nf=st.integers(1, 6))
def test_linop_vs_fortran(flin, kap, seed, nl, nf):
    o = flin
    rng = np.random.default_rng(seed)
    t = 10.0 ** rng.uniform(3.2, 5.45, nl)          # includes clipping at both ends of the ODF grid
    p = 10.0 ** rng.uniform(-5.0, 9.0, nl)
    o._itemp += 1
    o.set_common("rhox", nrhox=nl)
    o.set_common("temp", t=t, tlog=np.log(t), itemp=o._itemp)
    o.set_common("state", p=p)
    wls = 10.0 ** rng.uniform(np.log10(8.0), np.log10(12000.0), nf)
    if seed % 4 == 0:
        wls = np.append(wls, 8.5)                   # below WLBEG(1): LINOP returns without lines
    freqs = 2.997925e17 / np.sort(wls)              # increasing wavelength (IWL only advances)
    co = linop.setup(torch.from_numpy(np.log(t)), torch.from_numpy(p))
    wlbeg, wlend = linop.wl_bounds(wavedf())
    iwls = linop.iwl_sequence(freqs, wlbeg, wlend)
    a = linop.line_opacity(kap, torch.tensor(iwls), co, expf=linop.glibc_expf)     # the oracle's expf
    a_cr = linop.line_opacity(kap, torch.tensor(iwls), co)                          # default (correctly rounded)
    ulp = (a_cr.view(torch.int32) - a.view(torch.int32)).abs()
    assert int(ulp.max()) <= 1, "default EXP differs from libm expf by more than 1 ulp"
    for i, fr in enumerate(freqs):
        o.set_common("freq", freq=float(fr))
        for n in range(1, 13):
            o.common("ops")["alines"][:] = -1.0
            _, (nn, nsteps, stepwt) = o.call("linop", n, 0, 0.0)
            sv = o.saved("linop")
            if iwls[i] == 0:
                assert nsteps == 0 and np.all(o.common("ops")["alines"][:nl] == -1.0)   # early RETURN
                break
            assert nsteps == 12 and stepwt == float(linop.step_weights()[n - 1])
            assert int(sv["iwl"].reshape(-1)[0]) == iwls[i]
            got = o.common("ops")["alines"][:nl]
            assert np.array_equal(got, a[i, n - 1].double().numpy()), (fr, n)
            assert np.all(o.common("ops")["siglin"][:nl] == 0.0)
    sv = o.saved("linop")
    for c in ("co1", "co2", "co3", "co4"):
        assert np.array_equal(sv[c][:nl], co[c].numpy()), c
    assert np.array_equal(sv["itj"][:nl], co["it"].numpy()) and np.array_equal(sv["ipj"][:nl], co["ip"].numpy())
