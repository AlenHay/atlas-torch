"""LINOP (atlas9mem.for, in-memory big ODF): golden traced calls and random states/frequencies/steps vs the
Fortran (which reads the ODF from fort.9 of the working directory)."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9 import linop as L

STAGE = "sun_atlas9/atlas9"


@pytest.fixture(scope="module")
def odf(ext):
    return L.read_odf(str(Path(ext) / "traces/sun_atlas9/w/fort.9"))


@pytest.fixture(scope="module")
def linop_oracle(oracle):
    """One library instance for the module: its LINOP reads the 11 MB ODF once (IREAD)."""
    return oracle("atlas9"), [1000]


def _port(e, odf, n_step):
    n = int(e["rhox.nrhox"][0])
    it, ip, co, fscat = L.coefficients(torch.as_tensor(e["temp.tlog"][:n]), torch.as_tensor(e["state.p"][:n]),
                                       torch.as_tensor(e["abross.tauros"][:n]), float(e["if.tauscat"][0]))
    iwl = L.interval_index([2.997925e17 / float(e["freq.freq"][0])])[0]
    if iwl is None:
        return None
    al, sl = L.line_opacity(odf, torch.tensor([iwl]), it, ip, co, fscat)
    return al[0, n_step - 1], sl[0, n_step - 1], (it, ip, co)


def test_linop_golden(golden, odf):
    for hit in range(3):
        e, x = golden(STAGE, "linop", hit, "entry"), golden(STAGE, "linop", hit, "exit")
        n, N = int(e["rhox.nrhox"][0]), int(e["arg:n"][0])
        al, sl, (it, ip, co) = _port(e, odf, N)
        assert np.array_equal(it.numpy(), x["static:itj"][:n]) and np.array_equal(ip.numpy(), x["static:ipj"][:n])
        for i in range(4):
            assert np.array_equal(co[i].numpy(), x[f"static:co{i + 1}"][:n])
        assert np.array_equal(al.numpy(), x["ops.alines"][:n]) and np.array_equal(sl.numpy(), x["ops.siglin"][:n])
        assert L.WT[N - 1] == x["arg:stepwt"][0] and int(x["arg:nsteps"][0]) == 12


@settings(max_examples=30, deadline=None)
@given(dlt=st.floats(-0.3, 0.6), dlp=st.floats(-3.0, 3.0), lw=st.floats(1.0, 4.0), nstep=st.integers(1, 12),
       tauscat=st.sampled_from([0.0, 0.0, 1e-3]))
def test_linop_vs_fortran(linop_oracle, golden, odf, ext, dlt, dlp, lw, nstep, tauscat):
    e = dict(golden(STAGE, "linop", 0, "entry"))
    n = int(e["rhox.nrhox"][0])
    e["temp.tlog"] = e["temp.tlog"].copy(); e["temp.tlog"][:n] += dlt * 2.302585
    e["state.p"] = e["state.p"].copy(); e["state.p"][:n] *= 10.0 ** dlp
    e["freq.freq"] = np.array([2.997925e17 / (10.0 ** lw)])
    e["if.tauscat"] = np.array([tauscat])
    o, itemp = linop_oracle
    itemp[0] += 1                     # new ITEMP -> LINOP recomputes its coefficients and restarts IWL at 1
    for blk in ("temp", "state", "abross", "freq", "if", "rhox", "biglit", "turbpr"):
        o.set_common(blk, **{k.split(".")[1]: v for k, v in e.items() if k.startswith(blk + ".")})
    o.set_common("temp", itemp=itemp[0])
    import ctypes
    N, NS, SW = ctypes.c_int(nstep), ctypes.c_int(0), ctypes.c_double(0.0)
    old = os.getcwd()
    os.chdir(Path(ext) / "traces/sun_atlas9/w")
    try:
        o.call("linop", N, NS, SW)
    finally:
        os.chdir(old)
    al, sl, _ = _port(e, odf, nstep)
    assert np.array_equal(al.numpy(), o.common("ops")["alines"][:n])
    assert np.array_equal(sl.numpy(), o.common("ops")["siglin"][:n])
    assert SW.value == L.WT[nstep - 1] and NS.value == 12
