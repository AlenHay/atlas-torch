"""DFCALC, BISORT and the turbulent broadening of DFSYNTHE MAIN vs the Fortran.

1. golden: the debug executable's layer-1 spectrum (DFCALC entry, IVT=0) -> broaden(IVT=1,2) equals the traced
   DFCALC entry buffers for IVT=1,2 bit for bit; DFCALC of each traced buffer reproduces IBUFFER and the ODF steps
   the debug run wrote (fort.15/16/17, layer 1) exactly.
2. hypothesis: BISORT on random INTEGER arrays (incl. negative values and all-negative segments) vs the oracle.
3. hypothesis: DFCALC's step averaging on random spectra vs the oracle routine itself.
"""
import os
import tempfile

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.dfsynthe.dfsynthe import broaden, dfcalc
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines import tables as T
from atlas_torch.dfsynthe.odf_io import read_odf_bin

pytestmark = pytest.mark.skipif(not S.available(), reason="libdfsynthe.so or Intel runtime missing")
STAGE = "odf_solar/dfsynthe"
SCR = "/tmp/claude-1000" if os.path.isdir("/tmp/claude-1000") else None


@pytest.fixture(scope="module")
def intel():
    return S.intel_flavor()


def test_broaden_golden(golden, intel):
    tabs = T.device_tables(torch.device("cpu"), intel)
    b0 = torch.from_numpy(golden(STAGE, "dfcalc", 0, "entry")["buffer.buffer"])[None]
    for hit, ivt in ((1, 1), (2, 2)):
        g = golden(STAGE, "dfcalc", hit, "entry")
        assert int(g["arg:ivt"][0]) == ivt
        got = broaden(b0, ivt, tabs)[0].numpy()
        assert np.array_equal(got.view(np.int32), g["buffer.buffer"][: T.LENGTH].view(np.int32)), ivt


def test_dfcalc_golden(golden, intel):
    if not (S.TRACE_RUN / "fort.15").exists():
        pytest.skip("debug run outputs missing")
    tnp = T.build_tables(intel)
    ref = {0: "fort.15", 1: "fort.16", 2: "fort.17"}
    for hit in range(3):
        g = golden(STAGE, "dfcalc", hit, "entry")
        ivt = int(g["arg:ivt"][0])
        x = torch.from_numpy(g["buffer.buffer"][: T.LENGTH])[None]
        st = dfcalc(x, tnp, intel)[0].numpy()
        r = read_odf_bin(S.TRACE_RUN / ref[ivt])["isteps"][:1540]
        assert np.array_equal(st, r), ivt
    ge = golden(STAGE, "dfcalc", 0, "exit")["buffer.ibuffer"][: T.LENGTH]
    x = torch.from_numpy(golden(STAGE, "dfcalc", 0, "entry")["buffer.buffer"][: T.LENGTH])
    d = (intel.log10f(x) * 1000.0).double()
    ib = (torch.sign(d) * torch.floor(d.abs() + 0.5)).long().numpy()
    assert np.array_equal(ib, ge)


def _bisort_ref(o, k):
    a = np.asfortranarray(k.astype(np.int32))
    o.call("bisort", a, len(a), np.zeros(100000, np.int32))
    return a


def _bisort_port(k):
    """BISORT semantics as used in dfcalc(): stable sort by the low bit_length(|max|) bits of K+30000."""
    kk = torch.from_numpy(k.astype(np.int64)) + 30000
    mx = int(kk.max())
    nb = abs(mx).bit_length()
    key = kk & ((1 << nb) - 1)
    order = torch.sort(key, stable=True).indices
    return (kk[order] - 30000).numpy()


@settings(max_examples=150, deadline=None)
@given(k=st.lists(st.integers(-40000, 40000), min_size=1, max_size=300),
       shift=st.sampled_from([0, -60000, -30000]))
def test_bisort_vs_fortran(k, shift):
    k = np.array(k) + shift
    o = S.oracle()
    assert np.array_equal(_bisort_ref(o, k), _bisort_port(k))


@settings(max_examples=6, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), lo=st.floats(-12, 2), span=st.floats(0.5, 8))
def test_dfcalc_vs_fortran(seed, lo, span):
    fl = S.glibc_flavor()
    tnp = T.build_tables(fl)
    rng = np.random.default_rng(seed)
    x = (10.0 ** rng.uniform(lo, lo + span, T.LENGTH)).astype(np.float32)
    o = S.oracle()
    o.call("dfintervals")
    o.set_common("buffer", buffer=x)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(dir=SCR) as td:     # DFCALC WRITEs to unit 15 (default file fort.15)
        os.chdir(td)
        try:
            o.call("dfcalc", 21, 1, 0, np.array([5012.0], np.float32), np.array([1.0], np.float32),
                   np.array([0.0], np.float32))
        finally:
            os.chdir(cwd)
    ib = o.common("buffer")["ibuffer"][: T.LENGTH].copy()
    d = (fl.log10f(torch.from_numpy(x)) * 1000.0).double()
    mine_ib = (torch.sign(d) * torch.floor(d.abs() + 0.5)).long().numpy()
    assert np.array_equal(mine_ib, ib)
    st = dfcalc(torch.from_numpy(x)[None], tnp, fl)[0].numpy()
    idf = o.saved("dfcalc")["idfout"]
    last = idf[1:].view(np.int16)                       # ISTEPS of the last interval (IDF=1540)
    assert np.array_equal(st[1539], last)
