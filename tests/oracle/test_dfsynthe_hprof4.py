"""Oracle tests for DFSYNTHE's profile functions: VOIGT, VCSE1F, HFNM, SOFBET (incl. its stale-P cache quirk)
and HPROF4 / HPROF4m / HPROF4p (dfsynthe.for), REAL*4 bit for bit with Intel libimf (intel_flavor).
"""
import ctypes

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.dfsynthe import odf_io
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines.hydrogen import Hydrogen, hfnm, sofbet, sofbet_index, vcse1f
from atlas_torch.dfsynthe.lines.tables import build_tables, device_tables
from atlas_torch.dfsynthe.lines.xlinop import ehyd_tables, voigt

pytestmark = pytest.mark.skipif(not S.available(), reason="libdfsynthe.so or Intel runtime missing")
f4 = np.float32
CF = ctypes.c_float


def bits(x):
    return np.asarray(x, f4).view(np.int32)


@pytest.fixture(scope="module")
def fl():
    # the oracle library inside this process binds expf/logf/... to glibc libm
    return S.oracle_flavor()


@pytest.fixture(scope="module")
def tabs(fl):
    return device_tables(torch.device("cpu"), fl)


@pytest.fixture(scope="module")
def solar():
    p = S.STOCK / "xnfpdf.dat"
    if not p.exists():
        pytest.skip("stock xnfpdf.dat missing")
    return odf_io.read_xnfpdf(p, 21)


@settings(max_examples=300, deadline=None)
@given(v=st.floats(0, 60, width=32), a=st.one_of(st.floats(0, 3, width=32), st.floats(0, 300, width=32)))
def test_voigt_vs_fortran(tabs, v, a):
    o = S.oracle()
    o.call("tabvoigt", np.array([200.0], f4), 2001)
    r, _ = o.call("voigt", CF(v), CF(a), restype=CF)
    got = voigt(torch.tensor([v], dtype=torch.float32), torch.tensor([a], dtype=torch.float32), tabs)
    assert bits(got.numpy())[0] == bits([r])[0]


@settings(max_examples=200, deadline=None)
@given(x=st.one_of(st.floats(-1, float(np.float32(0.02)), width=32), st.floats(0, 40, width=32)))
def test_vcse1f_vs_fortran(fl, x):
    o = S.oracle()
    r, _ = o.call("vcse1f", CF(x), restype=CF)
    assert bits(vcse1f(torch.tensor([x], dtype=torch.float32), fl).numpy())[0] == bits([r])[0]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
def test_hfnm_vs_fortran(fl, n):
    o = S.oracle()
    for m in range(1, 101):
        r, _ = o.call("hfnm", n, m, restype=CF)
        assert bits([hfnm(n, m, fl)])[0] == bits([r])[0], (n, m)


@settings(max_examples=60, deadline=None)
@given(seq=st.lists(st.tuples(st.floats(0.5, 800, width=32), st.sampled_from([0.0, 0.13, 0.37, 0.6, 0.79]),
                              st.sampled_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (2, 9), (1, 20)])),
                    min_size=1, max_size=25))
def test_sofbet_cache_quirk(seq):
    """SOFBET keeps the P-interpolated tables of the last INDEX change: emulate with p_eff."""
    o = S.oracle()
    idx_last, p_eff = 0, None
    for b, p, (n, m) in seq:
        r, _ = o.call("sofbet", CF(b), CF(p), n, m, restype=CF)
        ix = sofbet_index(n, m)
        if b < 500 and ix != idx_last:
            idx_last, p_eff = ix, f4(p)
        pe = torch.tensor([p_eff if p_eff is not None else 0.0], dtype=torch.float32)
        got = sofbet(torch.tensor([b], dtype=torch.float32), pe, ix)
        assert bits(got.numpy())[0] == bits([r])[0], (b, p, n, m)


def _oracle_hprof4(solar, fl):
    o = S.oracle()
    t = build_tables(fl)
    st = solar.state
    o.set_common("temp", t=st["t"], itemp=1)
    o.set_common("state", xne=st["xne"])
    o.set_common("bhe", xnfhe=st["xnfhe"])
    o.set_common("txnxn", xnfh2=st["xnfh2"])
    o.set_common("bhyd", xnfph=solar.xnfpel[:, 0:2])
    o.set_common("extab", extab=t.extab)
    o.set_common("h1tab", h0tab=t.h0tab)
    o.set_common("rhox", nrhox=25)
    return o


def _hydrogen(solar, fl, tabs):
    st = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in solar.state.items()}
    return Hydrogen(st, torch.from_numpy(solar.xnfpel[:, 0:2].copy()), tabs["extab"], tabs["h0tab"], fl)


LINES = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 12), (2, 3), (2, 4), (2, 5), (2, 7), (3, 4), (3, 6), (4, 9), (5, 7),
         (2, 15), (1, 40)]


@settings(max_examples=120, deadline=None)
@given(nm=st.sampled_from(LINES), j=st.integers(1, 25), which=st.sampled_from(["hprof4", "hprof4m", "hprof4p"]),
       lg=st.floats(-5, 2.5), sgn=st.sampled_from([-1.0, 1.0]), dshift=st.floats(-0.3, 0.3))
def test_hprof4_vs_fortran(solar, fl, tabs, nm, j, which, lg, sgn, dshift):
    n, m = nm
    e, _ = ehyd_tables()
    wl0 = 1.0e7 / (e[m - 1] - e[n - 1]) * (1.0 + 1e-3 * dshift)
    delw = sgn * 10.0 ** lg
    dopph = (solar.dopple[:, 0] * (f4(1) if dshift >= 0 else f4(1) / f4(1.4142))).astype(f4)
    o = _oracle_hprof4(solar, fl)
    r, _ = o.call(which, n, m, j, float(wl0), float(delw), np.asfortranarray(dopph), restype=CF)
    hyd = _hydrogen(solar, fl, tabs)
    s = hyd.setup(n, m)
    jj = torch.tensor([j - 1])
    q = hyd.prep(s, jj, torch.tensor([wl0], dtype=torch.float64), torch.tensor([delw], dtype=torch.float64),
                 torch.from_numpy(dopph[j - 1:j]))
    pp = hyd.pp[jj]
    got = hyd.finish(q, pp, pp, {"hprof4": "4", "hprof4m": "m", "hprof4p": "p"}[which])
    assert bits(got.numpy())[0] == bits([r])[0], (n, m, j, which, wl0, delw)


def test_hprof4_depth_vectors(solar, fl, tabs):
    """Per-ITEMP depth vectors (PP, FO, Y1B, ...) equal the Fortran's SAVEd arrays."""
    o = _oracle_hprof4(solar, fl)
    o.call("hprof4", 1, 2, 1, 121.5, 0.01, np.asfortranarray(solar.dopple[:, 0]), restype=CF)
    sv = o.saved("hprof4")
    hyd = _hydrogen(solar, fl, tabs)
    for k in ("pp", "fo", "y1b", "y1s", "t3nhe", "t3nh2", "c1d", "c2d", "gcon1", "gcon2"):
        assert np.array_equal(bits(sv[k]), bits(getattr(hyd, k).numpy())), k
