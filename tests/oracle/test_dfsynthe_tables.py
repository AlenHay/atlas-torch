"""Oracle tests for DFSYNTHE's set-up routines (dfsynthe.for): TABVOIGT (+ its REAL*4 MAP1), VTTAB,
DFINTERVALS, EXPI, the MAIN tables EXTAB/E1TAB, and the libm / integer-power flavours.

Fortran = libdfsynthe.so (ifx -g -O0 -save).  Inside this Python process the library's expf/logf/... bind to
glibc libm, so the port uses glibc through ctypes (glibc_flavor) here and every table must agree bit for bit;
golden traces of the debug executable (libimf-bound) use intel_flavor.  With the default (float64-rounded)
transcendentals the agreement is within one ulp on a handful of entries (checked too).
"""
import ctypes

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines import tables as T
from atlas_torch.dfsynthe.lines.f32 import Flavor, powidf2_np, powr8i4_np

pytestmark = pytest.mark.skipif(not S.available(), reason="libdfsynthe.so or Intel runtime missing")
f4 = np.float32


@pytest.fixture(scope="module")
def fl():
    # the oracle library inside this process binds expf/logf/... to glibc libm
    return S.oracle_flavor()


def _bits(a):
    return np.asarray(a, f4).view(np.int32)


def test_tabvoigt_vs_fortran(fl):
    o = S.oracle()
    o.call("tabvoigt", np.array([200.0], f4), 2001)
    h = o.common("h1tab")
    h0, h1, h2, atab = T.tabvoigt(fl=fl)
    for k, v in (("h0tab", h0), ("h1tab", h1), ("h2tab", h2), ("atab", atab)):
        assert np.array_equal(_bits(h[k]), _bits(v)), k
    # default libm: at most a few one-ulp differences
    d0 = T.tabvoigt(fl=Flavor("O0"))[0]
    assert np.abs(_bits(d0) - _bits(h["h0tab"])).max() <= 1


def test_tabvoigt_golden(golden):
    """The debug executable (golden trace) binds expf to Intel libimf."""
    g = golden("odf_solar/dfsynthe", "tabvoigt", 0, "exit")
    assert np.array_equal(_bits(g["h1tab.h0tab"]), _bits(T.tabvoigt(fl=S.intel_flavor())[0]))


def test_vttab_vs_fortran(fl):
    o = S.oracle()
    o.call("vttab")
    v = o.common("vtprof")
    a, b, c = T.vttab(fl)
    assert np.array_equal(_bits(v["vtprof"]), _bits(a))
    assert np.array_equal(_bits(v["vtcenter"]), _bits(b))
    assert np.array_equal(v["nvt"], c)


def test_dfintervals_vs_fortran():
    o = S.oracle()
    o.call("dfintervals")
    nb = o.common("nbiglit")
    nbeg, nend, nsteps, nn = T.dfintervals()
    assert np.array_equal(nb["nbeg"], nbeg) and np.array_equal(nb["nend"], nend)
    assert np.array_equal(nb["nsteps"], nsteps) and np.array_equal(nb["nn"], nn)


def test_main_tables(fl):
    t = T.build_tables(fl)
    o = S.oracle()
    e1 = np.array([o.call("expi", 1, ctypes.c_float(x), restype=ctypes.c_float)[0]
                   for x in (np.arange(1, 2001).astype(f4) * T.lit(".01")).astype(f4)], f4)
    assert np.array_equal(_bits(e1), _bits(t.e1tab))
    assert t.extab[0] == 1.0 and np.all(np.diff(t.extab) <= 0)


@settings(max_examples=150, deadline=None)
@given(n=st.integers(1, 4), x=st.floats(float(np.float32(1e-4)), 60.0, width=32))
def test_expi_vs_fortran(fl, n, x):
    o = S.oracle()
    r, _ = o.call("expi", n, ctypes.c_float(x), restype=ctypes.c_float)
    assert _bits(T.expi_f32(n, np.array([x], f4), fl))[0] == _bits(np.array([r], f4))[0]


@settings(max_examples=80, deadline=None)
@given(n=st.integers(4, 40), m=st.integers(1, 60), seed=st.integers(0, 2**31 - 1))
def test_map1_f32_vs_fortran(n, m, seed):
    rng = np.random.default_rng(seed)
    xold = np.sort(rng.uniform(0, 10, n)).astype(f4)
    if np.any(np.diff(xold) == 0):
        return
    fold = rng.normal(size=n).astype(f4)
    xnew = np.sort(rng.uniform(-1, 11, m)).astype(f4)
    fnew = np.zeros(m, f4)
    o = S.oracle()
    o.call("map1", np.asfortranarray(xold), np.asfortranarray(fold), n, np.asfortranarray(xnew), fnew, m)
    got, _ = T.map1_f32(xold, fold, xnew)
    assert np.array_equal(_bits(got), _bits(fnew))


def test_integer_power_flavours():
    """REAL*8**INTEGER: debug/oracle builds call Intel __powr8i4 (x87 extended powering), stock -O2 calls
    compiler-rt __powidf2 (double powering).  Compare with the actual runtime routines."""
    d = S.preload()
    lib = ctypes.CDLL(d + "/libifcoremt.so.5")
    p8 = lib.__powr8i4
    p8.restype, p8.argtypes = ctypes.c_double, [ctypes.c_double, ctypes.c_int]
    ns = np.arange(0, 3510000, 7919)
    r = 1.0 + 1.0 / 500000.0
    ref = np.array([p8(r, int(k)) for k in ns])
    assert np.array_equal(ref, powr8i4_np(r, ns))
    assert not np.array_equal(ref, powidf2_np(r, ns))
    p4 = lib.__powr4i4
    p4.restype, p4.argtypes = ctypes.c_float, [ctypes.c_float, ctypes.c_int]
    xs = np.random.default_rng(0).uniform(0.01, 300, 3000).astype(f4)
    for k in (3, 4):
        ref = np.array([p4(float(x), k) for x in xs], f4)
        assert np.array_equal(ref, Flavor("O0").powi4(torch.from_numpy(xs), k).numpy())
