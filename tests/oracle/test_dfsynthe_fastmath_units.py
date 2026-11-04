"""Units of the stock (ifx -fp-model fast) emulation used by Flavor('stock').fastmath:
rcpps table + Newton division (lines/rcpps.py), the far-wing formula with one reciprocal
(LineAccumulator._asym_fast), TABVOIGT's reciprocal/SVML H tables, the rsqrtps emulation (vs the instruction),
the libimf and SVML-vector paths (lines/imf.py, svml.expf4/powf4).  The checks against the stock binary are
tests/stage/test_dfsynthe_stock_state.py (tables, SELECTLINES output and opacity buffers bit for bit) and
tests/stage/test_dfsynthe_prog.py (ODFs)."""
import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe.lines import rcpps, svml
from atlas_torch.dfsynthe.lines.accumulate import LineAccumulator
from atlas_torch.dfsynthe.lines.f32 import Flavor
from atlas_torch.dfsynthe.lines.tables import tabvoigt


def test_rcpps_table_scaling():
    t = rcpps._table_np()
    assert t.size == 4096
    one = torch.tensor([1.0, 2.0, 0.5, -4.0, 3.0e10], dtype=torch.float32)
    r = rcpps.rcpps(one)
    assert r.view(torch.int32)[0] == int(t[0])                      # input 1.0 -> table entry 0
    assert float(r[1]) == float(r[0]) / 2 and float(r[2]) == float(r[0]) * 2 and float(r[3]) == -float(r[0]) / 4
    x = torch.from_numpy((10.0 ** np.random.default_rng(0).uniform(-30, 30, 100000)).astype(np.float32))
    rel = (rcpps.rcpps(x).double() * x.double() - 1).abs().max()
    assert float(rel) < 1.5 * 2.0 ** -12                             # rcpps accuracy bound


def test_div_nr_within_two_ulp():
    rng = np.random.default_rng(1)
    a = torch.from_numpy((10.0 ** rng.uniform(-10, 10, 200000)).astype(np.float32))
    b = torch.from_numpy((10.0 ** rng.uniform(-10, 10, 200000)).astype(np.float32))
    q, e = rcpps.div_nr(a, b), a / b
    ulp = (q.view(torch.int32).long() - e.view(torch.int32).long()).abs()
    assert int(ulp.max()) <= 2 and 0 < int((ulp > 0).sum()) < 100000      # one Newton step: not always IEEE


def test_asym_fast_close_to_source_formula():
    tabs = {"h0tab": torch.zeros(1)}
    acc = LineAccumulator.__new__(LineAccumulator)
    LineAccumulator.__init__(acc, tabs, torch.zeros(1), torch.zeros(1), torch.ones(1, 1), torch.ones(1, 1),
                             torch.ones(1, 10), fastmath=True)
    rng = np.random.default_rng(2)
    profc = torch.from_numpy(rng.uniform(1, 100, (1, 300)).astype(np.float32))
    aa = torch.from_numpy(rng.uniform(2, 1e4, (1, 300)).astype(np.float32))
    kdv = torch.arange(1, 2001, dtype=torch.float32)[:, None] * torch.from_numpy(
        rng.uniform(1e-3, 1e-1, (1, 300)).astype(np.float32))
    f, s = acc._asym_fast(profc, aa, kdv.clone()), acc._asym(profc, aa, kdv.clone())
    rel = ((f.double() - s.double()) / s.double()).abs().max()
    assert float(rel) < 1e-5 and bool((f != s).any())


def test_tabvoigt_fast_tables():
    h0f, h1f, h2f, _ = tabvoigt(fl=Flavor("stock"))
    h0p, h1p, h2p, _ = tabvoigt(fl=Flavor("stock", fastmath=False))
    assert h0f[0] == 1.0 and bool(np.all(np.diff(h0f) <= 0))
    assert np.max(np.abs(h0f.astype(np.float64) - h0p) / h0p) < 1e-4
    if svml.available():
        assert int((h0f != h0p).sum()) > 500                          # reciprocal VV + SVML: ~790 of 2001 differ


def test_rsqrtps_matches_instruction(tmp_path):
    """rcpps.rsqrtps (table of the reference CPU) equals the rsqrtps instruction of this machine (skipped on other
    CPUs: the table is CPU specific, see rcpps.py)."""
    import ctypes
    import platform
    import shutil
    import subprocess
    if platform.machine() not in ("x86_64", "AMD64") or shutil.which("gcc") is None:
        pytest.skip("x86-64 + gcc needed")
    src = tmp_path / "rsq.c"
    src.write_text("#include <immintrin.h>\nvoid rsq(const float*x,float*y,long n){for(long i=0;i<n;i+=4){"
                   "float t[4]={1,1,1,1},r[4];for(int k=0;k<4&&i+k<n;k++)t[k]=x[i+k];"
                   "_mm_storeu_ps(r,_mm_rsqrt_ps(_mm_loadu_ps(t)));for(int k=0;k<4&&i+k<n;k++)y[i+k]=r[k];}}\n")
    so = tmp_path / "rsq.so"
    subprocess.run(["gcc", "-O2", "-msse2", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    lib = ctypes.CDLL(str(so))
    rng = np.random.default_rng(3)
    x = rng.integers(0, 2 ** 32, 1_000_000, dtype=np.uint64).astype(np.uint32).view(np.float32)
    x = np.concatenate([x, np.array([0.0, -0.0, 1.0, 2.0, 4.0, np.inf, -1.0, 1e-40], np.float32)])
    y = np.empty_like(x)
    lib.rsq(ctypes.c_void_p(x.ctypes.data), ctypes.c_void_p(y.ctypes.data), ctypes.c_long(x.size))
    tab_first = rcpps.rsqrtps(torch.tensor([1.0], dtype=torch.float32)).view(torch.int32)[0]
    if int(tab_first) != int(y[x == 1.0].view(np.int32)[0]):
        pytest.skip("rsqrtps table belongs to another CPU")
    for got in (rcpps.rsqrtps(torch.from_numpy(x)).numpy(), rcpps.rsqrtps_np(x)):
        same = (got.view(np.int32) == y.view(np.int32)) | (np.isnan(got) & np.isnan(y))
        assert bool(same.all())


def test_intel_math_paths():
    """libimf expf is not correctly rounded (the stock SELECTLINES/HPROF4 scalar calls use it), and the SVML
    vector entry points are reached through the C helper; skipped without the Intel runtime."""
    from atlas_torch.dfsynthe.lines import imf
    x = np.linspace(-20.0, 20.0, 400_001).astype(np.float32)
    e = imf.expf(x)
    if e is None or svml.expf4(x[:8]) is None:
        pytest.skip("Intel runtime or C helper missing")
    cr = np.exp(x.astype(np.float64)).astype(np.float32)
    assert 0 < int((e != cr).sum()) < 1000                          # ~1e-4 of the inputs, one ulp
    y = x[:4096] * np.float32(0.1)
    assert np.array_equal(svml.powf_xy(np.float32(10.0), y), svml.powf4(np.float32(10.0), y))
    assert imf.map1("powf", np.float32([2.0]), np.float32([10.0]))[0] == np.float32(1024.0)
