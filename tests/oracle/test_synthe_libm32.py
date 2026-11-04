"""The single-precision libm layer of the SYNTHE port (lineopac/libm32.py).

* vexpf: float32 emulation of libmvec's SSE4 expf (used by the -O3 binary's vectorised loops) vs libmvec.
* slog10f: emulation of glibc 2.39's scalar log10f (the container that produced the reference runs) vs that
  libm, evaluated inside the atlas-dev container (skipped when the container is not available).
* native scalar calls vs ctypes calls of the same libm functions; loopvec epilogue semantics.
* torch's CPU index_add_ adds in index order (the property the exact line-buffer accumulation relies on).
"""
import ctypes
import ctypes.util
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.lineopac import libm32 as L
from atlas_torch.synthe.synthe_prog import _index_add_is_sequential, ordered_add_sorted

F32 = np.float32
pytestmark = pytest.mark.skipif(L.native() is None, reason="native libm calls unavailable")


def _bits(a):
    return np.asarray(a, F32).view(np.uint32)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), lo=st.sampled_from([-120., -90., -20., -1.]), hi=st.sampled_from([0., 1., 50., 95.]))
def test_vexpf_vs_libmvec(seed, lo, hi):
    x = np.random.default_rng(seed).uniform(lo, hi, 50000).astype(F32)
    ref = L.vexpf_native(torch.from_numpy(x)).numpy()
    assert np.array_equal(_bits(L.vexpf(torch.from_numpy(x)).numpy()), _bits(ref))


def test_vexpf_tables():
    x = np.concatenate([-np.arange(1001, dtype=F32), -(np.arange(1001, dtype=F32) * F32(.001)),
                        -(np.arange(2001, dtype=F32) / F32(200.)) ** 2]).astype(F32)
    assert np.array_equal(_bits(L.vexpf(torch.from_numpy(x)).numpy()), _bits(L.vexpf_native(torch.from_numpy(x)).numpy()))


@pytest.mark.parametrize("name,nargs", [("expf", 1), ("logf", 1), ("powf", 2)])
def test_native_scalar_vs_ctypes(name, nargs):
    rng = np.random.default_rng(3)
    x = (10 ** rng.uniform(-6, 6, 3000)).astype(F32)
    y = rng.uniform(-3, 3, 3000).astype(F32)
    if name == "expf":
        x = rng.uniform(-100, 88, 3000).astype(F32)
    lib = ctypes.CDLL(ctypes.util.find_library("m"))
    fn = getattr(lib, name)
    fn.restype = ctypes.c_float
    fn.argtypes = [ctypes.c_float] * nargs
    ref = np.array([fn(*(float(a), float(b))[:nargs]) for a, b in zip(x, y)], F32)
    got = {"expf": lambda: L.sexpf(torch.from_numpy(x)), "logf": lambda: L.slogf(torch.from_numpy(x)),
           "powf": lambda: L.spowf(torch.from_numpy(x), torch.from_numpy(y))}[name]().numpy()
    assert np.array_equal(_bits(got), _bits(ref))


def test_loopvec_epilogue():
    x = torch.linspace(-5, 1, 4 * 7 + 3).float().reshape(1, -1)
    got = L.loopvec(L.vexpf, L.sexpf, x, dim=1)
    assert torch.equal(got[:, :28], L.vexpf(x[:, :28]))
    assert torch.equal(got[:, 28:], L.sexpf(x[:, 28:]))


def test_index_add_sequential():
    assert _index_add_is_sequential()
    rng = np.random.default_rng(0)
    key = torch.from_numpy(rng.integers(0, 50, 200000))
    val = torch.from_numpy((rng.uniform(0, 1, 200000) * 10 ** rng.uniform(-8, 8, 200000)).astype(F32))
    a = torch.zeros(50, dtype=torch.float32).index_add_(0, key, val)
    ref = np.zeros(50, F32)
    np.add.at(ref, key.numpy(), val.numpy())        # unbuffered, in order
    b = torch.zeros(50, dtype=torch.float32)
    ordered_add_sorted(b, key, val)
    assert np.array_equal(_bits(a.numpy()), _bits(ref)) and np.array_equal(_bits(b.numpy()), _bits(ref))


_HELPER = r'''
import ctypes, array, sys
lib = ctypes.CDLL("libm.so.6"); f = lib.log10f; f.restype = ctypes.c_float; f.argtypes = [ctypes.c_float]
a = array.array("f"); a.frombytes(open(sys.argv[1], "rb").read())
open(sys.argv[2], "wb").write(array.array("f", [f(v) for v in a]).tobytes())
'''


def test_log10f_glibc239_vs_container():
    if shutil.which("distrobox") is None:
        pytest.skip("distrobox not available")
    rng = np.random.default_rng(5)
    x = np.concatenate([10 ** rng.uniform(-37, 37, 100000), rng.uniform(0.5, 2, 50000),
                        rng.uniform(1e13, 1e17, 20000), rng.uniform(4000, 50000, 20000)]).astype(F32)
    with tempfile.TemporaryDirectory(dir="/tmp/claude-1000" if Path("/tmp/claude-1000").is_dir() else None) as d:
        d = Path(d)
        (d / "h.py").write_text(_HELPER)
        x.tofile(d / "x.bin")
        cmd = f"ldd --version | head -1; python3 {d / 'h.py'} {d / 'x.bin'} {d / 'y.bin'}"
        r = subprocess.run(["distrobox", "enter", "atlas-dev", "--", "bash", "-c", cmd], capture_output=True,
                           text=True, timeout=300)
        if r.returncode != 0 or not (d / "y.bin").exists():
            pytest.skip(f"atlas-dev container not usable: {r.stderr[-200:]}")
        if "2.39" not in r.stdout:
            pytest.skip(f"container libc is not glibc 2.39: {r.stdout.strip()}")
        ref = np.fromfile(d / "y.bin", F32)
    got = L.slog10f(torch.from_numpy(x)).numpy()
    assert np.array_equal(_bits(got), _bits(ref))


def test_double_libmvec_pairs():
    """dvexp/dvpow (2-lane REAL*8 libmvec _ZGVbN2v_exp/_ZGVbN2vv_pow through the 16-byte trampoline): within a
    few ulp of scalar glibc, lane-independent (any pairing gives the same values), and dloopvec puts the odd
    last iteration through the scalar function as the vectorised loop's epilogue does."""
    import math
    if L.native() is None:
        pytest.skip("no native calls")
    rng = np.random.default_rng(3)
    x = rng.uniform(-700.0, 700.0, 4001)
    b = rng.uniform(1e-3, 1e3, 4001)
    y = rng.uniform(-3.0, 3.0, 4001)
    got_e, got_p = L.dvexp(x), L.dvpow(b, y)
    ref_e = np.array([math.exp(v) for v in x])
    ref_p = np.array([math.pow(u, v) for u, v in zip(b, y)])
    for v, ref in ((got_e, ref_e), (got_p, ref_p)):
        assert np.all(np.abs(v - ref) <= 4 * np.spacing(np.abs(ref)))
    assert (got_e != ref_e).any() or (got_p != ref_p).any()          # it is not the scalar function
    assert np.array_equal(L.dvexp(x[1:]), got_e[1:]) and np.array_equal(L.dvpow(b[1:], y[1:]), got_p[1:])
    lo = L.dloopvec(L.dvexp, lambda t: np.array([math.exp(v) for v in t]), x[:7])
    assert np.array_equal(lo[:6], got_e[:6]) and lo[6] == ref_e[6]
