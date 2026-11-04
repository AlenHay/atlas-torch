"""Oracle tests for FUNCTION XKARSAS(FREQ,ZEFF2,N,L) (atlas7v.for, IMPLICIT REAL*4): Karzas & Latter
hydrogenic bound-free cross-sections, interpolated in REAL*4. Golden: the traced calls; property: every
branch (L>=N, N<=15 / L<N, N<=6 / N>15) over frequency and ZEFF2."""
import ctypes

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.hydrogen import xkarsas
from atlas_torch.testing.compare import assert_close


def test_xkarsas_golden(golden, oracle):
    for hit in range(3):
        d = golden(T.STAGE, "xkarsas", hit, "entry")
        f, z, n, l = (float(d["arg:freq"][0]), float(d["arg:zeff2"][0]), int(d["arg:n"][0]), int(d["arg:l"][0]))
        o = oracle("atlas7v")
        ref, _ = o.call("xkarsas", f, z, n, l, restype=ctypes.c_double)
        assert_close(xkarsas(torch.tensor([f]), z, n, l), np.array([ref]), rtol=0.0, name=f"xkarsas[{hit}]")
        e = golden(T.STAGE, "xkarsas", hit, "exit")
        assert float(e["static:freqlg"][0]) == float(np.float32(o.saved("xkarsas")["freqlg"][0]))


@settings(max_examples=200, deadline=None)
@given(lf=st.floats(12.0, 18.0), n=st.integers(1, 20), dl=st.integers(-6, 3),
       zeff2=st.sampled_from([1.0, 4.0, 9.0]) | st.floats(1.0, 30.0))
def test_xkarsas_vs_fortran(oracle, lf, n, dl, zeff2):
    l = max(0, n + dl) if n <= 6 else n
    o = oracle("atlas7v")
    f = 10.0**lf
    ref, _ = o.call("xkarsas", f, zeff2, n, l, restype=ctypes.c_double)
    assert_close(xkarsas(torch.tensor([f]), zeff2, n, l), np.array([ref]), rtol=0.0,
                 name=f"xkarsas({f:.6e},{zeff2},{n},{l})")


def test_xkarsas_batch_equals_scalar():
    f = torch.from_numpy(10.0 ** np.linspace(12.5, 16.5, 301))
    for n in (1, 2, 6, 15):
        b = xkarsas(f, 1.0, n, n)
        s = torch.stack([xkarsas(f[i:i + 1], 1.0, n, n)[0] for i in range(f.numel())])
        assert torch.equal(b, s)


def _freq_at_table_end(fn29: np.float32) -> float:
    """A REAL*8 FREQ whose LOG10(SNGL(FREQ)) is exactly FREQN(29,N) (the table search then falls through)."""
    from atlas_torch.synthe.opacity._core import log10f
    f0 = 10.0 ** float(fn29)
    for k in range(4000):
        for s in (1, -1):
            f = f0 * (1.0 + s * k * 1e-9)
            if log10f(np.array([f], np.float32))[0] == fn29:
                return f
    raise AssertionError("no frequency hits the table end")


def test_xkarsas_fall_through_vs_fortran(oracle):
    """FREQLG == FREQN(29,N): DO loop exits with I=30, FREQN(30,N)/XN(30,N)/XL(30,L+1,N) read out of bounds
    (next column, or the zero padding after the last one: N=15 at WAVENO 487.456, the xnfpelsyn NaN)."""
    from atlas_torch.synthe.opacity.hydrogen import _xk_tables
    fn = _xk_tables()[0]
    o = oracle("atlas7v")
    for n, l in ((15, 15), (14, 14), (7, 7), (6, 6), (6, 5), (3, 1), (1, 0)):
        f = _freq_at_table_end(fn[29 * (n - 1) + 28])
        ref, _ = o.call("xkarsas", f, 1.0, n, l, restype=ctypes.c_double)
        got = xkarsas(torch.tensor([f]), 1.0, n, l)
        assert np.isfinite(ref) and torch.isfinite(got).all()
        assert_close(got, np.array([ref]), rtol=0.0, name=f"xkarsas({f!r},1,{n},{l})")
