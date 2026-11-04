"""FTZ/DAZ environment scope (atlas_torch.dfsynthe.ftz): covers torch float32/float64 ops, float64->float32
conversion, numpy and libm on the calling thread; is fully restored on exit (no leak into other code)."""
import math
import platform

import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe import ftz

pytestmark = pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"), reason="x86 MXCSR only")


def test_scope_flushes_and_restores():
    n0 = torch.get_num_threads()
    assert not ftz.active()
    assert not any(ftz.probe().values())
    with ftz.scope(True):
        assert ftz.active()
        assert all(ftz.probe().values())
        assert torch.get_num_threads() == 1
        # REAL*4 store of a tiny REAL*8 (the xnfpdf.dat case) and subnormal products
        assert float(torch.tensor([1e-40], dtype=torch.float64).to(torch.float32)) == 0.0
        assert np.float32(np.float64(1e-40)) == 0
        assert math.exp(-720.0) == 0.0
        big = torch.full((200_000,), 1e-37, dtype=torch.float32) * 0.01
        assert bool((big == 0).all())
        with ftz.scope(True):                    # nested
            assert ftz.active()
        assert ftz.active()
        with ftz.scope(False):                   # IEEE island inside
            assert not ftz.active()
            assert math.exp(-720.0) > 0.0
        assert ftz.active()
    assert not ftz.active()
    assert not any(ftz.probe().values())
    assert torch.get_num_threads() == n0


def test_scope_restores_on_error():
    with pytest.raises(ZeroDivisionError):
        with ftz.scope(True):
            1 / 0
    assert not ftz.active()


def test_scope_off_is_noop():
    n0 = torch.get_num_threads()
    with ftz.scope(False):
        assert not ftz.active()
        assert torch.get_num_threads() == n0
        assert math.exp(-720.0) > 0.0


def test_flush_f32():
    x = torch.tensor([1e-39, -1e-40, 1.2e-38, 0.0, 1.0], dtype=torch.float32)
    y = ftz.flush_f32(x)
    assert y.tolist()[:2] == [0.0, 0.0] and y[2] == x[2] and y[4] == 1.0
