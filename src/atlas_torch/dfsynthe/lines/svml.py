"""Intel SVML single-precision exp as linked into the stock dfsynthe.exe (vectorised setup loops call
__svml_expf4/__svml_expf1, which are not correctly rounded).  Evaluated through the Intel runtime's libsvml.so
(the same compiler release) with ctypes; None when the runtime is not available (callers then fall back to the
correctly rounded expf and document the deviation)."""
from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _expf1():
    try:
        from ..xnf.oracle_support import ensure_ifx_runtime, ifx_lib_dir
        if not ensure_ifx_runtime():
            return None
        lib = ctypes.CDLL(ifx_lib_dir() + "/libsvml.so")
        f = lib.__svml_expf1
        f.restype = ctypes.c_float
        f.argtypes = [ctypes.c_float]
        return f
    except Exception:
        return None


def available() -> bool:
    return _expf1() is not None


def expf(x: np.ndarray) -> np.ndarray | None:
    f = _expf1()
    if f is None:
        return None
    x = np.asarray(x, np.float32)
    return np.fromiter((f(float(v)) for v in x.ravel()), dtype=np.float32, count=x.size).reshape(x.shape)
