"""Intel SVML single-precision exp and pow as linked into the stock dfsynthe.exe (vectorised setup loops call
__svml_expf4/__svml_expf1 and __svml_powf4/__svml_powf1, which are not correctly rounded).  Evaluated through the
Intel runtime's libsvml.so (the same compiler release) with ctypes; None when the runtime is not available (callers
then fall back to the correctly rounded function and document the deviation)."""
from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _fn(name: str, nargs: int):
    try:
        from ..xnf.oracle_support import ensure_ifx_runtime, ifx_lib_dir
        if not ensure_ifx_runtime():
            return None
        lib = ctypes.CDLL(ifx_lib_dir() + "/libsvml.so")
        f = getattr(lib, name)
        f.restype = ctypes.c_float
        f.argtypes = [ctypes.c_float] * nargs
        return f
    except Exception:
        return None


def _expf1():
    return _fn("__svml_expf1", 1)


def available() -> bool:
    return _expf1() is not None


def expf(x: np.ndarray) -> np.ndarray | None:
    f = _expf1()
    if f is None:
        return None
    x = np.asarray(x, np.float32)
    return np.fromiter((f(float(v)) for v in x.ravel()), dtype=np.float32, count=x.size).reshape(x.shape)


def powf(x: np.ndarray, y: float) -> np.ndarray | None:
    """__svml_powf1(x, y) elementwise (y a REAL*4 constant).  __svml_powf4 gives the same result lane by lane
    (checked on 3.2e8 inputs 1e2..1e22 with y = -2./15.); None without the Intel runtime."""
    f = _fn("__svml_powf1", 2)
    if f is None:
        return None
    x = np.asarray(x, np.float32)
    y = float(np.float32(y))
    return np.fromiter((f(float(v), y) for v in x.ravel()), dtype=np.float32, count=x.size).reshape(x.shape)
