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


def powf_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """__svml_powf1(x, y) elementwise over broadcast arrays (SELECTLINES TABLOG: __svml_powf4 lane by lane, checked
    against the stock TABLOG); None without the Intel runtime."""
    f = _fn("__svml_powf1", 2)
    if f is None:
        return None
    x, y = np.broadcast_arrays(np.asarray(x, np.float32), np.asarray(y, np.float32))
    return np.fromiter((f(float(a), float(b)) for a, b in zip(x.ravel(), y.ravel())), dtype=np.float32,
                       count=x.size).reshape(x.shape)


def powf(x: np.ndarray, y: float) -> np.ndarray | None:
    """__svml_powf1(x, y) elementwise (y a REAL*4 constant).  __svml_powf4 gives the same result lane by lane
    (checked on 3.2e8 inputs 1e2..1e22 with y = -2./15.); None without the Intel runtime."""
    f = _fn("__svml_powf1", 2)
    if f is None:
        return None
    x = np.asarray(x, np.float32)
    y = float(np.float32(y))
    return np.fromiter((f(float(v), y) for v in x.ravel()), dtype=np.float32, count=x.size).reshape(x.shape)


# ---- vector entry points (__svml_expf4, __svml_powf4): lanes of the 4-wide SSE variants, through the C helper
#      native/imfmap.c (map_v4f1/map_v4f2).  The stock vector loops call these; their results differ from the
#      scalar __svml_*1 variants for some inputs (e.g. 131 of the 2001 TABVOIGT H0TAB values).
@lru_cache(maxsize=None)
def _vaddr(name: str):
    try:
        from ..xnf.oracle_support import ensure_ifx_runtime, ifx_lib_dir
        if not ensure_ifx_runtime():
            return None
        lib = ctypes.CDLL(ifx_lib_dir() + "/libsvml.so")
        return ctypes.cast(getattr(lib, name), ctypes.c_void_p).value
    except Exception:
        return None


def _vmap(name: str, *xs) -> np.ndarray | None:
    from .native import load_extra
    lib, fp = load_extra("imfmap"), _vaddr(name)
    if lib is None or fp is None:
        return None
    xs = np.broadcast_arrays(*(np.asarray(x, np.float32) for x in xs))
    xs = [np.ascontiguousarray(x) for x in xs]
    y = np.empty(xs[0].shape, np.float32)
    if y.size:
        P = ctypes.c_void_p
        fn = lib.map_v4f1 if len(xs) == 1 else lib.map_v4f2
        fn(P(fp), *(P(x.ctypes.data) for x in xs), P(y.ctypes.data), ctypes.c_long(y.size))
    return y


def expf4(x: np.ndarray) -> np.ndarray | None:
    """__svml_expf4 lane by lane (vectorised stock loops); None without the Intel runtime or the C helper."""
    return _vmap("__svml_expf4", x)


def powf4(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """__svml_powf4 lane by lane; None without the Intel runtime or the C helper."""
    return _vmap("__svml_powf4", x, y)
