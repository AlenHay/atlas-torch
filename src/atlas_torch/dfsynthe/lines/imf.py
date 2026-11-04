"""Intel libimf scalar exp/expf as called by the stock dfsynthe.exe.

The executable's scalar `expf@plt` / `exp@plt` calls bind to libimf.so (it precedes libm in the NEEDED list), and
libimf expf is not correctly rounded: it differs from the correctly rounded expf for 1.1e-4 of the float32 inputs
in (-103, 88.7) (one ulp).  SELECTLINES evaluates EXP(-ELO*HCKT), EXP(IWL*RATIOLG4) and EXP(-FREQ4*HKT) per line
with it, so the stock flavour does too (Flavor.fastmath).  The functions come from the Intel runtime through
ctypes and are mapped over arrays by the small C helper native/imfmap.c (built by `make native`; without it a
Python loop gives the same values, slowly).  Without the Intel runtime the functions return None and callers use
the correctly rounded functions (documented deviation, as for svml.py).
"""
from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _fn(name: str, double: bool, nargs: int = 1):
    try:
        from ..xnf.oracle_support import ensure_ifx_runtime, ifx_lib_dir
        if not ensure_ifx_runtime():
            return None
        lib = ctypes.CDLL(ifx_lib_dir() + "/libimf.so")
        t = ctypes.c_double if double else ctypes.c_float
        f = ctypes.CFUNCTYPE(t, *([t] * nargs))((name, lib))
        return f
    except Exception:
        return None


def _map(name: str, x: np.ndarray, double: bool) -> np.ndarray | None:
    f = _fn(name, double)
    if f is None:
        return None
    dt = np.float64 if double else np.float32
    x = np.ascontiguousarray(x, dtype=dt)
    y = np.empty_like(x)
    from .native import load_extra
    lib = load_extra("imfmap")
    if lib is not None and x.size:
        P = ctypes.c_void_p
        getattr(lib, "map_d1" if double else "map_f1")(ctypes.cast(f, P), P(x.ctypes.data), P(y.ctypes.data),
                                                       ctypes.c_long(x.size))
        return y
    flat = y.reshape(-1)
    for i, v in enumerate(x.reshape(-1).tolist()):
        flat[i] = f(v)
    return y


def expf(x: np.ndarray) -> np.ndarray | None:
    """libimf expf elementwise (float32), or None without the Intel runtime."""
    return _map("expf", x, False)


def exp(x: np.ndarray) -> np.ndarray | None:
    """libimf exp elementwise (float64), or None without the Intel runtime."""
    return _map("exp", x, True)


def logf(x: np.ndarray) -> np.ndarray | None:
    """libimf logf elementwise (float32), or None without the Intel runtime."""
    return _map("logf", x, False)


def address(name: str, double: bool = False):
    """Address of the libimf function (for the C kernels), or None."""
    f = _fn(name, double)
    return None if f is None else ctypes.cast(f, ctypes.c_void_p).value


def powf(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """libimf powf elementwise over broadcast float32 arrays, or None without the Intel runtime."""
    f = _fn("powf", False, 2)
    if f is None:
        return None
    x, y = (np.ascontiguousarray(v) for v in np.broadcast_arrays(np.asarray(x, np.float32), np.asarray(y, np.float32)))
    out = np.empty_like(x)
    from .native import load_extra
    lib = load_extra("imfmap")
    if lib is not None and x.size:
        P = ctypes.c_void_p
        lib.map_f2(ctypes.cast(f, P), P(x.ctypes.data), P(y.ctypes.data), P(out.ctypes.data), ctypes.c_long(x.size))
        return out
    flat = out.reshape(-1)
    for i, (a, b) in enumerate(zip(x.reshape(-1).tolist(), y.reshape(-1).tolist())):
        flat[i] = f(a, b)
    return out


def map1(name: str, *a):
    """libimf `name` (expf, logf, powf; float32) on numpy arrays, or None."""
    return {"expf": expf, "logf": logf, "powf": powf}[name](*a)
