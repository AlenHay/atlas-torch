"""Elementary functions with the Fortran's libm semantics.

The oracle libraries (gfortran) call glibc `exp`, `log`, `log10`, `pow`, `log10f`. torch's vectorised CPU
kernels (SLEEF) differ from glibc by 1 ulp in ~0.5% of arguments. That is inside the porting tolerance, but
a bit-exact port makes oracle tests sharp (any mismatch is a porting error, not ulp noise) and keeps
iterative solvers (NMOLEC's convergence test) on exactly the Fortran's path.

EXACT = True (default): CPU float64 tensors go through Python's `math` (= the platform C libm, glibc on
Linux) element by element; float32 functions go through libm's *f entry points via ctypes.
EXACT = False, or tensors on other devices: native torch kernels (device-agnostic, ulp-level differences).

Scalars (python floats) always use `math` (glibc).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np
import torch

EXACT = True

_libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
for _n in ("log10f", "expf", "logf", "sqrtf"):
    getattr(_libm, _n).restype = ctypes.c_float
    getattr(_libm, _n).argtypes = [ctypes.c_float]
_libm.powf.restype = ctypes.c_float
_libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]

for _n in ("exp", "log", "log10", "sqrt", "pow"):
    getattr(_libm, _n).restype = ctypes.c_double
    getattr(_libm, _n).argtypes = [ctypes.c_double] * (2 if _n == "pow" else 1)


def _c_semantics(name):
    """math.<name>, but with the C library's IEEE results where Python raises instead (exp/pow overflow -> inf,
    log(0) -> -inf, log/sqrt of a negative -> nan, pow(0, -y) -> inf): the Fortran calls libm directly and
    carries on (e.g. NMOLEC's EQUILJ overflows on layers above 10000 K, which it then discards)."""
    f, c = getattr(math, name), getattr(_libm, name)

    def g(*a):
        try:
            return f(*a)
        except (OverflowError, ValueError):
            return c(*map(float, a))
    g.__name__ = name
    return g


_math = {name: _c_semantics(name) for name in ("exp", "log", "log10", "sqrt", "pow")}
_u = {name: np.frompyfunc(_math[name], 1, 1) for name in ("exp", "log", "log10", "sqrt")}
_pow = np.frompyfunc(_math["pow"], 2, 1)
_uf = {name: np.frompyfunc(getattr(_libm, name + "f"), 1, 1) for name in ("log10", "exp", "log")}


def _exact(x: torch.Tensor) -> bool:
    return EXACT and x.device.type == "cpu"


def _apply(name, x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float32:
            if _exact(x):
                with np.errstate(all="ignore"):
                    r = _uf[name](x.detach().numpy()).astype(np.float32)
                return torch.from_numpy(np.asarray(r, dtype=np.float32).reshape(x.shape))
            return getattr(torch, name)(x)
        if _exact(x):
            with np.errstate(all="ignore"):           # inf/nan results are IEEE, as in the Fortran
                r = _u[name](x.detach().numpy()).astype(np.float64)
            return torch.from_numpy(np.asarray(r, dtype=np.float64).reshape(x.shape))
        return getattr(torch, name)(x)
    if isinstance(x, np.float32):
        return np.float32(getattr(_libm, name + "f")(float(x)))
    return _math[name](x)


def exp(x):
    return _apply("exp", x)


def log(x):
    return _apply("log", x)


def log10(x):
    return _apply("log10", x)


def sqrt(x):
    """IEEE sqrt (correctly rounded, as the Fortran's SQRT/sqrtsd). torch's CPU sqrt kernel on this build is
    not correctly rounded (~1% of results 1 ulp off), so the exact path uses numpy."""
    if isinstance(x, torch.Tensor):
        if _exact(x):
            return torch.from_numpy(np.sqrt(x.detach().numpy()))
        return torch.sqrt(x)
    if isinstance(x, np.float32):
        return np.float32(np.sqrt(x))
    return _math["sqrt"](x)


def pow(x, y):
    """x**y with a REAL exponent (Fortran calls libm pow / powf)."""
    xt, yt = isinstance(x, torch.Tensor), isinstance(y, torch.Tensor)
    if not (xt or yt):
        if isinstance(x, np.float32) and isinstance(y, np.float32):
            return np.float32(_libm.powf(float(x), float(y)))
        return _math["pow"](x, y)
    ref = x if xt else y
    if ref.dtype == torch.float32:
        return torch.pow(x, y)
    if _exact(ref):
        xa = x.detach().numpy() if xt else np.float64(x)
        ya = y.detach().numpy() if yt else np.float64(y)
        with np.errstate(all="ignore"):
            r = _pow(xa, ya).astype(np.float64)
        return torch.from_numpy(np.asarray(r, dtype=np.float64).reshape(np.broadcast_shapes(np.shape(xa), np.shape(ya))))
    return torch.pow(x, y)


# gcc's powi addition chains (tree-ssa-math-opts.c powi_table) used by gfortran for constant integer
# exponents x**n: x**n = x**powi_table[n] * x**(n - powi_table[n]), with memoised subpowers.
_POWI_TABLE = [0, 1, 1, 2, 2, 3, 3, 4, 4, 6, 5, 6, 6, 10, 7, 9,
               8, 16, 9, 16, 10, 12, 11, 13, 12, 17, 13, 18, 14, 24, 15, 26,
               16, 17, 17, 19, 18, 33, 19, 26, 20, 25, 21, 40, 22, 27, 23, 44,
               24, 32, 25, 34, 26, 29, 27, 44, 28, 31, 29, 34, 30, 60, 31, 36, 32, 64]


def ipow_const(x, n: int):
    """x**n for a compile-time constant integer n, as gfortran expands it (powi addition chain)."""
    if n == 0:
        return x * 0 + 1 if isinstance(x, torch.Tensor) else 1.0
    if n < 0:
        return 1.0 / ipow_const(x, -n)
    cache = {1: x}

    def rec(k):
        if k in cache:
            return cache[k]
        if k < len(_POWI_TABLE):
            a = _POWI_TABLE[k]
            r = rec(a) * rec(k - a)
        else:  # binary method for large exponents
            r = rec(k // 2) * rec(k // 2) if k % 2 == 0 else rec(k - 1) * x
        cache[k] = r
        return r
    return rec(n)


def ipow_var(x, m):
    """x**m for a run-time INTEGER m: libgcc __powidf2 (square-and-multiply)."""
    if isinstance(m, torch.Tensor):
        n = m.abs()
        y = torch.where(n % 2 == 1, x, torch.ones_like(x))
        xx = x
        n = n // 2
        while bool((n > 0).any()):
            xx = xx * xx
            y = torch.where((n % 2 == 1), y * xx, y)
            n = n // 2
        return torch.where(m < 0, 1.0 / y, y)
    n = abs(int(m))
    y = x if n % 2 else (x * 0 + 1 if isinstance(x, torch.Tensor) else 1.0)
    xx = x
    n >>= 1
    while n:
        xx = xx * xx
        if n % 2:
            y = y * xx
        n >>= 1
    return 1.0 / y if m < 0 else y


def rdiv(c, t):
    """c / t for a python scalar c and a tensor t, correctly rounded.

    torch implements `scalar / tensor` as scalar * reciprocal(tensor) (two roundings, ~25% of results
    differ by an ulp); a 0-d tensor numerator takes the IEEE division path.
    """
    if isinstance(t, torch.Tensor):
        return torch.as_tensor(c, dtype=t.dtype, device=t.device) / t
    return c / t
