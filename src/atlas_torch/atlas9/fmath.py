"""Elementary functions with the Fortran's libm semantics.

gfortran -O0 calls glibc's exp/log/log10/pow/expf (and sqrtsd, correctly rounded). torch's CPU kernels use
their own vectorised implementations (Sleef / AVX512) whose results differ from glibc in the last ulp for a
fraction of the inputs -- including sqrt, which is not correctly rounded there. For bit-for-bit agreement the
port evaluates these functions through glibc on CPU tensors (Python's math module and ctypes' libm.expf, which
are the very same glibc entry points the Fortran binary resolves). On other devices the native torch functions
are used (documented accuracy difference: last-ulp only).

Plain +,-,*,/ , abs, min/max are IEEE in torch and are used directly everywhere.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np
import torch

from . import native as _native

_libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
_libm.expf.restype = ctypes.c_float
_libm.expf.argtypes = [ctypes.c_float]

for _n in ("exp", "log", "log10"):
    getattr(_libm, _n).restype = ctypes.c_double
    getattr(_libm, _n).argtypes = [ctypes.c_double]
_libm.pow.restype = ctypes.c_double
_libm.pow.argtypes = [ctypes.c_double, ctypes.c_double]


def _safe1(name):
    """math.<name>, with the C library's IEEE result where Python raises (overflow -> inf, log(0) -> -inf,
    log of a negative -> nan): the Fortran calls libm directly."""
    f, c = getattr(math, name), getattr(_libm, name)

    def g(v):
        try:
            return f(v)
        except (OverflowError, ValueError):
            return c(float(v))
    return g


def _pow1(u, v):
    try:
        return math.pow(u, v)
    except (OverflowError, ValueError):
        return _libm.pow(float(u), float(v))


_u_exp = np.frompyfunc(_safe1("exp"), 1, 1)
_u_log = np.frompyfunc(_safe1("log"), 1, 1)
_u_log10 = np.frompyfunc(_safe1("log10"), 1, 1)
_u_pow = np.frompyfunc(_pow1, 2, 1)
_u_expf = np.frompyfunc(_libm.expf, 1, 1)


def _cpu(x: torch.Tensor) -> bool:
    return x.device.type == "cpu"


def _apply(u, x: torch.Tensor, dtype=np.float64, name=None) -> torch.Tensor:
    a = x.detach().numpy()
    if a.size == 0:
        return x.clone()
    if name is not None:
        r = _native.vmath(name, np.asarray(a, dtype=dtype))   # same glibc calls in C (optional helper)
        if r is not None:
            return torch.from_numpy(r)
    return torch.from_numpy(np.asarray(u(a), dtype=dtype).reshape(a.shape))


def exp(x: torch.Tensor) -> torch.Tensor:
    return _apply(_u_exp, x, name="exp") if _cpu(x) else torch.exp(x)


def log(x: torch.Tensor) -> torch.Tensor:
    return _apply(_u_log, x, name="log") if _cpu(x) else torch.log(x)


def log10(x: torch.Tensor) -> torch.Tensor:
    return _apply(_u_log10, x, name="log10") if _cpu(x) else torch.log10(x)


def sqrt(x: torch.Tensor) -> torch.Tensor:
    if _cpu(x):
        with np.errstate(invalid="ignore"):
            return torch.from_numpy(np.sqrt(x.detach().numpy()))   # IEEE sqrtpd: correctly rounded
    return torch.sqrt(x)


def pow(x: torch.Tensor, y) -> torch.Tensor:
    """REAL*8 ** REAL*8 (glibc pow). Integer powers in the Fortran are products (gfortran powi) -- write them
    as products, not with this function."""
    if _cpu(x):
        yy = y.detach().numpy() if isinstance(y, torch.Tensor) else np.float64(y)
        a = x.detach().numpy()
        r = _native.vmath("pow", np.asarray(a, dtype=np.float64), np.asarray(yy, dtype=np.float64))
        if r is not None:
            return torch.from_numpy(r)
        return torch.from_numpy(np.asarray(_u_pow(a, yy), dtype=np.float64))
    return torch.pow(x, y)


def expf(x: torch.Tensor) -> torch.Tensor:
    """REAL*4 EXP (glibc expf) of a float32 tensor."""
    if _cpu(x):
        return _apply(_u_expf, x.to(torch.float32), dtype=np.float32, name="exp")
    return torch.exp(x.to(torch.float32))


# scalar helpers (Python floats), same libm
sexp, slog, slog10, spow = math.exp, math.log, math.log10, math.pow


def rdiv(c, x: torch.Tensor) -> torch.Tensor:
    """c / x with IEEE division. NOTE: torch evaluates `python_scalar / tensor` as x.reciprocal() * c, which is
    not correctly rounded; every scalar-over-tensor quotient of the port goes through this function."""
    return torch.div(torch.as_tensor(c, dtype=x.dtype, device=x.device), x)
