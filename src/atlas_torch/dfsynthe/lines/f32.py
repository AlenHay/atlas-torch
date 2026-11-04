"""REAL*4 arithmetic helpers.

The Fortran (ifx, no FMA, SSE) evaluates REAL*4 expressions in single precision with IEEE rounding after each
operation; torch float32 +,-,*,/,sqrt are the same IEEE operations.  libm single-precision transcendentals
(expf, logf, log10f, powf) are (nearly always) correctly rounded; we evaluate them in float64 and round once,
which gives the correctly rounded result except in astronomically rare double-rounding cases.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
import torch

F32 = torch.float32
F64 = torch.float64
I64 = torch.int64


@lru_cache(maxsize=None)
def lit(s: str) -> np.float32:
    """Default-REAL Fortran literal -> exactly rounded float32 (round to nearest, ties to even)."""
    x = Fraction(s.lower().replace("d", "e"))
    c = np.float32(float(x))
    best, bd = c, abs(Fraction(float(c)) - x)
    for d in (np.nextafter(c, np.float32(np.inf)), np.nextafter(c, np.float32(-np.inf))):
        e = abs(Fraction(float(d)) - x)
        if e < bd or (e == bd and (d.view(np.int32) & 1) == 0):
            best, bd = d, e
    return np.float32(best)


def lf(s: str) -> float:
    """A default-REAL literal used inside a REAL*8 expression (widened single)."""
    return float(lit(s))


def expf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x.to(F64)).to(F32)


def logf(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.to(F64)).to(F32)


def log10f(x: torch.Tensor) -> torch.Tensor:
    return torch.log10(x.to(F64)).to(F32)


def powf(a, b) -> torch.Tensor:
    a = a.to(F64) if isinstance(a, torch.Tensor) else float(a)
    b = b.to(F64) if isinstance(b, torch.Tensor) else float(b)
    return torch.pow(a, b).to(F32) if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor) \
        else torch.tensor(np.float32(a ** b))


def sqrtf(x: torch.Tensor) -> torch.Tensor:
    """IEEE sqrtf.  torch.sqrt on CPU float32 is NOT correctly rounded (~15% of results off by one ulp);
    the float64 square root rounded once to float32 is exact."""
    return torch.sqrt(x.to(F64)).to(x.dtype)


def rdiv(c, t: torch.Tensor) -> torch.Tensor:
    """c / t with a single IEEE division.  NB: torch's `python_scalar / tensor` is computed as
    `scalar * reciprocal(tensor)` (two roundings) and must never be used in this port."""
    return torch.tensor(float(c), dtype=t.dtype, device=t.device) / t


def tof32(x: torch.Tensor) -> torch.Tensor:
    return x.to(F32)


def nint(x: torch.Tensor) -> torch.Tensor:
    """Fortran NINT (half away from zero), exact for float32 input."""
    d = x.to(F64)
    return (torch.sign(d) * torch.floor(d.abs() + 0.5)).to(I64)


def ifix(x: torch.Tensor) -> torch.Tensor:
    """Fortran INT/IFIX: truncation toward zero."""
    return torch.trunc(x.to(F64)).to(I64)


def powidf2(a: float, n: torch.Tensor) -> torch.Tensor:
    """REAL*8 ** INTEGER as compiled (compiler-rt/libgcc __powidf2: binary powering, same product order)."""
    n = n.to(I64)
    b = n.abs()
    r = torch.ones(n.shape, dtype=F64, device=n.device)
    aa = torch.full(n.shape, float(a), dtype=F64, device=n.device)
    while bool((b != 0).any()):
        r = torch.where((b & 1) == 1, r * aa, r)
        b = b >> 1
        aa = aa * aa
    return torch.where(n < 0, 1.0 / r, r)


def powidf2_np(a: float, n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.int64)
    b = np.abs(n)
    r = np.ones(n.shape)
    aa = np.full(n.shape, float(a))
    while (b != 0).any():
        r = np.where((b & 1) == 1, r * aa, r)
        b = b >> 1
        aa = aa * aa
    return np.where(n < 0, 1.0 / r, r)


# --------------------------------------------------------------------------------------------- flavours
class Flavor:
    """Compiler/runtime-library semantics that differ between the Fortran builds.

    name='stock' (ifx -O2, BasicATLAS): REAL*8**INTEGER via compiler-rt __powidf2 (double binary powering),
    REAL*4**3/**4 expanded into multiplications.  name='O0' (ifx -O0 debug/oracle builds): Intel runtime
    __powr8i4 (binary powering in x87 80-bit extended, rounded once) and __powr4i4 (exact power rounded once).
    `libm` supplies the scalar transcendentals; the default evaluates in float64 and rounds (accurate);
    tests may install Intel libimf through ctypes (IntelLibm) to reproduce the Fortran bit-for-bit.
    """

    def __init__(self, name: str = "stock", libm=None, fastmath: bool | None = None):
        if name not in ("stock", "O0"):
            raise ValueError(name)
        self.name = name
        self.libm = libm
        # ifx -fp-model fast code generation of the stock binary (HPROF4's Lyman-alpha NaN term, reciprocal
        # rewrites, rcpps divisions, SVML table exps; list in dfsynthe.py).  Default: on for 'stock'.
        # Flavor('stock', fastmath=False) reproduces the ifx -fp-model precise build.
        self.fastmath = (name == "stock") if fastmath is None else bool(fastmath)

    # ---- integer powers
    def powi8_np(self, a: float, n: np.ndarray) -> np.ndarray:
        if self.name == "stock":
            return powidf2_np(a, n)
        return powr8i4_np(a, n)

    def powi4(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if self.name == "O0":
            return torch.pow(x.to(F64), k).to(F32)
        if k == 2:
            return x * x
        if k == 3:
            return x * (x * x)
        if k == 4:
            x2 = x * x
            return x2 * x2
        raise ValueError(k)

    # ---- transcendentals (float32 in/out unless noted)
    def _call(self, fn, *args):
        if self.libm is not None and hasattr(self.libm, fn):
            return getattr(self.libm, fn)(*args)
        return None

    def expf(self, x):
        r = self._call("expf", x)
        return expf(x) if r is None else r

    def logf(self, x):
        r = self._call("logf", x)
        return logf(x) if r is None else r

    def log10f(self, x):
        r = self._call("log10f", x)
        return log10f(x) if r is None else r

    def powf(self, a, b):
        r = self._call("powf", a, b)
        return powf(a, b) if r is None else r

    def exp(self, x):
        r = self._call("exp", x)
        return torch.exp(x) if r is None else r

    def log(self, x):
        r = self._call("log", x)
        return torch.log(x) if r is None else r

    def pow(self, a, b):
        r = self._call("pow", a, b)
        return torch.pow(a, b) if r is None else r


def powr8i4_np(a: float, n: np.ndarray) -> np.ndarray:
    """Intel __powr8i4: binary powering in x87 extended precision, rounded to double (CPU numpy longdouble)."""
    if np.finfo(np.longdouble).nmant < 63:
        raise RuntimeError("needs x87 80-bit long double")
    n = np.asarray(n, dtype=np.int64)
    b = np.abs(n)
    r = np.ones(n.shape, dtype=np.longdouble)
    aa = np.full(n.shape, a, dtype=np.longdouble)
    while (b != 0).any():
        r = np.where((b & 1) == 1, r * aa, r)
        b = b >> 1
        aa = aa * aa
    r = np.where(n < 0, np.longdouble(1) / r, r)
    return r.astype(np.float64)


class CtypesLibm:
    """Scalar libm transcendentals of a given shared library through ctypes (validation only; slow, CPU).

    The ifx executables bind expf/logf/powf/exp/log/pow to Intel libimf; the oracle library loaded into a
    Python process binds them to glibc libm (already in the global scope)."""

    def __init__(self, path: str):
        import ctypes
        lib = ctypes.CDLL(path)
        self._f = {}
        for nm, ty, na in (("expf", ctypes.c_float, 1), ("logf", ctypes.c_float, 1), ("log10f", ctypes.c_float, 1),
                           ("powf", ctypes.c_float, 2), ("exp", ctypes.c_double, 1), ("log", ctypes.c_double, 1),
                           ("pow", ctypes.c_double, 2)):
            f = getattr(lib, nm)
            f.restype = ty
            f.argtypes = [ty] * na
            self._f[nm] = (f, np.float32 if ty is ctypes.c_float else np.float64)

    def _apply(self, nm, *args):
        f, dt = self._f[nm]
        ts = [a for a in args if isinstance(a, torch.Tensor)]
        dev = ts[0].device if ts else torch.device("cpu")
        arrs = [a.detach().cpu().numpy().astype(dt) if isinstance(a, torch.Tensor) else np.asarray(a, dt)
                for a in args]
        arrs = np.broadcast_arrays(*arrs)
        flat = [a.ravel().tolist() for a in arrs]
        out = np.fromiter((f(*v) for v in zip(*flat)), dtype=dt, count=arrs[0].size).reshape(arrs[0].shape)
        return torch.from_numpy(out).to(dev)

    def expf(self, x): return self._apply("expf", x)
    def logf(self, x): return self._apply("logf", x)
    def log10f(self, x): return self._apply("log10f", x)
    def powf(self, a, b): return self._apply("powf", a, b)
    def exp(self, x): return self._apply("exp", x)
    def log(self, x): return self._apply("log", x)
    def pow(self, a, b): return self._apply("pow", a, b)


class IntelLibm(CtypesLibm):
    def __init__(self, libdir: str = "/opt/intel/oneapi/compiler/latest/lib"):
        import os
        super().__init__(os.path.join(libdir, "libimf.so"))


class GlibcLibm(CtypesLibm):
    def __init__(self):
        super().__init__("libm.so.6")
