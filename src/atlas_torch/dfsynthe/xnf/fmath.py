"""Numeric helpers with the Fortran (ifx -O0) semantics needed by the xnfdf port.

- F(x): a default-kind REAL literal widened to double (same as fnum.f32).
- fsqrt: correctly rounded square root (this torch build's CPU sqrt kernel is not correctly rounded;
  numpy's is, and so is the Fortran's sqrtsd).
- powi: X**N for an INTEGER N. ifx (-O0) calls libimf's __powr8i4, which is correctly rounded in
  >99.7% of cases (it is NOT repeated multiplication: plain binary exponentiation agrees with it only
  10-75% of the time for N>=3). powi returns the correctly rounded power: exact rational arithmetic for
  Python scalars, double-double binary exponentiation for tensors.
- powi_mul: the plain multiplication sequence (llvm.powi / __powidf2), kept for reference.
- r4: REAL*8 -> REAL*4 assignment (round to nearest).
Single-precision transcendental functions (LOG10 of a REAL*4) are evaluated in double and rounded,
which is what libimf's float entry points return in all but pathological cases.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import torch

from ...fnum import f32 as F, fint, nint  # noqa: F401  (re-exported)

DT = torch.float64


def fsqrt(x):
    if isinstance(x, torch.Tensor):
        if x.device.type == "cpu":
            return torch.from_numpy(np.sqrt(x.detach().numpy()))
        return torch.sqrt(x)
    return math.sqrt(x)


def rdiv(a, t):
    """a / t for a Python scalar a and tensor t with true IEEE division (torch's scalar.__rtruediv__
    computes reciprocal(t) * a, which rounds twice)."""
    if isinstance(t, torch.Tensor):
        return torch.full_like(t, a) / t
    return a / t


def powi(x, n: int):
    """Correctly rounded x**n for an integer n (see module docstring)."""
    n = int(n)
    if not isinstance(x, torch.Tensor):
        x = float(x)
        if n == 0:
            return 1.0
        if x == 0.0 or not math.isfinite(x):
            return x ** n
        try:
            return float(Fraction(x) ** n)
        except OverflowError:
            return math.copysign(math.inf, x) if (n > 0 and n % 2) else math.inf
    if n == 0:
        return torch.ones_like(x)
    m = abs(n)
    rh, rl = None, None
    ah, al = x, torch.zeros_like(x)
    while True:
        if m & 1:
            if rh is None:
                rh, rl = ah, al
            else:
                rh, rl = _dd_mul(rh, rl, ah, al)
        m //= 2
        if m == 0:
            break
        ah, al = _dd_mul(ah, al, ah, al)
    if n > 0:
        return rh
    q = torch.full_like(rh, 1.0) / rh
    p, e = _two_prod(q, rh)
    corr = (((1.0 - p) - e) - q * rl) * q
    return q + corr


_SPLIT = 134217729.0   # 2**27 + 1 (Veltkamp)


def _split(a):
    c = a * _SPLIT
    hi = c - (c - a)
    return hi, a - hi


def _two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def _dd_mul(xh, xl, yh, yl):
    p, e = _two_prod(xh, yh)
    e = e + (xh * yl + xl * yh)
    s = p + e
    return s, e - (s - p)


def powi_mul(x, n: int):
    """x**n by binary exponentiation (llvm.powi / __powidf2 semantics)."""
    if n == 0:
        return x * 0 + 1.0 if isinstance(x, torch.Tensor) else 1.0
    recip = n < 0
    b = -n if recip else n
    r = None
    a = x
    while True:
        if b & 1:
            r = a if r is None else r * a
        b //= 2
        if b == 0:
            break
        a = a * a
    return rdiv(1.0, r) if recip else r


def r4(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)


def log10_r4(x32: torch.Tensor) -> torch.Tensor:
    return torch.log10(x32.to(DT)).to(torch.float32)


LN10_R4 = F(2.30258509299405)     # 2.30258509299405E0 (READIN EXP10 statement function)
