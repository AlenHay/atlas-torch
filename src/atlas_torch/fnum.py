"""Numeric helpers reproducing Fortran semantics bit-for-bit.

- f32(x): a default-kind REAL literal (e.g. `.77`) used in REAL*8 arithmetic is single precision widened
  to double; write it as f32(0.77) so the port uses the same value (0.7699999809265137).
- fint(x): Fortran INT() / assignment of real to integer = truncation toward zero.
- nint(x): Fortran NINT = round half away from zero.
"""
from __future__ import annotations

import numpy as np
import torch


def f32(x: float) -> float:
    return float(np.float32(x))


def fint(x):
    if isinstance(x, torch.Tensor):
        return torch.trunc(x).to(torch.int64)
    return int(x)


def nint(x):
    if isinstance(x, torch.Tensor):
        return (torch.sign(x) * torch.floor(torch.abs(x) + 0.5)).to(torch.int64)
    return int(np.sign(x) * np.floor(abs(x) + 0.5))
