"""DATA tables of the atlas7v.for opacity routines, as float64 tensors of the decimal values in the source."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ....tables import table
from ...constants import decimal32


@lru_cache(maxsize=None)
def _array(unit: str, var: str) -> np.ndarray:
    a = np.asarray(table("atlas7v.for", unit, var))
    if a.dtype == np.float32:
        return decimal32(a)
    a = a.astype(np.float64)
    single = a == a.astype(np.float32).astype(np.float64)        # REAL*4 literals stored in REAL*8 arrays
    return np.where(single, decimal32(a.astype(np.float32)), a)


def fortran_data(unit: str, var: str, device=None) -> torch.Tensor:
    """DATA array `var` of routine `unit` (Fortran index order kept: a[i, j] = VAR(i+1, j+1))."""
    return torch.as_tensor(_array(unit.lower(), var.lower()), dtype=torch.float64, device=device)
