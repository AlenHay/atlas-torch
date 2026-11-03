"""table(source, unit, var): DATA-initialised Fortran table as numpy.

Uses atlas_torch.kurucz_data (parser over the fetched Fortran source, no Fortran build needed) when
available; otherwise falls back to the compiled oracle library (test environments only).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def table(source: str, unit: str, var: str) -> np.ndarray:
    try:
        from .kurucz_data import table as _t
        return _t(source, unit, var)
    except (ImportError, AttributeError):
        from .testing.so_tables import so_table
        return so_table(source, unit, var)
