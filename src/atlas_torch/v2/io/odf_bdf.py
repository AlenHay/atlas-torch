"""ODF step files p00big{v}.bdf / p00lit{v}.bdf (separatedf.for output): sequential unformatted, one record per
(interval, temperature) of INTEGER*2 KAP(25 pressures, 12 steps) = 1000 log10 kappa, steps in increasing opacity."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...io.formats import read_bdf as _read_records
from ...io.fortran import write_record


def read_bdf(path, n_t: int = 57, n_p: int = 25) -> np.ndarray:
    """int16 (I, n_t, n_p, 12)."""
    a = _read_records(path, nt=n_t)
    if a.shape[2] != n_p:
        raise ValueError(f"{path}: {a.shape[2]} pressures, expected {n_p}")
    return a


def write_bdf(path, steps: np.ndarray) -> None:
    """steps int16 (I, n_t, n_p, 12)."""
    s = np.ascontiguousarray(np.asarray(steps), dtype="<i2")
    if s.ndim != 4:
        raise ValueError("steps must be (I, n_t, n_p, 12)")
    rec = s.reshape(s.shape[0] * s.shape[1], -1)
    with open(Path(path), "wb") as f:
        for r in rec:
            write_record(f, r.tobytes())
