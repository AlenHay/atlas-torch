"""SEPARATEDF (separatedf.for): merge the 57 per-temperature sortp files into p00big / p00lit ODF files.

For inu=1..1540, it=1..57: read 25 lines (ip) of unit it+9, then write one unformatted record
((ISTEPS(ip,it,istep), istep=1,12), ip=1,25) as INTEGER*2 to fort.2 (inu<=328, BIG) or fort.3 (LITTLE).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .odf_io import KW, NBIG, NFREQ, parse_sortp, write_seq_records


def separatedf_arrays(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """steps int16 [57, 1540, 25, 12] (it, nu, ip, istep) -> (big payloads, lit payloads) uint8 records."""
    steps = np.asarray(steps, dtype="<i2")
    nt = steps.shape[0]
    rec = steps.transpose(1, 0, 2, 3).reshape(NFREQ, nt, KW * 12)          # (inu, it) records
    big = rec[:NBIG].reshape(NBIG * nt, -1).view(np.uint8)
    lit = rec[NBIG:].reshape((NFREQ - NBIG) * nt, -1).view(np.uint8)
    return big, lit


def read_sortp_steps(paths: Sequence[str | Path]) -> np.ndarray:
    """57 dfsortp files -> int16 [57, 1540, 25, 12]."""
    out = []
    for p in paths:
        _, st = parse_sortp(Path(p).read_bytes())
        out.append(st.reshape(NFREQ, KW, 12))
    return np.stack(out)


def separatedf(sortp_paths: Sequence[str | Path], big_path: str | Path, lit_path: str | Path) -> None:
    big, lit = separatedf_arrays(read_sortp_steps(sortp_paths))
    write_seq_records(big_path, big)
    write_seq_records(lit_path, lit)
