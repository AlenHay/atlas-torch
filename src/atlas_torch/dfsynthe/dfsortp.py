"""DFSORTP (dfsortp.for): reorder one temperature's ODF steps from (layer, nu) to (nu, layer) order, ascii.

The Fortran reads fort.1 (1540*25 fixed 32-byte records, layer J outer) and writes fort.2 with
'(1x,i10,1x,12i8)': for nu=1..1540, for ip=1..25: IDFOUT(1), ISTEPS(1..12).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .odf_io import KW, NFREQ, format_sortp, read_odf_bin


def dfsortp_arrays(idfout: np.ndarray, isteps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(idfout [25*1540], isteps [25*1540,12]) in file order -> same in (nu, ip) order."""
    idf = np.asarray(idfout)[: KW * NFREQ].reshape(KW, NFREQ).T.reshape(-1)
    st = np.asarray(isteps)[: KW * NFREQ].reshape(KW, NFREQ, 12).transpose(1, 0, 2).reshape(-1, 12)
    return idf, st


def dfsortp(fort1: str | Path, fort2: str | Path) -> None:
    rec = read_odf_bin(fort1)
    idf, st = dfsortp_arrays(rec["idfout"], rec["isteps"])
    Path(fort2).write_bytes(format_sortp(idf, st))
