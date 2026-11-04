"""spectrum.bin (SPECTRV fort.7) as BasicATLAS reads it (atlas.py load_binary_spectrum)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

N_ANGLES, N_EDGES = 20, 377


def _record(payload: bytes) -> bytes:
    n = struct.pack("<i", len(payload))
    return n + payload + n


def write_spectrum_bin(path: str | Path, *, teff: float, logg: float, title: str, wbegin: float, resolution: float,
                       line: np.ndarray, continuum: np.ndarray, mu: np.ndarray | None = None, vacuum: bool = True,
                       edges_nm: np.ndarray | None = None) -> None:
    """Header record (TEFF, GLOG, TITLE*74 with 'V'/'A' in column 1, WBEGIN, RESOLU, NUMNU, IFSURF, NMU,
    ANGLE(20), NEDGE, WLEDGE(377)); one record per wavelength (line(1..NMU), continuum(1..NMU)); a final
    NLINES = 0 record (spectrv.for, WRITE(7)). line, continuum: (W,) Eddington flux H_nu (IFSURF = 1) or
    (W, NMU) intensity at the angles mu (IFSURF = 2), erg s^-1 cm^-2 Hz^-1 sr^-1."""
    line = np.asarray(line, dtype="<f8")
    cont = np.asarray(continuum, dtype="<f8")
    surf = 1 if line.ndim == 1 else 2
    q = np.concatenate([line.reshape(len(line), -1), cont.reshape(len(cont), -1)], 1)
    nmu = q.shape[1] // 2
    angle = np.zeros(N_ANGLES)
    if surf == 2:
        angle[:nmu] = np.asarray(mu, dtype=np.float64)
    t = ("V" if vacuum else "A") + title.ljust(74)[1:74]
    e = np.zeros(N_EDGES)
    ne = 0 if edges_nm is None else min(len(edges_nm), N_EDGES)
    if ne:
        e[:ne] = np.asarray(edges_nm, dtype=np.float64)[:ne]
    head = (struct.pack("<dd", teff, logg) + t.encode("latin-1") + struct.pack("<ddiii", wbegin, resolution, len(q),
            surf, nmu) + angle.astype("<f8").tobytes() + struct.pack("<i", ne) + e.astype("<f8").tobytes())
    n = q.shape[1] * 8
    body = np.empty((len(q), n + 8), dtype=np.uint8)
    mark = np.frombuffer(struct.pack("<i", n), dtype=np.uint8)
    body[:, :4] = mark
    body[:, 4:4 + n] = np.ascontiguousarray(q).view(np.uint8).reshape(len(q), n)
    body[:, 4 + n:] = mark
    with open(path, "wb") as f:
        f.write(_record(head))
        f.write(body.tobytes())
        f.write(_record(struct.pack("<i", 0)))
