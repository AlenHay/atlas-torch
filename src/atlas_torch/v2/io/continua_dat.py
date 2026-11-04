"""continua.dat (XNFPELSYN unit 17): continuum edges, free format, read until a 0 (xnfpelsyn.for loop 11).
A value is a wavelength in nm if |v| < 1e6, a frequency in Hz if |v| < 1e25, else a wavenumber in cm^-1
times 1e25; negative values are filler points, not edges."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[DE][-+]?\d+)?", re.I)
C_NM = 2.99792458e17      # c in nm s^-1


def default_path() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "BasicATLAS" / "data" / "synthe_files" \
        / "continua.dat"


def read_continua_dat(path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Edges in file order converted to wavelength (nm), frequency (Hz) and wavenumber (cm^-1), with the sign
    of the input; then sorted by |wavelength| (stable), as XNFPELSYN orders them."""
    vals = []
    for line in Path(path or default_path()).read_text(encoding="latin-1").splitlines():
        for tok in _NUM.findall(line):
            v = float(tok.upper().replace("D", "E"))
            if v == 0.0:
                break
            vals.append(v)
        else:
            continue
        break
    wl, fr, cm = [], [], []
    for v in vals:
        if abs(v) < 1e6:
            w = v
        elif abs(v) < 1e25:
            w = C_NM / v
        else:
            w = 1e7 / (v / 1e25)
        wl.append(w)
        fr.append(C_NM / w)
        cm.append(1e7 / w)
    order = np.argsort(np.abs(wl), kind="stable")
    return {k: np.asarray(v)[order] for k, v in (("wavelength_nm", wl), ("frequency", fr), ("wavenumber", cm))}


def frequency_set(edges: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """XNFPELSYN's continuum frequencies FREQSET (loop 18), Hz: per interval between consecutive edges, the lower
    edge / (1 + 1e-7), the frequency of the mean wavelength, and the upper edge * (1 + 1e-7)."""
    e = read_continua_dat() if edges is None else edges
    fr, wl = np.abs(e["frequency"]), np.abs(e["wavelength_nm"])
    out = np.empty(3 * (len(fr) - 1))
    out[0::3] = fr[:-1] / (1.0 + 1e-7)
    out[1::3] = C_NM / (0.5 * (wl[:-1] + wl[1:]))
    out[2::3] = fr[1:] * (1.0 + 1e-7)
    return out
