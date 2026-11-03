"""Port of KAPREADTS: merge the five kapk{v}.dat KAPPA9 tables (v = 0,1,2,4,8 km/s) into kappa.ros.

Fortran semantics reproduced: values are read with FORMAT 3PF5.0 into REAL*4 (integer milli-dex / 1000,
rounded to float32), the temperature/pressure axes are REAL*4 DATA, columns log Ne / log Na / log rho are
never assigned (static zero), IPloCM-1 is 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.formats import read_kapk

TABT = np.array([3.30, 3.320, 3.340, 3.360, 3.380, 3.400, 3.420, 3.440, 3.460, 3.480, 3.500, 3.520, 3.540,
                 3.560, 3.580, 3.600, 3.620, 3.640, 3.660, 3.680, 3.700, 3.730, 3.760, 3.790, 3.820, 3.850,
                 3.880, 3.910, 3.940, 3.970, 4.000, 4.050, 4.100, 4.150, 4.200, 4.250, 4.300, 4.350, 4.400,
                 4.450, 4.500, 4.550, 4.600, 4.650, 4.700, 4.750, 4.800, 4.850, 4.900, 4.950, 5.000, 5.050,
                 5.100, 5.150, 5.200, 5.250, 5.300], dtype=np.float32)
TABP = np.array([-4.000, -3.500, -3.000, -2.500, -2.000, -1.500, -1.000, -0.500, 0.000, 0.500, 1.000,
                 1.250, 1.500, 1.750, 2.000, 2.250, 2.500, 2.750, 3.000, 3.200, 3.400, 3.600, 3.800, 4.000,
                 4.200, 4.400, 4.600, 4.800, 5.000, 5.200, 5.400, 5.600, 5.800, 6.000, 6.250, 6.500, 6.750,
                 7.000, 7.250, 7.500, 7.750, 8.000], dtype=np.float32)
HEADER = (" ROSSELAND MASS ABSORPTION COEFFICIENTS\n"
          " logT logP  0km/s  1km/s  2km/s  4km/s  8km/s  log Ne   log Na  log rho IPloCM-1\n")


def _f(x: float, w: int, d: int) -> str:
    s = f"{x:{w}.{d}f}"
    if len(s) > w and s.startswith("-0."):  # Fortran drops the leading zero when it doesn't fit
        s = "-" + s[2:]
    elif len(s) > w and s.startswith("0."):
        s = s[1:]
    return s if len(s) <= w else "*" * w


def kapreadts(kapk_paths: list[str | Path], out_path: str | Path) -> None:
    """kapk_paths: the five KAPPA9 outputs for vturb 0,1,2,4,8 (Fortran units 11..15)."""
    akap = np.stack([(read_kapk(p).astype(np.float64) / 1000.0).astype(np.float32) for p in kapk_paths], -1)  # [42,57,5]
    lines = [HEADER]
    zero9, one8 = _f(0.0, 9, 5), _f(1.0, 8, 3)
    for it in range(57):
        for ip in range(42):
            row = _f(float(TABT[it]), 5, 2) + _f(float(TABP[ip]), 5, 2)
            row += "".join(_f(float(akap[ip, it, iv]), 7, 3) for iv in range(5))
            row += zero9 * 3 + one8
            lines.append(row + "\n")
    Path(out_path).write_text("".join(lines))
