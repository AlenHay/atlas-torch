"""Python port of RH2OFAST (SYNTHE H2O line preparation program).

Reads direct-access h2ofastfix.bin, appends H2O line records to fort.12,
and rewinds/updates fort.93 (updating nlines and setting linout = -abs(linout)).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import struct

from atlas_torch.io.fortran import write_record
from atlas_torch.synthe.lines.synbeg import read_fort93, write_fort93_dict


# Constants exactly matching gfortran -O3 compilation of rh2ofast.for
# Single precision constants promoted to double in Fortran:
C_FREQ_NUMERATOR = np.float64(np.float32(2.99792458e17))
C1_GAMMAR = 22229999943680.0  # double(float32(2.223E13))
C2_GAMMAR = 0.0010000000474974513  # double(float32(0.001))
C_01502 = np.float32(0.01502)
C_125664 = np.float32(12.5664)
XISO = np.array([0.9976, 0.0004, 0.0020, 0.00001], dtype=np.float32)
RATIOLOG = math.log(1.0 + 1.0 / 2000000.0)


_TABLOG = None


def load_tablog() -> np.ndarray:
    """TABLOG(I) = 10.**((I-16384)*.001) in REAL*4, evaluated with glibc's scalar powf.

    The stock -O3 build vectorises this loop with glibc's libmvec powf (_ZGVbN4vv_powf), which differs from
    scalar powf by 1 ulp in 1684 of 32768 entries; scalar powf is what the gfortran -O0 noise variant computes,
    so this choice lies within the measured Fortran noise floor (see RESULTS.md).
    """
    global _TABLOG
    if _TABLOG is None:
        import ctypes
        import ctypes.util
        libm = ctypes.CDLL(ctypes.util.find_library("m"))
        powf = libm.powf
        powf.restype = ctypes.c_float
        powf.argtypes = [ctypes.c_float, ctypes.c_float]
        i = np.arange(1, 32769, dtype=np.int32)
        e = ((i - 16384).astype(np.float32) * np.float32(0.001)).astype(np.float32)
        _TABLOG = np.array([powf(10.0, float(x)) for x in e], dtype=np.float32)
    return _TABLOG


def run_rh2ofast(
    workdir: str | Path,
    h2o_path: str | Path,
    tablog: np.ndarray | None = None,
) -> dict[str, Any]:
    """Execute RH2OFAST in `workdir`.

    Reads `workdir/fort.93`, parses `h2o_path` (h2ofastfix.bin), appends to `fort.12`,
    and updates `fort.93`.
    Returns dict with run statistics (lines added, total lines).
    """
    workdir = Path(workdir)
    h2o_path = Path(h2o_path)

    if tablog is None:
        tablog = load_tablog()

    p = read_fort93(workdir / "fort.93")
    wlbeg = p["wlbeg"]
    wlend = p["wlend"]
    ratiolg = p["ratiolg"]
    ifvac = p["ifvac"]

    ixwlbeg = int(math.log(wlbeg) / ratiolg)
    if math.exp(ixwlbeg * ratiolg) < wlbeg:
        ixwlbeg += 1

    wlbeg1 = wlbeg - 1.0
    wlend1 = wlend + 1.0

    file_size = h2o_path.stat().st_size
    lengthfile = file_size // 8

    # Fortran binary search for istart
    with open(h2o_path, "rb") as f:
        limitblue = 1
        limitred = lengthfile
        while True:
            newlimit = (limitred + limitblue) // 2
            f.seek((newlimit - 1) * 8)
            (iwl,) = struct.unpack("<i", f.read(4))
            wlvac = math.exp(iwl * RATIOLOG)
            # In rh2ofast.for ifvac != 1 applies airshift, but BasicATLAS runs VAC (ifvac=1)
            if wlvac < wlbeg1:
                limitblue = newlimit
                if limitred - limitblue <= 1:
                    istart = newlimit
                    break
            else:
                limitred = newlimit
                if limitred - limitblue <= 1:
                    istart = newlimit
                    break

        # Read records starting at istart
        f.seek((istart - 1) * 8)
        n_added = 0
        line_records = []

        while True:
            data = f.read(8)
            if len(data) < 8:
                break
            iwl, ielo, igflog = struct.unpack("<ihh", data)
            wlvac = math.exp(iwl * RATIOLOG)
            if wlvac > wlend1:
                break

            freq = np.float32(C_FREQ_NUMERATOR / wlvac)
            frq4pi = freq * C_125664

            if ielo > 0 and igflog > 0:
                iso = 1
            elif ielo > 0:
                iso = 2
            elif igflog > 0:
                iso = 3
            else:
                iso = 4

            igflog_abs = abs(igflog)
            xmm0 = (C_01502 * tablog[igflog_abs - 1]) / freq
            congf = xmm0 * XISO[iso - 1]

            gammar_dp = (C1_GAMMAR / (wlvac * wlvac)) * C2_GAMMAR
            gammar_sp = np.float32(gammar_dp)
            gamrf = gammar_sp / frq4pi
            gamsf = tablog[0] / frq4pi
            gamwf = tablog[9383] / frq4pi

            ixwl = int(math.log(wlvac) / ratiolg + 0.5)
            nbuff = ixwl - ixwlbeg + 1
            elo = np.float32(abs(ielo))
            alpha = np.float32(0.0)

            line_records.append(
                struct.pack("<ifif4f", nbuff, congf, 534, elo, gamrf, gamsf, gamwf, alpha)
            )
            n_added += 1

    # Append to fort.12
    with open(workdir / "fort.12", "ab") as f12:
        for payload in line_records:
            write_record(f12, payload)

    # Update fort.93
    p["nlines"] += n_added
    p["linout"] = -abs(p["linout"])
    write_fort93_dict(workdir / "fort.93", p)

    return {
        "n_added": n_added,
        "nlines_total": p["nlines"],
    }
