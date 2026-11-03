"""Python port of RMOLECASC (SYNTHE molecular line preparation program).

Reads molecular line lists (.asc files), appends line records to fort.12,
and rewinds/updates fort.93 (updating nlines).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import struct

from atlas_torch.io.fortran import write_record
from atlas_torch.synthe.lines.synbeg import read_fort93, write_fort93_dict

C_FACTOR = np.float64(0.026538 / 1.77245)
LN10 = np.float64(2.30258509299405)
C_SPEED = np.float64(2.99792458e17)
C_125664 = np.float64(12.5664)


def get_c12c13(
    workdir: str | Path,
    c12: float | None = None,
    c13: float | None = None,
) -> tuple[float, float]:
    """Retrieve C12 and C13 isotope log abundance offsets.

    Defaults to (-0.005, -1.955), overridden by c12c13.dat in workdir if present,
    or explicitly provided parameters.
    """
    if c12 is not None and c13 is not None:
        return float(c12), float(c13)

    workdir = Path(workdir)
    c12c13_path = workdir / "c12c13.dat"
    if c12c13_path.exists():
        with open(c12c13_path, "r") as f:
            parts = f.read().split()
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])

    return (
        float(c12) if c12 is not None else -0.005,
        float(c13) if c13 is not None else -1.955,
    )


def dispatch_iso(
    iso: int,
    code: float,
    c12: float,
    c13: float,
) -> tuple[int, float, float, float]:
    """Dispatch ISO number to (nelion, fudge, x1, x2).

    Matches Fortran computed GO TO in rmolecasc.patched.for.
    """
    nelion = 0
    fudge = 0.0
    x1 = 0.0
    x2 = 0.0

    if iso == 10:  # H2
        nelion = 240
        fudge = 0.0
        x1 = 0.0
        x2 = -5.0
    elif iso == 20:  # HD
        nelion = 240
        fudge = 0.0
        x1 = 0.0
        x2 = -4.469
    elif iso == 120:
        if code == 606.0:  # C2
            nelion = 264
            fudge = 0.0
            x1 = c12
            x2 = c12
        elif code == 608.0:  # CO
            nelion = 276
            fudge = 0.0
            x1 = c12
            x2 = -0.001
        elif code == 106.0:  # CH
            nelion = 246
            fudge = 0.0
            x1 = 0.0
            x2 = c12
        else:  # CN
            nelion = 270
            fudge = 0.0
            x1 = c12
            x2 = -0.002
    elif iso == 130:
        if code == 606.0:  # C2
            nelion = 264
            fudge = 0.0
            x1 = c12
            x2 = c13
        elif code == 608.0:  # CO
            nelion = 276
            fudge = 0.0
            x1 = c13
            x2 = -0.001
        elif code == 106.0:  # CH
            nelion = 246
            fudge = 0.0
            x1 = 0.0
            x2 = c13
        else:  # CN
            nelion = 270
            fudge = 0.0
            x1 = c13
            x2 = -0.002
    elif iso == 140:  # NH
        nelion = 252
        fudge = 0.0
        x1 = 0.0
        x2 = -0.002
    elif iso == 150:
        if code == 607.0:  # CN
            nelion = 270
            fudge = 0.0
            x1 = c12
            x2 = -2.444
        else:  # NH
            nelion = 252
            fudge = 0.0
            x1 = 0.0
            x2 = -2.444
    elif iso == 160:
        if code == 813.0:  # AlO
            nelion = 324
            fudge = 0.0
            x1 = 0.0
            x2 = -0.001
        else:  # OH
            nelion = 258
            fudge = 0.0
            x1 = 0.0
            x2 = -0.001
    elif iso == 170:
        if code == 813.0:  # AlO
            nelion = 324
            fudge = 0.0
            x1 = 0.0
            x2 = -3.398
        else:  # CO
            nelion = 276
            fudge = 0.0
            x1 = c12
            x2 = -3.398
    elif iso == 180:
        if code == 814.0:  # SiO
            nelion = 330
            fudge = 0.0
            x1 = -0.035
            x2 = -2.690
        elif code == 608.0:  # CO
            nelion = 276
            fudge = 0.0
            x1 = c12
            x2 = -2.690
        elif code == 813.0:  # AlO
            nelion = 324
            fudge = 0.0
            x1 = 0.0
            x2 = -2.690
        else:  # OH
            nelion = 258
            fudge = 0.0
            x1 = 0.0
            x2 = -2.690
    elif iso == 230:  # NaH
        nelion = 492
        fudge = 0.0
        x1 = 0.0
        x2 = 0.0
    elif iso == 240:  # MgH
        nelion = 300
        fudge = 0.0
        x1 = 0.0
        x2 = -0.105
    elif iso == 250:  # MgH
        nelion = 300
        fudge = 0.0
        x1 = 0.0
        x2 = -0.996
    elif iso == 260:  # MgH
        nelion = 300
        fudge = 0.0
        x1 = 0.0
        x2 = -0.947
    elif iso == 280:
        if code == 814.0:  # SiO
            nelion = 330
            fudge = 0.0
            x1 = -0.035
            x2 = -0.001
        else:  # SiH
            nelion = 312
            fudge = 0.0
            x1 = 0.0
            x2 = -0.035
    elif iso == 290:
        if code == 814.0:  # SiO
            nelion = 330
            fudge = 0.0
            x1 = -1.328
            x2 = -0.001
        else:  # SiH
            nelion = 312
            fudge = 0.0
            x1 = 0.0
            x2 = -1.331
    elif iso == 300:
        if code == 814.0:  # SiO
            nelion = 330
            fudge = 0.0
            x1 = -1.510
            x2 = -0.001
        else:  # SiH
            nelion = 312
            fudge = 0.0
            x1 = 0.0
            x2 = -1.516
    elif iso == 330:  # C2
        nelion = 264
        fudge = 0.0
        x1 = c13
        x2 = c13
    elif iso == 390:  # KH
        nelion = 498
        fudge = 0.0
        x1 = -0.030
        x2 = 0.0
    elif iso == 400:  # CaH
        nelion = 342
        fudge = 0.0
        x1 = -0.013
        x2 = 0.0
    elif iso == 410:  # KH
        nelion = 498
        fudge = 0.0
        x1 = -1.172
        x2 = 0.0
    elif iso == 420:  # CaH
        nelion = 342
        fudge = 0.0
        x1 = -2.189
        x2 = 0.0
    elif iso == 430:  # CaH
        nelion = 342
        fudge = 0.0
        x1 = -2.870
        x2 = 0.0
    elif iso == 440:  # CaH
        nelion = 342
        fudge = 0.0
        x1 = -1.681
        x2 = 0.0
    elif iso == 460:
        if code == 120.0:  # CaH
            nelion = 342
            fudge = 0.0
            x1 = -4.398
            x2 = 0.0
        else:  # TiO
            nelion = 366
            fudge = 0.0
            x1 = 0.0
            x2 = -1.101
    elif iso == 470:  # TiO
        nelion = 366
        fudge = 0.0
        x1 = 0.0
        x2 = -1.138
    elif iso == 480:
        if code == 120.0:  # CaH
            nelion = 342
            fudge = 0.0
            x1 = -2.728
            x2 = 0.0
        else:  # TiO
            nelion = 366
            fudge = 0.0
            x1 = 0.0
            x2 = -0.131
    elif iso == 490:  # TiO
        nelion = 366
        fudge = 0.0
        x1 = 0.0
        x2 = -1.259
    elif iso == 500:
        if code == 124.0:  # CrH
            nelion = 432
            fudge = 0.0
            x1 = -1.362
            x2 = 0.0
        else:  # TiO
            nelion = 366
            fudge = 0.0
            x1 = 0.0
            x2 = -1.272
    elif iso == 510:  # VO
        nelion = 372
        fudge = 0.0
        x1 = 0.0
        x2 = -0.001
    elif iso == 520:  # CrH
        nelion = 432
        fudge = 0.0
        x1 = -0.077
        x2 = 0.0
    elif iso == 530:  # CrH
        nelion = 432
        fudge = 0.0
        x1 = -1.022
        x2 = 0.0
    elif iso == 540:
        if code == 124.0:  # CrH
            nelion = 432
            fudge = 0.0
            x1 = -1.626
            x2 = 0.0
        else:  # FeH
            nelion = 444
            fudge = 0.0
            x1 = -1.237
            x2 = 0.0
    elif iso == 560:  # FeH
        nelion = 444
        fudge = 0.0
        x1 = -0.038
        x2 = 0.0
    elif iso == 570:  # FeH
        nelion = 444
        fudge = 0.0
        x1 = -1.658
        x2 = 0.0
    elif iso == 580:  # FeH
        nelion = 444
        fudge = 0.0
        x1 = -2.553
        x2 = 0.0
    else:
        raise ValueError(f"Unrecognized ISO code {iso}")

    return nelion, fudge, x1, x2


def find_asc_start_offset(f: BinaryIO, start_wl: float, file_size: int) -> int:
    """Fast binary search to find file byte offset near `start_wl`."""
    if file_size < 65536:
        return 0

    low = 0
    high = file_size
    margin = 5.0  # Search 5 nm before to account for air-vacuum differences
    target = max(0.0, start_wl - margin)

    while high - low > 32768:
        mid = (low + high) // 2
        f.seek(mid)
        # Advance to start of next line
        f.readline()
        line = f.readline()
        if not line:
            high = mid
            continue
        try:
            wl = abs(float(line[:10]))
            if wl < target:
                low = mid
            else:
                high = mid
        except (ValueError, IndexError):
            high = mid

    return low


def run_rmolecasc(
    workdir: str | Path,
    mol_path: str | Path,
    c12: float | None = None,
    c13: float | None = None,
) -> dict[str, Any]:
    """Execute RMOLECASC for a single molecular list file in `workdir`.

    Reads `workdir/fort.93`, parses `mol_path` (.asc file), appends line records
    to `fort.12`, and updates `fort.93`.
    """
    workdir = Path(workdir)
    mol_path = Path(mol_path)

    c12_val, c13_val = get_c12c13(workdir, c12, c13)

    p = read_fort93(workdir / "fort.93")
    wlbeg = p["wlbeg"]
    wlend = p["wlend"]
    ratiolg = p["ratiolg"]
    ifvac = p["ifvac"]
    ifpred = p["ifpred"]

    ixwlbeg = int(math.log(wlbeg) / ratiolg)
    if math.exp(ixwlbeg * ratiolg) < wlbeg:
        ixwlbeg += 1

    start = wlbeg - 0.1
    stop = wlend + 1.0
    stop1 = stop + 1.0

    file_size = mol_path.stat().st_size
    records = []
    n_added = 0

    with open(mol_path, "r", encoding="latin1") as f:
        # Binary search for start offset
        start_offset = find_asc_start_offset(f.buffer, start, file_size)
        f.seek(start_offset)
        if start_offset > 0:
            f.readline()  # skip partial line

        for line in f:
            if len(line) < 70:
                continue
            try:
                wl = float(line[:10])
            except ValueError:
                continue

            abs_wl = abs(wl)
            if abs_wl > stop1:
                break

            e = float(line[22:32])
            ep = float(line[37:48])

            if ifpred == 0 and (e < 0.0 or ep < 0.0):
                continue

            icode = int(line[48:52])
            code = float(icode)

            if ifvac == 1:
                denom = abs(abs(ep) - abs(e))
                if denom == 0.0:
                    continue
                wlvac = 1.0e7 / denom
            else:
                wlvac = abs_wl

            if wlvac < start:
                continue
            if wlvac > stop:
                break

            gflog = float(line[10:17])
            labelp = line[60:68]
            iso = int(line[68:70])
            loggr_str = line[70:74].strip()
            loggr = int(loggr_str) if loggr_str else 0

            # ISO dispatch table
            # In rmolecasc.for: ISO ranges 1..60, scaled as 10*ISO in branches
            iso_key = iso * 10
            nelion, fudge, x1, x2 = dispatch_iso(iso_key, code, c12_val, c13_val)

            # Fortran arithmetic
            gf = np.float32(math.exp((np.float64(gflog) + np.float64(x1) + np.float64(x2) + np.float64(fudge)) * LN10))
            elo = np.float32(min(abs(e), abs(ep)))
            ixwl = int(math.log(wlvac) / ratiolg + 0.5)
            nbuff = ixwl - ixwlbeg + 1
            freq = np.float32(C_SPEED / wlvac)
            congf = np.float32(C_FACTOR * np.float64(gf) / np.float64(freq))
            frq4pi = np.float32(np.float64(freq) * C_125664)

            # gammar = 10.**(loggr * .01)
            gammar = np.float32(10.0 ** (loggr * 0.01))

            if labelp.strip().startswith("X"):
                gammas = np.float64(3.0e-8)
                gammaw = np.float64(1.0e-8)
            else:
                gammas = np.float64(3.0e-5)
                gammaw = np.float64(1.0e-7)

            gamrf = np.float32(np.float64(gammar) / np.float64(frq4pi))
            gamsf = np.float32(gammas / np.float64(frq4pi))
            gamwf = np.float32(gammaw / np.float64(frq4pi))
            alpha = np.float32(0.0)

            records.append(
                struct.pack("<ifif4f", nbuff, congf, nelion, elo, gamrf, gamsf, gamwf, alpha)
            )
            n_added += 1

    # Append records to fort.12
    with open(workdir / "fort.12", "ab") as f12:
        for payload in records:
            write_record(f12, payload)

    # Update fort.93
    p["nlines"] += n_added
    write_fort93_dict(workdir / "fort.93", p)

    return {
        "n_added": n_added,
        "nlines_total": p["nlines"],
    }
