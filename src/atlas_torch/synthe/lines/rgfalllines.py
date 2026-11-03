"""Python port of RGFALLLINESNEW (SYNTHE atomic line preparation program).

Reads atomic line list (BasicATLAS.dat), computes Stark and van der Waals widths,
appends normal lines to fort.12 and NLTE/predicted lines to fort.19,
and updates fort.93 (updating nlines and n19).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import struct

from atlas_torch.io.fortran import write_record
from atlas_torch.synthe.lines.synbeg import read_fort93, write_fort93_dict


CODEX = [
    1.0, 2.0, 2.01, 6.0, 6.01, 12.0, 12.01, 13.0, 13.01,
    14.0, 14.01, 20.0, 20.01, 8.0, 11.0, 5.0, 19.0,
]
DELLIM = [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1]
C_CGF = 0.026538 / 1.77245
C_SPEED = 2.99792458e17
C_125664 = 12.5664
C_RYDBERG = 109737.31


def load_potion() -> np.ndarray:
    """POTION(999): ionization potentials, COMMON /POTION/ filled by IONPOTS's EQUIVALENCEd DATA tables
    (parsed from rgfalllinesnew.for; bit-identical to the compiled program's values)."""
    from atlas_torch.tables import table
    return np.asarray(table("rgfalllinesnew.for", "ionpots", "potion"), dtype=np.float64)


def find_linelist_start_offset(f: BinaryIO, start_target: float, file_size: int) -> int:
    """Fast binary search to find file byte offset near `start_target`."""
    if file_size < 65536:
        return 0

    low = 0
    high = file_size
    while high - low > 32768:
        mid = (low + high) // 2
        f.seek(mid)
        f.readline()
        line = f.readline()
        if not line:
            high = mid
            continue
        try:
            wl = float(line[:11])
            if wl < start_target:
                low = mid
            else:
                high = mid
        except (ValueError, IndexError):
            high = mid
    return low


def run_rgfalllines(
    workdir: str | Path,
    linelist_path: str | Path,
    potion: np.ndarray | None = None,
) -> dict[str, Any]:
    """Execute RGFALLLINESNEW in `workdir`.

    Reads `workdir/fort.93`, parses `linelist_path` (BasicATLAS.dat), appends normal lines
    to `fort.12` and hydrogen/NLTE lines to `fort.19`, and rewinds/updates `fort.93`.
    """
    workdir = Path(workdir)
    linelist_path = Path(linelist_path)

    if potion is None:
        potion = load_potion()

    p = read_fort93(workdir / "fort.93")
    wlbeg = p["wlbeg"]
    wlend = p["wlend"]
    resolu = p["resolu"]
    ratiolg = p["ratiolg"]
    ifvac = p["ifvac"]

    ixwlbeg = int(math.log(wlbeg) / ratiolg)
    if math.exp(ixwlbeg * ratiolg) < wlbeg:
        ixwlbeg += 1

    delfactor = wlbeg / 500.0 if wlbeg > 500.0 else 1.0
    start_target = wlbeg - DELLIM[0] * delfactor - 5.0

    file_size = linelist_path.stat().st_size
    recs_12 = []
    recs_19 = []
    n14 = 0
    n19 = 0

    with open(linelist_path, "r", encoding="latin1") as f:
        low_offset = find_linelist_start_offset(f.buffer, start_target, file_size)
        f.seek(low_offset)
        if low_offset > 0:
            f.readline()  # Skip partial line

        for line in f:
            if len(line) < 50:
                continue
            line = line.rstrip("\r\n").ljust(166)

            try:
                wl = float(line[:11])
            except ValueError:
                continue

            ep = float(line[52:64])
            e = float(line[24:36])
            other1 = line[124:134]
            other2 = line[134:144]
            labelp = line[70:80]

            ishift = int(other1[0:5]) if other1[0:5].strip() else 0
            ishiftp = int(other1[5:10]) if other1[5:10].strip() else 0
            linesize = int(other2[6:7]) if len(other2) >= 7 and other2[6:7].isdigit() else 0
            auto = other2[7:10] if len(other2) >= 10 else "   "

            eshift = ishift * 0.001
            eshiftp = ishiftp * 0.001
            isoshift = int(line[154:160]) if line[154:160].strip() else 0
            dwliso = isoshift * 0.0001

            if ifvac == 1 or labelp == "CONTINUUM ":
                denom = abs(abs(ep) + eshiftp - (abs(e) + eshift))
                if denom == 0.0:
                    continue
                wlvac = (1.0e7 / denom) + dwliso
            else:
                wlvac = abs(wl)

            code = float(line[18:24])
            lim = min(8 - linesize, 7)
            if code == 1.0:
                lim = 1

            # Window filtering
            if wlvac < wlbeg - DELLIM[lim - 1] * delfactor:
                continue
            if wlvac > wlend + DELLIM[lim - 1] * delfactor:
                # Fortran IF(WLVAC.GT.WLEND+DELLIM(LIM)*DELFACTOR)GO TO 145 (exit)
                break

            if auto == "COR":
                continue

            gflog = float(line[11:18])
            x1 = float(line[109:115]) if line[109:115].strip() else 0.0
            x2 = float(line[118:124]) if line[118:124].strip() else 0.0
            gf = 10.0 ** (gflog + x1 + x2)
            elo = min(abs(e), abs(ep))

            gr = float(line[80:86])
            gs = float(line[86:92])
            gw = float(line[92:98])

            gammar = 10.0 ** gr
            gammas = 10.0 ** gs
            gammaw = 10.0 ** gw

            if auto == "AUT" and gs > 0.0:
                gammas = -10.0 ** (-gs)
            if gr == 0.0:
                gammar = 2.223e13 / (wlvac * wlvac)
                gr = math.log10(gammar)

            nelem = int(code)
            icharge = int((code - float(nelem)) * 100.0 + 0.1)
            zeff = float(icharge + 1)
            nelion = nelem * 6 - 6 + int(zeff)
            if 19 < nelem < 29 and icharge > 5:
                nelion = 6 * (nelem + icharge * 10 - 30) - 1

            if gs == 0.0 or gw == 0.0:
                if gs == 0.0:
                    if code >= 100.0:
                        gammas = 1.0e-5
                        gs = -5.0
                    else:
                        eup = np.float32(max(abs(e), abs(ep)))
                        effnsq = np.float32(25.0)
                        iz = int(code)
                        index = iz * (iz + 1) // 2 + icharge if iz <= 30 else iz * 5 + 341 + icharge
                        deleup = np.float32(potion[index - 1] - np.float64(eup))
                        if deleup > 0.0:
                            effnsq = np.float32(C_RYDBERG * (zeff ** 2) / deleup)
                        gammas = np.float32(1.0e-8 * (float(effnsq) ** 2) * math.sqrt(float(effnsq)))
                        gs = math.log10(float(gammas))
                        gs = min(gs, -3.0)

                if gw == 0.0:
                    if code >= 100.0:
                        gammaw = 1.0e-7 / zeff
                        gw = math.log10(gammaw)
                    else:
                        eup = np.float32(max(abs(e), abs(ep)))
                        effnsq = np.float32(25.0)
                        iz = int(code)
                        index = iz * (iz + 1) // 2 + icharge if iz <= 30 else iz * 5 + 341 + icharge
                        deleup = np.float32(potion[index - 1] - np.float64(eup))
                        if deleup > 0.0:
                            effnsq = np.float32(C_RYDBERG * (zeff ** 2) / deleup)
                        effnsq = min(float(effnsq), 1000.0)
                        rsqup = 2.5 * (effnsq / zeff) ** 2

                        delelo = np.float32(potion[index - 1] - np.float64(np.float32(elo)))
                        effnsq = min(float(np.float32(C_RYDBERG * (zeff ** 2) / delelo)), 1000.0)
                        rsqlo = 2.5 * (effnsq / zeff) ** 2

                        nseq = int(code - zeff + 1.0)
                        if 20 < nseq < 29:
                            rsqup = (45.0 - float(nseq)) / zeff
                            rsqlo = 0.0
                        if labelp == "CONTINUUM ":
                            rsqlo = 0.0
                        if rsqup < rsqlo:
                            rsqup = 2.0 * rsqlo
                        gammaw = 4.5e-9 * ((rsqup - rsqlo) ** 0.4)
                        gw = math.log10(gammaw)

            # Determine type
            iso1 = int(line[106:109]) if line[106:109].strip() else 0
            iso2 = int(line[115:118]) if line[115:118].strip() else 0

            type_val = 0
            if code == 1.00:
                type_val = -1
                if iso1 == 2:
                    type_val = -2
            elif code == 2.00:
                type_val = -3
                if iso1 == 3:
                    type_val = -4
            elif code == 2.01:
                type_val = -6
                if iso1 == 3:
                    type_val = -6

            if auto == "COR":
                type_val = 2
            elif auto == "AUT":
                type_val = 1
            elif auto == "PRD":
                type_val = 3

            if labelp == "CONTINUUM ":
                xjp = float(line[64:69])
                type_val = int(xjp)
                xj = float(line[36:41])
                gf = gf * (xj + xj + 1.0)

            ncon = 0
            if iso1 == 0 and iso2 > 0:
                ncon = iso2

            # Truncated single-precision FRELIN in Fortran
            frelin = np.float32(C_SPEED / wlvac)
            cgf = np.float32(C_CGF * gf / frelin)

            if type_val == 2:
                gammar_val = np.float32(gr)
            else:
                gammar_val = np.float32(gammar / C_125664 / frelin)
                gammas_val = np.float32(gammas / C_125664 / frelin)
                gammaw_val = np.float32(gammaw / C_125664 / frelin)

            nblo = int(line[102:104]) if line[102:104].strip() else 0
            nbup = int(line[104:106]) if line[104:106].strip() else 0
            nblo = abs(nblo)
            nbup = abs(nbup)

            ixwl = int(math.log(wlvac) / ratiolg + 0.5)
            nbuff = ixwl - ixwlbeg + 1
            alpha = float(line[160:166]) if len(line) >= 166 and line[160:166].strip() else 0.0
            alpha_f32 = np.float32(alpha)
            elo_f32 = np.float32(elo)

            if nblo + nbup > 0 or type_val == 1:
                # Special lines go to fort.19
                nelionx = 0
                if type_val != 1:
                    try:
                        nelionx = CODEX.index(code) + 1
                    except ValueError:
                        pass
                recs_19.append(
                    struct.pack(
                        "<d2f6i4f2i",
                        wlvac,
                        elo_f32,
                        cgf,
                        nblo,
                        nbup,
                        nelion,
                        type_val,
                        ncon,
                        nelionx,
                        gammar_val,
                        gammas_val,
                        gammaw_val,
                        alpha_f32,
                        nbuff,
                        lim,
                    )
                )
                n19 += 1
            else:
                # Normal lines go to fort.12
                recs_12.append(
                    struct.pack(
                        "<ifif4f",
                        nbuff,
                        cgf,
                        nelion,
                        elo_f32,
                        gammar_val,
                        gammas_val,
                        gammaw_val,
                        alpha_f32,
                    )
                )
                n14 += 1

    # Append to fort.12 and fort.19
    with open(workdir / "fort.12", "ab") as f12:
        for payload in recs_12:
            write_record(f12, payload)

    with open(workdir / "fort.19", "ab") as f19:
        for payload in recs_19:
            write_record(f19, payload)

    # Rewind and update fort.93
    p["nlines"] += n14
    p["n19"] += n19
    write_fort93_dict(workdir / "fort.93", p)

    return {
        "n14": n14,
        "n19": n19,
        "nlines_total": p["nlines"],
        "n19_total": p["n19"],
    }
