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

from atlas_torch.synthe.lines._fnum import dexp, dlog, pack12, vpowf, write_records12
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
    """DO 1 I=1,32768: TABLOG(I)=10.**((I-16384)*.001) in REAL*4, as the stock gfortran -O3 rh2ofast.exe computes it.

    gfortran -O3 vectorises the loop: (I-16384) converted to REAL*4 times .001 (REAL*4), then glibc libmvec's
    _ZGVbN4vv_powf on groups of 4 (objdump: the executable calls _ZGVbN4vv_powf and no scalar powf; 32768 = 4*8192
    so there is no scalar remainder). libmvec powf differs from scalar powf by 1 ulp in 1684 of the entries.
    """
    global _TABLOG
    if _TABLOG is None:
        i = np.arange(1, 32769, dtype=np.int32)
        e = (i - 16384).astype(np.float32) * np.float32(0.001)
        _TABLOG = vpowf(np.float32(10.0), e)
    return _TABLOG


def vacair(w: float) -> float:
    """FUNCTION VACAIR(W) (IMPLICIT REAL*8): vacuum -> air wavelength [nm]."""
    waven = 1.0e7 / w
    return w / (1.0000834213 + 2406030.0 / (1.30e10 - waven * waven) + 15997.0 / (3.89e9 - waven * waven))


def tabvacair() -> np.ndarray:
    """SUBROUTINE TABVACAIR(AIRSHIFT): AIRSHIFT(1..1999)=0; AIRSHIFT(IWL)=VACAIR(WLVAC)-WLVAC with WLVAC=IWL*.1
    (REAL*4 product widened to REAL*8), stored REAL*4."""
    out = np.zeros(100000, np.float32)
    for iwl in range(2000, 100001):
        w = float(np.float32(np.float32(iwl) * np.float32(0.1)))
        out[iwl - 1] = np.float32(vacair(w) - w)
    return out


def run_rh2ofast(
    workdir: str | Path,
    h2o_path: str | Path,
    tablog: np.ndarray | None = None,
    chunk: int = 1 << 21,
) -> dict[str, Any]:
    """Execute RH2OFAST in `workdir`.

    Reads `workdir/fort.93`, parses `h2o_path` (h2ofastfix.bin), appends to `fort.12`,
    and updates `fort.93`.
    Returns dict with run statistics (lines added, total lines).
    The records are processed in blocks of `chunk` (vectorised, element-wise the scalar loop's operations).
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
    lengthfile = file_size // 8                  # LENGTHFILE=65912356 in rh2ofast.for (= h2ofastfix.bin records)
    airshift = tabvacair() if p["ifpred"] != 1 else None   # IF(IFPRED.NE.1)CALL TABVACAIR(AIRSHIFT); else 0.

    def shifted(wlvac):
        """IF(IFVAC.NE.1): KWL=WLVAC*10.+.5; WLVAC=WLVAC+AIRSHIFT(KWL)."""
        if ifvac == 1:
            return wlvac
        kwl = int(wlvac * 10.0 + 0.5)
        if not 1 <= kwl <= 100000:
            raise ValueError(f"rh2ofast: AIRSHIFT({kwl}) out of bounds")
        return wlvac + (float(airshift[kwl - 1]) if airshift is not None else 0.0)

    def wl_of(iwl, vacair_too):
        w = math.exp(iwl * RATIOLOG)
        return vacair(w) if (vacair_too and ifvac != 1) else w

    n_added = 0
    with open(h2o_path, "rb") as f:
        def iwl_at(rec):
            f.seek((rec - 1) * 8)
            return struct.unpack("<i", f.read(4))[0]

        # first record (AIRSHIFT) and last record (VACAIR) bracket the file
        if shifted(wl_of(iwl_at(1), False)) > wlend1 or wlbeg1 > wl_of(iwl_at(lengthfile), True):
            istart = None
        else:
            limitblue = 1
            limitred = lengthfile
            while True:
                newlimit = (limitred + limitblue) // 2
                wlvac = wl_of(iwl_at(newlimit), True)
                if wlvac < wlbeg1:
                    limitblue = newlimit
                else:
                    limitred = newlimit
                if limitred - limitblue <= 1:
                    istart = newlimit
                    break

        # DO 20 ILINE=ISTART,LENGTHFILE, in blocks of `chunk` records, each written before the next is read
        f12 = workdir / "fort.12"
        open(f12, "ab").close()
        if istart is not None:
            rec = istart
            while rec <= lengthfile:
                m = min(chunk, lengthfile - rec + 1)
                f.seek((rec - 1) * 8)
                raw = np.frombuffer(f.read(m * 8), np.dtype([("iwl", "<i4"), ("ielo", "<i2"), ("igflog", "<i2")]))
                recs, done = _block(raw, tablog, airshift, ifvac, wlend1, ratiolg, ixwlbeg)
                if len(recs):
                    write_records12(f12, recs)
                    n_added += len(recs)
                if done:
                    break
                rec += m

    # Update fort.93
    p["nlines"] += n_added
    p["linout"] = -abs(p["linout"])
    write_fort93_dict(workdir / "fort.93", p)

    return {
        "n_added": n_added,
        "nlines_total": p["nlines"],
    }


def _block(raw: np.ndarray, tablog: np.ndarray, airshift, ifvac: int, wlend1: float, ratiolg: float,
           ixwlbeg: int):
    """fort.12 records of a block of h2ofastfix.bin records (IWL, IELO, IGFLOG) and whether the loop ended:
      WLVAC=EXP(IWL*RATIOLOG); FREQ=2.99792458E17/WLVAC (REAL*4); [air shift]; IF(WLVAC.GT.WLEND+1.)GO TO 21
      ISO from the signs of IELO/IGFLOG; CONGF=.01502*TABLOG(|IGFLOG|)/FREQ*XISO(ISO); FRQ4PI=FREQ*12.5664
      GAMRF=2.223E13/WLVAC**2*.001/FRQ4PI; GAMSF=TABLOG(1)/FRQ4PI; GAMWF=TABLOG(9384)/FRQ4PI;
      NBUFF=IXWL-IXWLBEG+1 with IXWL=DLOG(WLVAC)/RATIOLG+.5; NELION=534; ELO=|IELO|; ALPHA=0."""
    iwl = raw["iwl"].astype(np.float64)
    ielo = raw["ielo"].astype(np.int64)
    igflog = raw["igflog"].astype(np.int64)
    wlvac = dexp(iwl * RATIOLOG)
    freq = (C_FREQ_NUMERATOR / wlvac).astype(np.float32)
    if ifvac != 1:
        kwl = np.trunc(wlvac * 10.0 + 0.5).astype(np.int64)
        oob = (kwl < 1) | (kwl > 100000)
        sh = airshift[np.clip(kwl, 1, 100000) - 1].astype(np.float64) if airshift is not None else 0.0
        wlvac = wlvac + sh
        # sequential order: an out-of-bounds AIRSHIFT before the end of the loop stops the program
        ib = np.flatnonzero(oob)
        st = np.flatnonzero(~oob & (wlvac > wlend1))
        if ib.size and (not st.size or ib[0] < st[0]):
            raise ValueError(f"rh2ofast: AIRSHIFT({int(kwl[ib[0]])}) out of bounds")
    stops = np.flatnonzero(wlvac > wlend1)
    done = bool(stops.size)
    if done:
        k = int(stops[0])
        wlvac, freq, ielo, igflog = wlvac[:k], freq[:k], ielo[:k], igflog[:k]
    frq4pi = freq * C_125664
    iso = np.where(ielo > 0, np.where(igflog > 0, 1, 2), np.where(igflog > 0, 3, 4))
    ig = np.abs(igflog)
    if ((ig < 1) | (ig > 32768)).any():
        bad = int(ig[(ig < 1) | (ig > 32768)][0])
        raise ValueError(f"rh2ofast: TABLOG({bad}) out of bounds")
    congf = ((C_01502 * tablog[ig - 1]) / freq) * XISO[iso - 1]
    gammar = ((C1_GAMMAR / (wlvac * wlvac)) * C2_GAMMAR).astype(np.float32)
    nbuff = np.trunc(dlog(wlvac) / ratiolg + 0.5).astype(np.int64) - ixwlbeg + 1
    n = len(wlvac)
    recs = pack12(nbuff.astype(np.int32), congf, np.full(n, 534, np.int32), np.abs(ielo).astype(np.float32),
                  gammar / frq4pi, tablog[0] / frq4pi, tablog[9383] / frq4pi, np.zeros(n, np.float32))
    return recs, done
