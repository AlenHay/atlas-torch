"""Python port of RMOLECASC (SYNTHE molecular line preparation program).

Reads molecular line lists (.asc files), appends line records to fort.12,
and rewinds/updates fort.93 (updating nlines).

The list is streamed in blocks (`_fixed.iter_records`, ~32 MB at a time; tiototo.asc is 1.7 GB) and each
block's records are appended to fort.12 before the next one is read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from atlas_torch.synthe.lines._fixed import fcol, icol, iter_records
from atlas_torch.synthe.lines._fnum import F4, dexp, ixwlbeg_of, nbuff_of, pack12, powf, r4, write_records12
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

    Matches Fortran computed GO TO in rmolecasc.patched.for. Returns the values as assigned (X1/X2 are then
    stored as REAL*4 by the caller); nelion None = label 580, which does not set NELION.
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
    elif iso == 580:  # FeH: "NEILON=444" (typo in rmolecasc.for): NELION keeps its previous value
        nelion = None
        fudge = 0.0
        x1 = -2.553
        x2 = 0.0
    elif 10 <= iso <= 600 and iso % 10 == 0:
        # label 99 of the computed GO TO: PRINT 3 ... ; CALL EXIT (the program stops; fort.93 not rewritten)
        raise ValueError(f"rmolecasc: ISO={iso // 10} goes to label 99 (the Fortran stops here)")
    else:
        # ISO outside 1..60: the computed GO TO falls through to the next statement, label 10 (H2)
        nelion, fudge, x1, x2 = 240, 0.0, 0.0, -5.0

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
    chunk_bytes: int = 32 << 20,
) -> dict[str, Any]:
    """Execute RMOLECASC for a single molecular list file in `workdir`.

    Reads `workdir/fort.93`, parses `mol_path` (.asc file), appends line records
    to `fort.12`, and updates `fort.93`.

    Arithmetic as compiled (IMPLICIT REAL*4 (A-H,O-Z); WL, E, EP, WLVAC REAL*8; stock gfortran -O3 calls the
    scalar glibc exp/powf here, no vectorised loop):
      GF=EXP((GFLOG+X1+X2+FUDGE)*2.30258509299405D0)    REAL*4 sum, REAL*8 exp, rounded to REAL*4
      FREQ=2.99792458D17/WLVAC; CONGF=.026538D0/1.77245D0*GF/FREQ; FRQ4PI=FREQ*12.5664D0
      GAMMAR=10.**(LOGGR*.01)                             REAL*4 product, powf
      GAMRF=GAMMAR/FRQ4PI; GAMSF=GAMMAS/FRQ4PI; GAMWF=GAMMAW/FRQ4PI   REAL*4 (GAMMAS=3.D-5 etc. REAL*4)
    LINOUT>=0 (fort.14 LINDAT records) is not ported.  The file is processed in blocks of ~`chunk_bytes`.
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
    if p["linout"] >= 0:
        raise NotImplementedError("rmolecasc: LINOUT >= 0 (fort.14 LINDAT records) is not ported")

    ixwlbeg = ixwlbeg_of(wlbeg, ratiolg)

    start = wlbeg - 0.1
    stop = wlend + 1.0
    stop1 = stop + 1.0

    file_size = mol_path.stat().st_size
    with open(mol_path, "rb") as fb:
        start_offset = find_asc_start_offset(fb, start, file_size)

    f12 = workdir / "fort.12"
    open(f12, "ab").close()
    n_added = 0
    state = {"nelion": 0}                        # /LINDAT/ NELION: 0 at program start
    dispatch: dict = {}
    # READ(11,1111) WL,GFLOG,XJ,E,XJP,EP,ICODE,LABEL,LABELP,ISO,LOGGR
    # 1111 FORMAT(F10.4,F7.3,F5.1,F10.3,F5.1,F11.3,I4,A8,A8,I2,I4)   (short records padded with blanks)
    for a in iter_records(mol_path, start_offset, 74, chunk_bytes, skip_partial=start_offset > 0):
        recs, done = _block(a, ifvac, ifpred, start, stop, stop1, ratiolg, ixwlbeg, c12_val, c13_val, dispatch,
                            state)
        if recs is not None and len(recs):
            write_records12(f12, recs)
            n_added += len(recs)
        if done:
            break

    # Update fort.93
    p["nlines"] += n_added
    write_fort93_dict(workdir / "fort.93", p)

    return {
        "n_added": n_added,
        "nlines_total": p["nlines"],
    }


def _block(a, ifvac, ifpred, start, stop, stop1, ratiolg, ixwlbeg, c12_val, c13_val, dispatch, state):
    """fort.12 records of one block of records (record order) and whether the read loop ended in it.
    Per record: end when |WL| > STOP1; skip predicted levels (IFPRED=0); WLVAC; skip WLVAC < START; end when
    WLVAC > STOP."""
    wl = fcol(a, 0, 10, 4)
    n = len(wl)
    abs_wl = np.abs(wl)
    end1 = abs_wl > stop1
    e = fcol(a, 22, 32, 3)
    ep = fcol(a, 37, 48, 3)
    skip = ((e < 0.0) | (ep < 0.0)) if ifpred == 0 else np.zeros(n, bool)
    if ifvac == 1:
        denom = np.abs(np.abs(ep) - np.abs(e))
        with np.errstate(divide="ignore"):
            wlvac = 1.0e7 / denom                      # denom = 0 -> +inf
    else:
        wlvac = abs_wl
    lo = ~skip & (wlvac < start)
    end2 = ~skip & ~lo & (wlvac > stop)
    stops = np.flatnonzero(end1 | end2)
    ilast = int(stops[0]) if stops.size else n
    done = bool(stops.size)
    keep = np.flatnonzero((np.arange(n) < ilast) & ~skip & ~lo)
    if not keep.size:
        return None, done
    b = a[keep]
    wlvac = wlvac[keep]
    e, ep = e[keep], ep[keep]
    icode = icol(b, 48, 52)
    iso = icol(b, 68, 70)
    loggr = icol(b, 70, 74)
    xlab = b[:, 60] == ord("X")                         # IF(CLABELP(1:1).EQ.'X')
    gfl = fcol(b, 10, 17, 3)
    m = len(keep)
    nel = np.empty(m, np.int64)
    xa = np.empty(m, F4)
    xb = np.empty(m, F4)
    fud = np.empty(m, F4)
    has = np.ones(m, bool)
    key = iso * 100000 + icode
    uk, inv = np.unique(key, return_inverse=True)
    # ISO dispatch per distinct (ISO, ICODE); label 99 (stop) must happen at its first record, in order
    first = np.full(len(uk), m, np.int64)
    np.minimum.at(first, inv, np.arange(m))
    for u in np.argsort(first, kind="stable"):
        kk = (int(iso[first[u]]), int(icode[first[u]]))
        d = dispatch.get(kk)
        if d is None:
            n_, fu, x1, x2 = dispatch_iso(kk[0] * 10, r4(float(kk[1])), c12_val, c13_val)
            d = dispatch[kk] = (n_, r4(fu), r4(x1), r4(x2))
        sel = inv == u
        if d[0] is None:
            has[sel] = False
            nel[sel] = 0
        else:
            nel[sel] = d[0]
        fud[sel], xa[sel], xb[sel] = d[1], d[2], d[3]
    # ISO=58 (label 580) leaves NELION as the previous record set it (carried across blocks)
    src = np.maximum.accumulate(np.where(has, np.arange(m), -1))
    nel = np.where(src >= 0, nel[np.maximum(src, 0)], state["nelion"])
    state["nelion"] = int(nel[-1])

    s4 = ((gfl.astype(F4) + xa) + xb) + fud
    gf = dexp(s4.astype(np.float64) * LN10).astype(F4)
    freq = (C_SPEED / wlvac).astype(F4)
    congf = (C_FACTOR * gf.astype(np.float64) / freq.astype(np.float64)).astype(F4)
    frq4pi = (freq.astype(np.float64) * C_125664).astype(F4)
    gammar = powf(F4(10.0), loggr.astype(F4) * F4(0.01))
    gammas = np.where(xlab, F4(3.0e-8), F4(3.0e-5)).astype(F4)
    gammaw = np.where(xlab, F4(1.0e-8), F4(1.0e-7)).astype(F4)
    elo = np.minimum(np.abs(e), np.abs(ep)).astype(F4)
    recs = pack12(nbuff_of(wlvac, ratiolg, ixwlbeg), congf, nel.astype(np.int32), elo,
                  gammar / frq4pi, gammas / frq4pi, gammaw / frq4pi, np.zeros(m, F4))
    return recs, done
