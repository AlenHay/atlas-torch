"""Python port of RGFALLLINESNEW (SYNTHE atomic line preparation program).

Reads atomic line list (BasicATLAS.dat), computes Stark and van der Waals widths,
appends normal lines to fort.12 and NLTE/predicted lines to fort.19,
and updates fort.93 (updating nlines and n19).

The line list is streamed in blocks (`_fixed.iter_records`, ~32 MB at a time) and every block's records are
appended to fort.12 / fort.19 before the next block is read: memory does not grow with the window.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from atlas_torch.synthe.lines._fixed import fcol, icol, iter_records, scol
from atlas_torch.synthe.lines._fnum import F4, ixwlbeg_of, log10f, nbuff_of, pack12, powf, r4, write_records12
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


FORT19 = np.dtype([("wlvac", "<f8"), ("elo", "<f4"), ("gf", "<f4"), ("nblo", "<i4"), ("nbup", "<i4"),
                   ("nelion", "<i4"), ("type", "<i4"), ("ncon", "<i4"), ("nelionx", "<i4"), ("gammar", "<f4"),
                   ("gammas", "<f4"), ("gammaw", "<f4"), ("alpha", "<f4"), ("nbuff", "<i4"), ("lim", "<i4")])
assert FORT19.itemsize == 64      # WRITE(19)WLVAC,ELO,GF,NBLO,NBUP,NELION,TYPE,NCON,NELIONX,GAMMAR,GAMMAS,GAMMAW,alpha,NBUFF,LIM


def _write_fixed(path: Path, recs: np.ndarray) -> None:
    """Append fixed-length records (gfortran unformatted sequential, 4-byte markers)."""
    n = len(recs)
    out = np.empty(n, np.dtype([("m0", "<i4"), ("r", recs.dtype), ("m1", "<i4")]))
    out["m0"] = recs.dtype.itemsize
    out["m1"] = recs.dtype.itemsize
    out["r"] = recs
    with open(path, "ab") as fh:
        fh.write(out.tobytes())


def run_rgfalllines(
    workdir: str | Path,
    linelist_path: str | Path,
    potion: np.ndarray | None = None,
    chunk_bytes: int = 32 << 20,
) -> dict[str, Any]:
    """Execute RGFALLLINESNEW in `workdir`.

    Reads `workdir/fort.93`, parses `linelist_path` (BasicATLAS.dat), appends normal lines
    to `fort.12` and hydrogen/NLTE lines to `fort.19`, and rewinds/updates `fort.93`.

    PROGRAM RGFALL has no IMPLICIT statement: everything not declared REAL*8 (WL, E, EP, WLVAC, WLBEG, WLEND,
    RATIOLG, POTION) is REAL*4 (GFLOG, CODE, XJ, XJP, GR, GS, GW, X1, X2, ESHIFT, DWLISO, DELFACTOR, DELLIM,
    GF, ELO, GAMMAR/S/W, EFFNSQ, ZEFF, RSQUP/LO, FRELIN, ...), and the stock -O3 executable calls the scalar
    glibc powf (10.**x, x**.4) and log10f (ALOG10). Reproduced: GF and CGF share storage (EQUIVALENCE), so TYPE=1
    (AUT) and TYPE>3 lines, which skip the CGF/width scaling, write the raw GF and widths; a continuum line with
    NLAST=TYPE=2 jumps to 1253 past label 17 (NBLO/NBUP not made positive, NELIONX keeps the previous line's value);
    DGFLOG (never assigned; DLOGGF is) and DWL/DGAMMA* are 0. LINOUT>=0 (fort.14/fort.20 LINDAT) is not ported.

    The list is read in blocks of ~`chunk_bytes`; each block is decoded column-wise with the Fortran Fw.d/Iw
    semantics and its records are written before the next block is read (bounded memory).
    """
    workdir = Path(workdir)
    linelist_path = Path(linelist_path)

    if potion is None:
        potion = load_potion()

    p = read_fort93(workdir / "fort.93")
    wlbeg = p["wlbeg"]
    wlend = p["wlend"]
    ratiolg = p["ratiolg"]
    ifvac = p["ifvac"]
    if p["linout"] >= 0:
        raise NotImplementedError("rgfalllinesnew: LINOUT >= 0 (fort.14/fort.20 LINDAT records) is not ported")

    ixwlbeg = ixwlbeg_of(wlbeg, ratiolg)
    dellim = np.array([r4(v) for v in DELLIM], F4)        # DATA DELLIM (REAL*4)
    delfactor = r4(wlbeg / 500.0) if wlbeg > 500.0 else 1.0
    stop0 = wlend + float(dellim[0])                     # IF(WLVAC.GT.WLEND+DELLIM(1))GO TO 145
    start_target = wlbeg - DELLIM[0] * delfactor - 5.0
    codex = np.array([r4(c) for c in CODEX], F4)

    file_size = linelist_path.stat().st_size
    with open(linelist_path, "rb") as fb:
        low_offset = find_linelist_start_offset(fb, start_target, file_size)

    f12, f19 = workdir / "fort.12", workdir / "fort.19"
    open(f12, "ab").close()
    open(f19, "ab").close()
    n14 = n19 = 0
    carry = {"nelionx": 0}
    # 140 FORMAT(F11.4,F7.3,F6.2,F12.3,F5.1,1X,A10,F12.3,F5.1,1X,A10,F6.2,F6.2,F6.2,A4,I2,I2,I3,F6.3,I3,F6.3,
    #            A10,A10,2I5,I6,f6.3)       (records shorter than 166 characters are padded with blanks)
    for a in iter_records(linelist_path, low_offset, 166, chunk_bytes, skip_partial=low_offset > 0):
        sel, done = _select(a, ifvac, wlbeg, wlend, stop0, dellim, delfactor)
        if sel is not None:
            r12, r19 = _records(sel, potion, ratiolg, ixwlbeg, codex, carry)
            if len(r12):
                write_records12(f12, r12)
                n14 += len(r12)
            if len(r19):
                _write_fixed(f19, r19)
                n19 += len(r19)
        if done:
            break

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


def _select(a: np.ndarray, ifvac: int, wlbeg: float, wlend: float, stop0: float, dellim: np.ndarray,
            delfactor: float):
    """The read loop's window logic on a block of records -> (decoded columns of the kept records | None,
    whether the loop ended in this block).  In record order: GO TO 145 (end) when WLVAC > WLEND+DELLIM(1);
    LIM/DELLIM; skip WLVAC < WLBEG-DL; end when WLVAC > WLEND+DL; skip AUTO='COR'."""
    c001, c0001 = F4(0.001), F4(0.0001)
    wl = fcol(a, 0, 11, 4)
    e = fcol(a, 24, 36, 3)
    ep = fcol(a, 52, 64, 3)
    cont = scol(a, 70, 80) == b"CONTINUUM "
    ishift, ishiftp = icol(a, 124, 129), icol(a, 129, 134)          # READ(COTHER1,'(2I5)')
    linesize = icol(a, 140, 141)                                    # READ(COTHER2,'(A6,I1,A3)')
    auto = scol(a, 141, 144)
    dwliso = icol(a, 154, 160).astype(F4) * c0001

    with np.errstate(divide="ignore"):
        wlvac = np.abs(wl) + 0.0
        conv = (ifvac == 1) | cont
        if conv.any():
            eshift = (ishift.astype(F4) * c001).astype(np.float64)
            eshiftp = (ishiftp.astype(F4) * c001).astype(np.float64)
            d = np.abs((np.abs(ep) + eshiftp) - (np.abs(e) + eshift))
            wv = (1.0e7 / d + 0.0) + dwliso.astype(np.float64)         # d = 0 -> +inf (> stop0)
            wlvac = np.where(conv, wv, wlvac)
    n = len(wlvac)
    end0 = wlvac > stop0
    code = fcol(a, 18, 24, 2).astype(F4)
    lim = np.minimum(8 - linesize, 7)
    lim = np.where(code == F4(1.0), 1, lim)
    dl = (dellim[np.clip(lim - 1, 0, 6)] * F4(delfactor)).astype(np.float64)
    lo = wlvac < wlbeg - dl
    end1 = ~lo & (wlvac > wlend + dl)
    stops = np.flatnonzero(end0 | ((lim >= 1) & end1))
    ilast = int(stops[0]) if stops.size else n                       # records [0, ilast) are processed
    # LIM < 1 reads DELLIM out of bounds: the scalar loop stops with an error at the first such record that
    # it reaches (i.e. before the end of the loop and not itself ending it through WLEND+DELLIM(1))
    badlim = np.flatnonzero(~end0 & (lim < 1))
    if badlim.size and int(badlim[0]) < ilast:
        i = int(badlim[0])
        raise ValueError(f"rgfalllinesnew: LINESIZE={linesize[i]} reads DELLIM({lim[i]}) out of bounds")
    keep = (np.arange(n) < ilast) & ~lo & (auto != b"COR")
    done = bool(stops.size)
    if not keep.any():
        return None, done
    k = np.flatnonzero(keep)
    b = a[k]
    sel = dict(wlvac=wlvac[k], code=code[k], e=e[k], ep=ep[k], cont=cont[k], auto=auto[k], lim=lim[k],
               gflog=fcol(b, 11, 18, 3), xj=fcol(b, 36, 41, 1), xjp=fcol(b, 64, 69, 1), gr=fcol(b, 80, 86, 2),
               gs=fcol(b, 86, 92, 2), gw=fcol(b, 92, 98, 2), nblo=icol(b, 102, 104), nbup=icol(b, 104, 106),
               iso1=icol(b, 106, 109), x1=fcol(b, 109, 115, 3), iso2=icol(b, 115, 118), x2=fcol(b, 118, 124, 3),
               alpha=fcol(b, 160, 166, 3))
    return sel, done


def _records(cols: dict, potion: np.ndarray, ratiolg: float, ixwlbeg: int, codex: np.ndarray, carry: dict):
    """fort.12 and fort.19 records of the selected lines of one block (record order)."""
    n = len(cols["wlvac"])
    A = {k: np.asarray(cols[k]).astype(F4) for k in ("gflog", "xj", "xjp", "gr", "gs", "gw", "x1", "x2", "alpha")}
    A["code"] = cols["code"].astype(F4)
    wlvac = cols["wlvac"].astype(np.float64)
    e, ep = cols["e"], cols["ep"]
    cont = cols["cont"]
    auto = cols["auto"]
    iso1, iso2 = cols["iso1"], cols["iso2"]
    code, gr, gs, gw = A["code"], A["gr"], A["gs"], A["gw"]
    z4 = F4(0.0)

    with np.errstate(all="ignore"):
        ten = F4(10.0)
        gf = powf(ten, ((A["gflog"] + z4) + A["x1"]) + A["x2"])          # 10.**(GFLOG+DGFLOG+X1+X2)
        elo = np.minimum(np.abs(e), np.abs(ep)).astype(F4)
        gammar = powf(ten, gr + z4)
        gammas = powf(ten, gs + z4)
        gammaw = powf(ten, gw + z4)
        m = (auto == b"AUT") & (gs > 0)
        gammas = np.where(m, -powf(ten, -gs + z4), gammas)
        m = gr == 0
        gammar = np.where(m, (2.223e13 / (wlvac * wlvac)).astype(F4), gammar)
        gr = np.where(m, log10f(gammar), gr)

        nelem = np.trunc(code).astype(np.int64)
        icharge = np.trunc(((code - nelem.astype(F4)) * F4(100.0)) + F4(0.1)).astype(np.int64)
        zeff = (icharge + 1).astype(F4)
        nelion = nelem * 6 - 6 + np.trunc(zeff).astype(np.int64)
        nelion = np.where((nelem > 19) & (nelem < 29) & (icharge > 5), 6 * (nelem + icharge * 10 - 30) - 1, nelion)

        # Stark (GS=0) and van der Waals (GW=0) guesses
        iz = nelem
        index = np.where(iz <= 30, iz * (iz + 1) // 2 + icharge, iz * 5 + 341 + icharge)
        pot = potion[np.clip(index - 1, 0, potion.size - 1)]
        eup = np.maximum(np.abs(e), np.abs(ep)).astype(F4)
        deleup = (pot - eup.astype(np.float64)).astype(F4)
        z2 = zeff * zeff
        mol = code >= F4(100.0)
        # GS block: EFFNSQ=109737.31*ZEFF**2/DELEUP (REAL*4); GAMMAS=1.0D-8*EFFNSQ**2*SQRT(EFFNSQ)
        eff = np.where(deleup > 0, (F4(109737.31) * z2) / deleup, F4(25.0)).astype(F4)
        gam_s = ((1.0e-8 * (eff * eff).astype(np.float64)) * np.sqrt(eff).astype(np.float64)).astype(F4)
        do_gs = gs == 0
        gammas = np.where(do_gs & ~mol, gam_s, np.where(do_gs & mol, F4(1.0e-5), gammas))
        # GW block: EFFNSQ=109737.31D0*ZEFF**2/DELEUP (REAL*8 constant here)
        eff = np.where(deleup > 0, (109737.31 * z2.astype(np.float64) / deleup.astype(np.float64)).astype(F4),
                       F4(25.0))
        eff = np.minimum(eff, F4(1000.0))
        t = eff / zeff
        rsqup = F4(2.5) * (t * t)
        delelo = (pot - elo.astype(np.float64)).astype(F4)
        eff = (109737.31 * z2.astype(np.float64) / delelo.astype(np.float64)).astype(F4)
        eff = np.minimum(eff, F4(1000.0))
        t = eff / zeff
        rsqlo = F4(2.5) * (t * t)
        nseq = np.trunc((code - zeff) + F4(1.0)).astype(np.int64)
        m = (nseq > 20) & (nseq < 29)
        rsqup = np.where(m, (F4(45.0) - nseq.astype(F4)) / zeff, rsqup)
        rsqlo = np.where(m | cont, z4, rsqlo)
        rsqup = np.where(rsqup < rsqlo, F4(2.0) * rsqlo, rsqup)
        gam_w = (4.5e-9 * powf(rsqup - rsqlo, F4(0.4)).astype(np.float64)).astype(F4)
        do_gw = gw == 0
        gammaw = np.where(do_gw & ~mol, gam_w,
                          np.where(do_gw & mol, (1.0e-7 / zeff.astype(np.float64)).astype(F4), gammaw))
        need_pot = (do_gs | do_gw) & ~mol
        if bool((need_pot & ((index < 1) | (index > potion.size))).any()):
            raise ValueError("rgfalllinesnew: POTION index out of bounds")

        typ = np.zeros(n, np.int64)
        typ = np.where(code == F4(1.0), -1, typ)
        typ = np.where((code == F4(1.0)) & (iso1 == 2), -2, typ)
        typ = np.where(code == F4(2.0), -3, typ)
        typ = np.where((code == F4(2.0)) & (iso1 == 3), -4, typ)
        typ = np.where(code == F4(2.01), -6, typ)
        typ = np.where(auto == b"COR", 2, typ)
        typ = np.where(auto == b"AUT", 1, typ)
        typ = np.where(auto == b"PRD", 3, typ)
        typ = np.where(cont, np.trunc(A["xjp"]).astype(np.int64), typ)          # NLAST=XJP (EQUIVALENCE)
        gf = np.where(cont, gf * ((A["xj"] + A["xj"]) + F4(1.0)), gf)
        ncon = np.where((iso1 == 0) & (iso2 > 0), iso2, 0)

        conv = ~((typ == 1) | (typ > 3))
        frelin = (C_SPEED / wlvac).astype(F4)
        cgf = np.where(conv, (C_CGF * gf.astype(np.float64) / frelin.astype(np.float64)).astype(F4), gf)
        t2 = conv & (typ == 2)
        sc = conv & ~t2

        def scale(g):
            return np.where(sc, (g.astype(np.float64) / C_125664 / frelin.astype(np.float64)).astype(F4), g)
        gammar = np.where(t2, gr, scale(gammar))
        gammas = scale(gammas)
        gammaw = scale(gammaw)

    nbuff = nbuff_of(wlvac, ratiolg, ixwlbeg) if n else np.zeros(0, np.int32)
    # label 17 (every line but the NLAST=TYPE=2 continua): NBLO/NBUP made positive, NELIONX=0; TYPE.NE.1 lines
    # with NBLO+NBUP=0 go to fort.12 (1260), the others get NELIONX from CODEX (BAD CODE stops the program).
    # The TYPE=2 continua jump past it and keep the previous line's NELIONX.
    nblo = np.where(t2, cols["nblo"], np.abs(cols["nblo"]))
    nbup = np.where(t2, cols["nbup"], np.abs(cols["nbup"]))
    is12 = ~t2 & (typ != 1) & (nblo + nbup == 0)
    look = ~t2 & (typ != 1) & ~is12
    cidx = np.full(n, -1, np.int64)
    for k in range(len(codex) - 1, -1, -1):
        cidx = np.where(code == codex[k], k, cidx)
    if (look & (cidx < 0)).any():
        i = int(np.flatnonzero(look & (cidx < 0))[0])
        raise ValueError(f"rgfalllinesnew: BAD CODE {float(code[i]):10.2f} (the Fortran stops here)")
    nx = np.where(look, cidx + 1, 0)                  # NELIONX after label 17 (non-continuum lines)
    # TYPE=2 continua: the value left by the last line through label 17 (forward fill, carried across blocks)
    src = np.where(~t2, np.arange(n), -1)
    src = np.maximum.accumulate(src) if n else src
    nelionx = np.where(t2, np.where(src >= 0, nx[np.maximum(src, 0)], carry["nelionx"]), nx)
    if n:
        carry["nelionx"] = int(nelionx[-1])

    k = is12
    recs12 = pack12(nbuff[k], cgf[k], nelion[k].astype(np.int32), elo[k], gammar[k], gammas[k], gammaw[k],
                    A["alpha"][k])
    k = ~is12
    r19 = np.empty(int(k.sum()), FORT19)
    for name, v in (("wlvac", wlvac), ("elo", elo), ("gf", cgf), ("nblo", nblo), ("nbup", nbup),
                    ("nelion", nelion), ("type", typ), ("ncon", ncon), ("nelionx", nelionx), ("gammar", gammar),
                    ("gammas", gammas), ("gammaw", gammaw), ("alpha", A["alpha"]), ("nbuff", nbuff),
                    ("lim", cols["lim"])):
        r19[name] = v[k]
    return recs12, r19
