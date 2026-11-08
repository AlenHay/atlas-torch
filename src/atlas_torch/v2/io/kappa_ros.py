"""Rosseland-mean tables: kappa.ros (kapreadts.for, read by ATLAS9 READ KAPPA) and the KAPPA9 per-vturb tables
kapk{v}.dat (kappa9.for WRITE(7,75)/(7,77): Fortran DATA statements of 1000 log10 kappa_R)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...io.formats import read_kapk as _read_kapk
from ...io.fortran import ffloat

TITLE = " ROSSELAND MASS ABSORPTION COEFFICIENTS"
HEADER = " logT logP  0km/s  1km/s  2km/s  4km/s  8km/s  log Ne   log Na  log rho IPloCM-1"
_ROW = (5, 5, 7, 7, 7, 7, 7, 9, 9, 9, 8)          # kapreadts.for FORMAT 5: f5.2,f5.2,5F7.3,3F9.5,f8.3


def fortran_f(x: float, w: int, d: int) -> str:
    """Fortran Fw.d output (leading zero dropped when the field is full, asterisks on overflow)."""
    s = f"{x:{w}.{d}f}"
    if len(s) > w and s.startswith("-0."):
        s = "-" + s[2:]
    elif len(s) > w and s.startswith("0."):
        s = s[1:]
    return s if len(s) <= w else "*" * w


def read_kappa_ros(path) -> dict:
    """title (first line), log_t (57,), log_p (42,), log_kappa (5, 57, 42) (vturb first, log10 cm^2 g^-1),
    extra (57, 42, 4): log Ne, log Na, log rho, IPlo (cm^-1)."""
    lines = Path(path).read_text().splitlines()
    rows = []
    for ln in lines[2:]:
        if not ln.strip():
            continue
        vals, i = [], 0
        for w in _ROW:
            vals.append(ffloat(ln[i:i + w]))
            i += w
        rows.append(vals)
    a = np.array(rows, dtype=np.float64).reshape(57, -1, len(_ROW))
    return {"title": lines[0], "log_t": a[:, 0, 0].copy(), "log_p": a[0, :, 1].copy(),
            "log_kappa": np.moveaxis(a[:, :, 2:7], -1, 0).copy(), "extra": a[:, :, 7:].copy()}


def write_kappa_ros(path, log_t, log_p, log_kappa, extra=None, title: str = TITLE) -> None:
    """log_kappa (5, 57, 42) dex; extra (57, 42, 4) or None (zeros and IPlo = 1, as kapreadts writes)."""
    lk = np.asarray(log_kappa, dtype=np.float64)
    nt, np_ = len(log_t), len(log_p)
    if extra is None:
        extra = np.zeros((nt, np_, 4))
        extra[..., 3] = 1.0
    out = [title, HEADER]
    for it in range(nt):
        for ip in range(np_):
            row = fortran_f(log_t[it], 5, 2) + fortran_f(log_p[ip], 5, 2)
            row += "".join(fortran_f(lk[iv, it, ip], 7, 3) for iv in range(lk.shape[0]))
            row += "".join(fortran_f(extra[it, ip, k], 9, 5) for k in range(3)) + fortran_f(extra[it, ip, 3], 8, 3)
            out.append(row)
    Path(path).write_text("\n".join(out) + "\n")


def read_kapk(path) -> np.ndarray:
    """int (42, 57): 1000 log10 kappa_R per pressure (rows) and temperature."""
    return _read_kapk(path)


def _quantize(log_kappa, rounding: str) -> np.ndarray:
    x = np.round(np.asarray(log_kappa, dtype=np.float64) * 1000.0, 6)
    if rounding == "truncate":                       # KAPPA9: KABROS = INT(LOG10(ABROSS)*1000.)
        return np.trunc(x).astype(np.int64)
    if rounding == "nearest":
        return np.round(x).astype(np.int64)
    raise ValueError("rounding: 'nearest' | 'truncate'")


def write_kapk(path, log_t, log_p, log_kappa, rounding: str = "nearest") -> None:
    """log_kappa (57, 42) dex -> kapk text (kappa9.for FORMAT 75 header, FORMAT 77 per pressure)."""
    k = _quantize(log_kappa, rounding).T                     # (42, 57)
    f63 = lambda v: fortran_f(float(v), 6, 3)                # noqa: E731
    i5 = lambda v: f"{int(v):5d}" if -9999 <= v <= 99999 else "*****"   # noqa: E731
    tt, tp = [f63(v) for v in log_t], [f63(v) for v in log_p]
    lines = ["      DATA TABT/    " + "".join(v + "," for v in tt[0:7])]
    for r, i in zip(range(1, 6), range(7, 52, 9)):
        lines.append(f"     {r}" + "".join(v + "," for v in tt[i:i + 9]))
    lines.append("     6" + "".join(v + "," for v in tt[52:56]) + tt[56] + "/")
    lines.append("      DATA TABP/    " + "".join(v + "," for v in tp[0:7]))
    for r, i in zip(range(1, 4), range(7, 34, 9)):
        lines.append(f"     {r}" + "".join(v + "," for v in tp[i:i + 9]))
    lines.append("     4" + "".join(v + "," for v in tp[34:41]) + tp[41] + "/")
    for ip in range(len(log_p)):
        p = tp[ip]
        lines.append(f"      DATA KTAB{ip + 1:02d}/" + " " * 54 + p)
        for r in range(1, 6):
            lines.append(f"     {r}" + "".join(i5(v) + "," for v in k[ip, (r - 1) * 10:r * 10]) + " " * 6 + p)
        lines.append("     6" + "".join(i5(v) + "," for v in k[ip, 50:56]) + i5(k[ip, 56]) + "/" + " " * 24 + p)
    Path(path).write_text("\n".join(lines) + "\n")
