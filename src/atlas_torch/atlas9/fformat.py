"""Fortran formatted output as gfortran writes it (edit descriptors Iw, Fw.d, [kP]Ew.d, Aw).

Rounding: gfortran gets its digits from the C library's correctly rounded conversion (round-half-even on exact
binary ties), as does Python's float formatting, so the digit strings agree.
Non-finite values: gfortran writes NaN / Infinity / Inf right-justified (asterisks if they do not fit).
Carriage control characters (1H0, 1H1, ...) are written literally (gfortran does not interpret them).
"""
from __future__ import annotations

import math


def _stars(w: int) -> str:
    return "*" * w


def _nonfinite(v: float, w: int) -> str:
    if math.isnan(v):
        s = "NaN"
    else:
        neg = v < 0
        s = ("-" if neg else "") + ("Infinity" if w >= 8 + neg else "Inf")
    return s.rjust(w) if len(s) <= w else _stars(w)


def I(v, w: int) -> str:
    s = str(int(v))
    return s.rjust(w) if len(s) <= w else _stars(w)


def F(v, w: int, d: int) -> str:
    v = float(v)
    if not math.isfinite(v):
        return _nonfinite(v, w)
    s = f"{abs(v):.{d}f}" + ("." if d == 0 else "")
    neg = math.copysign(1.0, v) < 0
    if s.startswith("0.") and d > 0:
        body = s
        if len(body) + (1 if neg else 0) > w:
            body = s[1:]          # drop the optional leading zero
    else:
        body = s
    out = ("-" if neg else "") + body
    return out.rjust(w) if len(out) <= w else _stars(w)


def E(v, w: int, d: int, k: int = 0) -> str:
    """[kP]Ew.d with k in {0, 1} (the scale factors used by ATLAS9)."""
    v = float(v)
    if not math.isfinite(v):
        return _nonfinite(v, w)
    neg = math.copysign(1.0, v) < 0
    a = abs(v)
    ndig = d if k <= 0 else d + 1
    if a == 0.0:
        mant, ex = "0" * ndig, 0
    else:
        s = f"{a:.{ndig - 1}e}"
        m, e = s.split("e")
        mant = m.replace(".", "")
        ex = int(e) + 1            # value = 0.mant * 10**ex
    if k <= 0:
        body = "0." + mant
        expo = ex if a != 0.0 else 0
    else:
        body = mant[:k] + "." + mant[k:]
        expo = ex - k if a != 0.0 else 0
    if abs(expo) <= 99:
        es = "E" + ("-" if expo < 0 else "+") + f"{abs(expo):02d}"
    elif abs(expo) <= 999:
        es = ("-" if expo < 0 else "+") + f"{abs(expo):03d}"
    else:
        return _stars(w)
    out = ("-" if neg else "") + body + es
    if len(out) > w and k <= 0:
        out = ("-" if neg else "") + body[1:] + es
    return out.rjust(w) if len(out) <= w else _stars(w)


def A(s, w: int | None = None) -> str:
    if isinstance(s, (bytes, bytearray)):
        s = s.decode("latin-1")
    s = str(s)
    if w is None:
        return s
    return s[:w] if len(s) >= w else s.rjust(w)


def X(n: int) -> str:
    return " " * n
