"""DATA tables of the ATLAS9 variant, read from the fetched (patched) Fortran source at run time.

`kurucz_data.table` handles unit-local arrays. Two cases it does not cover are handled here with its own
statement reader / value converter (so literal conversion semantics, e.g. a default-REAL literal stored into a
REAL*8 array = single precision widened, stay in one place):
  * arrays dimensioned in a COMMON statement (ROSSTAB's TABT/TABP/TABV in /TABTP/), returned 0-d by kurucz_data;
  * arrays assembled through EQUIVALENCE of DATA-initialised pieces (BLOCKJ/BLOCKH COEFJ/COEFH, BLOCKBIG's
    WAVEBIG/WAVELIT, BLOCKR's FRESET <- WBIG <- WBIG1..3).
Validated bit-for-bit against the compiled library's initial image in tests/oracle/test_atlas9_data.py.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

SRC = "atlas9mem.patched.for"


def source_path() -> Path:
    ext = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
    return ext / "builds" / "src" / SRC


@lru_cache(maxsize=None)
def _units() -> dict[str, list[str]]:
    from ..kurucz_data import _read_statements
    return {u: s for u, s in _read_statements(source_path())}


@lru_cache(maxsize=None)
def _all_tables() -> dict:
    """kurucz_data.load_tables once (its per-variable front-end re-reads the whole npz cache each time)."""
    try:
        from ..kurucz_data import load_tables
        return load_tables(source_path())
    except ImportError:
        return {}


def table(unit: str, var: str) -> np.ndarray:
    t = _all_tables()
    key = (unit.lower(), var.lower())
    if key in t:
        return t[key]
    from ..tables import table as _t
    return _t(SRC, unit, var)


@lru_cache(maxsize=None)
def common_data(unit: str, var: str, n: int, dtype: str = "f8") -> np.ndarray:
    """DATA-initialised array `var(n)` of `unit` whose dimension is declared in a COMMON statement."""
    from ..kurucz_data import _convert_val, _split_data_pairs, _split_val_list
    dt = np.dtype(dtype)
    for st in _units()[unit.lower()]:
        if not re.match(r"^\s*DATA\b", st, re.I):
            continue
        for names, vals in _split_data_pairs(st):
            if names.replace(" ", "").lower() != var.lower():
                continue
            out = []
            for tok in _split_val_list(vals):
                m = re.match(r"^\s*(\d+)\s*\*(.*)$", tok)
                if m:
                    out += [_convert_val(m.group(2), dt, {})] * int(m.group(1))
                else:
                    out.append(_convert_val(tok, dt, {}))
            a = np.array(out, dtype=dt)
            assert a.size == n, (unit, var, a.size, n)
            return a
    raise KeyError((unit, var))


def _equivalences(unit: str) -> list[tuple[str, int, str, int]]:
    """EQUIVALENCE (A(i),B(j)) pairs of `unit` as (a, i, b, j) (1-based, 1-D subscripts, blanks removed)."""
    out = []
    for st in _units()[unit.lower()]:
        s = st.replace(" ", "").upper()
        if not s.startswith("EQUIVALENCE"):
            continue
        for a, i, b, j in re.findall(r"\(([A-Z0-9_]+)\((\d+)\),([A-Z0-9_]+)\((\d+)\)\)", s):
            out.append((a.lower(), int(i), b.lower(), int(j)))
    return out


@lru_cache(maxsize=None)
def assembled(unit: str, target: str, n: int, dtype: str = "f8", _parents: tuple = ()) -> np.ndarray:
    """Array `target(n)` of `unit` (COMMON member or local) built from the DATA pieces equivalenced into it.

    Pieces are placed recursively (a piece may itself be assembled from smaller pieces); elements not covered
    by any DATA stay zero (COMMON/.bss default)."""
    _t = lambda src, u, v: table(u, v)
    out = np.zeros(n, dtype=dtype)
    eqs = _equivalences(unit)
    for a, i, b, j in eqs:
        if b == target and a != target:
            piece, off = a, j - i
        elif a == target and b != target:
            piece, off = b, i - j
        else:
            continue
        if piece in _parents:
            continue
        try:
            v = np.asarray(_t(SRC, unit, piece)).ravel()
        except KeyError:
            v = None
        if v is None or v.ndim == 0 or v.size <= 1:
            # piece without its own DATA (e.g. WBIG = WBIG1//WBIG2//WBIG3): assemble it from its sub-pieces
            v = assembled(unit, piece, _declared_size(unit, piece), dtype, _parents + (target,))
        out[off:off + v.size] = v.astype(dtype)
    return out


def _declared_size(unit: str, var: str) -> int:
    for st in _units()[unit.lower()]:
        s = st.replace(" ", "").upper()
        if s.startswith("DIMENSION"):
            m = re.search(r"(?:^DIMENSION|,)" + var.upper() + r"\((\d+)\)", s)
            if m:
                return int(m.group(1))
    raise KeyError((unit, var))


# ------------------------------------------------------------------ named tables used by the port

@lru_cache(maxsize=None)
def rosstab_grid() -> dict[str, np.ndarray]:
    return {"tabt": common_data("rosstab", "tabt", 57), "tabp": common_data("rosstab", "tabp", 42),
            "tabv": common_data("rosstab", "tabv", 5)}


@lru_cache(maxsize=None)
def josh_tables() -> dict[str, np.ndarray]:
    cj = assembled("blockj", "cj", 2601, "f4").reshape(51, 51, order="F")   # COEFJ(L,M)
    ch = assembled("blockh", "ch", 2601, "f4").reshape(51, 51, order="F")   # COEFH(L,M)
    return {"coefj": cj, "coefh": ch, "ch": table("josh", "ch").astype(np.float32),
            "ck": table("josh", "ck").astype(np.float32), "xtau8": table("josh", "xtau8").astype(np.float64)}


@lru_cache(maxsize=None)
def wavedf() -> np.ndarray:
    """COMMON /BIGLIT/ WAVEDF(1213,2): column 1 = WAVELIT, column 2 = WAVEBIG (nm)."""
    lit = assembled("blockbig", "wavelit", 1213)
    big = assembled("blockbig", "wavebig", 1213)
    return np.stack([lit, big], axis=1)


@lru_cache(maxsize=None)
def freset_init() -> np.ndarray:
    """Initial COMMON /FRESET/ FRESET(1563) (wavelengths of the BIG/LITTLE sets at offsets 6 / 343)."""
    return assembled("blockr", "freset", 1563)


@lru_cache(maxsize=None)
def linop_tables() -> dict[str, np.ndarray]:
    return {"tabt": table("linop", "tabt"), "tabp": table("linop", "tabp"),
            "tenlog": np.float64(table("linop", "tenlog"))}


@lru_cache(maxsize=None)
def expi_coeffs() -> dict[str, float]:
    names = "a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 b4 c0 c1 c2 c3 c4 c5 c6 d1 d2 d3 d4 d5 d6 " \
            "e0 e1 e2 e3 e4 e5 e6 f1 f2 f3 f4 f5 f6".split()
    return {k: float(table("expi", k)) for k in names}


@lru_cache(maxsize=None)
def readin_ifop1() -> np.ndarray:
    return table("readin", "ifop1").astype(np.int64)
