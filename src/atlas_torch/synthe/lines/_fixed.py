"""Chunked fixed-width (Fortran formatted, sequential) input for the line-list programs.

`iter_records(path, offset, width)` streams a text file in blocks of ~`chunk_bytes` and yields each block as
a uint8 array [n, width]: one row per record, truncated or blank-padded to `width` characters (gfortran's
PAD='YES' for short records; characters past the format are ignored), with the record's trailing CR/LF
stripped as the scalar readers did (`line.rstrip("\\r\\n")`).  Memory is O(chunk), whatever the file size.

`fcol` / `icol` decode a field (columns [lo, hi) of every row) with the Fortran Fw.d / Iw semantics of
`_fnum.fread` / `_fnum.iread` (BLANK='NULL': blanks ignored, all-blank field = 0, no decimal point = d implied
decimals), vectorised and exact: the digits are accumulated into an integer mantissa M (< 10**15, exact in
REAL*8) and the value is M / 10**k (k = digits after the point, or d), a single correctly rounded IEEE
division of two exact doubles = the correctly rounded decimal value = what strtod / float() return.
Fields with any other character (exponents, several signs or points, ...) go through the scalar readers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ._fnum import fread, iread

SP, DOT, MINUS, PLUS = 32, 46, 45, 43
_POW10 = np.array([10.0 ** k for k in range(23)])


def iter_records(path: str | Path, offset: int, width: int, chunk_bytes: int = 32 << 20,
                 skip_partial: bool = False) -> Iterator[np.ndarray]:
    """Yield [n, width] uint8 blocks of the records of `path` from byte `offset` on.

    skip_partial: drop the first (partial) record, as `f.seek(offset); f.readline()` does."""
    with open(path, "rb") as f:
        f.seek(offset)
        rest = b""
        first = skip_partial
        while True:
            data = f.read(chunk_bytes)
            eof = not data
            buf = rest + data
            if not buf:
                return
            if eof:
                end = len(buf)
                if buf[-1:] != b"\n":
                    buf = buf + b"\n"
            else:
                end = buf.rfind(b"\n") + 1
                if end == 0:                       # no complete record yet: read on
                    rest = buf
                    continue
            rest = buf[end:] if not eof else b""
            arr = np.frombuffer(buf, np.uint8, count=end if not eof else len(buf))
            if first:
                k = int(np.argmax(arr == 10)) + 1
                arr = arr[k:]
                first = False
                if not arr.size:
                    if eof:
                        return
                    continue
            yield _to_fixed(arr, width)
            if eof:
                return


def _to_fixed(arr: np.ndarray, width: int, block: int = 65536) -> np.ndarray:
    """Records of `arr` (each terminated by LF) -> [n, width] blank-padded rows."""
    nl = np.flatnonzero(arr == 10)
    starts = np.empty(nl.size, np.int64)
    starts[0] = 0
    starts[1:] = nl[:-1] + 1
    ends = nl.copy()
    # rstrip("\r\n"): strip the trailing CRs (the LF terminators are already excluded)
    while True:
        cr = (ends > starts) & (arr[np.maximum(ends - 1, 0)] == 13)
        if not cr.any():
            break
        ends[cr] -= 1
    lens = ends - starts
    n = nl.size
    rec = nl - starts + 1
    if n and (lens == lens[0]).all() and (rec == rec[0]).all():
        # uniform records: a strided view
        stride = int(rec[0])
        L = int(lens[0])
        m = min(L, width)
        out = np.full((n, width), SP, np.uint8)
        view = np.lib.stride_tricks.as_strided(arr, shape=(n, m), strides=(stride, 1))
        out[:, :m] = view
        return out
    out = np.full((n, width), SP, np.uint8)
    col = np.arange(width, dtype=np.int64)
    for a in range(0, n, block):
        s = starts[a:a + block]
        ln = np.minimum(lens[a:a + block], width)
        idx = s[:, None] + col[None, :]
        ok = col[None, :] < ln[:, None]
        np.clip(idx, 0, arr.size - 1, out=idx)
        out[a:a + block] = np.where(ok, arr[idx], SP)
    return out


def _rows_str(a: np.ndarray, rows: np.ndarray) -> list[str]:
    return [bytes(a[r]).decode("latin1") for r in rows]


def fcol(a: np.ndarray, lo: int, hi: int, d: int) -> np.ndarray:
    """Fortran Fw.d input of columns [lo, hi) of every row of `a` (float64)."""
    c = a[:, lo:hi]
    n, w = c.shape
    isd = (c >= 48) & (c <= 57)
    isdot = c == DOT
    issg = (c == MINUS) | (c == PLUS)
    other = ~(isd | isdot | issg | (c == SP))
    ndot = isdot.sum(1)
    nsg = issg.sum(1)
    nd = isd.sum(1)
    # a sign must come before every digit and the point
    pos = np.arange(w)
    big = w + 1
    first_num = np.where(isd | isdot, pos, big).min(1)
    sgpos = np.where(issg, pos, -1).max(1)
    bad = other.any(1) | (ndot > 1) | (nsg > 1) | ((nsg == 1) & (sgpos > first_num)) | (nd > 15) | \
        ((nd == 0) & ((ndot > 0) | (nsg > 0)))
    mant = np.zeros(n, np.int64)
    kdig = np.zeros(n, np.int64)
    seen = np.zeros(n, bool)
    for j in range(w):
        dj = isd[:, j]
        mant = np.where(dj, mant * 10 + (c[:, j].astype(np.int64) - 48), mant)
        kdig += dj & seen
        seen |= isdot[:, j]
    k = np.where(ndot > 0, kdig, d)
    v = mant.astype(np.float64) / _POW10[np.minimum(k, 22)]
    v = np.where(((c == MINUS).any(1)) & (nd > 0), -v, v)
    if bad.any():
        rows = np.flatnonzero(bad)
        v[rows] = [fread(s, d) for s in _rows_str(c, rows)]
    return v


def icol(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Fortran Iw input of columns [lo, hi) of every row of `a` (int64)."""
    c = a[:, lo:hi]
    n, w = c.shape
    isd = (c >= 48) & (c <= 57)
    issg = (c == MINUS) | (c == PLUS)
    other = ~(isd | issg | (c == SP))
    pos = np.arange(w)
    first_d = np.where(isd, pos, w + 1).min(1)
    sgpos = np.where(issg, pos, -1).max(1)
    nsg = issg.sum(1)
    nd = isd.sum(1)
    bad = other.any(1) | (nsg > 1) | ((nsg == 1) & ((sgpos > first_d) | (nd == 0))) | (nd > 18)
    v = np.zeros(n, np.int64)
    for j in range(w):
        dj = isd[:, j]
        v = np.where(dj, v * 10 + (c[:, j].astype(np.int64) - 48), v)
    v = np.where((c == MINUS).any(1), -v, v)
    if bad.any():
        rows = np.flatnonzero(bad)
        v[rows] = [iread(s) for s in _rows_str(c, rows)]
    return v


def scol(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Columns [lo, hi) as fixed-length bytes (dtype S<hi-lo>; trailing blanks kept, compare with b'...')."""
    return np.ascontiguousarray(a[:, lo:hi]).view(f"S{hi - lo}").ravel()
