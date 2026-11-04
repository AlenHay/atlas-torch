"""Fortran DATA tables of the ATLAS9/KAPPA9 physics library, loaded from the fetched sources.

Nothing is transcribed: DATA statements are parsed by atlas_torch.kurucz_data; tables that the Fortran
assembles through EQUIVALENCE (PFSAHA's NNN, PFIRON's PFTAB, IONPOTS' POTION, CHOP/OHOP cross-sections, ...)
are assembled here from the EQUIVALENCE statements of the same source, so the result is the exact
storage image the compiled program sees (validated against the oracle .so images in the tests).

Variant -> source file: 'atlas9' = atlas9mem.patched.for (BasicATLAS-patched ATLAS9), 'kappa9' = kappa9.for.
"""
from __future__ import annotations

import os
import re
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

from ...kurucz_data import _read_statements, load_tables

VARIANTS = ("atlas9", "kappa9")
SOURCES = {"atlas9": "atlas9mem.patched.for", "kappa9": "kappa9.for"}


def source_path(variant: str) -> Path:
    if variant not in SOURCES:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    ext = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
    return ext / "builds" / "src" / SOURCES[variant]


@lru_cache(maxsize=None)
def _tables(variant: str):
    t = dict(load_tables(source_path(variant)))
    t.update(_unspaced_data(variant))
    return t


def _unspaced_data(variant: str) -> dict:
    """DATA statements written without a blank after DATA (fixed form ignores blanks: 'DATAC10/.../' in
    CHOP/OHOP) are missed by kurucz_data's parser; re-parse them from a normalised copy of their units."""
    import tempfile
    from ...kurucz_data import parse_data
    units = dict(_read_statements(source_path(variant)))
    out = {}
    for unit, stmts in units.items():
        bad = [x for x in stmts if re.match(r"^\s*DATA[A-Za-z]", x, re.I)]
        if not bad:
            continue
        keep = [x for x in stmts if re.match(r"^\s*(IMPLICIT|PARAMETER|DIMENSION|REAL|INTEGER|DOUBLE|CHARACTER|LOGICAL|COMMON)", x, re.I)]
        body = [stmts[0]] + keep + [re.sub(r"^(\s*DATA)", r"\1 ", x, flags=re.I) for x in bad] + ["END"]
        lines = []
        for st in body:
            st = st.strip()
            lines.append("      " + st[:60])
            st = st[60:]
            while st:
                lines.append("     1" + st[:60])
                st = st[60:]
        with tempfile.NamedTemporaryFile("w", prefix=f"{variant}_", suffix=".for", delete=False) as fh:
            fh.write("\n".join(lines) + "\n")
            tmp = fh.name
        try:
            for (u, var), arr in parse_data(tmp).items():
                if u == unit:
                    out[(u, var)] = arr
        finally:
            os.unlink(tmp)
    return out


def table(variant: str, unit: str, var: str) -> np.ndarray:
    return _tables(variant)[(unit.lower(), var.lower())]


@lru_cache(maxsize=None)
def _statements(variant: str) -> dict:
    return dict(_read_statements(source_path(variant)))


def _dims(variant: str, unit: str) -> dict[str, tuple]:
    """Declared array shapes in a unit (DIMENSION / type statements, numeric bounds only)."""
    out = {}
    for s in _statements(variant)[unit]:
        m = re.match(r"^\s*(DIMENSION|REAL\*8|REAL\*4|REAL|INTEGER|DOUBLE\s+PRECISION|COMMON)\b(.*)$", s, re.I)
        if not m or re.match(r"^\s*\w[\w*]*\s+FUNCTION", s, re.I):
            continue
        for name, dims in re.findall(r"(\w+)\s*\(([\d,\s]+)\)", m.group(2)):
            out[name.lower()] = tuple(int(d) for d in dims.split(","))
    return out


def _lin(idx: tuple, shape: tuple) -> int:
    """0-based column-major linear index of a 1-based Fortran subscript."""
    k, mul = 0, 1
    for i, n in zip(idx, shape):
        k += (i - 1) * mul
        mul *= n
    return k


@lru_cache(maxsize=None)
def equivalenced(variant: str, unit: str, big: str, dtype=None) -> np.ndarray:
    """Assemble array `big` of `unit` from the DATA-initialised arrays EQUIVALENCEd into it.

    Handles chains through an intermediate array (e.g. NNN <-> NNNvec <-> NNN01..). Returns a flat
    (column-major) view reshaped to the declared shape of `big`.
    """
    big = big.lower()
    dims = _dims(variant, unit)
    pairs = []
    for s in _statements(variant)[unit]:
        if not re.match(r"^\s*EQUIVALENCE", s, re.I):
            continue
        for a, ia, b, ib in re.findall(r"\(\s*(\w+)\s*\(([\d,\s]+)\)\s*,\s*(\w+)\s*\(([\d,\s]+)\)\s*\)", s):
            pairs.append((a.lower(), tuple(int(x) for x in ia.split(",")),
                          b.lower(), tuple(int(x) for x in ib.split(","))))
    # offsets (in elements) of every array relative to `big`'s first element
    off = {big: 0}
    changed = True
    while changed:
        changed = False
        for a, ia, b, ib in pairs:
            for x, ix, y, iy in ((a, ia, b, ib), (b, ib, a, ia)):
                if x in off and y not in off:
                    sx, sy = dims.get(x, (1,)), dims.get(y, (1,))
                    off[y] = off[x] + _lin(ix, sx) - _lin(iy, sy)
                    changed = True
    tabs = _tables(variant)
    shape = dims[big]
    n = int(np.prod(shape))
    out = None
    for name, o in off.items():
        if (unit, name) not in tabs:
            continue
        sub = tabs[(unit, name)].reshape(-1, order="F")
        if out is None:
            out = np.zeros(n, dtype=dtype or sub.dtype)
        out[o:o + sub.size] = sub
    if (unit, big) in tabs:     # DATA directly on the big array (overrides nothing else here)
        sub = tabs[(unit, big)].reshape(-1, order="F")
        out = sub.copy() if out is None else out
    return out.reshape(shape, order="F")


@lru_cache(maxsize=None)
def potion(variant: str) -> np.ndarray:
    """COMMON /POTION/POTION(999) as initialised by IONPOTS' DATA statements (cm^-1)."""
    return equivalenced(variant, "ionpots", "potion", np.float64)


# ---------------------------------------------------------------------------------------------------
# Fortran REAL*4 literal values (exact decimal -> binary32, round to nearest even)

@lru_cache(maxsize=None)
def r4(text: str) -> float:
    """Value of the default-kind REAL literal `text` (e.g. '13.595') widened to double."""
    t = text.strip().lower().replace("d", "e")
    q = Fraction(t)
    f = np.float32(float(t))
    best = f
    for c in (np.nextafter(f, np.float32(-np.inf)), np.nextafter(f, np.float32(np.inf))):
        dc, db = abs(Fraction(float(c)) - q), abs(Fraction(float(best)) - q)
        if dc < db or (dc == db and (np.frombuffer(np.float32(c).tobytes(), np.uint32)[0] & 1) == 0):
            best = c
    return float(best)


# ---------------------------------------------------------------------------------------------------
# PFGROUND: a 336-way computed-GOTO "table" of closed-form ground-term partition functions. It is code,
# not DATA, but it is a table in all but syntax; it is read from the source (with statement labels) and
# evaluated by a small Fortran expression evaluator with exact literal kinds (see _fexpr).

@lru_cache(maxsize=None)
def _labelled_unit(variant: str, unit: str) -> list[tuple[int | None, str]]:
    """(label, statement) list of one program unit, fixed form, with continuation lines joined."""
    lines = source_path(variant).read_text(encoding="latin-1").split("\n")
    hdr = re.compile(r"^[ \t]+(?:REAL\*8\s+|REAL\s+|INTEGER\s+)?(?:SUBROUTINE|FUNCTION)\s+(\w+)", re.I)
    out, inside = [], False
    for raw in lines:
        if not raw.strip() or raw[0] in "cC*!":
            continue
        if raw.startswith("\t"):
            lab, cont, text = "", raw[1:2] in tuple("123456789"), raw[2:] if raw[1:2] in tuple("123456789") else raw[1:]
        else:
            lab, cont, text = raw[:5], len(raw) > 5 and raw[5] not in " 0", raw[6:72]
        m = hdr.match(raw)
        if m:
            inside = m.group(1).lower() == unit.lower()
        if not inside:
            continue
        if cont and out:
            out[-1] = (out[-1][0], out[-1][1] + text)
        else:
            out.append((int(lab) if lab.strip() else None, text.strip()))
        if re.match(r"^\s*END\s*$", text, re.I):
            break
    return out


@lru_cache(maxsize=None)
def pfground_cases(variant: str) -> dict[int, str]:
    """NELION -> the Fortran expression assigned to PFGROUND on that path (fall-through followed)."""
    from ._fexpr import parse
    stmts = _labelled_unit(variant, "pfground")
    labels = {lab: i for i, (lab, _) in enumerate(stmts) if lab is not None}
    goto_lists = []
    for i, (_, s) in enumerate(stmts):
        m = re.match(r"^GO\s*TO\s*\(([\d,\s]+)\)\s*,?\s*(.*)$", s, re.I)
        if m:
            goto_lists.append(([int(x) for x in m.group(1).split(",")], m.group(2).replace(" ", "").upper()))

    def run_from(i):
        val = None
        while i < len(stmts):
            s = stmts[i][1]
            m = re.match(r"^PFGROUND\s*=(.*)$", s, re.I)
            if m:
                val = m.group(1).strip()
            elif re.match(r"^RETURN\s*$", s, re.I):
                return val
            elif re.match(r"^(ENDIF|END\s*IF|END)$", s, re.I):
                # falling out of the IF block: the later blocks' conditions are disjoint, so control
                # reaches the final default assignment PFGROUND=1.
                return DEFAULT
            i += 1
        return DEFAULT

    DEFAULT = "1."
    cases = {}
    first, second = goto_lists[0], goto_lists[1]
    for k, lab in enumerate(first[0], start=1):
        if k < 114:
            cases[k] = run_from(labels[lab])
    for nel in (431, 491, 501):
        cases[nel] = run_from(labels[nel])
    for k, lab in enumerate(second[0], start=1):
        cases[k + 168] = run_from(labels[lab])
    for k, v in cases.items():
        parse(v)            # validate now
    return cases


def pfground_expr(variant: str, nelion: int) -> str:
    return pfground_cases(variant).get(int(nelion), "1.")
