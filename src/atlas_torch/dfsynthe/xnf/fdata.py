"""DATA tables of xnfdf.for, loaded from the fetched Fortran source (never transcribed).

Tables split into many DATA arrays and glued together by EQUIVALENCE (NNN of PFSAHA, POTION of IONPOTS,
PFTAB of PFIRON, the Karzas-Latter tables of XKARSAS, FF of HMINOP) are assembled here by parsing the
unit's EQUIVALENCE statements. PFGROUND (a computed GO TO over 113 closed formulas) is compiled from its
source text into vectorised torch expressions with the Fortran's literal kinds (default REAL constants
are single precision, D constants double).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ...kurucz_data import _read_statements, load_tables

SRC = "xnfdf.for"


def source_path() -> Path:
    ext = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
    return ext / "builds" / "src" / SRC


@lru_cache(maxsize=None)
def _all_tables() -> dict:
    # kurucz_data.load_tables re-reads its whole npz cache per call: load once here
    return load_tables(source_path())


def T(unit: str, var: str) -> np.ndarray:
    key = (unit.lower(), var.lower())
    tabs = _all_tables()
    if key not in tabs:
        raise KeyError(f"{var} not in unit {unit} of {SRC}")
    return tabs[key]


@lru_cache(maxsize=None)
def _unit_statements() -> dict[str, list[str]]:
    u: dict[str, list[str]] = {}
    for name, st in _read_statements(source_path()):
        u.setdefault(name, []).extend(st)
    return u


_REF = re.compile(r"\s*([A-Za-z][A-Za-z0-9_ ]*?)\s*\(([^()]*)\)\s*")


def _equiv_pairs(unit: str):
    for s in _unit_statements()[unit.lower()]:
        if not s.strip().upper().startswith("EQUIVALENCE"):
            continue
        body = s.strip()[len("EQUIVALENCE"):]
        for grp in re.findall(r"\(((?:[^()]|\([^()]*\))*)\)", body):
            refs = _REF.findall(grp)
            yield [(n.replace(" ", "").lower(), [int(x) for x in sub.split(",")]) for n, sub in refs]


def _lin(sub, shape):
    off, mul = 0, 1
    for i, n in zip(sub, shape):
        off += (i - 1) * mul
        mul *= n
    return off if len(sub) > 1 else sub[0] - 1


def equiv_assemble(unit: str, target: str, shape: tuple, dtype, alias: dict | None = None) -> np.ndarray:
    """Fill array `target` (Fortran shape) of `unit` from the DATA arrays EQUIVALENCEd into it.

    alias maps another equivalenced name (e.g. P63) to (shape) of the same storage starting at target(1)."""
    flat = np.zeros(int(np.prod(shape)), dtype=dtype)
    shapes = {target: shape}
    if alias:
        shapes.update(alias)
    for pair in _equiv_pairs(unit):
        names = [n for n, _ in pair]
        tgt = [p for p in pair if p[0] in shapes]
        src = [p for p in pair if p[0] not in shapes]
        if not tgt or not src:
            continue
        (tn, tsub), (sn, ssub) = tgt[0], src[0]
        try:
            data = T(unit, sn)
        except KeyError:
            continue
        a = np.asarray(data).reshape(-1, order="F")
        o = _lin(tsub, shapes[tn]) - _lin(ssub, data.shape if data.ndim else (1,))
        flat[o:o + a.size] = a.astype(dtype)
    return flat.reshape(shape, order="F")


@lru_cache(maxsize=None)
def nnn() -> np.ndarray:
    """PFSAHA NNN(6,365) (+NNN67 overlay at 2119)."""
    return equiv_assemble("pfsaha", "nnn", (6, 365), np.int64)


@lru_cache(maxsize=None)
def potion() -> np.ndarray:
    """COMMON /POTION/POTION(999) as initialised by IONPOTS' EQUIVALENCEd DATA (cm-1)."""
    return equiv_assemble("ionpots", "potion", (999,), np.float64)


@lru_cache(maxsize=None)
def pftab() -> np.ndarray:
    """PFIRON PFTAB(7,56,10,9) REAL*4 (storage via P63(63,560))."""
    p63 = equiv_assemble("pfiron", "p63", (63, 560), np.float32)
    return p63.reshape(-1, order="F").reshape((7, 56, 10, 9), order="F")


@lru_cache(maxsize=None)
def karzas():
    """XKARSAS FREQN(29,15), XN(29,15), XL(29,6,6) (REAL*4), EKARSAS(29)."""
    fr = equiv_assemble("xkarsas", "freqn", (29, 15), np.float32)
    xn = equiv_assemble("xkarsas", "xn", (29, 15), np.float32)
    xl = equiv_assemble("xkarsas", "xl", (29, 6, 6), np.float32)
    return fr, xn, xl, T("xkarsas", "ekarsas").astype(np.float32)


@lru_cache(maxsize=None)
def hminop_ff() -> np.ndarray:
    return equiv_assemble("hminop", "ff", (11, 22), np.float64)


# ---------------------------------------------------------------------------------------------------
# PFGROUND compiled from source
# ---------------------------------------------------------------------------------------------------
_NUM = re.compile(r"(\d+\.\d*|\.\d+|\d+)([DdEe][+-]?\d+)?")


def _lit(tok: str, exp: str | None) -> str:
    if exp and exp[0] in "Dd":
        return repr(float(tok + "e" + exp[1:]))
    v = float(tok + ("e" + exp[1:] if exp else ""))
    return repr(float(np.float32(v)))


def _translate(expr: str) -> str:
    e = expr.replace(" ", "")
    out, i = [], 0
    while i < len(e):
        if e[i].isdigit() or (e[i] == "." and i + 1 < len(e) and e[i + 1].isdigit()):
            m = _NUM.match(e, i)
            out.append(_lit(m.group(1), m.group(2)))
            i = m.end()
            continue
        m = re.match(r"[A-Za-z][A-Za-z0-9]*", e[i:])
        if m:
            w = m.group(0).upper()
            out.append({"EXP": "_exp", "HCK": "HCK", "T": "T"}[w])
            i += len(w)
            continue
        if e[i:i + 2] == "**":
            raise ValueError("power in PFGROUND not supported")
        out.append(e[i])
        i += 1
    return "".join(out)


@lru_cache(maxsize=None)
def pfground_exprs() -> dict[int, str]:
    """{NELION: python expression in T (tensor), HCK, _exp} for the reachable labels.

    Dispatch of the Fortran: NELION<114 -> GO TO (1..114),NELION; 115..168 -> 1 (iron group, PFIRON);
    169..336 -> GO TO (169..336),NELION-168; NELION=114 and >336 -> 1."""
    lines = source_path().read_text(encoding="latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if re.match(r"\s+FUNCTION PFGROUND", l))
    stmts: list[tuple[int | None, str]] = []
    for l in lines[start + 1:]:
        if re.match(r"\s+END\s*$", l):
            break
        if not l.strip() or l[0] in "Cc*!":
            continue
        if len(l) > 5 and l[5] not in " 0" and not l.startswith("\t"):
            lab, txt = stmts[-1]
            stmts[-1] = (lab, txt + l[6:72])
            continue
        lab = l[:5].strip()
        stmts.append((int(lab) if lab.isdigit() else None, l[6:72]))
    exprs = {}
    for k, (lab, txt) in enumerate(stmts):
        if lab is None or not (1 <= lab <= 113 or 169 <= lab <= 336):
            continue
        m = re.match(r"\s*PFGROUND\s*=(.*)$", txt, re.I)
        if not m:
            raise ValueError(f"unexpected PFGROUND statement {lab}: {txt}")
        nxt = stmts[k + 1][1].strip().upper()
        assert nxt == "RETURN", (lab, nxt)
        exprs[lab] = _translate(m.group(1))
    want = list(range(1, 114)) + list(range(169, 337))
    assert sorted(exprs) == want, sorted(set(want) - set(exprs))
    return exprs


HCK = 6.6256e-27 * 2.99792458e10 / 1.38054e-16


@lru_cache(maxsize=None)
def _pfground_fns():
    # HCK/T is evaluated once with a true division (torch's float/tensor is reciprocal-multiply)
    out = {}
    for k, v in pfground_exprs().items():
        assert v.count("HCK") == v.count("HCK/T"), v
        out[k] = eval(compile("lambda _Q, _exp: " + v.replace("HCK/T", "_Q"), f"pfground_{k}", "eval"))
    return out


def pfground(nelion: int, t: torch.Tensor) -> torch.Tensor:
    """FUNCTION PFGROUND(NELION,T) vectorised over T."""
    if not (1 <= nelion <= 113 or 169 <= nelion <= 336):
        return torch.ones_like(t)
    r = _pfground_fns()[nelion](torch.full_like(t, HCK) / t, torch.exp)
    if not isinstance(r, torch.Tensor):
        r = torch.full_like(t, float(r))
    return r
