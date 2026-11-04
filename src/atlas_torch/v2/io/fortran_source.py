"""Reading data that the Kurucz Fortran sources hold in executable statements (not DATA): statement lists of
a program unit and evaluation of Fortran arithmetic expressions over caller-supplied values."""
from __future__ import annotations

import ast
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

_HEADER = re.compile(r"^\s+(?:REAL\*8\s+|REAL\s+|INTEGER\s+)?(?:SUBROUTINE|FUNCTION|PROGRAM)\s+(\w+)", re.I)


def source_path(name: str) -> Path:
    """$ATLAS_EXT/builds/src/<name> (the fetched Kurucz sources)."""
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "builds" / "src" / name


@lru_cache(maxsize=None)
def statements(source: str, unit: str) -> tuple[tuple[int | None, str], ...]:
    """(label, statement) of program unit `unit` in fixed-form `source`: comments dropped, columns 7-72,
    continuation lines joined, upper case."""
    out: list[tuple[int | None, str]] = []
    inside = False
    for raw in source_path(source).read_text(encoding="latin-1").splitlines():
        if not raw.strip() or raw[0] in "cC*!":
            continue
        m = _HEADER.match(raw)
        if m:
            inside = m.group(1).upper() == unit.upper()
        if not inside:
            continue
        lab, text = raw[:5].strip(), raw[6:72].upper()
        if len(raw) > 5 and raw[5] not in " 0" and out:
            out[-1] = (out[-1][0], out[-1][1] + text.strip())
        else:
            out.append((int(lab) if lab else None, text.strip()))
        if re.fullmatch(r"\s*END\s*", text):
            break
    if not out:
        raise KeyError(f"unit {unit} not found in {source}")
    return tuple(out)


_DEXP = re.compile(r"(\d|\.)D([+-]?\d)")


def to_python(expr: str) -> str:
    """Fortran arithmetic expression -> Python syntax (D exponents, layer index (J) dropped)."""
    e = _DEXP.sub(r"\1E\2", expr.upper().replace(" ", ""))
    return re.sub(r"\b([A-Z]\w*)\(J\)", r"\1", e)


def evaluate(expr: str, names: Mapping[str, Any], functions: Mapping[str, Callable] | None = None) -> Any:
    """Value of a Fortran expression over `names` (any objects supporting + - * / **) and `functions`
    (called with evaluated arguments; also used for array references such as GHYD(I))."""
    functions = functions or {}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            return names[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.Div: lambda: a / b, ast.Pow: lambda: a ** b}
            return ops[type(n.op)]()
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            return functions[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported construct in {expr!r}: {ast.dump(n)}")

    return ev(ast.parse(to_python(expr), mode="eval"))
