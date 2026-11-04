"""molecules.dat (ATLAS/SYNTHE unit 2, READ MOLECULES): one species per line, FORMAT(F18.2,F7.3,6E11.4),
terminated by a zero code (atlas7v.for SUBROUTINE READMOL)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MoleculeRecord:
    """code: Kurucz species code (608. = CO, 26.01 = Fe II, 100. = H-); d0 and b..g as in Kurucz 1970 eq. 4.35
    (d0 in eV); comment: text after column 91."""
    code: float
    d0: float
    coeffs: tuple[float, float, float, float, float, float]
    comment: str = ""


def default_path() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "BasicATLAS" / "data" / "synthe_files" \
        / "molecules.dat"


def _num(field: str) -> float:
    f = field.strip().upper().replace("D", "E")
    return float(f) if f else 0.0


def read_molecules_dat(path: str | Path | None = None) -> list[MoleculeRecord]:
    out = []
    for line in Path(path or default_path()).read_text(encoding="latin-1").splitlines():
        code = _num(line[:18])
        if code == 0.0:
            break
        cols = [_num(line[25 + 11 * i:36 + 11 * i]) for i in range(6)]
        out.append(MoleculeRecord(code, _num(line[18:25]), tuple(cols), line[91:].rstrip()))
    return out


def format_molecules_dat(records: list[MoleculeRecord]) -> str:
    """Text that read_molecules_dat reads back to `records` (blank fields for atoms and ions)."""
    lines = []
    for r in records:
        if r.d0 == 0.0 and not any(r.coeffs):
            body = f"{r.code:18.2f}"
        else:
            body = f"{r.code:18.2f}{r.d0:7.3f}" + "".join(f"{c:11.4E}" for c in r.coeffs)
        lines.append((body.ljust(91) + r.comment).rstrip() if r.comment else body)
    return "\n".join(lines + [f"{'0.':>18}"]) + "\n"
