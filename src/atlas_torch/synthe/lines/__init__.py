"""SYNTHE line-list preparation programs: faithful Python ports.

Provides:
- run_synbeg: initializes fort.93, empty fort.12, fort.14, fort.19, fort.20
- run_rgfalllines: processes atomic line list (BasicATLAS.dat)
- run_rmolecasc: processes molecular line lists (.asc files)
- run_rh2ofast: processes H2O direct-access file (h2ofastfix.bin)
- prepare_lines: orchestrates the full pipeline producing fort.93, fort.12, fort.14, fort.19, fort.20
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from atlas_torch.synthe.lines.synbeg import run_synbeg, read_fort93, write_fort93_dict, parse_synbeg_card
from atlas_torch.synthe.lines.rgfalllines import run_rgfalllines
from atlas_torch.synthe.lines.rmolecasc import run_rmolecasc
from atlas_torch.synthe.lines.rh2ofast import run_rh2ofast


def prepare_lines(
    workdir: str | Path,
    wlbeg: float,
    wlend: float,
    resolu: float,
    vturb: float,
    linelist: str | Path,
    mol_lists: Sequence[str | Path],
    h2o_path: str | Path,
    c12c13: tuple[float, float] | None = None,
    airorvac: str = "VAC",
    ifnlte: int = 0,
    linout: int = -1,
    cutoff: float = 0.0001,
    ifpred: int = 1,
    nread: int = 0,
) -> dict[str, Any]:
    """Execute complete SYNTHE line list preparation pipeline in `workdir`.

    Creates fort.93, fort.12, fort.14, fort.19, fort.20 matching BasicATLAS synthe_control.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    c12, c13 = (None, None) if c12c13 is None else c12c13

    # 1. SYNBEG
    synbeg_params = run_synbeg(
        workdir=workdir,
        wlbeg=wlbeg,
        wlend=wlend,
        resolu=resolu,
        vturb=vturb,
        ifnlte=ifnlte,
        linout=linout,
        cutoff=cutoff,
        ifpred=ifpred,
        nread=nread,
        airvac=airorvac,
    )

    # 2. RGFALLLINESNEW
    atomic_res = run_rgfalllines(workdir=workdir, linelist_path=linelist)

    # 3. RMOLECASC (for each molecular list in order)
    mol_total = 0
    for mol_path in mol_lists:
        res_mol = run_rmolecasc(workdir=workdir, mol_path=mol_path, c12=c12, c13=c13)
        mol_total += res_mol["n_added"]

    # 4. RH2OFAST
    h2o_res = run_rh2ofast(workdir=workdir, h2o_path=h2o_path)

    final_params = read_fort93(workdir / "fort.93")
    return {
        "nlines": final_params["nlines"],
        "n19": final_params["n19"],
        "length": final_params["length"],
        "atomic_lines": atomic_res["n14"],
        "atomic_n19": atomic_res["n19"],
        "mol_lines": mol_total,
        "h2o_lines": h2o_res["n_added"],
    }


__all__ = [
    "run_synbeg",
    "run_rgfalllines",
    "run_rmolecasc",
    "run_rh2ofast",
    "prepare_lines",
    "read_fort93",
    "write_fort93_dict",
    "parse_synbeg_card",
]
