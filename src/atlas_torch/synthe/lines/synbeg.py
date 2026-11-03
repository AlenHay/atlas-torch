"""Python port of SYNBEG (Fortran program for SYNTHE line preparation initialization).

Initializes fort.93 parameter record, and creates empty fort.12, fort.14, fort.19, fort.20.
"""
from __future__ import annotations

import math
import os
import struct
from pathlib import Path
from typing import Any, BinaryIO

from atlas_torch.io.fortran import write_record, read_records

FORT93_STRUCT = struct.Struct("<5i f 693f i 5d f i")


def parse_synbeg_card(card: str) -> dict[str, Any]:
    """Parse SYNBEG fixed-format input card:
    FORMAT(A3,7X,4F10.4,I3,I7,F10.5,2I5)
    """
    card = card.ljust(80)
    airvac = card[0:3].strip() or "VAC"
    wlbeg = float(card[10:20])
    wlend = float(card[20:30])
    resolu = float(card[30:40])
    turbv = float(card[40:50])
    ifnlte = int(card[50:53]) if card[50:53].strip() else 0
    linout = int(card[53:60]) if card[53:60].strip() else -1
    cutoff = float(card[60:70]) if card[60:70].strip() else 0.0001
    ifpred = int(card[70:75]) if card[70:75].strip() else 1
    nread = int(card[75:80]) if card[75:80].strip() else 0
    return {
        "airvac": airvac,
        "wlbeg": wlbeg,
        "wlend": wlend,
        "resolu": resolu,
        "turbv": turbv,
        "ifnlte": ifnlte,
        "linout": linout,
        "cutoff": cutoff,
        "ifpred": ifpred,
        "nread": nread,
    }


def pack_fort93(
    nlines: int,
    length: int,
    ifvac: int,
    ifnlte: int,
    n19: int,
    turbv: float,
    deckj: list[float] | tuple[float, ...],
    ifpred: int,
    wlbeg: float,
    wlend: float,
    resolu: float,
    ratio: float,
    ratiolg: float,
    cutoff: float,
    linout: int,
) -> bytes:
    """Pack parameters into 2848-byte fort.93 record payload."""
    if len(deckj) != 693:
        raise ValueError(f"deckj must have 693 elements (got {len(deckj)})")
    return FORT93_STRUCT.pack(
        nlines,
        length,
        ifvac,
        ifnlte,
        n19,
        turbv,
        *deckj,
        ifpred,
        wlbeg,
        wlend,
        resolu,
        ratio,
        ratiolg,
        cutoff,
        linout,
    )


def unpack_fort93(payload: bytes) -> dict[str, Any]:
    """Unpack parameters from 2848-byte fort.93 record payload."""
    vals = FORT93_STRUCT.unpack(payload)
    return {
        "nlines": vals[0],
        "length": vals[1],
        "ifvac": vals[2],
        "ifnlte": vals[3],
        "n19": vals[4],
        "turbv": vals[5],
        "deckj": list(vals[6 : 6 + 693]),
        "ifpred": vals[699],
        "wlbeg": vals[700],
        "wlend": vals[701],
        "resolu": vals[702],
        "ratio": vals[703],
        "ratiolg": vals[704],
        "cutoff": vals[705],
        "linout": vals[706],
    }


def read_fort93(path: str | Path) -> dict[str, Any]:
    records = read_records(path)
    if not records:
        raise ValueError(f"Empty fort.93 at {path}")
    return unpack_fort93(records[0])


def write_fort93_dict(path: str | Path, p: dict[str, Any]) -> None:
    payload = pack_fort93(
        p["nlines"],
        p["length"],
        p["ifvac"],
        p["ifnlte"],
        p["n19"],
        p["turbv"],
        p["deckj"],
        p["ifpred"],
        p["wlbeg"],
        p["wlend"],
        p["resolu"],
        p["ratio"],
        p["ratiolg"],
        p["cutoff"],
        p["linout"],
    )
    with open(path, "wb") as f:
        write_record(f, payload)


def run_synbeg(
    workdir: str | Path,
    wlbeg: float | None = None,
    wlend: float | None = None,
    resolu: float = 600000.0,
    vturb: float = 1.5,
    ifnlte: int = 0,
    linout: int = -1,
    cutoff: float = 0.0001,
    ifpred: int = 1,
    nread: int = 0,
    airvac: str = "VAC",
    deckj: list[float] | None = None,
    card: str | None = None,
) -> dict[str, Any]:
    """Execute SYNBEG logic in `workdir`.

    Creates empty fort.12, fort.14, fort.19, fort.20, and initial fort.93.
    Returns the parameter dict written to fort.93.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if card is not None:
        parsed = parse_synbeg_card(card)
        airvac = parsed["airvac"]
        wlbeg = parsed["wlbeg"]
        wlend = parsed["wlend"]
        resolu = parsed["resolu"]
        vturb = parsed["turbv"]
        ifnlte = parsed["ifnlte"]
        linout = parsed["linout"]
        cutoff = parsed["cutoff"]
        ifpred = parsed["ifpred"]
        nread = parsed["nread"]

    if wlbeg is None or wlend is None:
        raise ValueError("wlbeg and wlend must be specified")

    if deckj is None:
        deckj = [0.0] * 693
    elif len(deckj) != 693:
        raise ValueError(f"deckj must have 693 elements (got {len(deckj)})")

    airvac_upper = airvac.strip().upper()
    ifvac = 0 if airvac_upper == "AIR" else 1

    ratio = 1.0 + 1.0 / resolu
    ratiolg = math.log(ratio)

    ixwlbeg = int(math.log(wlbeg) / ratiolg)
    wbegin = math.exp(ixwlbeg * ratiolg)
    if wbegin < wlbeg:
        ixwlbeg += 1
        wbegin = math.exp(ixwlbeg * ratiolg)

    ixwlend = int(math.log(wlend) / ratiolg)
    wllast = math.exp(ixwlend * ratiolg)
    if wllast >= wlend:
        ixwlend -= 1
        wllast = math.exp(ixwlend * ratiolg)

    length = ixwlend - ixwlbeg + 1

    # Empty files: fort.12, fort.14, fort.19, fort.20
    for name in ("fort.12", "fort.14", "fort.19", "fort.20"):
        open(workdir / name, "wb").close()

    params = {
        "nlines": 0,
        "length": length,
        "ifvac": ifvac,
        "ifnlte": ifnlte,
        "n19": 0,
        "turbv": vturb,
        "deckj": deckj,
        "ifpred": ifpred,
        "wlbeg": wlbeg,
        "wlend": wlend,
        "resolu": resolu,
        "ratio": ratio,
        "ratiolg": ratiolg,
        "cutoff": cutoff,
        "linout": linout,
    }

    write_fort93_dict(workdir / "fort.93", params)
    return params
