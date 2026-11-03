"""Readers for the BasicATLAS pipeline products used in scoring (same files the Fortran writes)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .fortran import ffloat, read_records

# ---------------------------------------------------------------- DFSYNTHE / KAPPA9


def read_bdf(path: str | Path, nt: int = 57, nfreq: int | None = None) -> np.ndarray:
    """ODF file written by separatedf (p00big*.bdf / p00lit*.bdf).

    Records: for inu in freq bins, for it in 1..57: ((isteps(ip,it,istep), istep=1..12), ip=1..25) as int16.
    Returns int16 array [nfreq, nt, 25, 12].
    """
    recs = read_records(path)
    a = np.stack([np.frombuffer(r, dtype="<i2") for r in recs])
    a = a.reshape(-1, nt, 25, 12)
    if nfreq is not None:
        assert a.shape[0] == nfreq, a.shape
    return a


def read_kappa_ros(path: str | Path) -> dict[str, np.ndarray]:
    """kappa.ros from kapreadts: columns logT logP k(0,1,2,4,8 km/s) [+ unused cols]. Returns arrays on a (57,42) grid."""
    rows = []
    for ln in Path(path).read_text().splitlines()[2:]:
        if not ln.strip():
            continue
        # format(f5.2,f5.2,5F7.3,f9.5,f9.5,f9.5,f8.3)
        w = [5, 5, 7, 7, 7, 7, 7]
        vals, i = [], 0
        for k in w:
            vals.append(ffloat(ln[i:i + k])); i += k
        rows.append(vals)
    a = np.array(rows).reshape(57, 42, 7)
    return {"logT": a[:, 0, 0], "logP": a[0, :, 1], "logkappa": a[:, :, 2:]}  # logkappa [57,42,5]


# ---------------------------------------------------------------- SYNTHE


def read_spectrum_bin(path: str | Path) -> dict[str, np.ndarray]:
    """spectrum.bin (spectrv fort.7): header record then one record (wave?, flux, cont) per point.

    Mirrors BasicATLAS load_binary_spectrum but returns raw Fortran quantities (no unit conversion).
    """
    raw = Path(path).read_bytes()
    dt = np.dtype("i4,f8,f8,S74,f8,f8,i4,i4,i4," + ",".join(["f8"] * 20 + ["i4"] + ["f8"] * 377))
    header = np.frombuffer(raw, dtype=dt, count=1)
    data = np.frombuffer(raw, dtype=np.dtype("f8,f8,f8"), offset=dt.itemsize)
    wbegin, resolu, n = float(header["f4"][0]), float(header["f5"][0]), int(header["f6"][0])
    return {"wbegin": wbegin, "resolu": resolu, "n": n, "c0": data["f0"].copy(),
            "flux": data["f1"].copy(), "cont": data["f2"].copy()}


# ---------------------------------------------------------------- ATLAS9


def read_atlas_table(output_main: str | Path, which: str = "last") -> dict[str, np.ndarray]:
    """Iteration summary table from patched atlas9mem stdout (between 'START TABLE' and 'END TABLE').

    Columns (patched FORMAT 542): J RHOX T P XNE RHO ABROSS HEIGHT TAUROS FLXCNV PRADK FLXERR FLXDRV.
    `which` = 'last' or an iteration index (0-based among tables).
    """
    lines = Path(output_main).read_text(errors="replace").splitlines()
    tables, cur = [], None
    for ln in lines:
        if "START TABLE" in ln:
            cur = []
            continue
        if "END TABLE" in ln and cur is not None:
            tables.append(cur); cur = None
            continue
        if cur is not None:
            parts = ln.split()
            if len(parts) == 13 and parts[0].isdigit():
                cur.append([ffloat(p) for p in parts])
    t = np.array(tables[-1] if which == "last" else tables[which])
    names = ["J", "RHOX", "T", "P", "XNE", "RHO", "ABROSS", "HEIGHT", "TAUROS", "FLXCNV", "PRADK", "FLXERR", "FLXDRV"]
    out = {n: t[:, i] for i, n in enumerate(names)}
    out["ntables"] = len(tables)
    return out


def read_deck6(path: str | Path) -> dict[str, np.ndarray]:
    """'READ DECK6' block of a punched model (output_summary.out / restart): RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB,FLXRAD,VCONV,VELSND."""
    lines = Path(path).read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("READ DECK6"):
            n = int(ln[10:13])
            rows = []
            for r in lines[i + 1:i + 1 + n]:
                # FORMAT(1PE15.8,0PF9.1,1P8E10.3)
                f = [r[0:15], r[15:24]] + [r[24 + 10 * k:34 + 10 * k] for k in range(8)]
                rows.append([ffloat(x) for x in f])
            a = np.array(rows)
            names = ["RHOX", "T", "P", "XNE", "ABROSS", "ACCRAD", "VTURB", "FLXRAD", "VCONV", "VELSND"]
            return {nm: a[:, k] for k, nm in enumerate(names)}
    raise ValueError("no READ DECK6 block")
