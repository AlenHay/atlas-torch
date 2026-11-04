"""Line-opacity oracle: SYNTHE work dirs (scripts/synthe_workdir.sh) as stage inputs and references.

* gas/continuum of a work dir from its xnfpelsyn.dat (stage isolation: the Fortran's populations);
* fort.9 (ASYNTH per wavelength) read/write;
* class-resolved references: synthe.exe re-run on fort.12 / fort.19 filtered by line class;
* spectrum with a given line opacity: the Fortran SPECTRV run on the work dir's pre_spectrv inputs with fort.9
  replaced.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from atlas_torch.io.fortran import read_records, write_record
from atlas_torch.io.formats import read_spectrum_bin
from atlas_torch.v2.io.synthe_layout import idmol
from atlas_torch.v2.io.xnfpelsyn_dat import read_xnfpelsyn_dat
from atlas_torch.v2.opacity.lines import GasState, LogWavelengthGrid, interpolate_edge_table

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
WORK = EXT / "work"
BIN = EXT / "BasicATLAS" / "bin"
MODEL = EXT / "runs" / "stock" / "atlas" / "sun" / "output_synthe.out"
WINDOWS = {"500_510": (500.0, 510.0), "380_420": (380.0, 420.0), "640_660": (640.0, 660.0),
           "1500_1600": (1500.0, 1600.0)}
CLASSES = ("atomic", "molecular", "hydrogen", "helium")


def workdir(window: str) -> Path:
    return WORK / f"stocksun_{window}"


def gas_from_xnfpelsyn(path) -> GasState:
    """GasState with XNFPELSYN's n/U (n_atom := n/U, U := 1), number densities and model VTURB."""
    x = read_xnfpelsyn_dat(path)
    st = x.state
    n = len(st["T"])
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
    nou = np.zeros((n, 99, 10))
    nou[:, :, :6] = x.xnfpel[:, :, :99].transpose(0, 2, 1)
    for z in range(20, 29):
        for off, s in ((30, 7), (40, 8), (50, 9), (60, 10)):
            nou[:, z - 1, s - 1] = x.xnfpel[:, 4, off + z - 1]
    nou[:, 39:, 5] = 0.0
    codes = idmol()
    used = codes > 0
    return GasState(T=t(st["T"]), n_e=t(st["XNE"]), rho=t(st["RHO"]), n_h1=t(st["XNFH"]),
                    n_he1=t(st["XNFHE"][:, 0]),
                    n_h2=t(st["XNFH2"]), vturb=t(st["VTURB"]), n_over_u_atom=t(nou),
                    n_over_u_mol=t(x.xnfpel[:, 5, 39:139][:, used]), mol_codes=codes[used])


def continuum_from_xnfpelsyn(path, grid: LogWavelengthGrid) -> torch.Tensor:
    x = read_xnfpelsyn_dat(path)
    return interpolate_edge_table(x.wl_edge, torch.as_tensor(x.log_cont_total), grid)


def read_fort9(path, n_layers: int) -> np.ndarray:
    """ASYNTH records -> (L, N) float64."""
    r = read_records(path)
    return np.stack([np.frombuffer(x, "<f4")[:n_layers] for x in r[2:-1]], 1).astype(np.float64)


def write_fort9(template, path, asynth: np.ndarray) -> None:
    """fort.9 with the header and trailer records of `template` and ASYNTH (L, N) padded to 99 layers."""
    r = read_records(template)
    L, N = asynth.shape
    a = np.zeros((N, 99), "<f4")
    a[:, :L] = asynth.T
    with open(path, "wb") as f:
        write_record(f, r[0])
        write_record(f, r[1])
        rec = np.empty(N, dtype=np.dtype([("m1", "<i4"), ("d", "<f4", (99,)), ("m2", "<i4")]))
        rec["m1"] = rec["m2"] = 4 * 99
        rec["d"] = a
        f.write(rec.tobytes())
        write_record(f, r[-1])


def spectrum_with(window: str, asynth: np.ndarray) -> dict:
    """read_spectrum_bin dict of SPECTRV run with line opacity `asynth` (L, N) on the work dir's inputs."""
    wd = workdir(window)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for f in (wd / "pre_spectrv").iterdir():
            shutil.copy(f, d / f.name)
        shutil.copy(wd / "fort.25", d / "fort.25")
        for f in d.iterdir():
            os.chmod(f, 0o644)
        write_fort9(wd / "pre_spectrv" / "fort.9", d / "fort.9", asynth)
        with open(MODEL) as fin, open(d / "spectrv.out", "w") as fout:
            subprocess.run([str(BIN / "spectrv.exe")], stdin=fin, stdout=fout, stderr=subprocess.STDOUT, cwd=d,
                           check=True)
        return read_spectrum_bin(d / "fort.7")


def class_references(window: str) -> Path:
    """Directory with <class>/fort.9 of synthe.exe run on the class's fort.12/fort.19 records (made once)."""
    wd = workdir(window)
    out = WORK / f"stocksun_{window}_split"
    if all((out / c / "fort.9").exists() for c in CLASSES):
        return out
    raw93 = read_records(wd / "pre_synthe" / "fort.93")[0]
    r12 = read_records(wd / "pre_synthe" / "fort.12")
    r19 = read_records(wd / "pre_synthe" / "fort.19")
    nel = np.array([struct.unpack_from("<i", r, 8)[0] for r in r12], dtype=np.int64)
    typ = np.array([struct.unpack_from("<i", r, 28)[0] for r in r19], dtype=np.int64)
    mol = (nel % 6 == 0) & (nel >= 240)
    none12, none19 = np.zeros(len(r12), bool), np.zeros(len(r19), bool)
    sel = {"atomic": (~mol, typ >= 0), "molecular": (mol, none19), "hydrogen": (none12, np.isin(typ, (-1, -2))),
           "helium": (none12, np.isin(typ, (-3, -4, -5, -6)))}
    for name, (m12, m19) in sel.items():
        d = out / name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for fn, recs, m in (("fort.12", r12, m12), ("fort.19", r19, m19)):
            with open(d / fn, "wb") as f:
                for r, k in zip(recs, m):
                    if k:
                        write_record(f, r)
        b = bytearray(raw93)
        struct.pack_into("<i", b, 0, int(m12.sum()))
        struct.pack_into("<i", b, 16, int(m19.sum()))
        with open(d / "fort.93", "wb") as f:
            write_record(f, bytes(b))
        for fn in ("fort.14", "fort.20"):
            (d / fn).touch()
        shutil.copy(wd / "xnfpelsyn.dat", d / "fort.10")
        os.chmod(d / "fort.10", 0o644)
        shutil.copy(EXT / "BasicATLAS" / "data" / "synthe_files" / "he1tables.dat", d / "fort.18")
        with open(d / "synthe.out", "w") as fout:
            subprocess.run([str(BIN / "synthe.exe")], stdout=fout, stderr=subprocess.STDOUT, cwd=d, check=True)
    return out


def deviation_stats(v: np.ndarray, r: np.ndarray, cont: np.ndarray) -> dict:
    """Per-point deviations of line opacity v from reference r: |dk|/kc (max, p99, rms), relative error where
    the reference exceeds 1 % of the continuum (median, p99, max), and the ratio of the totals."""
    d = np.abs(v - r) / cont
    big = r > 0.01 * cont
    rel = np.abs(v - r)[big] / r[big] if big.any() else np.zeros(1)
    return {"dk_over_kc:max": float(d.max()), "dk_over_kc:p99": float(np.percentile(d, 99)),
            "dk_over_kc:rms": float(np.sqrt((d ** 2).mean())), "rel:median": float(np.median(rel)),
            "rel:p99": float(np.percentile(rel, 99)), "rel:max": float(rel.max()),
            "total_ratio": float(v.sum() / r.sum()) if r.sum() > 0 else float("nan"),
            "points_compared": int(big.sum())}


def flux_stats(ref: dict, cand: dict) -> dict:
    """Metrics of tolerances.json stages.synthe (normflux:abs, :p99, :rms_abs, flux_over_cont:abs)."""
    a = ref["flux"] / ref["cont"]
    b = cand["flux"] / cand["cont"]
    d = np.abs(a - b)
    return {"normflux:abs": float(d.max()), "normflux:p99": float(np.quantile(d, 0.99)),
            "normflux:rms_abs": float(np.sqrt((d ** 2).mean())),
            "flux_over_cont:abs": float((np.abs(ref["flux"] - cand["flux"]) / ref["cont"]).max()),
            "cont:rel": float((np.abs(ref["cont"] - cand["cont"]) / np.maximum(ref["cont"], cand["cont"])).max())}
