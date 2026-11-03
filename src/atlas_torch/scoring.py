"""Stage metrics shared by the noise-floor tool and `make score`.

Every scored stage maps a pair of run directories (reference, candidate) to {quantity: error}. Errors:
  rel  : max over points of |a-b| / max(|a|,|b|)              (positive physical quantities)
  abs  : max |a-b| in the quantity's own units                (ODF milli-dex, log kappa, % flux errors)
  frac : fraction of entries that differ at all (informational, not gated)
rms variants are reported alongside (rms_*), informational.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io.formats import read_atlas_table, read_bdf, read_deck6, read_kapk, read_kappa_ros, read_last_iteration, read_spectrum_bin

VTURBS = [0, 1, 2, 4, 8]


def _rel(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    den = np.maximum(np.abs(a), np.abs(b))
    r = np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1), 0.0)
    return float(r.max(initial=0)), float(np.sqrt(np.mean(r**2))) if r.size else 0.0


def _abs(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return float(d.max(initial=0)), float(np.sqrt(np.mean(d**2))) if d.size else 0.0


def _put(out, name, kind, pair):
    out[f"{name}:{kind}"] = pair[0]
    out[f"{name}:rms_{kind}"] = pair[1]


def synthe_metrics(ref: Path, cand: Path) -> dict:
    """ref/cand = SYNTHE output dirs containing spectrum.bin."""
    a, b = read_spectrum_bin(Path(ref) / "spectrum.bin"), read_spectrum_bin(Path(cand) / "spectrum.bin")
    if a["n"] != b["n"] or a["flux"].shape != b["flux"].shape:
        return {"error": f"grid mismatch {a['n']} vs {b['n']}"}
    out = {}
    _put(out, "flux", "rel", _rel(a["flux"], b["flux"]))
    _put(out, "cont", "rel", _rel(a["cont"], b["cont"]))
    _put(out, "normflux", "abs", _abs(a["flux"] / a["cont"], b["flux"] / b["cont"]))
    return out


def odf_metrics(ref: Path, cand: Path) -> dict:
    """ref/cand = DFSYNTHE output dirs (p00big*.bdf + kappa.ros)."""
    out = {}
    for v in VTURBS:
        a = read_bdf(Path(ref) / f"p00big{v}.bdf").astype(np.int64)
        b = read_bdf(Path(cand) / f"p00big{v}.bdf").astype(np.int64)
        if a.shape != b.shape:
            return {"error": f"bdf shape {a.shape} vs {b.shape}"}
        _put(out, f"odf_v{v}", "abs", _abs(a, b))
        out[f"odf_v{v}:frac"] = float((a != b).mean())
    ka, kb = read_kappa_ros(Path(ref) / "kappa.ros"), read_kappa_ros(Path(cand) / "kappa.ros")
    _put(out, "logkappa_ros", "abs", _abs(ka["logkappa"], kb["logkappa"]))
    return out


def kappa9_metrics(ref: Path, cand: Path) -> dict:
    """ref/cand = KAPPA9 work dirs (scripts/kappa9_workdir.sh layout): kapk{v}.dat + kappa.ros."""
    out = {}
    for v in VTURBS:
        a, b = read_kapk(Path(ref) / f"kapk{v}.dat"), read_kapk(Path(cand) / f"kapk{v}.dat")
        _put(out, f"kapk_v{v}", "abs", _abs(a, b))
        out[f"kapk_v{v}:frac"] = float((a != b).mean())
    ka, kb = read_kappa_ros(Path(ref) / "kappa.ros"), read_kappa_ros(Path(cand) / "kappa.ros")
    _put(out, "logkappa_ros", "abs", _abs(ka["logkappa"], kb["logkappa"]))
    return out


ATLAS_REL = ["RHOX", "T", "P", "XNE", "RHO", "ABROSS", "TAUROS"]
ATLAS_ABS = ["FLXERR", "FLXDRV"]


def atlas_metrics(ref: Path, cand: Path) -> dict:
    """ref/cand = ATLAS run dirs. Scores BasicATLAS's products: output_last_iteration.out (the best iteration,
    what read_structure() returns) and output_summary.out (its punched deck)."""
    a, b = read_last_iteration(Path(ref) / "output_last_iteration.out"), read_last_iteration(Path(cand) / "output_last_iteration.out")
    if a["T"].shape != b["T"].shape:
        return {"error": f"layers {a['T'].shape} vs {b['T'].shape}"}
    out = {}
    for q in ATLAS_REL:
        _put(out, q, "rel", _rel(a[q], b[q]))
    for q in ATLAS_ABS:
        _put(out, q, "abs", _abs(a[q], b[q]))
    # height: zero at the reference level -> absolute, km
    _put(out, "HEIGHT", "abs", _abs(a["HEIGHT"], b["HEIGHT"]))
    # convective flux relative to the total flux sigma*Teff^4/(4 pi) (zero in radiative layers)
    da, db = read_deck6(Path(ref) / "output_summary.out"), read_deck6(Path(cand) / "output_summary.out")
    tot = np.max(np.abs(da["FLXRAD"]) + np.abs(a["FLXCNV"]))
    _put(out, "FLXCNV_over_total", "abs", _abs(a["FLXCNV"] / tot, b["FLXCNV"] / tot))
    for q in ["T", "P", "XNE", "ABROSS"]:  # the punched deck SYNTHE consumes
        _put(out, f"deck_{q}", "rel", _rel(da[q], db[q]))
    out["iterations_tables"] = int(read_atlas_table(Path(cand) / "output_main.out")["ntables"])
    return out


def gated_keys(metrics: dict) -> list[str]:
    return [k for k in metrics if ":" in k and not k.split(":")[1].startswith(("rms_", "frac"))]
