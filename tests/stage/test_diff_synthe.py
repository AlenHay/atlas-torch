"""Stage: differentiable SYNTHE (atlas_torch.diff) on the solar 500-510 nm window (synthe/sun/w0).

1. Primal: the differentiable mode (float64 native torch math, straight-through tables) reproduces the stock
   Fortran spectrum.bin inside the frozen synthe/sun/w0 gates (results/tolerances.json).
2. Derivatives vs central finite differences of the same function with frozen decisions (profile="smooth",
   polished NMOLEC root): abundances (Fe, Mg, C), [M/H], vturb, T/P/RHOX of single layers.
3. Derivatives (profile="exact") vs finite differences of the exact path (decisions re-taken, abundances
   through the deck's printed cards): agreement in the norm, at the level of the exact path's staircase.
Run: pytest -m stage tests/stage/test_diff_synthe.py -s -p no:cacheprovider     (~15 min, < 3 GB)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.stage

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
MODEL = EXT / "runs" / "stock" / "atlas" / "sun"
STOCK_SPEC = EXT / "work" / "stocksun_500_510" / "spectrum.bin"
REPO = Path(__file__).resolve().parents[2]


def _need(p):
    if not Path(p).exists():
        pytest.skip(f"missing {p}")
    return p


def _read_spectrum_bin(path):
    hdr = np.dtype("i4,f8,f8,S74,f8,f8,i4,i4,i4," + ",".join(["f8"] * 20 + ["i4"] + ["f8"] * 377))
    with open(path, "rb") as f:
        np.fromfile(f, dtype=hdr, count=1)
        d = np.fromfile(f, dtype=np.dtype("f8,f8,f8"))
    return d["f1"], d["f2"]


@pytest.fixture(scope="module")
def sd_smooth():
    from atlas_torch.diff import SyntheDiff
    torch.set_num_threads(4)
    return SyntheDiff(_need(MODEL), 500.0, 510.0, profile="smooth", polish=True)


@pytest.fixture(scope="module")
def sd_exact():
    from atlas_torch.diff import SyntheDiff
    torch.set_num_threads(4)
    return SyntheDiff(_need(MODEL), 500.0, 510.0, profile="exact", polish=False)


def test_primal_inside_frozen_gates(sd_exact):
    fx, cx = _read_spectrum_bin(_need(STOCK_SPEC))
    o = sd_exact({})
    fl, co = o["flux"].numpy(), o["cont"].numpy()
    tol = json.loads((REPO / "results" / "tolerances.json").read_text())["stages"]["synthe/sun/w0"]
    err = {"flux:rel": np.max(np.abs(fl - fx) / np.abs(fx)), "cont:rel": np.max(np.abs(co - cx) / np.abs(cx)),
           "normflux:abs": np.max(np.abs(fl / co - fx / cx))}
    print({k: f"{v:.2e} (tol {tol[k]['tol']:.2e})" for k, v in err.items()})
    for k, v in err.items():
        assert v <= tol[k]["tol"], k


@pytest.mark.parametrize("name", ["abund:Fe", "abund:C", "vturb", "T[45]", "P[40]", "RHOX[40]"])
def test_ad_vs_fd_frozen(sd_smooth, name):
    from atlas_torch.diff.validate import check_diff
    r = check_diff(sd_smooth, [name])[name]
    print(name, {k: v for k, v in r.items() if k != "refined"}, r.get("refined"))
    assert r["rms_rel"] < 1e-4
    worst = r["refined"]["max_rel"] if "refined" in r else r["max_rel"]
    assert worst < 2e-3


def test_ad_exact_profile_vs_exact_path(sd_exact):
    from atlas_torch.diff.validate import check_exact
    r = check_exact(sd_exact, {"abund:Fe": 0.05, "vturb": 0.05}, scratch=EXT / "scratch" / "m8" / "exact_fd_test")
    print(r)
    assert r["abund:Fe"]["rms_rel"] < 2e-2
    assert r["vturb"]["rms_rel"] < 5e-2
