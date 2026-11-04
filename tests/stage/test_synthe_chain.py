"""Stage test: the whole SYNTHE chain in torch/Python (atlas_torch.synthe.chain.run_synthe_chain), 500-510 nm.

xnfpelsyn_prog -> lines.prepare_lines -> synthe_prog -> spectrv_prog, as BasicATLAS' synthe_control, for
  scratch  $ATLAS_EXT/scratch_sun/output_synthe.out           stock: work/sun_500_510_stock/spectrum.bin
  stocksun $ATLAS_EXT/runs/stock/atlas/sun/output_synthe.out  stock: work/stocksun_500_510/spectrum.bin
(stock work dirs: scripts/synthe_workdir.sh with BasicATLAS/bin in the atlas-dev container).

1. test_spectrum_vs_stock: flux / continuum max relative error vs the stock spectrum.bin (reported; bound: the
   Fortran's own compiler spread on this window, ~7e-7 in flux between stock and other builds).
2. test_chain_wiring: the stock synthe.exe + spectrv.exe run on the chain's own intermediate inputs
   (its xnfpelsyn.dat and the line lists prepare_lines writes) must give the chain's spectrum.bin byte for
   byte: the driver feeds every program what synthe_control feeds the executables.
Runtimes of every program are printed next to the Fortran's (stock run of synthe_workdir.sh, measured here).
Run: pytest -m stage tests/stage/test_synthe_chain.py -s -p no:cacheprovider
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.io.formats import read_spectrum_bin
from atlas_torch.synthe.chain import FORT25, MOLECULAR_LISTS, basicatlas_dir, run_synthe_chain

pytestmark = pytest.mark.stage

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
MODELS = {
    "scratch": (EXT / "scratch_sun" / "output_synthe.out", EXT / "work" / "sun_500_510_stock"),
    "stocksun": (EXT / "runs" / "stock" / "atlas" / "sun" / "output_synthe.out", EXT / "work" / "stocksun_500_510"),
}
WIN = (500.0, 510.0, 600000.0, 1.5)
BIN = basicatlas_dir() / "bin"
SPREAD = 7e-7          # stock vs other Fortran builds, flux, this window


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p


def _rel(a, b) -> tuple[float, int]:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.abs(b), initial=0.0)), int((a != b).sum())


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    torch.set_default_dtype(torch.float64)
    cache = {}

    def get(model):
        if model not in cache:
            deck, stock = MODELS[model]
            _need(deck)
            wd = tmp_path_factory.mktemp(model) / "synthe_1"
            t = time.perf_counter()
            r = run_synthe_chain(deck, *WIN, wd, "BasicATLAS", keep=True)
            r["total"] = time.perf_counter() - t
            cache[model] = dict(r, deck=deck, stock=stock)
        return cache[model]
    return get


def _fortran_chain_times(deck: Path, wd: Path) -> dict:
    """Wall time of each stock executable of synthe_control (run like scripts/synthe_workdir.sh)."""
    s, d = basicatlas_dir() / "data" / "synthe_files", basicatlas_dir() / "data" / "dfsynthe_files"
    wd.mkdir(parents=True)
    tm = {}

    def run(exe, stdin=None, inp=None):
        with open(wd / f"{exe}.out", "w") as fo:
            fi = open(stdin) if stdin else None
            try:
                subprocess.run([str(BIN / f"{exe}.exe")], stdin=fi, input=inp, stdout=fo, cwd=wd, check=True,
                               text=inp is not None)
            finally:
                if fi:
                    fi.close()

    def link(src, name):
        (wd / name).unlink(missing_ok=True)
        os.symlink(src, wd / name)

    t = time.perf_counter()
    link(s / "molecules.dat", "fort.2")
    link(s / "continua.dat", "fort.17")
    run("xnfpelsyn", stdin=deck)
    tm["xnfpelsyn"] = time.perf_counter() - t
    t = time.perf_counter()
    card = ("%-3s       %-10.4f%-10.4f%-9.2f %-10.4f%-3d%-7d%-10.5f%-5d%-5d\nAIRorVAC  WLBEG     WLEND     RESOLU    "
            "TURBV  IFNLTE LINOUT CUTOFF        NREAD\n" % ("VAC", WIN[0], WIN[1], WIN[2], WIN[3], 0, -1, 0.0001, 1, 0))
    shutil.move(wd / "fort.10", wd / "xnfpelsyn.dat")
    run("synbeg", inp=card)
    link(s / "BasicATLAS.dat", "fort.11")
    run("rgfalllinesnew")
    for m in MOLECULAR_LISTS:
        link(s / m, "fort.11")
        run("rmolecasc")
    link(d / "h2ofastfix.bin", "fort.11")
    run("rh2ofast")
    tm["lines"] = time.perf_counter() - t
    t = time.perf_counter()
    link(wd / "xnfpelsyn.dat", "fort.10")
    link(s / "he1tables.dat", "fort.18")
    run("synthe")
    tm["synthe"] = time.perf_counter() - t
    t = time.perf_counter()
    (wd / "fort.25").write_text(FORT25)
    run("spectrv", stdin=deck)
    tm["spectrv"] = time.perf_counter() - t
    return tm


@pytest.mark.parametrize("model", list(MODELS))
def test_spectrum_vs_stock(model, chains, tmp_path):
    r = chains(model)
    ref = _need(r["stock"] / "spectrum.bin")
    a, b = read_spectrum_bin(r["spectrum"]), read_spectrum_bin(ref)
    assert a["header"] == b["header"] and len(a["flux"]) == len(b["flux"]) and a["tail"] == b["tail"]
    ef, nf = _rel(a["flux"], b["flux"])
    ec, nc = _rel(a["cont"], b["cont"])
    tf = _fortran_chain_times(r["deck"], tmp_path / "fortran") if (BIN / "synthe.exe").exists() else {}
    print(f"\n{model}: {len(a['flux'])} points; flux max rel {ef:.3e} ({nf} differ), cont max rel {ec:.3e} "
          f"({nc} differ)")
    print("  runtime [s]   " + "  ".join(f"{k}: torch {v:.1f} / Fortran {tf.get(k, float('nan')):.1f}"
                                          for k, v in r["timings"].items())
          + f"  | total torch {r['total']:.1f} / Fortran {sum(tf.values()) if tf else float('nan'):.1f}")
    assert ef <= SPREAD and ec <= SPREAD


@pytest.mark.parametrize("model", list(MODELS))
def test_chain_wiring(model, chains, tmp_path):
    """Stock synthe.exe + spectrv.exe on the chain's intermediate files reproduce the chain's spectrum.bin."""
    if not (BIN / "synthe.exe").exists():
        pytest.skip("stock executables not available")
    from atlas_torch.synthe.lines import prepare_lines
    r = chains(model)
    s, d = basicatlas_dir() / "data" / "synthe_files", basicatlas_dir() / "data" / "dfsynthe_files"
    wd = tmp_path / "wiring"
    prepare_lines(wd, *WIN, s / "BasicATLAS.dat", [s / m for m in MOLECULAR_LISTS], d / "h2ofastfix.bin")
    shutil.copy(r["workdir"] / "xnfpelsyn.dat", wd / "fort.10")
    os.symlink(s / "he1tables.dat", wd / "fort.18")
    with open(wd / "synthe.out", "w") as fo:
        subprocess.run([str(BIN / "synthe.exe")], stdout=fo, cwd=wd, check=True)
    (wd / "fort.25").write_text(FORT25)
    os.symlink(s / "molecules.dat", wd / "fort.2")
    with open(r["deck"]) as fi, open(wd / "spectrv.out", "w") as fo:
        subprocess.run([str(BIN / "spectrv.exe")], stdin=fi, stdout=fo, cwd=wd, check=True)
    a, b = read_spectrum_bin(r["spectrum"]), read_spectrum_bin(wd / "fort.7")
    print(f"\n{model}: chain vs stock exes on the chain's inputs: flux {_rel(a['flux'], b['flux'])}, "
          f"cont {_rel(a['cont'], b['cont'])}")
    assert (wd / "fort.9").read_bytes() == (r["workdir"] / "fort.9").read_bytes()
    assert r["spectrum"].read_bytes() == (wd / "fort.7").read_bytes()
