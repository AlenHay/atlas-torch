"""Stage test: PROGRAM XNFPELSYN (atlas_torch.synthe.xnfpelsyn_prog) vs the Fortran, fort.10 record by record.

Models (500-510 nm work dirs made with scripts/synthe_workdir.sh in the atlas-dev container):
  scratch  $ATLAS_EXT/scratch_sun/output_synthe.out              stock: work/sun_500_510_stock/xnfpelsyn.dat
  stocksun $ATLAS_EXT/runs/stock/atlas/sun/output_synthe.out     stock: work/stocksun_500_510/xnfpelsyn.dat

1. program logic: the port driven by the Fortran physics (READIN/POPS/KAPP of oracle/libxnfpelsyn.so through
   the test-only adapter atlas_torch.testing.xnfpelsyn_oracle_physics) must write a fort.10 byte-identical to
   the -O0 Fortran program (gf-O0 build, run here in the same environment, i.e. with the same libm).
2. torch physics (the default backend) vs the same -O0 Fortran fort.10: max relative error per array
   (populations, state record, log10 continuum tables), target <= 1e-13.
3. torch physics vs the stock run (gfortran -O3; NMOLEC converges to ~1e-10 different populations at -O3):
   reported, and bounded by the Fortran's own O0-vs-O3 spread.

Note on libm: the reference runs were made in the container (glibc 2.39). On a host with glibc >= 2.41
(correctly rounded log10f) XKARSAS' REAL*4 LOG10 flips the edge branch at 7 FREQSET points next to H/C/Si
edges (91.1, 103.4, 113.1, 142.8, 173.3, 8725, 9118 nm): the gf-O0 Fortran itself then differs from the
container runs there by up to 7e-3 dex. Comparisons 1 and 2 use a Fortran run made in the current
environment; comparison 3 skips those 7 points only when not running on glibc 2.39.
Run: pytest -m stage tests/stage/test_xnfpelsyn_prog.py -s -p no:cacheprovider   (prints the error tables)
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.synthe.xnfpelsyn_prog import default_paths, read_fort10, xnfpelsyn

pytestmark = pytest.mark.stage

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
MODELS = {
    "scratch": (EXT / "scratch_sun" / "output_synthe.out", EXT / "work" / "sun_500_510_stock"),
    "stocksun": (EXT / "runs" / "stock" / "atlas" / "sun" / "output_synthe.out", EXT / "work" / "stocksun_500_510"),
}
GF_O0 = EXT / "builds" / "gf-O0" / "bin" / "xnfpelsyn.exe"
EDGE_NU = [276, 314, 347, 389, 449, 919, 920]            # 0-based FREQSET points flipped by glibc's log10f
ARRAYS = ["edges", "idmol", "momass", "freqset", "continall", "contabs", "contscat", "xnfpel", "dopple"]


def _glibc_239() -> bool:
    return platform.libc_ver() == ("glibc", "2.39")


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p


def _fortran_O0(deck: Path, wd: Path) -> tuple[Path, float]:
    """fort.10 of the -O0 Fortran program in `wd` (gf-O0 executable; else the oracle library's MAIN)."""
    p = default_paths()
    wd.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    if GF_O0.exists():
        for name, src in (("fort.17", p["continua"]), ("fort.2", p["molecules"])):
            os.symlink(src, wd / name)
        with open(deck) as fi:
            subprocess.run([str(GF_O0)], stdin=fi, stdout=subprocess.DEVNULL, cwd=wd, check=True)
        out = wd / "fort.10"
    else:
        from atlas_torch.testing.xnfpelsyn_oracle_physics import available, fortran_main
        if not available():
            pytest.skip("no -O0 Fortran xnfpelsyn available")
        out = fortran_main(deck.read_text(), p["continua"], p["molecules"], wd)
    return out, time.perf_counter() - t


def compare(a: dict, b: dict, skip_nu=()) -> dict:
    """Max relative error (and number of differing values) per fort.10 array; header fields must agree."""
    assert (a["nt"], a["teff"], a["glog"], a["title"], a["nedge"]) == (b["nt"], b["teff"], b["glog"], b["title"],
                                                                     b["nedge"])
    out = {}

    def rel(name, x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        same_nan = np.isnan(x) & np.isnan(y)
        d = np.abs(x - y)
        m = np.maximum(np.abs(x), np.abs(y))
        r = np.where(same_nan | (d == 0), 0.0, d / np.where(m > 0, m, 1.0))
        r = np.where(np.isnan(r), np.inf, r)
        out[name] = dict(max_rel=float(r.max(initial=0.0)), ndiff=int((r > 0).sum()), n=x.size,
                         max_abs=float(np.where(np.isfinite(d), d, np.inf).max(initial=0.0)))

    for k in ARRAYS:
        x, y = a[k], b[k]
        if k in ("continall", "contabs", "contscat") and len(skip_nu):
            keep = np.setdiff1d(np.arange(x.shape[1]), skip_nu)
            x, y = x[:, keep], y[:, keep]
        rel(k, x, y)
    for k in a["state"]:
        rel("state." + k, a["state"][k], b["state"][k])
    return out


def report(title, st) -> str:
    lines = [title, f"{'array':14s} {'n':>7s} {'ndiff':>7s} {'max_rel':>10s} {'max_abs':>10s}"]
    for k, s in st.items():
        lines.append(f"{k:14s} {s['n']:7d} {s['ndiff']:7d} {s['max_rel']:10.3e} {s['max_abs']:10.3e}")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    torch.set_default_dtype(torch.float64)
    cache = {}

    def get(model):
        if model not in cache:
            deck, stock = MODELS[model]
            _need(deck)
            tmp = tmp_path_factory.mktemp(model)
            ref0, t_f = _fortran_O0(deck, tmp / "fortran_O0")
            t = time.perf_counter()
            res = xnfpelsyn(deck.read_text(), out10=tmp / "torch.fort.10")
            t_p = time.perf_counter() - t
            cache[model] = dict(deck=deck, stock=stock, tmp=tmp, ref0=ref0, t_fortran=t_f, t_torch=t_p,
                                torch10=tmp / "torch.fort.10", timings=res.timings)
        return cache[model]
    return get


@pytest.mark.parametrize("model", list(MODELS))
def test_program_logic_with_fortran_physics(model, runs, tmp_path):
    """Main-program port + Fortran READIN/POPS/KAPP == the Fortran program, byte for byte."""
    from atlas_torch.testing.xnfpelsyn_oracle_physics import OracleXnfpelsynPhysics, available
    if not available():
        pytest.skip("oracle library libxnfpelsyn.so not available")
    r = runs(model)
    out = tmp_path / "adapter.fort.10"
    t = time.perf_counter()
    xnfpelsyn(r["deck"].read_text(), out10=out, physics=OracleXnfpelsynPhysics())
    print(f"\n{model}: port+Fortran physics {time.perf_counter() - t:.2f} s, Fortran gf-O0 {r['t_fortran']:.2f} s")
    st = compare(read_fort10(out), read_fort10(r["ref0"]))
    print(report(f"== {model}: port (Fortran physics) vs Fortran -O0", st))
    assert out.read_bytes() == r["ref0"].read_bytes()


@pytest.mark.parametrize("model", list(MODELS))
def test_torch_vs_fortran_O0(model, runs):
    r = runs(model)
    st = compare(read_fort10(r["torch10"]), read_fort10(r["ref0"]))
    tm = ", ".join(f"{k} {v:.1f}" for k, v in r["timings"].items())
    print(f"\n{model}: torch port {r['t_torch']:.1f} s ({tm}), Fortran gf-O0 {r['t_fortran']:.2f} s")
    print(report(f"== {model}: torch port vs Fortran -O0 (same libm)", st))
    bad = {k: s for k, s in st.items() if s["max_rel"] > 1e-13}
    assert not bad, bad


@pytest.mark.parametrize("model", list(MODELS))
def test_torch_vs_stock(model, runs):
    r = runs(model)
    ref = _need(r["stock"] / "xnfpelsyn.dat")
    skip = () if _glibc_239() else EDGE_NU
    st = compare(read_fort10(r["torch10"]), read_fort10(ref), skip_nu=skip)
    st0 = compare(read_fort10(r["ref0"]), read_fort10(ref), skip_nu=skip)
    print(report(f"\n== {model}: torch port vs stock (gfortran -O3)" + (" [7 edge points skipped]" if skip else ""),
                 st))
    print(report(f"== {model}: Fortran -O0 vs stock (the Fortran's own spread)", st0))
    for k, s in st.items():
        # populations: NMOLEC at -O3 vs -O0 differs by ~1e-10; the port must be inside 10x that spread
        assert s["max_rel"] <= max(10 * st0[k]["max_rel"], 1e-13), (k, s, st0[k])
