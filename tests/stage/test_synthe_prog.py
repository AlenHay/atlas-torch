"""Stage test: PROGRAM SYNTHE (BasicATLAS-patched) from the stage-isolated inputs of the reference runs.

Inputs: $ATLAS_EXT/work/<window>_stock/pre_synthe (fort.10 = xnfpelsyn.dat, fort.12/19/20 line lists,
fort.93 from SYNBEG) + fort.18 = he1tables.dat.  Reference outputs: pre_spectrv/ (fort.9, fort.8, fort.14,
fort.15 after synthe.exe ran), synthe.out and progress.dat of the stock (gfortran -O3) run.

The reference runs were made in the atlas-dev container (Ubuntu 24.04, glibc 2.39).  The port reproduces
that libm (lineopac/libm32.py); note that the stock binary run on a host with glibc >= 2.41 gives a different
fort.9 for 380-420 nm (its correctly rounded log10f changes the He I 4026 BCS profiles).

Every output file must be byte-identical; fort.9 is also decoded and compared as float arrays (max relative
error reported, must be 0: all of SYNTHE's opacities are REAL*4 and reproduced exactly in float32).
"""
import os
import resource
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from atlas_torch.io.fortran import read_records
from atlas_torch.synthe.synthe_prog import run_synthe

pytestmark = pytest.mark.stage

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
WINDOWS = ["sun_500_510", "sun_380_420"]
HE1 = EXT / "BasicATLAS" / "data" / "synthe_files" / "he1tables.dat"
STOCK = EXT / "BasicATLAS" / "bin" / "synthe.exe"


def _need(p):
    if not Path(p).exists():
        pytest.skip(f"missing {p}")
    return Path(p)


def _workdir(tmp: Path, win: str) -> Path:
    src = _need(EXT / "work" / f"{win}_stock")
    tmp.mkdir(parents=True, exist_ok=True)
    for f in (src / "pre_synthe").glob("fort.*"):
        shutil.copy(f, tmp / f.name)
        os.chmod(tmp / f.name, 0o644)
    shutil.copy(_need(HE1), tmp / "fort.18")
    return tmp


def _decode9(path: Path, nrhox: int):
    recs = read_records(path)
    head = recs[:2]
    a = np.stack([np.frombuffer(r, "<f4") for r in recs[2:-1]])
    n9 = struct.unpack("<i", recs[-1])[0]
    return head, a, n9


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for win in WINDOWS:
        wd = _workdir(tmp_path_factory.mktemp(win), win)
        r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t = time.perf_counter()
        res = run_synthe(wd, stdout_path=wd / "synthe.out")
        dt = time.perf_counter() - t
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Fortran timing on identical inputs (same machine, same moment) for the speed ratio
        t_f = None
        if STOCK.exists():
            fw = _workdir(tmp_path_factory.mktemp(win + "_f"), win)
            t0 = time.perf_counter()
            with open(fw / "synthe.out", "w") as fo:
                subprocess.run([str(STOCK)], stdout=fo, cwd=fw, check=True)
            t_f = time.perf_counter() - t0
        out[win] = dict(wd=wd, res=res, t=dt, t_fortran=t_f, maxrss_mb=max(r0, rss) / 1024)
    return out


@pytest.mark.parametrize("win", WINDOWS)
def test_fort9(runs, win):
    r = runs[win]
    ref = EXT / "work" / f"{win}_stock" / "pre_spectrv" / "fort.9"
    got = r["wd"] / "fort.9"
    nr = r["res"]["buffer"].shape[0]
    hg, ag, ng = _decode9(got, nr)
    hr, ar, nref = _decode9(ref, nr)
    assert hg == hr, "fort.9 header records"
    assert ng == nref
    assert ag.shape == ar.shape
    den = np.where(ar != 0, np.abs(ar.astype(np.float64)), 1.0)
    rel = np.abs(ag.astype(np.float64) - ar) / den
    print(f"\n{win}: fort.9 ASYNTH {ar.shape}, max rel err {rel.max():.3e}, "
          f"bitwise-different values {(ag.view(np.uint32) != ar.view(np.uint32)).sum()}")
    assert rel.max() <= 1e-12
    assert got.read_bytes() == ref.read_bytes()


@pytest.mark.parametrize("win", WINDOWS)
def test_other_outputs(runs, win):
    r = runs[win]
    ref = EXT / "work" / f"{win}_stock"
    wd = r["wd"]
    for f in ("fort.8", "fort.14", "fort.15"):
        a, b = wd / f, ref / "pre_spectrv" / f
        assert a.exists() == b.exists(), f
        if b.exists():
            assert a.read_bytes() == b.read_bytes(), f
    # fort.14 decoded: the per-layer line buffers (REAL*4)
    nr = r["res"]["buffer"].shape[0]
    b14 = np.fromfile(ref / "pre_spectrv" / "fort.14", "<f4").reshape(nr, -1)[:, :r["res"]["buffer"].shape[1]]
    assert np.array_equal(r["res"]["buffer"].view(np.uint32), b14.view(np.uint32))
    assert (wd / "progress.dat").read_text() == (ref / "progress.dat").read_text()
    assert (wd / "synthe.out").read_text() == (ref / "synthe.out").read_text()
    # inputs deleted / kept exactly as synthe.exe does
    assert sorted(p.name for p in wd.glob("fort.*")) == sorted(p.name for p in (ref / "pre_spectrv").glob("fort.*"))


def test_speed_and_memory(runs):
    for win, r in runs.items():
        tf = r["t_fortran"]
        ratio = r["t"] / tf if tf else float("nan")
        print(f"\n{win}: torch {r['t']:.1f} s, Fortran {tf if tf else float('nan'):.1f} s, ratio {ratio:.2f}, "
              f"peak RSS {r['maxrss_mb']:.0f} MB")
        assert r["maxrss_mb"] < 4096


@pytest.mark.parametrize("win", WINDOWS)
def test_layer_blocks_and_chunks_invariant(tmp_path, win):
    """The bounded-memory scheme (layer blocks re-reading fort.12, small fort.12 batches, small scatter-add
    batches, chunked fort.14 -> fort.9 transposition) gives the same bytes as the stock executable."""
    from atlas_torch.synthe.synthe_prog import SyntheProgram
    wd = _workdir(tmp_path / win, win)
    ref = EXT / "work" / f"{win}_stock"
    SyntheProgram(wd, layer_block=7, line_chunk=3000, max_contrib=50_000, point_chunk=1000,
                  keep_arrays=False).run(stdout_path=wd / "synthe.out")
    for f in ("fort.9", "fort.8", "fort.14", "fort.15"):
        a, b = wd / f, ref / "pre_spectrv" / f
        assert a.exists() == b.exists(), f
        if b.exists():
            assert a.read_bytes() == b.read_bytes(), f
    assert (wd / "synthe.out").read_text() == (ref / "synthe.out").read_text()
