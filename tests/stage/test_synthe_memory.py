"""Stage test: bounded memory of the torch SYNTHE chain (atlas_torch.synthe.chain.run_synthe_chain).

BasicATLAS runs SYNTHE over batches of up to 2,010,001 wavelength points (e.g. 100-2850 nm at R=600000, with
50M+ fort.12 lines and a 1.7 GB tiototo.asc); the line programs stream their inputs and outputs in blocks and
SYNTHE/SPECTRV work on layer blocks / point chunks, so the peak RSS does not grow with the window.  Here a
100 nm window (500-600 nm, 109,378 points, ~10x the reference windows) runs the whole chain in a child process;
the peak RSS (VmHWM, reset before each program) of every program must stay below 1.5 GB and the chain below
3 GB (the budget of the 2M-point batches; measured there: see RESULTS.md / the wide-window report).
The same SYNTHE inputs run again with small layer blocks and point chunks must give the same fort.9 bytes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.stage

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
MODEL = EXT / "runs" / "stock" / "atlas" / "sun" / "output_synthe.out"
WIN = (500.0, 600.0)

CHILD = r"""
import json, sys, time
from pathlib import Path
from atlas_torch.synthe.chain import run_synthe_chain

def hwm_reset():
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")

def hwm():
    for l in open("/proc/self/status"):
        if l.startswith("VmHWM"):
            return int(l.split()[1]) / 1024.0

peaks, cur = {}, [None]
def progress(step):
    if cur[0] is not None:
        peaks[cur[0]] = hwm()
    hwm_reset()
    cur[0] = step

r = run_synthe_chain(sys.argv[1], float(sys.argv[3]), float(sys.argv[4]), 600000., 1.5, sys.argv[2],
                     "BasicATLAS", keep=True, progress=progress)
peaks[cur[0]] = hwm()
import resource
print(json.dumps(dict(peaks=peaks, timings=r["timings"],
                      maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)))
"""


def _hwm_resettable() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def scratch():
    """~0.5 GB of work files: on $ATLAS_EXT/scratch (disk), not in the RAM-backed /tmp."""
    base = EXT / "scratch" / f"test_synthe_memory_{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(base).free < 2 << 30:
        pytest.skip("needs ~1 GB of scratch space")
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="module")
def chain(scratch):
    if not MODEL.exists():
        pytest.skip(f"missing {MODEL}")
    if not _hwm_resettable():
        pytest.skip("needs /proc/self/clear_refs (Linux) to measure per-program peak RSS")
    wd = scratch / "synthe_1"
    out = subprocess.run([sys.executable, "-c", CHILD, str(MODEL), str(wd), *map(str, WIN)], capture_output=True,
                         text=True, check=True)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    return dict(wd=wd, **res)


def test_peak_rss_per_program(chain):
    print("\n" + "\n".join(f"  {k:12s} {v:8.0f} MB  {chain['timings'][k if k != 'line lists' else 'lines']:7.1f} s"
                           for k, v in chain["peaks"].items()))
    assert set(chain["peaks"]) == {"xnfpelsyn", "line lists", "synthe", "spectrv"}
    for step, mb in chain["peaks"].items():
        assert mb < 1536, (step, mb)
    assert chain["maxrss"] < 3072


def test_synthe_layer_blocks_same_bytes(chain, scratch):
    """Re-run SYNTHE on the chain's inputs with 5-layer blocks and 10k-point transposition chunks."""
    from atlas_torch.synthe.lines import prepare_lines
    from atlas_torch.synthe.chain import MOLECULAR_LISTS, basicatlas_dir
    from atlas_torch.synthe.synthe_prog import SyntheProgram
    wd = chain["wd"]
    s = basicatlas_dir() / "data" / "synthe_files"
    d = basicatlas_dir() / "data" / "dfsynthe_files"
    w2 = scratch / "blocks"
    prepare_lines(w2, *WIN, 600000., 1.5, s / "BasicATLAS.dat", [s / m for m in MOLECULAR_LISTS],
                  d / "h2ofastfix.bin")
    for name, src in (("fort.10", wd / "xnfpelsyn.dat"), ("fort.18", s / "he1tables.dat")):
        os.symlink(src, w2 / name)
    SyntheProgram(w2, layer_block=5, point_chunk=10_000, keep_arrays=False).run(stdout_path=w2 / "synthe.out")
    assert (w2 / "fort.9").read_bytes() == (wd / "fort.9").read_bytes()
    assert (w2 / "fort.14").read_bytes() == (wd / "fort.14").read_bytes()
    assert (w2 / "synthe.out").read_text() == (wd / "synthe.out").read_text()
