"""Benchmark: torch SYNTHE (synthe_prog.run_synthe) vs the stock synthe.exe on stage-isolated inputs (pre_synthe/).

    .venv/bin/python tools/bench/synthe_synthe.py --window 380_420 [--core 3] [--threads 1] [--fortran]
                     [--profile out.prof] [--json out.json] [--no-check]

Copies pre_synthe/ (+ he1tables.dat as fort.18) to $ATLAS_EXT/scratch/bench, runs the torch program in-process
(peak RSS = VmHWM of this process during the run), optionally the stock exe on the same core, and checks that the
torch fort.9 / fort.14 / fort.8 are byte-identical to the stock ones (pre_spectrv/) unless --no-check.
For the wide window (100_2850) the copy is ~2 GB; the Fortran there takes ~20 min.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "bench"))
from _common import EXT, bench_env, hwm_mb, hwm_reset, pin, run_fortran, same_file, workdir_for  # noqa: E402

HE1 = EXT / "BasicATLAS" / "data" / "synthe_files" / "he1tables.dat"


def _prepare(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for f in (src / "pre_synthe").glob("fort.*"):
        shutil.copy(f, dst / f.name)
        os.chmod(dst / f.name, 0o644)
    shutil.copy(HE1, dst / "fort.18")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", default="500_510")
    ap.add_argument("--core", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fortran", action="store_true")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--layers", default=None, help="profile only these layer blocks, e.g. '0:4,34:38,68:72' "
                    "(0-based, end exclusive; outputs incomplete, implies --no-check)")
    a = ap.parse_args()
    pin(a.core, a.threads)
    from atlas_torch.synthe.synthe_prog import run_synthe

    src = workdir_for(a.window)
    base = EXT / "scratch" / "bench" / f"synthe_{a.window}_{os.getpid()}"
    try:
        wd = _prepare(src, base / "torch")
        hwm_reset()
        t = time.perf_counter()
        if a.profile:
            import cProfile
            pr = cProfile.Profile(); pr.enable()
        if a.layers:
            from atlas_torch.synthe.synthe_prog import SyntheProgram
            blocks = [tuple(map(int, b.split(":"))) for b in a.layers.split(",")]
            prog = SyntheProgram(wd)
            prog._blocks = lambda: blocks
            prog.run(stdout_path=wd / "synthe.out")
            a.no_check = True
        else:
            run_synthe(wd, stdout_path=wd / "synthe.out")
        if a.profile:
            pr.disable(); pr.dump_stats(a.profile)
        rec = dict(bench="synthe_synthe", window=a.window, layers=a.layers, torch_seconds=time.perf_counter() - t,
                   torch_peak_rss_mb=hwm_mb(), **bench_env(a.core, a.threads))
        if not a.no_check:
            ref = src / "pre_spectrv"
            rec["identical_to_stock"] = {f: same_file(wd / f, ref / f) for f in ("fort.9", "fort.14", "fort.8")}
            rec["identical_to_stock"]["synthe.out"] = same_file(wd / "synthe.out", src / "synthe.out")
        shutil.rmtree(wd)
        if a.fortran:
            fw = _prepare(src, base / "fortran")
            el, kb = run_fortran(EXT / "BasicATLAS" / "bin" / "synthe.exe", fw, a.core, stdout=fw / "synthe.out")
            rec.update(fortran_seconds=el, fortran_peak_rss_mb=kb / 1024)
            rec["speed_ratio_torch_over_fortran"] = rec["torch_seconds"] / el
        print(json.dumps(rec, indent=1))
        if a.json:
            Path(a.json).write_text(json.dumps(rec, indent=1) + "\n")
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
