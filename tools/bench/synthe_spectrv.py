"""Benchmark: torch SPECTRV vs the stock Fortran spectrv.exe on stage-isolated inputs (a pre_spectrv/ snapshot).

    .venv/bin/python tools/bench/synthe_spectrv.py --window 500_510 [--core 3] [--threads 1] [--fortran]
                     [--profile] [--json out.json]

Windows: 500_510, 380_420 ($ATLAS_EXT/work/sun_<w>_stock) and 100_2850 ($ATLAS_EXT/scratch/wide/stock_100_2850).
The inputs are copied (fort.9 symlinked) into a scratch dir under $ATLAS_EXT/scratch/bench, the torch program
runs in-process (READIN of the stock solar deck, then spectrv(libm="libmvec") as the chain does), the Fortran
(stock spectrv.exe, optional) as a subprocess on the same core. Checks that the torch fort.7 is byte-identical
to the stock spectrum.bin. Prints/writes JSON: seconds, peak RSS, load average, threads, core, identical.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "bench"))
from _common import EXT, bench_env, deck_for, hwm_mb, hwm_reset, pin, run_fortran, same_file, workdir_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", default="500_510")
    ap.add_argument("--core", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fortran", action="store_true", help="also time the stock spectrv.exe on the same core")
    ap.add_argument("--profile", default=None, help="write a cProfile .prof here (torch run)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pin(a.core, a.threads)
    import torch  # noqa: F401
    from atlas_torch.synthe import spectrv_prog as sp
    from atlas_torch.synthe.chain import FORT25, basicatlas_dir
    from atlas_torch.synthe.readin import readin

    src = workdir_for(a.window)
    deck = deck_for(a.window)
    wd = EXT / "scratch" / "bench" / f"spectrv_{a.window}_{os.getpid()}"
    wd.mkdir(parents=True, exist_ok=True)
    try:
        for f in (src / "pre_spectrv").glob("fort.*"):
            (wd / f.name).symlink_to(f)
        (wd / "fort.25").write_text(FORT25)
        state = readin(deck.read_text(), molecules_path=basicatlas_dir() / "data/synthe_files/molecules.dat",
                       mode=20)
        times = []
        hwm_reset()
        for _ in range(a.repeat):
            t = time.perf_counter()
            if a.profile:
                import cProfile
                pr = cProfile.Profile(); pr.enable()
            res = sp.spectrv(state, sp.load_inputs(wd), libm="libmvec")
            if a.profile:
                pr.disable(); pr.dump_stats(a.profile)
            out = wd / "torch_fort7"
            res.write(out)
            times.append(time.perf_counter() - t)
        identical = same_file(out, src / "spectrum.bin")
        rec = dict(bench="synthe_spectrv", window=a.window, torch_seconds=min(times), torch_all=times,
                   torch_peak_rss_mb=hwm_mb(),
                   identical_to_stock=identical, npoints=int(res.q.shape[0]), **bench_env(a.core, a.threads))
        if a.fortran:
            fw = wd / "fortran"
            fw.mkdir()
            for f in (src / "pre_spectrv").glob("fort.*"):     # the Fortran opens its units read-write
                shutil.copy(f, fw / f.name)
                os.chmod(fw / f.name, 0o644)
            (fw / "fort.25").write_text(FORT25)
            el, kb = run_fortran(basicatlas_dir() / "bin" / "spectrv.exe", fw, a.core, stdin=deck,
                                 stdout=fw / "spectrv.out")
            rec.update(fortran_seconds=el, fortran_peak_rss_mb=kb / 1024,
                       fortran_identical_to_stock=same_file(fw / "fort.7", src / "spectrum.bin"))
            rec["speed_ratio_torch_over_fortran"] = rec["torch_seconds"] / rec["fortran_seconds"]
        print(json.dumps(rec, indent=1))
        if a.json:
            Path(a.json).write_text(json.dumps(rec, indent=1) + "\n")
    finally:
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    main()
