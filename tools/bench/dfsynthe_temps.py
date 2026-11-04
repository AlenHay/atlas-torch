"""Benchmark: torch DFSYNTHE per temperature (dfsynthe.run_temperature, stock flavour, FTZ, 1 torch thread),
as one worker of the run_odf pool runs it.

    .venv/bin/python tools/bench/dfsynthe_temps.py --temps 1,21,38 [--mix solar] [--core 3] [--profile DIR]
                     [--baseline DIR --save | --baseline DIR] [--json out.json]

Temperature indices are 1-based (1 = 1995 K, 21 = 5012 K, 38 = 22387 K).  Inputs: the stock XNFDF outputs of the
mix ($ATLAS_EXT/runs/stock/odf/<mix>/xnfpdf.dat, xnfpdfmax.dat) and BasicATLAS data/dfsynthe_files.  Per temperature
it records the wall/CPU time of every phase (selectlines, continuum, xlinop, lines, broaden+dfcalc), the peak RSS
(VmHWM, reset per temperature) and, with --baseline, whether the ODF steps are identical to a baseline saved earlier
(--save writes it: run once on the unmodified code).  The stock dfsynthe.exe per temperature: see
dfsynthe_fortran.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "bench"))
from _common import EXT, bench_env, hwm_mb, hwm_reset, pin  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--temps", default="1,21,38")
    ap.add_argument("--mix", default="solar")
    ap.add_argument("--core", type=int, default=None)
    ap.add_argument("--profile", default=None, help="directory for cProfile files t<NN>.prof")
    ap.add_argument("--baseline", default=None, help="directory of baseline steps t<NN>.npz")
    ap.add_argument("--save", action="store_true", help="write the baseline instead of comparing")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pin(a.core, 1)
    import numpy as np
    import torch  # noqa: F401
    from atlas_torch.dfsynthe.dfsynthe import Config, run_temperature
    from atlas_torch.dfsynthe.lines.f32 import Flavor
    from atlas_torch.dfsynthe.lines.native import load as native_load
    from atlas_torch.dfsynthe.pipeline import default_data_dir
    from atlas_torch.dfsynthe import odf_io

    src = EXT / "runs" / "stock" / "odf" / a.mix
    out = dict(bench="dfsynthe_temps", mix=a.mix, native=native_load() is not None, temps={},
               **bench_env(a.core, 1))
    for it in [int(x) for x in a.temps.split(",")]:
        cfg = Config(flavor=Flavor("stock", fastmath=True))
        hwm_reset()
        if a.profile:
            import cProfile
            pr = cProfile.Profile(); pr.enable()
        t0, c0 = time.time(), time.process_time()
        res = run_temperature(src / "xnfpdf.dat", src / "xnfpdfmax.dat", default_data_dir(), it, cfg)
        wall, cpu = time.time() - t0, time.process_time() - c0
        if a.profile:
            pr.disable()
            Path(a.profile).mkdir(parents=True, exist_ok=True)
            pr.dump_stats(Path(a.profile) / f"t{it:02d}.prof")
        info = res["info"]
        rec = dict(teff=float(odf_io.DFTS[it - 1]), wall=wall, cpu=cpu, peak_rss_mb=hwm_mb(),
                   phases={k: round(v, 2) for k, v in info["timing"].items()},
                   phases_cpu={k: round(v, 2) for k, v in info["cpu"].items()}, counts=info["counts"])
        steps = {f"v{v}": res[v][1] for v in odf_io.VTURBS}
        if a.baseline:
            bp = Path(a.baseline) / f"{a.mix}_t{it:02d}.npz"
            if a.save:
                bp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(bp, **steps)
            elif bp.exists():
                z = np.load(bp)
                rec["identical_to_baseline"] = all(np.array_equal(z[k], steps[k]) for k in steps)
        out["temps"][it] = rec
        print(json.dumps({it: rec}), flush=True)
    out["loadavg_end"] = list(__import__("os").getloadavg())
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
