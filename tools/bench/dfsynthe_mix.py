"""Benchmark: a full DFSYNTHE mix (XNFDF -> 57 temperatures -> SEPARATEDF -> KAPPA9) with atlas_torch.dfsynthe.pipeline
.run_odf and N worker processes (the "full mix with 4 workers" figure), byte-compared with reference outputs.

    .venv/bin/python tools/bench/dfsynthe_mix.py --mix solar [--workers 4] [--ref DIR] [--json out.json] [--keep]

Inputs: the stock xnfdf.com of the mix ($ATLAS_EXT/runs/stock/odf/<mix>/xnfdf.com).  Output in
$ATLAS_EXT/scratch/bench/odf_<mix>_<pid> (deleted unless --keep).  --ref: directory with p00big*/p00lit*.bdf and
kappa.ros to compare byte-for-byte (default $ATLAS_EXT/scratch/m7bench/odf_before/<mix>, the outputs of the code
before M7).  Records: total wall, per-temperature worker walls, load average, peak memory note.
Memory: ~1.6-1.9 GB per worker.
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
from _common import EXT, bench_env, same_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mix", default="solar")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    from atlas_torch.dfsynthe.pipeline import run_odf
    out = EXT / "scratch" / "bench" / f"odf_{a.mix}_{os.getpid()}"
    ref = Path(a.ref) if a.ref else EXT / "scratch" / "m7bench" / "odf_before" / a.mix
    rec = dict(bench="dfsynthe_mix", mix=a.mix, workers=a.workers, **bench_env(None, 1))
    loads = []
    try:
        t0 = time.time()
        info = run_odf(out, xnfdf_com=EXT / "runs" / "stock" / "odf" / a.mix / "xnfdf.com", workers=a.workers,
                       log=lambda s: (loads.append(os.getloadavg()[0]), print(s, flush=True)))
        rec["wall_seconds"] = time.time() - t0
        rec["timing"] = info["timing"]
        rec["walls"] = {int(k): v for k, v in info["walls"].items()}
        rec["loadavg_during_max"] = max(loads) if loads else None
        rec["loadavg_end"] = list(os.getloadavg())
        if ref.exists():
            names = [f"p00{p}{v}.bdf" for v in (0, 1, 2, 4, 8) for p in ("big", "lit")] + ["kappa.ros"]
            rec["identical_to_ref"] = {n: same_file(out / n, ref / n) for n in names}
            rec["all_identical"] = all(rec["identical_to_ref"].values())
        print(json.dumps({k: v for k, v in rec.items() if k != "walls"}, indent=1))
        if a.json:
            Path(a.json).write_text(json.dumps(rec, indent=1) + "\n")
    finally:
        if not a.keep:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
