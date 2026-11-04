"""Timing of the differentiable SYNTHE path (atlas_torch.diff, M8): setup, one forward pass, one JVP.

    scripts/dx '.venv/bin/python tools/bench/diff_gradients.py [--window 500 510] [--threads 4] [--out f.json]'

Reports wall times (s) with the load average, torch thread count and peak RSS; JSON to --out
(default results/m8/bench_diff_gradients.json). Pin with `taskset -c <cores>` for per-core comparisons.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch


def main():
    ext = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ext / "runs/stock/atlas/sun"))
    ap.add_argument("--window", nargs=2, type=float, default=[500.0, 510.0])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(repo / "results/m8/bench_diff_gradients.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from atlas_torch.diff import SyntheDiff
    res = {"window": a.window, "threads": a.threads, "load_before": os.getloadavg()}
    t = time.perf_counter()
    sd = SyntheDiff(a.model, *a.window)
    res["setup_s"] = time.perf_counter() - t
    res["points"] = int(sd.wave.numel())
    res["pairs"] = int(sd.lines.pairs.j.numel())
    t = time.perf_counter()
    with torch.no_grad():
        o = sd({})
    res["forward_s"] = time.perf_counter() - t
    res["forward_parts_s"] = o["timings"]
    for name in ("abund:Fe", "vturb", "T[40]"):
        t = time.perf_counter()
        sd.jacobian({}, [name])
        res[f"jvp_{name}_s"] = time.perf_counter() - t
    res["load_after"] = os.getloadavg()
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
