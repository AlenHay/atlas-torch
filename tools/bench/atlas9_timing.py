"""Per-model timing table: stock Fortran atlas9mem.exe vs the torch port (this checkout) and optionally a baseline
torch source tree, interleaved round by round on the same core, one torch thread (re-runnable on a quiet machine).

    # baseline tree (read-only copy of a commit):  git archive f6a611f src | tar -x -C $ATLAS_EXT/scratch/atlas9_bench/head
    scripts/dx '.venv/bin/python tools/bench/atlas9_timing.py --core 2 --rounds 3 \\
        --baseline $HOME/atlas-ext/scratch/atlas9_bench/head/src --json results/m7/atlas9_timing.json'

Each torch run is a fresh process running tools/bench/atlas9_profile.py --check (stdout and fort.7 compared with the
stock run, which must be byte-identical); the reported time is the model run (READIN .. last iteration) inside the
process (imports excluded), plus the process wall time. The Fortran is timed as a subprocess (wall and CPU).
Output: min / median per program, the min-ratio torch/Fortran, load averages before and after every run.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))


def torch_run(core: int, iters: int, src: str | None, label: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    cmd = ["taskset", "-c", str(core), sys.executable, str(HERE / "atlas9_profile.py"), "--iters", str(iters),
           "--check", "--json", out, "--label", label]
    if src:
        cmd += ["--src", src]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, cwd=REPO)
    wall = time.perf_counter() - t0
    d = json.loads(Path(out).read_text())
    os.unlink(out)
    return {"run_s": d["torch_seconds"], "cpu_s": d.get("torch_cpu_seconds"), "process_s": round(wall, 3),
            "identical": bool(d.get("identical_stdout") and d.get("identical_fort7")),
            "loadavg": [d["loadavg_before"][0], d["loadavg_after"][0]]}


def fortran_run(core: int, iters: int) -> dict:
    import atlas9_profile as P
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        P.setup(d)
        la = os.getloadavg()[0]
        r0 = os.times()
        t0 = time.perf_counter()
        subprocess.run(["taskset", "-c", str(core), str(P.EXE)], input=P.stdin_text(iters).encode(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=d, timeout=3600)
        wall = time.perf_counter() - t0
        r1 = os.times()
    return {"run_s": round(wall, 3), "cpu_s": round(r1.children_user - r0.children_user
                                                   + r1.children_system - r0.children_system, 3),
            "loadavg": [la, os.getloadavg()[0]]}


def summary(rows):
    v = [r["run_s"] for r in rows]
    return {"min_s": min(v), "median_s": statistics.median(v), "runs": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--baseline", default=None, help="src/ of a baseline atlas_torch tree")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = {"fortran": [], "torch": [], "baseline": []}
    for r in range(a.rounds):
        res["fortran"].append(fortran_run(a.core, a.iters))
        res["torch"].append(torch_run(a.core, a.iters, None, "timing"))
        if a.baseline:
            res["baseline"].append(torch_run(a.core, a.iters, a.baseline, "timing-baseline"))
        print("round", r, {k: v[-1]["run_s"] for k, v in res.items() if v}, flush=True)
    out = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "core": a.core, "iters": a.iters, "rounds": a.rounds,
           "model": "sun (traced BasicATLAS solar run, 15 iterations)",
           **{k: summary(v) for k, v in res.items() if v}}
    f = out["fortran"]["min_s"]
    out["ratio_torch_over_fortran_min"] = round(out["torch"]["min_s"] / f, 2)
    if a.baseline:
        out["ratio_baseline_over_fortran_min"] = round(out["baseline"]["min_s"] / f, 2)
        out["speedup_vs_baseline_min"] = round(out["baseline"]["min_s"] / out["torch"]["min_s"], 2)
        out["speedup_vs_baseline_median"] = round(out["baseline"]["median_s"] / out["torch"]["median_s"], 2)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))
    for k in ("fortran", "torch", "baseline"):
        if k in out:
            print(k, "min", out[k]["min_s"], "median", out[k]["median_s"])
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
