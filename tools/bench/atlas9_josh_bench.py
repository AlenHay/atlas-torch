"""Micro-benchmark of JOSH on the inputs of one ATLAS9 iteration (solar traced run, 337 frequencies x 13 problems).
Captures JOSH's arguments from iteration 1 of the program, then times josh() (min of --repeat), optionally with
cProfile. Usage (container): taskset -c 2 .venv/bin/python tools/bench/atlas9_josh_bench.py [--repeat 3] [--cprofile 25]
"""
import argparse
import cProfile
import json
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cprofile", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(1)
    import atlas9_profile as P
    from atlas_torch.atlas9 import main as M
    from atlas_torch.atlas9.josh import josh
    cap = {}

    def grab(*args, **kw):
        cap.setdefault("args", (args, kw))
        return josh(*args, **kw)
    M.josh = grab
    with tempfile.TemporaryDirectory() as td:
        P.run_torch(1, Path(td))
    M.josh = josh
    args, kw = cap["args"]
    with P_float64():
        ts = []
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            josh(*args, **kw)
            ts.append(time.perf_counter() - t0)
        print(json.dumps({"B": int(args[0].shape[0]), "min_s": round(min(ts), 4), "loadavg": os.getloadavg()}))
        if a.cprofile:
            pr = cProfile.Profile()
            pr.enable()
            josh(*args, **kw)
            pr.disable()
            pstats.Stats(pr).sort_stats("tottime").print_stats(a.cprofile)


def P_float64():
    from atlas_torch.atlas9.program import float64_default
    return float64_default()


if __name__ == "__main__":
    main()
