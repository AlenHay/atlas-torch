"""Micro-benchmark of NMOLEC on the traced solar states (golden hits 0..2: iteration-1 populations with IFEDNS=0,
the first two CONVEC calls with IFEDNS=1). Usage (container):
    taskset -c 2 .venv/bin/python tools/bench/atlas9_nmolec_bench.py [--repeat 3] [--cprofile 30] [--json out.json]
"""
import argparse
import cProfile
import json
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
TR = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "traces"


def golden(stage, routine, hit=0, phase="entry"):
    with np.load(TR / stage / f"{routine.lower()}_{hit}_{phase}.npz") as d:
        return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cprofile", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    from atlas_torch.atlas9.physics import _testing as T
    from atlas_torch.atlas9.physics.molec import nmolec
    from atlas_torch.atlas9.physics import pfsaha as PF
    states = [(T.nmolec_state(golden, h), int(golden(T.TRACE_STAGE, "nmolec", h, "entry")["arg:mode"][0])) for h in range(3)]
    nmolec(*states[0])                                   # warm-up (tables, lru caches)
    out = {"loadavg": os.getloadavg(), "hits": {}}
    prof = cProfile.Profile() if a.cprofile else None
    for h, (s, mode) in enumerate(states):
        ts = []
        for _ in range(a.repeat):
            PF._STAGE1.clear(); PF._SIZES.clear(); PF._STAGE1_BYTES[0] = 0      # cold PFSAHA stage-1 memo
            t0 = time.perf_counter()
            if prof:
                prof.enable()
            nmolec(s, mode)
            if prof:
                prof.disable()
            ts.append(time.perf_counter() - t0)
        out["hits"][h] = {"ifedns": int(s.iscalar("edens.ifedns")), "min_s": round(min(ts), 4)}
    print(json.dumps(out))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    if prof:
        pstats.Stats(prof).sort_stats("tottime").print_stats(a.cprofile)


if __name__ == "__main__":
    main()
