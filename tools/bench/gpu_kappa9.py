"""M7 study, KAPPA9 (continuum KAPP on the 42x57 grid + Rosseland means of the ODF): whole kappa9/<mix> stage per mode.

  tools/bench/gpu_run.sh cpu tools/bench/gpu_kappa9.py fortran --mix solar        # times scripts/kappa9_workdir.sh
  tools/bench/gpu_run.sh cpu tools/bench/gpu_kappa9.py run --mode exact --mix solar   # exact|native|cpu32|cpu32p
  tools/bench/gpu_run.sh gpu tools/bench/gpu_kappa9.py run --mode gpu64 --mix solar   # gpu64|gpu32|gpu32p
Scored with atlas_torch.scoring.kappa9_metrics vs $ATLAS_EXT/work/kappa9_<mix>_stock, gated on the frozen
kappa9/<mix> tolerances. Results: results/m7/gpu_float32_compile.json["kappa9/<mix>/<mode>"].

Device notes: kappa9_prog.run_workdir(device=...) puts the grid, the continuum tables and the Rosseland sums on the
device; the physics (populations per pressure column, NMOLEC, continuum per frequency on 42x57 points) follows the
device of the grid tensors it is given.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_common as G  # noqa: E402

import torch  # noqa: E402

BASE = G.SCRATCH / "kappa9"


def fortran(mix: str, repeat: int = 1) -> dict:
    ts = []
    for _ in range(repeat):
        wd = BASE / f"fortran_{mix}"
        shutil.rmtree(wd, ignore_errors=True)
        t0 = time.perf_counter()
        subprocess.run(["bash", str(G.REPO / "scripts/kappa9_workdir.sh"), str(G.EXT / "BasicATLAS/bin"),
                        str(G.EXT / "runs/stock/odf" / mix), str(wd)], check=True, stdout=subprocess.DEVNULL)
        ts.append(time.perf_counter() - t0)
    from atlas_torch.scoring import kappa9_metrics
    m = kappa9_metrics(G.EXT / "work" / f"kappa9_{mix}_stock", wd)
    rec = {"seconds": min(ts), "runs_s": ts, "same_as_stock": G.gate(f"kappa9/{mix}", m)["worst"][1] == 0,
           "env": G.env_info()}
    G.merge_result(f"kappa9/{mix}/fortran", rec)
    return rec


def run(mix: str, mode: str, threads: int, repeat: int) -> dict:
    torch.set_num_threads(threads)
    dev = G.mode_device(mode)
    from atlas_torch.kappa9.kappa9_prog import run_workdir
    from atlas_torch.scoring import kappa9_metrics
    rec = {"mode": mode, "env": G.env_info(), "runs_s": []}
    wd = BASE / f"{mode}_{mix}"
    try:
        for _ in range(repeat):
            shutil.rmtree(wd, ignore_errors=True)
            co = G.DeviceCoerce(dev) if dev.type != "cpu" else None
            with G.mode_context(mode, co), G.Timer(dev) as t:
                info = run_workdir(G.EXT / "runs/stock/odf" / mix, wd, device=dev)
            rec["runs_s"].append(t.s)
            if co is not None:
                rec["device_coercions"] = dict(sorted(co.sites.items(), key=lambda kv: -kv[1]))
            rec["timing"] = {k: round(v, 3) for k, v in info["timing"].items()}
        rec["seconds"], rec["cold_s"] = min(rec["runs_s"]), rec["runs_s"][0]
        m = kappa9_metrics(G.EXT / "work" / f"kappa9_{mix}_stock", wd)
        rec["metrics"] = {k: v for k, v in m.items() if ":" in k and "rms" not in k}
        rec["gate"] = G.gate(f"kappa9/{mix}", m)
        rec["status"] = "ok"
        shutil.rmtree(wd, ignore_errors=True)
    except ValueError as e:
        if "*****" not in str(e):
            raise
        rec["status"] = "fail_overflow"          # the kapk table overflowed its print field (Fortran prints *****)
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["gate"] = {"pass": False, "worst": ["kapk print overflow", float("inf")]}
    except Exception as e:  # noqa: BLE001
        rec["status"] = "crash"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["loadavg_end"] = round(os.getloadavg()[0], 2)
    print(f"kappa9/{mix} {mode:6s} {rec['status']} {rec.get('seconds', float('nan')):.2f}s "
          f"pass={rec.get('gate', {}).get('pass')} worst={rec.get('gate', {}).get('worst')} {rec.get('error', '')}")
    if rec["status"] == "crash":
        print(rec["traceback"])
    G.merge_result(f"kappa9/{mix}/{mode}", {k: v for k, v in rec.items() if k != "traceback"})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fortran", "run"])
    ap.add_argument("--mix", default="solar")
    ap.add_argument("--mode", default="exact", choices=G.MODES)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "fortran":
        print(fortran(a.mix, a.repeat))
    else:
        run(a.mix, a.mode, a.threads, a.repeat)


if __name__ == "__main__":
    main()
