"""M7 study, ATLAS9 (populations/continuum KAPP, JOSH, TCORR, ... per iteration): one whole model per mode.

  tools/bench/gpu_run.sh cpu tools/bench/gpu_atlas9.py --model sun --mode fortran   # stock atlas9mem.exe, timed
  tools/bench/gpu_run.sh cpu tools/bench/gpu_atlas9.py --model sun --mode exact     # exact|native|cpu32|cpu32p
  tools/bench/gpu_run.sh gpu tools/bench/gpu_atlas9.py --model sun --mode gpu64     # gpu64|gpu32|gpu32p
Same driver as the atlas/<model> stage (compat.atlas.atlas, stock ODF of the model's mix, BasicATLAS defaults,
restart auto, niter 450), output under $ATLAS_EXT/scratch/m7_gpu/atlas9/<mode>_<model> (not the scorer's dir),
scored with atlas_torch.scoring.atlas_metrics on the frozen atlas/<model> gates.

GPU: ATLAS9 has no device argument (the State is created by READIN on the CPU and the program keeps the Fortran's
scalar control flow). The GPU modes run the whole program under torch.set_default_device('cuda') (factory
functions allocate on the GPU) plus gpu_common.DeviceCoerce (stray CPU tensors from numpy tables are moved when
an op mixes devices; .numpy() of device tensors copies to the CPU). The coercion sites are the device bug list.
--max-iter limits the run (timing per iteration; the gates need the full run).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_common as G  # noqa: E402

import torch  # noqa: E402

BASE = G.SCRATCH / "atlas9"


@contextlib.contextmanager
def default_device(dev):
    if dev.type == "cpu":
        yield
        return
    old = torch.get_default_device()
    torch.set_default_device(dev)
    try:
        yield
    finally:
        torch.set_default_device(old)


@contextlib.contextmanager
def iteration_counter(rec, max_iter=None):
    """Counts ATLAS9 iterations (atlas9.main.iteration) and their wall times; optionally stops the run."""
    from atlas_torch.atlas9 import main
    orig = main.iteration
    rec["iteration_s"] = []

    class Stop(Exception):
        def __str__(self):
            return "M7_STOP"

    def it(*a, **k):
        if max_iter is not None and len(rec["iteration_s"]) >= max_iter:
            raise Stop()
        G.sync()
        t0 = time.perf_counter()
        try:
            r = orig(*a, **k)
        except Exception:
            rec["inner_traceback"] = traceback.format_exc()[-4000:]     # compat's script runner swallows it
            raise
        G.sync()
        rec["iteration_s"].append(time.perf_counter() - t0)
        return r
    main.iteration = it
    try:
        yield Stop
    finally:
        main.iteration = orig


def run(model: str, mode: str, threads: int, max_iter=None, record: bool = True) -> dict:
    torch.set_num_threads(threads)
    ref = json.loads((G.REPO / "reference/reference_set.json").read_text())
    m = ref["models"][model]
    from atlas_torch.compat import atlas as api
    from atlas_torch.compat.settings import Settings
    from atlas_torch.scoring import atlas_metrics
    s = Settings()
    s.zscale = ref["odfs"][m["odf"]]["zscale"]
    s.abun = dict(ref["odfs"][m["odf"]]["abun"])
    s.teff, s.logg, s.vturb = m["teff"], m["logg"], ref["atlas"]["vturb"]
    out = BASE / f"{mode}_{model}"
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["ATLAS_TORCH_BACKEND"] = "fortran" if mode == "fortran" else "torch"
    dev = G.mode_device(mode)
    rec = {"mode": mode, "model": model, "env": G.env_info()}
    co = G.DeviceCoerce(dev) if dev.type != "cpu" else None
    try:
        with contextlib.ExitStack() as st:
            Stop = st.enter_context(iteration_counter(rec, max_iter)) if mode != "fortran" else None
            if mode != "fortran":
                st.enter_context(G.mode_context(mode, co))
                st.enter_context(default_device(dev))
            t = st.enter_context(G.Timer(dev))
            try:
                api.atlas(str(out), s, restart=ref["atlas"]["restart"], niter=ref["atlas"]["niter"],
                          ODF=str(G.EXT / "runs/stock/odf" / m["odf"]), silent=True)
            except Exception as e:  # noqa: BLE001
                if "M7_STOP" in str(e):
                    rec["stopped_after"] = max_iter
                else:
                    raise
        rec["seconds"] = t.s
        rec["loadavg"] = t.load
        if "iteration_s" in rec and rec["iteration_s"]:
            its = sorted(rec["iteration_s"])
            rec["iterations"] = len(its)
            rec["iteration_median_s"] = its[len(its) // 2]
            rec["iteration_first_s"] = rec["iteration_s"][0]
            del rec["iteration_s"]
        if "stopped_after" not in rec:
            mt = atlas_metrics(G.EXT / "runs/stock/atlas" / model, out)
            rec["metrics"] = {k: v for k, v in mt.items() if ":" in k and "rms" not in k}
            rec["iterations_tables"] = mt.get("iterations_tables")
            rec["gate"] = G.gate(f"atlas/{model}", mt)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "crash"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-4000:]
    if co is not None:
        rec["device_coercions"] = dict(sorted(co.sites.items(), key=lambda kv: -kv[1])[:60])
        rec["device_coercion_sites"] = len(co.sites)
    print(f"atlas/{model} {mode:7s} {rec['status']} {rec.get('seconds', float('nan')):.1f}s "
          f"its={rec.get('iterations')} it_med={rec.get('iteration_median_s')} "
          f"pass={rec.get('gate', {}).get('pass')} worst={rec.get('gate', {}).get('worst')} {rec.get('error', '')[:300]}",
          flush=True)
    if rec["status"] == "crash":
        print(rec.get("inner_traceback") or rec["traceback"])
    key = f"atlas/{model}/{mode}" + (f"/max{max_iter}" if max_iter else "")
    if record:
        G.merge_result(key, {k: v for k, v in rec.items() if k != "traceback"})
    if rec["status"] == "ok" and not os.environ.get("M7_KEEP"):
        shutil.rmtree(out, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sun")
    ap.add_argument("--mode", default="exact", choices=G.MODES + ("fortran",))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.mode, a.threads, a.max_iter)


if __name__ == "__main__":
    main()
