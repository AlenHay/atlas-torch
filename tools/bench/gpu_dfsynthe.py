"""M7 study, DFSYNTHE: single temperatures (1-based index into the 57 DFTS) on the stock xnfpdf.dat/xnfpdfmax.dat of a
mix, compared with the stock ODF entries of that temperature (p00big/p00lit*.bdf rows of that IT).

  tools/bench/gpu_run.sh cpu tools/bench/gpu_dfsynthe.py --temps 33 --mode fortran    # stock dfsynthe.exe, timed
  tools/bench/gpu_run.sh cpu tools/bench/gpu_dfsynthe.py --temps 33 --mode exact      # C line kernel, FTZ, stock
  tools/bench/gpu_run.sh cpu tools/bench/gpu_dfsynthe.py --temps 33 --mode native     # torch line path (no C kernel)
  tools/bench/gpu_run.sh cpu tools/bench/gpu_dfsynthe.py --temps 33 --mode cpu32      # REAL*8 parts in float32
  tools/bench/gpu_run.sh gpu tools/bench/gpu_dfsynthe.py --temps 33 --mode gpu_lines  # Config.lines_device=cuda
  tools/bench/gpu_run.sh gpu tools/bench/gpu_dfsynthe.py --temps 33 --mode gpu64      # Config.device=cuda (all)
  tools/bench/gpu_run.sh gpu tools/bench/gpu_dfsynthe.py --temps 33 --mode gpu32      # all on GPU + float32 emu
  tools/bench/gpu_run.sh cpu tools/bench/gpu_dfsynthe.py --temps 33 --mode xnfdf32    # XNFDF (REAL*8 populations)
                                                         in float32, then exact DFSYNTHE on its xnfpdf.dat
The line accumulation is REAL*4 in the Fortran (float32 in the port in every mode); "float32" here means the REAL*8
parts (populations, frequencies, layer set-up) in float32.

Metrics per temperature (all 5 microturbulences): max |diff| in ODF steps (milli-dex), share of entries off by > 1
step, p99.9, compared with the frozen odf/<mix> gates for orientation only: the gates are over the 57-temperature
table (max-abs for T >= 4571 K = index 19+, share/p99.9 over all temperatures), a single temperature is a subset.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

BASE = G.SCRATCH / "dfsynthe"
DATA = G.EXT / "BasicATLAS" / "data" / "dfsynthe_files"
LINKS = (("xnfpdf.dat", 10), ("xnfpdfmax.dat", 22), ("lowlinesdf.bin", 11), ("highlinesdf.bin", 21),
         ("diatomicsdf.bin", 31), ("tiolinesdf.bin", 41), ("h2olinesdf.bin", 43), ("nltelinesdf.bin", 51))


def stock_steps(mix: str, it: int) -> dict:
    """{vt: int16 [1540, 25, 12]} of temperature `it` from the stock p00big/p00lit files."""
    from atlas_torch.io.formats import read_bdf
    d = G.EXT / "runs/stock/odf" / mix
    out = {}
    for v in (0, 1, 2, 4, 8):
        big, lit = read_bdf(d / f"p00big{v}.bdf"), read_bdf(d / f"p00lit{v}.bdf")
        out[v] = np.concatenate([big, lit], axis=0)[:, it - 1]
    return out


def compare(mix: str, it: int, res: dict) -> dict:
    ref = stock_steps(mix, it)
    tol = G.tolerances().get(f"odf/{mix}", {})
    out = {}
    worst = 0.0
    for v in (0, 1, 2, 4, 8):
        cand = res[v][1].reshape(25, 1540, 12).transpose(1, 0, 2).astype(np.int64)
        d = np.abs(cand - ref[v].astype(np.int64))
        out[f"odf_v{v}:abs"] = int(d.max())
        out[f"odf_v{v}:share_gt1"] = float((d > 1).mean())
        out[f"odf_v{v}:p999"] = float(np.quantile(d, 0.999))
        out[f"odf_v{v}:frac"] = float((d != 0).mean())
        for k in ("p999", "share_gt1"):
            t = tol.get(f"odf_v{v}:{k}", {}).get("tol")
            if t:
                worst = max(worst, out[f"odf_v{v}:{k}"] / t)
        if it >= 19:
            t = tol.get(f"odf_v{v}_hot:abs", {}).get("tol")
            if t:
                worst = max(worst, out[f"odf_v{v}:abs"] / t)
    out["worst_ratio_vs_table_gates"] = worst
    out["within_table_gates"] = worst <= 1.0
    return out


def fortran(mix: str, it: int) -> dict:
    wd = BASE / f"fortran_{mix}_t{it}"
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    src = G.EXT / "runs/stock/odf" / mix
    for f, u in LINKS:
        os.symlink((src / f) if f.startswith("xnfpdf") else (DATA / f), wd / f"fort.{u}")
    sw = "0" * (it - 1) + "1" + "0" * (57 - it) + "\n"
    t0 = time.perf_counter()
    subprocess.run([str(G.EXT / "BasicATLAS/bin/dfsynthe.exe")], cwd=wd, input=sw.encode(), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sec = time.perf_counter() - t0
    shutil.rmtree(wd, ignore_errors=True)
    return {"seconds": sec, "env": G.env_info(), "status": "ok"}


def run(mix: str, it: int, mode: str, threads: int) -> dict:
    torch.set_num_threads(threads)
    if mode == "fortran":
        rec = fortran(mix, it)
    else:
        if mode == "native":
            os.environ["ATLAS_TORCH_NO_NATIVE"] = "1"       # torch line path (bit-identical by design)
        from atlas_torch.dfsynthe.dfsynthe import Config, run_temperature
        src = G.EXT / "runs/stock/odf" / mix
        rec = {"mode": mode, "env": G.env_info()}
        x10, x11 = src / "xnfpdf.dat", src / "xnfpdfmax.dat"
        cfg = Config()
        dev = torch.device("cpu")
        if mode == "exact_noftz":
            cfg.ftz = False
        if mode == "gpu_lines":
            cfg.lines_device = torch.device("cuda")
            dev = cfg.lines_device
        elif mode in ("gpu64", "gpu32"):
            cfg.device = dev = torch.device("cuda")
        co = G.DeviceCoerce(dev) if mode in ("gpu64", "gpu32") else None
        try:
            if mode == "xnfdf32":
                from atlas_torch.dfsynthe import pipeline, xnfdf
                stdin, cps = pipeline.parse_xnfdf_com(src / "xnfdf.com")
                wd = BASE / f"xnfdf32_{mix}"
                wd.mkdir(parents=True, exist_ok=True)
                x10, x11 = wd / "xnfpdf.dat", wd / "xnfpdfmax.dat"
                if not x10.exists():
                    with G.mode_context("cpu32"), G.Timer() as tx:
                        xnfdf.run(stdin, continua_text=cps["fort.17"].read_text(),
                                  molecules_text=cps["fort.2"].read_text(), out10=x10, out11=x11)
                    rec["xnfdf_seconds"] = tx.s
                ctx = G.mode_context("exact")
            elif mode in ("exact", "exact_noftz"):
                ctx = G.mode_context("exact")
            elif mode == "native":
                ctx = G.mode_context("native")
            else:
                ctx = G.mode_context({"gpu_lines": "gpu64"}.get(mode, mode), co)
            with ctx, G.Timer(dev) as t:
                res = run_temperature(x10, x11, DATA, it, cfg)
            rec["seconds"] = t.s
            rec["loadavg"] = t.load
            rec["timing"] = {k: round(v, 2) for k, v in res["info"]["timing"].items()}
            rec["metrics"] = compare(mix, it, res)
            rec["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            rec["status"] = "crash"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-4000:]
        if co is not None:
            rec["device_coercions"] = dict(sorted(co.sites.items(), key=lambda kv: -kv[1])[:40])
            rec["device_coercion_sites"] = len(co.sites)
    m = rec.get("metrics", {})
    print(f"odf/{mix} t{it} {mode:9s} {rec['status']} {rec.get('seconds', float('nan')):.1f}s "
          f"max={[m.get(f'odf_v{v}:abs') for v in (0, 1, 2, 4, 8)]} share>1={m.get('odf_v0:share_gt1')} "
          f"worst/gate={m.get('worst_ratio_vs_table_gates')} {rec.get('error', '')[:300]}", flush=True)
    if rec["status"] == "crash":
        print(rec["traceback"])
    G.merge_result(f"dfsynthe/{mix}/t{it}/{mode}", {k: v for k, v in rec.items() if k != "traceback"})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="solar")
    ap.add_argument("--temps", default="33")
    ap.add_argument("--mode", default="exact",
                    choices=["fortran", "exact", "exact_noftz", "native", "cpu32", "gpu_lines", "gpu64", "gpu32", "xnfdf32"])
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    for it in [int(x) for x in a.temps.split(",")]:
        run(a.mix, it, a.mode, a.threads)


if __name__ == "__main__":
    main()
