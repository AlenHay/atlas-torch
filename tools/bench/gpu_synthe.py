"""M7 study, SYNTHE family: xnfpelsyn (KAPP continuum batch), synthe (line opacity), spectrv (radiative transfer).

Stage-isolated, per program: the Fortran chain of a SYNTHE window is built once with the stock executables
(scripts/synthe_workdir.sh: every intermediate fort.* kept). For each mode, each torch program runs on the stock
inputs of that program and the rest of the chain is the stock Fortran, so the spectrum.bin error vs the stock
window (atlas_torch.scoring.synthe_metrics, frozen synthe/sun/w<i> gates) is that program's error alone.
`chain` runs the three torch programs back to back in the mode (Fortran line readers; their torch ports are
byte-identical).

  tools/bench/gpu_run.sh cpu tools/bench/gpu_synthe.py prepare --windows 0,3        # Fortran work dirs + timings
  tools/bench/gpu_run.sh cpu tools/bench/gpu_synthe.py run --mode exact --windows 0   # exact|native|cpu32
  tools/bench/gpu_run.sh gpu tools/bench/gpu_synthe.py run --mode gpu64 --windows 0   # gpu64|gpu32
Results: $ATLAS_EXT/scratch/m7_gpu/synthe/w<i>/<mode>.json, merged into results/m7/gpu_float32_compile.json
under "synthe/w<i>/<mode>". Threads: 1 torch thread (--threads), CORE=<n> pins the process (gpu_run.sh).
"""
from __future__ import annotations

import argparse
import contextlib
import json
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

BIN = G.EXT / "BasicATLAS" / "bin"
S_FILES = G.EXT / "BasicATLAS" / "data" / "synthe_files"
MODEL = G.EXT / "runs" / "stock" / "atlas" / "sun" / "output_synthe.out"
BASE = G.SCRATCH / "synthe"
FORT25 = ("0.0       0.        1.        0.        0.        0.        0.        0.\n0.\n"
          "RHOXJ     R1        R101      PH1       PC1       PSI1      PRDDOP    PRDPOW\n")


def windows():
    ref = json.loads((G.REPO / "reference/reference_set.json").read_text())
    return ref["synthe"]["windows_nm"]


def stock_dir(i):
    return G.EXT / "runs" / "stock" / "atlas" / "sun" / f"window_{i}"


def fdir(i):
    return BASE / f"w{i}" / "fortran"


def _copy_inputs(src: Path, dst: Path, skip=()):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.name not in skip:
            shutil.copyfile(f, dst / f.name)
            os.chmod(dst / f.name, 0o644)


def _run_fortran(exe, wd: Path, stdin=None) -> float:
    t0 = time.perf_counter()
    with open(wd / f"{exe}.out", "w") as out:
        subprocess.run([str(BIN / f"{exe}.exe")], cwd=wd, stdin=open(stdin) if stdin else subprocess.DEVNULL,
                       stdout=out, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - t0


def fortran_synthe(wd: Path, xnfpelsyn_dat: Path) -> float:
    """synthe.exe in wd (pre_synthe inputs already there) with the given fort.10."""
    for n in ("fort.10", "fort.18"):
        (wd / n).unlink(missing_ok=True)
    shutil.copyfile(xnfpelsyn_dat, wd / "fort.10")
    os.symlink(S_FILES / "he1tables.dat", wd / "fort.18")
    return _run_fortran("synthe", wd)


def fortran_spectrv(wd: Path) -> float:
    (wd / "fort.25").write_text(FORT25)
    for n in ("fort.2",):
        (wd / n).unlink(missing_ok=True)
        os.symlink(S_FILES / "molecules.dat", wd / n)
    t = _run_fortran("spectrv", wd, stdin=MODEL)
    shutil.copyfile(wd / "fort.7", wd / "spectrum.bin")
    return t


# ------------------------------------------------------------------------------------------- prepare
def prepare(i: int, repeat: int = 3) -> dict:
    lo, hi = windows()[i]
    wd = fdir(i)
    if not (wd / "spectrum.bin").exists():
        wd.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["bash", str(G.REPO / "scripts/synthe_workdir.sh"), str(BIN), str(MODEL), str(lo), str(hi),
                        str(wd), "600000.", "1.5"], check=True, stdout=subprocess.DEVNULL)
    same = (wd / "spectrum.bin").read_bytes() == (stock_dir(i) / "spectrum.bin").read_bytes()
    tm = {"xnfpelsyn": [], "synthe": [], "spectrv": []}
    for _ in range(repeat):
        t = BASE / f"w{i}" / "ftime"
        _copy_inputs(wd / "pre_synthe", t)
        for n in ("fort.10",):
            (t / n).unlink(missing_ok=True)
        # xnfpelsyn.exe < model: fort.2 molecules, fort.17 continua
        for n, src in (("fort.2", S_FILES / "molecules.dat"), ("fort.17", S_FILES / "continua.dat")):
            (t / n).unlink(missing_ok=True)
            os.symlink(src, t / n)
        tm["xnfpelsyn"].append(_run_fortran("xnfpelsyn", t, stdin=MODEL))
        tm["synthe"].append(fortran_synthe(t, wd / "xnfpelsyn.dat"))
        tm["spectrv"].append(fortran_spectrv(t))
    shutil.rmtree(BASE / f"w{i}" / "ftime", ignore_errors=True)
    info = {"window_nm": [lo, hi], "fortran_spectrum_equals_stock": same,
            "fortran_seconds": {k: float(np.median(v)) for k, v in tm.items()}, "env": G.env_info()}
    G.save_json(BASE / f"w{i}" / "fortran.json", info)
    return info


# ------------------------------------------------------------------------------------------- torch runs
def score(i, spectrum_dir: Path) -> dict:
    from atlas_torch.scoring import synthe_metrics
    m = synthe_metrics(stock_dir(i), spectrum_dir)
    g = G.gate(f"synthe/sun/w{i}", m)
    return {"metrics": {k: v for k, v in m.items() if ":" in k and "rms" not in k}, "gate": g}


@contextlib.contextmanager
def save_get_on_state_device():
    """Device bug (reported, not fixed in src): synthe.opacity._core.save_get(s, ...) creates the zero initial
    value of an absent SAVE array with device=None (CPU) even when the State is on the GPU; put_rows() then keeps
    that CPU tensor and LINTER/HMINOP mix devices. This patch gives save_get the State's device by default
    (identical behaviour for a CPU State)."""
    from atlas_torch.synthe.opacity import _core
    orig = _core.save_get

    def save_get(s, unit, var, shape=(), dtype=torch.float64, device=None):
        if device is None:
            device = next((v.device for v in s.values() if isinstance(v, torch.Tensor)), None)
        return orig(s, unit, var, shape, dtype, device)
    mods = [_core] + G._modules_importing("atlas_torch.synthe", "save_get", orig)
    for m in mods:
        m.save_get = save_get
    try:
        yield
    finally:
        for m in mods:
            m.save_get = orig


def device_physics(dev):
    """xnfpelsyn physics whose KAPP batch runs on `dev` (device helper of this study, not in the package).

    xnfpelsyn(device=...) does not work off the CPU (reported): the READIN/POPS State of TorchPhysics stays on
    the CPU, so KAPP mixes devices (HOP), and the DOPPLE block mixes device layer arrays with CPU atomic masses.
    The program therefore runs on the CPU and only its KAPP batch (all frequencies x layers, the hot part) runs on
    `dev`: the State and the frequency arrays are moved for the call, results come back to the CPU. The opacity
    routines are device-agnostic (Ctx takes the device of its frequencies) apart from save_get (patched).
    POPS calls made inside KAPP (LUKEOP/HOTOP ITEMP caches) run on the device with the State."""
    from atlas_torch.synthe.xnfpelsyn_prog import TorchPhysics
    if torch.device(dev).type == "cpu":
        return None

    class DeviceTorchPhysics(TorchPhysics):
        def kapp(self, *freqargs):
            cpu = self.s
            self.s = type(cpu)({k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in cpu.items()})
            try:
                with save_get_on_state_device():
                    acont, sigmac = super().kapp(*[a.to(dev) for a in freqargs])
            finally:
                self.s = type(cpu)({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in self.s.items()})
            return acont.cpu(), sigmac.cpu()
    return DeviceTorchPhysics()


def run_xnfpelsyn(wd: Path, dev):
    from atlas_torch.synthe.xnfpelsyn_prog import xnfpelsyn
    with G.Timer(dev) as t:
        r = xnfpelsyn(MODEL.read_text(), (S_FILES / "continua.dat").read_text(), S_FILES / "molecules.dat",
                      out10=wd / "xnfpelsyn.dat", physics=device_physics(dev))   # device: KAPP batch only
    return t.s, r


def run_synthe_prog(wd: Path, dev):
    from atlas_torch.synthe.synthe_prog import run_synthe
    for n in ("fort.18",):
        (wd / n).unlink(missing_ok=True)
        os.symlink(S_FILES / "he1tables.dat", wd / n)
    with G.Timer(dev) as t:
        run_synthe(wd, device=dev, stdout_path=wd / "synthe.out", delete_inputs=False)
    return t.s


def run_spectrv(wd: Path, dev, libm):
    from atlas_torch.synthe import spectrv_prog as sp
    from atlas_torch.synthe.readin import readin
    (wd / "fort.25").write_text(FORT25)
    with G.Timer(dev) as t:
        state = readin(MODEL.read_text(), molecules_path=S_FILES / "molecules.dat", mode=20)
        res = sp.spectrv(state, sp.load_inputs(wd), device=dev, libm=libm)
        res.write(wd / "spectrum.bin")
    return t.s


def xnf_compare(a: Path, b: Path) -> dict:
    from atlas_torch.synthe.xnfpelsyn_prog import read_fort10
    x, y = read_fort10(a), read_fort10(b)
    out = {}
    for k in ("continall", "contabs", "contscat"):
        n = x["freqset"].size
        out[f"{k}:abs_dex"] = float(np.max(np.abs(x[k][:, :n] - y[k][:, :n])))
    for k in ("xnfpel", "dopple"):
        a_, b_ = x[k], y[k]
        den = np.maximum(np.abs(a_), np.abs(b_))
        out[f"{k}:rel"] = float(np.max(np.where(den > 0, np.abs(a_ - b_) / np.where(den > 0, den, 1), 0)))
    return out


def run_mode(i: int, mode: str, programs, threads: int, repeat: int = 2) -> dict:
    """Each program `repeat` times (outputs of the last run scored): seconds = fastest run, cold = first run."""
    res = None
    for r in range(repeat):
        cur = _run_mode(i, mode, programs, threads, final=(r == repeat - 1))
        if res is None:
            res = cur
        for p, rec in cur["programs"].items():
            prev = res["programs"][p]
            if "seconds" in rec:
                prev.setdefault("runs_s", []).append(rec["seconds"])
                prev["cold_s"] = prev["runs_s"][0]
                prev["seconds"] = min(prev["runs_s"])
            if r == repeat - 1:
                for k in ("metrics", "gate", "status", "error", "traceback", "xnfpelsyn_vs_fortran"):
                    if k in rec:
                        prev[k] = rec[k]
    G.save_json(BASE / f"w{i}" / f"{mode}.json", res)
    G.merge_result(f"synthe/w{i}/{mode}", {p: {k: v for k, v in r.items() if k != "traceback"}
                                           for p, r in res["programs"].items()})
    return res


def _run_mode(i: int, mode: str, programs, threads: int, final: bool = True) -> dict:
    torch.set_num_threads(threads)
    dev = G.mode_device(mode)
    fw = fdir(i)
    libm = "libmvec" if mode == "exact" else "torch"
    res = {"mode": mode, "window": i, "env": G.env_info(), "programs": {}}
    for prog in programs:
        wd = BASE / f"w{i}" / f"{mode}_{prog}"
        rec = {}
        try:
            if prog == "xnfpelsyn":
                _copy_inputs(fw / "pre_synthe", wd, skip=("fort.10",))
                with G.mode_context(mode):
                    rec["seconds"], _ = run_xnfpelsyn(wd, dev)
                rec["xnfpelsyn_vs_fortran"] = xnf_compare(fw / "xnfpelsyn.dat", wd / "xnfpelsyn.dat")
                fortran_synthe(wd, wd / "xnfpelsyn.dat")
                fortran_spectrv(wd)
            elif prog == "synthe":
                _copy_inputs(fw / "pre_synthe", wd)
                (wd / "fort.10").unlink()
                shutil.copyfile(fw / "xnfpelsyn.dat", wd / "fort.10")
                with G.mode_context(mode):
                    rec["seconds"] = run_synthe_prog(wd, dev)
                fortran_spectrv(wd)
            elif prog == "spectrv":
                _copy_inputs(fw / "pre_spectrv", wd)
                with G.mode_context(mode):
                    rec["seconds"] = run_spectrv(wd, dev, libm)
            elif prog == "chain":
                _copy_inputs(fw / "pre_synthe", wd, skip=("fort.10",))
                with G.mode_context(mode):
                    t1, _ = run_xnfpelsyn(wd, dev)
                    shutil.copyfile(wd / "xnfpelsyn.dat", wd / "fort.10")
                    t2 = run_synthe_prog(wd, dev)
                    t3 = run_spectrv(wd, dev, libm)
                rec["seconds"] = t1 + t2 + t3
                rec["seconds_parts"] = {"xnfpelsyn": t1, "synthe": t2, "spectrv": t3}
            if final:
                rec.update(score(i, wd))
            rec["status"] = "ok"
        except Exception as e:  # noqa: BLE001  (a crash is a result of the study)
            rec["status"] = "crash"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-3000:]
        rec["loadavg"] = round(os.getloadavg()[0], 2)
        res["programs"][prog] = rec
        print(f"w{i} {mode:6s} {prog:9s} {rec['status']:5s} "
              f"{rec.get('seconds', float('nan')):8.2f}s  pass={rec.get('gate', {}).get('pass')} "
              f"worst={rec.get('gate', {}).get('worst')} {rec.get('error', '')[:200]}", flush=True)
        if rec["status"] == "ok" and not os.environ.get("M7_KEEP") and final:
            shutil.rmtree(wd, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run"])
    ap.add_argument("--windows", default="0")
    ap.add_argument("--mode", default="exact", choices=G.MODES)
    ap.add_argument("--programs", default="xnfpelsyn,synthe,spectrv,chain")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    for i in [int(x) for x in a.windows.split(",")]:
        if a.cmd == "prepare":
            print(json.dumps(prepare(i), indent=1))
        else:
            run_mode(i, a.mode, a.programs.split(","), a.threads, a.repeat)


if __name__ == "__main__":
    main()
