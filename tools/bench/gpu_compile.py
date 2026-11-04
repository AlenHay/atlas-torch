"""M7 study: torch.compile on the hot kernels (CPU inductor and GPU inductor/triton) -- speed and output identity.

1. capture (CPU env): real inputs of each kernel are recorded from short real runs (native torch math, the mode in
   which these kernels can be compiled at all: the exact path calls glibc/libmvec through numpy/ctypes, which are
   graph breaks), saved to $ATLAS_EXT/scratch/m7_gpu/compile/<kernel>.pt:
     spectrv_josh    synthe.josh.josh_batch       SPECTRV radiative transfer, 8192 points x 2 rows (w0)
     atlas9_josh     atlas9.josh.josh             ATLAS9 JOSH, all frequencies of iteration 1 (sun)
     synthe_voigt    synthe.lineopac.voigt.voigt  SYNTHE Voigt profiles of the largest batch (w0)
     dfsynthe_chunk  LineAccumulator._chunk       DFSYNTHE torch line path, first 50k-line chunk (12589 K)
2. bench (CPU or GPU env): eager vs torch.compile(fn) (default inductor, dynamic=False) on the device: median
   time, first-call (compile) time, and identity of every output tensor vs eager on the same device (bitwise,
   max |diff|, max rel diff), plus GPU-eager vs CPU-eager.

  tools/bench/gpu_run.sh cpu tools/bench/gpu_synthe.py prepare --windows 0     # Fortran w0 work dir (needed once)
  tools/bench/gpu_run.sh cpu tools/bench/gpu_compile.py capture
  tools/bench/gpu_run.sh cpu tools/bench/gpu_compile.py bench --device cpu
  tools/bench/gpu_run.sh gpu tools/bench/gpu_compile.py bench --device cuda
Results: results/m7/gpu_float32_compile.json["compile/<kernel>/<device>"].
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_common as G  # noqa: E402

import torch  # noqa: E402

BASE = G.SCRATCH / "compile"
KERNELS = ("spectrv_josh", "atlas9_josh", "synthe_voigt", "dfsynthe_chunk")


# ------------------------------------------------------------------------------------------- capture
class Stop(Exception):
    def __str__(self):
        return "M7_STOP"


def _cpu(o):
    return G._map(o, lambda t: t.detach().cpu().clone())


def _patch_capture(mods, name, key, pick=lambda a, k: True, stop_after=None, method_of=None):
    """Replace `name` in each module of `mods` (or the method of class `method_of`) by a recorder; saves the first
    call accepted by pick() (or the largest by first-arg numel when pick returns a size)."""
    best = {"size": -1, "n": 0}
    orig = getattr(method_of, name) if method_of is not None else getattr(mods[0], name)

    def rec(*a, **k):
        p = pick(a, k)
        if p is not False and (p is True and best["size"] < 0 or (p is not True and p > best["size"])):
            payload = {"args": _cpu(list(a)), "kwargs": _cpu(dict(k))}
            if method_of is not None:
                payload = {"self": copy.deepcopy(a[0]), "args": _cpu(list(a[1:])), "kwargs": _cpu(dict(k))}
            best["size"] = 0 if p is True else p
            best["payload"] = payload
        best["n"] += 1
        r = orig(*a, **k)
        if stop_after is not None and best["n"] >= stop_after:
            BASE.mkdir(parents=True, exist_ok=True)
            torch.save(best["payload"], BASE / f"{key}.pt")
            raise Stop()
        return r
    if method_of is not None:
        setattr(method_of, name, rec)
    else:
        for m in mods:
            setattr(m, name, rec)

    def done():
        if "payload" in best and not (BASE / f"{key}.pt").exists():
            BASE.mkdir(parents=True, exist_ok=True)
            torch.save(best["payload"], BASE / f"{key}.pt")
        if method_of is not None:
            setattr(method_of, name, orig)
        else:
            for m in mods:
                setattr(m, name, orig)
        return best["n"]
    return done


def capture(which):
    torch.set_num_threads(1)
    import shutil
    import gpu_synthe as S
    BASE.mkdir(parents=True, exist_ok=True)
    if "spectrv_josh" in which:
        from atlas_torch.synthe import josh, spectrv_prog
        done = _patch_capture([josh, spectrv_prog], "josh_batch", "spectrv_josh", stop_after=1)
        wd = S.BASE / "w0" / "capture_spectrv"
        S._copy_inputs(S.fdir(0) / "pre_spectrv", wd)
        try:
            with G.native_math():
                S.run_spectrv(wd, torch.device("cpu"), "torch")
        except Exception as e:  # noqa: BLE001
            if "M7_STOP" not in str(e):
                raise
        print("spectrv_josh calls", done())
        shutil.rmtree(wd, ignore_errors=True)
    if "synthe_voigt" in which:
        from atlas_torch.synthe import synthe_prog
        from atlas_torch.synthe.lineopac import voigt
        done = _patch_capture([voigt, synthe_prog], "voigt", "synthe_voigt", pick=lambda a, k: int(a[0].numel()))
        wd = S.BASE / "w0" / "capture_synthe"
        S._copy_inputs(S.fdir(0) / "pre_synthe", wd)
        (wd / "fort.10").unlink()
        shutil.copyfile(S.fdir(0) / "xnfpelsyn.dat", wd / "fort.10")
        with G.native_math():
            S.run_synthe_prog(wd, torch.device("cpu"))
        print("synthe_voigt calls", done())
        shutil.rmtree(wd, ignore_errors=True)
    if "atlas9_josh" in which:
        import gpu_atlas9 as A
        from atlas_torch.atlas9 import josh as aj
        from atlas_torch.atlas9 import main
        done = _patch_capture([aj, main], "josh", "atlas9_josh", stop_after=1)
        A.run("sun", "native", 1, max_iter=None, record=False)          # stops (M7_STOP) at the first JOSH call
        print("atlas9_josh calls", done())
    if "dfsynthe_chunk" in which:
        os.environ["ATLAS_TORCH_NO_NATIVE"] = "1"
        from atlas_torch.dfsynthe.dfsynthe import Config, run_temperature
        from atlas_torch.dfsynthe.lines.accumulate import LineAccumulator
        done = _patch_capture(None, "_chunk", "dfsynthe_chunk", stop_after=1, method_of=LineAccumulator)
        src = G.EXT / "runs/stock/odf/solar"
        try:
            run_temperature(src / "xnfpdf.dat", src / "xnfpdfmax.dat", G.EXT / "BasicATLAS/data/dfsynthe_files",
                            33, Config())
        except Exception as e:  # noqa: BLE001
            if "M7_STOP" not in str(e):
                raise
        print("dfsynthe_chunk calls", done())


# ------------------------------------------------------------------------------------------- bench
def _load(key, dev):
    p = torch.load(BASE / f"{key}.pt", weights_only=False)
    mv = lambda t: t.to(dev)  # noqa: E731
    p["args"] = G._map(p["args"], mv)
    p["kwargs"] = G._map(p["kwargs"], mv)
    return p


def make_fn(key, dev):
    """(fn() -> outputs, kernel callable to compile, its args/kwargs)."""
    p = _load(key, dev)
    if key == "spectrv_josh":
        from atlas_torch.synthe.josh import josh_batch as k
    elif key == "atlas9_josh":
        from atlas_torch.atlas9.josh import josh as k
    elif key == "synthe_voigt":
        from atlas_torch.synthe.lineopac.voigt import voigt as k
    elif key == "dfsynthe_chunk":
        acc0 = p["self"]
        # move the accumulator to the device (its tables/layer arrays are tensor attributes)
        for n, v in list(vars(acc0).items()):
            if isinstance(v, torch.Tensor):
                setattr(acc0, n, v.to(dev))
            elif isinstance(v, dict):
                setattr(acc0, n, G._map(v, lambda t: t.to(dev)))
        acc0.dev = torch.device(dev)
        carry = acc0.aa_carry.clone()
        buf0 = p["args"][0]

        def k(buf, t, stats):
            acc0.aa_carry = carry.clone()
            b = buf.clone()
            acc0._chunk(b, t, dict(stats))
            return b, acc0.aa_carry
        p["args"] = [buf0] + p["args"][1:]
    return k, p["args"], p["kwargs"]


def _outs(o):
    flat = []
    G._map(o, lambda t: flat.append(t) or t)
    return [t for t in flat if isinstance(t, torch.Tensor)]


def compare(a, b) -> dict:
    fa, fb = _outs(a), _outs(b)
    if len(fa) != len(fb):
        return {"identical": False, "note": f"{len(fa)} vs {len(fb)} outputs"}
    same, mabs, mrel = True, 0.0, 0.0
    for x, y in zip(fa, fb):
        x, y = x.detach().cpu(), y.detach().cpu()
        if x.shape != y.shape:
            return {"identical": False, "note": f"shape {tuple(x.shape)} vs {tuple(y.shape)}"}
        if x.is_floating_point():
            eq = torch.equal(torch.nan_to_num(x, nan=1.2345e300), torch.nan_to_num(y, nan=1.2345e300))
            if not eq:
                same = False
                d = (x.double() - y.double()).abs()
                fin = torch.isfinite(d)
                if fin.any():
                    mabs = max(mabs, float(d[fin].max()))
                    den = torch.maximum(x.double().abs(), y.double().abs())[fin]
                    r = d[fin] / torch.where(den > 0, den, torch.ones_like(den))
                    mrel = max(mrel, float(r.max()))
        else:
            same = same and torch.equal(x, y)
    return {"identical": same, "max_abs": mabs, "max_rel": mrel}


def bench_one(key, dev, repeat, threads):
    torch.set_num_threads(threads)
    rec = {"device": str(dev), "env": G.env_info()}
    try:
        k, args, kwargs = make_fn(key, dev)
        with G.native_math():
            run_e = lambda: k(*args, **kwargs)  # noqa: E731
            out_e = run_e()
            rec["eager"] = G.bench(run_e, dev, repeat=repeat)
            if dev.type != "cpu":
                kc, ac, kwc = make_fn(key, torch.device("cpu"))
                rec["gpu_vs_cpu_eager"] = compare(out_e, kc(*ac, **kwc))
            torch._dynamo.reset()
            kcmp = torch.compile(k, dynamic=False)
            t0 = time.perf_counter()
            out_c = kcmp(*args, **kwargs)
            G.sync(dev)
            rec["compile_first_call_s"] = time.perf_counter() - t0
            rec["compiled"] = G.bench(lambda: kcmp(*args, **kwargs), dev, repeat=repeat)
            rec["compiled_vs_eager"] = compare(out_c, out_e)
            rec["speedup"] = rec["eager"]["median_s"] / rec["compiled"]["median_s"]
            try:
                import torch._dynamo.utils as du
                rec["graph_breaks"] = int(sum(v for k_, v in du.counters["graph_break"].items()))
                rec["graph_break_reasons"] = [str(k_)[:200] for k_ in list(du.counters["graph_break"])[:8]]
            except Exception:  # noqa: BLE001
                pass
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "crash"
        rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        rec["traceback"] = traceback.format_exc()[-3000:]
    print(f"{key:15s} {str(dev):5s} {rec['status']} eager={rec.get('eager', {}).get('median_s')} "
          f"compiled={rec.get('compiled', {}).get('median_s')} x{rec.get('speedup')} "
          f"first={rec.get('compile_first_call_s')} same={rec.get('compiled_vs_eager')} "
          f"gpu_vs_cpu={rec.get('gpu_vs_cpu_eager')} breaks={rec.get('graph_breaks')} {rec.get('error', '')}",
          flush=True)
    G.merge_result(f"compile/{key}/{dev.type}", {k_: v for k_, v in rec.items() if k_ != "traceback"})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["capture", "bench"])
    ap.add_argument("--kernels", default=",".join(KERNELS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    ks = a.kernels.split(",")
    if a.cmd == "capture":
        capture(ks)
    else:
        for k in ks:
            bench_one(k, torch.device(a.device), a.repeat, a.threads)


if __name__ == "__main__":
    main()
