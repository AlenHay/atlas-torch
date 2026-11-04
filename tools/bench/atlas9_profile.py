"""Profile / time one solar ATLAS9 model with the torch port (and optionally the stock Fortran on the same core).

Inputs: the traced solar run ($ATLAS_EXT/traces/sun_atlas9/w: kappa.ros, molecules.dat, restart deck, p00big2 ODF,
BasicATLAS control cards with ITERATIONS 15). Run inside the container (scripts/dx), e.g.

    taskset -c 3 .venv/bin/python tools/bench/atlas9_profile.py --iters 15 --json results/bench/atlas9_profile.json
    taskset -c 3 .venv/bin/python tools/bench/atlas9_profile.py --iters 3 --cprofile 40      # top-40 cProfile
    taskset -c 3 .venv/bin/python tools/bench/atlas9_profile.py --fortran                   # stock exe, same core

--check compares the torch stdout / fort.7 with the stock executable's (byte-identical expected; the stock run is
cached in --cache). The JSON holds wall time, per-iteration times, the step breakdown (main.iteration hooks), the
load average and, with --cprofile, the top functions by cumulative and total time.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
_SRC = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--src"), str(REPO / "src"))
sys.path.insert(0, _SRC)
EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
W = EXT / "traces/sun_atlas9/w"
EXE = EXT / "BasicATLAS/bin/atlas9mem.exe"


def stdin_text(iters: int) -> str:
    text = (W / "stdin.txt").read_text()
    head = text[:text.index("ITERATIONS")]
    return (head + "ITERATIONS {} PRINT".format(iters) + " 1" * 15 + "\nPUNCH" + " 1" * 15 + "\n"
            + "BEGIN" + " " * 20 + "ITERATION  10 COMPLETED\n")


def setup(d: Path):
    for u in ("1", "2", "9"):
        os.symlink(os.path.realpath(W / f"fort.{u}"), d / f"fort.{u}")
    shutil.copy(W / "restart.dat", d / "fort.3")


def run_fortran(iters: int, d: Path) -> dict:
    setup(d)
    r0 = os.times()
    t0 = time.perf_counter()
    p = subprocess.run([str(EXE)], input=stdin_text(iters).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       cwd=d, timeout=3600)
    dt = time.perf_counter() - t0
    r1 = os.times()
    return {"seconds": dt, "cpu_seconds": (r1.children_user - r0.children_user) + (r1.children_system - r0.children_system), "stdout": p.stdout.decode("latin-1"), "punch": (d / "fort.7").read_text(),
            "code": p.returncode}


def run_torch(iters: int, d: Path, prof=None, steps=None) -> dict:
    from atlas_torch.atlas9 import main
    from atlas_torch.atlas9.program import float64_default, make_physics
    from atlas_torch.atlas9.cards import StdinReader  # noqa: F401
    setup(d)
    files = {u: d / f"fort.{u}" for u in (1, 2, 3, 9)}
    it_times = []
    last = [time.perf_counter()]

    def on_it(count, s):
        t = time.perf_counter()
        it_times.append(t - last[0])
        last[0] = t

    hook = None
    if steps is not None:
        prev = {"name": None, "t": None}

        def hook(name, s):
            t = time.perf_counter()
            if prev["name"] is not None:
                steps[prev["name"]] = steps.get(prev["name"], 0.0) + t - prev["t"]
            prev["name"], prev["t"] = (None if name == "end" else name), t
    with float64_default():
        phys = make_physics("torch")
        t0, c0 = time.perf_counter(), time.process_time()
        last[0] = t0
        if prof is not None:
            prof.enable()
        r = main.run(stdin_text(iters), files, phys, on_iteration=on_it, hook=hook, keep_states=False)
        if prof is not None:
            prof.disable()
        dt, dc = time.perf_counter() - t0, time.process_time() - c0
    return {"seconds": dt, "cpu_seconds": dc, "iter_seconds": it_times, "stdout": "".join(x + "\n" for x in r.stdout),
            "punch": "".join(x + "\n" for x in r.punch), "exit": r.exit_message}


TIMED = ["atlas_torch.atlas9.physics.molec:nmolec", "atlas_torch.atlas9.physics.molec:_edens_molecular",
         "atlas_torch.atlas9.physics.molec:solvit", "atlas_torch.atlas9.physics.molec:_partition_divide",
         "atlas_torch.atlas9.physics.api:Atlas9Physics.continuum", "atlas_torch.atlas9.physics.api:Atlas9Physics.populations",
         "atlas_torch.atlas9.josh:josh", "atlas_torch.atlas9.josh:_gs_numpy", "atlas_torch.atlas9.ttaup:ttaup",
         "atlas_torch.atlas9.rosstab:Rosstab.__call__", "atlas_torch.atlas9.linop:line_opacity",
         "atlas_torch.atlas9.main:frequency_integration", "atlas_torch.atlas9.main:hydrostatic",
         "atlas_torch.atlas9.tcorr:tcorr3", "atlas_torch.atlas9.tcorr:tcorr_terms", "atlas_torch.atlas9.convec:convec",
         "atlas_torch.atlas9.readin:readin", "atlas_torch.atlas9.putout:mode5", "atlas_torch.atlas9.kapp:kapp_combine",
         "atlas_torch.atlas9.radiap:radiap_terms", "atlas_torch.atlas9.ross:ross_terms"]


def install_timers(acc: dict):
    """Wrap the TIMED functions (every module attribute bound to them) with wall-clock accumulators
    (inclusive time; nested timers overlap)."""
    import functools
    import importlib
    for spec in TIMED:
        mod_name, qual = spec.split(":")
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        owner, attr = mod, qual
        if "." in qual:
            cls, attr = qual.split(".")
            owner = getattr(mod, cls, None)
        orig = getattr(owner, attr, None) if owner is not None else None
        if orig is None:
            continue

        def wrap(f, key):
            @functools.wraps(f)
            def g(*a, **k):
                t0 = time.perf_counter()
                try:
                    return f(*a, **k)
                finally:
                    c = acc.setdefault(key, [0.0, 0])
                    c[0] += time.perf_counter() - t0
                    c[1] += 1
            return g
        w = wrap(orig, qual)
        setattr(owner, attr, w)
        if owner is mod:
            for m in list(sys.modules.values()):
                if getattr(m, "__name__", "").startswith("atlas_torch") and getattr(m, attr, None) is orig:
                    setattr(m, attr, w)


def top(prof, key, n):
    s = io.StringIO()
    st = pstats.Stats(prof, stream=s)
    st.sort_stats(key)
    rows = []
    for func, (cc, nc, tt, ct, callers) in sorted(st.stats.items(), key=lambda kv: -kv[1][3 if key == "cumulative" else 2])[:n]:
        rows.append({"func": "{}:{}({})".format(Path(func[0]).name, func[1], func[2]), "ncalls": nc,
                     "tottime": round(tt, 3), "cumtime": round(ct, 3)})
    return rows


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--cprofile", type=int, default=0, help="report the top N functions (cProfile)")
    ap.add_argument("--steps", action="store_true", help="per-step breakdown through main.iteration's hook")
    ap.add_argument("--timers", action="store_true", help="inclusive wall time of the TIMED functions")
    ap.add_argument("--fortran", action="store_true", help="also time the stock executable")
    ap.add_argument("--check", action="store_true", help="compare with the stock executable (byte-identical)")
    ap.add_argument("--cache", default=str(EXT / "scratch/atlas9_bench"), help="stock output cache dir")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--src", default=None, help="atlas_torch source tree to import (default: this checkout's src)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import atlas_torch
    out = {"label": a.label, "src": str(Path(atlas_torch.__file__).parent), "iters": a.iters, "threads": a.threads, "loadavg_before": os.getloadavg(),
           "affinity": sorted(os.sched_getaffinity(0)), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    prof = cProfile.Profile() if a.cprofile else None
    steps = {} if a.steps else None
    timers = {}
    if a.timers:
        import atlas_torch.atlas9.main  # noqa: F401  (import everything before wrapping)
        import atlas_torch.atlas9.physics  # noqa: F401
        install_timers(timers)
    with tempfile.TemporaryDirectory() as td:
        r = run_torch(a.iters, Path(td), prof, steps)
    out["torch_seconds"] = round(r["seconds"], 3)
    out["torch_cpu_seconds"] = round(r["cpu_seconds"], 3)
    out["torch_iter_seconds"] = [round(x, 3) for x in r["iter_seconds"]]
    out["exit"] = r["exit"]
    if steps is not None:
        out["steps"] = {k: round(v, 3) for k, v in sorted(steps.items(), key=lambda kv: -kv[1])}
    if timers:
        out["timers"] = {k: [round(v[0], 3), v[1]] for k, v in sorted(timers.items(), key=lambda kv: -kv[1][0])}
    if prof is not None:
        out["cprofile_cumulative"] = top(prof, "cumulative", a.cprofile)
        out["cprofile_tottime"] = top(prof, "tottime", a.cprofile)
    if a.fortran or a.check:
        cache = Path(a.cache) / "stock_{}".format(a.iters)
        if a.fortran or not (cache / "stdout.txt").exists():
            with tempfile.TemporaryDirectory() as td:
                f = run_fortran(a.iters, Path(td))
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "stdout.txt").write_text(f["stdout"], encoding="latin-1")
            (cache / "fort7.txt").write_text(f["punch"], encoding="latin-1")
            out["fortran_seconds"] = round(f["seconds"], 3)
            out["fortran_cpu_seconds"] = round(f["cpu_seconds"], 3)
        if a.check:
            so = (cache / "stdout.txt").read_text(encoding="latin-1")
            pu = (cache / "fort7.txt").read_text(encoding="latin-1")
            out["identical_stdout"] = so == r["stdout"]
            out["identical_fort7"] = pu == r["punch"]
    import resource
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    out["loadavg_after"] = os.getloadavg()
    txt = json.dumps(out, indent=1)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(txt + "\n")
    summary = {k: v for k, v in out.items() if not k.startswith("cprofile")}
    print(json.dumps(summary, indent=1))
    if prof is not None:
        print("--- cumulative")
        for row in out["cprofile_cumulative"]:
            print("{cumtime:9.3f} {tottime:9.3f} {ncalls:>9} {func}".format(**row))
        print("--- tottime")
        for row in out["cprofile_tottime"]:
            print("{cumtime:9.3f} {tottime:9.3f} {ncalls:>9} {func}".format(**row))


if __name__ == "__main__":
    main_()
