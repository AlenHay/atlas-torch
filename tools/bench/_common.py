"""Shared helpers of the tools/bench scripts: core pinning, thread count, environment record."""
from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
REPO = Path(__file__).resolve().parents[2]

# window -> (stock step-by-step workdir, wlbeg, wlend, model deck it was made from)
WINDOWS = {
    "500_510": (EXT / "work" / "sun_500_510_stock", 500.0, 510.0, EXT / "scratch_sun" / "output_synthe.out"),
    "380_420": (EXT / "work" / "sun_380_420_stock", 380.0, 420.0, EXT / "scratch_sun" / "output_synthe.out"),
    "100_2850": (EXT / "scratch" / "wide" / "stock_100_2850", 100.0, 2850.0,
                 EXT / "runs" / "stock" / "atlas" / "sun" / "output_synthe.out"),
}


def deck_for(window: str) -> Path:
    return WINDOWS[window][3]


def workdir_for(window: str) -> Path:
    """The stock step-by-step SYNTHE workdir (scripts/synthe_workdir.sh) of a window: pre_synthe/, pre_spectrv/,
    spectrum.bin, ..."""
    return WINDOWS[window][0]


def pin(core: int | None, threads: int | None) -> None:
    """Pin this process to one core (before torch spins up its pool) and fix the torch/OpenMP thread count."""
    if threads:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = str(threads)
    if core is not None:
        os.sched_setaffinity(0, {core})
    if threads:
        import torch
        torch.set_num_threads(threads)


def bench_env(core=None, threads=None) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                             text=True).strip())
    except Exception:  # noqa: BLE001
        commit, dirty = "?", None
    import torch
    return dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), loadavg=list(os.getloadavg()), core=core,
                threads=torch.get_num_threads(), host=platform.node(), commit=commit, dirty=dirty,
                torch=torch.__version__, native_disabled=bool(os.environ.get("ATLAS_TORCH_NO_NATIVE")))


def peak_rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_fortran(exe, cwd, core=None, stdin=None, stdout=None, args=()) -> tuple[float, int]:
    """Run a Fortran executable (pinned to `core`) under /usr/bin/time; returns (wall s, peak RSS kB)."""
    import tempfile
    pre = ["taskset", "-c", str(core)] if core is not None else []
    with tempfile.NamedTemporaryFile("r", suffix=".time") as tf:
        fi = open(stdin) if stdin else subprocess.DEVNULL
        fo = open(stdout, "w") if stdout else subprocess.DEVNULL
        try:
            p = subprocess.run(pre + ["/usr/bin/time", "-o", tf.name, "-f", "%e %M", str(exe), *map(str, args)],
                               stdin=fi, stdout=fo, stderr=subprocess.DEVNULL, cwd=cwd)
        finally:
            if stdin:
                fi.close()
            if stdout:
                fo.close()
        if p.returncode != 0:
            raise RuntimeError(f"{exe} failed with status {p.returncode} in {cwd}")
        el, kb = open(tf.name).read().split()[-2:]
    return float(el), int(kb)


def hwm_reset() -> bool:
    """Reset this process's peak RSS (VmHWM) so that hwm_mb() measures from here on (Linux)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def hwm_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024.0
    return peak_rss_mb()


def same_file(a, b, block: int = 1 << 24) -> bool:
    a, b = Path(a), Path(b)
    if not a.exists() or not b.exists() or a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(block), fb.read(block)
            if x != y:
                return False
            if not x:
                return True
