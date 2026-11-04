"""Several ATLAS9 models at once.

``run_models(jobs, workers)`` runs independent models concurrently, one per worker process (spawned, one torch
thread each). Every model is still the exact single-model computation (byte-identical to the stock program), so
this is throughput batching, not a tensor batch.

    from atlas_torch.atlas9.batch import AtlasJob, run_models
    jobs = [AtlasJob(out_dir, settings, ODF=odf) for ...]          # compat.atlas.atlas() arguments
    results = run_models(jobs, workers=4)                         # [{'output_dir', 'seconds', 'error'}]

Why there is no leading model dimension (yet)
---------------------------------------------
A tensor batch over models (``[M, n]`` layer arrays) is not done in this pass because the per-iteration work is
not uniform across models and the port keeps the Fortran's scalar control flow:
  * the State mirrors the Fortran COMMONs (``"block.var"`` with Fortran shapes); no module has a model axis, and
    READIN/PUTOUT are per-model card interpreters / formatted writers;
  * models iterate a different number of times (GOLD at iteration 15 vs 30 ...) and stop on per-model
    conditions (PUTOUT's GOLD / MODEL DIVERGED / INVALID STRUCTURE, HYDROFAIL);
  * the opacity-sampling early stop of the frequency integration (main.frequency_integration, step 5) is
    replayed per frequency on Python scalars, and its result fixes the order of the TCORR/RADIAP/ROSS sums
    (sequential float64 cumsums) -- per model;
  * the exact elementary functions (physics/libm.py, fmath.py) are glibc calls element by element, whose cost
    scales with the number of elements whatever the batching; most of the remaining cost is Python dispatch on
    72-layer tensors (NMOLEC's per-layer recurrence, the per-frequency continuum sources), which a model axis
    would amortise -- that is the payoff of a later batched version.
Hook for that version: the per-iteration steps are already separate functions of a State
(``main.iteration``: hydrostatic -> physics.electrons/populations -> tcorr1 -> frequency_integration ->
ross3/high/radiap3/convec/tcorr3/turb -> putout.mode5), JOSH already takes ``[B, n]`` problems, and
``Atlas9Physics.continuum`` takes ``[F, n]`` frequency batches; a batched driver would give each of these a
leading model axis (rhox becoming ``[M, n]`` in JOSH/INTEG), keep a per-model "active" mask for models that
stopped, and replay the opacity-sampling control flow per (model, frequency).
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import time
from dataclasses import dataclass, field

__all__ = ["AtlasJob", "run_models"]


@dataclass
class AtlasJob:
    """Arguments of compat.atlas.atlas() for one model."""
    output_dir: str
    settings: object
    restart: str = "auto"
    niter: int = 450
    ODF: str | None = None
    molecules: bool = True
    kwargs: dict = field(default_factory=dict)


_CORES = None


def _init_worker(cores, threads: int):
    """Worker start: one core per worker (taken from a shared queue) and the native helpers loaded once."""
    import torch
    torch.set_num_threads(threads)
    if cores is not None:
        try:
            c = cores.get_nowait()
            os.sched_setaffinity(0, {c})
        except Exception:                 # queue empty or affinity not permitted: run unpinned
            pass
    from . import native
    native.load()


def _one(job: AtlasJob, backend: str, threads: int) -> dict:
    os.environ["ATLAS_TORCH_BACKEND"] = backend
    from atlas_torch.compat import atlas as api
    t0 = time.perf_counter()
    kw = dict(restart=job.restart, niter=job.niter, molecules=job.molecules, silent=True, **job.kwargs)
    if job.ODF is not None:
        kw["ODF"] = job.ODF
    try:
        api.atlas(job.output_dir, job.settings, **kw)
        err = None
    except Exception as e:                       # reported per model; the other models carry on
        err = "{}: {}".format(type(e).__name__, e)
    return {"output_dir": job.output_dir, "seconds": time.perf_counter() - t0, "error": err}


def run_models(jobs: list[AtlasJob], workers: int | None = None, backend: str = "torch", threads: int = 1,
               pin: bool = True) -> list[dict]:
    """Run ``jobs`` concurrently (``workers`` processes, default: half the CPUs, at most len(jobs)); results in
    job order: {'output_dir', 'seconds', 'error' (None or message)}. pin=True gives every worker its own core
    (the first `workers` cores of this process's affinity set, one per physical core first when the set allows).
    Memory: ~0.5 GB per worker (one model each)."""
    if not jobs:
        return []
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) // 2))
    ctx = multiprocessing.get_context("spawn")
    cores = None
    if pin:
        avail = sorted(os.sched_getaffinity(0))
        cores = ctx.Queue()
        for c in _spread(avail)[:workers]:
            cores.put(c)
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker,
                                                initargs=(cores, threads)) as ex:
        futs = [ex.submit(_one, j, backend, threads) for j in jobs]
        return [f.result() for f in futs]


def _spread(cores):
    """Order cores so that SMT siblings come last (one worker per physical core first)."""
    first, later, seen = [], [], set()
    for c in cores:
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
        except OSError:
            sib = str(c)
        (later if sib in seen else first).append(c)
        seen.add(sib)
    return first + later
