"""Whole DFSYNTHE chain in torch: XNFDF -> DFSYNTHE (57 temperatures) -> DFSORTP -> SEPARATEDF -> KAPPA9 +
KAPREADTS, producing the final products of BasicATLAS ``dfsynthe()``:

    xnfpdf.dat, xnfpdfmax.dat, p00big{v}.bdf, p00lit{v}.bdf (v = 0, 1, 2, 4, 8), kappa.ros

Usage (the compat API, or directly)::

    from atlas_torch.dfsynthe.pipeline import run_odf
    info = run_odf("/path/to/odf_dir", xnfdf_stdin=stdin_text)          # or xnfdf_com="xnfdf.com"
    info = run_odf(out, xnfdf_com=".../xnfdf.com", workers=6, kappa=True)

* XNFDF and DFSYNTHE run in the stock floating-point environment (flush-to-zero/denormals-are-zero, ftz.py) and
  with the stock ifx -fp-model fast semantics of HPROF4's Lyman-alpha term (Flavor('stock'), fastmath=True)
  unless ``ftz=False`` / ``fastmath=False`` (then they reproduce the ifx -O0 / -fp-model precise builds).
* Each temperature is an independent DFSYNTHE run (as BasicATLAS runs them) in a worker process (spawn, one
  torch thread each, FTZ scoped inside); finished temperatures are kept in ``<out>/.odf_parts/t<NN>.npz`` so an
  interrupted run resumes.  DFSORTP/SEPARATEDF are done in memory (byte-identical files).
* KAPPA9+KAPREADTS: atlas_torch.kappa9.kappa9_prog.run_workdir (if importable) with BasicATLAS's control cards
  (built from the XNFDF abundance cards, or taken from ``kappa9_controls`` / ``<out>/kappa9v{v}.com``).
* Not produced: the Fortran printouts (xnfdf.out, dfp00t*.out, kapm40k2.out).  compat.meta() identifies an ODF
  directory by xnfdf.out (abundances parsed from the XNFDF printout), which this port does not write.
"""
from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import odf_io
from .separatedf import separatedf_arrays

VTURBS = odf_io.VTURBS
NTEMP = len(odf_io.DFTS)


def _ext() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def default_data_dir() -> Path:
    try:
        from ..compat.settings import basicatlas_dir
        return Path(basicatlas_dir()) / "data" / "dfsynthe_files"
    except Exception:
        return _ext() / "BasicATLAS" / "data" / "dfsynthe_files"


def parse_xnfdf_com(com: str | Path) -> tuple[str, dict]:
    """(stdin heredoc, {fort.N name: source path}) of a BasicATLAS xnfdf.com command file."""
    text = Path(com).read_text()
    stdin = text.split("<<EOF>xnfdf.out\n", 1)[1].split("\nEOF", 1)[0] + "\n"
    cps = {Path(d).name: Path(s) for s, d in re.findall(r"^cp\s+(\S+)\s+(fort\.\d+)\s*$", text, re.M)}
    return stdin, cps


def abundance_cards(xnfdf_stdin: str) -> str:
    """The ABUNDANCE SCALE / ABUNDANCE CHANGE cards of the first model of an XNFDF deck."""
    out, started = [], False
    for ln in xnfdf_stdin.splitlines():
        if ln.startswith("ABUNDANCE SCALE"):
            if started:
                break
            started = True
        elif started and not ln.startswith(" ABUNDANCE CHANGE"):
            break
        if started:
            out.append(ln)
    if not out:
        raise ValueError("no ABUNDANCE cards in the XNFDF deck")
    return "\n".join(out)


def kappa9_controls_from_xnfdf(xnfdf_stdin: str) -> dict:
    """BasicATLAS's KAPPA9 control cards (atlas.py _kappa9_script) for each microturbulence."""
    from ..compat import atlas as A
    head = "\n".join(["MOLECULES ON", "READ MOLECULES", "FREQUENCIES 337 1 337 BIG", "ITERATIONS 1 PRINT 1 PUNCH 0 ",
                      "TITLE ROSSELAND OPACITY", A._OPACITY_CARD, A._CONVECTION_CARD,
                      "TEFF   5777.  GRAVITY 4.43770 LTE", abundance_cards(xnfdf_stdin)]) + "\n"
    return {v: head + A._kappa9_deck() + "VTURB {}.0E5\n{}\nEND\n".format(v, A._BEGIN_15) for v in VTURBS}


# ------------------------------------------------------------------------------------------ workers
def _worker_init(parent: int):
    """Pool initializer: exit when the parent dies (no orphaned 2-3 GB workers after a kill)."""
    import threading

    def watch():
        while True:
            if os.getppid() != parent:
                os._exit(3)
            time.sleep(5)
    threading.Thread(target=watch, daemon=True).start()


def _temperature_job(args):
    """Worker: one DFSYNTHE temperature -> parts file.  Runs in a fresh (spawned) process."""
    xnfpdf, xnfpdfmax, data_dir, it, cfgd, part = args
    import resource
    import torch
    torch.set_num_threads(1)
    from .dfsynthe import Config, run_temperature
    from .lines.f32 import Flavor
    cfg = Config(**{k: v for k, v in cfgd.items() if k not in ("flavor", "device", "fastmath", "lines_device")})
    cfg.flavor = Flavor(cfgd.get("flavor", "stock"), fastmath=cfgd.get("fastmath"))
    cfg.device = torch.device(cfgd.get("device", "cpu"))
    if cfgd.get("lines_device"):
        cfg.lines_device = torch.device(cfgd["lines_device"])
    t0 = time.time()
    res = run_temperature(xnfpdf, xnfpdfmax, data_dir, it, cfg)
    info = res["info"]
    tmp = Path(str(part) + ".tmp.npz")
    np.savez(tmp, **{f"v{v}": res[v][1] for v in VTURBS}, **{f"idf{v}": res[v][0] for v in VTURBS},
             wall=time.time() - t0, timing=np.array([info["timing"].get(k, 0.0) for k in
                                                     ("selectlines", "continuum", "xlinop", "lines",
                                                      "broaden+dfcalc")]))
    os.replace(tmp, part)
    return it, time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def default_workers(n_jobs: int) -> int:
    """Bounded by CPUs (half of them: SMT siblings) and by memory (~2.5 GB per DFSYNTHE worker)."""
    cpus = os.cpu_count() or 1
    try:
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    avail = int(ln.split()[1]) * 1024
    except (OSError, ValueError):
        avail = 8 << 30
    by_mem = max(1, int(avail // (2.5 * (1 << 30))))
    return max(1, min(n_jobs, max(1, cpus // 2), by_mem))


def run_temperatures(xnfpdf, xnfpdfmax, data_dir, parts_dir, temps, cfgd, workers, log=print):
    """Run the missing temperatures (1-based indices) into parts_dir/t<NN>.npz with a process pool."""
    import concurrent.futures as cf
    import multiprocessing as mp
    from concurrent.futures.process import BrokenProcessPool
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    walls = {}

    def done(it, w, rss):
        walls[it] = w
        log(f"  T index {it} ({odf_io.DFTS[it - 1]} K) done in {w:.0f} s, peak RSS {rss:.2f} GB")

    breaks = 0
    for attempt in range(6):
        todo = [it for it in temps if not (parts_dir / f"t{it:02d}.npz").exists()]
        if not todo:
            break
        jobs = [(str(xnfpdf), str(xnfpdfmax), str(data_dir), it, cfgd, str(parts_dir / f"t{it:02d}.npz"))
                for it in sorted(todo)]                      # coolest (most expensive) first
        if workers <= 1:
            for j in jobs:
                done(*_temperature_job(j))
            continue
        try:
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                        initializer=_worker_init, initargs=(os.getpid(),)) as pool:
                futs = [pool.submit(_temperature_job, j) for j in jobs]
                for f in cf.as_completed(futs):
                    done(*f.result())
        except BrokenProcessPool as e:          # e.g. a worker killed by the (early)OOM killer
            breaks += 1
            if breaks >= 2:                     # shrink the pool only when it keeps happening
                workers = max(1, workers - 1)
                breaks = 0
            log(f"  worker pool broken ({e}); retrying the missing temperatures with {workers} worker(s) "
                f"(attempt {attempt + 2})")
    missing = [it for it in temps if not (parts_dir / f"t{it:02d}.npz").exists()]
    if missing:
        raise RuntimeError(f"DFSYNTHE temperatures not completed: {missing}")
    return walls


def load_steps(parts_dir, temps=range(1, NTEMP + 1)) -> dict:
    """{vt: int16 [57, 1540, 25, 12]} (it, nu, ip, step) = the DFSORTP order SEPARATEDF reads."""
    out = {v: np.zeros((NTEMP, odf_io.NFREQ, odf_io.KW, 12), dtype=np.int16) for v in VTURBS}
    for it in temps:
        z = np.load(Path(parts_dir) / f"t{it:02d}.npz")
        for v in VTURBS:
            idf = z[f"idf{v}"]
            j = np.arange(1, odf_io.KW + 1)
            idf_ref = (v * 100000000 + np.arange(1, odf_io.NFREQ + 1)[None, :] * 10000 + it * 100 + j[:, None]).reshape(-1)
            if not np.array_equal(idf, idf_ref.astype(np.int32)):
                raise ValueError(f"parts file of temperature {it} has unexpected IDFOUT")
            out[v][it - 1] = z[f"v{v}"].reshape(odf_io.KW, odf_io.NFREQ, 12).transpose(1, 0, 2)
    return out


def write_bdf(steps: dict, outdir, vturbs=VTURBS) -> None:
    for v in vturbs:
        big, lit = separatedf_arrays(steps[v])
        odf_io.write_seq_records(Path(outdir) / f"p00big{v}.bdf", big)
        odf_io.write_seq_records(Path(outdir) / f"p00lit{v}.bdf", lit)


# ------------------------------------------------------------------------------------------ main entry
def run_odf(output_dir, xnfdf_stdin: str | None = None, *, xnfdf_com=None, continua_text: str | None = None,
            molecules_text: str | None = None, data_dir=None, vturbs=VTURBS, temps=None, workers: int | None = None,
            ftz: bool = True, fastmath: bool = True, kappa: bool = True, kappa9_controls: dict | None = None,
            keep_parts: bool = False, device: str = "cpu", lines_device: str | None = None, log=print,
            **cfg_overrides) -> dict:
    """Run the DFSYNTHE chain into output_dir (created if needed; existing parts are reused).

    xnfdf_stdin: the XNFDF control deck (what BasicATLAS feeds xnfdf.exe), or xnfdf_com: a BasicATLAS xnfdf.com
    (its heredoc and the molecules.dat/continua.dat it copies are used).  data_dir: BasicATLAS
    data/dfsynthe_files (line lists).  temps: 1-based temperature indices (default all 57; the .bdf files and
    kappa.ros need all 57).  vturbs: microturbulences whose .bdf files are written (kappa.ros needs all five).
    workers: parallel DFSYNTHE processes (default: bounded by CPUs/2 and available memory).  lines_device:
    optional accelerator (e.g. 'cuda') for the TAPE12 line accumulation only (rounding-level, not default).
    Returns {'files': [...], 'timing': {...}, 'walls': {it: s}, 'gaps': [...]}."""
    from .xnfdf import default_inputs, run as xnfdf_run
    t00 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    if xnfdf_stdin is None:
        if xnfdf_com is None:
            raise ValueError("need xnfdf_stdin or xnfdf_com")
        xnfdf_stdin, cps = parse_xnfdf_com(xnfdf_com)
        if continua_text is None and "fort.17" in cps:
            continua_text = cps["fort.17"].read_text()
        if molecules_text is None and "fort.2" in cps:
            molecules_text = cps["fort.2"].read_text()
    if continua_text is None or molecules_text is None:
        d = default_inputs()
        continua_text = continua_text or d["continua_text"]
        molecules_text = molecules_text or d["molecules_text"]
    timing, gaps, files = {}, [], []
    x10, x11 = out / "xnfpdf.dat", out / "xnfpdfmax.dat"
    if not (x10.exists() and x11.exists()):
        t0 = time.time()
        xnfdf_run(xnfdf_stdin, continua_text=continua_text, molecules_text=molecules_text,
                  out10=x10, out11=x11, ftz=ftz)
        timing["xnfdf"] = time.time() - t0
        log(f"XNFDF done in {timing['xnfdf']:.1f} s")
    files += [x10, x11]
    temps = list(range(1, NTEMP + 1)) if temps is None else sorted(int(t) for t in temps)
    parts = out / ".odf_parts"
    todo = [it for it in temps if not (parts / f"t{it:02d}.npz").exists()]
    workers = default_workers(len(todo)) if workers is None else max(1, int(workers))
    cfgd = dict(flavor="stock", fastmath=fastmath, ftz=ftz, device=device, lines_device=lines_device, **cfg_overrides)
    t0 = time.time()
    log(f"DFSYNTHE: {len(todo)} temperature(s) to run with {workers} worker(s)")
    walls = run_temperatures(x10, x11, data_dir, parts, temps, cfgd, workers, log)
    timing["dfsynthe"] = time.time() - t0
    if len(temps) == NTEMP:
        t0 = time.time()
        steps = load_steps(parts)
        write_bdf(steps, out, vturbs)
        files += [out / f"p00{p}{v}.bdf" for v in vturbs for p in ("big", "lit")]
        timing["separatedf"] = time.time() - t0
        if kappa:
            if tuple(sorted(vturbs)) != tuple(sorted(VTURBS)):
                gaps.append("kappa.ros needs all five microturbulences")
            else:
                try:
                    from ..kappa9 import kappa9_prog
                except Exception as e:                      # KAPPA9 port not available
                    gaps.append(f"KAPPA9 port not importable ({e!r}): kappa.ros not written")
                    kappa9_prog = None
                if kappa9_prog is not None:
                    t0 = time.time()
                    wd = out / ".kappa9"
                    wd.mkdir(exist_ok=True)
                    if kappa9_controls is None and not all((out / f"kappa9v{v}.com").exists() for v in VTURBS):
                        kappa9_controls = kappa9_controls_from_xnfdf(xnfdf_stdin)
                    if kappa9_controls is not None:
                        for v in VTURBS:
                            (wd / f"stdin_v{v}.txt").write_text(kappa9_controls[v])
                    kinfo = kappa9_prog.run_workdir(out, wd, molecules=_molecules_path(xnfdf_com, data_dir))
                    shutil.copy(wd / "kappa.ros", out / "kappa.ros")
                    files.append(out / "kappa.ros")
                    timing["kappa9"] = time.time() - t0
                    if not keep_parts:
                        shutil.rmtree(wd, ignore_errors=True)
        if not keep_parts:
            shutil.rmtree(parts, ignore_errors=True)
    else:
        gaps.append("not all 57 temperatures requested: no .bdf / kappa.ros (parts kept in .odf_parts)")
    timing["total"] = time.time() - t00
    return dict(files=[str(f) for f in files], timing=timing, walls=walls, gaps=gaps, workers=workers)


def _molecules_path(xnfdf_com, data_dir):
    if xnfdf_com is not None:
        _, cps = parse_xnfdf_com(xnfdf_com)
        if "fort.2" in cps and cps["fort.2"].exists():
            return cps["fort.2"]
    return Path(data_dir) / "molecules.dat"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="torch DFSYNTHE chain (XNFDF..KAPREADTS)")
    ap.add_argument("out")
    ap.add_argument("--xnfdf-com", required=True, help="BasicATLAS xnfdf.com (heredoc + input files)")
    ap.add_argument("--data", default=None, help="BasicATLAS data/dfsynthe_files")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--temps", default="all")
    ap.add_argument("--no-ftz", action="store_true")
    ap.add_argument("--no-fastmath", action="store_true")
    ap.add_argument("--no-kappa", action="store_true")
    ap.add_argument("--keep-parts", action="store_true")
    ap.add_argument("--lines-device", default=None)
    a = ap.parse_args(argv)
    temps = None if a.temps == "all" else [int(x) for x in a.temps.split(",")]
    info = run_odf(a.out, xnfdf_com=a.xnfdf_com, data_dir=a.data, workers=a.workers, temps=temps, ftz=not a.no_ftz,
                   fastmath=not a.no_fastmath, kappa=not a.no_kappa, keep_parts=a.keep_parts,
                   lines_device=a.lines_device,
                   log=lambda s: print(s, flush=True))
    print(info)


if __name__ == "__main__":
    main()
