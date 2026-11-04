"""Benchmark: the whole torch SYNTHE chain per program (xnfpelsyn, synbeg, rgfalllinesnew, rmolecasc x21, rh2ofast,
synthe, spectrv) as atlas_torch.synthe.chain.run_synthe_chain runs it, optionally vs the stock Fortran chain
(scripts/synthe_workdir.sh with /usr/bin/time wrappers) on the same core.

    .venv/bin/python tools/bench/synthe_chain.py --window 500_510 [--core 3] [--threads 1] [--fortran]
                     [--json out.json] [--keep]

--window: 500_510 | 380_420 (scratch_sun deck, as the work/ reference runs) | 100_2850 (stock solar deck, the
BasicATLAS wide batch: ~20 min of Fortran SYNTHE, ~2 GB of fort.12; the stock chain is NOT re-run for it unless
--fortran is given: earlier stock timings are in $ATLAS_EXT/scratch/wide/stock_100_2850.time).
Per program: wall seconds and peak RSS (VmHWM reset before each step; heap kept by earlier steps of the same
process counts too); spectrum.bin compared byte-for-byte with the stock spectrum.bin of the window.  NB: for 100_2850
(stock solar deck) the torch XNFPELSYN physics agrees with the Fortran to <= 1e-13 relative but not bitwise, so the
end-to-end spectrum is not byte-identical there by design; synthe_synthe.py / synthe_spectrv.py byte-check those
programs on the stock stage-isolated inputs.  Work files go to $ATLAS_EXT/scratch/bench (deleted unless --keep).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "bench"))
from _common import EXT, WINDOWS, bench_env, deck_for, hwm_mb, hwm_reset, pin, same_file, workdir_for  # noqa: E402


def torch_chain(deck: Path, wlbeg: float, wlend: float, wd: Path) -> dict:
    from atlas_torch.synthe import spectrv_prog as sp
    from atlas_torch.synthe.chain import FORT25, MOLECULAR_LISTS, basicatlas_dir
    from atlas_torch.synthe.lines import run_rgfalllines, run_rh2ofast, run_rmolecasc, run_synbeg
    from atlas_torch.synthe.synthe_prog import run_synthe
    from atlas_torch.synthe.xnfpelsyn_prog import xnfpelsyn
    ba = basicatlas_dir()
    s_files, d_files = ba / "data" / "synthe_files", ba / "data" / "dfsynthe_files"
    wd.mkdir(parents=True, exist_ok=True)
    rec = {}

    def step(name, fn):
        hwm_reset()
        t = time.perf_counter()
        r = fn()
        rec[name] = dict(seconds=time.perf_counter() - t, peak_rss_mb=hwm_mb())
        return r
    deck_text = deck.read_text()
    xr = step("xnfpelsyn", lambda: xnfpelsyn(deck_text, (s_files / "continua.dat").read_text(),
                                             s_files / "molecules.dat", out10=wd / "xnfpelsyn.dat"))
    step("synbeg", lambda: run_synbeg(workdir=wd, wlbeg=wlbeg, wlend=wlend, resolu=600000., vturb=1.5, ifnlte=0,
                                      linout=-1, cutoff=0.0001, ifpred=1, nread=0, airvac="VAC"))
    step("rgfalllinesnew", lambda: run_rgfalllines(workdir=wd, linelist_path=s_files / "BasicATLAS.dat"))
    mol = {}
    for m in MOLECULAR_LISTS:
        step("_m", lambda: run_rmolecasc(workdir=wd, mol_path=s_files / m, c12=None, c13=None))
        mol[m] = rec.pop("_m")
    rec["rmolecasc"] = dict(seconds=sum(v["seconds"] for v in mol.values()),
                            peak_rss_mb=max(v["peak_rss_mb"] for v in mol.values()), per_list=mol)
    step("rh2ofast", lambda: run_rh2ofast(workdir=wd, h2o_path=d_files / "h2ofastfix.bin"))
    for name, src in (("fort.10", wd / "xnfpelsyn.dat"), ("fort.18", s_files / "he1tables.dat")):
        (wd / name).unlink(missing_ok=True)
        os.symlink(os.path.abspath(src), wd / name)
    step("synthe", lambda: run_synthe(wd, stdout_path=wd / "synthe.out"))
    (wd / "fort.25").write_text(FORT25)

    def spectrv():
        res = sp.spectrv(xr.readin_state, sp.load_inputs(wd), libm="libmvec")
        res.write(wd / "spectrum.bin")
    step("spectrv", spectrv)
    rec["total"] = dict(seconds=sum(v["seconds"] for k, v in rec.items() if k != "total"),
                        peak_rss_mb=max(v["peak_rss_mb"] for k, v in rec.items() if k != "total"))
    return rec


def fortran_chain(deck: Path, wlbeg: float, wlend: float, wd: Path, core) -> dict:
    """scripts/synthe_workdir.sh with per-program /usr/bin/time wrappers (stock BasicATLAS executables)."""
    tb = wd.parent / f"timebin_{os.getpid()}"
    tb.mkdir(parents=True, exist_ok=True)
    tf = wd.parent / f"fortran_{os.getpid()}.time"
    tf.unlink(missing_ok=True)
    for e in ("xnfpelsyn", "synbeg", "rgfalllinesnew", "rmolecasc", "rh2ofast", "synthe", "spectrv"):
        (tb / f"{e}.exe").write_text(f'#!/bin/bash\nexec /usr/bin/time -a -o {tf} -f "{e} %e %M" '
                                     f'{EXT}/BasicATLAS/bin/{e}.exe "$@"\n')
        (tb / f"{e}.exe").chmod(0o755)
    pre = ["taskset", "-c", str(core)] if core is not None else []
    subprocess.run(pre + [str(REPO / "scripts" / "synthe_workdir.sh"), str(tb), str(deck), str(wlbeg), str(wlend),
                          str(wd)], check=True, stdout=subprocess.DEVNULL)
    rec = {}
    for line in tf.read_text().splitlines():
        name, el, kb = line.split()
        r = rec.setdefault(name, dict(seconds=0.0, peak_rss_mb=0.0))
        r["seconds"] += float(el)
        r["peak_rss_mb"] = max(r["peak_rss_mb"], int(kb) / 1024)
    rec["total"] = dict(seconds=sum(v["seconds"] for v in rec.values()),
                        peak_rss_mb=max(v["peak_rss_mb"] for v in rec.values()))
    shutil.rmtree(tb, ignore_errors=True)
    tf.unlink(missing_ok=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", default="500_510", choices=sorted(WINDOWS))
    ap.add_argument("--core", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fortran", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    pin(a.core, a.threads)
    src, wlbeg, wlend, deck = WINDOWS[a.window]
    base = EXT / "scratch" / "bench" / f"chain_{a.window}_{os.getpid()}"
    out = dict(bench="synthe_chain", window=a.window, **bench_env(a.core, a.threads))
    try:
        out["torch"] = torch_chain(deck, wlbeg, wlend, base / "torch")
        out["identical_to_stock"] = same_file(base / "torch" / "spectrum.bin", src / "spectrum.bin")
        if not a.keep:
            shutil.rmtree(base / "torch", ignore_errors=True)
        if a.fortran:
            out["fortran"] = fortran_chain(deck, wlbeg, wlend, base / "fortran", a.core)
            out["speed_ratio_torch_over_fortran"] = out["torch"]["total"]["seconds"] / out["fortran"]["total"]["seconds"]
        out["loadavg_end"] = list(os.getloadavg())
        print(json.dumps(out, indent=1))
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    finally:
        if not a.keep:
            subprocess.run(["chmod", "-R", "u+w", str(base)], stderr=subprocess.DEVNULL)  # read-only pre_* snapshots
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
