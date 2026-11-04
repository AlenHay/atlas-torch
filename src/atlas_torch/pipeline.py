"""Torch stage runners used by `make score` (stage-isolated: each consumes the stock Fortran inputs).

STAGES[kind](stage, reference_set) -> path of the candidate output dir, comparable with the stock output by
atlas_torch.scoring. Each runner imports its torch implementation lazily; a missing implementation raises
NotImplementedError and the stage is scored `not_implemented`.
"""
from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
OUT = EXT / "runs" / "torch"


def _need(mod: str, attr: str):
    try:
        m = importlib.import_module(mod)
    except ImportError as e:
        raise NotImplementedError(f"{mod} not available ({e})") from e
    if not hasattr(m, attr):
        raise NotImplementedError(f"{mod}.{attr} missing")
    return getattr(m, attr)


def _fresh(p: Path) -> Path:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p


def kappa9_stage(stage: str, ref: dict) -> Path:
    """kappa9/<mix>: torch KAPPA9 on the stock ODF of that mix + torch KAPREADTS."""
    mix = stage.split("/")[1]
    run = _need("atlas_torch.kappa9.kappa9_prog", "run_workdir")
    out = _fresh(OUT / "kappa9" / mix)
    run(odf_dir=EXT / "runs/stock/odf" / mix, workdir=out)
    return out


def odf_stage(stage: str, ref: dict) -> Path:
    """odf/<mix>: full torch DFSYNTHE pipeline (xnfdf -> dfsynthe -> dfsortp -> separatedf -> kappa9 -> kapreadts)."""
    mix = stage.split("/")[1]
    run = _need("atlas_torch.compat.atlas", "dfsynthe")
    settings_cls = _need("atlas_torch.compat.settings", "Settings")
    os.environ.setdefault("ATLAS_TORCH_BACKEND", "torch")
    s = settings_cls()
    s.zscale = ref["odfs"][mix]["zscale"]
    s.abun = dict(ref["odfs"][mix]["abun"])
    out = OUT / "odf" / mix
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run(str(out), s, parallel=True, silent=True)
    return out


def atlas_stage(stage: str, ref: dict) -> Path:
    """atlas/<model>: torch ATLAS9 with the stock ODF of the model's mix, BasicATLAS defaults (restart auto)."""
    name = stage.split("/")[1]
    m = ref["models"][name]
    run = _need("atlas_torch.compat.atlas", "atlas")
    settings_cls = _need("atlas_torch.compat.settings", "Settings")
    if "atlas9mem" not in _need("atlas_torch.compat.runners", "registered")():
        raise NotImplementedError("no torch atlas9mem runner registered")
    s = settings_cls()
    s.zscale = ref["odfs"][m["odf"]]["zscale"]
    s.abun = dict(ref["odfs"][m["odf"]]["abun"])
    s.teff, s.logg, s.vturb = m["teff"], m["logg"], ref["atlas"]["vturb"]
    out = OUT / "atlas" / name
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = os.environ.get("ATLAS_TORCH_BACKEND")
    os.environ["ATLAS_TORCH_BACKEND"] = "torch"          # every program of the run must be the torch port
    try:
        run(str(out), s, restart=ref["atlas"]["restart"], niter=ref["atlas"]["niter"],
            ODF=str(EXT / "runs/stock/odf" / m["odf"]), silent=True)
    finally:
        if old is None:
            os.environ.pop("ATLAS_TORCH_BACKEND", None)
        else:
            os.environ["ATLAS_TORCH_BACKEND"] = old
    return out


def synthe_stage(stage: str, ref: dict) -> Path:
    """synthe/sun/w<i>: torch SYNTHE on the stock solar model for window i."""
    i = int(stage.split("/w")[1])
    lo, hi = ref["synthe"]["windows_nm"][i]
    run = _need("atlas_torch.compat.atlas", "synthe")
    os.environ.setdefault("ATLAS_TORCH_BACKEND", "torch")
    model = OUT / "synthe_models" / f"sun_w{i}"
    src = EXT / "runs/stock/atlas/sun"
    if model.exists():
        shutil.rmtree(model)
    shutil.copytree(src, model, ignore=shutil.ignore_patterns("synthe_*", "window_*"))
    run(str(model), lo, hi, res=ref["synthe"]["res"], vturb=ref["synthe"]["vturb"], silent=True, progress=False)
    return model / "synthe_1"


def api_stage(ref: dict) -> dict:
    """Run BasicATLAS's example notebooks with the torch backend (scripts/run_examples.py torch); returns
    {example: HOME dir}. Reuses an existing complete run unless ATLAS_TORCH_RERUN_EXAMPLES=1."""
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    base = EXT / "runs" / "torch" / "examples"
    done = base / "timing.json"
    if not done.exists() or os.environ.get("ATLAS_TORCH_RERUN_EXAMPLES") == "1":
        env = dict(os.environ, ATLAS_TORCH_BACKEND="torch")
        r = subprocess.run([sys.executable, str(repo / "scripts/run_examples.py"), "torch"], env=env)
        if r.returncode != 0:
            raise RuntimeError(f"examples failed (exit {r.returncode}); see run_examples output")
    return {ex: base / ex for ex in ref["examples"]}


STAGES = {"kappa9": kappa9_stage, "odf": odf_stage, "atlas": atlas_stage, "synthe": synthe_stage, "api": api_stage}
