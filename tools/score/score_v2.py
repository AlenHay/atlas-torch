"""Score the v2 synthesis on the solar SYNTHE windows of reference/reference_set.json against results/v2/tolerances.json.

For each window: the stock solar model (BasicATLAS output_summary.out -> synthe deck) through atlas_torch.v2.synthesis
with either the v2 opacity modules or, as stand-ins, the exact port's opacity tables of the stock stage work dir
(fort.10, fort.9 of work/stocksun_<lo>_<hi>; isolates the transfer). Writes runs/v2/sun/window_<i>/spectrum.bin,
scores it with atlas_torch.scoring.synthe_metrics_v2 against runs/stock/atlas/sun/window_<i>, appends one record
to results/scoreboard.jsonl (note prefixed "v2").
usage: python tools/score/score_v2.py [--windows 0,1,2,3] [--opacity auto|v2|exact-port] [--kurucz] [--note ...]
       python tools/score/score_v2.py --stage atlas [--kurucz] [--models sun,t9000g40] [--batch 2] [--note ...]
       python tools/score/score_v2.py --stage kappa_ros|odf [--kurucz] [--mixes solar,alpha] (tools/score/score_v2_odf.py)
  --kurucz: atlas_torch.v2.synthesis.SynthesisConfig.kurucz() instead of SynthesisConfig(): the stock Fortran's
            choice in every module (EOS network/lowering, constants, continuum switches and edge sampling, line
            options/selection/isotopes, transfer).
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def _fmt(x: float) -> str:
    return f"{x:g}" if x != int(x) else str(int(x))


def reset_peak_rss() -> None:
    """Linux: restart the VmHWM high-water mark of this process (proc(5), clear_refs value 5)."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        pass


def peak_rss_mb() -> float:
    """Peak resident memory since the last reset_peak_rss (VmHWM), else of the whole process (ru_maxrss)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def v2_opacity(grid, config):
    """(continuum, lines) providers of atlas_torch.v2.opacity (NotImplementedError while it is missing)."""
    try:
        v2 = config.opacity(grid)
    except ImportError as e:
        raise NotImplementedError(f"atlas_torch.v2.opacity not available ({e})") from e
    return v2.continuum, v2.lines


def _record(note: str, config: dict, stages: dict) -> dict:
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note,
            "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                                  text=True).strip()),
            "v2": config, "tolerances": "results/v2/tolerances.json", "stages": stages,
            "summary": {s: sum(1 for v in stages.values() if v["status"] == s)
                        for s in ("pass", "fail", "crash", "not_implemented")}}


ATLAS_KEYS = ["T:rel", "P:rel", "XNE:rel", "RHOX:rel", "ABROSS:rel", "FLXERR:own_max", "FLXDRV:own_max"]


def main_atlas(a) -> None:
    """--stage atlas: the reference models from their stock restart.dat and atlas_control.com with the stock
    p00big{v}.bdf and kappa.ros of their mixture through atlas_torch.v2.structure (StructureConfig.kurucz() with
    --kurucz, else StructureConfig()); products in $ATLAS_EXT/runs/v2/atlas/<config>/<model>/, scored with
    scoring.atlas_metrics_v2 against runs/stock/atlas/<model> and gated on stages.atlas plus its also_require
    conditions (own max |FLXERR| < 1, |FLXDRV| < 10 over log tau_Ross in [-4, 1])."""
    import torch
    from atlas_torch.scoring import atlas_also_require, atlas_metrics_v2, gate_v2
    from atlas_torch.v2.io.atlas_control import read_restart
    from atlas_torch.v2.opacity.odf import OpacityDistribution
    from atlas_torch.v2.opacity.rosseland import RosselandTable
    from atlas_torch.v2.structure import ModelSpec, StructureConfig, StructureSolver

    torch.set_num_threads(a.threads)
    ref = json.loads((REPO / "reference/reference_set.json").read_text())
    tols = json.loads((REPO / "results/v2/tolerances.json").read_text())["stages"]["atlas"]
    names = list(ref["models"]) if a.models == "all" else a.models.split(",")
    cfg = StructureConfig.kurucz() if a.kurucz else StructureConfig()
    cname = "kurucz" if a.kurucz else "v2"
    tables = {}
    specs = []
    for name in names:
        mix = ref["models"][name]["odf"]
        odf_dir = EXT / "runs/stock/odf" / mix
        stock = EXT / "runs/stock/atlas" / name
        if mix not in tables:
            v = int(round(float(ref["atlas"]["vturb"])))
            tables[mix] = (OpacityDistribution.read(odf_dir, "big", vturbs=(v,)), RosselandTable.read(odf_dir / "kappa.ros"))
        specs.append(ModelSpec.from_control(stock / "atlas_control.com", odf_dir, odf=tables[mix][0],
                                            kappa_ros=tables[mix][1]))
    stages = {}
    for i in range(0, len(names), a.batch):
        group, gspecs = names[i:i + a.batch], specs[i:i + a.batch]
        t0 = time.time()
        reset_peak_rss()
        try:
            solver = StructureSolver(cfg, ModelSpec.stack(gspecs))
            start = solver.start([read_restart(EXT / "runs/stock/atlas" / n / "restart.dat") for n in group])
            res = solver.run(start, max_iterations=a.max_iterations,
                             on_iteration=lambda k, s, rec, act: print(
                                 f"  {group} it {k}: max|err| {rec['max_flux_error'].round(3).tolist()} "
                                 f"max|drv| {rec['max_flux_derivative'].round(2).tolist()}", flush=True))
        except Exception as e:  # noqa: BLE001  (a crash is a failing score, recorded)
            for n in group:
                stages[f"v2/atlas/{cname}/{n}"] = {"status": "crash", "detail": f"{type(e).__name__}: {e}"}
            continue
        seconds, rss = round(time.time() - t0, 1), round(peak_rss_mb(), 1)
        for m, n in enumerate(group):
            out = EXT / "runs/v2/atlas" / cname / n
            res.write(out, m)
            met = atlas_metrics_v2(EXT / "runs/stock/atlas" / n, out, tols)
            fails = gate_v2(met, tols)
            fails.update(atlas_also_require(met))
            stages[f"v2/atlas/{cname}/{n}"] = {
                "status": "pass" if not fails and "error" not in met else "fail", "metrics": met, "fails": fails,
                "iterations": len(res.records[m]), "best_iteration": res.best(m) + 1, "stopped": res.stopped[m],
                "seconds": seconds, "batch": group, "peak_rss_mb": rss}
    rec = _record(("v2 atlas: " + a.note if a.note else "v2 atlas") + (" [kurucz-compatible]" if a.kurucz else ""),
                  {"config": cname, **cfg.describe()}, stages)
    with open(REPO / "results/scoreboard.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps(rec["summary"]))
    for s, v in stages.items():
        m = v.get("metrics", {})
        print(f"{s:34s} {v['status']:6s} {v.get('seconds', '')}s it {v.get('iterations', '')} best "
              f"{v.get('best_iteration', '')} " + " ".join(f"{k}={m[k]:.3g}" for k in ATLAS_KEYS if k in m)
              + (f"  FAILS {sorted(v['fails'])}" if v.get("fails") else "") + (f"  {v['detail']}" if "detail" in v else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="synthe", choices=["synthe", "atlas", "odf", "kappa_ros"])
    ap.add_argument("--windows", default="0,1,2,3")
    ap.add_argument("--opacity", default="auto", choices=["auto", "v2", "exact-port"])
    ap.add_argument("--kurucz", action="store_true")
    ap.add_argument("--deck", default=None, help="synthe: model deck (output_summary.out) instead of the stock sun's")
    ap.add_argument("--models", default="all", help="--stage atlas: comma-separated reference models")
    ap.add_argument("--batch", type=int, default=1, help="--stage atlas: models per tensor batch")
    ap.add_argument("--max-iterations", type=int, default=450)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mixes", default="solar,alpha", help="--stage kappa_ros / odf: ODF mixtures")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.stage == "atlas":
        return main_atlas(a)
    if a.stage in ("kappa_ros", "odf"):
        import score_v2_odf
        fn = score_v2_odf.main_kappa_ros if a.stage == "kappa_ros" else score_v2_odf.main_odf
        return fn(a, _record, reset_peak_rss, peak_rss_mb)
    import torch
    from atlas_torch.scoring import gate_v2, synthe_metrics_v2
    from atlas_torch.v2.io import read_atlas_deck, to_synthe_deck
    from atlas_torch.v2.opacity.lines.grid import LogWavelengthGrid
    from atlas_torch.v2.synthesis import SynthesisConfig, TableOpacity, deck_title, synthesize

    ref = json.loads((REPO / "reference/reference_set.json").read_text())
    tols = json.loads((REPO / "results/v2/tolerances.json").read_text())["stages"]["synthe"]
    syn = ref["synthe"]
    deck_path = Path(a.deck) if a.deck else EXT / "runs/stock/atlas" / syn["model"] / "output_summary.out"
    deck = to_synthe_deck(read_atlas_deck(deck_path))
    atm, comp = deck.atmosphere, deck.composition
    cfg = (SynthesisConfig.kurucz if a.kurucz else SynthesisConfig)(microturbulence=syn["vturb"] * 1e5)
    transfer = cfg.transfer_solver()
    config = {"opacity": None, "config": "kurucz" if a.kurucz else "v2", "deck": str(deck_path), **cfg.describe(),
              "transfer": {"method": transfer.method, "n_mu": transfer.n_mu, "opaque_top": transfer.opaque_top,
                           "tau_scheme": transfer.tau_scheme, "scattering": transfer.scattering}}
    stages = {}
    for i in map(int, a.windows.split(",")):
        lo, hi = syn["windows_nm"][i]
        stage = f"v2/synthe/{syn['model']}/w{i}"
        stock = EXT / "runs/stock/atlas" / syn["model"] / f"window_{i}"
        out = EXT / "runs/v2" / (syn["model"] if not a.deck else f"synthe_from_{deck_path.parent.parent.name}_"
                                 f"{deck_path.parent.name}") / f"window_{i}"
        out.mkdir(parents=True, exist_ok=True)
        grid = LogWavelengthGrid.window(float(lo), float(hi), float(syn["res"]))
        t0 = time.time()
        reset_peak_rss()
        try:
            if a.opacity in ("auto", "v2"):
                try:
                    cont, lines = v2_opacity(grid, cfg)
                    kind, vacuum = "v2", True
                except NotImplementedError:
                    if a.opacity == "v2":
                        raise
                    kind = None
            if a.opacity == "exact-port" or (a.opacity == "auto" and kind is None):
                tab = TableOpacity(EXT / "work" / f"stocksun_{_fmt(lo)}_{_fmt(hi)}")
                if tab.grid != grid:
                    raise ValueError(f"stage grid {tab.grid} != {grid}")
                cont, lines, kind = tab.continuum, tab.lines, "exact-port (fort.10, fort.9 stand-ins)"
                vacuum = tab.vacuum
            config["opacity"] = kind
            with torch.no_grad():
                spec = synthesize(atm, comp, grid, continuum=cont, lines=lines, transfer=transfer, config=cfg,
                                  title=deck_title(deck), vacuum=vacuum)
            spec.write(out / "spectrum.bin")
            del spec, cont, lines
        except NotImplementedError as e:
            stages[stage] = {"status": "not_implemented", "detail": str(e)}
            continue
        except Exception as e:  # noqa: BLE001  (a crash is a failing score, recorded)
            stages[stage] = {"status": "crash", "detail": f"{type(e).__name__}: {e}"}
            continue
        m = synthe_metrics_v2(stock, out)
        fails = gate_v2(m, tols)
        stages[stage] = {"status": "pass" if not fails and "error" not in m else "fail", "metrics": m,
                         "fails": fails, "seconds": round(time.time() - t0, 1), "opacity": kind,
                         "peak_rss_mb": round(peak_rss_mb(), 1)}
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "note": ("v2" + (": " + a.note if a.note else "")) + (" [kurucz-compatible]" if a.kurucz else ""),
           "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip(),
           "dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                                 text=True).strip()),
           "v2": config, "tolerances": "results/v2/tolerances.json", "stages": stages,
           "summary": {s: sum(1 for v in stages.values() if v["status"] == s)
                       for s in ("pass", "fail", "crash", "not_implemented")}}
    with open(REPO / "results/scoreboard.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in rec if k != "stages"}, indent=1))
    keys = ["cont:rel", "cont:rms_rel", "flux:rms_rel", "normflux:abs", "normflux:p99", "normflux:rms_abs",
            "flux_over_cont:abs"]
    for s, v in stages.items():
        print(f"{s:22s} {v['status']:8s} {v.get('seconds', '')}s " + " ".join(
            f"{k}={v['metrics'][k]:.2e}" for k in keys if k in v.get("metrics", {})) + (
            f"  peak {v['peak_rss_mb']:.0f} MB" if "peak_rss_mb" in v else "") + (
            f"  FAILS {sorted(v['fails'])}" if v.get("fails") else "") + (f"  {v['detail']}" if "detail" in v else ""))


if __name__ == "__main__":
    main()
