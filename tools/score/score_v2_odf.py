"""score_v2 --stage kappa_ros and --stage odf (imported by score_v2.py).

kappa_ros: the stock p00big{v}.bdf of each mixture through atlas_torch.v2.opacity.rosseland (RosselandConfig.kurucz()
with --kurucz, else RosselandConfig()); kapk{v}.dat and kappa.ros in $ATLAS_EXT/runs/v2/kappa_ros/<config>/<mix>/,
scored with scoring.kappa9_metrics_v2 against the stock KAPPA9 tables (work/kappa9_<mix>_stock, kappa.ros of
runs/stock/odf/<mix>) and, when present, against the stock tables recomputed with KAPPA9's HCKT updated per
temperature ($ATLAS_EXT/runs/v2/kappa_ros/kappa9_hckt_fixed/<mix>/: the stale-HCKT artefact removed).
odf: scores the v2 ODF products $ATLAS_EXT/runs/v2/odf/<config>/<mix>/ (tools/score/build_v2_odf.py) with
scoring.odf_metrics_v2 against runs/stock/odf/<mix>. When only some temperatures are built (.parts/t{index}.pt),
scores those rows with scoring.odf_rows_metrics_v2: BIG steps vs the stock p00big{v}.bdf rows, and log kappa_R at
those temperatures from the stock BIG table with the v2 rows substituted (quantized as in p00big{v}.bdf;
RosselandConfig of the same configuration; all 57 temperatures tabulated so KAPPA9's lagged lowering chain is
unchanged) vs the stock kappa.ros and the HCKT-fixed one; logkappa_ros_odf_only:* compares it with the same
Rosseland configuration on the unchanged stock ODF (the ODF's own effect). 'not_built' when nothing exists."""
import json
import os
import time
from pathlib import Path

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
REPO = Path(__file__).resolve().parents[2]


def main_kappa_ros(a, record, reset_peak_rss, peak_rss_mb) -> None:
    from atlas_torch.scoring import gate_v2, kappa9_metrics_v2
    from atlas_torch.v2.io.odf_directory import read_abundance_cards
    from atlas_torch.v2.opacity.odf import OpacityDistribution
    from atlas_torch.v2.opacity.rosseland import RosselandConfig, RosselandOpacity

    tols = json.loads((REPO / "results/v2/tolerances.json").read_text())["stages"]["kappa_ros"]
    cfg = RosselandConfig.kurucz() if a.kurucz else RosselandConfig()
    cname = "kurucz" if a.kurucz else "v2"
    stages = {}
    for mix in a.mixes.split(","):
        src = EXT / "runs/stock/odf" / mix
        out = EXT / "runs/v2/kappa_ros" / cname / mix
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        reset_peak_rss()
        try:
            tab = RosselandOpacity(OpacityDistribution.read(src, "big"), cfg).tabulate(
                read_abundance_cards(src / "kappa9v0.com"))
            tab.write_kapk(out, cfg.kapk_rounding)
            tab.write(out / "kappa.ros")
        except Exception as e:  # noqa: BLE001  (a crash is a failing score, recorded)
            stages[f"v2/kappa_ros/{cname}/{mix}"] = {"status": "crash", "detail": f"{type(e).__name__}: {e}"}
            continue
        seconds, rss = round(time.time() - t0, 1), round(peak_rss_mb(), 1)
        refs = {"stock": EXT / "work" / f"kappa9_{mix}_stock",
                "stock_hckt_fixed": EXT / "runs/v2/kappa_ros/kappa9_hckt_fixed" / mix}
        for rname, ref in refs.items():
            if not (ref / "kapk0.dat").exists():
                continue
            met = kappa9_metrics_v2(ref, out, src / "kappa.ros" if rname == "stock" else ref / "kappa.ros")
            fails = gate_v2(met, tols)
            stages[f"v2/kappa_ros/{cname}/{mix}/vs_{rname}"] = {
                "status": "pass" if not fails else "fail", "metrics": met, "fails": fails, "seconds": seconds,
                "peak_rss_mb": rss}
    rec = record(("v2 kappa_ros: " + a.note if a.note else "v2 kappa_ros") + (" [kurucz-compatible]" if a.kurucz else ""),
                 {"config": cname, **cfg.describe()}, stages)
    _finish(rec, stages, ["kapk_v0:abs", "kapk_v2:abs", "kapk_v8:abs", "logkappa_ros:abs", "logkappa_ros:rms_abs"])


def main_odf(a, record, reset_peak_rss, peak_rss_mb) -> None:
    from atlas_torch.scoring import gate_v2, odf_metrics_v2
    tols = json.loads((REPO / "results/v2/tolerances.json").read_text())["stages"]["odf"]
    cname = "kurucz" if a.kurucz else "v2"
    stages = {}
    for mix in a.mixes.split(","):
        cand = EXT / "runs/v2/odf" / cname / mix
        key = f"v2/odf/{cname}/{mix}"
        if not all((cand / f"p00big{v}.bdf").exists() for v in (0, 1, 2, 4, 8)) or not (cand / "kappa.ros").exists():
            if sorted((cand / ".parts").glob("t*.pt")):
                stages.update(_score_rows(a, mix, cname, cand, tols, reset_peak_rss, peak_rss_mb))
            else:
                stages[key] = {"status": "not_built", "detail": f"no ODF products in {cand} (tools/score/build_v2_odf.py)"}
            continue
        t0 = time.time()
        met = odf_metrics_v2(EXT / "runs/stock/odf" / mix, cand)
        fails = gate_v2(met, tols)
        stages[key] = {"status": "pass" if not fails and "error" not in met else "fail", "metrics": met,
                       "fails": fails, "seconds": round(time.time() - t0, 1)}
    rec = record(("v2 odf: " + a.note if a.note else "v2 odf") + (" [kurucz-compatible]" if a.kurucz else ""),
                 {"config": cname}, stages)
    rec["summary"]["not_built"] = sum(1 for v in stages.values() if v["status"] == "not_built")
    _finish(rec, stages, ["odf_v0:rms_abs", "odf_v2:rms_abs", "odf_v2:p999", "odf_v8:p999", "logkappa_ros:abs",
                          "logkappa_ros:rms_abs", "logkappa_ros_odf_only:abs"])


def _score_rows(a, mix, cname, cand, tols, reset_peak_rss, peak_rss_mb) -> dict:
    """Subset scoring of the built temperatures (see the module docstring); one stage per reference kappa.ros."""
    import numpy as np
    import torch
    from atlas_torch.scoring import VTURBS, _abs, _put, gate_v2, odf_rows_metrics_v2
    from atlas_torch.v2.io.odf_directory import read_abundance_cards
    from atlas_torch.v2.opacity.odf import ODF_LOG_T, OpacityDistribution, quantize
    from atlas_torch.v2.opacity.rosseland import RosselandConfig, RosselandOpacity, RosselandTable

    src = EXT / "runs/stock/odf" / mix
    t0 = time.time()
    reset_peak_rss()
    idx = sorted(int(p.stem[1:]) for p in (cand / ".parts").glob("t*.pt"))
    stock = OpacityDistribution.read(src, "big")
    lk = stock.log_kappa.clone()
    steps = {v: [] for v in VTURBS}
    for i in idx:
        d = torch.load(cand / ".parts" / f"t{i:02d}.pt")
        vt = list(d.get("vturbs", VTURBS))
        for v in VTURBS:
            q = quantize(d["big"][vt.index(v)])
            steps[v].append(q)
            lk[VTURBS.index(v), :, i] = torch.from_numpy(q.astype(np.float64)) / 1000.0
    steps = {v: np.stack(s, axis=1) for v, s in steps.items()}
    rcfg = RosselandConfig.kurucz() if cname == "kurucz" else RosselandConfig()
    comp = read_abundance_cards(src / "kappa9v0.com")
    hybrid = RosselandOpacity(OpacityDistribution(stock.intervals, lk, stock.vturb_kms), rcfg).tabulate(comp)
    own = hybrid.log_kappa.numpy()[:, idx].transpose(1, 2, 0)
    base = EXT / "runs/v2/kappa_ros" / cname / mix / "kappa.ros"
    odf_only = {}
    if base.exists():
        _put(odf_only, "logkappa_ros_odf_only", "abs", _abs(RosselandTable.read(base).log_kappa.numpy()[:, idx]
                                                            .transpose(1, 2, 0), own))
    seconds, rss = round(time.time() - t0, 1), round(peak_rss_mb(), 1)
    temps = [round(float(10 ** ODF_LOG_T[i])) for i in idx]
    out = {}
    for rname, kr in (("stock", src / "kappa.ros"),
                      ("stock_hckt_fixed", EXT / "runs/v2/kappa_ros/kappa9_hckt_fixed" / mix / "kappa.ros")):
        if not kr.exists():
            continue
        met = odf_rows_metrics_v2(src, steps, idx, own, kr)
        met.update(odf_only)
        per_t = {}
        for j, i in enumerate(idx):
            m1 = odf_rows_metrics_v2(src, {v: s[:, j:j + 1] for v, s in steps.items()}, [i], own[j:j + 1], kr)
            per_t[str(temps[j])] = {k: m1[k] for k in ("odf_v0:rms_abs", "odf_v2:rms_abs", "odf_v8:rms_abs",
                                                       "odf_v2:p999", "logkappa_ros:abs", "logkappa_ros:rms_abs")}
        fails = gate_v2(met, tols)
        build = [torch.load(cand / ".parts" / f"t{i:02d}.pt").get("stats", {}).get("seconds") for i in idx]
        out[f"v2/odf/{cname}/{mix}/T{','.join(map(str, temps))}/vs_{rname}"] = {
            "status": "pass" if not fails and "error" not in met else "fail", "metrics": met, "fails": fails,
            "per_temperature": per_t,
            "subset": {"temperature_index": idx, "T_K": temps, "build_seconds": build,
                       "note": "temperature subset of the 57; logkappa_ros at these temperatures only"},
            "seconds": seconds, "peak_rss_mb": rss}
    return out


def _finish(rec, stages, keys) -> None:
    with open(REPO / "results/scoreboard.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps(rec["summary"]))
    for s, v in stages.items():
        m = v.get("metrics", {})
        print(f"{s:44s} {v['status']:9s} {v.get('seconds', '')}s " + " ".join(f"{k}={m[k]:.3g}" for k in keys if k in m)
              + (f"  FAILS {sorted(v['fails'])}" if v.get("fails") else "") + (f"  {v['detail']}" if "detail" in v else ""))
