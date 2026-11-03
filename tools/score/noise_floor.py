"""Measure the Fortran's own noise floor and freeze tolerances (toolkit item 5).

For each scored stage, every noise variant (reference_set.json noise_variants: other compilers / flags,
stage-isolated inputs) is compared with the stock reference using atlas_torch.scoring. Per quantity:
    spread = max over variants of the error
    tol    = max(3 * spread, quantum)
`quantum` is one unit of the output representation (1 ulp for binary float64, the last printed digit for
text tables, 1 for integer ODF steps); it only matters when every variant agrees exactly.
Writes results/noise_floor.json (all per-variant numbers) and results/tolerances.json, which is FROZEN:
the tool refuses to overwrite it (gate changes must be argued in RESULTS.md, see BRIEF).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from atlas_torch.scoring import atlas_metrics, gated_keys, odf_metrics, synthe_metrics  # noqa: E402

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
RUNS = EXT / "runs"
EPS = 2.220446049250313e-16
QUANTUM = {  # by quantity prefix
    "flux": EPS, "cont": EPS, "normflux": EPS,
    "odf_v": 1.0, "logkappa_ros": 1e-3,
    "RHOX": 1e-5, "P": 1e-5, "XNE": 1e-5, "RHO": 1e-5, "ABROSS": 1e-5, "TAUROS": 1e-5, "T": 2e-5,
    "FLXERR": 0.0, "FLXDRV": 0.0, "HEIGHT": 0.0, "FLXCNV_over_total": 1e-5,
}


def quantum(key):
    q = key.split(":")[0]
    if q.startswith("odf_v"):
        return QUANTUM["odf_v"]
    return QUANTUM.get(q, 0.0)


def stage_pairs(ref):
    """yield (stage, metric_fn, stock_dir, {variant: dir})"""
    nv = ref["noise_variants"]
    for i, _ in enumerate(ref["synthe"]["windows_nm"]):
        yield (f"synthe/sun/w{i}", synthe_metrics, RUNS / "stock/atlas/sun" / f"window_{i}",
               {v: RUNS / v / "atlas/sun" / f"window_{i}" for v in nv["synthe"]})
    for o in ref["scored_odfs"]:
        yield (f"odf/{o}", odf_metrics, RUNS / "stock/odf" / o, {v: RUNS / v / "odf" / o for v in nv["dfsynthe"]})
    for m in ref["models"]:
        yield (f"atlas/{m}", atlas_metrics, RUNS / "stock/atlas" / m, {v: RUNS / v / "atlas" / m for v in nv["atlas"]})


def main():
    ref = json.loads((REPO / "reference/reference_set.json").read_text())
    floor, tols, missing = {}, {}, []
    for stage, fn, sdir, vdirs in stage_pairs(ref):
        per = {}
        for v, d in vdirs.items():
            try:
                per[v] = fn(sdir, d)
            except (FileNotFoundError, ValueError, IndexError) as e:
                per[v] = {"error": f"{type(e).__name__}: {e}"}
        floor[stage] = per
        ok = {v: m for v, m in per.items() if "error" not in m}
        if not ok:
            missing.append(stage)
            continue
        keys = gated_keys(next(iter(ok.values())))
        t = {}
        for k in keys:
            spread = max(m[k] for m in ok.values())
            t[k] = {"tol": max(3.0 * spread, quantum(k)), "spread": spread, "quantum": quantum(k),
                    "worst_variant": max(ok, key=lambda v: ok[v][k]), "n_variants": len(ok)}
        tols[stage] = t
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results/noise_floor.json").write_text(json.dumps(floor, indent=1))
    print("stages without noise data:", missing)
    out = REPO / "results/tolerances.json"
    if out.exists() and "--write" in sys.argv:
        sys.exit("results/tolerances.json is frozen; refusing to overwrite (see BRIEF: gates never weaken)")
    if "--write" in sys.argv:
        if missing:
            sys.exit(f"refusing to freeze with missing stages: {missing}")
        body = {"policy": "tol = max(3*spread over Fortran variants vs stock, 1 quantum of the output format)",
                "variants": ref["noise_variants"], "stages": tols}
        s = json.dumps(body, indent=1, sort_keys=True)
        body["sha256"] = hashlib.sha256(s.encode()).hexdigest()
        out.write_text(json.dumps(body, indent=1, sort_keys=True))
        print("froze", out)
    else:
        print(json.dumps(tols, indent=1)[:4000])


if __name__ == "__main__":
    main()
