"""`make score`: scoreboard of the torch port vs the frozen Fortran reference (toolkit item 6).

1. fast tier: pytest (oracle/unit tests) -> routines ported/passing per variant
   (tests/oracle/test_<variant>_<routine>.py; a routine passes if all its tests pass).
2. every scored stage: run the torch implementation on the stage-isolated Fortran inputs
   (atlas_torch.pipeline.STAGES; missing = not_implemented), compute atlas_torch.scoring metrics
   against the stock Fortran output, gate each quantity on results/tolerances.json.
3. append one JSON line to results/scoreboard.jsonl, rewrite the summary block of RESULTS.md,
   residual plots in results/plots/.
usage: python tools/score/score.py [--stages synthe,odf,atlas,api] [--note "attempt description"] [--no-tests]
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "score"))
from noise_floor import EXAMPLE_PRODUCTS, stage_pairs  # noqa: E402

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def run_tests():
    xml = REPO / "results" / ".pytest.xml"
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not stage and not full", "-p", "no:cacheprovider",
                    f"--junitxml={xml}", "tests"], cwd=REPO, capture_output=True)
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # variant -> routine -> [passed, total]
    tot = {"passed": 0, "failed": 0, "skipped": 0}
    for tc in ET.parse(xml).getroot().iter("testcase"):
        mod = tc.get("classname", "").split(".")[-1]
        status = "failed" if tc.find("failure") is not None or tc.find("error") is not None else (
            "skipped" if tc.find("skipped") is not None else "passed")
        tot[status] += 1
        if mod.startswith("test_") and tc.get("classname", "").startswith("tests.oracle"):
            parts = mod.split("_", 2)
            if len(parts) == 3 and status != "skipped":
                per[parts[1]][parts[2]][1] += 1
                per[parts[1]][parts[2]][0] += status == "passed"
    routines = {v: {"ported": len(r), "passing": sum(1 for p, t in r.values() if p == t)} for v, r in per.items()}
    return {"tests": tot, "routines": routines, "test_seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="synthe,odf,atlas,api")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args()
    want = set(a.stages.split(","))
    ref = json.loads((REPO / "reference/reference_set.json").read_text())
    tolf = REPO / "results/tolerances.json"
    tols = json.loads(tolf.read_text())["stages"] if tolf.exists() else {}
    from atlas_torch import pipeline  # torch stage runners
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": a.note,
           "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip(),
           "dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True).strip())}
    if not a.no_tests:
        rec.update(run_tests())
    stages = {}
    for stage, fn, sdir, _ in stage_pairs(ref):
        kind = stage.split("/")[0]
        if kind not in want:
            continue
        runner = pipeline.STAGES.get(kind)
        if runner is None:
            stages[stage] = {"status": "not_implemented"}
            continue
        t0 = time.time()
        r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            cand = runner(stage, ref)
        except NotImplementedError as e:
            stages[stage] = {"status": "not_implemented", "detail": str(e)}
            continue
        except Exception as e:  # noqa: BLE001  (a crash is a failing score, recorded)
            stages[stage] = {"status": "crash", "detail": f"{type(e).__name__}: {e}"}
            continue
        m = fn(sdir, cand)
        tol = tols.get(stage, {})
        fails = {k: (m.get(k), t["tol"]) for k, t in tol.items() if not (m.get(k, float("inf")) <= t["tol"])}
        stages[stage] = {"status": "pass" if (tol and not fails and "error" not in m) else "fail",
                         "metrics": m, "fails": fails, "seconds": round(time.time() - t0, 1),
                         "peak_rss_mb": round(max(r0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024, 1)}
    if "api" in want:
        from atlas_torch.scoring import atlas_metrics, odf_metrics, spectrum_batches_metrics
        fns = {"structure": atlas_metrics, "spectrum": spectrum_batches_metrics, "odf": odf_metrics}
        t0 = time.time()
        try:
            homes = pipeline.STAGES["api"](ref)          # runs the example notebooks with the torch backend
            err = None
        except NotImplementedError as e:
            homes, err = None, ("not_implemented", str(e))
        except Exception as e:  # noqa: BLE001
            homes, err = None, ("crash", f"{type(e).__name__}: {e}")
        for ex, prods in EXAMPLE_PRODUCTS.items():
            for prod, rel in prods.items():
                stage = f"api/{ex}/{prod}"
                if err:
                    stages[stage] = {"status": err[0], "detail": err[1]}
                    continue
                sdir = EXT / "runs/stock/examples" / ex / rel
                m = fns[prod](sdir, homes[ex] / rel)
                tol = tols.get(stage, {})
                fails = {k: (m.get(k), t["tol"]) for k, t in tol.items() if not (m.get(k, float("inf")) <= t["tol"])}
                stages[stage] = {"status": "pass" if (tol and not fails and "error" not in m) else "fail",
                                 "metrics": m, "fails": fails, "seconds": round(time.time() - t0, 1)}
    rec["stages"] = stages
    rec["summary"] = {s: sum(1 for v in stages.values() if v["status"] == s) for s in ("pass", "fail", "crash", "not_implemented")}
    with open(REPO / "results/scoreboard.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    import results_md
    results_md.write(REPO, rec, tols)
    print(json.dumps({k: rec[k] for k in rec if k != "stages"}, indent=1))
    for s, v in stages.items():
        line = f"{s:28s} {v['status']:16s}"
        if "fails" in v and v["fails"]:
            k = next(iter(v["fails"]))
            line += f" e.g. {k}={v['fails'][k][0]:.3e} > tol {v['fails'][k][1]:.3e}"
        print(line)


if __name__ == "__main__":
    main()
