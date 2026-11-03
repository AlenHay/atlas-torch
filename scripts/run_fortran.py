"""Run the Fortran BasicATLAS pipeline (reference outputs) for one build variant.

usage: python scripts/run_fortran.py <variant> <kind> <name> [--out DIR]
  kind = odf    : atlas.dfsynthe() for reference_set odfs[name]
  kind = atlas  : atlas.atlas() for reference_set models[name] (needs the matching ODF run of the same variant,
                  unless --odf-from VARIANT is given)
  kind = synthe : atlas.synthe() on the model run of `name`, for every window in reference_set synthe
Outputs go to $ATLAS_EXT/runs/<variant>/{odf,atlas}/<name>; a provenance.json is written next to each.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def tree(variant):
    return Path(subprocess.check_output([str(REPO / "scripts/make_tree.sh"), variant], text=True).strip())


def settings_for(atlas, spec, teff=None, logg=None, vturb=None):
    s = atlas.Settings()
    s.zscale = spec["zscale"]
    s.abun = dict(spec["abun"])
    if teff is not None:
        s.teff, s.logg = teff, logg
    if vturb is not None:
        s.vturb = vturb
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant"); ap.add_argument("kind"); ap.add_argument("name")
    ap.add_argument("--odf-from", default=None)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--model-from", default=None, help="synthe: copy the ATLAS model of this variant first")
    a = ap.parse_args()
    ref = json.loads((REPO / "reference/reference_set.json").read_text())
    t = tree(a.variant)
    sys.path.insert(0, str(t)); os.chdir(t)
    import atlas  # BasicATLAS (external, pinned)
    runs = EXT / "runs" / a.variant
    t0 = time.time()
    if a.kind == "odf":
        out = runs / "odf" / a.name
        if out.exists(): shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        atlas.dfsynthe(str(out), settings_for(atlas, ref["odfs"][a.name]), parallel=a.parallel, silent=False)
    elif a.kind == "atlas":
        m = ref["models"][a.name]
        odf = EXT / "runs" / (a.odf_from or a.variant) / "odf" / m["odf"]
        out = runs / "atlas" / a.name
        if out.exists(): shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        s = settings_for(atlas, ref["odfs"][m["odf"]], m["teff"], m["logg"], ref["atlas"]["vturb"])
        atlas.atlas(str(out), s, restart=ref["atlas"]["restart"], niter=ref["atlas"]["niter"], ODF=str(odf), silent=False)
    elif a.kind == "synthe":
        out = runs / "atlas" / a.name
        if a.model_from:
            src = EXT / "runs" / a.model_from / "atlas" / a.name
            if out.exists(): shutil.rmtree(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, out, ignore=shutil.ignore_patterns("synthe_*", "window_*"))
        sy = ref["synthe"]
        for i, (lo, hi) in enumerate(sy["windows_nm"]):
            atlas.synthe(str(out), lo, hi, res=sy["res"], vturb=sy["vturb"], overwrite_prev=True, silent=False, progress=False)
            dst = out / f"window_{i}"
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(out / "synthe_1", dst, ignore=shutil.ignore_patterns("rgfalllinesnew.out", "xnfpelsyn.out"))
    else:
        raise SystemExit("unknown kind")
    prov = {"variant": a.variant, "kind": a.kind, "name": a.name, "seconds": time.time() - t0,
            "build": json.loads((EXT / "builds" / a.variant / "build.json").read_text()) if a.variant != "stock" else
                     {"variant": "stock", "note": "BasicATLAS compile.com", "basicatlas_commit":
                      subprocess.check_output(["git", "-C", str(EXT / "BasicATLAS"), "rev-parse", "HEAD"], text=True).strip()},
            "odf_from": a.odf_from, "model_from": a.model_from, "command": " ".join(sys.argv), "atlas_torch_commit":
            subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()}
    (out / f"provenance_{a.kind}.json").write_text(json.dumps(prov, indent=1))
    print(f"done {a.kind} {a.name} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
