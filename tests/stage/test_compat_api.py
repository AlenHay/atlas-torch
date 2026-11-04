"""Stage test of the BasicATLAS-compatible API (atlas_torch.compat.atlas) with the Fortran backend.

The same driver script runs once against BasicATLAS's own ``atlas`` module (pinned checkout on sys.path) and once
against ours (``src/atlas_torch/compat`` on sys.path, ATLAS_TORCH_BACKEND=fortran), each in its own subprocess
and run directory under /tmp/claude-1000/. Both must produce identical run directories (every file byte for byte;
command files after replacing the run root), identical return values of the readers (read_structure, meta,
read_spectrum, synphot, ...) and identical printed messages (timing lines excepted).

Scenario: ATLAS Sun (default Settings, restart='auto'); 10 nm SYNTHE (500-510 nm); a second ATLAS model restarted
from the first run directory (teff 6000, logg 4.0, niter=20, molecules off); a SYNTHE run on it in two buffer
batches with abun_adjust, C12C13 and air wavelengths; file and gray restarts; synphot; error messages.
DFSYNTHE (an hour of CPU) is not run: with stub programs on both sides only its command files are compared.
"""
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
BA = Path(os.environ.get("ATLAS_TORCH_BASICATLAS", "/home/deck/atlas-ext/BasicATLAS")).resolve()
SCRATCH = Path("/tmp/claude-1000")

pytestmark = [pytest.mark.stage,
              pytest.mark.skipif(not (BA / "atlas.py").is_file(), reason="BasicATLAS checkout not found")]

PRELUDE = textwrap.dedent("""
    import os, sys, pickle, traceback
    import numpy as np
    side, root = sys.argv[1], os.path.realpath(sys.argv[2])
    # BasicATLAS predates numpy 2 (np.alltrue / np.trapz); give its code the old names back
    if not hasattr(np, "alltrue"): np.alltrue = np.all
    if not hasattr(np, "trapz"): np.trapz = np.trapezoid
    if side == "basicatlas":
        sys.path.insert(0, {ba!r})
    else:
        sys.path.insert(0, {compat!r}); sys.path.insert(1, {src!r})
    import atlas
    import matplotlib; matplotlib.use("Agg")
    results = {{}}
    def norm(x):
        return x.replace(root, "<ROOT>") if isinstance(x, str) else x
    def attempt(key, fn):
        try:
            results[key] = ("ok", fn())
        except Exception as e:
            results[key] = ("error", type(e).__name__, norm(str(e)))
    def p(*parts):
        return os.path.join(root, *parts)
""")

DRIVER_RUNS = """
s = atlas.Settings()
atlas.atlas(p("sun"))
results["sun.structure"] = atlas.read_structure(p("sun"))
results["sun.meta"] = atlas.meta(p("sun"))
attempt("sun.converged", lambda: atlas.atlas_converged(p("sun"), True, False, 450))
attempt("sun.exists", lambda: atlas.atlas(p("sun")))
attempt("sun.synthe_before_atlas", lambda: atlas.synthe(root, 500, 510))

atlas.synthe(p("sun"), 500, 510, progress=False)
results["sun.spectrum"] = atlas.read_spectrum(p("sun"))
results["sun.spectrum_binned"] = atlas.read_spectrum(p("sun"), num_bins=1000)
results["sun.meta_synthe"] = atlas.meta(p("sun"))
results["sun.validate"] = atlas.validate_run(p("sun"))
attempt("sun.synthe_again", lambda: atlas.synthe(p("sun"), 500, 510))
results["sun.binary"] = atlas.load_binary_spectrum(p("sun", "synthe_1", "spectrum.bin"))

hot = atlas.Settings(); hot.teff = 6000; hot.logg = 4.0
atlas.atlas(p("hot"), hot, restart=p("sun"), niter=20, molecules=False)
results["hot.structure"] = atlas.read_structure(p("hot"))
results["hot.meta"] = atlas.meta(p("hot"))
atlas.synthe(p("hot"), 600, 603, res=300000, vturb=2.0, abun_adjust={"Fe": 0.3, "C": -0.2, "Li": 1.0},
             C12C13=20, air_wl=True, buffsize=1200, progress=True)
results["hot.spectrum"] = atlas.read_spectrum(p("hot"))
results["hot.meta_synthe"] = atlas.meta(p("hot"))

results["restart.file"] = (norm(atlas.restarts.prepare_restart(p("sun", "output_summary.out"), p("restart_file.dat"), hot)),
                           open(p("restart_file.dat")).read())
results["restart.gray"] = (atlas.restarts.prepare_restart("gray", p("restart_gray.dat"), hot),
                           open(p("restart_gray.dat")).read())

os.mkdir(p("filters"))
wl = np.linspace(5010, 5090, 81)
np.savetxt(p("filters", "inside.dat"), np.c_[wl, np.exp(-((wl - 5050) / 20) ** 2)])
np.savetxt(p("filters", "spill.dat"), np.c_[wl + 50, np.ones(81)])
spec = dict(results["sun.spectrum"], teff=5770.0)
srt = lambda d: dict(sorted(d.items()))
attempt("synphot.ab", lambda: srt(atlas.synphot(None, "ABMAG", spectrum=spec, filters_dir=p("filters"))))
attempt("synphot.vega", lambda: srt(atlas.synphot(p("sun"), "VEGAMAG", filters_dir=p("filters"))))
os.mkdir(p("filters_in"))
np.savetxt(p("filters_in", "inside.dat"), np.c_[wl, np.exp(-((wl - 5050) / 20) ** 2)])
attempt("synphot.array", lambda: srt(atlas.synphot(None, "ABMAG", reddening=np.zeros(3), spectrum=spec,
                                                    filters_dir=p("filters_in"))))
# BasicATLAS raises UnboundLocalError here when the first filter listed is skipped (filesystem order); we don't
attempt("synphot.array_mixed", lambda: srt(atlas.synphot(None, "ABMAG", reddening=np.zeros(3), spectrum=spec,
                                                          filters_dir=p("filters"))))
attempt("synphot.bad", lambda: atlas.synphot(None, "STMAG", spectrum=spec, filters_dir=p("filters")))
results["synbeg"] = [atlas.synbeg(a, b, r) for a, b, r in [(500, 510, 600000), (100, 3000, 1e5), (300.3, 300.4, 5e5)]]
results["bin_spec"] = atlas.bin_spec(np.linspace(0, 1, 1001), np.linspace(0, 1, 1001) ** 2, 37)
results["blackbody"] = [f(*a) for f, a in [(atlas.blackbody, (1e14, 5000)), (atlas.blackbody, (1e17, 300)),
                        (atlas.blackbody_dBdT, (1e14, 5000)), (atlas.blackbody_peak, (5000,)),
                        (atlas.blackbody_dBdT_peak, (5000,))]]
attempt("meta.missing", lambda: atlas.meta(p("nothing")))
attempt("meta.unknown", lambda: atlas.meta(p("filters")))
attempt("structure.missing", lambda: atlas.read_structure(p("filters")))
attempt("parse.classic", lambda: atlas.parse_atlas_abundances(p("hot", "atlas_control.com"), lookbehind=4,
                                                               params=["VTURB", "SCALE 72/3", "SCALE 72/4"], to_float=False))
attempt("parse.printed", lambda: atlas.parse_atlas_abundances(p("hot", "output_main.out"), classic_style=False))
pickle.dump(results, open(p("results.pkl"), "wb"))
"""

DRIVER_DFSYNTHE = """
captured = {}
def capture(path):
    captured[norm(os.path.realpath(path))] = norm(open(path).read())
    return path

if side == "basicatlas":
    import re
    def fake_cmd(command):
        path = capture(command.split(" ", 1)[1])
        text = open(path).read()
        d = os.path.dirname(path)
        if path.endswith("xnfdf.com"):
            for f in ("xnfpdf.dat", "xnfpdfmax.dat"):
                open(os.path.join(d, f), "w").close()
        for f in re.findall(r"mv fort\\.[23] (p00(?:big|lit)[0-9]\\.bdf)", text):
            open(os.path.join(d, f), "w").close()
    atlas.cmd = fake_cmd
else:
    from atlas_torch.compat import runners
    outputs = {"xnfdf": (10, 11), "dfsynthe": (15, 16, 17, 18, 20), "dfsortp": (2,), "separatedf": (2, 3),
               "kappa9": (7,), "kapreadts": (2,)}
    def fake_program(name, workdir, stdin_text, unit_files):
        for u in outputs[name]:
            open(os.path.join(workdir, "fort.%d" % u), "w").close()
        return ""
    runners.run_program = fake_program
    real_bash = atlas._bash
    atlas._bash = lambda path: real_bash(capture(path))

s = atlas.Settings()
s.teff, s.logg, s.zscale = 4286, 1.66, -0.52
s.abun = {"C": 0.43, "O": 0.50, "Na": 0.11, "Mg": 0.37, "Al": 0.34, "Si": 0.33, "K": 0.20, "Ca": 0.11, "Sc": 0.23,
          "Ti": 0.23, "V": 0.20, "Cr": -0.05, "Mn": -0.21, "Co": 0.09, "Ni": 0.06, "Zn": 0.22, "U": -30.0}
attempt("dfsynthe", lambda: atlas.dfsynthe(p("odf"), s, parallel=(side != "basicatlas")))
results["captured"] = captured
attempt("dfsynthe.exists", lambda: atlas.dfsynthe(p("odf"), s))
pickle.dump(results, open(p("results.pkl"), "wb"))
"""


def _driver(body):
    return PRELUDE.format(ba=str(BA), compat=str(REPO / "src" / "atlas_torch" / "compat"), src=str(REPO / "src")) + body


def _launch(side, root, body):
    root.mkdir(parents=True)
    script = root.parent / "driver_{}_{}.py".format(root.name, side)
    script.write_text(_driver(body))
    env = dict(os.environ, ATLAS_TORCH_BACKEND="fortran", MPLBACKEND="Agg")
    log = open(root.parent / "{}.log".format(root.name), "wb")
    proc = subprocess.Popen([sys.executable, str(script), side, str(root)], stdout=subprocess.PIPE,
                            stderr=log, env=env, cwd=str(root.parent))
    return proc


def _finish(proc, timeout=3600):
    out, _ = proc.communicate(timeout=timeout)
    assert proc.returncode == 0, "driver failed, see logs next to it:\n" + out.decode()[-3000:]
    return out.decode()


def _run_both(body, name):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="compat_api_{}_".format(name), dir=SCRATCH)).resolve()
    roots = {"basicatlas": base / "basicatlas", "torch": base / "ours"}
    procs = {side: _launch(side, root, body) for side, root in roots.items()}
    stdout = {side: _finish(proc) for side, proc in procs.items()}
    results = {side: pickle.loads((root / "results.pkl").read_bytes()) for side, root in roots.items()}
    return base, roots, stdout, results


def same(a, b, path="result"):
    """Strict structural equality (types, dict order, exact floats; NaN == NaN)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a.keys()) == list(b.keys()), path
        for k in a:
            same(a[k], b[k], "{}[{!r}]".format(path, k))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, "{}[{}]".format(path, i))
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape, path
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), path
    elif isinstance(a, (float, np.floating)) and np.isnan(a):
        assert isinstance(b, (float, np.floating)) and np.isnan(b), path
    else:
        assert type(a) is type(b) and a == b, "{}: {!r} != {!r}".format(path, a, b)


def _messages(text, root):
    return [ln.replace(str(root), "<ROOT>") for ln in text.splitlines() if "Finished running" not in ln]


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.name != "results.pkl")


@pytest.fixture(scope="module")
def runs():
    base, roots, stdout, results = _run_both(DRIVER_RUNS, "runs")
    yield roots, stdout, results
    shutil.rmtree(base, ignore_errors=True)


def test_run_directories_identical(runs):
    roots, _, _ = runs
    a, b = roots["basicatlas"], roots["torch"]
    assert _tree(a) == _tree(b)
    for rel in _tree(a):
        pa, pb = a / rel, b / rel
        assert pa.is_dir() == pb.is_dir() and pa.is_symlink() == pb.is_symlink(), rel
        if pa.is_dir():
            continue
        da, db = pa.read_bytes(), pb.read_bytes()
        if rel.endswith(".com"):          # command files name the run directory
            da, db = da.replace(bytes(a), b"<ROOT>"), db.replace(bytes(b), b"<ROOT>")
        assert da == db, rel


@pytest.mark.parametrize("rel", ["sun/output_summary.out", "sun/output_main.out", "sun/output_last_iteration.out",
                                 "sun/restart.dat", "sun/synthe_1/spectrum.bin", "hot/output_summary.out",
                                 "hot/output_main.out", "hot/synthe_2/spectrum.bin"])
def test_key_outputs_byte_identical(runs, rel):
    roots, _, _ = runs
    assert (roots["basicatlas"] / rel).read_bytes() == (roots["torch"] / rel).read_bytes()


def test_api_results_identical(runs):
    _, _, results = runs
    a, b = results["basicatlas"], results["torch"]
    assert list(a) == list(b)
    # known BasicATLAS bug (see driver): accept its crash, require the intended result from us
    mixed_a, mixed_b = a.pop("synphot.array_mixed"), b.pop("synphot.array_mixed")
    if mixed_a[0] == "error":
        assert mixed_a[1] == "UnboundLocalError"
    else:
        same(mixed_a, mixed_b, "synphot.array_mixed")
    assert mixed_b[0] == "ok" and mixed_b[1]["inside.dat"].shape == (3,) and np.isnan(mixed_b[1]["spill.dat"]).all()
    same(mixed_b[1]["inside.dat"], b["synphot.array"][1]["inside.dat"])
    for key in a:
        same(a[key], b[key], key)
    assert a["sun.meta"]["type"] == "ATLAS" and a["sun.meta_synthe"]["type"] == "SYNTHE"
    assert a["hot.meta_synthe"]["medium"] == "air" and a["hot.meta_synthe"]["abun_adjust"]


def test_printed_messages_identical(runs):
    roots, stdout, _ = runs
    assert _messages(stdout["basicatlas"], roots["basicatlas"]) == _messages(stdout["torch"], roots["torch"])


def test_dfsynthe_command_files():
    base, roots, stdout, results = _run_both(DRIVER_DFSYNTHE, "dfsynthe")
    a, b = results["basicatlas"], results["torch"]
    ca, cb = a.pop("captured"), b.pop("captured")
    # 1 xnfdf + 57 DFSYNTHE temperatures + 5 SEPARATEDF + 5 KAPPA9 + KAPREADTS; separatedf.com is rewritten per
    # velocity, the last one is compared here
    assert sorted(ca) == sorted(cb) and len(ca) == 1 + 57 + 1 + 5 + 1
    for k in ca:
        assert ca[k] == cb[k], k
    assert a["dfsynthe.exists"] == b["dfsynthe.exists"]
    assert a["dfsynthe.exists"][0] == "error"
    # BasicATLAS stops at validate_run (no real program output); so do we
    assert a["dfsynthe"][0] == b["dfsynthe"][0] == "error"
    ma = _messages(stdout["basicatlas"], roots["basicatlas"])
    mb = _messages(stdout["torch"], roots["torch"])
    # parallel=True on our side: the per-temperature lines may come out of order
    assert sorted(ma) == sorted(mb)
    shutil.rmtree(base, ignore_errors=True)
