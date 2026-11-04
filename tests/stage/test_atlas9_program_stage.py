"""Stage tests of the torch ``atlas9mem`` program (torch physics, no Fortran in the loop) against the stock
BasicATLAS executable run here (same container libm):

  1. the program interface on the traced solar run's inputs (kappa.ros, molecules.dat, restart deck, p00big2 ODF)
     with BasicATLAS's control cards cut to two ITERATIONS batches (2 + 1 iterations) and no END card: stdout
     (output_main.out) and fort.7 must be identical character for character, and both programs must end on the
     exhausted control cards (Fortran: runtime error, status 2; runner: ValueError after writing fort.7);
  2. the BasicATLAS-compatible API end to end (compat.atlas.atlas(), ATLAS_TORCH_BACKEND=torch) for the Sun with
     the reference-set settings, against the stock reference run: the run directory's products identical.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.stage

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
REPO = Path(__file__).resolve().parents[2]
W = EXT / "traces/sun_atlas9/w"
EXE = EXT / "BasicATLAS/bin/atlas9mem.exe"


def _stdin_batches(batches):
    text = (W / "stdin.txt").read_text()
    head = text[:text.index("ITERATIONS")]
    body = "".join("ITERATIONS {} PRINT".format(n) + " 1" * 15 + "\nPUNCH" + " 1" * 15 + "\n"
                   + "BEGIN" + " " * 20 + "ITERATION  10 COMPLETED\n" for n in batches)
    return head + body


def _setup(d):
    for u in ("1", "2", "9"):
        os.symlink(os.path.realpath(W / f"fort.{u}"), d / f"fort.{u}")
    shutil.copy(W / "restart.dat", d / "fort.3")


@pytest.fixture(scope="module")
def stdin_text():
    if not (W / "stdin.txt").exists():
        pytest.skip("traced solar ATLAS9 run not found")
    return _stdin_batches([2, 1])


@pytest.fixture(scope="module")
def stock(stdin_text, tmp_path_factory):
    if not EXE.exists():
        pytest.skip("stock atlas9mem.exe not built")
    d = tmp_path_factory.mktemp("atlas9_stock")
    _setup(d)
    p = subprocess.run([str(EXE)], input=stdin_text.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       cwd=d, timeout=600)
    return {"stdout": p.stdout.decode("latin-1"), "punch": (d / "fort.7").read_text(), "code": p.returncode,
            "stderr": p.stderr.decode()}


@pytest.fixture(scope="module")
def port(stdin_text, tmp_path_factory):
    from atlas_torch.atlas9.program import atlas9mem_program
    d = tmp_path_factory.mktemp("atlas9_torch")
    _setup(d)
    units = {u: d / f"fort.{u}" for u in (1, 2, 3, 9)}
    err = None
    try:
        atlas9mem_program(str(d), stdin_text, units)
    except ValueError as e:
        err = str(e)
    from atlas_torch.atlas9.program import atlas9mem
    stdout, punch, r = atlas9mem(stdin_text, units)
    return {"stdout": stdout, "punch": punch, "fort7": (d / "fort.7").read_text(), "error": err, "result": r}


def test_two_batches_stdout_identical(port, stock):
    assert stock["code"] == 2 and "End of file" in stock["stderr"]
    assert port["result"].exit_code == 2 and "End of file" in port["result"].exit_message
    assert port["stdout"].count(" START TABLE") == 3
    assert port["stdout"] == stock["stdout"]


def test_two_batches_punch_identical(port, stock):
    assert port["fort7"] == port["punch"] == stock["punch"]
    assert port["punch"].count("==========") == 3


def test_runner_raises_on_exhausted_cards(port):
    assert port["error"] is not None and "End of file" in port["error"]


def _ref_set():
    return json.loads((REPO / "reference/reference_set.json").read_text())


def test_compat_atlas_sun_identical_to_stock(tmp_path, monkeypatch):
    ref_run = EXT / "runs/stock/atlas/sun"
    odf = EXT / "runs/stock/odf/solar"
    if not (ref_run / "output_main.out").exists() or not (odf / "p00big2.bdf").exists():
        pytest.skip("stock reference runs not found")
    from atlas_torch.compat import atlas as api
    from atlas_torch import scoring
    ref = _ref_set()
    m = ref["models"]["sun"]
    s = api.Settings()
    s.zscale = ref["odfs"][m["odf"]]["zscale"]
    s.abun = dict(ref["odfs"][m["odf"]]["abun"])
    s.teff, s.logg, s.vturb = m["teff"], m["logg"], ref["atlas"]["vturb"]
    monkeypatch.setenv("ATLAS_TORCH_BACKEND", "torch")
    out = tmp_path / "sun"
    api.atlas(str(out), s, restart=ref["atlas"]["restart"], niter=ref["atlas"]["niter"], ODF=str(odf), silent=True)
    for f in ("output_main.out", "output_summary.out", "output_last_iteration.out", "restart.dat"):
        assert (out / f).read_text() == (ref_run / f).read_text(), f
    for f in ("atlas_control.com", "atlas_control_start.com", "atlas_control_end.com"):
        a = (out / f).read_text().replace(str(out.resolve()), "<RUN>")
        b = (ref_run / f).read_text().replace(str(ref_run), "<RUN>")
        assert re.sub(r"\S*/BasicATLAS", "<BA>", a) == re.sub(r"\S*/BasicATLAS", "<BA>", b), f
    met = scoring.atlas_metrics(ref_run, out)
    assert all(v == 0 for k, v in met.items() if k != "iterations_tables"), met
    assert not list(out.glob("fort.*"))
