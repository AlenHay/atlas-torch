"""Stage test: 15 ATLAS9 iterations of the torch port (Fortran physics through the test-only oracle adapter)
from the traced solar run's exact inputs (BasicATLAS control cards, kappa.ros, p00big2.bdf ODF, molecules.dat,
restart deck), compared with

  1. the stock BasicATLAS atlas9mem.exe rerun on this machine in a scratch copy of the run directory: same
     libm as the port's elementary functions -> the iteration tables (stdout) and the punched models
     (fort.7 = output_summary.out) must agree character for character, and the program must end the same way
     (control cards exhausted -> gfortran 'End of file' runtime error, status 2);
  2. the gdb-traced run (traces/sun_atlas9/atlas9/gdb.log), produced in the atlas-dev container (other glibc:
     its exp/log differ in the last ulp for a few arguments, which the iteration amplifies): agreement to the
     cross-environment noise floor of the Fortran itself, measured here as stock-vs-traced.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.stage

COLS = ("j", "rhox", "t", "p", "xne", "rho", "abross", "height", "tauros", "flxcnv", "pradk", "flxerr", "flxdrv")


def tables(text: str) -> list[np.ndarray]:
    """The iteration tables of a stdout (rows of the 13 columns), stderr lines interleaved are ignored."""
    out, cur = [], None
    for ln in text.split("\n"):
        if ln.startswith(" START TABLE"):
            cur = []
        elif ln.startswith(" END TABLE"):
            out.append(np.array(cur))
            cur = None
        elif cur is not None and re.match(r"^\s*\d+\s+[-0-9.]+E[-+]\d\d", ln):
            cur.append([float(v) for v in ln.split()])
    return out


def table_text(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if ln.strip() and not ln.startswith(("At line", "Fortran runtime",
                                                                              "Error termination", "#"))]


@pytest.fixture(scope="module")
def run_dir(ext):
    return Path(ext) / "traces/sun_atlas9/w"


@pytest.fixture(scope="module")
def stock(ext, run_dir, tmp_path_factory):
    exe = Path(ext) / "BasicATLAS/bin/atlas9mem.exe"
    if not exe.exists():
        pytest.skip("stock atlas9mem.exe not built")
    d = tmp_path_factory.mktemp("atlas9_stock")
    for u in ("1", "2", "9"):
        os.symlink(os.path.realpath(run_dir / f"fort.{u}"), d / f"fort.{u}")
    shutil.copy(run_dir / "restart.dat", d / "fort.3")
    with open(run_dir / "stdin.txt") as fi, open(d / "output_main.out", "w") as fo:
        p = subprocess.run([str(exe)], stdin=fi, stdout=fo, stderr=subprocess.PIPE, cwd=d, timeout=1800)
    return {"stdout": (d / "output_main.out").read_text(), "punch": (d / "fort.7").read_text(),
            "stderr": p.stderr.decode(), "code": p.returncode}


@pytest.fixture(scope="module")
def port(run_dir):
    from atlas_torch.atlas9.main import run
    from atlas_torch.testing.atlas9_oracle_physics import OraclePhysics
    torch.set_default_dtype(torch.float64)
    files = {1: run_dir / "fort.1", 2: run_dir / "fort.2", 3: run_dir / "fort.3", 9: run_dir / "fort.9"}
    r = run((run_dir / "stdin.txt").read_text(), files, OraclePhysics())
    return {"stdout": "\n".join(r.stdout) + "\n", "punch": "\n".join(r.punch) + "\n", "message": r.exit_message,
            "code": r.exit_code, "states": r.states}


def _report(name, a_tabs, b_tabs):
    lines = []
    worst = {}
    for it, (a, b) in enumerate(zip(a_tabs, b_tabs), 1):
        row = []
        for k in ("t", "p", "xne", "abross"):
            i = COLS.index(k)
            r = float(np.max(np.abs(a[:, i] - b[:, i]) / np.abs(b[:, i])))
            worst[k] = max(worst.get(k, 0.0), r)
            row.append(f"{k}={r:.2e}")
        i = COLS.index("flxerr")
        d = float(np.max(np.abs(a[:, i] - b[:, i])))
        worst["flxerr_abs"] = max(worst.get("flxerr_abs", 0.0), d)
        row.append(f"flxerr_abs={d:.2e}")
        lines.append(f"{name} iter {it:2d}: " + " ".join(row))
    print("\n".join(lines))
    return worst


def test_iteration_tables_equal_stock(port, stock):
    assert port["code"] == stock["code"] == 2 and "End of file" in stock["stderr"]
    assert port["message"] == "Fortran runtime error: End of file"
    pt, st = tables(port["stdout"]), tables(stock["stdout"])
    assert len(pt) == len(st) == 15
    _report("port-vs-stock", pt, st)
    assert table_text(port["stdout"]) == table_text(stock["stdout"])


def test_punch_equal_stock(port, stock):
    assert port["punch"].split("\n") == stock["punch"].split("\n")


def test_vs_traced_run_within_fortran_noise_floor(port, stock, ext):
    """The traced (container) run differs from the host stock run by libm ulps amplified over 15 iterations;
    the port must be at least as close to it as the Fortran itself is."""
    ref = tables((Path(ext) / "traces/sun_atlas9/atlas9/gdb.log").read_text())
    pt, st = tables(port["stdout"]), tables(stock["stdout"])
    assert len(ref) == 15
    floor = _report("stock-vs-traced", st, ref)
    mine = _report("port-vs-traced", pt, ref)
    for k, v in mine.items():
        assert v <= max(floor[k], 1e-5 if k != "flxerr_abs" else 1e-3) * 1.000001, (k, v, floor[k])
