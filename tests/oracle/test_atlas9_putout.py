"""PUTOUT (atlas9mem.for as patched by BasicATLAS), MODE 5: the iteration table on unit 6, captured from the
Fortran subroutine run on the same iteration states, must match the port's text exactly; plus the patch's stop
conditions. The punched model (unit 7) cannot be produced by the oracle library: it is built without
-std=legacy and gfortran's runtime then rejects FORMAT 552 ('IFOP20I2', missing comma); the punch text is
compared with the stock BasicATLAS executable in tests/stage/test_atlas9_core.py."""
import ctypes
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.atlas9 import putout
from atlas_torch.atlas9.errors import FortranExit
from atlas_torch.state import State
from atlas_torch.testing.atlas9_oracle_physics import OraclePhysics, iteration_snapshots


def _fortran_putout5(o, s):
    """Run PUTOUT(5) in a scratch directory with fd 1 redirected; returns (stdout lines, fort.7 lines)."""
    gf = ctypes.CDLL("libgfortran.so.5")
    ph = OraclePhysics.__new__(OraclePhysics)
    ph.o = o
    ph.push(s)
    o.saved("putout")["amaxerr"][0] = 0.0
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        # flush output that earlier Fortran calls (other tests, same process) left in libgfortran's unit-6
        # buffer, otherwise it lands in this capture
        for u in (6, 7):
            gf._gfortran_flush_i4(ctypes.byref(ctypes.c_int(u)))
        out = os.open(os.path.join(d, "stdout"), os.O_WRONLY | os.O_CREAT)
        saved = os.dup(1)
        os.dup2(out, 1)
        try:
            o.call("putout", 5)
            for u in (6, 7):
                gf._gfortran_flush_i4(ctypes.byref(ctypes.c_int(u)))
        finally:
            os.dup2(saved, 1)
            os.close(saved); os.close(out)
            os.chdir(old_cwd)
        so = Path(d, "stdout").read_text().split("\n")[:-1]
        f7 = Path(d, "fort.7").read_text().split("\n")[:-1] if Path(d, "fort.7").exists() else []
    return so, f7


def test_putout5_vs_fortran(oracle):
    snaps = iteration_snapshots(2)
    for it in (1, 2):
        s = State(snaps[(it, "putout5")])
        s["iter.ifpnch"] = torch.zeros_like(s["iter.ifpnch"])
        so, f7 = _fortran_putout5(oracle("atlas9"), s)
        mine_so, mine_f7 = [], []
        putout.mode5(s, mine_so, mine_f7)
        assert so and mine_so == so
        assert mine_f7 == f7 == []


def test_putout5_stops():
    s = State(iteration_snapshots(1)[(1, "putout5")])
    n = s.iscalar("rhox.nrhox")
    # gold tolerances at ITER=15
    s15 = State(s)
    s15.update({"iter.iter": torch.tensor([15]), "flux.flxerr": torch.zeros(99), "flux.flxdrv": torch.zeros(99)})
    with pytest.raises(FortranExit) as e:
        putout.mode5(s15, [], [])
    assert e.value.message == "REACHED GOLD TOLERANCES"
    # divergence
    d = torch.zeros(99); d[3] = -2e5
    s15.update({"flux.flxdrv": d})
    with pytest.raises(FortranExit) as e:
        putout.mode5(s15, [], [])
    assert e.value.message == "MODEL DIVERGED"
    # invalid structure (negative pressure) is detected before the table
    p = s["state.p"].clone(); p[5] = -1.0
    so = []
    with pytest.raises(FortranExit) as e:
        putout.mode5(s.updated({"state.p": p}), so, [])
    assert e.value.message == "INVALID STRUCTURE" and so[-1] == " INVALID STRUCTURE" and " START TABLE" not in so
