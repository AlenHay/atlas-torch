"""RADIAP (atlas9mem.for): MODE 1/2 golden traced calls, MODE 3 on iteration states vs the Fortran."""
import numpy as np
import torch

from atlas_torch.atlas9.radiap import radiap3, radiap_terms
from atlas_torch.testing.atlas9_oracle_physics import iteration_snapshots, run_fortran

STAGE = "sun_atlas9/atlas9"


def test_radiap_golden(golden):
    e, x = golden(STAGE, "radiap", 0, "entry"), golden(STAGE, "radiap", 0, "exit")
    n = int(e["rhox.nrhox"][0])
    assert int(e["arg:mode"][0]) == 1
    assert not x["pzero.raden"][:n].any() and not x["rad.accrad"][:n].any() and x["pzero.pradk0"][0] == 0
    for hit in (1, 2):
        e, x = golden(STAGE, "radiap", hit, "entry"), golden(STAGE, "radiap", hit, "exit")
        g = lambda k: torch.as_tensor(e[k][:n])[None]
        t = radiap_terms(g("taushj.jnu"), g("taushj.hnu"), g("abtot.abtot"), torch.as_tensor(e["pzero.knu"][:1]),
                         torch.as_tensor(e["arg:rcowt"]))
        for k, key in (("raden", "pzero.raden"), ("accrad", "rad.accrad"), ("h", "static:h")):
            assert np.array_equal((torch.as_tensor(e[key][:n]) + t[k][0]).numpy(), x[key][:n]), (hit, k)
        assert float(e["pzero.pradk0"][0] + t["pradk0"][0]) == x["pzero.pradk0"][0]


def test_radiap_mode3_vs_fortran(oracle):
    snaps = iteration_snapshots(2)
    for it in (1, 2):
        s = snaps[(it, "radiap3")]
        _, ref = run_fortran(oracle("atlas9"), s, "radiap", 3, 0.0, unit_statics={"radiap": {"h": s["save.radiap.h"]}})
        out = radiap3(s)
        n = s.iscalar("rhox.nrhox")
        for k in ("pzero.raden", "rad.accrad", "rad.prad", "pzero.pradk", "pzero.pradk0"):
            assert np.array_equal(out[k].numpy().reshape(-1)[:n], ref[k].numpy().reshape(-1)[:n]), (it, k)
