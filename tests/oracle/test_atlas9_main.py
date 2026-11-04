"""ATLAS9 main program: the batched frequency integration (continuum + 12 ODF sub-steps per frequency, the
RESIDC early stop, the continuum remainder with the stale KNU(1), the order of the TCORR/RADIAP/ROSS(2)
accumulations) against the Fortran main loop replayed with the Fortran KAPP/JOSH/LINOP/TCORR/RADIAP/ROSS
via ctypes from the same state; and the hydrostatic step."""
import numpy as np
import torch

from atlas_torch.atlas9.main import frequency_integration, hydrostatic, linop
from atlas_torch.state import State
from atlas_torch.testing.atlas9_oracle_physics import (EXT, OraclePhysics, fortran_frequency_integration,
                                                        iteration_snapshots, mem_restore)

KEYS = ("flux.flxrad", "abross.abross", "pzero.raden", "rad.accrad", "pzero.pradk0", "save.tcorr.rdabh",
        "save.tcorr.rjmins", "save.tcorr.rdiagj", "save.radiap.h", "taushj.hnu", "taushj.taunu", "abtot.abtot",
        "pzero.knu", "rr.rjminsnu", "rr.rdiagjnu", "freq.freq", "freq.bnu")


def test_frequency_integration_vs_fortran(oracle):
    snaps = iteration_snapshots(2)
    odf = linop.read_odf(str(EXT / "traces/sun_atlas9/w/fort.9"))
    for it in (1, 2):
        s = snaps[(it, "frequencies")]
        o = oracle("atlas9")
        mem_restore(o, snaps[(it, "frequencies", "mem")])
        ref = fortran_frequency_integration(o, s)
        ph = OraclePhysics()
        mem_restore(ph.o, snaps[(it, "frequencies", "mem")])
        out = frequency_integration(s, ph, odf, [], [])
        n = s.iscalar("rhox.nrhox")
        for k in KEYS:
            a, b = out[k].numpy().reshape(-1), ref[k].numpy().reshape(-1)
            m = n if a.size > 1 else 1
            assert np.array_equal(a[:m], b[:m]), (it, k)


def test_hydrostatic_fallbacks():
    s = State(iteration_snapshots(1)[(1, "populations")])
    out = hydrostatic(s)
    n = s.iscalar("rhox.nrhox")
    grav = s.scalar("teff.grav")
    p = grav * s["rhox.rhox"][:n] - s["rad.prad"][:n] - s["turbpr.pturb"][:n] - s.scalar("pzero.pcon")
    assert np.array_equal(out["state.p"][:n].numpy(), p.numpy())
    # PRAD larger than the column weight at the top: falls back to PRAD-PRAD(1), then to P(J-1)
    pr = s["rad.prad"].clone(); pr[:3] = grav * s["rhox.rhox"][:3] * 2
    out = hydrostatic(s.updated({"rad.prad": pr}))
    assert (out["state.p"][:n] > 0).all()
