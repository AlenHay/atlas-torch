"""CONVEC (atlas9mem.for): on iteration states of the solar run vs the Fortran CONVEC with the same Fortran
physics memory image (its four POPS calls run NMOLEC from the same SAVEd starting guesses), plus the golden
traced calls (traced in another build environment: compared to roundoff, see stage test for why)."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9.convec import convec
from atlas_torch.state import State
from atlas_torch.testing.atlas9_oracle_physics import OraclePhysics, iteration_snapshots, mem_restore, run_fortran

KEYS = ("conv.dltdlp", "conv.heatcp", "conv.dlrdlt", "conv.velsnd", "conv.grdadb", "conv.hscale", "conv.flxcnv",
        "conv.vconv", "conv.flxcnv0", "conv.flxcnv1", "temp.t", "temp.tk", "temp.tkev", "temp.tlog", "state.p",
        "state.xne", "state.xnatom", "state.rho", "edens.edens", "temp.itemp")


def _compare(s, mem, oracle):
    o = oracle("atlas9")
    mem_restore(o, mem)
    _, ref = run_fortran(o, s, "convec")
    ph = OraclePhysics()
    mem_restore(ph.o, mem)
    out = convec(s, ph)
    n = s.iscalar("rhox.nrhox")
    for k in KEYS:
        a, b = out[k].numpy().reshape(-1), ref[k].numpy().reshape(-1)
        m = n if a.size > 1 else 1
        assert np.array_equal(a[:m], b[:m]), k


def test_convec_vs_fortran(oracle):
    snaps = iteration_snapshots(2)
    for it in (1, 2):
        _compare(snaps[(it, "convec")], snaps[(it, "convec", "mem")], oracle)


@settings(max_examples=6, deadline=None)
@given(mix=st.sampled_from([0.5, 1.25, 2.0]), overwt=st.sampled_from([0.0, 0.5, 1.0]), ifconv=st.sampled_from([1, 0]))
def test_convec_options_vs_fortran(oracle, mix, overwt, ifconv):
    """Mixing length, overshooting weight (the HEIGHT-integral smoothing branch) and IFCONV=0 (one pass)."""
    snaps = iteration_snapshots(1)
    s = State(snaps[(1, "convec")])
    s.update({"conv.mixlth": torch.tensor([mix]), "conv.overwt": torch.tensor([overwt]),
              "conv.ifconv": torch.tensor([ifconv])})
    _compare(s, snaps[(1, "convec", "mem")], oracle)
