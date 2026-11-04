"""TURB (atlas9mem.for): turbulent velocity law (TURBULENCE ON cards) vs the Fortran on iteration states.
Not traced (BasicATLAS runs use TURBULENCE OFF)."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9.turb import turb
from atlas_torch.state import State
from atlas_torch.testing.atlas9_oracle_physics import iteration_snapshots, run_fortran


@settings(max_examples=30, deadline=None)
@given(fdg=st.floats(0.0, 5.0), pw=st.floats(-1.0, 1.0), snd=st.floats(0.0, 1.0), con=st.floats(0.0, 3.0))
def test_turb_vs_fortran(oracle, fdg, pw, snd, con):
    s = State(iteration_snapshots(1)[(1, "putout5")])
    s.update({"turbpr.trbfdg": torch.tensor([fdg]), "turbpr.trbpow": torch.tensor([pw]),
              "turbpr.trbsnd": torch.tensor([snd]), "turbpr.trbcon": torch.tensor([con]),
              "turbpr.ifturb": torch.tensor([1])})
    _, ref = run_fortran(oracle("atlas9"), s, "turb")
    out = turb(s)
    n = s.iscalar("rhox.nrhox")
    for k in ("turbpr.vturb", "turbpr.pturb"):
        assert np.array_equal(out[k][:n].numpy(), ref[k][:n].numpy()), k
