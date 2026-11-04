"""HIGH (atlas9mem.for): golden traced calls (three iterations) and iteration states vs the Fortran."""
import numpy as np

from atlas_torch.atlas9.high import high
from atlas_torch.state import State
from atlas_torch.testing.atlas9_oracle_physics import iteration_snapshots, run_fortran

STAGE = "sun_atlas9/atlas9"


def test_high_golden(golden):
    for hit in range(3):
        e, x = golden(STAGE, "high", hit, "entry"), golden(STAGE, "high", hit, "exit")
        n = int(e["rhox.nrhox"][0])
        assert np.array_equal(high(State.from_trace(e))["height.height"][:n].numpy(), x["height.height"][:n])


def test_high_vs_fortran(oracle):
    snaps = iteration_snapshots(2)
    for it in (1, 2):
        s = snaps[(it, "high")]
        _, ref = run_fortran(oracle("atlas9"), s, "high")
        n = s.iscalar("rhox.nrhox")
        assert np.array_equal(high(s)["height.height"][:n].numpy(), ref["height.height"][:n].numpy())
