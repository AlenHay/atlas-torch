"""ROSS (atlas9mem.for): MODE 1/2 golden traced calls, MODE 3 on iteration states vs the Fortran."""
import numpy as np
import torch

from atlas_torch.atlas9.ross import ross3, ross_terms
from atlas_torch.testing.atlas9_oracle_physics import iteration_snapshots, run_fortran

STAGE = "sun_atlas9/atlas9"


def test_ross_golden(golden):
    for hit in (1, 2):
        e, x = golden(STAGE, "ross", hit, "entry"), golden(STAGE, "ross", hit, "exit")
        n = int(e["rhox.nrhox"][0])
        g = lambda k: torch.as_tensor(e[k][:n])
        inc = ross_terms(g("freq.bnu")[None], torch.as_tensor(e["freq.freq"][:1]), g("temp.hkt"), g("temp.t"),
                         g("freq.stim")[None], g("abtot.abtot")[None], torch.as_tensor(e["arg:rcowt"]),
                         int(e["freset.numnu"][0]))
        assert np.array_equal((g("abross.abross") + inc[0]).numpy(), x["abross.abross"][:n])


def test_ross_mode3_vs_fortran(oracle):
    snaps = iteration_snapshots(2)
    for it in (1, 2):
        s = snaps[(it, "ross3")]
        _, ref = run_fortran(oracle("atlas9"), s, "ross", 3, 0.0)
        out = ross3(s)
        n = s.iscalar("rhox.nrhox")
        for k in ("abross.abross", "abross.tauros"):
            assert np.array_equal(out[k].numpy()[:n], ref[k].numpy()[:n]), (it, k)
