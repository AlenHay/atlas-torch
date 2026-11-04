"""KAPP's own combination of /OPS/ into /OPTOT/ (atlas9mem.for loop 100), as used by the iteration machinery
(atlas_torch.atlas9.kapp.kapp_combine): golden traced KAPP(0) calls (sources as the Fortran left them in
/OPS/) and Fortran KAPP(0) at perturbed temperatures / frequencies (sources from the oracle library)."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9.kapp import OPS, kapp_combine
from atlas_torch.atlas9.main import frequencies
from atlas_torch.state import State
from atlas_torch.testing.atlas9_oracle_physics import OraclePhysics, iteration_snapshots, mem_restore

STAGE = "sun_atlas9/atlas9"
OPT = ("acont", "scont", "aline", "sline", "sigmac", "sigmal")


def test_kapp_combine_golden(golden):
    for hit in range(3):
        x = golden(STAGE, "kapp", hit, "exit")
        n = int(x["rhox.nrhox"][0])
        out = kapp_combine({k: torch.as_tensor(x[f"ops.{k}"][:n]) for k in OPS}, torch.as_tensor(x["freq.bnu"][:n]))
        for k in OPT:
            assert np.array_equal(out[k].numpy(), x[f"optot.{k}"][:n]), (hit, k)


@settings(max_examples=15, deadline=None)
@given(tsc=st.floats(0.9, 1.1), f=st.integers(0, 336))
def test_kapp_combine_vs_fortran(tsc, f):
    snaps = iteration_snapshots(1)
    s = State(snaps[(1, "frequencies")])
    n = s.iscalar("rhox.nrhox")
    t = s["temp.t"].clone(); t[:n] *= tsc; s["temp.t"] = t
    s["temp.itemp"] = s["temp.itemp"] + 1000          # fresh ITEMP: the sources recompute their T caches
    ph = OraclePhysics()
    mem_restore(ph.o, snaps[(1, "frequencies", "mem")])
    fq = {k: (v[f:f + 1] if k != "nu" else v) for k, v in frequencies(s).items()}
    ops = ph.continuum(s, fq)
    out = kapp_combine({k: v[0] for k, v in ops.items()}, fq["bnu"][0])
    opt = ph.o.common("optot")
    for k in OPT:
        assert np.array_equal(out[k].numpy(), opt[k][:n]), k
