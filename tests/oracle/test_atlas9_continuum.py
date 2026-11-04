"""Entry points of the ATLAS9 physics library: continuum(state, freq) and the Physics-protocol adapter.
/FREQ/ derivation (FREQLG, EHVKT, STIM, BNU as the ATLAS9 main program computes them) vs the trace; the
adapter's [F, n] /OPS/ output vs per-frequency KAPP."""
import numpy as np
import torch

from atlas_torch.atlas9.physics import Atlas9Physics, continuum
from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics.opacity import OPS, Freq


def test_freq_derivation_matches_trace(golden):
    for hit in range(3):
        s = T.opacity_state(golden, "kapp", hit)
        n = s.iscalar("rhox.nrhox")
        fq = Freq.derive(s, n, s.scalar("freq.freq"))
        assert fq.freqlg == s.scalar("freq.freqlg")
        for k in ("ehvkt", "stim", "bnu"):
            assert T.max_rel(getattr(fq, k).numpy(), s["freq." + k][:n].numpy()) <= T.golden_tol(), k


def test_continuum_entry_point(golden):
    s = T.opacity_state(golden, "kapp", 0)
    x = golden(T.TRACE_STAGE, "kapp", 0, "exit")
    n = s.iscalar("rhox.nrhox")
    out = continuum(s, s.scalar("freq.freq"))           # uses the State's /FREQ/
    for k in ("acont", "scont", "sigmac", "aline", "sline", "sigmal"):
        assert T.max_rel(out["optot." + k][:n].numpy(), x["optot." + k][:n]) <= T.golden_tol(), k
    out2 = continuum(s, 2.0e15)                          # derives /FREQ/ for a new frequency
    exp = T.oracle_opacity(T.make_oracle("atlas9"), s.updated({k: v for k, v in out2.items() if k.startswith("freq.")}), "kapp")
    for k in exp:
        assert T.max_rel(out2["optot." + k][:n].numpy(), exp[k]) == 0.0, k


def test_physics_protocol_adapter(golden):
    s = T.opacity_state(golden, "kapp", 0)
    n = s.iscalar("rhox.nrhox")
    freqs = [3.0e13, 6.0e14, 2.5e15, 1.0e16]
    fqs = [Freq.derive(s, n, f) for f in freqs]
    fq = {"freq": torch.tensor(freqs), "freqlg": torch.tensor([f.freqlg for f in fqs]),
          "ehvkt": torch.stack([f.ehvkt for f in fqs]), "stim": torch.stack([f.stim for f in fqs]),
          "bnu": torch.stack([f.bnu for f in fqs])}
    ops = Atlas9Physics().continuum(s, fq)
    assert set(ops) == set(OPS) and ops["ahyd"].shape == (4, n)
    for i, f in enumerate(freqs):
        one = continuum(s, f)
        for k in OPS:
            assert torch.equal(ops[k][i], one["ops." + k][:n]), (f, k)
