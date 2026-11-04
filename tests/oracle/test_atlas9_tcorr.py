"""TCORR (atlas9mem.for): MODE 1/2 golden traced calls; MODE 2 increments on perturbed inputs and MODE 3 on
iteration states of the solar run (torch ATLAS9 + Fortran physics) vs the Fortran subroutine."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9.tcorr import tcorr1, tcorr3, tcorr_terms
from atlas_torch.state import State
from atlas_torch.testing.atlas9_oracle_physics import iteration_snapshots, mem_restore, run_fortran

STAGE = "sun_atlas9/atlas9"
SAVED = ("rjmins", "rdabh", "rdiagj")


def _terms(e, rcowt=None):
    n = int(e["rhox.nrhox"][0])
    g = lambda k: torch.as_tensor(np.asarray(e[k])[:n])
    rc = torch.tensor([float(e["arg:rcowt"][0]) if rcowt is None else rcowt])
    return n, tcorr_terms(g("rhox.rhox"), g("abtot.abtot")[None], g("taushj.hnu")[None], g("taushj.jmins")[None],
                          g("taushj.taunu")[None], g("abtot.alpha")[None], g("freq.bnu")[None],
                          torch.as_tensor(e["freq.freq"][:1]), g("temp.hkt"), g("temp.t"), g("freq.stim")[None], rc,
                          float(e["teff.teff"][0]), int(e["freset.numnu"][0]), float(e["flux.flux"][0]))


def test_tcorr_golden_modes_1_2(golden):
    e, x = golden(STAGE, "tcorr", 0, "entry"), golden(STAGE, "tcorr", 0, "exit")
    assert int(e["arg:mode"][0]) == 1
    s = State.from_trace(e)
    out = tcorr1(s)
    n = int(e["rhox.nrhox"][0])
    assert np.array_equal(out["flux.flxrad"][:n].numpy(), x["flux.flxrad"][:n])
    for hit in (1, 2):
        e, x = golden(STAGE, "tcorr", hit, "entry"), golden(STAGE, "tcorr", hit, "exit")
        assert int(e["arg:mode"][0]) == 2
        n, t = _terms(e)
        for k, key in (("rdabh", "static:rdabh"), ("rjminsnu", "static:rjmins"), ("rdiagjnu", "static:rdiagj"),
                       ("flxrad", "flux.flxrad")):
            assert np.array_equal((torch.as_tensor(e[key][:n]) + t[k][0]).numpy(), x[key][:n]), (hit, k)
        assert np.array_equal(t["rjminsnu"][0].numpy(), x["rr.rjminsnu"][:n])
        assert np.array_equal(t["rdiagjnu"][0].numpy(), x["rr.rdiagjnu"][:n])


@settings(max_examples=40, deadline=None)
@given(hit=st.integers(1, 2), lrc=st.floats(8.0, 15.0), teff=st.sampled_from([None, 4000.0, 7000.0]),
       sc=st.floats(0.01, 100.0))
def test_tcorr_mode2_vs_fortran(oracle, golden, hit, lrc, teff, sc):
    e = dict(golden(STAGE, "tcorr", hit, "entry"))
    if teff is not None:
        e["teff.teff"] = np.array([teff])
    e["abtot.abtot"] = e["abtot.abtot"] * sc
    e["taushj.taunu"] = e["taushj.taunu"] * sc
    s = State.from_trace(e)
    o = oracle("atlas9")
    _, ref = run_fortran(o, s, "tcorr", 2, 10.0 ** lrc,
                         unit_statics={"tcorr": {k: torch.as_tensor(e[f"static:{k}"]) for k in SAVED}})
    n, t = _terms(e, 10.0 ** lrc)
    assert np.array_equal((torch.as_tensor(e["flux.flxrad"][:n]) + t["flxrad"][0]).numpy(), ref["flux.flxrad"][:n].numpy())
    for k, sk in (("rdabh", "rdabh"), ("rjminsnu", "rjmins"), ("rdiagjnu", "rdiagj")):
        assert np.array_equal((torch.as_tensor(e[f"static:{sk}"][:n]) + t[k][0]).numpy(),
                              ref[f"save.tcorr.{sk}"][:n].numpy()), k


def _mode3(o, s):
    st = {"tcorr": {k: s[f"save.tcorr.{k}"] for k in SAVED + ("ddel", "oldt1") if f"save.tcorr.{k}" in s}}
    return run_fortran(o, s, "tcorr", 3, 0.0, unit_statics=st)[1]


KEYS3 = ("temp.t", "temp.tk", "temp.hkt", "temp.tlog", "rhox.rhox", "state.p", "state.xne", "abross.abross",
         "abross.tauros", "rad.prad", "pzero.pradk", "turbpr.pturb", "flux.flxerr", "flux.flxdrv", "conv.flxcnv",
         "depart.bmin", "depart.bhyd")


def test_tcorr_mode3_vs_fortran(oracle):
    snaps = iteration_snapshots(2)
    for it in (1, 2):
        s = snaps[(it, "tcorr3")]
        o = oracle("atlas9")
        mem_restore(o, snaps[(it, "tcorr3", "mem")])
        ref = _mode3(o, s)
        out = tcorr3(s)
        n = s.iscalar("rhox.nrhox")
        for k in KEYS3:
            assert np.array_equal(out[k].numpy()[:n], ref[k].numpy()[:n]), (it, k)
        for k in ("oldt1", "ddel"):
            assert np.array_equal(out[f"save.tcorr.{k}"].numpy()[:n], ref[f"save.tcorr.{k}"].numpy()[:n]), (it, k)


@settings(max_examples=12, deadline=None)
@given(scale=st.floats(0.97, 1.03), ferr=st.floats(-0.05, 0.05), it=st.sampled_from([1, 2]), smooth=st.booleans())
def test_tcorr_mode3_perturbed_vs_fortran(oracle, scale, ferr, it, smooth):
    """Perturb the frequency integrals (flux error) and T before TCORR(3); optional SMOOTH card."""
    snaps = iteration_snapshots(2)
    s = State(snaps[(it, "tcorr3")])
    n = s.iscalar("rhox.nrhox")
    fr = s["flux.flxrad"].clone(); fr[:n] *= 1.0 + ferr * torch.linspace(-1, 1, n); s["flux.flxrad"] = fr
    t = s["temp.t"].clone(); t[:n] *= scale; s["temp.t"] = t
    if smooth:
        s.update({"tsmooth.j1smooth": torch.tensor([20]), "tsmooth.j2smooth": torch.tensor([30])})
    o = oracle("atlas9")
    mem_restore(o, snaps[(it, "tcorr3", "mem")])
    ref = _mode3(o, s)
    out = tcorr3(s)
    for k in KEYS3:
        assert np.array_equal(out[k].numpy()[:n], ref[k].numpy()[:n]), k
