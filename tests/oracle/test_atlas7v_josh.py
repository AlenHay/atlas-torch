"""JOSH (atlas7v.for, SYNTHE variant): golden SPECTRV calls (entry -> exit, including the REAL*4 internals)
and batched property tests vs the Fortran routine over IFSCAT in {0,1}, IFSURF in {1,2}, scattering and
line-opacity scales from transparent to TAUNU(1) > 20 (the DERIV branch), and non-monotonic TAUNU."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.state import State
from atlas_torch.synthe.josh import josh, josh_batch, josh_tables
from atlas_torch.testing.compare import assert_close

STAGE = "sun_500_510/spectrv"


def test_josh_tables_match_compiled(golden):
    ent = golden(STAGE, "josh", 0, "entry")
    T = josh_tables()
    assert np.array_equal(T["coefj"].numpy(), ent["matxj.coefj"])
    assert np.array_equal(T["coefh"].numpy(), ent["matxh.coefh"])
    for k in ("ch", "ck", "xtau8"):
        assert np.array_equal(T[k].numpy(), ent[f"static:{k}"]), k


def test_josh_golden(golden):
    for hit in range(3):
        ent, ext = golden(STAGE, "josh", hit, "entry"), golden(STAGE, "josh", hit, "exit")
        ifscat, ifsurf = int(ent["arg:ifscat"][0]), int(ent["arg:ifsurf"][0])
        out = josh(State.from_trace(ent), ifscat, ifsurf)
        det = out.pop("_detail")
        for k, v in out.items():
            assert_close(v, ext[k], rtol=0, atol=0, name=f"{k}[{hit}]")
        assert "taushj.hnu" in out
        n = int(ent["rhox.nrhox"][0])
        for k in ("xs", "xsbar", "xalpha", "diag", "xsbar8", "xalpha8"):
            assert_close(det[k][0], ext[f"static:{k}"], rtol=0, atol=0, name=f"static:{k}[{hit}]")
        assert_close(det["snubar"][0], ext["static:snubar"][:n], rtol=0, atol=0, name="snubar")
        assert int(det["nsweep"][0]) == int(ext["static:l"][0])
        # everything JOSH does not write is unchanged
        for k in ("taushj.snu", "taushj.jnu", "taushj.jmins", "mus.surfi", "pzero.knu"):
            assert np.array_equal(ext[k], ent[k]), k


def _oracle_josh(o, base, rows, ifscat, ifsurf, angle):
    n = int(base["rhox.nrhox"][0])
    o.set_common("rhox", rhox=base["rhox.rhox"], nrhox=n)
    o.set_common("mus", angle=np.pad(angle, (0, 20 - len(angle))), nmu=len(angle))
    res = []
    for r in rows:
        o.set_common("optot", **{k: np.pad(v, (0, 99 - n)) for k, v in r.items()})
        o.call("josh", ifscat, ifsurf)
        t, a, m, p = o.common("taushj"), o.common("abtot"), o.common("mus"), o.common("pzero")
        res.append({"hnu": t["hnu"].copy(), "snu": t["snu"].copy(), "taunu": t["taunu"].copy(),
                    "abtot": a["abtot"].copy(), "alpha": a["alpha"].copy(), "surfi": m["surfi"].copy(),
                    "knu": p["knu"].copy()})
    return res


def _rows(base, rng, nrow, lg_line, lg_scat, wild):
    n = int(base["rhox.nrhox"][0])
    ac, sc = base["optot.acont"][:n], base["optot.scont"][:n]
    sg = base["optot.sigmac"][:n]
    rows = []
    for _ in range(nrow):
        prof = 10.0 ** (lg_line + rng.uniform(-1, 1) + (rng.uniform(-2, 2, n) if wild else 0.0))
        rows.append({"acont": ac * 10.0 ** rng.uniform(-0.5, 0.5), "scont": sc * rng.uniform(0.8, 1.2),
                     "aline": ac * prof, "sline": sc * rng.uniform(0.5, 1.5, n), "sigmac": sg * rng.uniform(0.5, 2),
                     "sigmal": ac * prof * 10.0 ** lg_scat * (rng.random() < 0.7)})
    return rows


def _check(base, rows, ref, ifscat, ifsurf, angle, tag):
    n = int(base["rhox.nrhox"][0])
    T = lambda k: torch.as_tensor(np.stack([r[k] for r in rows]))
    got = josh_batch(T("acont"), T("scont"), T("aline"), T("sline"), T("sigmac"), T("sigmal"),
                     torch.as_tensor(base["rhox.rhox"][:n]), ifscat, ifsurf, angle=torch.as_tensor(angle))
    for i, r in enumerate(ref):
        for k in ("abtot", "alpha", "taunu"):
            assert_close(got[k][i], r[k][:n], rtol=0, atol=0, name=f"{tag} {k}[{i}]")
        if ifsurf == 1:
            assert_close(got["hnu1"][i], r["hnu"][0], rtol=0, atol=0, name=f"{tag} hnu(1)[{i}]")
        else:
            assert_close(got["surfi"][i], r["surfi"][:len(angle)], rtol=0, atol=0, name=f"{tag} surfi[{i}]")
    return got


@settings(max_examples=60, deadline=None)
@given(ifscat=st.sampled_from([0, 1]), ifsurf=st.sampled_from([1, 2]), nmu=st.integers(1, 4),
       lg_line=st.floats(-4, 8.5), lg_scat=st.floats(-6, 1), wild=st.booleans(), seed=st.integers(0, 2**31 - 1))
def test_josh_vs_fortran(oracle, golden, ifscat, ifsurf, nmu, lg_line, lg_scat, wild, seed):
    rng = np.random.default_rng(seed)
    base = golden(STAGE, "josh", 0, "entry")
    angle = np.sort(rng.uniform(0.02, 1.0, nmu))[::-1].copy()
    rows = _rows(base, rng, 3, lg_line, lg_scat, wild)
    with np.errstate(all="ignore"):
        ref = _oracle_josh(oracle("atlas7v"), base, rows, ifscat, ifsurf, angle)
        _check(base, rows, ref, ifscat, ifsurf, angle, f"ifscat={ifscat} ifsurf={ifsurf}")


def test_josh_thick_branch(oracle, golden):
    """TAUNU(1) > XTAU8(51) = 20 (strong-line cores, e.g. Ca II H/K): MAXJ=1 DERIV iteration (label 401)."""
    base = golden(STAGE, "josh", 0, "entry")
    rng = np.random.default_rng(7)
    for ifsurf in (1, 2):
        rows = _rows(base, rng, 4, 9.5, -3, False) + _rows(base, rng, 2, 0.0, -3, False)
        angle = np.array([1.0, 0.5, 0.1])
        ref = _oracle_josh(oracle("atlas7v"), base, rows, 1, ifsurf, angle)
        got = _check(base, rows, ref, 1, ifsurf, angle, f"thick ifsurf={ifsurf}")
        assert got["rows_thick"].tolist() == [0, 1, 2, 3]
        if ifsurf == 1:
            n = int(base["rhox.nrhox"][0])
            for j, i in enumerate(got["rows_thick"].tolist()):
                assert_close(got["thick"]["snu"][j], ref[i]["snu"][:n], rtol=0, atol=0, name="snu")
                assert_close(got["thick"]["jnu"][j, 0] / 3.0, ref[i]["knu"][0], rtol=0, atol=0, name="knu")
