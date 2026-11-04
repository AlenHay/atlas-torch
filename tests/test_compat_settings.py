"""Fast tests of the BasicATLAS-compatible front end (atlas_torch.compat): Settings conversions and restarts
against BasicATLAS's own modules (loaded from the pinned checkout under private names), the command-file
interpreter and backend selection of the runner registry."""
import importlib.util
import os
import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings as hsettings, strategies as st

from atlas_torch.compat import restarts as my_restarts
from atlas_torch.compat import runners
from atlas_torch.compat.settings import Settings, python_path

BA = python_path
pytestmark = [pytest.mark.skipif(not os.path.isfile(os.path.join(BA, "settings.py")),
                                 reason="BasicATLAS checkout not found ($ATLAS_TORCH_BASICATLAS)"),
              pytest.mark.filterwarnings("ignore::RuntimeWarning")]   # Y = 1 etc. overflow on both sides


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, os.path.join(BA, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ba_settings():
    return _load("settings", "_basicatlas_settings")


@pytest.fixture(scope="module")
def ba_restarts():
    return _load("restarts", "_basicatlas_restarts")


SYMBOLS = list(Settings().abun_solar().keys())
METALS = [s for s in SYMBOLS if s not in ("H", "He")]
TOL = 1e-12


def close(a, b):
    return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(a)))


def outcome(fn):
    try:
        return "ok", fn()
    except ValueError as e:
        return "error", str(e)


abun_dicts = st.dictionaries(st.sampled_from(METALS), st.floats(-3.0, 3.0, allow_nan=False), max_size=12)
zscales = st.floats(-7.0, 2.5, allow_nan=False)
helium = st.one_of(st.just(-0.1), st.floats(0.0, 1.0), st.floats(-1.0, 2.0))
SETTINGS = hsettings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def make(cls, Y, zscale, abun):
    s = cls()
    s.zscale = zscale
    s.Y = Y
    s.abun = dict(abun)
    return s


# --------------------------------------------------------------------------------------------- Settings
def test_defaults_and_solar(ba_settings):
    mine, theirs = Settings(), ba_settings.Settings()
    for k in ("teff", "logg", "zscale", "Y", "vturb", "abun"):
        assert getattr(mine, k) == getattr(theirs, k)
    a, b = mine.abun_solar(), theirs.abun_solar()
    assert list(a.keys()) == list(b.keys()) and all(a[k] == b[k] for k in a)


@SETTINGS
@given(z=zscales)
def test_zscale_validation(ba_settings, z):
    def setter(cls):
        def f():
            s = cls()
            s.zscale = z
            return s.zscale
        return f
    assert outcome(setter(Settings)) == outcome(setter(ba_settings.Settings))


@SETTINGS
@given(Y=helium, z=st.floats(-4.0, 1.0), abun=abun_dicts)
def test_std_to_atlas_and_derived(ba_settings, Y, z, abun):
    mine, theirs = make(Settings, Y, z, abun), make(ba_settings.Settings, Y, z, abun)
    ra, rb = outcome(mine.atlas_abun), outcome(theirs.atlas_abun)
    assert ra[0] == rb[0]
    if ra[0] == "error":
        assert ra[1] == rb[1]
        return
    a, b = ra[1], rb[1]
    assert len(a) == len(b) == 100 and a[0] == b[0] == 0
    assert all(close(x, y) for x, y in zip(a[1:], b[1:]))
    assert all(close(x, y) for x, y in zip(mine.mass_fractions(), theirs.mass_fractions()))
    assert close(mine.effective_zscale(), theirs.effective_zscale())

    for rounding in (2, 7):
        sa = mine.abun_atlas_to_std(a, z, rounding=rounding)
        sb = theirs.abun_atlas_to_std(b, z, rounding=rounding)
        assert list(sa["abun"].keys()) == list(sb["abun"].keys())
        assert all(close(sa["abun"][k], sb["abun"][k]) for k in sa["abun"])
        assert close(sa["Y"], sb["Y"]) and sa["zscale"] == sb["zscale"]


@SETTINGS
@given(Y=helium, z=st.floats(-4.0, 1.0), abun=abun_dicts)
def test_control_cards_identical(ba_settings, Y, z, abun):
    """The quantities as they are written into the control files must be identical, not just close."""
    from atlas_torch.compat.atlas import _abundance_cards
    a = outcome(make(Settings, Y, z, abun).atlas_abun)
    b = outcome(make(ba_settings.Settings, Y, z, abun).atlas_abun)
    assert a[0] == b[0]
    if a[0] == "ok":
        assert _abundance_cards(10 ** z, a[1]) == _abundance_cards(10 ** z, b[1])


@SETTINGS
@given(el=st.lists(st.floats(-25.0, -1.0), min_size=97, max_size=97), h=st.floats(0.5, 0.99),
       z=st.floats(-3.0, 1.0), rounding=st.sampled_from([2, 5, 7]))
def test_atlas_to_std_arbitrary(ba_settings, el, h, z, rounding):
    elements = [0, h, 1.0 - h] + el
    a = outcome(lambda: Settings().abun_atlas_to_std(elements, z, rounding))
    b = outcome(lambda: ba_settings.Settings().abun_atlas_to_std(elements, z, rounding))
    assert a[0] == b[0]
    if a[0] == "error":
        assert a[1] == b[1]
        return
    assert list(a[1]["abun"]) == list(b[1]["abun"])
    assert all(close(a[1]["abun"][k], b[1]["abun"][k]) for k in a[1]["abun"])
    assert close(a[1]["Y"], b[1]["Y"])


@pytest.mark.parametrize("abun", [{"H": 0.1}, {"He": 0.1}, {"Xx": 0.1}, {"Fe": 0.1, "H": 1}])
def test_std_to_atlas_errors(ba_settings, abun):
    a = outcome(lambda: Settings().abun_std_to_atlas(abun=abun))
    b = outcome(lambda: ba_settings.Settings().abun_std_to_atlas(abun=abun))
    assert a == b and a[0] == "error"


@SETTINGS
@given(Y=st.one_of(st.just(-0.1), st.floats(0.2, 0.35)), z=st.floats(-3.0, 0.5), abun=abun_dicts,
       dz=st.sampled_from([0.0, 0.005, 0.02]), dY=st.sampled_from([0.0, 0.0005, 0.005]),
       dabun=st.dictionaries(st.sampled_from(METALS), st.sampled_from([0.005, 0.0101, 0.05, -25.0]), max_size=3))
def test_check_ODF(ba_settings, Y, z, abun, dz, dY, dabun):
    odf_settings = Settings()
    odf_settings.zscale, odf_settings.Y = z + dz, (Y + dY if Y > 0 else Y)
    odf_settings.abun = {k: abun.get(k, 0.0) + dabun.get(k, 0.0) for k in set(abun) | set(dabun)}
    if outcome(odf_settings.atlas_abun)[0] == "error":
        return
    mf = odf_settings.mass_fractions()
    odf = {"zscale": odf_settings.zscale, "Y": mf[1],
           "abun": Settings().abun_atlas_to_std(odf_settings.atlas_abun(), odf_settings.zscale)["abun"]}
    a = outcome(lambda: make(Settings, Y, z, abun).check_ODF(odf))
    b = outcome(lambda: make(ba_settings.Settings, Y, z, abun).check_ODF(odf))
    assert a == b


# --------------------------------------------------------------------------------------------- restarts
def _restart_text(mod, restart, settings, path, **kw):
    msg = mod.prepare_restart(restart, path, settings, **kw)
    with open(path) as f:
        return msg, f.read()


@hsettings(max_examples=25, deadline=None)
@given(teff=st.floats(2500, 60000), logg=st.floats(-1.0, 6.0), z=st.floats(-6.0, 1.0),
       abun=st.dictionaries(st.sampled_from(["Fe", "Mg", "C", "O", "Ti"]), st.floats(-1.5, 1.5), max_size=4),
       kind=st.sampled_from(["auto", "gray", "grey"]))
def test_prepare_restart(ba_restarts, tmp_path_factory, teff, logg, z, abun, kind):
    s = Settings()
    try:
        s.zscale = z
    except ValueError:
        return
    s.teff, s.logg, s.abun = teff, logg, abun
    d = tmp_path_factory.mktemp("restart")
    assert _restart_text(my_restarts, kind, s, str(d / "a.dat")) == _restart_text(ba_restarts, kind, s, str(d / "b.dat"))


def test_prepare_restart_invalid(ba_restarts, tmp_path):
    for mod in (my_restarts, ba_restarts):
        with pytest.raises(ValueError, match="is an invalid choice of restart"):
            mod.prepare_restart(str(tmp_path / "nothing"), str(tmp_path / "r.dat"), Settings())


@pytest.fixture(scope="module")
def headers(ba_restarts):
    return my_restarts.load_header(), ba_restarts.load_header()


@hsettings(max_examples=25, deadline=None)
@given(u=st.lists(st.floats(0.0, 1.0), min_size=5, max_size=5), snap=st.lists(st.booleans(), min_size=5, max_size=5))
def test_interpolate_and_generate(ba_restarts, headers, u, snap):
    mine_h, ba_h = headers
    params = {}
    for k, x, s in zip(my_restarts.PARAM_KEYS, u, snap):
        axis = mine_h[k]
        params[k] = axis[int(round(x * (len(axis) - 1)))] if s else axis[0] + x * (axis[-1] - axis[0])
    a = my_restarts.interpolate_structure(dict(params), header=mine_h)
    b = ba_restarts.interpolate_structure(dict(params), header=ba_h)
    assert np.array_equal(a[0], b[0]) and a[3] == b[3] and a[4] == b[4]
    assert my_restarts.generate_model(a[0], params, mine_h, a[3], a[4]) == \
        ba_restarts.generate_model(b[0], params, ba_h, b[3], b[4])


def test_interpolate_out_of_grid(ba_restarts, headers):
    p = {"teff": 1000.0, "logg": 4.0, "zscale": 0.0, "alpha": 0.0, "carbon": 0.0}
    for mod, h in zip((my_restarts, ba_restarts), headers):
        with pytest.raises(ValueError, match="exceed grid bounds along axis teff"):
            mod.interpolate_structure(p, header=h)


# --------------------------------------------------------------------------------------------- runners
def test_script_interpreter(tmp_path, monkeypatch):
    data = tmp_path / "data.txt"
    data.write_text("payload\n")
    calls = []

    def fake(name, workdir, stdin_text, unit_files):
        calls.append((name, workdir, stdin_text, {u: os.path.realpath(p) for u, p in unit_files.items()}))
        with open(os.path.join(workdir, "fort.7"), "w") as f:
            f.write("out7")
        return "stdout of {}\n".format(name)

    script = tmp_path / "x.com"
    script.write_text(
        "\ncd {d}\nmkdir -p sub\ncd sub/\nln -s {data} fort.2\n# comment\n"
        "{d}/bin//synbeg.exe<<\"EOF\">synbeg.out\nline 1  \n  line 2\nEOF\n"
        "{d}/bin/synthe.exe< {data}>synthe.out\n"
        "echo \"-0.01 -1.9\" > c12c13.dat\ncat <<\"EOF\" >fort.25\na b\nEOF\n"
        "mv fort.7 spec.bin\nln spec.bin hard.bin\ncp {data} copy.txt\n{d}/bin/kapreadts.exe\nrm fort.*\nrm -f nothing\n"
        .format(d=tmp_path, data=data))
    out = runners.run_script(str(script), program=fake)
    sub = tmp_path / "sub"
    assert out == "stdout of kapreadts\n"
    assert [c[0] for c in calls] == ["synbeg", "synthe", "kapreadts"]
    assert calls[0][1] == str(sub) and calls[0][2] == "line 1  \n  line 2\n"
    assert calls[0][3] == {2: str(data)}
    assert calls[1][2] == "payload\n"
    assert (sub / "synbeg.out").read_text() == "stdout of synbeg\n"
    assert (sub / "c12c13.dat").read_text() == "-0.01 -1.9\n"
    assert (sub / "spec.bin").read_text() == "out7" and os.path.samefile(sub / "spec.bin", sub / "hard.bin")
    assert (sub / "copy.txt").read_text() == "payload\n"
    assert not list(sub.glob("fort.*"))

    bad = tmp_path / "bad.com"
    bad.write_text("cd {}\nrm fort.*\nmv a b\n".format(tmp_path))
    with pytest.raises(ValueError, match="returned an error"):
        runners.run_script(str(bad), program=fake)


def test_backend_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(runners, "_discovered", True)
    monkeypatch.setattr(runners, "_registry", {})
    used = []
    monkeypatch.setattr(runners, "_fortran", lambda *a: used.append("fortran") or "F")
    runners.register("spectrv", lambda wd, stdin, units: used.append("torch") or "T")

    monkeypatch.setenv("ATLAS_TORCH_BACKEND", "auto")
    assert runners.run_program("spectrv", tmp_path, "") == "T"
    assert runners.run_program("synthe", tmp_path, "") == "F"
    monkeypatch.setenv("ATLAS_TORCH_BACKEND", "fortran")
    assert runners.run_program("spectrv", tmp_path, "") == "F"
    monkeypatch.setenv("ATLAS_TORCH_BACKEND", "torch")
    assert runners.run_program("spectrv", tmp_path, "") == "T"
    with pytest.raises(NotImplementedError):
        runners.run_program("synthe", tmp_path, "")
    with pytest.raises(ValueError):
        runners.register("nonsense", lambda *a: "")
    assert used == ["torch", "fortran", "fortran", "torch"]


def test_fortran_runner_links_units(tmp_path):
    """FortranRunner links unit_files as fort.N for the run and removes them afterwards (synbeg: stdin only)."""
    card = "{:<3s}       {:<10.4f}{:<10.4f}{:<9.2f} {:<10.4f}{:<3d}{:<7d}{:<10.5f}{:<5d}{:<5d}\n".format(
        "VAC", 500.0, 501.0, 600000.0, 1.5, 0, -1, 0.0001, 1, 0) + "AIRorVAC  WLBEG\n"
    extra = tmp_path / "extra.dat"
    extra.write_text("x")
    out = runners.FortranRunner()("synbeg", str(tmp_path), card, {33: str(extra)})
    assert "WLBEG" in out
    assert not os.path.lexists(tmp_path / "fort.33")
