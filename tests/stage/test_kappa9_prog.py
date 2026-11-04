"""Stage tests for the KAPPA9 program (atlas_torch.kappa9.kappa9_prog).

1. run_workdir on the stock solar / alpha ODFs: kapk{0,1,2,4,8}.dat and kappa.ros byte-identical to the stock
   KAPPA9 + KAPREADTS work dirs (all ifx build variants agree byte for byte, so the gate is exact).
2. hypothesis: perturbed control decks (ABUNDANCE SCALE / CHANGE, velocity) -> port vs the whole Fortran
   program (oracle library MAIN__) on the same ODF: identical kapk text.

Test 1 runs with both physics backends: 'torch' (atlas_torch.atlas9.physics, the port) and 'oracle' (the
test-only Fortran adapter). Test 2 compares program against program, so both sides use the Fortran physics.
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.io.formats import read_kapk
from atlas_torch.kappa9 import kappa9_prog as kp
from atlas_torch.testing import kappa9_oracle_physics as k9o

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
pytestmark = pytest.mark.stage


def _diff_entries(cand: Path, ref: Path) -> list:
    a, b = read_kapk(cand), read_kapk(ref)
    return [(int(ip) + 1, int(it) + 1, int(a[ip, it]), int(b[ip, it])) for ip, it in np.argwhere(a != b)]


def _backend(name):
    try:
        return kp.get_physics(name)
    except (NotImplementedError, FileNotFoundError, OSError) as e:
        pytest.skip(f"{name} physics unavailable: {e}")


@pytest.mark.parametrize("physics", ["torch", "oracle"])
@pytest.mark.parametrize("mix", ["solar", "alpha"])
def test_run_workdir_matches_stock(mix, physics, tmp_path):
    ref = EXT / "work" / f"kappa9_{mix}_stock"
    odf = EXT / "runs" / "stock" / "odf" / mix
    if not (ref / "kapk0.dat").exists() or not (odf / "p00big0.bdf").exists():
        pytest.skip("stock KAPPA9 work dir / ODFs missing")
    info = kp.run_workdir(odf, tmp_path, physics=_backend(physics))
    print(f"\n{mix}: physics={info['physics']} timing={ {k: round(v, 2) for k, v in info['timing'].items()} }")
    bad = {}
    for v in kp.VTURBS:
        if (tmp_path / f"kapk{v}.dat").read_bytes() != (ref / f"kapk{v}.dat").read_bytes():
            bad[v] = _diff_entries(tmp_path / f"kapk{v}.dat", ref / f"kapk{v}.dat")
    assert not bad, f"differing (KTAB, T index, port, stock): {bad}"
    assert (tmp_path / "kappa.ros").read_bytes() == (ref / "kappa.ros").read_bytes()


@settings(max_examples=4, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(xscale=st.floats(0.003, 10.0), dfe=st.floats(-1.0, 1.0), dc=st.floats(-1.0, 1.0),
       v=st.sampled_from([0, 2, 8]))
def test_perturbed_deck_vs_fortran_main(xscale, dfe, dc, v):
    if not k9o.available():
        pytest.skip("libkappa9.so / Intel runtime missing")
    odf_dir = EXT / "runs" / "stock" / "odf" / "solar"
    mol = kp.default_molecules()
    txt = kp.control_from_com(odf_dir / f"kappa9v{v}.com")
    txt = re.sub(r"ABUNDANCE SCALE +[0-9.]+", f"ABUNDANCE SCALE {xscale:9.6f}", txt)
    txt = re.sub(r"( 6 +)(-?[0-9.]+)", lambda m: f"{m.group(1)}{float(m.group(2)) + dc:6.2f}", txt, count=1)
    txt = re.sub(r"(26 +)(-?[0-9.]+)", lambda m: f"{m.group(1)}{float(m.group(2)) + dfe:6.2f}", txt, count=1)
    odf = odf_dir / f"p00big{v}.bdf"
    ref = k9o.fortran_kappa9_main(txt, odf, mol)
    got = kp.kappa9(odf, v, control=txt, molecules=mol, physics="oracle")   # program vs program, same physics
    if got != ref:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a").write_text(got)
            Path(d, "b").write_text(ref)
            diffs = _diff_entries(Path(d, "a"), Path(d, "b"))
        pytest.fail(f"kapk differs from the Fortran program at (KTAB, T index, port, fortran): {diffs}")


class _OracleAsBatchedPhysics:
    """Fake `atlas_torch.atlas9.physics` module (populations/continuum entry points) backed by the oracle
    adapter, so that the batched torch-physics code path of kappa9_prog (TorchKappa9Physics: per-column
    populations, one continuum call per frequency on all 42 x 57 points) is exercised before the torch physics
    exists. Before evaluating each column of a batched continuum call, the COMMON members that differ between
    columns are restored from that column's snapshot taken after its POPS block."""

    def __init__(self):
        self.a = k9o.OracleKappa9Physics()
        self.snaps = []
        self.keys = None
        self.loaded = False

    def _snapshot(self):
        o = self.a.o
        return {(b, v): np.array(a, copy=True) for b in o.commons_layout for v, a in o.common(b).items()}

    def populations(self, s, variant):
        assert variant == "kappa9"
        if not self.loaded:
            from atlas_torch.state import State
            self.a.load(State({k: v for k, v in s.items() if not k.startswith(("temp.", "state.", "rhox."))}))
            self.a.set("temp", "hckt", s["temp.hckt"])
            self.loaded = True
        k = int(s["temp.itemp"][0])
        grid = {v: s[f"temp.{v}"] for v in ("t", "tk", "hkt", "tkev", "tlog")}
        grid["p"] = s["state.p"]
        out = self.a.populations(k, grid)
        self.snaps.append(self._snapshot())
        return out

    def continuum(self, s, freq, variant, derive_freq=False):
        if self.keys is None:
            s0, s1 = self.snaps[0], self.snaps[1]
            self.keys = [key for key in s0 if not np.array_equal(s0[key], s1[key])]
        n = 57
        res = []
        for k, snap in enumerate(self.snaps):
            views = {}
            for (b, v) in self.keys:
                views.setdefault(b, self.a.o.common(b))[v][...] = snap[(b, v)]
            self.a.nrhox = n
            sl = slice(k * n, (k + 1) * n)
            res.append(self.a.continuum(float(s["freq.freq"][0]), float(s["freq.freqlg"][0]), s["freq.ehvkt"][sl],
                                        s["freq.stim"][sl], s["freq.bnu"][sl]))
        return {"ops." + v: __import__("torch").cat([r[v] for r in res]) for v in res[0]}


def test_batched_physics_path_matches_stock():
    if not k9o.available():
        pytest.skip("libkappa9.so / Intel runtime missing")
    ref = EXT / "work" / "kappa9_solar_stock" / "kapk2.dat"
    odf = EXT / "runs" / "stock" / "odf" / "solar" / "p00big2.bdf"
    if not ref.exists() or not odf.exists():
        pytest.skip("stock files missing")
    phys = object.__new__(kp.TorchKappa9Physics)
    phys.ph, phys.rd, phys.cur, phys.cols, phys.state = _OracleAsBatchedPhysics(), None, None, [], None
    control = (EXT / "work" / "kappa9_solar_stock" / "stdin_v2.txt").read_text()
    from atlas_torch.kappa9.readin import readin
    rd = readin(control, kp.default_molecules())
    cont = kp.continuum_tables(rd, phys)
    assert cont["physics"] == "torch"
    got = kp.kappa9(odf, 2, control=control, cont=cont)
    assert got == ref.read_text()


def test_compat_runner_interface(tmp_path):
    """PROGRAMS['kappa9'](workdir, stdin_text, unit_files) writes fort.7 = the stock kapk file."""
    ref = EXT / "work" / "kappa9_alpha_stock"
    odf = EXT / "runs" / "stock" / "odf" / "alpha" / "p00big4.bdf"
    if not (ref / "kapk4.dat").exists() or not odf.exists():
        pytest.skip("stock files missing")
    os.environ.setdefault(kp.PHYSICS_ENV, "oracle")      # interface test: fast backend unless chosen
    out = kp.PROGRAMS["kappa9"](str(tmp_path), (ref / "stdin_v4.txt").read_text(),
                                {9: str(odf), 2: str(kp.default_molecules())})
    assert "KAPPA9" in out
    assert (tmp_path / "fort.7").read_bytes() == (ref / "kapk4.dat").read_bytes()
