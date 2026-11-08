"""v2 ATLAS9 control / restart readers and run-product writers (BasicATLAS layouts)."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.v2.io.atlas_control import read_control, read_restart
from atlas_torch.v2.io.atlas_output import fortran_e, format_iteration_table, select_best_iteration, write_atlas_run

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
SUN = EXT / "runs" / "stock" / "atlas" / "sun"


def _need(p):
    if not p.exists():
        pytest.skip(f"missing {p}")


def test_control_cards_of_basicatlas_run():
    _need(SUN / "atlas_control.com")
    c = read_control(SUN / "atlas_control.com")
    assert (c.teff, c.logg, c.vturb) == (5777.0, 4.44, 2.0e5)
    assert (c.mixing_length, c.overshoot_weight, c.nconv, c.convection) == (1.25, 0.0, 36, True)
    assert c.tau_grid == (72, -6.875, 0.125) and c.iterations == 450 and c.molecules
    assert c.frequency_set == ("BIG", 337) and c.title.strip() == "BasicATLAS"
    a = c.composition.abundance
    assert abs(float(a[0]) - 0.91183) < 1e-12 and abs(float(a[1]) - 0.08708) < 1e-12
    assert abs(float(torch.log10(a[25])) - (-4.52)) < 1e-12
    np.testing.assert_allclose(c.log_tau()[[0, -1]], [-6.875, 2.0], atol=1e-12)


def test_restart_deck_with_nans():
    _need(SUN / "restart.dat")
    r = read_restart(SUN / "restart.dat")
    a = r.atmosphere
    assert float(a.teff) == 5800.0 and torch.isnan(a.logg)
    assert r.composition is None and r.pradk0 == 0.0 and r.iteration == 15
    assert torch.isnan(a.P).all() and torch.isfinite(a.T).all() and a.T.shape == (72,)
    # the RHOX column holds tau_Ross (ABROSS = 1)
    torch.testing.assert_close(r.tau_ross(), a.rhox, rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.log10(a.rhox.numpy())[[0, -1]], [-6.875, 2.0], atol=1e-6)


def test_stock_summary_reads_as_restart():
    _need(SUN / "output_summary.out")
    r = read_restart(SUN / "output_summary.out")
    assert r.composition is not None and abs(r.pradk0 - 1.4874) < 1e-12 and float(r.atmosphere.logg) == 4.44
    assert float(r.atmosphere.extra["ABROSS"][0]) == pytest.approx(2.693e-4)


def test_fortran_e_edit_descriptors():
    assert fortran_e(4.95206e-4, 17, 5) == "      4.95206E-04"
    assert fortran_e(3696.3, 17, 5, 0) == "      0.36963E+04"
    assert fortran_e(-0.00164, 15, 3, 0) == "     -0.164E-02"
    assert fortran_e(0.0, 17, 5) == "      0.00000E+00"


def _records(n_it=3, L=72):
    rng = np.random.default_rng(1)
    recs = []
    for k in range(n_it):
        r = {q: np.abs(rng.normal(size=L)) + 1.0 for q in ("RHOX", "T", "P", "XNE", "RHO", "ABROSS", "HEIGHT",
                                                             "TAUROS", "FLXCNV", "PRADK")}
        r["T"] = 4000.0 + 100.0 * np.arange(L)
        r["FLXERR"] = rng.normal(size=L) * (3 - k)
        r["FLXDRV"] = rng.normal(size=L) * (30 - 10 * k)
        for q in ("RHOX", "T", "P", "XNE", "ABROSS", "ACCRAD", "VTURB", "FLXRAD", "VCONV", "VELSND"):
            r["deck_" + q] = r.get(q, np.full(L, 2.0e5))
        r["PRADK0"] = np.float64(1.4874)
        r["max_flux_error"] = np.abs(r["FLXERR"]).max()
        r["max_flux_derivative"] = np.abs(r["FLXDRV"]).max()
        r["iteration"] = r["iteration_in_batch"] = k + 1
        recs.append(r)
    return recs


def test_writers_round_trip(tmp_path):
    _need(SUN / "atlas_control.com")
    from atlas_torch.io.formats import read_atlas_table, read_deck6, read_last_iteration
    from atlas_torch.v2.io import read_atlas_deck
    from atlas_torch.v2.structure import ModelSpec
    from atlas_torch.v2.structure.grid import RosselandGrid
    c = read_control(SUN / "atlas_control.com")
    spec = ModelSpec(c.teff, c.logg, c.composition, c.vturb, c.mixing_length, c.overshoot_weight, c.nconv,
                     RosselandGrid(), odf=None, title=c.title)
    recs = _records()
    best = select_best_iteration([r["max_flux_error"] for r in recs], [r["max_flux_derivative"] for r in recs])
    paths = write_atlas_run(tmp_path, spec, recs, best, "gold")
    tab = read_atlas_table(paths["main"], "last")
    assert tab["ntables"] == 3
    np.testing.assert_allclose(tab["T"], recs[-1]["T"], rtol=1e-5)
    last = read_last_iteration(paths["last"])
    np.testing.assert_allclose(last["FLXERR"], recs[best]["FLXERR"], rtol=5e-3, atol=1e-12)
    np.testing.assert_allclose(last["P"], recs[best]["P"], rtol=1e-5)
    # BasicATLAS atlas_converged layout: 72 rows after the 4 header lines, columns 11 and 12
    import re
    text = paths["main"].read_text()
    its = re.findall("START TABLE.+?END TABLE", text, re.DOTALL)
    rows = its[best].split("\n")[4:-1]
    assert len(rows) == 72
    err, de = np.loadtxt(rows, unpack=True, usecols=[11, 12])
    np.testing.assert_allclose(err, recs[best]["FLXERR"], rtol=5e-3, atol=1e-12)
    d6 = read_deck6(paths["summary"])
    np.testing.assert_allclose(d6["T"], recs[best]["deck_T"], atol=0.051)
    deck = read_atlas_deck(paths["summary"])
    assert float(deck.atmosphere.teff) == 5777.0 and float(deck.atmosphere.logg) == 4.44
    torch.testing.assert_close(deck.composition.abundance, c.composition.abundance, rtol=1e-5, atol=0)
    assert "REACHED GOLD TOLERANCES" in text


def test_best_iteration_rule():
    assert select_best_iteration([5, 0.5, 0.8], [5, 20, 9]) == 2           # GOLD wins over a smaller error
    assert select_best_iteration([5, 3, 0.8], [500, 50, 900]) == 1         # SILVER
    assert select_best_iteration([float("nan"), 2e3, 900], [1, 1, 1e6]) == 2


def test_iteration_table_header():
    t = format_iteration_table(_records(1)[0], 5777.0, 4.44, " BasicATLAS", 15)
    assert t[0] == " START TABLE" and t[-1] == " END TABLE" and len(t) == 72 + 5
    assert t[1].startswith(" TEFF   5777.   LOG G  4.44000            B a s i c A T L A S")
    assert t[1].endswith("ITERATION 15")


def test_main_output_passes_basicatlas_checks(tmp_path):
    """BasicATLAS validate_run (abundances, XSCALE, TEFF, LOG G, VTURB) and the 0FREQID frequency-range parse of
    atlas() on a written run directory."""
    import shutil
    import sys
    basic = EXT / "BasicATLAS"
    _need(SUN / "atlas_control.com")
    _need(basic / "atlas.py")
    from atlas_torch.v2.opacity.frequencies import FrequencyQuadrature
    from atlas_torch.v2.structure import ModelSpec
    from atlas_torch.v2.structure.grid import RosselandGrid
    sys.path.insert(0, str(basic))
    try:
        import atlas as basicatlas
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"BasicATLAS not importable: {e}")
    finally:
        sys.path.remove(str(basic))
    c = read_control(SUN / "atlas_control.com")
    spec = ModelSpec(c.teff, c.logg, c.composition, c.vturb, c.mixing_length, c.overshoot_weight, c.nconv,
                     RosselandGrid(), odf=None, title=c.title, convection=c.convection)
    recs = _records()
    for r in recs:
        r["deck_VTURB"] = np.full(72, c.vturb)
    q = FrequencyQuadrature.atlas_big()
    write_atlas_run(tmp_path, spec, recs, 2, "gold", frequencies=(q.nu, q.weight))
    shutil.copy(SUN / "atlas_control.com", tmp_path / "atlas_control.com")
    assert basicatlas.validate_run(str(tmp_path), silent=True)
    nus = []
    reading = False
    for line in (tmp_path / "output_main.out").read_text().splitlines(True):      # atlas.py lines 330-343
        if line.find("0FREQID") != -1:
            reading = True
            continue
        if reading and len(line) < 60:
            break
        if reading:
            nus += list(np.loadtxt([line.replace("-", " -")])[1::3])
    assert len(nus) == 337
    np.testing.assert_allclose(sorted(nus), np.sort(q.nu.numpy()), rtol=1e-6)
    stock = (SUN / "output_main.out").read_text().splitlines()[9:26]          # TEFF .. IFTURB of the stock run
    ours = (tmp_path / "output_main.out").read_text().splitlines()[5:22]
    assert [ln.rstrip() for ln in ours] == [ln.rstrip() for ln in stock]
