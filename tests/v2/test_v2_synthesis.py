"""v2 synthesis pipeline: spectrum.bin writer, SYNTHE grid, transfer alone against stock SPECTRV (exact-port
opacities as stand-ins)."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.io.formats import read_spectrum_bin
from atlas_torch.v2.opacity.lines.grid import LogWavelengthGrid
from atlas_torch.v2.synthesis import Spectrum

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
STOCK = EXT / "runs/stock/atlas/sun"


def test_spectrum_bin_roundtrip(tmp_path):
    g = LogWavelengthGrid.window(500.0, 500.1, 600000.0)
    f = torch.linspace(1.0, 2.0, g.length, dtype=torch.float64)
    s = Spectrum(g, f * 0.9, f, teff=5777.0, logg=4.44, title="  TEST")
    s.write(tmp_path / "spectrum.bin")
    r = read_spectrum_bin(tmp_path / "spectrum.bin")
    assert r["n"] == g.length and r["wbegin"] == g.wl_begin and r["resolu"] == 600000.0
    assert np.array_equal(r["flux"], (f * 0.9).numpy()) and np.array_equal(r["cont"], f.numpy())
    assert r["header"][16:21] == b"V TES" and [len(t) for t in r["tail"]] == [4]


def test_grid_matches_stock_window():
    p = STOCK / "window_0/spectrum.bin"
    if not p.exists():
        pytest.skip(f"missing {p}")
    r = read_spectrum_bin(p)
    g = LogWavelengthGrid.window(500.0, 510.0, 600000.0)
    assert g.length == r["n"] and g.wl_begin == r["wbegin"]


@pytest.mark.stage
def test_transfer_alone_vs_stock_window0(tmp_path):
    """Stock solar model, stock XNFPELSYN/SYNTHE opacities (fort.10, fort.9), v2 populations unused: the deviation
    from stock spectrum.bin is the transfer's (measured: cont 2.2e-4 rel, normalised flux 1.3e-4 max)."""
    from atlas_torch.scoring import synthe_metrics_v2
    from atlas_torch.v2.constants import KURUCZ
    from atlas_torch.v2.io import read_atlas_deck, to_synthe_deck
    from atlas_torch.v2.synthesis import TableOpacity, deck_title, synthesize
    work = EXT / "work/stocksun_500_510"
    if not (work / "fort.9").exists():
        pytest.skip(f"missing {work}")
    deck = to_synthe_deck(read_atlas_deck(STOCK / "output_summary.out"))
    tab = TableOpacity(work)
    with torch.no_grad():
        spec = synthesize(deck.atmosphere, deck.composition, tab.grid, continuum=tab.continuum, lines=tab.lines,
                          const=KURUCZ, title=deck_title(deck))
    spec.write(tmp_path / "spectrum.bin")
    m = synthe_metrics_v2(STOCK / "window_0", tmp_path)
    assert m["cont:rel"] < 5e-4 and m["normflux:abs"] < 5e-4 and m["normflux:rms_abs"] < 1e-4
    assert open(tmp_path / "spectrum.bin", "rb").read(12) == open(STOCK / "window_0/spectrum.bin", "rb").read(12)


def test_kurucz_configuration_reaches_every_module():
    """SynthesisConfig.kurucz() switches the EOS, continuum, line opacity, line selection and transfer together."""
    from atlas_torch.v2.opacity.continuum import ContinuousOpacity, EdgeSampled, KuruczCompat
    from atlas_torch.v2.opacity.lines import KURUCZ_OPTIONS, LineList
    from atlas_torch.v2.synthesis import SynthesisConfig
    k = SynthesisConfig.kurucz()
    eq = k.equation_of_state()
    assert (eq.network, eq.debye_mode, eq.lagged, eq.const.name) == ("kurucz", "kurucz", True, "KURUCZ")
    op = k.opacity(LogWavelengthGrid.window(500.0, 500.1, 600000.0), line_list=LineList.empty())
    assert isinstance(op.continuum, EdgeSampled) and isinstance(op.continuum.continuum, ContinuousOpacity)
    assert op.continuum.continuum.compat == KuruczCompat.all() and op.continuum.continuum.const.name == "KURUCZ"
    opt = op.line_opacity.opt
    assert opt.const.name == "KURUCZ" and all(getattr(opt, a) == v for a, v in KURUCZ_OPTIONS.items())
    assert (k.selection, k.kurucz_isotopes) == ("kurucz", True)
    tr = k.transfer_solver()
    assert (tr.opaque_top, tr.tau_scheme) == ("diffusion", "atlas")
    v = SynthesisConfig()
    assert v.equation_of_state().network == "full" and v.const.name == "CODATA2018"
    assert isinstance(v.opacity(LogWavelengthGrid.window(500.0, 500.1, 600000.0),
                                line_list=LineList.empty()).continuum, ContinuousOpacity)


@pytest.mark.stage
def test_kurucz_compatible_pipeline_vs_stock_window0(tmp_path):
    """The whole v2 pipeline in SynthesisConfig.kurucz() against stock spectrum.bin, window 0, gated by
    tolerances.json stages.synthe (measured: cont 2.2e-4 rel, normalised flux 5.3e-4 max)."""
    import json
    from atlas_torch.scoring import gate_v2, synthe_metrics_v2
    from atlas_torch.v2.io import read_atlas_deck, to_synthe_deck
    from atlas_torch.v2.synthesis import deck_title, synthesize
    if not (STOCK / "window_0/spectrum.bin").exists():
        pytest.skip(f"missing {STOCK}")
    deck = to_synthe_deck(read_atlas_deck(STOCK / "output_summary.out"))
    with torch.no_grad():
        spec = synthesize(deck.atmosphere, deck.composition, LogWavelengthGrid.window(500.0, 510.0, 600000.0),
                          kurucz=True, title=deck_title(deck))
    spec.write(tmp_path / "spectrum.bin")
    tol = json.loads((Path(__file__).resolve().parents[2] / "results/v2/tolerances.json").read_text())
    m = synthe_metrics_v2(STOCK / "window_0", tmp_path)
    assert not gate_v2(m, tol["stages"]["synthe"]), m
