"""Kurucz file adapters: decks, molecules.dat, continua.dat, BasicATLAS abundances."""
import numpy as np
import pytest

from _oracle import MODELS, model_path


@pytest.mark.parametrize("name", MODELS)
def test_deck_round_trip(name):
    from atlas_torch.v2.io import format_atlas_deck, read_atlas_deck
    p = model_path(name)
    if not p.exists():
        pytest.skip(f"missing {p}")
    assert format_atlas_deck(read_atlas_deck(p)).splitlines() == p.read_text().splitlines()


def test_synthe_deck_matches_basicatlas(solar_deck):
    from atlas_torch.v2.io import format_atlas_deck
    ref = model_path("sun").with_name("output_synthe.out")
    assert format_atlas_deck(solar_deck).splitlines() == ref.read_text().splitlines()


def test_composition_equals_readin_xabund(solar_deck):
    from atlas_torch.synthe.readin import readin
    from atlas_torch.v2.io import format_atlas_deck
    s = readin(format_atlas_deck(solar_deck))
    a = solar_deck.composition.abundance
    np.testing.assert_allclose(a.numpy(), s["xabund.xabund"].numpy()[:99], rtol=1e-12)
    np.testing.assert_allclose(float(solar_deck.composition.mean_atomic_mass()), float(s["xabund.wtmole"]),
                               rtol=1e-7)


def test_basicatlas_solar_composition_reproduces_deck(solar_deck):
    from atlas_torch.v2.composition import Composition
    c = Composition.from_basicatlas()
    h, he, logm = c.atlas_cards()
    h0, he0, logm0 = solar_deck.composition.atlas_cards()
    assert (round(h, 5), round(he, 5)) == (h0, he0)
    np.testing.assert_allclose(np.round(logm, 2), logm0, atol=1e-9)
    x, y, z = c.mass_fractions()
    assert abs(float(x + y + z) - 1.0) < 1e-12 and 0.2 < float(y) < 0.3


def test_molecules_dat_matches_readmol_and_round_trips(tmp_path):
    from atlas_torch.synthe.readin import readmol
    from atlas_torch.v2.io import format_molecules_dat, read_molecules_dat
    rec = read_molecules_dat()
    ref = readmol()
    n = int(ref["xnmol.nummol"])
    np.testing.assert_array_equal([r.code for r in rec], ref["xnmol.code"].numpy()[:n])
    eq = ref["ifequa.equil"].numpy()[:, :n]
    np.testing.assert_array_equal([r.d0 for r in rec], eq[0])
    np.testing.assert_array_equal(np.array([r.coeffs for r in rec]).T, eq[1:])
    p = tmp_path / "molecules.dat"
    p.write_text(format_molecules_dat(rec))
    assert [(r.code, r.d0, r.coeffs) for r in read_molecules_dat(p)] == [(r.code, r.d0, r.coeffs) for r in rec]


def test_continua_edges_match_xnfpelsyn():
    from atlas_torch.synthe.freeform import Free
    from atlas_torch.synthe.xnfpelsyn_prog import default_paths, read_edges
    from atlas_torch.v2.io import read_continua_dat
    e = read_continua_dat()
    frq, wl, cm = read_edges(default_paths()["continua"].read_text(), Free(numcol=1, last=81, more=1, maxpow=99))
    np.testing.assert_allclose(e["wavelength_nm"], wl, rtol=1e-14)
    np.testing.assert_allclose(e["frequency"], frq, rtol=1e-14)
    np.testing.assert_allclose(e["wavenumber"], cm, rtol=1e-14)


def test_species_codes():
    from atlas_torch.v2.constants import parse_species, species_name
    from atlas_torch.v2.eos.molecules import decode_code, formula
    assert parse_species("Fe II") == (26, 2) == parse_species("26.01")
    assert species_name(8, 1) == "O I"
    assert decode_code(10108.0) == ({1: 2, 8: 1}, 0, 0)
    assert decode_code(100.0) == ({1: 1}, 1, 0)
    assert decode_code(101.01) == ({1: 2}, 0, 1)
    assert formula(60808.0) == "CO2" and formula(100.0) == "H-"
