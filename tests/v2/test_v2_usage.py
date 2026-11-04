"""Usage: solar model deck -> Atmosphere -> populations."""
import pytest


def test_solar_model_populations():
    from _oracle import model_path
    from atlas_torch.v2 import ChemicalEquilibrium, read_atlas_deck

    path = model_path("sun")                             # BasicATLAS output_summary.out
    if not path.exists():
        pytest.skip(f"missing {path}")
    deck = read_atlas_deck(path)
    atm, comp = deck.atmosphere, deck.composition        # T, P, rhox, ... (72 layers); XABUND
    pops = ChemicalEquilibrium().solve(atm, comp)

    tau1 = int((atm.T - atm.teff).abs().argmin())        # the layer where T = Teff
    n_e = pops.n_e[tau1]                                 # cm^-3
    fe_ii = pops.density("Fe II")[tau1]                  # number density of Fe+
    co_nu = pops.n_over_u(608.0)[tau1]                   # CO: n/U, the line-opacity prefactor
    assert 1e13 < n_e < 1e15 and fe_ii > 10 * pops.density("Fe I")[tau1] and co_nu > 0
    assert 1.2 < float(pops.mu[tau1]) < 1.35             # mean molecular weight of neutral solar gas
