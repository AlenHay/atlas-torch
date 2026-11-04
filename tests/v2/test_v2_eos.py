"""Chemical equilibrium on the solar model: conservation laws, agreement with the exact port (NMOLEC/PFSAHA)."""
import pytest
import torch

from _oracle import deviations, solve_tier, spectators


@pytest.fixture(scope="module")
def v2_default(solar_deck):
    from atlas_torch.v2.eos import ChemicalEquilibrium
    return ChemicalEquilibrium().solve(solar_deck.atmosphere, solar_deck.composition)


def test_conservation_laws(v2_default, solar_deck):
    p = v2_default
    nu = torch.as_tensor(p.molecules.stoich)
    nuclei = p.n_atom.sum(-1) + p.n_mol @ nu                                    # (L, 99)
    a = solar_deck.composition.abundance
    assert torch.allclose(nuclei, a * p.n_nuclei[:, None], rtol=1e-9, atol=0)
    particles = p.n_atom.sum((-1, -2)) + p.n_mol.sum(-1) + p.n_e
    assert torch.allclose(particles, p.n_total, rtol=1e-9)
    stage = torch.arange(10, dtype=torch.float64)
    charge = (p.n_atom * stage).sum((-1, -2)) + p.n_mol @ torch.as_tensor(p.molecules.charge) - p.n_e
    assert (charge.abs() / p.n_e).max() < 1e-9
    assert torch.allclose(p.rho, p.n_nuclei * solar_deck.composition.mean_atomic_mass() * 1.66053906660e-24)


def test_fortran_equivalent_tier_matches_exact_port(solar_deck, solar_oracle):
    """Same species list, lowering and constants as NMOLEC: what is left is NMOLEC's convergence test
    (|dX/X| < 1e-3 per Newton step)."""
    d = deviations(solve_tier(solar_deck, solar_oracle, "A_fortran_equivalent"), solar_oracle,
                   skip_elements=spectators())
    assert d["n_e"] < 1e-4 and d["n_nuclei"] < 1e-6 and d["rho"] < 1e-6
    assert d["atoms"] < 2e-3 and d["molecules"] < 2e-3


def test_v2_default_against_exact_port(v2_default, solar_oracle):
    """Physics differences (full species list, self-consistent lowering, CODATA constants) stay at the
    percent level for n_e and below 1e-3 for the mass density; layer 1 excluded for the atoms (NMOLEC's
    lowering there uses n_e = P/kT/20)."""
    d = deviations(v2_default, solar_oracle, first_layer=1)
    assert d["n_e"] < 0.02 and d["rho"] < 1e-3 and d["mu"] < 1e-3
    worst = {k: v for k, v in d["atoms_top"].items() if k not in ("19.00", "11.00", "13.00", "12.00")}
    assert max(worst.values()) < 0.02


def test_batched_models_and_layers(solar_deck):
    from atlas_torch.v2 import Atmosphere, Composition
    from atlas_torch.v2.eos import ChemicalEquilibrium
    eq = ChemicalEquilibrium()
    atm = solar_deck.atmosphere
    hot = atm.with_(T=atm.T * 1.1)
    batch = Atmosphere.stack([atm, hot])
    comp = Composition(torch.stack([solar_deck.composition.abundance, solar_deck.composition.abundance * 1.0]))
    pb = eq.solve(batch, comp)
    p1 = eq.solve(hot, solar_deck.composition)
    assert pb.n_e.shape == (2, atm.n_layers) and pb.n_atom.shape == (2, atm.n_layers, 99, 10)
    torch.testing.assert_close(pb.n_e[1], p1.n_e, rtol=1e-10, atol=0)
    torch.testing.assert_close(pb.n_over_u_mol[1], p1.n_over_u_mol, rtol=1e-8, atol=1e-300)
