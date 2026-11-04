"""Convergence and conservation over 1500-200000 K and P = 1e-4 - 1e9 dyn cm^-2."""
import numpy as np
import torch


def test_wide_grid_converges_and_conserves():
    from atlas_torch.v2 import Atmosphere, ChemicalEquilibrium, Composition
    T = torch.logspace(np.log10(1500.0), np.log10(2.0e5), 9, dtype=torch.float64)
    P = torch.logspace(-4, 9, 6, dtype=torch.float64)
    TT, PP = torch.meshgrid(T, P, indexing="ij")
    comp = Composition.from_basicatlas(zscale=-1.0)
    p = ChemicalEquilibrium().solve(Atmosphere(rhox=torch.zeros_like(TT), T=TT, P=PP), comp)
    nu = torch.as_tensor(p.molecules.stoich)
    nuclei = p.n_atom.sum(-1) + p.n_mol @ nu
    assert torch.allclose(nuclei, comp.abundance * p.n_nuclei[..., None], rtol=1e-9, atol=0)
    assert torch.allclose(p.n_atom.sum((-1, -2)) + p.n_mol.sum(-1) + p.n_e, p.n_total, rtol=1e-9)
    hot = TT > 1e5
    assert (p.n_e[hot] / p.n_total[hot] > 0.5).all()                 # H and He fully ionised
    cold = (TT < 2000) & (PP > 1e3)
    h2 = p.density(101.0)
    assert (2 * h2[cold] / (comp.abundance[0] * p.n_nuclei[cold]) > 0.99).all()     # hydrogen molecular
