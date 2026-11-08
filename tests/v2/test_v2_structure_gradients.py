"""Derivatives through one structure iteration's radiation integrals (EOS, continuum, ODF, transfer) on a reduced
problem: 2 models, 8 layers, 16 frequencies."""
import os
from pathlib import Path

import pytest
import torch

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
SUN = EXT / "runs" / "stock" / "atlas" / "sun"
SOLAR = EXT / "runs" / "stock" / "odf" / "solar"
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def test_radiation_moments_gradient():
    if not (SUN / "output_summary.out").exists() or not (SOLAR / "p00big2.bdf").exists():
        pytest.skip("stock files missing")
    from atlas_torch.v2.atmosphere import Atmosphere
    from atlas_torch.v2.io import read_atlas_deck
    from atlas_torch.v2.opacity.frequencies import FrequencyQuadrature
    from atlas_torch.v2.opacity.odf import OpacityDistribution
    from atlas_torch.v2.structure import StructureConfig
    from atlas_torch.v2.structure.radiation import RadiationMoments
    deck = read_atlas_deck(SUN / "output_summary.out")
    a = deck.atmosphere
    sl = slice(24, 72, 6)
    rhox = torch.stack([a.rhox[sl], a.rhox[sl]])
    T0 = torch.stack([a.T[sl], a.T[sl] * 1.02])
    P = torch.stack([a.P[sl], a.P[sl]])
    cfg = StructureConfig()
    odf = OpacityDistribution.read(SOLAR, "big", vturbs=(2,))
    q = cfg.quadrature(odf.intervals)
    idx = torch.arange(60, 300, 15)
    q = FrequencyQuadrature(q.nu[idx], q.weight[idx], q.interval[idx])
    eos, cont, tr = cfg.equation_of_state(), cfg.continuous_opacity(), cfg.transfer_solver()
    comp = deck.composition

    def f(T):
        pops = eos.solve(Atmosphere(rhox=rhox, T=T, P=P), comp)
        r = RadiationMoments.compute(rhox, pops, q, q.samples(odf.fractions), cont, odf, tr, const=cfg.const)
        return r.flux_rad[:, 5].sum() + 1e7 * r.kappa_ross[:, 3].sum()

    T = T0.clone().requires_grad_(True)
    g, = torch.autograd.grad(f(T), T)
    for m, j in ((0, 5), (1, 3)):
        h = 0.05
        tp, tm = T0.clone(), T0.clone()
        tp[m, j] += h
        tm[m, j] -= h
        fd = (f(tp) - f(tm)) / (2 * h)
        assert float(g[m, j]) == pytest.approx(float(fd), rel=2e-4)
