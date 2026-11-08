"""Continuous opacity: interpolation, batching, gradients, and the Kurucz-equivalent tier against the exact port on the
solar model (the three-model validation is test_v2_continuum_models.py)."""
import dataclasses

import pytest
import torch

from atlas_torch.v2.opacity.continuum import PROCESSES, ContinuousOpacity, KuruczCompat
from atlas_torch.v2.opacity.continuum.interpolate import linear, map1


def test_interpolation_exact_on_polynomials():
    xp = torch.linspace(0.0, 10.0, 11, dtype=torch.float64) ** 1.3
    x = torch.linspace(-1.0, 25.0, 301, dtype=torch.float64)
    assert torch.allclose(linear(x, xp, 3.0 * xp - 2.0), 3.0 * x - 2.0)
    inner = (x >= xp[2]) & (x <= xp[-1])                               # MAP1 is linear on the first two intervals
    assert torch.allclose(map1(x, xp, xp ** 2 - xp)[inner], (x ** 2 - x)[inner])


@pytest.fixture(scope="module")
def solar_pops(solar_deck):
    from atlas_torch.v2.eos import ChemicalEquilibrium
    atm = solar_deck.atmosphere
    sel = slice(5, 70, 8)
    sub = atm.with_(T=atm.T[sel], P=atm.P[sel], rhox=atm.rhox[sel], n_e=atm.n_e[sel], vturb=atm.vturb[sel], extra={})
    return ChemicalEquilibrium().solve(sub, solar_deck.composition)


@pytest.fixture(scope="module")
def nu():
    from atlas_torch.v2.io import frequency_set
    return torch.as_tensor(frequency_set())


def test_shapes_and_batching(solar_pops, nu):
    op = ContinuousOpacity()(solar_pops, nu, parts=True)
    n = solar_pops.T.shape[0]
    assert op.absorption.shape == op.scattering.shape == (n, nu.shape[0])
    assert set(op.absorbers) | set(op.scatterers) == set(PROCESSES)
    assert torch.allclose(sum(op.absorbers.values()), op.absorption) and bool((op.absorption > 0).all())
    stacked = {f.name: getattr(solar_pops, f.name) for f in dataclasses.fields(solar_pops)}
    batched = dataclasses.replace(solar_pops, **{k: torch.stack([v, v.flip(0)]) for k, v in stacked.items()
                                                 if isinstance(v, torch.Tensor)})
    ob = ContinuousOpacity()(batched, nu)
    assert ob.total.shape == (2, n, nu.shape[0])
    assert torch.allclose(ob.total[0], op.total, rtol=1e-13) and torch.allclose(ob.total[1], op.total.flip(0), rtol=1e-13)


def test_gradients(solar_pops, nu):
    """d log kappa / dT and d/dn_e at fixed populations: autograd against central differences."""
    for name in ("T", "n_e"):
        x = getattr(solar_pops, name).clone().requires_grad_(True)
        f = lambda v: ContinuousOpacity()(dataclasses.replace(solar_pops, **{name: v}), nu).total.log().sum()  # noqa
        (g,) = torch.autograd.grad(f(x), x)
        k = 4
        h = 1e-6 * float(x[k].detach())
        e = torch.zeros_like(x)
        e[k] = h
        fd = (f(x.detach() + e) - f(x.detach() - e)) / (2 * h)
        assert torch.isfinite(g).all() and abs(float(g[k]) / float(fd) - 1.0) < 1e-6, name


def test_kurucz_tier_solar(solar_deck, solar_oracle):
    """Exact-port populations, Kurucz constants, all KuruczCompat switches: every exact-port KAPP routine reproduced to
    < 2E-4 where it holds >= 1E-3 of the extinction. H I bound-free in rms only: whether the exact port includes n = 10
    at the point 1.2E-7 above its edge depends on the libm's REAL*4 log10 (XKARSAS threshold window)."""
    from _opacity_oracle import deviations, oracle_kapp, populations_from_oracle
    from atlas_torch.v2 import constants
    from atlas_torch.v2.eos import ChemicalEquilibrium
    from atlas_torch.v2.io import format_atlas_deck
    ok = oracle_kapp(format_atlas_deck(solar_deck))
    pops = populations_from_oracle(solar_oracle, ChemicalEquilibrium().molecules)
    op = ContinuousOpacity(const=constants.KURUCZ, compat=KuruczCompat.all())(pops, torch.as_tensor(ok["freqset"]),
                                                                              parts=True)
    d = deviations(op, ok, add_floor=True, pops=pops)
    for g, v in d["groups"].items():
        if "all" in v["rel"] and g != "bhyd.ahyd":
            assert v["rel"]["all"]["max"] < 2e-4, g
    assert d["groups"]["bhyd.ahyd"]["rel"]["all"]["rms"] < 2e-3
    assert d["totals"]["continall"]["all"]["rms"] < 1e-4


def test_chunked_evaluation_and_edge_sampling(solar_pops):
    """Frequency blocks change the result only by round-off; EdgeSampled is exact at the interval midpoints (XNFPELSYN's
    middle frequency) and within 0.5 % of the direct evaluation across 500-510 nm."""
    from atlas_torch.v2.opacity.continuum import EdgeSampled
    wl = torch.linspace(500.0, 510.0, 101, dtype=torch.float64)
    nu = 2.99792458e17 / wl
    one = ContinuousOpacity(chunk=1 << 20)(solar_pops, nu)
    blocks = ContinuousOpacity(chunk=7)(solar_pops, nu)
    torch.testing.assert_close(blocks.absorption, one.absorption, rtol=1e-13, atol=0)
    torch.testing.assert_close(blocks.scattering, one.scattering, rtol=1e-13, atol=0)
    es = EdgeSampled(ContinuousOpacity())
    mid = torch.as_tensor(es.freqset[1::3][100:110])
    torch.testing.assert_close(es(solar_pops, mid).total, ContinuousOpacity()(solar_pops, mid).total, rtol=1e-10, atol=0)
    assert float((es(solar_pops, nu).total / one.total - 1).abs().max()) < 5e-3

