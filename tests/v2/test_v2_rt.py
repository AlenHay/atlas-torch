"""v2 transfer: optical depth quadrature, analytic emergent fluxes, scattering, ALI vs direct solve, gradients."""
import math

import pytest
import torch

from atlas_torch.v2.rt import Transfer, optical_depth

F64 = torch.float64


def _grid(L=72, lo=-8.0, hi=2.5):
    return torch.logspace(lo, hi, L, dtype=F64)


def test_optical_depth_linear_and_quadratic():
    m = _grid(40, -4, 2)
    lin = 2.0 + 3.0 * m
    tau = optical_depth(m, lin)
    exact = lin[0] * m[0] + 2.0 * (m - m[0]) + 1.5 * (m ** 2 - m[0] ** 2)
    assert torch.allclose(tau, exact, rtol=1e-13)
    q = 1.0 + m + 0.5 * m ** 2                                    # interior intervals: both parabolas exact
    d = optical_depth(m, q).diff()
    ex = (m + 0.5 * m ** 2 + m ** 3 / 6).diff()
    assert torch.allclose(d[1:-1], ex[1:-1], rtol=1e-12)


@pytest.mark.parametrize("scattering", ["absorption", "coherent"])
def test_linear_source_flux_and_intensity(scattering):
    """S = a + b tau: H(0) = a/4 + b/6, I(0, mu) = a + b mu (Kurucz 1970 eqs. 2.27b, 2.42) for any ray set."""
    rhox = _grid()
    kap = torch.ones_like(rhox)
    tau = optical_depth(rhox, kap)
    a, b = 1.0, 1.5
    r = Transfer(4, scattering=scattering).solve(rhox, kap, torch.zeros_like(rhox), a + b * tau, mu=[0.1, 0.5, 1.0])
    assert abs(float(r.flux) / (a / 4 + b / 6) - 1) < 1e-13
    assert torch.allclose(r.intensity, a + b * torch.tensor([0.1, 0.5, 1.0], dtype=F64), rtol=1e-12)


def test_quadratic_source_flux():
    rhox = _grid()
    tau = optical_depth(rhox, torch.ones_like(rhox))
    r = Transfer(6, scattering="absorption").solve(rhox, torch.ones_like(rhox), torch.zeros_like(rhox),
                                                   1 + 1.5 * tau + 0.3 * tau ** 2)
    assert abs(float(r.flux) / (0.5 * (0.5 + 1.5 / 3 + 0.3 * 0.5)) - 1) < 1e-12


def test_sqrt_epsilon_law():
    """Constant B and photon destruction probability eps: S(0) = sqrt(eps) B (semi-infinite atmosphere)."""
    rhox = _grid(72, -8, 4)
    for eps, tol in ((0.1, 5e-4), (0.01, 3e-3)):
        r = Transfer(8).solve(rhox, torch.full_like(rhox, eps), torch.full_like(rhox, 1 - eps), torch.ones_like(rhox))
        assert abs(float(r.S[0]) / math.sqrt(eps) - 1) < tol
        assert abs(float(r.S[-1]) - 1) < 1e-10


def _solar_like(W=6, L=40):
    torch.manual_seed(1)
    rhox = _grid(L, -3.3, 3.0)
    T = 4000 + 6000 * torch.sigmoid(torch.log10(rhox) - 0.5)
    ab = (1e-3 * rhox ** 0.8 * (T / 5000) ** 6)[None] * (1 + 50 * torch.rand(W, 1, dtype=F64) * torch.exp(-rhox))
    sc = (0.05 * torch.ones_like(rhox))[None].expand(W, L) * torch.rand(W, 1, dtype=F64)
    B = (1 / torch.expm1(3e4 / T))[None].expand(W, L) * (1 + 0.1 * torch.rand(W, 1, dtype=F64))
    return rhox, ab, sc, B


def test_ali_matches_direct():
    rhox, ab, sc, B = _solar_like()
    sc = sc * 18.0                                                 # albedo up to ~0.9 at the top
    d = Transfer(6, method="direct").solve(rhox, ab, sc, B)
    a = Transfer(6, method="ali").solve(rhox, ab, sc, B)
    assert torch.allclose(a.S, d.S, rtol=1e-11) and torch.allclose(a.flux, d.flux, rtol=1e-12)


def test_batched_models_and_wavelengths():
    rhox, ab, sc, B = _solar_like(W=4)
    rh2 = torch.stack([rhox, rhox * 1.3])[:, None, :]              # 2 models x 4 wavelengths
    ab2, sc2, B2 = (torch.stack([v, v * 0.7]) for v in (ab, sc, B))
    rt = Transfer(6)
    r = rt.solve(rh2, ab2, sc2, B2)
    assert r.flux.shape == (2, 4)
    for m in range(2):
        assert torch.allclose(r.flux[m], rt.solve(rh2[m, 0], ab2[m], sc2[m], B2[m]).flux, rtol=1e-13)


def test_gradients():
    rhox, ab, sc, B = _solar_like(W=2, L=10)
    la = ab.log().requires_grad_(True)                            # gradcheck's absolute step: use log opacity
    B = B.clone().requires_grad_(True)
    f = lambda a, b: Transfer(4, method="direct").solve(rhox, a.exp(), sc, b).flux
    assert torch.autograd.gradcheck(f, (la, B))
    g_ali = torch.autograd.grad(Transfer(4).solve(rhox, la.exp(), sc, B).flux.sum(), (la, B))
    g_dir = torch.autograd.grad(f(la, B).sum(), (la, B))
    assert all(torch.allclose(x, y, rtol=1e-8, atol=1e-14) for x, y in zip(g_ali, g_dir))


def test_opaque_top_option():
    rhox, ab, sc, B = _solar_like(W=2)
    thick = ab * torch.tensor([[1.0], [1e12]], dtype=F64)           # second wavelength: tau_1 > 20
    f = Transfer(opaque_top="formal").solve(rhox, thick, sc, B)
    k = Transfer(opaque_top="diffusion").solve(rhox, thick, sc, B)
    assert float(f.tau[1, 0]) > 20 and float(f.tau[0, 0]) < 20
    assert float(f.flux[0]) == float(k.flux[0])
    assert float(k.flux[1]) < 1e-3 * float(f.flux[1])              # diffusion flux << the slab's emergent flux
