"""v2 structure modules in isolation: depth-grid calculus, hydrostatics, transfer moments, Lambda diagonal,
ODF truncation, mixing-length closed forms, thermodynamic derivatives."""
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.v2.rt import Transfer
from atlas_torch.v2.rt.optical_depth import cumulative_integral
from atlas_torch.v2.structure.convection import _kurucz_series, bubble_gradient_excess
from atlas_torch.v2.structure.grid import derivative, map1
from atlas_torch.v2.structure.hydrostatic import gas_pressure, pressure_on_tau
from atlas_torch.v2.structure.radiation import _truncate, expint3, kurucz_lambda_diagonal

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
SUN = EXT / "runs" / "stock" / "atlas" / "sun"
SOLAR_ODF = EXT / "runs" / "stock" / "odf" / "solar"
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def test_calculus_on_polynomials():
    x = torch.linspace(0.1, 3.0, 40).expand(3, 40)
    f = 2.0 + 0.5 * x - 0.3 * x ** 2
    torch.testing.assert_close(derivative(x, f)[:, 1:-1], (0.5 - 0.6 * x)[:, 1:-1], rtol=1e-12, atol=1e-12)
    # interior intervals integrate parabolas exactly; the two end intervals are linear (trapezoid)
    dF = torch.diff(cumulative_integral(x, f, initial=1.0), dim=-1)
    prim = 2.0 * x + 0.25 * x ** 2 - 0.1 * x ** 3
    torch.testing.assert_close(dF[:, 1:-1], torch.diff(prim, dim=-1)[:, 1:-1], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(cumulative_integral(x, 3.0 * x, initial=1.0), 1.0 + 1.5 * (x ** 2 - 0.01),
                               rtol=1e-13, atol=1e-13)
    # atlas DERIV: exact for straight lines
    torch.testing.assert_close(derivative(x, 3.0 * x + 1.0, "atlas"), torch.full_like(x, 3.0), rtol=1e-12, atol=0)


def test_map1_batched_equals_1d():
    from atlas_torch.v2.opacity.rosseland import map1 as map1_1d
    xp = torch.sort(torch.rand(30) * 10)[0]
    fp = torch.sin(xp)
    x = torch.linspace(-1, 11, 57)
    ref = map1_1d(x, xp, fp)
    got = map1(x.expand(2, 57), xp.expand(2, 30), fp.expand(2, 30))
    torch.testing.assert_close(got[1], ref, rtol=1e-14, atol=1e-14)


def test_gas_pressure_is_algebraic():
    rhox = torch.logspace(-4, 1, 10)[None]
    prad = 1e-3 * rhox
    p = gas_pressure(rhox, torch.tensor([1e4]), prad, 0.5 * prad)
    torch.testing.assert_close(p, 1e4 * rhox - prad - 0.5 * prad, rtol=1e-15, atol=0)


def test_pressure_on_tau_power_law():
    """kappa = k0 P^b: P^(1+b) = (g / k0)((1+b) tau - b tau_1), the solution with P(tau_1) = g tau_1 / kappa_1."""
    lt = torch.linspace(-6, 2, 65)[None]
    k0, b, g = 1e-3, 0.6, torch.tensor([2.75e4])
    kap = lambda T, P: k0 * P ** b                                        # noqa: E731
    P, pt, rhox = pressure_on_tau(lt, torch.full_like(lt, 5000.0), g, kap)
    tau = 10.0 ** lt
    exact = (g / k0 * ((1 + b) * tau - b * tau[:, :1])) ** (1 / (1 + b))
    # third-order quadrature in ln tau; the first interval (from the constant-opacity top) is linear
    torch.testing.assert_close(pt[:, :10], exact[:, :10], rtol=2e-3, atol=0)
    torch.testing.assert_close(pt[:, 10:], exact[:, 10:], rtol=3e-4, atol=0)
    torch.testing.assert_close(rhox * g, pt, rtol=1e-14, atol=0)


def test_pressure_on_tau_against_ttaup_listing():
    """Start model of the stock sun (restart T scaled to 5777 K, kappa.ros) against READIN's printed TTAUP result."""
    if not (SUN / "output_main.out").exists() or not (SOLAR_ODF / "kappa.ros").exists():
        pytest.skip("stock files missing")
    from atlas_torch.v2.io.atlas_control import read_control, read_restart
    from atlas_torch.v2.opacity.rosseland import RosselandTable
    c, r = read_control(SUN / "atlas_control.com"), read_restart(SUN / "restart.dat")
    tab = RosselandTable.read(SOLAR_ODF / "kappa.ros")
    lt = torch.as_tensor(c.log_tau())
    T = map1(10.0 ** lt, r.tau_ross(), r.atmosphere.T) * c.teff / float(r.atmosphere.teff)
    P, _, rhox = pressure_on_tau(lt[None], T[None], torch.tensor([10.0 ** c.logg]),
                                 lambda t, p: tab.kappa(t, p, c.vturb), scheme="atlas")
    lines = (SUN / "output_main.out").read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if "RHOX         T        P        XNE" in ln)
    ref = np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[i + 1:i + 73]])
    np.testing.assert_allclose(T.numpy(), ref[:, 1], atol=0.051)
    np.testing.assert_allclose(P[0].numpy(), ref[:, 2], rtol=1.5e-3)
    np.testing.assert_allclose(rhox[0].numpy(), ref[:, 0], rtol=1.5e-3)


def test_transfer_moments_eddington():
    tau = torch.logspace(-6, 3, 72)
    S = 1.0 + 1.5 * tau
    for method in ("direct", "ali"):
        sol = Transfer(method=method).solve(tau, torch.ones(72), 0.3 * torch.ones(72), S, moments=True)
        so = Transfer(method=method).solve(tau, torch.ones(72), torch.zeros(72), S, moments=True)
        torch.testing.assert_close(so.H[-10:], torch.full((10,), 0.5), rtol=1e-9, atol=0)    # H = (1/3) dS/dtau
        assert float(so.K0) == pytest.approx(1.0 / 6.0 + 1.5 / 8.0, rel=1e-6)               # K(0) = a/6 + b/8
        assert float(so.flux) == pytest.approx(float(so.H[0]), rel=1e-3)
        assert torch.isfinite(sol.lam_diag).all() and float(sol.lam_diag[-1]) == pytest.approx(1.0, abs=1e-4)
    d = Transfer(method="direct", deep="diffusion").solve(tau, torch.ones(72), torch.zeros(72), S, moments=True)
    torch.testing.assert_close(d.J[-10:], S[-10:], rtol=1e-12, atol=0)


def test_expint3_and_kurucz_diagonal():
    x = torch.tensor([0.02, 0.5, 1.0, 3.0, 9.9])
    ref = torch.tensor([0.48096829147697, 0.22160436427518, 0.10969196719776, 0.0089306465560227,
                        3.9533493884138e-06])                                                  # scipy expn(3, x)
    torch.testing.assert_close(expint3(x), ref, rtol=1e-6, atol=1e-15)
    tau = torch.logspace(-5, 2, 30)
    lam = kurucz_lambda_diagonal(tau)
    assert float(lam[0]) < 1e-3 and float(lam[-1]) == pytest.approx(1.0, abs=0.02)
    assert torch.all(lam > 0) and torch.all(lam <= 1.0 + 1e-12)


def test_odf_truncation_rule():
    from atlas_torch.v2.opacity.frequencies import SampleSet
    k = 12
    step = torch.arange(k)
    sub = SampleSet(torch.ones(k), torch.full((k,), 1.0 / 12), torch.zeros(k, dtype=torch.long),
                    torch.zeros(k, dtype=torch.long), step)
    L = 5
    h_c = torch.ones(1, 1, L)
    # flux ratio rises with weaker steps; stored step 8 is the first (strongest-first order: 11, 10, 9, 8) to reach .998
    ratio = torch.where(step >= 9, 0.5, 0.999).to(torch.float64)
    h = ratio[None, :, None].expand(1, k, L).clone()
    w, rem = _truncate(sub, h, h_c, torch.full((1, k), 1.0 / 12))
    assert torch.equal(w[0] > 0, step >= 8)
    assert float(rem[0, 0]) == pytest.approx(8.0 / 12.0)


def test_mlt_closed_form():
    delta = torch.logspace(-6, 0, 30)
    for d in (1e-8, 1e-3, 0.1, 10.0):
        D = torch.full_like(delta, d)
        direct = D + delta - torch.sqrt(D ** 2 + 2.0 * D * delta)                           # eq. 6.15
        torch.testing.assert_close(bubble_gradient_excess(delta, D), direct, rtol=1e-6, atol=1e-14)
        q = (delta / (D + delta)) ** 2
        torch.testing.assert_close(_kurucz_series(q) * delta ** 2 / (D + delta), direct, rtol=1e-5, atol=1e-14)
    torch.testing.assert_close(bubble_gradient_excess(delta, torch.zeros_like(delta)), delta, rtol=1e-14, atol=0)


def test_thermodynamics_exact_against_differences():
    from atlas_torch.v2.atmosphere import Atmosphere
    from atlas_torch.v2.composition import Composition
    from atlas_torch.v2.eos import ChemicalEquilibrium
    from atlas_torch.v2.structure.thermodynamics import ThermoDerivatives
    comp = Composition.from_atlas(0.92, 0.078, np.full(97, -20.0))
    comp = Composition(comp.abundance.clone())
    comp.abundance[5], comp.abundance[7], comp.abundance[25] = 2.7e-4, 4.9e-4, 3.0e-5
    T = torch.tensor([[3800.0, 5000.0, 6500.0, 9000.0, 14000.0]])
    P = torch.tensor([[50.0, 3e3, 6e4, 1.5e5, 3e5]])
    eos = ChemicalEquilibrium()
    pr = 1e-3 * P
    pt = P + pr
    dil = torch.full_like(T, 0.7)
    pops = eos.solve(Atmosphere(rhox=torch.zeros_like(T), T=T, P=P), comp)
    ex = ThermoDerivatives.of(eos, T, P, comp, pt, pr, dil, "exact", pops=pops)
    fd = ThermoDerivatives.of(eos, T, P, comp, pt, pr, dil, "kurucz", pops=pops)
    for k in ("heat_cp", "heat_cv", "dlnrho_dlnT", "grad_ad", "sound_speed"):
        torch.testing.assert_close(getattr(ex, k), getattr(fd, k), rtol=2e-4, atol=0, msg=k)
    # ideal monatomic limit: hot, fully ionised hydrogen-like gas has grad_ad near 2/5 only without ionisation zones
    assert torch.all(ex.grad_ad > 0.05) and torch.all(ex.grad_ad < 0.41)


def test_transfer_on_xtau_grid():
    """grid="xtau" (JOSH's fixed depths) for S linear in tau: H(0) = a/4 + b/6, K(0) = a/6 + b/8, J = S and
    H = b/3 at depth, H and J - S as on the model's own depths; with scattering the same S as there."""
    tau = torch.logspace(-6, 3, 72)
    S = 1.0 + 1.5 * tau
    one, zero = torch.ones(72), torch.zeros(72)
    x = Transfer(grid="xtau").solve(tau, one, zero, S, moments=True)
    f = Transfer(grid="model").solve(tau, one, zero, S, moments=True)
    assert float(x.flux) == pytest.approx(0.25 + 1.5 / 6.0, rel=1e-9)
    assert float(x.K0) == pytest.approx(1.0 / 6.0 + 1.5 / 8.0, rel=1e-6)
    torch.testing.assert_close(x.J[tau > 20], S[tau > 20], rtol=1e-12, atol=0)
    torch.testing.assert_close(x.H[tau > 20], torch.full_like(x.H[tau > 20], 0.5), rtol=1e-12, atol=0)
    torch.testing.assert_close(x.H, f.H, rtol=2e-5, atol=0)
    torch.testing.assert_close(x.J - S, f.J - S, rtol=0, atol=1e-4)               # MAP1 back from XTAU
    m = Transfer(grid="model").solve(tau, one, 0.3 * one, S, moments=True)
    s = Transfer(grid="xtau").solve(tau, one, 0.3 * one, S, moments=True)
    torch.testing.assert_close(s.S, m.S, rtol=3e-5, atol=0)
    assert float(s.flux) == pytest.approx(float(m.flux), rel=1e-5)


def test_radiative_loss_forms():
    """eq. 6.14's D is 8 times CONVEC's for an opaque bubble (tau_b >> 1); the thin-bubble factor
    tau_b^2/(2 + tau_b^2) enters CONVEC's D squared."""
    from atlas_torch.v2.structure.convection import radiative_loss
    T, rho, hp, v0, f0 = (torch.tensor([[x]]) for x in (6000.0, 2e-7, 1.5e7, 2e5, 3e9))
    l = torch.tensor([1.25])
    for kappa, thin in ((1e3, 1.0), (1e-7, None)):
        k = torch.tensor([[kappa]])
        a = radiative_loss(T, rho, k, hp, v0, f0, l, 5.67e-5, "atlas9")
        p = radiative_loss(T, rho, k, hp, v0, f0, l, 5.67e-5, "kurucz1970")
        taub = kappa * float(rho) * 1.25 * float(hp)
        factor = (taub ** 2 / (2 + taub ** 2)) ** 2 if thin is None else 1.0
        assert float(p / a) == pytest.approx(8.0 / factor, rel=1e-6)
