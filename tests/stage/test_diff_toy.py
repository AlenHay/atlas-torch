"""Fast-tier toy checks of atlas_torch.diff (forward-mode AD vs finite differences), < 10 s in total.

1. NMOLEC implicit differentiation on the first layers of the solar SYNTHE deck: the primal is the exact
   path's iterate (bit-identical XNE), the tangents of XNE/XNATOM w.r.t. P and an abundance match central
   differences of the polished (machine-precision) root.
2. JOSH (float64 direct scattering solve) on a synthetic atmosphere: tangent vs FD, decisions frozen.
3. VOIGT with straight-through tables: values are the REAL*4 table values, tangents those of the smooth
   functions.
4. The fixed-point IFT algebra (tangents, chord Newton) on a linear toy map with a known answer.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwad

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
DECK = EXT / "runs" / "stock" / "atlas" / "sun" / "output_synthe.out"
D = torch.float64


def _need(p):
    if not Path(p).exists():
        pytest.skip(f"missing {p}")
    return p


@pytest.fixture(scope="module")
def small_state():
    """READIN state of the solar SYNTHE deck truncated to its first 6 layers."""
    from atlas_torch.diff import xnf
    base = xnf.Base.from_deck(_need(DECK))
    s = base.state0.updated({"rhox.nrhox": torch.full_like(base.state0["rhox.nrhox"], 6)})
    return base, s


def test_nmolec_ift_primal_and_tangent(small_state):
    from atlas_torch.diff import nmolec as NM, xnf
    from atlas_torch.diff.native import native_math
    from atlas_torch.synthe.pops import nmolec as nmolec_exact
    base, s0 = small_state
    plan = NM.build_plan(s0)
    n = 6
    P0 = s0["state.p"][:n].to(D).clone()
    ex = NM.solve_exact(s0, plan)
    ref = nmolec_exact(s0)
    assert np.array_equal(ex.xn[:, -1], ref["state.xne"][:n].numpy())      # bit-identical iterate

    def run(params, polish):
        b = xnf.Base(**{**base.__dict__, "state0": s0, "n": n, "exact": {}})
        s = xnf.inject(b, params)
        e = xnf.exact_nmolec(b, s)
        with native_math():
            att = NM.attach(plan, s, e, polish=polish)
        return att["xne"], att["xnatom"]

    tanP = 1e-3 * P0

    def tangent(polish):
        with fwad.dual_level():
            xne, xna = run({"P": fwad.make_dual(P0.clone(), tanP.clone()),
                            "abund": {"Fe": fwad.make_dual(torch.tensor(0.0, dtype=D), torch.tensor(0.5, dtype=D))}},
                           polish=polish)
            return fwad.unpack_dual(xne).tangent, fwad.unpack_dual(xna).tangent, fwad.unpack_dual(xne).primal
    t0_xne, _, p_xne = tangent(False)
    assert torch.equal(p_xne, torch.as_tensor(ex.xn[:, -1]))                 # primal = exact iterate
    t_xne, t_xna, _ = tangent(True)
    # the IFT tangent at the exact iterate differs from the one at the exact root by NMOLEC's convergence
    assert float(((t0_xne - t_xne).abs() / t_xne.abs()).max()) < 1e-4
    h = 1e-4
    xp, ap = run({"P": P0 + h * tanP, "abund": {"Fe": torch.tensor(0.5 * h, dtype=D)}}, polish=True)
    xm, am = run({"P": P0 - h * tanP, "abund": {"Fe": torch.tensor(-0.5 * h, dtype=D)}}, polish=True)
    fd_xne, fd_xna = (xp - xm) / (2 * h), (ap - am) / (2 * h)
    assert float(((t_xne - fd_xne).abs() / fd_xne.abs()).max()) < 1e-6
    assert float(((t_xna - fd_xna).abs() / fd_xna.abs()).max()) < 1e-6


def _toy_atmosphere(B=8, n=72):
    rhox = torch.logspace(-4, 2, n, dtype=D)
    lam = torch.linspace(0.0, 1.0, B, dtype=D)[:, None]
    acont = 0.3 * (1 + lam) * rhox[None, :] ** 0.4
    sigmac = 0.02 * acont * torch.exp(-rhox[None, :])
    aline = 5.0 * (lam + 0.1) * rhox[None, :] ** 0.1 * torch.exp(-rhox[None, :] / 10)
    bnu = 1e-5 * (1.0 + 0.3 * torch.log10(rhox[None, :] / 1e-4)) ** 4 * torch.ones(B, 1, dtype=D)
    return rhox, acont, sigmac, aline, bnu


def test_josh_float64_tangent():
    from atlas_torch.diff.rt import Josh
    try:
        josh = Josh()
    except Exception as e:                                   # tables come from the fetched Fortran sources
        pytest.skip(f"JOSH tables unavailable: {e}")
    rhox, acont, sigmac, aline, bnu = _toy_atmosphere()
    zero = torch.zeros_like(acont)
    tan = 0.1 * aline + 0.05 * acont
    dec = {}
    josh.flux(acont, bnu, aline, bnu, sigmac, zero, rhox, 1, dec)                 # record decisions
    with fwad.dual_level():
        a = fwad.make_dual(aline.clone(), tan.clone())
        t = fwad.unpack_dual(josh.flux(acont, bnu, a, bnu, sigmac, zero, rhox, 1, dec)).tangent
    h = 1e-6
    fd = (josh.flux(acont, bnu, aline + h * tan, bnu, sigmac, zero, rhox, 1, dec)
          - josh.flux(acont, bnu, aline - h * tan, bnu, sigmac, zero, rhox, 1, dec)) / (2 * h)
    assert float((t - fd).abs().max() / fd.abs().max()) < 1e-6


def test_voigt_straight_through():
    from atlas_torch.diff.lines import Tables, voigt64
    from atlas_torch.synthe.lineopac.voigt import voigt as voigt32
    try:
        tabs = Tables()
    except Exception as e:
        pytest.skip(f"SYNTHE tables unavailable: {e}")
    v = torch.linspace(0.001, 6.001, 301, dtype=D)        # (no point exactly on a branch boundary)
    for a0 in (0.05, 0.5, 2.0):
        a = torch.full_like(v, a0)
        ref = voigt32(v.to(torch.float32), a.to(torch.float32), tabs.h32).to(D)
        val = voigt64(v, a, tabs, None, "exact")
        assert float(((val - ref).abs() / ref.abs()).max()) < 2e-6           # the table values (float32 ops)
        with fwad.dual_level():
            vd = fwad.make_dual(v.clone(), torch.ones_like(v))
            t_exact = fwad.unpack_dual(voigt64(vd, a, tabs, None, "exact")).tangent
            vs = fwad.make_dual(v.clone(), torch.ones_like(v))
            t_smooth = fwad.unpack_dual(voigt64(vs, a, tabs, None, "smooth")).tangent
        # straight-through: the derivative of the smooth functions, evaluated with the table values (they
        # agree to the tables' resolution, 1/200 Doppler width)
        assert torch.allclose(t_exact, t_smooth, rtol=5e-3, atol=2e-4)
    # exp(-v^2) derivative in the small-A branch
    a = torch.full_like(v, 1e-3)
    with fwad.dual_level():
        vd = fwad.make_dual(v.clone(), torch.ones_like(v))
        t = fwad.unpack_dual(voigt64(vd, a, tabs, None, "smooth")).tangent
    expect = -2 * v * torch.exp(-v * v)
    assert float((t - expect).abs()[v < 4].max()) < 2e-2                     # H1 term (a=1e-3) aside


class _LinearMap:
    """G(x, theta) = A x + B theta + c on the RHOX/T slots of a 6n+1 vector (the other slots: copies)."""

    def __init__(self, n=4, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n
        N = 6 * n + 1
        self.N = N
        M = rng.normal(size=(N, N)) / np.sqrt(N)
        self.A = 0.6 * M / np.max(np.abs(np.linalg.eigvals(M)))
        self.B = rng.normal(size=(N, 3))
        self.c = 5.0 + rng.normal(size=N)
        self.theta0 = np.zeros(3)

    def G(self, x, theta=None):
        theta = self.theta0 if theta is None else theta
        return self.A @ x + self.B @ np.asarray(theta) + self.c


def test_ift_algebra_linear_map():
    from atlas_torch.diff.atlas9_ift import Atlas9FixedPoint, Jac, chord_newton
    m = _LinearMap()
    I = np.eye(m.N)
    xstar = np.linalg.solve(I - m.A, m.c)
    J = Jac(xstar, m.theta0, m.G(xstar), m.A.copy(), m.B.copy(), np.ones(m.N), np.ones(3), 0.0)
    dx = Atlas9FixedPoint.tangents(J)
    assert np.allclose(dx, np.linalg.solve(I - m.A, m.B), rtol=1e-10, atol=1e-12)
    th = np.array([0.3, -0.2, 0.1])
    x, hist = chord_newton(m, J, xstar, th, n=5, tol=1e-13)
    assert np.allclose(x, np.linalg.solve(I - m.A, m.c + m.B @ th), rtol=1e-12)
    assert len(hist) <= 3
