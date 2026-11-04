"""Derivatives of the equilibrium (implicit function theorem) against central finite differences."""
import pytest
import torch
from torch.func import jvp


@pytest.fixture(scope="module")
def setup(solar_deck):
    from atlas_torch.v2.eos import ChemicalEquilibrium
    idx = torch.tensor([0, 30, 55, 71])
    atm = solar_deck.atmosphere
    atm = atm.with_(T=atm.T[idx], P=atm.P[idx], rhox=atm.rhox[idx])
    return ChemicalEquilibrium(), atm, solar_deck.composition.abundance


def _outputs(eq, atm, T, P, a):
    from atlas_torch.v2 import Composition
    p = eq.solve(atm.with_(T=T, P=P), Composition(a))
    return torch.stack([p.n_e.log(), p.rho.log(), p.n_over_u("Fe I").log(), p.n_over_u("Ca II").log(),
                        p.n_over_u(608.0).log(), p.n_over_u(108.0).log(), p.density("H I").log()])


def test_reverse_mode(setup):
    eq, atm, a0 = setup
    T = atm.T.clone().requires_grad_()
    P = atm.P.clone().requires_grad_()
    a = a0.clone().requires_grad_()
    w = torch.linspace(0.5, 1.5, 7)[:, None]
    (w * _outputs(eq, atm, T, P, a)).sum().backward()
    h = 1e-5
    f = lambda T_, P_, a_: (w * _outputs(eq, atm, T_, P_, a_)).sum()
    for i in range(4):
        e = torch.zeros(4)
        e[i] = h * atm.T[i]
        fd = (f(atm.T + e, atm.P, a0) - f(atm.T - e, atm.P, a0)) / (2 * e[i])
        assert abs(float(T.grad[i]) / float(fd) - 1) < 1e-6
        e = torch.zeros(4)
        e[i] = h * atm.P[i]
        fd = (f(atm.T, atm.P + e, a0) - f(atm.T, atm.P - e, a0)) / (2 * e[i])
        assert abs(float(P.grad[i]) / float(fd) - 1) < 1e-6
    for z in (1, 2, 6, 8, 26):
        e = torch.zeros(99)
        e[z - 1] = h * a0[z - 1]
        fd = (f(atm.T, atm.P, a0 + e) - f(atm.T, atm.P, a0 - e)) / (2 * e[z - 1])
        assert abs(float(a.grad[z - 1]) / float(fd) - 1) < 1e-5


def test_forward_mode(setup):
    eq, atm, a0 = setup
    h = 1e-5
    _, tan = jvp(lambda T: _outputs(eq, atm, T, atm.P, a0), (atm.T,), (atm.T,))
    fd = (_outputs(eq, atm, atm.T * (1 + h), atm.P, a0) - _outputs(eq, atm, atm.T * (1 - h), atm.P, a0)) / (2 * h)
    assert ((tan - fd).abs() / fd.abs().clamp(min=1e-3)).max() < 1e-5
