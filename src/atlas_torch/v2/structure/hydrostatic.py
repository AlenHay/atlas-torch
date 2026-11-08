"""Hydrostatic equilibrium, P_total = P_gas + P_rad + P_turb = g m (Kurucz 1970 sec. 4.1, eqs. 4.2-4.6), on a
column-mass grid or on a Rosseland optical-depth grid (sec. 4.2, eqs. 4.9-4.11)."""
from __future__ import annotations

import math
from typing import Callable

import torch

from ..rt.optical_depth import cumulative_integral
from ..solve import newton


def gas_pressure(rhox: torch.Tensor, gravity: torch.Tensor, p_rad: torch.Tensor, p_turb=0.0,
                 p_con: float = 0.0) -> torch.Tensor:
    """P_gas = g m - P_rad - P_turb - P_con (eq. 4.5), (..., L); where that is not positive, P_rad is counted from
    the top layer's value, and a still negative value takes the layer above's (atlas9mem.for main program loop 11)."""
    gm = gravity[..., None] * rhox
    p = gm - p_rad - p_turb - p_con
    p = torch.where(p > 0, p, gm - (p_rad - p_rad[..., :1]) - p_turb - p_con)
    if bool((p < 0).any()):
        cols = list(p.unbind(-1))
        for j in range(1, len(cols)):
            cols[j] = torch.where(cols[j] < 0, cols[j - 1], cols[j])
        p = torch.stack(cols, -1)
    return p


def height(rhox: torch.Tensor, rho: torch.Tensor, scheme: str = "parabolic") -> torch.Tensor:
    """Geometrical depth x = int dm / rho (eq. 4.46), cm, 0 at the first layer, increasing inward."""
    return cumulative_integral(rhox, 1.0 / rho, scheme=scheme)


def scale_height(p_total: torch.Tensor, rho: torch.Tensor, gravity: torch.Tensor) -> torch.Tensor:
    """Pressure scale height h = P_total / (rho g) (eq. 4.47), cm."""
    return p_total / (rho * gravity[..., None])


Kappa = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def pressure_on_tau(log_tau: torch.Tensor, T: torch.Tensor, gravity: torch.Tensor, kappa: Kappa, *, p_rad=None,
                    p_turb=None, p_total0: torch.Tensor | None = None, dlnk_dlnp: Kappa | None = None,
                    scheme: str = "parabolic", tol: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(P_gas, P_total, rhox), (..., L), in hydrostatic equilibrium on the Rosseland depths 10**log_tau (..., L)
    for T (..., L), gravity (...) and kappa_R = kappa(T, P_gas) (cm^2 g^-1): d ln P_total / d ln tau =
    g tau / (kappa P_total) (eq. 4.10), integrated in ln tau with cumulative_integral (scheme), and
    P_total(tau_1) = g tau_1 / kappa_1 (constant opacity above the first depth, eq. 4.11). P_total = g m here;
    P_gas = P_total - p_rad - p_turb.

    The whole profile is one Newton system in ln P_total (v2.solve.newton, gradients by the implicit function
    theorem); its Jacobian is that of the residual with ln kappa linear in ln P_gas, slope dlnk_dlnp(T, P_gas)
    (default: central difference of kappa)."""
    s = log_tau * math.log(10.0)
    tau = torch.exp(s)
    g = gravity[..., None].expand(T.shape)
    zero = torch.zeros_like(T)
    p_rad = zero if p_rad is None else p_rad.expand(T.shape)
    p_turb = zero if p_turb is None else p_turb.expand(T.shape)

    def p_gas(y, p_rad, p_turb):
        pt = torch.exp(y)
        return torch.clamp(pt - p_rad - p_turb, min=1e-6 * pt)

    def model(y, kap, s, g):
        f = g * torch.exp(s - y) / kap
        return cumulative_integral(s, f, initial=torch.log(g[..., 0] * torch.exp(s[..., 0]) / kap[..., 0]),
                                   scheme=scheme)

    def residual(y, s, T, g, p_rad, p_turb):
        return y - model(y, kappa(T, p_gas(y, p_rad, p_turb)), s, g)

    def slope(T, p):
        if dlnk_dlnp is not None:
            return dlnk_dlnp(T, p)
        h = 1e-4
        return (torch.log(kappa(T, p * math.exp(h))) - torch.log(kappa(T, p * math.exp(-h)))) / (2.0 * h)

    def linearize(y, params):
        s_, T_, g_, pr, pt = params
        p0 = p_gas(y, pr, pt)
        k0, b = kappa(T_, p0), slope(T_, p0)
        return lambda y2, s2, T2, g2, pr2, pt2: y2 - model(
            y2, k0 * torch.exp(b * (torch.log(p_gas(y2, pr2, pt2)) - torch.log(p0))), s2, g2)

    with torch.no_grad():
        if p_total0 is None:
            y = torch.log(g * tau / 0.1)
            for _ in range(4):                       # Picard sweeps from a constant-opacity guess
                y = model(y, kappa(T, p_gas(y, p_rad, p_turb)), s, g)
        else:
            y = torch.log(p_total0)
    y = newton(residual, y, (s, T, g, p_rad, p_turb), tol=tol, maxiter=40, max_step=2.0, linearize=linearize)
    pt = torch.exp(y)
    return p_gas(y, p_rad, p_turb), pt, pt / g
