"""Monochromatic optical depth on the column-mass grid (Kurucz 1970 sec. 2.5)."""
from __future__ import annotations

import torch


def _curvature(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Second divided difference f[x_{j-1}, x_j, x_{j+1}] (the c of the parabola through three points, eq. 2.46c),
    for the interior points j = 1..L-2: shape (..., L-2)."""
    h0 = x[..., 1:-1] - x[..., :-2]
    h1 = x[..., 2:] - x[..., 1:-1]
    return ((f[..., 2:] - f[..., 1:-1]) / h1 - (f[..., 1:-1] - f[..., :-2]) / h0) / (h0 + h1)


def optical_depth(rhox: torch.Tensor, opacity: torch.Tensor, *, scheme: str = "parabolic") -> torch.Tensor:
    """tau_j = int_0^{rhox_j} opacity d(rhox) (Kurucz 1970 eq. 2.44), opacity in cm^2 g^-1, shapes (..., L).

    The top layer gets tau_1 = opacity_1 * rhox_1 (opacity constant above it; atlas7v.for JOSH,
    CALL INTEG(RHOX,ABTOT,TAUNU,NRHOX,ABTOT(1)*RHOX(1))). Each interval [x_j, x_j+1] integrates the mean of the
    backward (x_j-1, x_j, x_j+1) and forward (x_j, x_j+1, x_j+2) parabolas weighted inversely by their curvatures
    (eqs. 2.45-2.48); the two end intervals, which have only one parabola, are linear.
    scheme="atlas": as atlas7v.for PARCOE, which also makes the second and third intervals linear.
    """
    if scheme not in ("parabolic", "atlas"):
        raise ValueError("scheme: 'parabolic' | 'atlas'")
    x = rhox.expand(opacity.shape) if rhox.shape != opacity.shape else rhox
    f = opacity
    h = x[..., 1:] - x[..., :-1]                                     # (..., L-1)
    trapezoid = 0.5 * h * (f[..., 1:] + f[..., :-1])
    if f.shape[-1] >= 4:
        c = _curvature(x, f)                                         # parabola centred on j = 1..L-2
        cb, cf = c[..., :-1].abs(), c[..., 1:].abs()                 # interval j = 1..L-3: backward, forward
        tot = cb + cf
        w = torch.where(tot > 0, cf / torch.where(tot > 0, tot, torch.ones_like(tot)), torch.ones_like(tot))
        cm = w * c[..., :-1] + (1.0 - w) * c[..., 1:]
        if scheme == "atlas":
            cm = torch.cat([torch.zeros_like(cm[..., :2]), cm[..., 2:]], -1)
        # a parabola through both interval ends integrates to the trapezoid - c h^3 / 6
        corr = torch.cat([torch.zeros_like(h[..., :1]), -cm * h[..., 1:-1] ** 3 / 6.0, torch.zeros_like(h[..., :1])], -1)
        dtau = trapezoid + corr
    else:
        dtau = trapezoid
    tau1 = f[..., :1] * x[..., :1]
    return torch.cat([tau1, tau1 + torch.cumsum(dtau, -1)], -1)
