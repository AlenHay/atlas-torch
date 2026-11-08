"""Monochromatic optical depth on the column-mass grid (Kurucz 1970 sec. 2.5); integrals and derivatives on a depth
grid."""
from __future__ import annotations

import torch


def _curvature(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Second divided difference f[x_{j-1}, x_j, x_{j+1}] (the c of the parabola through three points, eq. 2.46c),
    for the interior points j = 1..L-2: shape (..., L-2)."""
    h0 = x[..., 1:-1] - x[..., :-2]
    h1 = x[..., 2:] - x[..., 1:-1]
    return ((f[..., 2:] - f[..., 1:-1]) / h1 - (f[..., 1:-1] - f[..., :-2]) / h0) / (h0 + h1)


def cumulative_integral(x: torch.Tensor, f: torch.Tensor, *, initial=0.0, scheme: str = "parabolic") -> torch.Tensor:
    """F_j = initial + int_{x_1}^{x_j} f dx, shapes (..., L) (x increasing or decreasing along the last axis).

    Each interval [x_j, x_j+1] integrates the mean of the backward (x_j-1, x_j, x_j+1) and forward (x_j, x_j+1,
    x_j+2) parabolas weighted inversely by their curvatures (Kurucz 1970 eqs. 2.45-2.48); the two end intervals,
    which have only one parabola, are linear. scheme="atlas": as atlas9mem.for INTEG/PARCOE, which also makes the
    second and third intervals linear."""
    if scheme not in ("parabolic", "atlas"):
        raise ValueError("scheme: 'parabolic' | 'atlas'")
    shape = torch.broadcast_shapes(x.shape, f.shape)
    x, f = x.expand(shape), f.expand(shape)
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
        d = trapezoid + corr
    else:
        d = trapezoid
    first = torch.as_tensor(initial, dtype=f.dtype, device=f.device).expand(shape[:-1])[..., None]
    return torch.cat([first, first + torch.cumsum(d, -1)], -1)


def optical_depth(rhox: torch.Tensor, opacity: torch.Tensor, *, scheme: str = "parabolic") -> torch.Tensor:
    """tau_j = int_0^{rhox_j} opacity d(rhox) (Kurucz 1970 eq. 2.44), opacity in cm^2 g^-1, shapes (..., L).

    The top layer gets tau_1 = opacity_1 * rhox_1 (opacity constant above it; atlas7v.for JOSH,
    CALL INTEG(RHOX,ABTOT,TAUNU,NRHOX,ABTOT(1)*RHOX(1))); below it cumulative_integral (scheme "parabolic" or
    "atlas" = atlas7v.for PARCOE).
    """
    x = rhox.expand(opacity.shape) if rhox.shape != opacity.shape else rhox
    return cumulative_integral(x, opacity, initial=opacity[..., 0] * x[..., 0], scheme=scheme)


def derivative(x: torch.Tensor, f: torch.Tensor, scheme: str = "parabolic") -> torch.Tensor:
    """df/dx on the nodes, (..., L): "parabolic": slope of the parabola through j-1, j, j+1 (one-sided chords at
    the ends); "atlas": atlas9mem.for DERIV, the bisector of the two chord angles after scaling f by
    max(|f_j-1|, |f_j|, |f_j+1|)/|x_j| (one-sided chords at the ends)."""
    x, f = torch.broadcast_tensors(x, f)
    h = x[..., 1:] - x[..., :-1]
    s = (f[..., 1:] - f[..., :-1]) / h                                     # chords (..., L-1)
    if scheme == "parabolic":
        h0, h1 = h[..., :-1], h[..., 1:]
        mid = (s[..., :-1] * h1 + s[..., 1:] * h0) / (h0 + h1)
    elif scheme == "atlas":
        scale = torch.maximum(torch.maximum(f[..., :-2].abs(), f[..., 1:-1].abs()), f[..., 2:].abs()) / x[..., 1:-1].abs()
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        sg = torch.sign(h[..., :1])
        d1, d0 = s[..., 1:] / scale, s[..., :-1] / scale
        t1 = d1 / (sg * torch.sqrt(1.0 + d1 ** 2) + 1.0)
        t0 = d0 / (sg * torch.sqrt(1.0 + d0 ** 2) + 1.0)
        mid = (t1 + t0) / (1.0 - t1 * t0) * scale
    else:
        raise ValueError("scheme: 'parabolic' | 'atlas'")
    return torch.cat([s[..., :1], mid, s[..., -1:]], -1)
