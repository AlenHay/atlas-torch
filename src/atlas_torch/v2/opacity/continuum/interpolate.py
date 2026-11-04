"""Batched one-dimensional interpolation used by the opacity tables (abscissae increasing)."""
from __future__ import annotations

import torch


def _cell(x: torch.Tensor, xp: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """0-based index i of the node pair (i-1, i) with xp[i-1] <= x < xp[i], clamped to [lo, hi]."""
    return torch.searchsorted(xp.contiguous(), x.contiguous(), right=True).clamp(lo, hi)


def linear(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear f(x) through (xp, fp), extrapolated linearly from the end segments.
    xp (K,), fp (K,) or (..., K) with batch dims broadcasting against x."""
    i = _cell(x, xp, 1, xp.shape[-1] - 1)
    x0, x1 = xp[i - 1], xp[i]
    if fp.dim() == 1:
        f0, f1 = fp[i - 1], fp[i]
    else:
        shape = torch.broadcast_shapes(fp.shape[:-1], x.shape)
        fpb = fp.expand(shape + fp.shape[-1:])
        ib = i.expand(shape)
        f0 = torch.gather(fpb, -1, (ib - 1).unsqueeze(-1)).squeeze(-1)
        f1 = torch.gather(fpb, -1, ib.unsqueeze(-1)).squeeze(-1)
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0)


def _parabola(x, x0, x1, x2, f0, f1, f2):
    """Value and curvature (second divided difference) of the parabola through three points."""
    d01 = (f1 - f0) / (x1 - x0)
    d12 = (f2 - f1) / (x2 - x1)
    c = (d12 - d01) / (x2 - x0)
    return f0 + (x - x0) * (d01 + (x - x1) * c), c


def blended_parabola(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Kurucz's MAP1 interpolation (atlas7v.for FUNCTION MAP1) for xp (K,), fp (K,): on an interior interval
    (x_{i-1}, x_i) the parabolas through (i-2, i-1, i) and (i-1, i, i+1) averaged with weights inversely
    proportional to their curvature; the parabola through the last three nodes on the last interval; linear
    on the first two intervals and outside the table (extrapolation)."""
    k = xp.shape[0]
    i = torch.searchsorted(xp.contiguous(), x.contiguous(), right=True)  # nodes <= x
    lin = linear(x, xp, fp)
    ib = i.clamp(2, k - 1)
    g = lambda j: (xp[j], fp[j])                                          # noqa: E731
    (xa, fa), (xb, fb), (xc, fc) = g(ib - 2), g(ib - 1), g(ib)
    back, cb = _parabola(x, xa, xb, xc, fa, fb, fc)
    jf = ib.clamp(max=k - 2)
    (xd, fd), (xe, fe), (xf, ff) = g(jf - 1), g(jf), g(jf + 1)
    fwd, cf = _parabola(x, xd, xe, xf, fd, fe, ff)
    den = cf.abs() + cb.abs()
    wt = torch.where(den > 0, cf.abs() / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(den))
    inner = fwd + wt * (back - fwd)
    out = torch.where(i == k - 1, back, inner)
    return torch.where((i <= 2) | (i >= k), lin, out)


def linear_rows(x: torch.Tensor, xp: torch.Tensor, rows: torch.Tensor, clamp: bool = False) -> torch.Tensor:
    """Linear interpolation between the rows of a table: rows (K, F) tabulated at xp (K,), x (..., 1) -> (..., F);
    linear extrapolation beyond the table unless clamp (then the end rows)."""
    if clamp:
        x = torch.clamp(x, xp[0], xp[-1])
    i = _cell(x, xp, 1, xp.shape[0] - 1)[..., 0]
    t = ((x[..., 0] - xp[i - 1]) / (xp[i] - xp[i - 1]))[..., None]
    return rows[i - 1] + (rows[i] - rows[i - 1]) * t
