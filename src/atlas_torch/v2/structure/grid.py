"""Depth-grid calculus of the structure: derivatives, interpolation between depth grids, the Rosseland grid.

    d = derivative(rhox, T)                         # dT/drhox, (..., L)
    T_std = map1(tau_std, tau_model, T)            # T on another grid
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..opacity.continuum.interpolate import map1  # noqa: F401  (re-exported)
from ..rt.optical_depth import derivative  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class RosselandGrid:
    """The fixed Rosseland optical-depth grid: log10 tau_j = log_tau1 + (j - 1) step (SCALE card, READIN 2700)."""
    n: int = 72
    log_tau1: float = -6.875
    step: float = 0.125

    def log_tau(self, dtype=torch.float64, device=None) -> torch.Tensor:
        return self.log_tau1 + self.step * torch.arange(self.n, dtype=dtype, device=device)

    def tau(self, dtype=torch.float64, device=None) -> torch.Tensor:
        return torch.pow(10.0, self.log_tau(dtype, device))


def linear(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation with batched increasing xp (..., K), linear extrapolation."""
    xp, fp = torch.broadcast_tensors(xp, fp)
    k = xp.shape[-1]
    batch = torch.broadcast_shapes(x.shape[:-1], xp.shape[:-1])
    x = x.expand(batch + x.shape[-1:])
    xp, fp = xp.expand(batch + (k,)), fp.expand(batch + (k,))
    i = torch.searchsorted(xp.detach().contiguous(), x.detach().contiguous(), right=True).clamp(1, k - 1)
    x0, x1 = torch.gather(xp, -1, i - 1), torch.gather(xp, -1, i)
    f0, f1 = torch.gather(fp, -1, i - 1), torch.gather(fp, -1, i)
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0)
