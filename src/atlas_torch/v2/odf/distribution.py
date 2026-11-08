"""Opacity distribution functions: the line opacity of each interval sorted into increasing order and averaged over
the 12 steps (Kurucz 1970 sec. 5.15, eqs. 5.23-5.24; Castelli 2005, MSAIS 8, 34, p. 35; dfsynthe.for DFCALC).
The step value is the mean of log10 kappa over the step (DFCALC averages 1000 log10 kappa)."""
from __future__ import annotations

import numpy as np
import torch

from .grid import step_bounds


def distribution_steps(log_kappa: torch.Tensor, interval: torch.Tensor, n_intervals: int, kind: str = "exact",
                       fractions=None) -> torch.Tensor:
    """Step means of log10 kappa (..., N) over the points of each interval: interval (N,) long, 0..n_intervals-1,
    non-decreasing (every interval present). Returns (..., n_intervals, 12) in increasing opacity.

    One sort per row with the composite key (interval, value); the step sums are differences of the cumulative
    sum at the step boundaries (fractional boundaries take the straddling point's share: exact quantiles of the
    step function of the sorted values). kind: step_bounds kind ('exact' | 'kurucz')."""
    from ..opacity.odf import STEP_FRACTIONS
    fr = STEP_FRACTIONS if fractions is None else fractions
    dev = log_kappa.device
    counts = torch.bincount(interval, minlength=n_intervals)
    start = torch.cumsum(counts, 0) - counts
    span = float((log_kappa.detach().amax() - log_kappa.detach().amin()).clamp(min=1.0)) * 2.0
    key = log_kappa.detach() + interval.to(log_kappa.dtype) * span
    order = torch.sort(key, dim=-1).indices
    srt = torch.gather(log_kappa, -1, order)
    cs = torch.cat([torch.zeros_like(srt[..., :1]), torch.cumsum(srt, -1)], -1)          # (..., N+1)
    b = torch.as_tensor(step_bounds(counts.cpu().numpy(), kind, fr), dtype=log_kappa.dtype, device=dev)
    pos = start.to(log_kappa.dtype)[:, None] + b                                           # (I, 13) absolute
    lo = pos.floor().long().clamp(max=cs.shape[-1] - 1)
    frac = pos - lo.to(pos.dtype)
    nxt = (lo + 1).clamp(max=cs.shape[-1] - 1)
    c_lo = cs[..., lo.reshape(-1)].reshape(cs.shape[:-1] + lo.shape)
    c_hi = cs[..., nxt.reshape(-1)].reshape(cs.shape[:-1] + lo.shape)
    s = c_lo + frac * (c_hi - c_lo)                                                         # S(pos)
    width = (b[:, 1:] - b[:, :-1])
    return (s[..., 1:] - s[..., :-1]) / width.clamp(min=1e-300)


def dfcalc_round(step_sum: np.ndarray, count: np.ndarray) -> np.ndarray:
    """DFCALC's integer step value (ISUM + N/2)/N with Fortran integer division (truncation toward zero)."""
    num = np.asarray(step_sum, dtype=np.int64) + np.asarray(count, dtype=np.int64) // 2
    return (np.sign(num) * (np.abs(num) // np.asarray(count, dtype=np.int64))).astype(np.int64)


__all__ = ["distribution_steps", "dfcalc_round"]
