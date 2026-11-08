"""Microturbulent broadening of the 0 km/s opacity spectrum: convolution in velocity with the Gaussian
exp(-(dv/xi)^2) / (sqrt(pi) xi) of the microturbulent velocity xi (Castelli 2005, MSAIS 8, 34, p. 35: ODFs for
xi = 0, 1, 2, 4, 8 km/s; dfsynthe.for VTTAB and MAIN labels 411-420)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

C_KMS = 2.99792458e5


def turbulence_kernel(vturb_kms: float, resolution: float = 500000.0, kind: str = "kurucz",
                      floor: float = 1e-5) -> torch.Tensor:
    """Weights (2 n + 1,) of the grid offsets -n..n, summing to 1.
    'kurucz' (VTTAB): exp(-(i c/(R xi))^2) for i = 1.. up to and including the first value < 1e-5, divided by
    1 + 2 sum (velocity step c/R). 'exact': velocity step c ln(1 + 1/R) of the log-wavelength grid, cut below
    `floor`, normalised to unit sum."""
    if vturb_kms <= 0:
        return torch.ones(1, dtype=torch.float64)
    dv = C_KMS / resolution if kind == "kurucz" else C_KMS * math.log1p(1.0 / resolution)
    if kind not in ("kurucz", "exact"):
        raise ValueError("kind: 'kurucz' | 'exact'")
    n = 1
    while math.exp(-(n * dv / vturb_kms) ** 2) >= floor:
        n += 1
    i = torch.arange(1, n + 1, dtype=torch.float64)
    w = torch.exp(-(i * dv / vturb_kms) ** 2)
    k = torch.cat([w.flip(0), torch.ones(1, dtype=torch.float64), w])
    return k / k.sum()


def broaden(spectrum: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolution of spectrum (..., N) with kernel (2n+1,) on the grid ('valid' part: (..., N - 2n))."""
    shape = spectrum.shape
    x = spectrum.reshape(-1, 1, shape[-1])
    y = F.conv1d(x, kernel.to(x).flip(0)[None, None, :])
    return y.reshape(shape[:-1] + (y.shape[-1],))


__all__ = ["turbulence_kernel", "broaden"]
