"""SYNTHE's wavelength grid: lambda_i = (1 + 1/R)^i nm for consecutive integers i (synbeg.for)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LogWavelengthGrid:
    """Points lambda_k = exp((first + k) ln(1 + 1/resolution)) nm, k = 0..length-1 (vacuum)."""
    resolution: float
    first: int
    length: int

    @classmethod
    def window(cls, wl_begin: float, wl_end: float, resolution: float = 600000.0) -> "LogWavelengthGrid":
        """The points strictly inside [wl_begin, wl_end) as SYNBEG chooses them (first point >= wl_begin, last
        point < wl_end)."""
        step = math.log(1.0 + 1.0 / resolution)
        i0 = int(math.log(wl_begin) / step)
        if math.exp(i0 * step) < wl_begin:
            i0 += 1
        i1 = int(math.log(wl_end) / step)
        if math.exp(i1 * step) >= wl_end:
            i1 -= 1
        return cls(float(resolution), i0, i1 - i0 + 1)

    @property
    def step(self) -> float:
        """ln(lambda_{k+1}/lambda_k)."""
        return math.log(1.0 + 1.0 / self.resolution)

    def wavelengths(self, device=None) -> torch.Tensor:
        k = torch.arange(self.length, dtype=torch.float64, device=device)
        return torch.exp((self.first + k) * self.step)

    def position(self, wavelength: torch.Tensor) -> torch.Tensor:
        """Fractional grid index k of `wavelength` (nm)."""
        return torch.log(wavelength) / self.step - self.first

    @property
    def wl_begin(self) -> float:
        return math.exp(self.first * self.step)

    @property
    def wl_end(self) -> float:
        return math.exp((self.first + self.length - 1) * self.step)
