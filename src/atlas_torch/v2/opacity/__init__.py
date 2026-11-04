"""Opacities on a (depth layer, frequency) grid: the result type shared by the continuum and line opacities.

    from atlas_torch.v2.opacity.continuum import ContinuousOpacity
    op = ContinuousOpacity()(pops, nu)          # Opacity, (..., L, F)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..constants import CODATA2018, PhysicalConstants


@dataclass(frozen=True)
class Opacity:
    """Mass extinction coefficients, cm^2 g^-1, shape (..., L, F) (batch, depth layers, frequencies).

    absorption  true absorption corrected for stimulated emission (Kurucz 1970 eq. 5.1, the kappa terms)
    scattering  coherent isotropic scattering, source function J_nu (eq. 5.1, the sigma terms)
    source      source function of `absorption` (erg s^-1 cm^-2 Hz^-1 sr^-1); None = Planck B_nu(T) (LTE)
    absorbers, scatterers   per-process contributions by name (summing to absorption, scattering)
    """
    absorption: torch.Tensor
    scattering: torch.Tensor
    source: torch.Tensor | None = None
    absorbers: dict = field(default_factory=dict)
    scatterers: dict = field(default_factory=dict)

    @property
    def total(self) -> torch.Tensor:
        return self.absorption + self.scattering

    @property
    def albedo(self) -> torch.Tensor:
        return self.scattering / self.total

    def per_volume(self, rho: torch.Tensor) -> "Opacity":
        """Extinction per unit volume (cm^-1), rho (..., L) in g cm^-3."""
        r = rho[..., None]
        return Opacity(self.absorption * r, self.scattering * r, self.source,
                       {k: v * r for k, v in self.absorbers.items()}, {k: v * r for k, v in self.scatterers.items()})


def combine(parts: list[Opacity], planck: torch.Tensor | None = None) -> Opacity:
    """Sum of opacities; the source function is the absorption-weighted mean (Kurucz 1970 eq. 5.2), which needs
    planck (B_nu, (..., L, F)) when any part has a non-LTE source."""
    a = sum(p.absorption for p in parts)
    s = sum(p.scattering for p in parts)
    src = None
    if any(p.source is not None for p in parts):
        if planck is None:
            raise ValueError("combining a non-LTE source function needs planck")
        num = sum(p.absorption * (planck if p.source is None else p.source) for p in parts)
        src = torch.where(a > 0, num / a, planck)
    absorbers, scatterers = {}, {}
    for p in parts:
        for d, new in ((p.absorbers, absorbers), (p.scatterers, scatterers)):
            for k, v in d.items():
                new[k] = new[k] + v if k in new else v
    return Opacity(a, s, src, absorbers, scatterers)


def h_nu_over_kt(nu: torch.Tensor, T: torch.Tensor, const: PhysicalConstants = CODATA2018) -> torch.Tensor:
    """h nu / kT, (..., L, F) from T (..., L) and nu (F,) in Hz."""
    return (const.h / const.k) * nu / T[..., None]


def stimulated_emission(nu: torch.Tensor, T: torch.Tensor, const: PhysicalConstants = CODATA2018) -> torch.Tensor:
    """1 - exp(-h nu / kT), the LTE stimulated-emission factor of an absorption coefficient, (..., L, F)."""
    return -torch.expm1(-h_nu_over_kt(nu, T, const))


def planck(nu: torch.Tensor, T: torch.Tensor, const: PhysicalConstants = CODATA2018) -> torch.Tensor:
    """B_nu(T) = 2 h nu^3 / c^2 / (exp(h nu/kT) - 1), erg s^-1 cm^-2 Hz^-1 sr^-1, (..., L, F)."""
    return 2.0 * const.h * nu ** 3 / const.c ** 2 / torch.expm1(h_nu_over_kt(nu, T, const))


__all__ = ["Opacity", "combine", "h_nu_over_kt", "stimulated_emission", "planck"]
