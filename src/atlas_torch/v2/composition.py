"""Chemical composition: number abundances per nucleus, ATLAS / BasicATLAS conventions."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .constants import N_ELEMENTS, atomic_masses


@dataclass(frozen=True)
class Composition:
    """abundance[..., Z-1] = N(Z) / N(all nuclei) (ATLAS XABUND); batch dims lead.

    scale: the ATLAS 'ABUNDANCE SCALE' (linear metallicity factor) the metal abundances are quoted
    relative to when written to a deck; it does not change `abundance`.
    """
    abundance: torch.Tensor
    scale: float = 1.0

    @property
    def batch_shape(self) -> torch.Size:
        return self.abundance.shape[:-1]

    def to(self, device) -> "Composition":
        return Composition(self.abundance.to(device), self.scale)

    def mean_atomic_mass(self) -> torch.Tensor:
        """Mean mass per nucleus in amu (ATLAS WTMOLE = sum XABUND*ATMASS), shape batch."""
        return (self.abundance * atomic_masses(self.abundance.device)).sum(-1)

    def mass_fractions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hydrogen, helium and metal mass fractions X, Y, Z."""
        m = self.abundance * atomic_masses(self.abundance.device)
        tot = m.sum(-1)
        return m[..., 0] / tot, m[..., 1] / tot, m[..., 2:].sum(-1) / tot

    # ---------------------------------------------------------------- ATLAS deck convention

    @classmethod
    def from_atlas(cls, h: float, he: float, log_metals, scale: float = 1.0) -> "Composition":
        """READIN abundance cards: H, He as number fractions (log10 if negative), metals Z=3..99 as log10 number
        fractions before the ABUNDANCE SCALE factor (atlas7v.for READIN labels 1510-1514)."""
        h = 10.0 ** h if h < 0 else h
        he = 10.0 ** he if he < 0 else he
        a = np.empty(N_ELEMENTS)
        a[0], a[1] = h, he
        a[2:] = 10.0 ** np.asarray(log_metals, dtype=np.float64) * scale
        return cls(torch.as_tensor(a), float(scale))

    def atlas_cards(self) -> tuple[float, float, np.ndarray]:
        """(H, He, log10 metals / scale) as written on ABUNDANCE CHANGE cards (single model)."""
        a = self.abundance.detach().cpu().numpy().reshape(-1, N_ELEMENTS)[0]
        return float(a[0]), float(a[1]), np.log10(a[2:] / self.scale)

    # ---------------------------------------------------------------- BasicATLAS convention

    @classmethod
    def from_basicatlas(cls, Y: float | None = None, zscale: float = 0.0, abun: dict | None = None,
                        solar_csv: str | Path | None = None) -> "Composition":
        """BasicATLAS Settings.abun_std_to_atlas: solar log eps (data/solar.csv) with [M/H]=zscale applied to
        Z>2, element enhancements abun={'C': +0.2}, helium set from the mass fraction Y (None = solar);
        metals floored at log10 fraction -20 before scaling (as BasicATLAS writes them)."""
        path = Path(solar_csv) if solar_csv else (Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
                                                  / "BasicATLAS" / "data" / "solar.csv")
        rows = np.genfromtxt(path, delimiter=",", dtype=None, encoding="utf-8", usecols=(0, 1, 2, 5))
        z = np.array([int(r[0]) for r in rows])
        mass = np.array([float(r[1]) for r in rows])
        log_eps = np.array([float(r[2]) for r in rows])
        sym = [str(r[3]).strip() for r in rows]
        for key, d in (abun or {}).items():
            if key in ("H", "He"):
                raise ValueError("H and He are set through Y, not abun")
            log_eps[sym.index(key)] += d
        n = 10.0 ** (log_eps + zscale * (z > 2))
        if Y is not None:
            he = z == 2
            n[he] = -np.sum(n[~he] * mass[~he]) * Y / (mass[he][0] * Y - mass[he][0])
        frac = n / n.sum()
        order = np.argsort(z)
        frac, z = frac[order], z[order]
        assert np.array_equal(z, np.arange(1, N_ELEMENTS + 1)), "solar.csv must list Z=1..99"
        log_metals = np.maximum(np.log10(frac[2:]) - zscale, -20.0)
        return cls.from_atlas(float(frac[0]), float(frac[1]), log_metals, 10.0 ** zscale)
