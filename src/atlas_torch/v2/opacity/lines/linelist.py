"""Spectral line data as a structure of arrays (one entry per line), independent of the file format.

Conventions (Kurucz & Avrett 1981, SAO SR 391, pp. 3-4; gfall format, rgfalllinesnew.for header):
  wavelength      vacuum wavelength, nm
  species         Kurucz code: Z + charge/100 for atoms and ions (26.01 = Fe II), molecule code (606. = C2)
  log_gf          log10 of (statistical weight of the lower level x oscillator strength)
  log_isotope     log10 of the isotopic (isotopologue) fraction the line belongs to
  e_lower         lower-level energy, cm^-1 (above the ground state of the species)
  log_gamma_rad   log10 radiative damping constant Gamma_R, s^-1
  log_gamma_stark log10 Gamma_S / n_e, s^-1 cm^3 (temperature independent)
  log_gamma_vdw   log10 Gamma_W / n_H at T = 10^4 K, s^-1 cm^3; scaled by (T/10^4)^0.3 and by the
                  perturber weights 1 (H), 0.42 (He), 0.85 (H2) (K&A 1981 p. 4) unless vdw_alpha != 0
  vdw_alpha       velocity exponent of the Anstee-Barklem-O'Mara cross section (0: K&A 1981 scaling)
  mass            mass of the absorber, amu (isotope / isotopologue mass where the list specifies it)
  kind            LineKind
  n_lower/n_upper principal quantum numbers of hydrogen lines; level indices of departure coefficients
                  (NBLO/NBUP) for the other lines given them
  series_limit    index into the merged-continuum edge table (NCON, XLINOP CONTX) of lines of a series
                  dissolving into a continuum; 0 = none
  wing_class      LINESIZE digit of the gfall OTHER2 field (rgfalllinesnew.for line selection margin)
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum

import numpy as np
import torch


class LineKind(IntEnum):
    ATOMIC = 0
    MOLECULAR = 1
    HYDROGEN = 2
    HELIUM_I = 3
    HELIUM_II = 4
    AUTOIONIZING = 5


_FLOAT = ("wavelength", "species", "log_gf", "log_isotope", "e_lower", "log_gamma_rad", "log_gamma_stark",
          "log_gamma_vdw", "vdw_alpha", "mass")
_INT = ("kind", "n_lower", "n_upper", "series_limit", "wing_class")


@dataclass(frozen=True)
class LineList:
    """Lines as parallel 1-d tensors (float64 / int64), see the module docstring for units."""
    wavelength: torch.Tensor
    species: torch.Tensor
    log_gf: torch.Tensor
    log_isotope: torch.Tensor
    e_lower: torch.Tensor
    log_gamma_rad: torch.Tensor
    log_gamma_stark: torch.Tensor
    log_gamma_vdw: torch.Tensor
    vdw_alpha: torch.Tensor
    mass: torch.Tensor
    kind: torch.Tensor
    n_lower: torch.Tensor
    n_upper: torch.Tensor
    series_limit: torch.Tensor
    wing_class: torch.Tensor

    def __len__(self) -> int:
        return self.wavelength.shape[0]

    @classmethod
    def from_numpy(cls, **arrays) -> "LineList":
        n = len(arrays["wavelength"])
        out = {}
        for f in fields(cls):
            a = arrays.get(f.name)
            if a is None:
                a = np.zeros(n)
            dt = torch.int64 if f.name in _INT else torch.float64
            out[f.name] = torch.as_tensor(np.ascontiguousarray(a), dtype=dt)
        return cls(**out)

    @classmethod
    def empty(cls) -> "LineList":
        return cls.from_numpy(wavelength=np.zeros(0))

    @staticmethod
    def concat(lists: list["LineList"]) -> "LineList":
        lists = [x for x in lists if len(x)] or [LineList.empty()]
        return LineList(**{f.name: torch.cat([getattr(x, f.name) for x in lists]) for f in fields(LineList)})

    def select(self, mask_or_index) -> "LineList":
        return LineList(**{f.name: getattr(self, f.name)[mask_or_index] for f in fields(self)})

    def sorted(self) -> "LineList":
        return self.select(torch.argsort(self.wavelength, stable=True))

    def window(self, wl_min: float, wl_max: float) -> "LineList":
        return self.select((self.wavelength >= wl_min) & (self.wavelength <= wl_max))

    def of_kind(self, *kinds: LineKind) -> "LineList":
        m = torch.zeros_like(self.kind, dtype=torch.bool)
        for k in kinds:
            m |= self.kind == int(k)
        return self.select(m)

    def to(self, device) -> "LineList":
        return LineList(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def with_(self, **kw) -> "LineList":
        return replace(self, **kw)

    @property
    def is_molecule(self) -> torch.Tensor:
        return self.species >= 100.0

    @property
    def element(self) -> torch.Tensor:
        """Z of atomic species (0 for molecules)."""
        return torch.where(self.is_molecule, torch.zeros_like(self.kind), self.species.floor().long())

    @property
    def stage(self) -> torch.Tensor:
        """Ionisation stage, 1 = neutral (0 for molecules)."""
        s = torch.round((self.species - self.species.floor()) * 100.0).long() + 1
        return torch.where(self.is_molecule, torch.zeros_like(s), s)
