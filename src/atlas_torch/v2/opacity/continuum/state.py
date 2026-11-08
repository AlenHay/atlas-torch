"""Gas and radiation quantities shared by the continuous-opacity processes, and the bound-free level sum."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from ...constants import PhysicalConstants
from ...eos.equilibrium import Populations

C_CM = 2.99792458e10            # cm s^-1 (exact)
RYDBERG_INF = 109737.31568160   # R_infinity, cm^-1 (CODATA 2018)


@dataclass(frozen=True)
class RadiativeConstants:
    """kramers    hydrogenic bound-free cross-section times n^5 nu^3 / Z^4, cm^2 s^-3:
                  64 pi^4 m_e e^10 / (3 sqrt(3) c h^6) (Kurucz 1970 eq. 5.5, 2.815E29);
    free_free  hydrogenic free-free coefficient, kappa = free_free Z^2 g n_e n_ion / (nu^3 sqrt T) (1 - e^-hnu/kT):
               (4 e^6 / 3 h c m_e) (2 pi / 3 k m_e)^1/2 (eq. 5.8, 3.6919E8);
    thomson    electron scattering cross-section 8 pi r_e^2 / 3, cm^2 (sec. 5.12; ELECOP .6653D-24);
    rayleigh   unit of Gavrila's H I Rayleigh cross-sections, the Thomson cross-section (HRAYOP 6.65D-25)."""
    kramers: float
    free_free: float
    thomson: float
    rayleigh: float


def _codata_radiative() -> RadiativeConstants:
    c, h, k = 2.99792458e10, 6.62607015e-27, 1.380649e-16
    e, m = 4.803204712570263e-10, 9.1093837015e-28
    return RadiativeConstants(
        kramers=64.0 * math.pi ** 4 * m * e ** 10 / (3.0 * math.sqrt(3.0) * c * h ** 6),
        free_free=4.0 * e ** 6 / (3.0 * h * c * m) * math.sqrt(2.0 * math.pi / (3.0 * k * m)),
        thomson=6.6524587321e-25, rayleigh=6.6524587321e-25)


CODATA_RADIATIVE = _codata_radiative()
KURUCZ_RADIATIVE = RadiativeConstants(kramers=2.815e29, free_free=3.6919e8, thomson=0.6653e-24, rayleigh=6.65e-25)


def radiative_constants(const: PhysicalConstants) -> RadiativeConstants:
    """The atlas7v.for literals with the KURUCZ constant set, else values derived from CODATA 2018."""
    return KURUCZ_RADIATIVE if const.name == "KURUCZ" else CODATA_RADIATIVE


@dataclass(frozen=True)
class KuruczCompat:
    """Switches to the atlas7v.for choices where v2 deliberately differs (all False: v2 physics; KuruczCompat.all()).

    h2_fits          n(H2) for H2 Rayleigh / collision-induced absorption from H2RAOP's / H2COLLOP's own K(T) fits of
                     n(H I, n=1)^2 instead of the equation of state's n(H2)
    choh_tables      CH, OH absorption as (n/U)_EOS x U(T) of the CHOP/OHOP tables x sigma, instead of n_EOS x sigma
    hot_edge_cut     HOTOP adds an edge only where n g sigma exceeds 1% of the running total (before its Boltzmann
                     factor)
    metal_ff_hotop   metal free-free from C, N, O, Ne, Mg, Si, S, Fe ions of charge 1-5 only (HOTOP), instead of
                     all ions of Z >= 3
    he1_ff_typo      He I free-free (He II + e) with HE1OP's 3.619E8 instead of the hydrogenic constant 3.6919E8
    he2_gaunt_at_nu  He II bound-free Gaunt fits evaluated at nu instead of nu/Z^2 (HE2OP)
    gavrila_typos    HRAYOP's abscissae 1.664026D13 (for 1.644026D13) and 3.28851D15 (for 3.288051D15)

    ATLAS9-only (atlas9mem.for; KuruczCompat.atlas9()):
    coolop_frequency_cut  no C I, Mg I, Al I, Si I, Fe I bound-free, CH, OH or H2 collision-induced absorption above
                     3.28805E15 Hz (COOLOP: IF(FREQ.GT.3.28805D15)RETURN)
    cool_atlas9      C I, Mg I, Al I bound-free of atlas9mem.for C1OP, MG1OP, AL1OP (continuum/atlas9.py)
    hop_atlas9       H I bound-free of atlas9mem.for HOP: n <= 8 hydrogenic, the rest merged (continuum/atlas9.py)"""
    h2_fits: bool = False
    choh_tables: bool = False
    hot_edge_cut: bool = False
    metal_ff_hotop: bool = False
    he1_ff_typo: bool = False
    he2_gaunt_at_nu: bool = False
    gavrila_typos: bool = False
    coolop_frequency_cut: bool = False
    cool_atlas9: bool = False
    hop_atlas9: bool = False

    ATLAS9_ONLY = ("coolop_frequency_cut", "cool_atlas9", "hop_atlas9")

    @classmethod
    def all(cls) -> "KuruczCompat":
        """atlas7v.for's choices (the SYNTHE-side continuum: XNFPELSYN, SPECTRV)."""
        return cls(**{k: k not in cls.ATLAS9_ONLY for k in cls.__dataclass_fields__})

    @classmethod
    def atlas9(cls) -> "KuruczCompat":
        """atlas9mem.for's choices: all() plus the ATLAS9-only switches (ATLAS9, KAPPA9)."""
        return cls(**{k: True for k in cls.__dataclass_fields__})


class Conditions:
    """Per-layer gas state (..., L, 1) and per-frequency radiation quantities (F,) / (..., L, F).

    T, n_e (cm^-3), rho (g cm^-3); nu (Hz), waveno = nu/c (cm^-1); hnu_kt; stim = 1 - exp(-h nu/kT);
    hc_kt = hc/kT (cm, multiplies energies in cm^-1); kt_ev = kT in eV."""

    def __init__(self, pops: Populations, nu: torch.Tensor, const: PhysicalConstants,
                 compat: KuruczCompat = KuruczCompat()):
        self.pops, self.const, self.compat = pops, const, compat
        self.rad = radiative_constants(const)
        self.T = pops.T[..., None]
        self.n_e = pops.n_e[..., None]
        self.rho = pops.rho[..., None]
        self.nu = nu
        self.waveno = nu / C_CM
        self.hnu_kt = (const.h / const.k) * nu / self.T
        self.stim = -torch.expm1(-self.hnu_kt)
        self.hc_kt = const.hc_k / self.T
        self.kt_ev = const.k_ev * self.T

    def n(self, z: int, stage: int) -> torch.Tensor:
        """Number density of element z in ionisation stage `stage` (1 = neutral), (..., L, 1)."""
        return self.pops.n_atom[..., z - 1, stage - 1, None]

    def n_over_u(self, z: int, stage: int) -> torch.Tensor:
        """n/U, the population of a level of statistical weight 1 at zero excitation, (..., L, 1)."""
        return self.pops.n_over_u_atom[..., z - 1, stage - 1, None]

    def n_molecule(self, code: float) -> torch.Tensor:
        """Number density of a molecule by Kurucz code (101. = H2), (..., L, 1)."""
        return self.pops.density(code)[..., None]

    def boltzmann(self, energy: torch.Tensor, unit: str = "cm-1") -> torch.Tensor:
        """exp(-E/kT) for level energies E (K,), (..., L, K)."""
        return torch.exp(-energy * (self.hc_kt if unit == "cm-1" else 1.0 / self.kt_ev))

    def tensor(self, values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=self.nu.dtype, device=self.nu.device)


@dataclass(frozen=True)
class Levels:
    """Bound-free absorbing levels of one species: kappa = n/U sum_k g_k exp(-E_k/kT) sigma_k(nu) (1 - e^-hnu/kT)
    for nu >= edge_k (Kurucz 1970 eqs. 5.3-5.4 with b = 1).

    g (K,); energy (K,) excitation energies in `unit` ('cm-1' or 'eV'); edge (K,) threshold frequencies, Hz;
    sigma(nu) -> (F, K) photoionisation cross-sections, cm^2 (evaluated above threshold only)."""
    g: tuple
    energy: tuple
    edge: tuple
    sigma: Callable[[torch.Tensor], torch.Tensor]
    unit: str = "cm-1"

    def cross_sections(self, c: Conditions) -> torch.Tensor:
        """sigma_k(nu), zero below each threshold, (F, K)."""
        edge = c.tensor(self.edge)
        return torch.where(c.nu[:, None] >= edge, self.sigma(c.nu), torch.zeros((), dtype=c.nu.dtype))

    def weighted_sum(self, c: Conditions) -> torch.Tensor:
        """sum_k g_k exp(-E_k/kT) sigma_k(nu), (..., L, F)."""
        w = c.tensor(self.g) * c.boltzmann(c.tensor(self.energy), self.unit)
        return w @ self.cross_sections(c).transpose(0, 1)


def stack(*cols: torch.Tensor) -> torch.Tensor:
    """(F,) columns -> (F, K)."""
    return torch.stack(torch.broadcast_tensors(*cols), dim=-1)
