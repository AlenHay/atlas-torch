"""Hydrogenic bound-free and free-free absorption (Kurucz 1970 sec. 5.1, eqs. 5.5-5.8).

Cross-sections of levels n <= 15: Karzas & Latter (1961) as tabulated in atlas7v.for FUNCTION XKARSAS; free-free
Gaunt factors: the Karzas & Latter table of Kurucz 1970 p. 77 (FUNCTION COULFF)."""
from __future__ import annotations

import torch

from .data import fortran_data
from .interpolate import linear
from .state import RYDBERG_INF, Conditions


def karzas_latter(nu: torch.Tensor, n_max: int = 15, z: int = 1) -> torch.Tensor:
    """Total photoionisation cross-section of levels n = 1..n_max of a hydrogenic ion of charge z, cm^2, (F, n_max):
    log10 sigma interpolated linearly in log10 nu (XKARSAS tables FREQN, XN; sigma_z(nu) = sigma_1(nu/z^2)/z^2);
    below the table's first frequency (the Rydberg threshold R c/n^2) the threshold value, above the table linear
    log-log extrapolation."""
    fn = fortran_data("xkarsas", "freqn", nu.device).flip(0)           # (29, 15), increasing frequency
    xn = fortran_data("xkarsas", "xn", nu.device).flip(0)
    x = torch.log10(nu / z ** 2)
    cols = [linear(torch.maximum(x, fn[0, n]), fn[:, n], xn[:, n]) for n in range(n_max)]
    return 10.0 ** torch.stack(cols, -1) / z ** 2


def kramers(c: Conditions, n, z: int = 1) -> torch.Tensor:
    """Hydrogenic cross-section with unit Gaunt factor, kramers z^4 / (n^5 nu^3), (F,) (eq. 5.5)."""
    return c.rad.kramers * z ** 4 / (n ** 5 * c.nu ** 3)


def merged_levels(c: Conditions, n_first: int, chi: float, rz2: float, g_per_n2: float, z: int = 1) -> torch.Tensor:
    """sum_{n >= n_first} g_n exp(-E_n/kT) sigma_n(nu) for Kramers levels g_n = g_per_n2 n^2, E_n = chi - rz2/n^2
    (cm^-1, rz2 = R Z^2), as the integral over n of eq. 5.6:
    kramers z^4/nu^3 (g_per_n2/2) kT/(hc rz2) [exp(-E_low/kT) - exp(-chi/kT)], E_low = max(E_{n_first}, chi - h nu)
    (levels whose edge lies above nu do not absorb). (..., L, F)."""
    e_low = torch.clamp(chi - c.waveno, min=chi - rz2 / n_first ** 2)
    return (c.rad.kramers * z ** 4 / c.nu ** 3 * (0.5 * g_per_n2) / (rz2 * c.hc_kt)
            * (torch.exp(-e_low * c.hc_kt) - torch.exp(-chi * c.hc_kt)))


def gaunt_free_free(c: Conditions, z: float = 1.0) -> torch.Tensor:
    """Free-free Gaunt factor g(gamma^2, u) of a charge-z ion, gamma^2 = z^2 Ry/kT, u = h nu/kT, (..., L, F):
    bilinear in (2 log10 gamma^2, 2 log10 u) on the table of Kurucz 1970 p. 77 (COULFF DATA A), extrapolated
    linearly beyond it."""
    a = fortran_data("coulff", "a", c.nu.device)                         # (11 gamma^2, 12 u)
    gam = 2.0 * torch.log10(z * z * RYDBERG_INF * c.hc_kt)               # (..., L, 1)
    u = 2.0 * torch.log10(c.hnu_kt)                                      # (..., L, F)
    i = torch.clamp(torch.floor(gam + 6.0), 0, 9).long()                 # cell (i, i+1), node i at 2 log g^2 = i-6
    j = torch.clamp(torch.floor(u + 8.0), 0, 10).long()
    p = gam - (i - 6).to(gam.dtype)
    q = u - (j - 8).to(u.dtype)
    i, j = torch.broadcast_tensors(i, j)
    a00, a01, a10, a11 = a[i, j], a[i, j + 1], a[i + 1, j], a[i + 1, j + 1]
    return (1.0 - p) * ((1.0 - q) * a00 + q * a01) + p * ((1.0 - q) * a10 + q * a11)


def free_free(c: Conditions, z: float, n_ion: torch.Tensor, coefficient: float | None = None) -> torch.Tensor:
    """Free-free absorption by electrons in the field of ions of charge z (eq. 5.8), cm^-1, (..., L, F):
    C z^2 g n_e n_ion / (nu^3 sqrt T) (1 - e^-hnu/kT), C = free_free constant unless given."""
    k = c.rad.free_free if coefficient is None else coefficient
    return k * z * z * gaunt_free_free(c, z) * c.n_e * n_ion / (c.nu ** 3 * torch.sqrt(c.T)) * c.stim

