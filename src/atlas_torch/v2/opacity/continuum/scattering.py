"""Continuum scattering: Thomson scattering by free electrons and Rayleigh scattering by H I, He I and H2 (Kurucz 1970
secs. 5.4, 5.8, 5.12, 5.13; atlas7v.for ELECOP, HRAYOP, HERAOP, H2RAOP). Per unit volume, cm^-1, (..., L, F)."""
from __future__ import annotations

import torch

from .data import fortran_data
from .interpolate import map1
from .molecules import h2_density
from .state import Conditions

NU_LYMAN_GAVRILA = 3.288051e15
GAP_885, GAP_890 = 2.909925e15, 2.926365e15      # 0.885, 0.890 nu_1 as the continuum edges of continua.dat
C_ANGSTROM = 2.99792458e18


def _uniform(x, x0, dx, t, dx_origin=None, x0_index=None):
    """Linear interpolation in t tabulated at x0 + k dx (k = 0..K-1); with dx_origin / x0_index the abscissa typos
    of HRAYOP (KuruczCompat.gavrila_typos): node k placed at x0 + k dx_origin, cell chosen from x0_index."""
    k = torch.clamp(torch.floor((x - (x0 if x0_index is None else x0_index)) / dx), 0, t.shape[0] - 2).long()
    xk = x0 + k * (dx if dx_origin is None else dx_origin)
    return t[k] + (t[k + 1] - t[k]) * (x - xk) / dx


def thomson(c: Conditions) -> torch.Tensor:
    """sigma_T n_e (sec. 5.12; ELECOP), (..., L, 1) (frequency independent)."""
    return c.rad.thomson * c.n_e


def h1_rayleigh_cross_section(c: Conditions) -> torch.Tensor:
    """Rayleigh / Raman cross-section of H I 1s, Gavrila (1967) (HRAYOP), (F,): sigma_T G^2 with G tabulated in
    x = nu/nu_1 (nu_1 = 3.288051E15 Hz) on 0.01-0.74 (step 0.01), 0.755-0.885 (0.005), 0.890-0.936 (0.002),
    0.938-0.959 (0.001), sigma_T G(0.01)^2 (x/0.01)^4 below x = 0.01; zero in the cut-out Lyman line cores
    (0.7357-0.77, 0.885-0.890, 0.936-0.938, 0.959-0.961; the 0.885 and 0.890 limits at the continua.dat edges);
    sigma_T G above (constant to x = 1, Lyman continuum table beyond, blended-parabola interpolation)."""
    dev = c.nu.device
    x = c.nu / NU_LYMAN_GAVRILA
    tab = lambda k: fortran_data("hrayop", k, dev)                         # noqa: E731
    zero = torch.zeros((), dtype=x.dtype)
    g_m, g_ab, g_bc, g_cd = tab("gavrilam"), tab("gavrilamab"), tab("gavrilambc"), tab("gavrilamcd")
    g_lc, f_lc = tab("gavrilalymancont"), tab("fgavrilalymancont")
    sq = lambda t: t * t                                                   # noqa: E731
    s = torch.where(x < 0.01, sq(g_m[0]) * (x / 0.01) ** 4, zero)
    typo = c.compat.gavrila_typos
    for lo, hi, x0, dx, t, dx_origin, x0_index in (
            (0.01, 2.4190611e15 / NU_LYMAN_GAVRILA, 0.01, 0.01, g_m, None, None),
            (0.77, GAP_885 / NU_LYMAN_GAVRILA, 0.755, 0.005, g_ab,
             1.664026e13 / NU_LYMAN_GAVRILA if typo else None, None),
            (GAP_890 / NU_LYMAN_GAVRILA, 0.936, 0.890, 0.002, g_bc, None,
             0.890 * 3.28851e15 / NU_LYMAN_GAVRILA if typo else None),
            (0.938, 0.959, 0.938, 0.001, g_cd, None, None)):
        g = _uniform(x, x0, dx, t, dx_origin, x0_index)
        s = torch.where((x >= lo) & (x <= hi), g * g, s)
    s = torch.where((x >= 0.961) & (x <= 1.0), g_lc[0], s)
    s = torch.where(x > 1.0, map1(x, f_lc, g_lc), s)
    return c.rad.rayleigh * s


def h1_rayleigh(c: Conditions) -> torch.Tensor:
    """H I Rayleigh scattering, n(H I, 1s) sigma (sec. 5.4 with Gavrila's cross-sections; HRAYOP)."""
    return 2.0 * c.n_over_u(1, 1) * h1_rayleigh_cross_section(c)


def he1_rayleigh(c: Conditions) -> torch.Tensor:
    """He I Rayleigh scattering, Dalgarno (1962) (sec. 5.8; HERAOP): n(He I, 1s) 5.484E-14/lambda^4
    (1 + (2.44E5 + 5.94E10/(lambda^2 - 2.90E5))/lambda^2)^2, lambda in A, nu capped at 5.15E15 Hz."""
    ww = (C_ANGSTROM / torch.clamp(c.nu, max=5.15e15)) ** 2
    t = 1.0 + (2.44e5 + 5.94e10 / (ww - 2.90e5)) / ww
    return c.n_over_u(2, 1) * 5.484e-14 / (ww * ww) * t * t


def h2_rayleigh(c: Conditions) -> torch.Tensor:
    """H2 Rayleigh scattering, Dalgarno (1962) (sec. 5.13; H2RAOP): n(H2) (8.14E-13 + 1.28E-6/lambda^2
    + 1.61/lambda^4)/lambda^4, lambda in A, nu capped at 2.922E15 Hz."""
    ww = (C_ANGSTROM / torch.clamp(c.nu, max=2.922e15)) ** 2
    return h2_density(c, "h2raop") * (8.14e-13 + 1.28e-6 / ww + 1.61 / (ww * ww)) / (ww * ww)
