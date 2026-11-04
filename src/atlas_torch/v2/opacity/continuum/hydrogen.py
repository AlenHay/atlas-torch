"""Neutral hydrogen and the negative hydrogen ion: bound-free and free-free absorption (Kurucz 1970 secs. 5.1, 5.3;
atlas7v.for HOP, HMINOP). All coefficients per unit volume, cm^-1, (..., L, F)."""
from __future__ import annotations

import torch

from .data import fortran_data
from .hydrogenic import free_free, karzas_latter, merged_levels
from .interpolate import blended_parabola, linear, linear_rows
from .state import C_CM, Conditions, Levels

# H I ionisation limit and Rydberg constant (cm^-1); levels n = 1..15: edge (cm^-1) and excitation energy (cm^-1)
# of the fine-structure-averaged level (HOP), g_n = 2 n^2
H_LIMIT, R_H = 109678.764, 109677.576
H_EDGE = (109678.764, 27419.659, 12186.462, 6854.871, 4387.113, 3046.604, 2238.320, 1713.713, 1354.044, 1096.776,
          906.426, 761.649, 648.980, 559.579, 487.456)
H_ENERGY = (0.0, 82259.105, 97492.302, 102823.893, 105291.651, 106632.160, 107440.444, 107965.051, 108324.719,
            108581.992, 108772.336, 108917.117, 109029.789, 109119.188, 109191.313)

_H1_LEVELS = Levels(tuple(2.0 * n * n for n in range(1, 16)), H_ENERGY, tuple(e * C_CM for e in H_EDGE),
                    lambda nu: karzas_latter(nu, 15))

H_MINUS_BINDING_EV = 0.754209       # Hotop & Lineberger 1985 (HMINOP)


def h1_bound_free(c: Conditions) -> torch.Tensor:
    """H I photoionisation: n(H I)/U sum_n 2n^2 exp(-E_n/kT) sigma_n (1 - e^-hnu/kT), n = 1..15 with Karzas-Latter
    cross-sections, n >= 16 by the integral of eq. 5.6 (HOP)."""
    total = _H1_LEVELS.weighted_sum(c) + merged_levels(c, 16, H_LIMIT, R_H, 2.0)
    return c.n_over_u(1, 1) * total * c.stim


def h1_free_free(c: Conditions) -> torch.Tensor:
    """Proton-electron free-free ('H I free-free'), eq. 5.8 with Karzas-Latter Gaunt factors (HOP)."""
    return free_free(c, 1.0, c.n(1, 2))


def h_minus_density(c: Conditions) -> torch.Tensor:
    """n(H-) from the Saha equation with n(H I, n=1) = 2 n/U and g(H-) = 1 (Kurucz 1970 sec. 5.3; HMINOP), (..., L, 1):
    n(H-) = n(H I, 1s) n_e / (4 (2 pi m_e kT/h^2)^1.5) exp(0.754209 eV / kT)."""
    return (2.0 * c.n_over_u(1, 1) * c.n_e * torch.exp(H_MINUS_BINDING_EV / c.kt_ev)
            / (4.0 * c.const.saha_e * c.T ** 1.5))


def h_minus_bound_free(c: Conditions) -> torch.Tensor:
    """H- photodetachment, n(H-) sigma (1 - e^-hnu/kT); sigma(lambda) of Mathisen (1984) (HMINOP DATA WBF, BF in Mb)
    by Kurucz's blended-parabola interpolation, zero beyond the last tabulated wavelength (the threshold) and where the
    extrapolation below 18 nm turns negative."""
    wbf, bf = fortran_data("hminop", "wbf", c.nu.device), fortran_data("hminop", "bf", c.nu.device)
    lam = C_CM * 1e7 / c.nu
    sigma = torch.where(lam < wbf[-1], torch.clamp(blended_parabola(lam, wbf, bf), min=0.0),
                        torch.zeros((), dtype=lam.dtype)) * 1e-18
    return h_minus_density(c) * sigma * c.stim


def h_minus_free_free(c: Conditions) -> torch.Tensor:
    """H- free-free (H I + e), Bell & Berrington (1987) (HMINOP DATA FF: 1e-26 cm^4 dyn^-1 per H atom per unit electron
    pressure, stimulated emission included): kT FF / 1e26 interpolated linearly in ln(lambda), then FF kT linearly
    in theta = 5040/T; times n(H I, 1s) n_e."""
    dev = c.nu.device
    ff = fortran_data("hminop", "ff", dev)                              # (11 theta, 22 wavelengths)
    theta_grid = fortran_data("hminop", "thetaff", dev)
    x_grid = torch.log(91.134 / fortran_data("hminop", "wavek", dev))  # ln(lambda/nm), increasing
    log_ffkt = torch.log(ff / theta_grid[:, None] * 5040.0 * c.const.k)   # (11, 22)
    lam_log = torch.log(C_CM * 1e7 / c.nu)
    ffkt = torch.exp(torch.stack([linear(lam_log, x_grid, log_ffkt[i]) for i in range(ff.shape[0])]))  # (11, F)
    ff_t = linear_rows(5040.0 / c.T, theta_grid, ffkt)
    return ff_t * 2.0 * c.n_over_u(1, 1) * c.n_e * 1e-26

