"""Molecular continuous absorption: H2+ photodissociation, CH and OH photodissociation, H2-H2 and H2-He collision-
induced absorption (Kurucz 1970 sec. 5.2; atlas7v.for H2PLOP, CHOP, OHOP, H2COLLOP). Per unit volume, cm^-1."""
from __future__ import annotations

import torch

from .data import fortran_data
from .interpolate import linear, linear_rows
from .state import Conditions

LYMAN_LIMIT_HZ = 3.28805e15


def h2_density(c: Conditions, fit: str | None = None) -> torch.Tensor:
    """n(H2), (..., L, 1): the equation of state's, or with KuruczCompat.h2_fits the routine's own fit
    n(H2) = n(H I, 1s)^2 K(T), fit = 'h2raop' (Kurucz 1970 eq. 4.62) or 'h2collop' (NMOLEC's fit)."""
    if not c.compat.h2_fits or fit is None:
        return c.n_molecule(101.0)
    t = c.T
    if fit == "h2raop":
        ln_k = (4.477 / c.kt_ev - 4.6628e1 + (1.8031e-3 + (-5.0239e-7 + (8.1424e-11 - 5.0501e-15 * t) * t) * t) * t
                - 1.5 * torch.log(t))
    else:
        ln_k = (4.478 / c.kt_ev - 4.64584e1 + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (
            1.06206e-18 - 3.08720e-23 * t) * t) * t) * t) * t) * t - 1.5 * torch.log(t))
    return (2.0 * c.n_over_u(1, 1)) ** 2 * torch.exp(ln_k)


def h2_plus(c: Conditions) -> torch.Tensor:
    """H2+ bound-free + free-free (1s sigma_g -> 2p sigma_u), Bates' results as fit by Kurucz 1970 p. 79 (H2PLOP):
    n(H I, 1s) n(H II) exp(-E_s/kT + F(nu)) (1 - e^-hnu/kT) below the Lyman limit."""
    nu = torch.clamp(c.nu, max=LYMAN_LIMIT_HZ)
    ln = torch.log(nu)
    fr = -3.0233e3 + (3.7797e2 + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * ln) * ln) * ln) * ln
    f15 = nu / 1e15
    es = -7.342e-3 + (-2.409 + (1.028 + (-4.230e-1 + (1.224e-1 - 1.351e-2 * f15) * f15) * f15) * f15) * f15
    k = torch.exp(fr - es / c.kt_ev) * 2.0 * c.n_over_u(1, 1) * c.n(1, 2) * c.stim
    return torch.where(c.nu > LYMAN_LIMIT_HZ, torch.zeros((), dtype=k.dtype), k)


def _photodissociation(c: Conditions, unit: str, e_first: float, e_range: tuple, code: float) -> torch.Tensor:
    """n sigma(E, T) (1 - e^-hnu/kT) with log10 sigma tabulated on T = 2000, 2500, ..., 9000 K and photon energy
    E_j = e_first + 0.1 j eV; zero outside e_range and for T >= 9000 K."""
    dev = c.nu.device
    cross = fortran_data(unit, "cross" + unit[:2], dev)                 # (15, n_E)
    e_grid = e_first + 0.1 * torch.arange(cross.shape[1], dtype=cross.dtype, device=dev)
    ev = c.nu / 2.99792458e10 / c.const.ev_cm
    log_s = linear(torch.clamp(ev, *e_range)[None, :], e_grid, cross[:, None, :])     # (15, F)
    t_grid = 2000.0 + 500.0 * torch.arange(15, dtype=cross.dtype, device=dev)
    sigma = 10.0 ** linear_rows(torch.clamp(c.T, max=9000.0), t_grid, log_s)          # (..., L, F)
    if c.compat.choh_tables:
        part = fortran_data(unit, "part" + unit[:2], dev)
        t_part = 1000.0 + 200.0 * torch.arange(part.shape[0], dtype=part.dtype, device=dev)
        u = linear(torch.clamp(c.T, max=9000.0), t_part, part)
        n = c.pops.n_over_u(code)[..., None] * u
    else:
        n = c.n_molecule(code)
    inside = (ev >= e_range[0]) & (ev < e_range[1])
    k = n * sigma * c.stim
    return torch.where(inside & (c.T < 9000.0), k, torch.zeros((), dtype=k.dtype))


def ch(c: Conditions) -> torch.Tensor:
    """CH photodissociation, LTE cross-sections of Kurucz, van Dishoeck & Tarafdar (1987) (CHOP), 2.0-10.5 eV."""
    return _photodissociation(c, "chop", 0.1, (2.0, 10.5), 106.0)


def oh(c: Conditions) -> torch.Tensor:
    """OH photodissociation (OHOP), 2.1-15.0 eV."""
    return _photodissociation(c, "ohop", 2.1, (2.1, 15.0), 108.0)


def _cia(c: Conditions, var: str) -> torch.Tensor:
    """log10 of the collision-induced absorption coefficient (cm^5) on wavenumbers 0-20000 cm^-1 (step 250) and
    T = 1000-7000 K (clamped), (..., L, F)."""
    dev = c.nu.device
    tab = fortran_data("h2collop", var, dev)                            # (7, 81)
    w_grid = 250.0 * torch.arange(tab.shape[1], dtype=tab.dtype, device=dev)
    rows = linear(torch.clamp(c.waveno, max=20000.0)[None, :], w_grid, tab[:, None, :])  # (7, F)
    t_grid = 1000.0 * torch.arange(1, 8, dtype=tab.dtype, device=dev)
    return linear_rows(c.T, t_grid, rows, clamp=True)


def h2_h2_cia(c: Conditions) -> torch.Tensor:
    """H2-H2 collision-induced absorption, Borysow, Jorgensen & Zheng (1997) (H2COLLOP), below 20000 cm^-1."""
    n2 = h2_density(c, "h2collop")
    k = 10.0 ** _cia(c, "h2h2") * n2 * n2 * c.stim
    return torch.where(c.waveno > 20000.0, torch.zeros((), dtype=k.dtype), k)


def h2_he_cia(c: Conditions) -> torch.Tensor:
    """H2-He collision-induced absorption, Borysow, Jorgensen & Zheng (1997) (H2COLLOP), below 20000 cm^-1."""
    k = 10.0 ** _cia(c, "h2he") * c.n(2, 1) * h2_density(c, "h2collop") * c.stim
    return torch.where(c.waveno > 20000.0, torch.zeros((), dtype=k.dtype), k)
