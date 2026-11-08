"""atlas9mem.for's (older) versions of continuous absorbers where they differ from the atlas7v.for routines of
metals.py / hydrogen.py: C I from its ground term only, Mg I from Peach's tabulated cross-sections times partition
function, Al I as one power law (KuruczCompat.cool_atlas9); H I with hydrogenic levels n <= 8 and the rest merged
from 13.427 eV (KuruczCompat.hop_atlas9). Per unit volume, cm^-1, (..., L, F)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ....tables import table
from ...constants import decimal32
from .hydrogenic import karzas_latter
from .state import Conditions


@lru_cache(maxsize=None)
def _mg1_tables():
    g = lambda v: decimal32(np.asarray(table("atlas9mem.for", "mg1op", v), dtype=np.float32))   # noqa: E731
    return g("peach"), g("freqmg"), g("flog"), g("tlg")


def c1_bound_free(c: Conditions) -> torch.Tensor:
    """C I (atlas9mem.for C1OP): 2p2 3P, 1D, 1S with the Luo & Pradhan (1989) power laws and the Burke & Taylor
    (1979) resonance fits, Boltzmann factors 9, 5 exp(-1.264 eV/kT), exp(-2.683 eV/kT)."""
    ryd = 109732.298
    w = c.waveno
    ground = lambda w0: 10.0 ** (-16.80 - (w - w0) / 3.0 / ryd)                  # noqa: E731

    def res(w0, width, a, b):
        eps = (w - w0) * 2.0 / width
        return (a * eps + b) / (eps * eps + 1.0)

    zero = torch.zeros_like(w)
    x1100 = torch.where(c.nu >= 2.7254e15, ground(90777.000), zero)
    x1240 = torch.where(c.nu >= 2.4196e15, ground(80627.760) + res(93917.0, 9230.0, 22.0e-18, 26.0e-18)
                        + res(111130.0, 2743.0, -10.5e-18, 46.0e-18), zero)
    x1444 = torch.where(c.nu >= 2.0761e15, ground(69172.400) + res(97700.0, 2743.0, 68.0e-18, 118.0e-18), zero)
    s = 9.0 * x1100 + x1240 * 5.0 * torch.exp(-1.264 / c.kt_ev) + x1444 * torch.exp(-2.683 / c.kt_ev)
    return c.n_over_u(6, 1) * s * c.stim


def mg1_bound_free(c: Conditions) -> torch.Tensor:
    """Mg I (atlas9mem.for MG1OP): ln(sigma U) tabulated by Peach at 4000-10000 K and 15 wavelengths (pairs at the
    edges), linear in ln nu between the edge nodes FLOG and linear in ln T (extrapolated outside)."""
    peach, freqmg, flog, tlg = (c.tensor(a) for a in _mg1_tables())
    nu = c.nu
    n = (nu[:, None] <= freqmg).sum(-1)                                     # node interval (0-based), edges decrease
    d = (torch.log(nu) - flog[n]) / (flog[n + 1] - flog[n])
    k = torch.where(n > 1, 2 * n - 1, n)                                    # PEACH column below the edge pair
    x = peach[:, k + 1] * d + peach[:, k] * (1.0 - d)                       # (7 temperatures, F)
    T = c.T[..., 0]
    it = torch.clamp(torch.floor(T / 1000.0) - 3.0, 1.0, 6.0).long() - 1
    dt = ((torch.log(T) - tlg[it]) / (tlg[it + 1] - tlg[it]))[..., None]
    s = torch.exp(x[it] * (1.0 - dt) + x[it + 1] * dt)                      # (..., L, F)
    return c.n_over_u(12, 1) * s * c.stim


def al1_bound_free(c: Conditions) -> torch.Tensor:
    """Al I (atlas9mem.for AL1OP): 6 x 6.5E-17 (1.443E15/nu)^4.4 cm^2 above 1.443E15 Hz."""
    s = torch.where(c.nu >= 1.443e15, 6.5e-17 * (1.443e15 / c.nu) ** 4.4 * 6.0, torch.zeros_like(c.nu))
    return c.n_over_u(13, 1) * s * c.stim


def h1_bound_free(c: Conditions) -> torch.Tensor:
    """H I (atlas9mem.for HOP): levels n = 1..8 at E_n = 13.595 (1 - 1/n^2) eV, g = 2 n^2, Karzas-Latter
    cross-sections; n >= 9 merged (eq. 5.6) from 13.427 eV (from 13.595 eV - h nu below 4.05933E15/100 Hz)."""
    n = torch.arange(1, 9, dtype=c.nu.dtype, device=c.nu.device)
    e = 13.595 * (1.0 - 1.0 / n ** 2)
    boltz = 2.0 * n ** 2 * torch.exp(-e / c.kt_ev)                              # (..., L, 8)
    sigma = torch.where(c.nu[:, None] >= 3.28805e15 / n ** 2, karzas_latter(c.nu, 8), torch.zeros((), dtype=c.nu.dtype))
    levels = boltz @ sigma.transpose(0, 1)
    hnu_ev = c.hnu_kt * c.kt_ev
    e_low = torch.where(c.nu < 4.05933e13, 13.595 - hnu_ev, torch.full_like(hnu_ev, 13.427))
    merged = c.rad.kramers / c.nu ** 3 * c.kt_ev / 13.595 * (torch.exp(-e_low / c.kt_ev) - torch.exp(-13.595 / c.kt_ev))
    return c.n_over_u(1, 1) * (levels + merged) * c.stim


COOL = {"C I bf": c1_bound_free, "Mg I bf": mg1_bound_free, "Al I bf": al1_bound_free}
HOP = {"H I bf": h1_bound_free}
