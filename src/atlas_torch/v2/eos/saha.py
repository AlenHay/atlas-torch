"""Ionisation equilibrium: Debye lowering of the continuum and the Saha equation."""
from __future__ import annotations

import math

import torch

from ..constants import CODATA2018, MAX_STAGES, PhysicalConstants


def debye_lowering(T: torch.Tensor, n_charge2: torch.Tensor, const: PhysicalConstants = CODATA2018) -> torch.Tensor:
    """Lowering of the ionisation energy per unit charge, eV (Kurucz 1970 eqs. 4.36-4.37):
    dE = e^2 / r_D, r_D = sqrt(kT / (4 pi e^2 n_c)), n_c = sum_i q_i^2 n_i over electrons and ions;
    capped at 1 eV as in PFSAHA (POTLOW=MIN(1.D0,1.44D-7/DEBYE))."""
    r_d = torch.sqrt(const.k * T / (const.four_pi_e2 * n_charge2))
    return torch.clamp(const.e2_ev_cm / r_d, max=1.0)


def kurucz_charge_density(n_e: torch.Tensor, n_total: torch.Tensor) -> torch.Tensor:
    """PFSAHA's n_c: 2 n_e, plus 2 (2 n_e - n_tot) when electrons exceed half the particles (He III)."""
    excess = 2.0 * n_e - n_total
    return 2.0 * n_e + 2.0 * torch.clamp(excess, min=0.0)


def log_saha(T: torch.Tensor, ln_u: torch.Tensor, chi: torch.Tensor, dE: torch.Tensor,
             const: PhysicalConstants = CODATA2018) -> torch.Tensor:
    """ln[n(s+1) n_e / n(s)] for s = 1..9, shape (..., 99, 9) (Kurucz 1970 eq. 4.41):
    n(s+1) n_e / n(s) = 2 (2 pi m_e k T/h^2)^1.5 U(s+1)/U(s) exp(-(chi_s - s dE)/kT).

    ln_u (..., 99, 10) log partition functions; chi (99, 10) eV; dE (...) lowering per unit charge, eV."""
    s = torch.arange(1, MAX_STAGES, dtype=T.dtype, device=T.device)
    kt = (const.k_ev * T)[..., None, None]
    lnq = (math.log(2.0 * const.saha_e) + 1.5 * torch.log(T))[..., None, None]
    return lnq + ln_u[..., 1:] - ln_u[..., :-1] - (chi[:, :-1] - s * dE[..., None, None]) / kt


def log_stage_ratios(ln_phi: torch.Tensor, ln_ne: torch.Tensor, exists: torch.Tensor) -> torch.Tensor:
    """ln[n(s)/n(I)] for s = 1..10, (..., 99, 10), from ln_phi of log_saha; -inf-like (-1e4) for absent stages."""
    step = ln_phi - ln_ne[..., None, None]
    cum = torch.cat([torch.zeros_like(step[..., :1]), torch.cumsum(step, -1)], -1)
    return torch.where(exists, cum, torch.full_like(cum, -1.0e4))
