"""Thermodynamic derivatives of gas + radiation for the convective flux (Kurucz 1970 sec. 4.6, eqs. 4.48-4.61).

    td = ThermoDerivatives.of(eos, T, P, comp, p_total, p_rad, dilution)          # (M, L) fields

Energy per unit mass E = [3/2 P + sum_i n_i (E_i + kT^2 dln U_i/dT)] / rho + 3 P_rad / rho (eq. 4.50), with E_i the
ionisation energy above the neutral ground state; for a molecule (formation constant K(T) of molecules.dat,
n_mol = K prod n_a n_e^-q), the van 't Hoff energy kT^2 dln K/dT plus the internal energies of its atoms and
3/2 kT (N_atoms - 1 - q), so that energy and equilibrium use the same K(T). The radiation pressure varies as
T^4 times the dilution 1 - exp(-tau_Ross) (atlas9mem.for CONVEC: DPDT = 4 PRADK/T DILUT).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import torch

from ..atmosphere import Atmosphere
from ..composition import Composition


def internal_energy(pops, eos) -> torch.Tensor:
    """Gas energy per unit mass E (erg g^-1), (..., L), of equilibrium populations (eq. 4.50; no radiation)."""
    c = eos.const
    T = pops.T
    kt = c.k * T
    ev = c.k / c.k_ev                                                    # erg per eV
    de = pops.debye
    ln_u = lambda t: torch.log(eos.pf(t, de))                          # noqa: E731
    _, dlnu = torch.func.jvp(ln_u, (T,), (torch.ones_like(T),))         # (..., L, 99, 10)
    chi = eos.pf.chi.to(T.dtype)
    e_ion = torch.cat([torch.zeros_like(chi[:, :1]), torch.cumsum(chi[:, :-1], -1)], -1) * ev
    kt2 = (kt * T)[..., None, None]
    atoms = (pops.n_atom * (e_ion + kt2 * dlnu)).sum((-1, -2))
    mol = eos.molecules
    _, dlnk = torch.func.jvp(mol.log_k, (T,), (torch.ones_like(T),))    # (..., L, nmol)
    tt = lambda a: torch.as_tensor(a, dtype=T.dtype, device=T.device)  # noqa: E731
    n_at = tt(mol.stoich.sum(1))
    comp_u = dlnu[..., 0] @ tt(mol.stoich).T                            # sum_a dln U_a,I/dT
    e_mol = (kt * T)[..., None] * (dlnk + comp_u) + 1.5 * kt[..., None] * (n_at - 1.0 - tt(mol.charge))
    molecules = (pops.n_mol * e_mol).sum(-1)
    return (1.5 * pops.P + atoms + molecules) / pops.rho


@dataclass(frozen=True)
class ThermoDerivatives:
    """At constant total pressure P_total = P_gas + P_rad (P_turb ignored, eq. 4.60-4.61), (M, L):

    heat_cp, heat_cv   specific heats, erg g^-1 K^-1 (eqs. 4.48-4.49, 4.56-4.57)
    dlnrho_dlnT        (dln rho/dln T) at constant P_total (DLRDLT, eq. 4.58)
    grad_ad            adiabatic gradient (dln T/dln P_total)_S (GRDADB, eq. 4.51)
    sound_speed        cm s^-1 (VELSND, eq. 4.59; 0 where c_v <= 0)
    rho                g cm^-3"""
    heat_cp: torch.Tensor
    heat_cv: torch.Tensor
    dlnrho_dlnT: torch.Tensor
    grad_ad: torch.Tensor
    sound_speed: torch.Tensor
    rho: torch.Tensor

    @classmethod
    def of(cls, eos, T: torch.Tensor, P: torch.Tensor, comp, p_total: torch.Tensor, p_rad: torch.Tensor,
           dilution: torch.Tensor, mode: str = "exact", pops=None) -> "ThermoDerivatives":
        """mode "exact": dE/dT, dE/dP, drho/dT, drho/dP by forward-mode AD through the EOS solution (implicit
        function theorem; needs eos without the lagged lowering, which makes a layer depend on the one above);
        "kurucz": central differences at T (1 +- 0.001) and P (1 +- 0.001) (CONVEC loops 10-30).
        pops: the populations at (T, P), if known (Newton start of the perturbed solutions)."""
        def state(lt, lp, n=None):
            """E and rho at (ln T, ln P) (..., L); n: that many states stacked on a new leading axis."""
            c, guess = comp, pops
            if n is not None:
                c = Composition(comp.abundance.expand((n,) + comp.abundance.shape))
                guess = None if pops is None else _stacked(pops, n)
            pops2 = eos.solve(Atmosphere(rhox=torch.zeros_like(lt), T=torch.exp(lt), P=torch.exp(lp)), c,
                              guess=guess)
            return internal_energy(pops2, eos), pops2.rho

        lt, lp = torch.log(T), torch.log(P)
        if mode == "exact":
            if getattr(eos, "lagged", False):
                raise ValueError("thermodynamics='exact' needs an EOS without the lagged lowering")
            one, zero = torch.ones_like(lt), torch.zeros_like(lt)
            (e, rho), (e_t, r_t) = torch.func.jvp(state, (lt, lp), (one, zero))
            _, (e_p, r_p) = torch.func.jvp(state, (lt, lp), (zero, one))
            dedt, drdt, dedp, drdp = e_t / T, r_t / T, e_p / P, r_p / P
        elif mode == "kurucz":
            h = 0.001
            up, dn = math.log1p(h), math.log1p(-h)
            es, rs = state(torch.stack([lt + up, lt + dn, lt, lt]), torch.stack([lp, lp, lp + up, lp + dn]), 4)
            (e1, e2, e3, e4), (r1, r2, r3, r4) = es.unbind(0), rs.unbind(0)
            rho = pops.rho if pops is not None else state(lt, lp)[1]
            # radiation energy 3 P_rad / rho, P_rad ~ T^(4 dilution), as CONVEC's EDENS1..4
            e1 = e1 + 3.0 * p_rad / r1 * (1.0 + dilution * ((1.0 + h) ** 4 - 1.0))
            e2 = e2 + 3.0 * p_rad / r2 * (1.0 + dilution * ((1.0 - h) ** 4 - 1.0))
            e3, e4 = e3 + 3.0 * p_rad / r3, e4 + 3.0 * p_rad / r4
            dedt, drdt = (e1 - e2) / (2.0 * h * T), (r1 - r2) / (2.0 * h * T)
            dedp, drdp = (e3 - e4) / (2.0 * h * P), (r3 - r4) / (2.0 * h * P)
            return cls._combine(T, rho, p_total, p_rad, dilution, dedt, dedp, drdt, drdp)
        else:
            raise ValueError("mode: 'exact' | 'kurucz'")
        # radiation energy 3 P_rad / rho with P_rad ~ T^(4 dilution)
        er = 3.0 * p_rad / rho
        dedt = dedt + er * (4.0 * dilution / T - drdt / rho)
        dedp = dedp - er * drdp / rho
        return cls._combine(T, rho, p_total, p_rad, dilution, dedt, dedp, drdt, drdp)

    @classmethod
    def _combine(cls, T, rho, p_total, p_rad, dilution, dedt, dedp, drdt, drdp):
        """Eqs. 4.56-4.61 with (dP_total/dP)_T = 1, (dP_total/dT)_P = 4 P_rad dilution / T (CONVEC loop 30)."""
        dpdt = 4.0 * p_rad / T * dilution
        cv = dedt - dedp * drdt / drdp
        drdt_ptot = drdt - drdp * dpdt
        cp = dedt - dedp * dpdt - p_total / rho ** 2 * drdt_ptot
        vs = torch.sqrt(torch.clamp(cp / torch.where(cv > 0, cv, torch.ones_like(cv)) / drdp, min=0.0))
        vs = torch.where(cv > 0, vs, torch.zeros_like(vs))
        dlrdlt = T / rho * drdt_ptot
        grad_ad = -p_total / (rho * T) * dlrdlt / cp
        return cls(cp, cv, dlrdlt, grad_ad, vs, rho)


def _stacked(pops, n: int):
    """Populations repeated n times on a new leading axis (Newton start of stacked solves)."""
    return replace(pops, **{f.name: getattr(pops, f.name).expand((n,) + getattr(pops, f.name).shape)
                            for f in fields(pops) if isinstance(getattr(pops, f.name), torch.Tensor)})
