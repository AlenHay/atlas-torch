"""Ionisation + molecular equilibrium of a gas of given T, P_gas and composition (Kurucz 1970 sec. 4.3).

Unknowns per layer: ln n(Z I) of the elements bound in molecules, ln n_e, ln N (nuclei) and ln n_c (squared-
charge density setting the Debye lowering). Ions follow by the Saha equation (eq. 4.41), molecules by mass
action with K(T) (eq. 4.35), neutral atoms of the other elements in closed form from their abundance equation
(p. 57, eq. 4.33). Equations, as differences of logarithms of positive sums: abundance (eqs. 4.19-4.20) of the
molecular elements, P/kT = sum n (eq. 4.22), charge neutrality (eq. 4.21), n_c = n_e + sum q^2 n (eq. 4.37).
Molecules (and H-) are absent above t_molecules_max (NMOLEC: IF(T(J).GT.10000.)GO TO 37).

network="full": all elements and tabulated stages, molecules of molecules.dat (v2 default).
network="kurucz": NMOLEC's species list: the atoms/ions listed in molecules.dat and the neutral atoms of molecule
components (READMOL; e.g. Li of LiH) are in the conservation equations; other elements are spectators at the
solved n_e and N (POPS label 400); unlisted stages of listed elements are zero (POPS label 313). Elements
with no atom/ion line in molecules.dat (the spectators, and Li, Be, B, which enter only as molecule components)
are reported from the Saha equation at the converged n_e, N and lowering, as POPS does (label 400).
debye="kurucz": n_c = 2 n_e (+ He III excess) as in PFSAHA instead of the computed sum.
lagged_lowering=True (with debye="kurucz"): NMOLEC's lowering. The equilibrium of layer J is solved with the
lowering at the Newton start n_e(J-1) P(J)/P(J-1) (layer 1: P/kT/20); partition functions of n/U and molecular
n/U use the lowering at the converged n_e (NMOLEC labels 120-136). The same holds for an explicit debye_n_e.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..atmosphere import Atmosphere
from ..composition import Composition
from ..constants import CODATA2018, MAX_STAGES, N_ELEMENTS, PhysicalConstants, atomic_masses, parse_species
from ..io.molecules_dat import MoleculeRecord, read_molecules_dat
from ..solve import newton
from .molecules import MoleculeSet
from .partition import PartitionFunctions
from .saha import debye_lowering, kurucz_charge_density, log_saha, log_stage_ratios

BIG = 1.0e4


@dataclass(frozen=True)
class Populations:
    """Equilibrium number densities (cm^-3) on the atmosphere's grid; shapes (..., L[, 99, 10 | nmol]).

    n_atom[..., Z-1, s-1]: element Z in stage s (1 = neutral); u_atom: its partition function; n_mol,
    n_over_u_mol: molecules of `molecules` (n/U from eq. 4.43); n_nuclei: all nuclei (ATLAS XNATOM);
    rho = n_nuclei * mean atomic mass; mu = rho / (amu P/kT) (mean molecular weight per particle);
    debye: lowering per unit charge, eV.
    """
    T: torch.Tensor
    P: torch.Tensor
    n_total: torch.Tensor
    n_e: torch.Tensor
    n_nuclei: torch.Tensor
    rho: torch.Tensor
    mu: torch.Tensor
    debye: torch.Tensor
    n_atom: torch.Tensor
    u_atom: torch.Tensor
    n_mol: torch.Tensor
    n_over_u_mol: torch.Tensor
    molecules: MoleculeSet

    @property
    def n_over_u_atom(self) -> torch.Tensor:
        return self.n_atom / self.u_atom

    def _lookup(self, species):
        if isinstance(species, str) and " " in species.strip():
            z, s = parse_species(species)
            return ("atom", z, s)
        code = float(species)
        if code >= 100.0:
            return ("mol", self.molecules.index(code), None)
        z, s = parse_species(f"{code:.2f}")
        return ("atom", z, s)

    def density(self, species) -> torch.Tensor:
        """n of 'Fe II', 26.01 or a molecule code (608. = CO), (..., L)."""
        kind, a, b = self._lookup(species)
        return self.n_atom[..., a - 1, b - 1] if kind == "atom" else self.n_mol[..., a]

    def n_over_u(self, species) -> torch.Tensor:
        """n/U (the Boltzmann prefactor of level populations, ATLAS XNFP*) of a species, (..., L)."""
        kind, a, b = self._lookup(species)
        return self.n_over_u_atom[..., a - 1, b - 1] if kind == "atom" else self.n_over_u_mol[..., a]


class ChemicalEquilibrium:
    """EOS solver: ChemicalEquilibrium().solve(atmosphere, composition) -> Populations."""

    def __init__(self, records: list[MoleculeRecord] | None = None, *, network: str = "full",
                 debye: str = "exact", lagged_lowering: bool = False, t_molecules_max: float = 1.0e4,
                 const: PhysicalConstants = CODATA2018, tol: float = 1e-11, maxiter: int = 60, device=None):
        if network not in ("full", "kurucz") or debye not in ("exact", "kurucz"):
            raise ValueError("network: 'full'|'kurucz', debye: 'exact'|'kurucz'")
        if lagged_lowering and debye != "kurucz":
            raise ValueError("lagged_lowering requires debye='kurucz'")
        self.lagged = lagged_lowering
        records = read_molecules_dat() if records is None else records
        self.network, self.debye_mode, self.const = network, debye, const
        self.t_mol_max, self.tol, self.maxiter = t_molecules_max, tol, maxiter
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        tt = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)
        self.pf = PartitionFunctions(const, self.device)
        self.molecules = MoleculeSet.from_records(records)
        exists = self.pf.exists.clone()
        balance = torch.ones(N_ELEMENTS, dtype=torch.bool, device=self.device)
        self.reported = balance.clone()
        if network == "kurucz":
            listed = np.zeros((N_ELEMENTS, MAX_STAGES), dtype=bool)
            for r in records:
                if r.code < 100.0:
                    z, s = parse_species(f"{r.code:.2f}")
                    listed[z - 1, s - 1] = True
            self.reported = tt(listed.any(1), torch.bool)
            listed[self.molecules.elements - 1, 0] = True       # READMOL: every molecule component gets an equation
            bal = listed.any(1)
            balance = tt(bal, torch.bool)
            exists = torch.where(balance[:, None], tt(listed, torch.bool), exists)
        self.stage_mask, self.balance = exists, balance
        mol_z = self.molecules.elements
        if not bool(balance[mol_z - 1].all()):
            raise ValueError("molecule elements must take part in the equilibrium")
        self.mol_z = torch.as_tensor(mol_z - 1, device=self.device)
        self.select = torch.zeros(len(mol_z), N_ELEMENTS, dtype=torch.float64, device=self.device)
        self.select[torch.arange(len(mol_z)), self.mol_z] = 1.0
        self.is_mol = self.select.sum(0) > 0
        m = self.molecules
        lg = lambda v: np.where(v > 0, np.log(np.maximum(v, 1e-300)), -BIG)
        self.stoich = tt(m.stoich)
        self.ln_nu = tt(lg(m.stoich.T))[self.mol_z]                                  # (M, nmol)
        self.ln_pos, self.ln_el = tt(lg(m.positive)), tt(lg(m.electrons))
        self.ln_q2 = tt(lg(m.charge ** 2))
        self.charge = tt(m.charge)
        s1 = np.arange(MAX_STAGES, dtype=np.float64)
        self.ln_s1, self.ln_s1sq = tt(lg(s1)), tt(lg(s1 ** 2))
        self.mass = atomic_masses(self.device)

    # ------------------------------------------------------------------ species from the unknowns

    def _lowering(self, x, T, ln_ntot, ln_ne_debye):
        nm = len(self.mol_z)
        nc = torch.exp(x[..., nm + 2])
        if self.debye_mode == "kurucz":
            ne_d = torch.exp(torch.where(torch.isnan(ln_ne_debye), x[..., nm], ln_ne_debye))
            nc = kurucz_charge_density(ne_d, torch.exp(ln_ntot))
        return debye_lowering(T, nc, self.const)

    def _species(self, x, T, ln_ntot, ln_a, ln_ne_debye, ln_u_of=None):
        c = self.const
        nm = len(self.mol_z)
        ln_ne, ln_n, ln_nc = x[..., nm], x[..., nm + 1], x[..., nm + 2]
        de = self._lowering(x, T, ln_ntot, ln_ne_debye)
        ln_u = torch.log(self.pf(T, de)) if ln_u_of is None else ln_u_of(de)
        ln_r = log_stage_ratios(log_saha(T, ln_u, self.pf.chi, de, c), ln_ne, self.stage_mask)
        closed = ln_a + ln_n[..., None] - torch.logsumexp(ln_r, -1)
        ln_n1 = torch.where(self.is_mol, x[..., :nm] @ self.select, closed)
        ln_atoms = ln_n1[..., None] + ln_r
        ln_mol = self.molecules.log_k(T) + ln_n1 @ self.stoich.T - self.charge * ln_ne[..., None]
        ln_mol = torch.where(T[..., None] > self.t_mol_max, torch.full_like(ln_mol, -BIG), ln_mol)
        return dict(ln_ne=ln_ne, ln_n=ln_n, ln_nc=ln_nc, de=de, ln_u=ln_u, ln_n1=ln_n1, ln_atoms=ln_atoms,
                    ln_mol=ln_mol)

    def _linearize(self, x, params):
        """Residual with ln U replaced by its tangent in the lowering at x (central difference): same value and
        (to ~1e-8) the same Jacobian at x, without a partition-function evaluation per Jacobian column."""
        T, ln_ntot, _, ln_ne_debye = params
        de0 = self._lowering(x, T, ln_ntot, ln_ne_debye)
        h = 1e-4 * de0
        ln_u0 = torch.log(self.pf(T, de0))
        dln_u = (torch.log(self.pf(T, de0 + h)) - torch.log(self.pf(T, de0 - h))) / (2.0 * h)[..., None, None]
        lin = lambda de: ln_u0 + dln_u * (de - de0)[..., None, None]
        return lambda y, *p: self._residual(y, *p, ln_u_of=lin)

    def _residual(self, x, T, ln_ntot, ln_a, ln_ne_debye, ln_u_of=None):
        sp = self._species(x, T, ln_ntot, ln_a, ln_ne_debye, ln_u_of)
        ln_atoms, ln_mol, ln_ne = sp["ln_atoms"], sp["ln_mol"], sp["ln_ne"][..., None]
        lse = lambda *parts: torch.logsumexp(torch.cat(parts, -1), -1)
        la = ln_atoms[..., self.mol_z, :]
        lm = self.ln_nu + ln_mol[..., None, :]
        r_el = torch.logsumexp(torch.cat([la, lm], -1), -1) - ln_a[..., self.mol_z] - sp["ln_n"][..., None]
        bal = torch.where(self.balance[:, None], ln_atoms, torch.full_like(ln_atoms, -BIG))
        r_p = lse(bal.flatten(-2), ln_mol, ln_ne) - ln_ntot
        r_q = lse((bal + self.ln_s1).flatten(-2), ln_mol + self.ln_pos) - lse(ln_ne, ln_mol + self.ln_el)
        if self.debye_mode == "exact":
            r_c = lse(ln_ne, (bal + self.ln_s1sq).flatten(-2), ln_mol + self.ln_q2) - sp["ln_nc"]
        else:
            r_c = sp["ln_nc"] - torch.log(kurucz_charge_density(torch.exp(sp["ln_ne"]), torch.exp(ln_ntot)))
            r_c = torch.where(torch.isnan(ln_ne_debye), r_c, sp["ln_nc"] - sp["ln_ne"])
        return torch.cat([r_el, torch.stack([r_p, r_q, r_c], -1)], -1)

    # ------------------------------------------------------------------ start: ionisation only

    def _initial(self, T, ln_ntot, ln_a, sweeps: int = 4):
        """x0: n_e from the molecule-free ionisation balance (bisection in ln n_e, the charge excess being
        monotonic in n_e), then Gauss-Seidel sweeps over the molecular elements in decreasing abundance, each
        abundance equation solved for ln n(Z I) with the others fixed (one convex equation at a time,
        Kurucz 1970 p. 57)."""
        ntot = torch.exp(ln_ntot)
        lo, hi = ln_ntot + np.log(1e-30), ln_ntot + np.log(0.999)
        ln_u = torch.log(self.pf(T, debye_lowering(T, 2e-4 * ntot, self.const)))

        def parts(ln_ne):
            ne = torch.exp(ln_ne)
            de = debye_lowering(T, 2.0 * ne, self.const)
            ln_r = log_stage_ratios(log_saha(T, ln_u, self.pf.chi, de, self.const), ln_ne, self.stage_mask)
            ln_n = torch.log(ntot - ne)
            return ln_n, ln_r, ln_a + ln_n[..., None] - torch.logsumexp(ln_r, -1)

        for _ in range(40):
            mid = 0.5 * (lo + hi)
            _, ln_r, ln_n1 = parts(mid)
            la = torch.where(self.balance[:, None], ln_n1[..., None] + ln_r, torch.full_like(ln_r, -BIG))
            f = torch.logsumexp((la + self.ln_s1).flatten(-2), -1) - mid
            lo, hi = torch.where(f > 0, mid, lo), torch.where(f > 0, hi, mid)
        ln_ne = 0.5 * (lo + hi)
        ln_n, ln_r, ln_n1 = parts(ln_ne)
        ln_k = self.molecules.log_k(T) - self.charge * ln_ne[..., None]
        ln_k = torch.where(T[..., None] > self.t_mol_max, torch.full_like(ln_k, -BIG), ln_k)
        order = sorted(range(len(self.mol_z)), key=lambda i: -float(ln_a[..., self.mol_z[i]].max()))
        for _ in range(sweeps):
            for i in order:
                z = int(self.mol_z[i])
                nu = self.stoich[:, z]
                c_mol = ln_k + ln_n1 @ self.stoich.T - nu * ln_n1[..., z:z + 1]
                c = torch.cat([ln_r[..., z, :], torch.where(nu > 0, c_mol, torch.full_like(c_mol, -BIG))], -1)
                k = torch.cat([torch.ones_like(ln_r[0, z]), torch.where(nu > 0, nu, torch.ones_like(nu))])
                b = ln_a[..., z] + ln_n
                y = ln_n1[..., z]
                for _ in range(30):
                    t = c + k * y[..., None]
                    g = torch.logsumexp(t, -1) - b
                    y = y - g / (torch.softmax(t, -1) * k).sum(-1)
                ln_n1 = torch.cat([ln_n1[..., :z], y[..., None], ln_n1[..., z + 1:]], -1)
        return torch.cat([ln_n1[..., self.mol_z], torch.stack([ln_ne, ln_n, ln_ne + np.log(2.0)], -1)], -1)

    # ------------------------------------------------------------------ public

    def solve(self, atm: Atmosphere, comp: Composition, debye_n_e: torch.Tensor | None = None) -> Populations:
        """Populations for every layer (and batch element) of `atm`. debye_n_e (debye="kurucz" only): electron
        density at which the lowering is evaluated instead of the solved one (NMOLEC uses its Newton start, the
        previous layer's n_e scaled by the pressure ratio; lagged_lowering=True computes it self-consistently)."""
        c = self.const
        T = atm.T.to(self.device, torch.float64)
        P = atm.P.to(self.device, torch.float64)
        shape = T.shape
        a = comp.abundance.to(self.device, torch.float64)
        a = a.reshape(a.shape[:-1] + (1,) * (len(shape) - a.dim() + 1) + (N_ELEMENTS,)).expand(shape + (N_ELEMENTS,))
        tf, pf_, af = T.reshape(-1), P.reshape(-1), a.reshape(-1, N_ELEMENTS)
        ln_ntot = torch.log(pf_ / (c.k * tf))
        ln_a = torch.log(torch.clamp(af, min=1e-40))
        if debye_n_e is not None and self.debye_mode != "kurucz":
            raise ValueError("debye_n_e requires debye='kurucz'")
        if debye_n_e is not None and self.lagged:
            raise ValueError("debye_n_e and lagged_lowering exclude each other")
        ln_ne_d = torch.full_like(tf, float("nan")) if debye_n_e is None else \
            torch.log(debye_n_e.to(self.device, torch.float64)).reshape(-1)
        with torch.no_grad():
            x0 = self._initial(tf.detach(), ln_ntot.detach(), ln_a.detach())
        solve = lambda x_start, lag: newton(self._residual, x_start, (tf, ln_ntot, ln_a, lag), tol=self.tol,
                                            maxiter=self.maxiter, linearize=self._linearize)
        x = solve(x0, ln_ne_d)
        if self.lagged:
            nm = len(self.mol_z)
            lag_of = lambda x: self._lagged_ne(x[..., nm].reshape(shape), torch.log(P), ln_ntot.reshape(shape))
            for _ in range(shape[-1] + 1):
                ln_ne_d = lag_of(x).reshape(-1)
                x = solve(x.detach(), ln_ne_d)
                if float((lag_of(x).reshape(-1) - ln_ne_d).abs().max()) < 1e-13:
                    break
            ln_ne_d = lag_of(x).reshape(-1)
        sp = self._species(x, tf, ln_ntot, ln_a, ln_ne_d)
        ln_u, n_atom_ln = sp["ln_u"], sp["ln_atoms"]
        mask = self.stage_mask
        lagged = not bool(torch.isnan(ln_ne_d).all())
        if lagged or self.network == "kurucz":
            de = self._lowering(x, tf, ln_ntot, torch.full_like(tf, float("nan"))) if lagged else sp["de"]
            ln_uf = torch.log(self.pf(tf, de)) if lagged else ln_u
            if lagged:
                ln_u = ln_uf
            ln_r = log_stage_ratios(log_saha(tf, ln_uf, self.pf.chi, de, c), sp["ln_ne"], self.pf.exists)
            free = ln_a + sp["ln_n"][..., None] - torch.logsumexp(ln_r, -1)
            n_atom_ln = torch.where(self.reported[:, None], sp["ln_atoms"], free[..., None] + ln_r)
            mask = torch.where(self.reported[:, None], self.stage_mask, self.pf.exists)
        off = (tf > self.t_mol_max)[..., None]
        zero = lambda v: torch.where(off, torch.zeros_like(v), v)
        n_atom = torch.where(mask, torch.exp(n_atom_ln), torch.zeros_like(n_atom_ln))
        ln_nu = self.molecules.log_n_over_u(tf, sp["ln_n1"], ln_u[..., 0], sp["ln_ne"], c)
        n_nuc = torch.exp(sp["ln_n"])
        rho = n_nuc * (af * self.mass).sum(-1) * c.amu
        ntot = torch.exp(ln_ntot)
        r = lambda v, extra=(): v.reshape(shape + tuple(extra))
        nmol = len(self.molecules)
        return Populations(
            T=T, P=P, n_total=r(ntot), n_e=r(torch.exp(sp["ln_ne"])), n_nuclei=r(n_nuc), rho=r(rho),
            mu=r(rho / (c.amu * ntot)), debye=r(sp["de"]), n_atom=r(n_atom, (N_ELEMENTS, MAX_STAGES)),
            u_atom=r(torch.exp(ln_u), (N_ELEMENTS, MAX_STAGES)), n_mol=r(zero(torch.exp(sp["ln_mol"])), (nmol,)),
            n_over_u_mol=r(zero(torch.exp(ln_nu)), (nmol,)), molecules=self.molecules)

    @staticmethod
    def _lagged_ne(ln_ne: torch.Tensor, ln_p: torch.Tensor, ln_ntot: torch.Tensor) -> torch.Tensor:
        """ln of NMOLEC's Newton-start n_e per layer: n_e(J-1) P(J)/P(J-1), layer 1: (P/kT)/20 (loops 32-34)."""
        first = ln_ntot[..., :1] - np.log(20.0)
        return torch.cat([first, ln_ne[..., :-1] + ln_p[..., 1:] - ln_p[..., :-1]], -1)
