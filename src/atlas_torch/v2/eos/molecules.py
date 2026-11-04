"""Molecular equilibrium constants (Kurucz 1970 sec. 4.4, eqs. 4.34-4.35, 4.43) for the species of molecules.dat."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from ..constants import CODATA2018, FIT_K_EV, N_ELEMENTS, PhysicalConstants, atomic_masses, element_symbols
from ..io.fortran_source import evaluate, statements
from ..io.molecules_dat import MoleculeRecord, read_molecules_dat


def decode_code(code: float) -> tuple[dict[int, int], int, int]:
    """Kurucz species code -> ({Z: count}, electrons attached, positive charge).

    Integer part: atomic numbers as base-100 digits, '00' an attached electron (100. = H-); fractional part
    *100 = positive charge (101.01 = H2+, 26.01 = Fe II) (Kurucz 1970 p. 59; READMOL)."""
    whole = int(round(code * 100)) // 100
    charge = int(round(code * 100)) % 100
    atoms: dict[int, int] = {}
    electrons = 0
    while whole:
        whole, d = divmod(whole, 100)
        if d == 0:
            electrons += 1
        else:
            atoms[d] = atoms.get(d, 0) + 1
    return atoms, electrons, charge


def formula(code: float) -> str:
    atoms, el, q = decode_code(code)
    sym = element_symbols()
    s = "".join(sym[z - 1] + (str(n) if n > 1 else "") for z, n in sorted(atoms.items()))
    return s + "-" * el + "+" * q


@lru_cache(maxsize=None)
def h2_log_k_expression() -> str:
    """NMOLEC's built-in H2 equilibrium constant (used instead of the molecules.dat entry for code 101.):
    the argument of EXP in the statement after IF(CODE(JMOL).EQ.101.)THEN."""
    st = [s.replace(" ", "") for _, s in statements("atlas7v.for", "NMOLEC")]
    i = st.index("IF(CODE(JMOL).EQ.101.)THEN")
    m = re.fullmatch(r"EQUILJ\(JMOL\)=EXP\((.*)\)", st[i + 1])
    return m.group(1)


@dataclass(frozen=True)
class MoleculeSet:
    """Molecules (and molecular / negative ions) entering the equilibrium.

    stoich (nmol, 99): atoms of element Z per molecule; electrons: attached electrons (H- = 1); positive:
    positive charge; d0 (eV) and coeffs b..g of eq. 4.35; mass (amu); h2: True for code 101. (NMOLEC's fit).
    """
    codes: np.ndarray
    stoich: np.ndarray
    electrons: np.ndarray
    positive: np.ndarray
    d0: np.ndarray
    coeffs: np.ndarray
    mass: np.ndarray
    h2: np.ndarray

    @classmethod
    def from_records(cls, records: list[MoleculeRecord]) -> "MoleculeSet":
        mol = [r for r in records if r.code >= 100.0]
        stoich = np.zeros((len(mol), N_ELEMENTS))
        electrons = np.zeros(len(mol))
        positive = np.zeros(len(mol))
        am = atomic_masses().numpy()
        mass = np.zeros(len(mol))
        for i, r in enumerate(mol):
            atoms, el, q = decode_code(r.code)
            for z, n in atoms.items():
                stoich[i, z - 1] = n
            electrons[i], positive[i] = el, q
            mass[i] = sum(n * am[z - 1] for z, n in atoms.items())
        codes = np.array([r.code for r in mol])
        return cls(codes, stoich, electrons, positive, np.array([r.d0 for r in mol]),
                   np.array([r.coeffs for r in mol], dtype=np.float64), mass, np.isclose(codes, 101.0))

    @classmethod
    def default(cls) -> "MoleculeSet":
        return cls.from_records(read_molecules_dat())

    @property
    def charge(self) -> np.ndarray:
        return self.positive - self.electrons

    def __len__(self) -> int:
        return len(self.codes)

    def index(self, code: float) -> int:
        hit = np.flatnonzero(np.abs(self.codes - code) < 1e-3)
        if not len(hit):
            raise KeyError(f"molecule {code} not in the set")
        return int(hit[0])

    @property
    def elements(self) -> np.ndarray:
        """Atomic numbers occurring in any molecule."""
        return np.flatnonzero(self.stoich.sum(0)) + 1

    def log_k(self, T: torch.Tensor) -> torch.Tensor:
        """ln K_m(T), (..., nmol), with n_m = K_m prod_Z n(Z I)^nu_mZ n_e^-charge_m (cm^-3 units):
        eq. 4.35, ln K = D0/kT - b + cT - dT^2 + eT^3 - fT^4 + gT^5 - 1.5 (m - 1 - k) ln T, m = atoms plus
        attached electrons, k = positive charge; evaluated with the fits' kT (FIT_K_EV). Code 101. (H2) uses
        NMOLEC's built-in fit."""
        t = T[..., None]
        tt = lambda a: torch.as_tensor(a, dtype=T.dtype, device=T.device)
        c = tt(self.coeffs)
        n_atoms = tt(self.stoich.sum(1) + self.electrons)
        k = tt(self.positive)
        poly = (c[:, 1] + (-c[:, 2] + (c[:, 3] + (-c[:, 4] + c[:, 5] * t) * t) * t) * t) * t
        lnk = tt(self.d0) / (FIT_K_EV * t) - c[:, 0] + poly - 1.5 * (n_atoms - 1 - k) * torch.log(t)
        if self.h2.any():
            h2 = evaluate(h2_log_k_expression(), {"T": T, "TKEV": FIT_K_EV * T, "TLOG": torch.log(T)})
            lnk = torch.where(torch.as_tensor(self.h2, device=T.device), h2[..., None], lnk)
        return lnk

    def log_n_over_u(self, T: torch.Tensor, ln_n1: torch.Tensor, ln_u1: torch.Tensor, ln_ne: torch.Tensor,
                     const: PhysicalConstants = CODATA2018) -> torch.Tensor:
        """ln(n/U) of each molecule from the neutral-atom densities ln_n1 and partition functions ln_u1
        (..., 99) and ln n_e (...): eq. 4.43,
        n/U = exp(D0/kT) prod_atoms [n_a/(U_a Q_a)] [n_e/(2 Q_e)]^(electrons - charge+) Q_mol,
        Q = (2 pi m k T/h^2)^1.5."""
        tt = lambda a: torch.as_tensor(a, dtype=T.dtype, device=T.device)
        lt = torch.log(T)[..., None]
        ln_qa = np.log(const.saha_amu) + 1.5 * (torch.log(atomic_masses(T.device)) + lt)        # (..., 99)
        ln_qe = np.log(2.0 * const.saha_e) + 1.5 * lt[..., 0]
        ln_qm = np.log(const.saha_amu) + 1.5 * (torch.log(tt(self.mass)) + lt)
        per_atom = (ln_n1 - ln_u1 - ln_qa) @ tt(self.stoich).T
        return tt(self.d0) / (const.k_ev * T[..., None]) + per_atom + ln_qm \
            + tt(-self.charge) * (ln_ne - ln_qe)[..., None]
