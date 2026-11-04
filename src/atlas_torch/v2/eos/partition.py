"""Atomic partition functions U(T, dE) and ionisation energies for Z=1..99, all stages ATLAS tabulates.

Physics: Kurucz 1970 (SAO SR 309) sec. 4.4, pp. 63-66: U = sum_i g_i exp(-E_i/kT) truncated at the continuum
lowered by dE (Debye, eq. 4.36); the levels above the explicit ones are treated as hydrogenic and
integrated in closed form (eq. 4.40). Data, as used by atlas7v.for SUBROUTINE PFSAHA:
  * pretabulated U at dE = 0.1 eV on T/T0 = 1.5, 2.5, ..., 10.5, T0 = chi[eV]*2000/11, linear in T, plus the
    eq. 4.40 integral from dE = 0.1 eV to the actual dE with weight G (DATA NNN);
  * explicit level lists + eq. 4.40 tail for the ions ATLAS can treat in NLTE (PFSAHA labels 1100-1210,
    DATA EHYD/GHYD ... and the pseudo-levels in those statements);
  * low-lying levels (FUNCTION PFGROUND) as a lower bound at T < 2 T0;
  * Ca-Ni (Z=20-28), stages I-X: U(log T, dE) table of SUBROUTINE PFIRON (DATA PFTAB);
  * ionisation energies: SUBROUTINE IONPOTS (DATA POTION, cm^-1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from ...tables import table
from ..constants import CODATA2018, MAX_STAGES, N_ELEMENTS, PhysicalConstants, decimal32
from ..io.fortran_source import evaluate, statements

SRC = "atlas7v.for"
_TABLE, _LEVELS, _IRON = 0, 1, 2


# ---------------------------------------------------------------------------------- Boltzmann-sum algebra

class _Lin:
    """c * hc/kT (E in cm^-1)."""

    def __init__(self, c: float):
        self.c = c

    def __mul__(self, o):
        return _Lin(self.c * float(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return _Lin(self.c / float(o))

    def __neg__(self):
        return _Lin(-self.c)


class _Lev:
    """sum_i g_i exp(-E_i hc/kT): {E_i (cm^-1): g_i}."""

    def __init__(self, terms: dict):
        self.t = dict(terms)

    @staticmethod
    def of(v):
        if isinstance(v, _Lin):
            raise ValueError("energy term outside EXP")
        return v if isinstance(v, _Lev) else _Lev({0.0: float(v)})

    def __add__(self, o):
        o = _Lev.of(o)
        t = dict(self.t)
        for e, g in o.t.items():
            t[e] = t.get(e, 0.0) + g
        return _Lev(t)

    __radd__ = __add__

    def __mul__(self, o):
        o = _Lev.of(o)
        t: dict = {}
        for e1, g1 in self.t.items():
            for e2, g2 in o.t.items():
                t[e1 + e2] = t.get(e1 + e2, 0.0) + g1 * g2
        return _Lev(t)

    __rmul__ = __mul__

    def arrays(self):
        e = np.array(sorted(self.t))
        return np.array([self.t[x] for x in e]), e


def _exp(v):
    if not isinstance(v, _Lin):
        raise ValueError("EXP of a non-energy argument in a level sum")
    return _Lev({-v.c: 1.0})


# ------------------------------------------------------------------------------------------- the data

def nnn_row(z: int, stage: int) -> int:
    """0-based row of DATA NNN(6,374) for element z, stage (PFSAHA: N=LOCZ(IZ)+ION-1; C, N, O moved to
    rows 354, 360, 367; Z>28 at 3*IZ+54)."""
    locz = np.asarray(table(SRC, "pfsaha", "locz"))
    first = {6: 354, 7: 360, 8: 367}.get(z, locz[z - 1] if z <= 28 else 3 * z + 54)
    return int(first) + stage - 2


def n_stages(z: int) -> int:
    """Number of ionisation stages tabulated for element z (PFSAHA NIONS)."""
    locz = np.asarray(table(SRC, "pfsaha", "locz"))
    if z in (6, 7, 8):
        return {6: 6, 7: 7, 8: 8}[z]
    if 20 <= z <= 28:
        return 10
    return int(locz[z] - locz[z - 1]) if z <= 28 else 3


@lru_cache(maxsize=None)
def ionisation_energies_cm() -> np.ndarray:
    """chi[Z-1, s-1] (cm^-1): energy to ionise stage s (IONPOTS POTION; PFSAHA falls back to the previous
    entry where POTION is 0, which happens only for the last stage). 0 for absent stages."""
    pot = np.asarray(table(SRC, "ionpots", "potion"), dtype=np.float64)
    chi = np.zeros((N_ELEMENTS, MAX_STAGES))
    for z in range(1, N_ELEMENTS + 1):
        for s in range(1, n_stages(z) + 1):
            i = (z * (z + 1) // 2 + s - 1) if z <= 30 else (z * 5 + 341 + s - 1)
            chi[z - 1, s - 1] = pot[i - 1] if pot[i - 1] != 0 else pot[i - 2]
    return chi


@lru_cache(maxsize=None)
def _nnn_values() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decoded NNN: U at the 10 nodes (374, 10), G (374,), and the lower bound applied below the first node."""
    nnn = np.asarray(table(SRC, "pfsaha", "nnn"), dtype=np.int64)          # (6, 374)
    scale = np.array([0.001, 0.01, 0.1, 1.0])
    k1 = nnn[:5] // 100000
    k3 = (nnn[:5] % 100000) // 10
    ks = scale[nnn[:5] % 10 - 1]
    vals = np.empty((374, 10))
    vals[:, 0::2] = (k1 * ks).T
    vals[:, 1::2] = (k3 * ks).T
    g = (nnn[5] % 100).astype(np.float64)
    # PFSAHA: below the first node the extrapolation is floored at the ground weight when the first two
    # nodes (scale 0.001) both round to the same integer.
    floor_ok = (nnn[0] % 10 == 1) & (np.floor(vals[:, 0]) == np.floor(vals[:, 1] + 0.5))
    pmin = np.where(floor_ok, np.floor(vals[:, 0]), 1.0)
    return vals, g, pmin


@dataclass(frozen=True)
class LevelBlock:
    """Explicit levels of one ion and its hydrogenic tail (G, start of the tail R/n*^2 in cm^-1)."""
    g: np.ndarray
    e_cm: np.ndarray
    tail_g: float | None
    tail_start_cm: float | None


@lru_cache(maxsize=None)
def level_blocks() -> dict[tuple[int, int], LevelBlock]:
    """(Z, stage) -> LevelBlock from PFSAHA labels 1100-1210, read from the statements of atlas7v.for."""
    st = statements(SRC, "PFSAHA")
    where = {lab: i for i, (lab, _) in enumerate(st) if lab is not None}
    targets = {}
    for _, s in st:
        m = re.fullmatch(r"IF\(N\.EQ\.(\d+)\)GOTO(\d+)", s.replace(" ", ""))
        if m:
            targets[int(m.group(1))] = int(m.group(2))
    row_of = {nnn_row(z, s): (z, s) for z in range(1, N_ELEMENTS + 1) if not 20 <= z <= 28
              for s in range(1, n_stages(z) + 1)}
    _, gtab, _ = _nnn_values()
    arrays = {}

    def array(name):
        if name not in arrays:
            arrays[name] = decimal32(table(SRC, "pfsaha", name.lower())).astype(np.float64)
        return arrays[name]

    out = {}
    for n, lab in targets.items():
        if n - 1 not in row_of:              # Ca I/II rows: Z=20-28 go to PFIRON before these labels
            continue
        z, stage = row_of[n - 1]
        names = {"B": 1.0, "HCKT": _Lin(1.0)}
        fns = {"EXP": _exp}
        part, g_tail, start, tail = None, gtab[n - 1], None, None
        i = where[lab]
        while tail is None:
            s = st[i][1].replace(" ", "")
            i += 1
            if s in ("GOTO14", "GOTO18"):
                tail = s == "GOTO14"
            elif s.startswith("B=") or s.startswith("IF(NLTEON"):
                continue
            elif (m := re.fullmatch(r"DO(\d+)I=(\d+),(\d+)", s)):
                body = st[where[int(m.group(1))]][1].replace(" ", "")
                rhs = body.split("=", 1)[1].split("+", 1)[1]
                f = {**fns, **{a: (lambda k, a=a: array(a)[int(k) - 1]) for a in re.findall(r"([A-Z]\w*)\(I\)", rhs)}}
                for k in range(int(m.group(2)), int(m.group(3)) + 1):
                    part = part + evaluate(rhs, {**names, "I": float(k)}, f)
                i = where[int(m.group(1))] + 1
            elif (m := re.fullmatch(r"PART\((\d+)\)=(.*)", s)):
                assert int(m.group(1)) == stage, (n, s)
                rhs, acc = m.group(2), f"PART({stage})+"
                part = part + evaluate(rhs[len(acc):], names, fns) if rhs.startswith(acc) else \
                    _Lev.of(evaluate(rhs, names, fns))
            elif s.startswith("G="):
                g_tail = float(evaluate(s[2:], {}))
            elif s.startswith("D1="):
                start = evaluate(s[3:], names, fns).c
            else:
                raise ValueError(f"unexpected statement in PFSAHA level block {lab}: {s}")
        g, e = part.arrays()
        out[(z, stage)] = LevelBlock(g, e, g_tail if tail else None, start if tail else None)
    return out


@lru_cache(maxsize=None)
def _pfground_dispatch() -> tuple[dict[int, int], dict[int, str]]:
    """NELION -> label through FUNCTION PFGROUND's IF(range)THEN / computed GO TO blocks; label -> expression."""
    st = statements(SRC, "PFGROUND")
    stmt = {lab: s.replace(" ", "") for lab, s in st if lab is not None}
    nelion_label = {}
    for i, (_, s) in enumerate(st):
        s = s.replace(" ", "")
        m = re.fullmatch(r"IF\((?:NELION\.GT\.(\d+)\.AND\.)?NELION\.LT\.(\d+)\)THEN", s)
        if not m:
            continue
        nxt = st[i + 1][1].replace(" ", "")
        g = re.fullmatch(r"GOTO\(([\d,]+)\),?NELION(?:-(\d+))?", nxt)
        if not g:
            continue
        labels = [int(x) for x in g.group(1).split(",")]
        lo, hi, off = int(m.group(1) or 0), int(m.group(2)), int(g.group(2) or 0)
        for nelion in range(lo + 1, hi):
            if 1 <= nelion - off <= len(labels):
                nelion_label[nelion] = labels[nelion - off - 1]
    return nelion_label, {lab: s.split("=", 1)[1] for lab, s in stmt.items() if s.startswith("PFGROUND=")}


@lru_cache(maxsize=None)
def ground_levels() -> dict[tuple[int, int], LevelBlock]:
    """(Z, stage) -> low-lying levels of FUNCTION PFGROUND, with NELION = 6*(Z-1)+stage and PFSAHA's
    redirections (O I -> 431, F I -> 491, F II -> 501, which no IF range of PFGROUND reaches); only ions the
    dispatch reaches (H-K except NELION 114, Cu-Ba).

    Deviations from the Fortran: label 43 (N VII, '2.') has no RETURN and falls through to label 431 (O I
    levels); v2 uses its own value. Label 54 (F VI) has '5.*(-HCK/T*97427.D0)' (EXP missing): skipped (the
    Fortran value is below the table at every T, so its MAX keeps the table). Label 66 (Na VI) multiplies
    two level terms ('+' typed as '*'): kept as the Fortran evaluates it.
    """
    nelion_label, expr = _pfground_dispatch()
    redirected = set()
    for _, s in statements(SRC, "PFSAHA"):
        m = re.fullmatch(r"IF\(IZ\.EQ\.(\d+)\.AND\.ION\.EQ\.(\d+)\)NELION=\d+", s.replace(" ", ""))
        if m:
            redirected.add((int(m.group(1)), int(m.group(2))))
    out = {}
    for z in range(1, N_ELEMENTS + 1):
        for stage in range(1, n_stages(z) + 1):
            nelion = 6 * (z - 1) + stage
            if (z, stage) in redirected or nelion not in nelion_label:
                continue
            try:
                lev = _Lev.of(evaluate(expr[nelion_label[nelion]], {"HCK": _Lin(1.0), "T": 1.0}, {"EXP": _exp}))
            except ValueError:
                continue
            g, e = lev.arrays()
            if len(g) > 1 or g[0] > 1.0:
                out[(z, stage)] = LevelBlock(g, e, None, None)
    return out


# PFIRON grids: log10 T nodes 3.32(.02)3.70(.03)4.00(.05)5.30; lowering 500*2^k cm^-1, k=0..6
_IRON_LOGT = np.concatenate([3.32 + 0.02 * np.arange(20), 3.70 + 0.03 * np.arange(1, 11),
                             4.00 + 0.05 * np.arange(1, 27)])
_IRON_LOWER = 500.0 * 2.0 ** np.arange(7)


@lru_cache(maxsize=None)
def _iron_table() -> np.ndarray:
    """PFTAB(7,56,10,9) -> [Z-20, stage-1, lowering, logT]."""
    return np.ascontiguousarray(decimal32(table(SRC, "pfiron", "pftab")).transpose(3, 2, 0, 1))


# ------------------------------------------------------------------------------------- evaluation

def _poly(d):
    """Series of the hydrogenic-level integral, Kurucz 1970 eq. 4.40: 1/3 + d - d^2/2 - d^3/18 - d^4/120."""
    return 1.0 / 3.0 + (1.0 - (0.5 + (1.0 / 18.0 + d / 120.0) * d) * d) * d


def _pad(blocks, keys):
    k = max(len(blocks[key].g) for key in keys)
    g = np.zeros((len(keys), k))
    e = np.zeros((len(keys), k))
    for i, key in enumerate(keys):
        g[i, :len(blocks[key].g)] = blocks[key].g
        e[i, :len(blocks[key].e_cm)] = blocks[key].e_cm
    return g, e


class PartitionFunctions:
    """U[..., Z-1, s-1](T, dE) for every element and stage; 1 for absent stages.

    T (...) in K; dE (...) Debye lowering of the ionisation energy per unit charge in eV (stage s is
    lowered by s*dE, Kurucz 1970 eq. 4.36 with Z_eff = s). Where the lowering exceeds the start of the eq. 4.40
    tail the tail is negative (levels removed); U is floored at the ground weight (not in PFSAHA, which only
    reaches this at densities beyond stellar atmospheres).
    """

    def __init__(self, const: PhysicalConstants = CODATA2018, device=None, dtype=torch.float64):
        self.const = const
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        ti = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
        chi_cm = ionisation_energies_cm()
        self.chi = tt(chi_cm / const.ev_cm)                                   # eV
        self.exists = torch.zeros(N_ELEMENTS, MAX_STAGES, dtype=torch.bool, device=device)
        for z in range(1, N_ELEMENTS + 1):
            self.exists[z - 1, :n_stages(z)] = True
        lev = level_blocks()
        kind = {}
        for z in range(1, N_ELEMENTS + 1):
            for s in range(1, n_stages(z) + 1):
                kind[(z, s)] = _IRON if 20 <= z <= 28 else (_LEVELS if (z, s) in lev else _TABLE)
        keys = {k: sorted(key for key, v in kind.items() if v == k) for k in (_TABLE, _LEVELS, _IRON)}
        flat = lambda key: (key[0] - 1) * MAX_STAGES + key[1] - 1
        order = keys[_TABLE] + keys[_LEVELS] + keys[_IRON]
        absent = [i for i in range(N_ELEMENTS * MAX_STAGES) if i not in {flat(k) for k in order}]
        perm = [flat(k) for k in order] + absent
        self._inverse = ti(np.argsort(perm))
        self._n_absent = len(absent)

        vals, gtab, pmin = _nnn_values()
        rows = [nnn_row(*k) for k in keys[_TABLE]]
        chi_t = np.array([chi_cm[k[0] - 1, k[1] - 1] for k in keys[_TABLE]]) / const.ev_cm
        gr = ground_levels()
        with_ground = [k for k in keys[_TABLE] if k in gr]
        gg, ge = _pad(gr, with_ground)
        gidx = {k: i for i, k in enumerate(with_ground)}
        self.t_vals = tt(vals[rows])
        self.t_t0 = tt(chi_t * 2000.0 / 11.0)
        self.t_chi = tt(chi_t)
        self.t_g = tt(gtab[rows])
        self.t_pmin = tt(pmin[rows])
        self.t_z = tt([k[1] for k in keys[_TABLE]])
        self.t_ground_g = tt(np.concatenate([gg, np.zeros((1, gg.shape[1]))]))
        self.t_ground_e = tt(np.concatenate([ge, np.zeros((1, ge.shape[1]))]))
        self.t_ground_idx = ti([gidx.get(k, len(with_ground)) for k in keys[_TABLE]])
        self.t_has_ground = torch.as_tensor([k in gidx for k in keys[_TABLE]], device=device)

        lg, le = _pad(lev, keys[_LEVELS])
        self.l_g, self.l_e = tt(lg), tt(le)
        self.l_g0 = tt([lev[k].g[np.argmin(lev[k].e_cm)] for k in keys[_LEVELS]])
        self.l_chi = tt([chi_cm[k[0] - 1, k[1] - 1] / const.ev_cm for k in keys[_LEVELS]])
        self.l_z = tt([k[1] for k in keys[_LEVELS]])
        self.l_tail_g = tt([lev[k].tail_g or 0.0 for k in keys[_LEVELS]])
        self.l_tail_start = tt([lev[k].tail_start_cm or 1.0 for k in keys[_LEVELS]])

        it = _iron_table()
        self.i_tab = tt(np.stack([it[k[0] - 20, k[1] - 1] for k in keys[_IRON]]).reshape(len(keys[_IRON]), -1))
        self.i_z = tt([k[1] for k in keys[_IRON]])
        self.i_logt = tt(_IRON_LOGT)
        self.i_loglow = tt(np.log10(_IRON_LOWER))

    def _tail(self, g, chi, z, lower, start_ev, tv):
        """Kurucz 1970 eq. 4.40 between start_ev and the lowered continuum `lower` (eV)."""
        ry = self.const.ry_ev
        lower = lower.clamp(min=1e-8)
        return g * torch.exp(-chi / tv) * ((ry * z * z / lower) ** 1.5 * _poly(lower / tv)
                                           - (ry * z * z / start_ev) ** 1.5 * _poly(start_ev / tv))

    def _tabulated(self, T, dE):
        c = self.const
        x = T[..., None] / self.t_t0
        it = torch.clamp(torch.floor(x - 0.5), 1, 9).detach()
        dt = x - it - 0.5
        i = it.long() - 1
        vals = self.t_vals.expand(x.shape + (10,))
        p1 = torch.gather(vals, -1, i[..., None])[..., 0]
        p2 = torch.gather(vals, -1, i[..., None] + 1)[..., 0]
        u = torch.maximum(torch.where(dt < 0, self.t_pmin, torch.ones_like(dt)), p1 + (p2 - p1) * dt)
        hckt = c.hc_k / T[..., None, None]
        ug = (self.t_ground_g * torch.exp(-self.t_ground_e * hckt)).sum(-1)[..., self.t_ground_idx]
        u = torch.where(self.t_has_ground & (x < 2.0), torch.maximum(ug, u), u)
        lower = self.t_z * dE[..., None]
        tv = torch.minimum(c.k_ev * T[..., None], 11.0 * self.t_t0 * c.k_ev)
        add = (self.t_g != 0) & (lower >= 0.1) & (x >= 4.0)
        u = u + torch.where(add, self._tail(self.t_g, self.t_chi, self.t_z, lower, 0.1, tv), torch.zeros_like(u))
        return torch.maximum(u, self.t_pmin)

    def _levels(self, T, dE):
        c = self.const
        hckt = c.hc_k / T[..., None]
        u = (self.l_g * torch.exp(-self.l_e * hckt[..., None])).sum(-1)
        tv = c.k_ev * T[..., None]
        start = self.l_tail_start * hckt * tv
        tail = self._tail(self.l_tail_g, self.l_chi, self.l_z, self.l_z * dE[..., None], start, tv)
        u = u + torch.where(self.l_tail_g != 0, tail, torch.zeros_like(u))
        return torch.maximum(u, self.l_g0)

    def _iron(self, T, dE):
        lt = torch.log10(T)
        k = torch.clamp(torch.bucketize(lt.detach(), self.i_logt), 1, len(self.i_logt) - 1)
        t0, t1 = self.i_logt[k - 1], self.i_logt[k]
        f = ((lt - t0) / (t1 - t0))[..., None]
        low = torch.log10(self.i_z * dE[..., None] * self.const.ev_cm)
        j = torch.clamp(torch.bucketize(low.detach(), self.i_loglow), 1, len(self.i_loglow) - 1)
        p = torch.clamp((low - self.i_loglow[j - 1]) / (self.i_loglow[j] - self.i_loglow[j - 1]), 0.0, 1.0)
        tab = self.i_tab.expand(low.shape + self.i_tab.shape[-1:])
        n = len(self.i_logt)
        at = lambda jj, kk: torch.gather(tab, -1, (jj * n + kk)[..., None])[..., 0]
        kk0, kk1 = k[..., None] - 1, k[..., None]
        hi = f * at(j, kk1) + (1 - f) * at(j, kk0)
        lo = f * at(j - 1, kk1) + (1 - f) * at(j - 1, kk0)
        return p * hi + (1 - p) * lo

    def __call__(self, T: torch.Tensor, dE: torch.Tensor) -> torch.Tensor:
        parts = [self._tabulated(T, dE), self._levels(T, dE), self._iron(T, dE),
                 torch.ones(T.shape + (self._n_absent,), dtype=T.dtype, device=T.device)]
        return torch.cat(parts, -1)[..., self._inverse].reshape(T.shape + (N_ELEMENTS, MAX_STAGES))
