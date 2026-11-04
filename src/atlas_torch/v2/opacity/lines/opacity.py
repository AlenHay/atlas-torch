"""LTE line absorption coefficient per gram on a log-uniform wavelength grid (Kurucz & Avrett 1981, SAO SR 391,
pp. 4-5; synthe.for main line loop and XLINOP):

  kappa_l(nu) = (pi e^2 / m_e c) gf (n/U) / rho exp(-E_lo / kT) phi(nu) (1 - exp(-h nu / kT)),
  phi(nu) = H(a, v) / (sqrt(pi) Delta nu_D),  Delta nu_D = (nu_0/c) sqrt(2kT/m + xi^2),
  a = (Gamma_R + Gamma_S n_e + Gamma_W n_pert) / (4 pi Delta nu_D),  v = |nu - nu_0| / Delta nu_D,
  n_pert = (n_H I + 0.42 n_He I + 0.85 n_H2) (T / 10^4 K)^0.3.

A line contributes at a depth only where kappa_l >= cutoff x kappa_continuum (absorption + scattering) at the same
wavelength (SYNTHE CUTOFF, K&A p. 5). Hydrogen lines use the Stark profiles of hydrogen.py; autoionizing lines
the Fano profile of XLINOP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from ....tables import table
from ...atmosphere import Atmosphere
from ...constants import CODATA2018, PhysicalConstants, atomic_masses
from ...eos.equilibrium import Populations
from .. import Opacity
from .grid import LogWavelengthGrid
from .linelist import LineKind, LineList
from .voigt import SQRT_PI, voigt, voigt_kurucz

C_CM = 2.99792458e10
C_NM = 2.99792458e17


def pi_e2_over_mc(const: PhysicalConstants) -> float:
    """pi e^2 / (m_e c), cm^2 s^-1 (classical line strength per unit gf); 0.026538 in the Kurucz codes
    (rgfalllinesnew.for CGF)."""
    if const.name == "KURUCZ":
        return 0.026538
    e_esu, m_e = 4.803204712570263e-10, 9.1093837015e-28
    return math.pi * e_esu ** 2 / (m_e * const.c)


@dataclass(frozen=True)
class GasState:
    """What the line opacity needs per layer, every tensor (M,) or (M, ...) with M = flattened layers.

    n_over_u_atom (M, 99, 10), n_over_u_mol (M, nmol) with codes mol_codes; number densities cm^-3; vturb cm/s.
    """
    T: torch.Tensor
    n_e: torch.Tensor
    rho: torch.Tensor
    n_h1: torch.Tensor
    n_he1: torch.Tensor
    n_h2: torch.Tensor
    vturb: torch.Tensor
    n_over_u_atom: torch.Tensor
    n_over_u_mol: torch.Tensor
    mol_codes: np.ndarray
    batch_shape: tuple = ()

    @classmethod
    def from_populations(cls, atm: Atmosphere, pops: Populations) -> "GasState":
        """From the v2 EOS (n_over_u of every atom, ion and molecule; atm.vturb = model microturbulence)."""
        shape = tuple(pops.T.shape)
        fl = lambda x: x.reshape(-1)
        h2 = pops.molecules.index(101.0)
        vt = atm.vturb if atm.vturb is not None else torch.zeros_like(pops.T)
        return cls(T=fl(pops.T), n_e=fl(pops.n_e), rho=fl(pops.rho), n_h1=fl(pops.n_atom[..., 0, 0]),
                   n_he1=fl(pops.n_atom[..., 1, 0]), n_h2=fl(pops.n_mol[..., h2]),
                   vturb=fl(vt.to(pops.T)), n_over_u_atom=pops.n_over_u_atom.reshape(-1, 99, 10),
                   n_over_u_mol=pops.n_over_u_mol.reshape(-1, pops.n_over_u_mol.shape[-1]),
                   mol_codes=np.asarray(pops.molecules.codes), batch_shape=shape)

    def n_over_u(self, species: torch.Tensor) -> torch.Tensor:
        """(M, n) n/U of each line's species (0 where the EOS does not carry it)."""
        sp = species.to(self.T.device)
        mol = sp >= 100.0
        z = sp.floor().long().clamp(1, 99)
        st = torch.round((sp - sp.floor()) * 100.0).long().clamp(0, 9)
        atom = self.n_over_u_atom[:, z - 1, st]
        codes = torch.as_tensor(self.mol_codes, dtype=sp.dtype, device=sp.device)
        hit = (sp[:, None] - codes[None, :]).abs() < 1e-3
        idx = hit.float().argmax(1)
        molv = self.n_over_u_mol[:, idx] * hit.any(1)
        return torch.where(mol, molv, atom)


@dataclass(frozen=True)
class LineOpacityOptions:
    """cutoff: SYNTHE CUTOFF (BasicATLAS 1e-4); microturbulence: added in quadrature to the model's, cm/s
    (SYNTHE TURBV, BasicATLAS 1.5 km/s). Kurucz-compatible switches, to attribute differences:
      voigt="kurucz"        SYNTHE's profile approximations (voigt.voigt_kurucz)
      line_center="exact"   line centres at their own wavelengths; the default "grid" moves them to the nearest
                            grid point as SYNTHE does (K&A p. 4: up to half a step, c/2R = 0.25 km/s at R = 6e5)
      doppler_mass="species" element / main-isotopologue mass in Delta nu_D instead of the line's isotope
      max_wing_points=10000 wings truncated as SYNTHE's MAXPROF
      cutoff_rule="kurucz"  the point nearest the centre always kept, wings compared with cutoff x continuum at
                            the line centre (instead of at each wavelength), and the first point below it kept
      helium_ii="skip"      He II lines left out (XLINOP label 400)
    budget: elements per profile-evaluation block (memory bound)."""
    cutoff: float = 1e-4
    microturbulence: float = 1.5e5
    voigt: str = "exact"
    line_center: str = "grid"
    doppler_mass: str = "line"
    max_wing_points: int | None = None
    cutoff_rule: str = "local"
    helium_ii: str = "voigt"
    budget: int = 1 << 21
    const: PhysicalConstants = field(default=CODATA2018)


def merging_wavenumber(n_e: torch.Tensor, hydrogen: bool) -> torch.Tensor:
    """Wavenumber below the series limit where lines merge into the continuum, cm^-1: R/n_m^2 with
    n_m = 1600 n_e^(-2/15) - 1.5 (empirical Inglis-Teller, XLINOP loop 91)."""
    nm = 1600.0 / n_e ** (2.0 / 15.0) - 1.5
    return (109677.576 if hydrogen else 109737.312) / nm ** 2


# species of the columns of the merged-continuum edge table (XLINOP DATA CONTX)
CODEX = (1.0, 2.0, 2.01, 6.0, 6.01, 12.0, 12.01, 13.0, 13.01, 14.0, 14.01, 20.0, 20.01, 8.0, 11.0, 5.0, 19.0)


def series_ramp(wl_nm: torch.Tensor, species: torch.Tensor, ncon: torch.Tensor, n_e: torch.Tensor) -> torch.Tensor:
    """Weight of lines of a series near its merged limit (XLINOP label 200, NCON > 0): 0 shortward of
    lambda_con = 1/(sigma_lim - sigma_merge), rising linearly to 1 at 1/(sigma_lim - sigma_merge - 500 cm^-1);
    sigma_lim from the CONTX edge table. Shapes: wl (M, n, w), species/ncon (n,), n_e (M,)."""
    contx = torch.as_tensor(table("synthe.for", "xlinop", "contx"), dtype=wl_nm.dtype, device=wl_nm.device)
    col = torch.zeros_like(ncon)
    for i, c in enumerate(CODEX):
        col = torch.where((species - c).abs() < 1e-3, torch.full_like(col, i), col)
    lim = contx[(ncon - 1).clamp(min=0), col]
    em = merging_wavenumber(n_e, False)[:, None]
    wcon = (1e7 / (lim[None, :] - em))[..., None]
    wtail = (1e7 / (lim[None, :] - em - 500.0))[..., None]
    r = ((wl_nm - wcon) / (wtail - wcon)).clamp(0.0, 1.0)
    return torch.where((ncon > 0)[None, :, None], r, torch.ones_like(r))


class LineOpacity:
    """kappa_lines(gas, continuum) -> (..., L, N) cm^2 g^-1 on `grid` for the lines of `lines`."""

    def __init__(self, lines: LineList, grid: LogWavelengthGrid, options: LineOpacityOptions = LineOpacityOptions()):
        self.lines, self.grid, self.opt = lines, grid, options

    # ------------------------------------------------------------------ common per-line quantities

    def _doppler(self, gas: GasState, mass: torch.Tensor) -> torch.Tensor:
        """Delta v_D / c, (M, n)."""
        c = self.opt.const
        xi2 = gas.vturb ** 2 + self.opt.microturbulence ** 2
        return torch.sqrt(2.0 * c.k * gas.T[:, None] / (mass[None, :] * c.amu) + xi2[:, None]) / C_CM

    def _mass(self, ll: LineList) -> torch.Tensor:
        if self.opt.doppler_mass == "line":
            return ll.mass
        z = ll.species.floor().long().clamp(1, 99)
        el = atomic_masses(ll.mass.device)[z - 1]
        codes, mm = (torch.as_tensor(a, dtype=ll.species.dtype, device=ll.species.device) for a in _molecule_masses())
        hit = (ll.species[:, None] - codes[None, :]).abs() < 1e-3
        mol = (mm[None, :] * hit).sum(1)
        mass = torch.where(ll.is_molecule, mol, el)
        mass = torch.where((ll.kind == int(LineKind.HYDROGEN)) & (ll.mass > 1.5), mass * 2.0, mass)    # D
        return torch.where((ll.kind == int(LineKind.HELIUM_I)) & (ll.mass > 2.5) & (ll.mass < 3.5),
                           mass * 0.75, mass)                                                          # 3He

    def _strength(self, gas: GasState, ll: LineList) -> torch.Tensor:
        """(pi e^2/mc) gf (n/U) exp(-E hc/kT) / rho, cm^2 s^-1 g^-1 (M, n)."""
        c = self.opt.const
        gf = 10.0 ** (ll.log_gf + ll.log_isotope)
        boltz = torch.exp(-ll.e_lower[None, :] * c.hc_k / gas.T[:, None])
        return pi_e2_over_mc(c) * gf[None, :] * gas.n_over_u(ll.species) * boltz / gas.rho[:, None]

    def _continuum_at(self, cont: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        k = torch.round(pos).long().clamp(0, self.grid.length - 1)
        return cont[:, k]

    def _positions(self, ll: LineList) -> torch.Tensor:
        pos = self.grid.position(ll.wavelength)
        return torch.round(pos) if self.opt.line_center == "grid" else pos

    # ------------------------------------------------------------------ Voigt lines

    def _perturbers(self, gas: GasState, ll: LineList) -> torch.Tensor:
        """n_pert for van der Waals broadening, (M, n): K&A 1981 p. 4 (vdw_alpha = 0) or the
        Anstee-Barklem-O'Mara velocity scaling (synthe.for, alpha != 0)."""
        T = gas.T[:, None]
        base = (gas.n_h1 + 0.42 * gas.n_he1 + 0.85 * gas.n_h2)[:, None] * (T / 1e4) ** 0.3
        al = ll.vdw_alpha[None, :]
        if not bool((al != 0).any()):
            return base.expand(-1, len(ll))
        e = (1.0 - al) / 2.0
        m = ll.mass[None, :]
        abo = (gas.n_h1[:, None] * (T / 1e4) ** e
               + gas.n_he1[:, None] * 0.628 * (2.0991e-4 * T * (0.25 + 1.008 / m)) ** e
               + gas.n_h2[:, None] * 1.08 * (2.0991e-4 * T * (0.5 + 1.008 / m)) ** e)
        return torch.where(al != 0, abo, base)

    def _voigt_lines(self, gas: GasState, cont: torch.Tensor, ll: LineList, which: torch.Tensor, out: torch.Tensor,
                     he1: bool = False) -> None:
        """Voigt-profile lines ll[which] added to out (M, N), in chunks of lines."""
        if not len(which):
            return
        opt, g = self.opt, self.grid
        M, N = cont.shape
        step = g.step
        chunk = max(1, opt.budget // max(M, 1))
        for c0 in range(0, len(which), chunk):
            sub = ll.select(which[c0:c0 + chunk])
            pos = self._positions(sub)
            dop = self._doppler(gas, self._mass(sub))
            nu0 = C_NM / sub.wavelength
            k0 = self._strength(gas, sub) / (SQRT_PI * nu0[None, :] * dop)
            ne = gas.n_e[:, None]
            gam = 10.0 ** sub.log_gamma_rad[None, :] + 10.0 ** sub.log_gamma_stark[None, :] * ne
            if not he1:
                gam = gam + 10.0 ** sub.log_gamma_vdw[None, :] * self._perturbers(gas, sub)
            a = gam / (4.0 * math.pi * nu0[None, :] * dop)
            kc = self._continuum_at(cont, pos) * opt.cutoff
            ratio = k0 / kc
            active = ratio >= 1.0
            keep = active.any(0)
            if not bool(keep.any()):
                continue
            idx = keep.nonzero().squeeze(1)
            r = ratio[:, idx].clamp(min=1.0)
            v_core = torch.sqrt(torch.log(r))
            v_wing = torch.sqrt(a[:, idx] * r / SQRT_PI)
            vmax = torch.where(active[:, idx], torch.maximum(v_core, v_wing) * 1.5 + 2.0, torch.zeros_like(r))
            half = torch.ceil((vmax * dop[:, idx]).amax(0) / step)
            if opt.max_wing_points is not None:
                half = half.clamp(max=opt.max_wing_points)
            cpos = torch.round(pos[idx])
            lo = (cpos - half).clamp(min=0)
            hi = (cpos + half).clamp(max=N - 1)
            ok = hi >= lo
            idx, lo, hi = idx[ok], lo[ok].long(), hi[ok].long()
            # lines wider than one block are split into segments of `seg` points
            seg = max(64, opt.budget // M)
            nseg = (hi - lo) // seg + 1
            if bool((nseg > 1).any()):
                rep = torch.repeat_interleave(torch.arange(len(idx), device=idx.device), nseg)
                first = torch.cumsum(nseg, 0) - nseg
                part = torch.arange(len(rep), device=idx.device) - first[rep]
                idx, lo, hi = idx[rep], lo[rep] + part * seg, torch.minimum(lo[rep] + part * seg + seg - 1, hi[rep])
            self._deposit(gas, cont, sub, idx, lo, hi, pos, dop, k0, a, active, out)

    def _deposit(self, gas, cont, sub, idx, lo, hi, pos, dop, k0, a, active, out):
        """Profiles of lines idx over grid points lo..hi, bucketed by width so that each block holds
        <= budget elements; scatter-added into out (M, N)."""
        opt = self.opt
        M, N = cont.shape
        width = hi - lo + 1
        order = torch.argsort(width)
        idx, lo, hi, width = idx[order], lo[order], hi[order], width[order]
        series = (sub.series_limit[idx] > 0).any()
        i = 0
        n = len(idx)
        while i < n:
            w = int(width[min(n - 1, i)])
            # lines whose width is within 2x of the first one share a block padded to the block maximum
            j = int(torch.searchsorted(width, torch.tensor(2 * w, device=width.device), right=True))
            j = max(j, i + 1)
            wmax = int(width[j - 1])
            per = max(1, opt.budget // (M * wmax))
            j = min(j, i + per)
            wmax = int(width[j - 1])
            b = slice(i, j)
            li = idx[b]
            k = lo[b, None] + torch.arange(wmax, device=lo.device)[None, :]
            valid = k <= hi[b, None]
            k = torch.where(valid, k, lo[b, None])
            x = torch.exp(self.grid.step * (pos[li][:, None] - k))           # nu_k / nu_0
            v = (x - 1.0).abs()[None, :, :] / dop[:, li][:, :, None]
            h = (voigt_kurucz if opt.voigt == "kurucz" else voigt)(a[:, li][:, :, None], v)
            kap = k0[:, li][:, :, None] * h
            if series:
                wl = C_NM / (C_NM / sub.wavelength[li][None, :, None] * x[None])
                kap = kap * series_ramp(wl, sub.species[li], sub.series_limit[li], gas.n_e)
            if opt.cutoff_rule == "kurucz":
                cen = torch.round(pos[li]).long()[:, None]
                ck = self._continuum_at(cont, pos[li])[:, :, None]
                above = (kap >= opt.cutoff * ck) | (k == cen)[None]
                inner_r = torch.cat([torch.zeros_like(above[..., :1]), above[..., :-1]], -1) & (k > cen)[None]
                inner_b = torch.cat([above[..., 1:], torch.zeros_like(above[..., :1])], -1) & (k < cen)[None]
                above = above | inner_r | inner_b
            else:
                above = kap >= opt.cutoff * cont[:, k]
            m = valid[None] & active[:, li][:, :, None] & above
            flat = (torch.arange(M, device=k.device)[:, None, None] * N + k[None]).expand_as(kap)
            out.view(-1).index_add_(0, flat[m], kap[m])
            i = j

    # ------------------------------------------------------------------ autoionizing lines

    def _fano_lines(self, gas: GasState, cont: torch.Tensor, ll: LineList, out: torch.Tensor) -> None:
        """kappa = gf (n/U)/rho exp(-E/kT) (A eps + B)/(eps^2 + 1), eps = 2 (nu - nu0)/Gamma (Shore 1967 form;
        XLINOP label 700; A = GAMMAS, B = GAMMAW fields, Gamma = GAMMAR field, s^-1)."""
        if not len(ll):
            return
        c = self.opt.const
        M, N = cont.shape
        gf = 10.0 ** (ll.log_gf + ll.log_isotope)
        boltz = torch.exp(-ll.e_lower[None, :] * c.hc_k / gas.T[:, None])
        base = gf[None, :] * gas.n_over_u(ll.species) * boltz / gas.rho[:, None]
        A = torch.where(ll.log_gamma_stark > 0, -(10.0 ** -ll.log_gamma_stark), 10.0 ** ll.log_gamma_stark)
        B = 10.0 ** ll.log_gamma_vdw
        G = 10.0 ** ll.log_gamma_rad
        nu = C_NM / self.grid.wavelengths(cont.device)
        nu0 = C_NM / ll.wavelength
        for i in range(len(ll)):
            eps = 2.0 * (nu - nu0[i]) / G[i]
            kap = base[:, i:i + 1] * B[i] * ((A[i] * eps + B[i]) / (eps ** 2 + 1.0) / B[i])[None, :]
            center = self._continuum_at(cont, self.grid.position(ll.wavelength[i:i + 1]))
            on = base[:, i:i + 1] * B[i] >= self.opt.cutoff * center
            out += torch.where(on & (kap >= self.opt.cutoff * cont), kap, torch.zeros_like(kap))

    # ------------------------------------------------------------------ public

    def __call__(self, gas: GasState, continuum, kinds: tuple[LineKind, ...] | None = None,
                 stimulated: bool = True) -> torch.Tensor:
        """Line absorption coefficient per gram, (*gas.batch_shape, N) (layers flattened in gas -> restored).
        continuum: kappa_abs + sigma per gram on the grid (tensor of the same leading shape, or an Opacity).
        kinds: restrict to these line kinds (default all). stimulated: include the factor 1 - exp(-h nu/kT)."""
        from .hydrogen import hydrogen_lines
        if isinstance(continuum, Opacity):
            continuum = continuum.total
        M = gas.T.shape[0]
        cont = continuum.reshape(M, self.grid.length)
        ll = self.lines.to(cont.device)
        want = set(LineKind) if kinds is None else set(kinds)
        out = torch.zeros(M, self.grid.length, dtype=cont.dtype, device=cont.device)
        def pick(*kinds):
            m = torch.zeros_like(ll.kind, dtype=torch.bool)
            for k in kinds:
                if k in want:
                    m |= ll.kind == int(k)
            return m.nonzero().squeeze(1)

        voigt_kinds = [LineKind.ATOMIC, LineKind.MOLECULAR]
        if self.opt.helium_ii == "voigt":
            voigt_kinds.append(LineKind.HELIUM_II)
        self._voigt_lines(gas, cont, ll, pick(*voigt_kinds), out)
        self._voigt_lines(gas, cont, ll, pick(LineKind.HELIUM_I), out, he1=True)
        self._fano_lines(gas, cont, ll.select(pick(LineKind.AUTOIONIZING)), out)
        if LineKind.HYDROGEN in want:
            out = out + hydrogen_lines(self, gas, cont, ll.of_kind(LineKind.HYDROGEN))
        if stimulated:
            hnu_kt = self.opt.const.hc_k * 1e7 / (self.grid.wavelengths(cont.device)[None, :] * gas.T[:, None])
            out = out * -torch.expm1(-hnu_kt)
        return out.reshape(tuple(gas.batch_shape) + (self.grid.length,)) if gas.batch_shape else out

    def opacity(self, gas: GasState, continuum) -> Opacity:
        """The line absorption as an Opacity (no scattering, LTE source), absorbers per line class (CLASSES)."""
        parts = {name: self(gas, continuum, kinds) for name, kinds in CLASSES.items()}
        a = sum(parts.values())
        return Opacity(a, torch.zeros_like(a), None, parts, {})


CLASSES = {"atomic lines": (LineKind.ATOMIC, LineKind.HELIUM_II, LineKind.AUTOIONIZING),
           "molecular lines": (LineKind.MOLECULAR,), "H I lines": (LineKind.HYDROGEN,), "He I lines": (LineKind.HELIUM_I,)}


@lru_cache(maxsize=None)
def _molecule_masses() -> tuple[np.ndarray, np.ndarray]:
    from ...eos.molecules import MoleculeSet
    ms = MoleculeSet.default()
    return np.asarray(ms.codes, dtype=np.float64), np.asarray(ms.mass, dtype=np.float64)


def interpolate_edge_table(wl_edge: np.ndarray, log_kappa: torch.Tensor, grid: LogWavelengthGrid) -> torch.Tensor:
    """kappa on the grid from log10 kappa tabulated at each continuum edge, the interval midpoint and the next
    edge (3 values per interval, xnfpelsyn.dat): parabola in wavelength through the three log values, as SYNTHE
    interpolates its continuum (synthe.for loop 2005). log_kappa (..., 3 (nedge-1)) -> (..., N)."""
    wl = grid.wavelengths(log_kappa.device)
    e = torch.as_tensor(np.abs(wl_edge), dtype=wl.dtype, device=wl.device)
    i = (torch.searchsorted(e, wl, right=True) - 1).clamp(0, len(e) - 2)
    w0, w2 = e[i], e[i + 1]
    w1 = 0.5 * (w0 + w2)
    t = log_kappa.reshape(log_kappa.shape[:-1] + (len(e) - 1, 3))
    y0, y1, y2 = t[..., i, 0], t[..., i, 1], t[..., i, 2]
    l0 = (wl - w1) * (wl - w2) / ((w0 - w1) * (w0 - w2))
    l1 = (wl - w0) * (wl - w2) / ((w1 - w0) * (w1 - w2))
    l2 = (wl - w0) * (wl - w1) / ((w2 - w0) * (w2 - w1))
    return 10.0 ** (y0 * l0 + y1 * l1 + y2 * l2)
