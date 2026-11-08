"""The gas of the ODF grid: equilibrium populations, Doppler widths and continuous opacity at the 57 x 25
(T, P_gas) nodes, the quantities XNFDF passes to DFSYNTHE (xnfdf.for; Castelli 2005, MSAIS 8, 34, p. 34)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..atmosphere import Atmosphere
from ..composition import Composition
from ..eos.equilibrium import Populations
from ..io.continua_dat import frequency_set, read_continua_dat
from ..opacity.lines.opacity import GasState, interpolate_edge_table, perturber_density
from .grid import OdfGrid
from .lines import doppler_slots, nelion_species


def dfsynthe_continua_path() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "BasicATLAS" / "data" / "dfsynthe_files" \
        / "continua.dat"


@dataclass(frozen=True)
class OdfGas:
    """One temperature row of the ODF grid, every tensor (L,) or (L, k) over the L = 25 pressures.

    n_over_u (L, 594) n/U per NELION slot (cm^-3); doppler (L, 594) Delta nu_D/nu at zero microturbulence;
    n_pert van der Waals perturbers (cm^-3); edge_wl (E,) continua.dat edges (nm) and log_cont (L, 3 (E-1)) log10
    of the continuous absorption + scattering (cm^2 g^-1) at the lower edge, middle and upper edge of each
    continua interval; state: GasState of the v2 line opacity."""
    T: torch.Tensor
    P: torch.Tensor
    rho: torch.Tensor
    n_e: torch.Tensor
    n_pert: torch.Tensor
    n_over_u: torch.Tensor
    doppler: torch.Tensor
    edge_wl: np.ndarray
    log_cont: torch.Tensor
    state: GasState

    def continuum(self, grid) -> torch.Tensor:
        """Continuous extinction (L, N) on a LogWavelengthGrid: parabola in wavelength through the three log10
        values of the continua interval (dfsynthe.for MAIN loops 2002-2005)."""
        return interpolate_edge_table(self.edge_wl, self.log_cont, grid)

    def continuum_at(self, layer: torch.Tensor, wl: torch.Tensor) -> torch.Tensor:
        """Continuous extinction at wavelengths wl (nm) of layers `layer` (both (k,)), as continuum()."""
        e = torch.as_tensor(self.edge_wl, dtype=wl.dtype, device=wl.device)
        i = (torch.searchsorted(e, wl.contiguous(), right=True) - 1).clamp(0, len(e) - 2)
        w0, w2 = e[i], e[i + 1]
        w1 = 0.5 * (w0 + w2)
        y = lambda k: self.log_cont[layer, 3 * i + k]                                  # noqa: E731
        return 10.0 ** (y(0) * (wl - w1) * (wl - w2) / ((w0 - w1) * (w0 - w2))
                        + y(1) * (wl - w0) * (wl - w2) / ((w1 - w0) * (w1 - w2))
                        + y(2) * (wl - w0) * (wl - w1) / ((w2 - w0) * (w2 - w1)))


def slot_populations(pops: Populations, molecular_ions: bool = True) -> torch.Tensor:
    """n/U (..., 594) per NELION slot from v2 Populations; molecular_ions=False leaves the molecular ions
    (CH+, NH+, OH+, MgH+, AlH+, SiH+, CaH+, CO+) empty, as stock XNFDF's output has them (its REAL*8 IDMOL DATA are
    single-precision literals such as 106.01, which never equal the codes read from molecules.dat)."""
    code, _, mol = nelion_species()
    out = torch.zeros(pops.T.shape + (594,), dtype=torch.float64, device=pops.T.device)
    nou = pops.n_over_u_atom
    for i in np.flatnonzero((code > 0) & ~mol):
        z = int(code[i])
        s = int(round((code[i] - z) * 100)) + 1
        out[..., i] = nou[..., z - 1, s - 1]
    codes = np.asarray(pops.molecules.codes, dtype=np.float64)
    for i in np.flatnonzero(mol):
        if not molecular_ions and code[i] != np.floor(code[i]):
            continue
        hit = np.flatnonzero(np.abs(codes - code[i]) < 1e-3)
        if len(hit):
            out[..., i] = pops.n_over_u_mol[..., int(hit[0])]
    return out


def odf_gas(comp: Composition, config, log_t=None, nodes: str = "exact", device=None) -> list[OdfGas]:
    """OdfGas for each temperature of the ODF grid (default all 57), on `device`. The EOS runs on (T, P) with P
    last, so a lagged lowering chains over the 25 pressures of a temperature as in XNFDF. config: OdfConfig."""
    grid = OdfGrid() if log_t is None else OdfGrid(np.asarray(log_t, dtype=np.float64))
    t, p = grid.nodes(nodes)
    T = torch.as_tensor(t, device=device)[:, None].expand(len(t), len(p)).contiguous()
    P = torch.as_tensor(p, device=device)[None, :].expand(len(t), len(p)).contiguous()
    eos = config.equation_of_state()
    pops = eos.solve(Atmosphere(rhox=torch.zeros_like(T), T=T, P=P), comp)
    edges = read_continua_dat(config.continua_path or dfsynthe_continua_path())
    fs = torch.as_tensor(frequency_set(edges), device=device)
    cont = config.continuous_opacity()
    nou = slot_populations(pops, config.molecular_ions)
    h2 = pops.molecules.index(101.0)
    out = []
    for i in range(len(t)):
        row = _row(pops, i)
        tot = cont(row, fs).total
        dop = doppler_slots(row.T, torch.zeros_like(row.T), config.const)
        npert = perturber_density(row.T, row.n_atom[..., 0, 0], row.n_atom[..., 1, 0], row.n_mol[..., h2])
        atm = Atmosphere(rhox=torch.zeros_like(row.T), T=row.T, P=row.P, vturb=torch.zeros_like(row.T))
        out.append(OdfGas(row.T, row.P, row.rho, row.n_e, npert, nou[i], dop, np.abs(edges["wavelength_nm"]),
                          torch.log10(tot), GasState.from_populations(atm, row)))
    return out


def _row(pops: Populations, i: int) -> Populations:
    from dataclasses import fields
    return Populations(**{f.name: (getattr(pops, f.name)[i] if isinstance(getattr(pops, f.name), torch.Tensor)
                                   else getattr(pops, f.name)) for f in fields(Populations)})


__all__ = ["OdfGas", "odf_gas", "slot_populations", "dfsynthe_continua_path"]
