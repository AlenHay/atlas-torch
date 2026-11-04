"""Spectrum synthesis: Atmosphere + Composition -> Populations -> continuum + line opacity -> transfer -> Spectrum.

    grid = LogWavelengthGrid.window(500.0, 510.0, 600000.0)
    spec = synthesize(atm, comp, grid)             # v2 physics in every module (SynthesisConfig())
    spec = synthesize(atm, comp, grid, kurucz=True)   # the stock Fortran's choices in every module
    spec.write("spectrum.bin")                     # BasicATLAS load_binary_spectrum / io.formats.read_spectrum_bin

Opacity providers (atlas_torch.v2.opacity conventions: cm^2 g^-1, shape (..., L, F)):
    continuum(pops, nu) -> Opacity                 absorption (stimulated emission included), scattering, source
    lines(atm, pops, continuum) -> (..., L, F)     line absorption; source function B_nu (LTE)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import torch

from .atmosphere import Atmosphere
from .composition import Composition
from .constants import CODATA2018, KURUCZ, PhysicalConstants
from .eos import ChemicalEquilibrium, Populations
from .io.spectrum_bin import write_spectrum_bin
from .opacity import Opacity, planck
from .opacity.lines.grid import LogWavelengthGrid
from .rt import Transfer

ContinuumOpacity = Callable[[Populations, torch.Tensor], Opacity]
LineOpacity = Callable[[Atmosphere, Populations, Opacity], torch.Tensor]


@dataclass(frozen=True)
class Spectrum:
    """Emergent spectrum on `grid`: flux and continuum (..., F) are Eddington fluxes H_nu, or (..., F, n_mu)
    intensities I_nu(0, mu) when mu is set; erg s^-1 cm^-2 Hz^-1 sr^-1."""
    grid: LogWavelengthGrid
    flux: torch.Tensor
    continuum: torch.Tensor
    mu: tuple | None = None
    teff: torch.Tensor | float | None = None
    logg: torch.Tensor | float | None = None
    title: str = "atlas_torch v2"
    vacuum: bool = True

    @property
    def wavelength(self) -> torch.Tensor:
        """nm."""
        return self.grid.wavelengths(device=self.flux.device)

    @property
    def normalized(self) -> torch.Tensor:
        return self.flux / self.continuum

    def f_lambda(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Surface flux F_lambda = 4 pi H_nu c / lambda^2, erg s^-1 cm^-2 cm^-1 (flux mode)."""
        conv = 4.0 * math.pi * CODATA2018.c / (self.wavelength * 1e-7) ** 2
        return self.flux * conv, self.continuum * conv

    def write(self, path: str | Path, edges_nm=None) -> None:
        """BasicATLAS spectrum.bin of one model."""
        if self.flux.dim() != (1 if self.mu is None else 2):
            raise ValueError("write() takes the spectrum of one model")
        f = lambda v: float(v) if v is not None else 0.0
        write_spectrum_bin(path, teff=f(self.teff), logg=f(self.logg), title=self.title, wbegin=self.grid.wl_begin,
                           resolution=self.grid.resolution, line=self.flux.detach().cpu().numpy(),
                           continuum=self.continuum.detach().cpu().numpy(), mu=self.mu, vacuum=self.vacuum,
                           edges_nm=edges_nm)


@dataclass(frozen=True)
class SynthesisConfig:
    """Physics choices of every module. SynthesisConfig() = v2 physics; SynthesisConfig.kurucz() = the stock
    Fortran's choice in each module (XNFPELSYN/NMOLEC, atlas7v.for KAPP, SYNTHE, SPECTRV), to attribute
    differences.

    const             constant set (EOS, opacities, Planck function)
    eos               ChemicalEquilibrium keywords: network, debye, lagged_lowering
    continuum         KuruczCompat switches of ContinuousOpacity
    continuum_sampling  "exact": at every wavelength; "edges": at XNFPELSYN's three frequencies per continua.dat
                      interval, quadratic in log10 kappa between them (xnfpelsyn.for loop 18, spectrv.for 2004-2006)
    lines             LineOpacityOptions keywords (KURUCZ_OPTIONS)
    selection         LineSelection rule ("wide" | "kurucz": the SYNTHE readers' wavelength ranges)
    kurucz_isotopes   rmolecasc.for's isotopic fractions of the molecular lists
    transfer          Transfer keywords
    microturbulence   cm/s, added in quadrature to the model's (SYNTHE TURBV; BasicATLAS 1.5 km/s)"""
    const: PhysicalConstants = CODATA2018
    eos: dict = field(default_factory=dict)
    continuum: object = None
    continuum_sampling: str = "exact"
    lines: dict = field(default_factory=dict)
    selection: str = "wide"
    kurucz_isotopes: bool = False
    transfer: dict = field(default_factory=dict)
    microturbulence: float = 1.5e5

    @classmethod
    def kurucz(cls, **changes) -> "SynthesisConfig":
        from .opacity.continuum import KuruczCompat
        from .opacity.lines import KURUCZ_OPTIONS
        base = cls(const=KURUCZ, eos=dict(network="kurucz", debye="kurucz", lagged_lowering=True),
                   continuum=KuruczCompat.all(), continuum_sampling="edges", lines=dict(KURUCZ_OPTIONS),
                   selection="kurucz", kurucz_isotopes=True, transfer=dict(opaque_top="diffusion", tau_scheme="atlas"))
        return replace(base, **changes)

    def equation_of_state(self) -> ChemicalEquilibrium:
        return ChemicalEquilibrium(const=self.const, **self.eos)

    def transfer_solver(self) -> Transfer:
        return Transfer(**self.transfer)

    def opacity(self, grid: LogWavelengthGrid, line_list=None) -> "V2Opacity":
        from .opacity.continuum import ContinuousOpacity, EdgeSampled, KuruczCompat
        from .opacity.lines import LineOpacityOptions, LineSelection
        cont = ContinuousOpacity(const=self.const, compat=self.continuum or KuruczCompat())
        if self.continuum_sampling == "edges":
            cont = EdgeSampled(cont)
        elif self.continuum_sampling != "exact":
            raise ValueError("continuum_sampling: 'exact' | 'edges'")
        opt = LineOpacityOptions(microturbulence=self.microturbulence, const=self.const, **self.lines)
        return V2Opacity(grid, continuum=cont, line_options=opt, selection=LineSelection(self.selection),
                         line_list=line_list, kurucz_isotopes=self.kurucz_isotopes)

    def describe(self) -> dict:
        """JSON-able summary (scoreboard records)."""
        c = self.continuum
        return {"const": self.const.name, "eos": dict(self.eos),
                "continuum": {} if c is None else {k: v for k, v in vars(c).items() if v},
                "continuum_sampling": self.continuum_sampling, "lines": dict(self.lines),
                "selection": self.selection, "kurucz_isotopes": self.kurucz_isotopes, "transfer": dict(self.transfer),
                "microturbulence": self.microturbulence}


def synthesize(atm: Atmosphere, comp: Composition, grid: LogWavelengthGrid, *,
               continuum: ContinuumOpacity | None = None, lines: LineOpacity | None = None,
               transfer: Transfer | None = None,
               eos: ChemicalEquilibrium | None = None, populations: Populations | None = None, mu=None,
               config: SynthesisConfig | None = None, kurucz: bool = False,
               const: PhysicalConstants | None = None, title: str = "atlas_torch v2", vacuum: bool = True) -> Spectrum:
    """Spectrum of `atm` (batch dims allowed) on `grid`. Every module not given explicitly is built from `config`
    (default SynthesisConfig(), or SynthesisConfig.kurucz() with kurucz=True): populations from `eos`, opacities
    from the providers (default: config.opacity(grid) for both; lines=None with an explicit continuum: no
    lines), transfer of the continuum alone and of continuum + lines; S_thermal is the absorption-weighted source
    function (Kurucz 1970 eq. 2.65). mu: emergent intensities at these angles instead of the flux."""
    config = config or (SynthesisConfig.kurucz() if kurucz else SynthesisConfig())
    const = const or config.const
    transfer = transfer or config.transfer_solver()
    if continuum is None:
        v2 = config.opacity(grid)
        continuum, lines = v2.continuum, lines or v2.lines
    pops = populations if populations is not None else (eos or config.equation_of_state()).solve(atm, comp)
    wl = grid.wavelengths(device=atm.T.device)
    nu = const.c / (wl * 1e-7)
    cont = continuum(pops, nu)
    line = lines(atm, pops, cont) if lines is not None else None
    B = planck(nu, atm.T, const)
    lf = lambda v: v.transpose(-1, -2)                                   # (..., L, F) -> (..., F, L)
    rhox = atm.rhox[..., None, :]
    s_cont = B if cont.source is None else cont.source
    pick = (lambda s: s.flux) if mu is None else (lambda s: s.intensity)
    c = pick(transfer.solve(rhox, lf(cont.absorption), lf(cont.scattering), lf(s_cont), mu=mu))
    if line is None:
        r = c
    else:
        ab = cont.absorption + line
        src = cont.absorption * s_cont
        src += line * B
        src /= ab
        del line, B, s_cont
        r = pick(transfer.solve(rhox, lf(ab), lf(cont.scattering), lf(src), mu=mu))
    return Spectrum(grid, r, c, None if mu is None else tuple(float(m) for m in mu), atm.teff, atm.logg,
                    title, vacuum)


def deck_title(deck) -> str:
    """The TITLE card text of an AtlasDeck (columns 7-80), as READIN stores it."""
    for card in list(deck.head) + list(deck.middle):
        if card.startswith("TITLE"):
            return card[6:80]
    return "atlas_torch v2"


class V2Opacity:
    """The v2 opacity modules as providers: ContinuousOpacity (Kurucz 1970 ch. 5) and LineOpacity of the BasicATLAS
    line lists around the grid (Kurucz & Avrett 1981), atlas_torch.v2.opacity."""

    def __init__(self, grid: LogWavelengthGrid, *, continuum=None, line_options=None, selection=None, line_list=None,
                 kurucz_isotopes: bool = False):
        from .opacity.continuum import ContinuousOpacity
        from .opacity.lines import LineOpacity, LineOpacityOptions, LineSelection, read_basicatlas_lines
        self.continuum = continuum or ContinuousOpacity()
        ll = line_list if line_list is not None else read_basicatlas_lines(
            grid.wl_begin, grid.wl_end, selection or LineSelection(), kurucz_isotopes=kurucz_isotopes)
        self.line_opacity = LineOpacity(ll, grid, line_options or LineOpacityOptions())

    def lines(self, atm: Atmosphere, pops: Populations, continuum: Opacity) -> torch.Tensor:
        from .opacity.lines import GasState
        return self.line_opacity(GasState.from_populations(atm, pops), continuum.total)


class TableOpacity:
    """Stand-in providers from one model's SYNTHE work files: the XNFPELSYN continuum (fort.10) and the SYNTHE line
    opacity (fort.9) of the exact port / stock Fortran. Frequencies must be those of the fort.9 grid."""

    def __init__(self, workdir: str | Path):
        from .io.synthe_files import read_synthe_lines, read_xnfpelsyn_continuum
        wd = Path(workdir)
        self.continuum_table = read_xnfpelsyn_continuum(wd / "fort.10")
        self.line_table = lt = read_synthe_lines(wd / "fort.9")
        self.vacuum = lt.vacuum
        self.grid = LogWavelengthGrid.window(lt.wlbeg, lt.wlend, lt.resolution)
        if self.grid.length != lt.length:
            raise ValueError(f"fort.9 LENGTH {lt.length} != SYNBEG grid {self.grid.length}")

    def continuum(self, pops, nu: torch.Tensor) -> Opacity:
        ab, sc = self.continuum_table.at(CODATA2018.c / nu * 1e7)
        return Opacity(ab.transpose(-1, -2), sc.transpose(-1, -2))

    def lines(self, atm, pops, continuum) -> torch.Tensor:
        n = continuum.absorption.shape[-1]
        if n != self.grid.length:
            raise ValueError("lines() covers the whole fort.9 grid")
        return torch.as_tensor(self.line_table.opacity(), device=continuum.absorption.device).transpose(-1, -2)
