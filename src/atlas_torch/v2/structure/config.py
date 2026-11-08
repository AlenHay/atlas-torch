"""Physics choices of the structure calculation (StructureConfig) and the parameters of one model (ModelSpec).

    cfg = StructureConfig()             # v2 physics
    cfg = StructureConfig.kurucz()      # atlas9mem.for's choice in every module, to attribute differences
    spec = ModelSpec.from_control("atlas_control.com", odf_dir)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from ..composition import Composition
from ..constants import CODATA2018, KURUCZ, PhysicalConstants
from .grid import RosselandGrid


def _basicatlas() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "BasicATLAS"


@dataclass(frozen=True)
class StructureConfig:
    """const            constant set (EOS, opacities, Planck function, sigma)
    eos              ChemicalEquilibrium keywords
    molecules_path   molecules.dat of the EOS (None: the v2 default records)
    continuum        KuruczCompat of ContinuousOpacity (None: v2 physics)
    processes        ContinuousOpacity processes (None: all)
    frequencies      "atlas_big" (ATLAS9's 337 frequencies) | "intervals" (one node per ODF interval)
    rosseland        "table": kappa.ros for the start-model hydrostatics, the bubble opacity ratio and the
                     hydrostatic update (TTAUP, ROSSTAB, TCORR); "model": the model's own kappa_R(T, P)
    rosseland_norm   "quadrature" | "analytic" (4 sigma T^3/pi, ATLAS9 ROSS)
    transfer         Transfer keywords (the depth-dependent moments are always computed)
    depth_scheme     integrals and derivatives on the depth grid ("parabolic" | "atlas": INTEG/PARCOE/DERIV):
                     tau_Ross, height, P_rad, nabla, dln(kappa)/dm of the moments, the hydrostatic integration in
                     ln tau, the temperature correction
    radiation_pressure_fudge  g_rad and P_rad(0) divided by H_rad/H where H_rad > H (RADIAP(3))
    thermodynamics   "exact": derivatives by forward-mode AD through the EOS; "kurucz": +-0.1 % central
                     differences (CONVEC)
    convection       kurucz: CONVEC's MixingLength options; bubble_average: MixingLength.bubble_average;
                     radiative_loss: MixingLength.loss ("atlas9" default | "kurucz1970"); smooth_flux (default
                     True): TemperatureCorrection.smooth_flux
    correction       "v2" | "kurucz" (TemperatureCorrection.kurucz: TCORR's limits and cuts)
    odf_truncation   stop the ODF steps of a frequency once the line flux equals the continuum flux
                     (ATLAS9 main program labels 24-79)
    lambda_diagonal  "transfer": diag(Lambda) of the transfer solution; "kurucz": TCORR's E3 estimate (eq. 7.22)
    accelerate       the sign-history factors x1.25 / x0.5 of the correction in radiative layers above the lower
                     third (Kurucz 1970 sec. 7.5; TCORR loop 61): a convergence aid that damps the oscillation of the
                     Lambda correction in cool giants' upper layers; the fixed point is unchanged
    t_min            lower limit of T, K, or "odf" (the ODF table's lowest temperature, 1995 K; default); None: none
    monotonic_T      T increases by at least 1 K per layer inward (TCORR loop 6050)
    stop             "converged" (dT, flux error and derivative, every iteration) | "gold_every_15" (BasicATLAS)
    turbulent_pressure  P_turb = rho vturb^2 / 2 from the model's vturb (eq. 4.6; ATLAS9 TURBULENCE ON)"""
    const: PhysicalConstants = CODATA2018
    eos: dict = field(default_factory=dict)
    molecules_path: str | None = None
    continuum: object = None
    processes: tuple | None = None
    frequencies: str = "atlas_big"
    rosseland: str = "model"
    rosseland_norm: str = "quadrature"
    transfer: dict = field(default_factory=lambda: dict(method="ali", tol=1e-9))
    depth_scheme: str = "parabolic"
    radiation_pressure_fudge: bool = False
    thermodynamics: str = "exact"
    convection: dict = field(default_factory=dict)
    correction: str = "v2"
    odf_truncation: bool = False
    lambda_diagonal: str = "transfer"
    accelerate: bool = True
    t_min: float | str | None = "odf"
    monotonic_T: bool = False
    stop: str = "converged"
    turbulent_pressure: bool = False

    @classmethod
    def kurucz(cls, **changes) -> "StructureConfig":
        """atlas9mem.for as BasicATLAS runs it: Kurucz constants, NMOLEC's network and lagged lowering with
        atlas_files/molecules.dat, ATLAS9's continuum switches, the kappa.ros table where ATLAS9 uses ROSSTAB,
        analytic Rosseland normalisation, JOSH's transfer on the fixed XTAU depths with PARCOE optical depths and
        DERIV diffusion below tau = 20, INTEG/DERIV depth calculus, RADIAP(3)'s flux fudge, CONVEC's finite
        differences and smoothing, TCORR's corrections with their fudges, ODF truncation, the ODF table's T floor,
        monotonic T, the GOLD stop every 15 iterations."""
        from ..opacity.continuum import KuruczCompat
        base = cls(const=KURUCZ, eos=dict(network="kurucz", debye="kurucz", lagged_lowering=True),
                   molecules_path=str(_basicatlas() / "data" / "atlas_files" / "molecules.dat"),
                   continuum=KuruczCompat.atlas9(), rosseland="table", rosseland_norm="analytic",
                   transfer=dict(grid="xtau", tol=1e-9, tau_scheme="atlas", deep_scheme="atlas"),
                   depth_scheme="atlas", radiation_pressure_fudge=True, thermodynamics="kurucz", convection=dict(kurucz=True), correction="kurucz",
                   odf_truncation=True, lambda_diagonal="kurucz", t_min="odf", monotonic_T=True,
                   stop="gold_every_15")
        return replace(base, **changes)

    def describe(self) -> dict:
        """JSON-able summary (scoreboard records)."""
        c = self.continuum
        out = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in ("const", "continuum")}
        out["const"] = self.const.name
        out["continuum"] = {} if c is None else {k: v for k, v in vars(c).items() if v}
        out["processes"] = None if self.processes is None else list(self.processes)
        return out

    def equation_of_state(self, device=None, **changes):
        from ..eos import ChemicalEquilibrium
        from ..io.molecules_dat import read_molecules_dat
        rec = read_molecules_dat(self.molecules_path) if self.molecules_path else None
        return ChemicalEquilibrium(rec, const=self.const, device=device, **{**self.eos, **changes})

    def continuous_opacity(self):
        from ..opacity.continuum import ContinuousOpacity, KuruczCompat
        return ContinuousOpacity(self.processes, const=self.const, compat=self.continuum or KuruczCompat())

    def transfer_solver(self):
        from ..rt import Transfer
        return Transfer(**self.transfer)

    def rosseland_config(self):
        """The RosselandConfig of the same physics (kappa_R of the model and of the tables use one code path)."""
        from ..opacity.rosseland import RosselandConfig
        return RosselandConfig(const=self.const, eos=dict(self.eos), molecules_path=self.molecules_path,
                               continuum=self.continuum, processes=self.processes, quadrature=self.frequencies,
                               normalization=self.rosseland_norm)

    def quadrature(self, intervals):
        return self.rosseland_config().frequencies(intervals)


@dataclass(frozen=True, eq=False)
class ModelSpec:
    """One model: teff (K), logg (cgs), composition, vturb (cm/s), mixing_length (l/H_p), overshoot_weight,
    nconv, turbulence (TRBFDG, TRBPOW, TRBSND, TRBCON) or None, grid, odf (OpacityDistribution of this vturb),
    kappa_ros (RosselandTable or None), title."""
    teff: float
    logg: float
    composition: Composition
    vturb: float
    mixing_length: float
    overshoot_weight: float
    nconv: int
    grid: RosselandGrid
    odf: object
    kappa_ros: object = None
    title: str = "atlas_torch v2"
    convection: bool = True
    turbulence: tuple | None = None

    @classmethod
    def from_control(cls, control, odf_dir, *, odf=None, kappa_ros=None) -> "ModelSpec":
        """From atlas_control.com (or a ControlSpec) and the directory of p00big{v}.bdf and kappa.ros; odf /
        kappa_ros: already loaded tables to share between models of the same mixture."""
        from ..io.atlas_control import ControlSpec, read_control
        from ..opacity.odf import OpacityDistribution
        from ..opacity.rosseland import RosselandTable
        c = control if isinstance(control, ControlSpec) else read_control(control)
        v_kms = c.vturb / 1e5
        if odf is None:
            odf = OpacityDistribution.read(odf_dir, "big", vturbs=(int(round(v_kms)),))
        if kappa_ros is None and (Path(odf_dir) / "kappa.ros").exists():
            kappa_ros = RosselandTable.read(Path(odf_dir) / "kappa.ros")
        if c.composition is None:
            raise ValueError("the control cards carry no abundances")
        return cls(teff=c.teff, logg=c.logg, composition=c.composition, vturb=c.vturb, mixing_length=c.mixing_length,
                   overshoot_weight=c.overshoot_weight, nconv=c.nconv, grid=RosselandGrid(*c.tau_grid), odf=odf,
                   kappa_ros=kappa_ros, title=c.title, convection=c.convection, turbulence=c.turbulence)

    @staticmethod
    def stack(specs: list["ModelSpec"], device=None) -> "ModelBatch":
        return ModelBatch.of(specs, device)


@dataclass(frozen=True, eq=False)
class ModelBatch:
    """M models as tensors (M,) / (M, L) on one device; the distinct ODF tables stacked along their B axis with
    table_index (M,), the distinct kappa.ros tables as a tuple with kappa_index (M,)."""
    specs: tuple
    teff: torch.Tensor
    gravity: torch.Tensor
    composition: Composition
    vturb: torch.Tensor
    mixing_length: torch.Tensor
    overshoot_weight: torch.Tensor
    nconv: torch.Tensor
    convection: torch.Tensor
    log_tau: torch.Tensor
    odf: object
    table_index: torch.Tensor
    kappa_tables: tuple
    kappa_index: torch.Tensor

    @classmethod
    def of(cls, specs: list[ModelSpec], device=None) -> "ModelBatch":
        """The batch of `specs` with every tensor and table on `device` (default CPU)."""
        from ..opacity.odf import OpacityDistribution
        dev = torch.device(device) if device is not None else torch.device("cpu")
        specs = tuple(specs)
        L = {s.grid.n for s in specs}
        if len(L) != 1:
            raise ValueError("models of one batch need the same number of layers")

        def distinct(objs):
            uniq, index = [], []
            for o in objs:
                k = next((i for i, u in enumerate(uniq) if u is o), None)
                if k is None:
                    uniq.append(o)
                    k = len(uniq) - 1
                index.append(k)
            return uniq, torch.as_tensor(index, device=dev)

        odfs, t_idx = distinct([s.odf for s in specs])
        kaps, k_idx = distinct([s.kappa_ros for s in specs])
        odf = odfs[0] if len(odfs) == 1 else OpacityDistribution.stack(odfs)
        odf = odf if odf.log_kappa.device == dev else odf.to(dev)
        kaps = [k if k is None or k.log_kappa.device == dev else replace(k, log_kappa=k.log_kappa.to(dev))
                for k in kaps]
        t = lambda v, **kw: torch.as_tensor(v, device=dev, **kw)                                   # noqa: E731
        f = lambda name: t([float(getattr(s, name)) for s in specs], dtype=torch.float64)          # noqa: E731
        comp = Composition(torch.stack([s.composition.abundance for s in specs]).to(dev))
        return cls(specs, f("teff"), torch.pow(10.0, f("logg")), comp, f("vturb"), f("mixing_length"),
                   f("overshoot_weight"), t([s.nconv for s in specs]), t([s.convection for s in specs]),
                   torch.stack([s.grid.log_tau(device=dev) for s in specs]), odf, t_idx if len(odfs) > 1 else None,
                   tuple(kaps), k_idx)

    @property
    def device(self) -> torch.device:
        return self.teff.device

    def __len__(self) -> int:
        return len(self.specs)

    def subset(self, keep: torch.Tensor) -> "ModelBatch":
        """The models where keep (M,) bool is True."""
        idx = torch.nonzero(keep).flatten()
        specs = [self.specs[int(i)] for i in idx]
        return ModelBatch.of(specs, self.device)


def t_floor(config: StructureConfig, odf) -> float | None:
    """config.t_min in K; "odf": the lowest temperature of the ODF table (10**3.30 = 1995 K, TCORR's 1995.)."""
    if config.t_min == "odf":
        return float(10.0 ** np.asarray(odf.log_t)[0])
    return config.t_min
