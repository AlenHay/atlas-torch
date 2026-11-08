"""Rosseland mean opacity (Kurucz 1970 sec. 2.12, eqs. 2.84-2.86): 1/kappa_R = int (1/kappa_nu) dB_nu/dT dnu /
int dB_nu/dT dnu, kappa_nu = continuous absorption + scattering + line opacity (ATLAS9 ROSS, ABTOT; KAPPA9 MAIN).
Tables on the KAPPA9 grid (57 log T x 42 log P_gas) are interpolated as ROSSTAB does.

    ross = RosselandOpacity(OpacityDistribution.read(odf_dir, "big"), RosselandConfig.kurucz())
    table = ross.tabulate(comp)                # RosselandTable (V, 57, 42)
    table.write("kappa.ros")
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from ...tables import table
from ..atmosphere import Atmosphere
from ..composition import Composition
from ..constants import CODATA2018, KURUCZ, PhysicalConstants, decimal32
from ..eos import ChemicalEquilibrium, Populations
from . import planck_derivative
from .continuum.interpolate import map1
from .frequencies import KAPPA9_LOWER_EDGE_HZ, FrequencyQuadrature, SampleSet
from .odf import ODF_LOG_T, IntervalSet, OpacityDistribution

# KAPPA9 DATA TABP (kappa9.for; = atlas9mem.for ROSSTAB TABP), log10 P_gas
KAPPA9_LOG_P = decimal32(table("kappa9.for", "KAPPA9", "TABP"))
ROSSTAB_LOG_T = ODF_LOG_T


def _basicatlas() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "BasicATLAS"


def rosseland_mean(extinction: torch.Tensor, samples, T: torch.Tensor, *, normalization: str = "quadrature",
                   const: PhysicalConstants = CODATA2018) -> torch.Tensor:
    """kappa_R (..., L) from extinction (..., L, K) cm^2 g^-1 at the samples (SampleSet, or (nu, weight) in Hz).
    normalization "quadrature": the same quadrature of dB/dT in the denominator; "analytic": 4 sigma T^3 / pi
    (ATLAS9 ROSS, KAPPA9: 4*5.6697E-5/3.14159*T**3)."""
    nu, w = (samples.nu, samples.weight) if isinstance(samples, SampleSet) else samples
    nu, w = nu.to(extinction.device), w.to(extinction.device)
    d = planck_derivative(nu, T, const) * w
    num = (d / extinction).sum(-1)
    if normalization == "quadrature":
        den = d.sum(-1)
    elif normalization == "analytic":
        den = 4.0 * const.sigma * T ** 3 / math.pi
    else:
        raise ValueError("normalization: 'quadrature' | 'analytic'")
    return den / num


@dataclass(frozen=True, eq=False)
class RosselandTable:
    """log10 kappa_R (V, 57, 42) on log_t (57,) x log_p (42,) (P_gas) per microturbulence vturb_kms."""
    log_t: np.ndarray
    log_p: np.ndarray
    vturb_kms: tuple
    log_kappa: torch.Tensor
    title: str = " ROSSELAND MASS ABSORPTION COEFFICIENTS"

    @classmethod
    def read(cls, path) -> "RosselandTable":
        from ..io.kappa_ros import read_kappa_ros
        d = read_kappa_ros(path)
        return cls(d["log_t"], d["log_p"], (0, 1, 2, 4, 8), torch.as_tensor(d["log_kappa"]), d["title"])

    def write(self, path) -> None:
        from ..io.kappa_ros import write_kappa_ros
        write_kappa_ros(path, self.log_t, self.log_p, self.log_kappa.detach().cpu().numpy(), title=self.title)

    @classmethod
    def read_kapk(cls, directory, vturbs=(0, 1, 2, 4, 8)) -> "RosselandTable":
        from ..io.kappa_ros import read_kapk
        lk = np.stack([read_kapk(Path(directory) / f"kapk{v}.dat").T / 1000.0 for v in vturbs])
        return cls(ROSSTAB_LOG_T.copy(), KAPPA9_LOG_P.copy(), tuple(vturbs), torch.as_tensor(lk))

    def write_kapk(self, directory, rounding: str = "nearest") -> None:
        from ..io.kappa_ros import write_kapk
        Path(directory).mkdir(parents=True, exist_ok=True)
        for j, v in enumerate(self.vturb_kms):
            write_kapk(Path(directory) / f"kapk{int(v)}.dat", self.log_t, self.log_p,
                       self.log_kappa[j].detach().cpu().numpy(), rounding)

    def kappa(self, T, P, vturb) -> torch.Tensor:
        """kappa_R, cm^2 g^-1, at T (K), P_gas (dyn cm^-2), vturb (cm/s) (broadcast): MAP1 in log10 T per
        (vturb, P) row, then in vturb, then in log10 P clamped below at the first node (ROSSTAB)."""
        dev = self.log_kappa.device
        T = torch.as_tensor(T, dtype=torch.float64, device=dev)
        P = torch.as_tensor(P, dtype=torch.float64, device=dev)
        v = torch.as_tensor(vturb, dtype=torch.float64, device=dev)
        shape = torch.broadcast_shapes(T.shape, P.shape, v.shape)
        T, P, v = T.expand(shape), P.expand(shape), v.expand(shape)
        lt = torch.as_tensor(self.log_t, dtype=torch.float64, device=dev)
        lp = torch.as_tensor(self.log_p, dtype=torch.float64, device=dev)
        tv = torch.as_tensor([float(x) * 1e5 for x in self.vturb_kms], dtype=torch.float64, device=dev)
        rows = self.log_kappa.permute(0, 2, 1)                                   # (V, 42, 57)
        nv, npr = rows.shape[:2]
        x = torch.log10(T)[..., None, None].expand(shape + (nv, npr))
        pk_v = map1(x[..., None], lt, rows)[..., 0]                               # (..., V, 42)
        pk = map1(v[..., None, None].expand(shape + (npr, 1)), tv, pk_v.transpose(-1, -2))[..., 0]   # (..., 42)
        plog = torch.maximum(torch.log10(P), lp[0])
        return torch.pow(10.0, map1(plog[..., None], lp, pk)[..., 0])


@dataclass(frozen=True)
class RosselandConfig:
    """Physics choices of RosselandOpacity. RosselandConfig() = v2 physics; .kurucz() = KAPPA9's.

    const          constant set (EOS, continuum, Planck function, sigma)
    eos            ChemicalEquilibrium keywords
    molecules_path molecules.dat of the EOS (None: the v2 default records)
    continuum      KuruczCompat switches of ContinuousOpacity (None: v2 physics)
    processes      ContinuousOpacity processes (None: all)
    quadrature     "atlas_big": ATLAS9's 337 frequencies; "intervals": one node per ODF interval
    lower_edge_nm  short-wavelength edge of the first atlas_big weight
    normalization  "quadrature" | "analytic" (rosseland_mean)
    uv_to_nm       continuum-only extension of the "intervals" quadrature shortward of the ODF
    kapk_rounding  "nearest" | "truncate" (KAPPA9 INT) when writing kapk files"""
    const: PhysicalConstants = CODATA2018
    eos: dict = field(default_factory=dict)
    molecules_path: str | None = None
    continuum: object = None
    processes: tuple | None = None
    quadrature: str = "atlas_big"
    lower_edge_nm: float = 8.97666
    normalization: str = "quadrature"
    uv_to_nm: float | None = None
    kapk_rounding: str = "nearest"

    @classmethod
    def kurucz(cls, **changes) -> "RosselandConfig":
        """KAPPA9 (kappa9.for as BasicATLAS runs it): Kurucz constants; NMOLEC's network and lagged Debye
        lowering; dfsynthe_files/molecules.dat; ATLAS9's continuum (KuruczCompat.atlas9(), else .all()); the
        BIG frequencies with the first weight from FRESET(89) = 3.364687e16 Hz; 4 sigma T^3/pi; INT() rounding."""
        from .continuum import KuruczCompat
        compat = KuruczCompat.atlas9() if hasattr(KuruczCompat, "atlas9") else KuruczCompat.all()
        base = cls(const=KURUCZ, eos=dict(network="kurucz", debye="kurucz", lagged_lowering=True),
                   molecules_path=str(_basicatlas() / "data" / "dfsynthe_files" / "molecules.dat"), continuum=compat,
                   quadrature="atlas_big", lower_edge_nm=KURUCZ.c * 1e7 / KAPPA9_LOWER_EDGE_HZ,
                   normalization="analytic", kapk_rounding="truncate")
        return replace(base, **changes)

    def describe(self) -> dict:
        c = self.continuum
        return {"const": self.const.name, "eos": dict(self.eos), "molecules_path": self.molecules_path,
                "continuum": {} if c is None else {k: v for k, v in vars(c).items() if v},
                "processes": None if self.processes is None else list(self.processes), "quadrature": self.quadrature,
                "lower_edge_nm": self.lower_edge_nm, "normalization": self.normalization, "uv_to_nm": self.uv_to_nm,
                "kapk_rounding": self.kapk_rounding}

    def equation_of_state(self) -> ChemicalEquilibrium:
        from ..io.molecules_dat import read_molecules_dat
        rec = read_molecules_dat(self.molecules_path) if self.molecules_path else None
        return ChemicalEquilibrium(rec, const=self.const, **self.eos)

    def continuous_opacity(self):
        from .continuum import ContinuousOpacity, KuruczCompat
        return ContinuousOpacity(self.processes, const=self.const, compat=self.continuum or KuruczCompat())

    def frequencies(self, intervals: IntervalSet) -> FrequencyQuadrature:
        if self.quadrature == "atlas_big":
            return FrequencyQuadrature.atlas_big(intervals, self.lower_edge_nm, self.const)
        if self.quadrature == "intervals":
            return FrequencyQuadrature.intervals(intervals, self.uv_to_nm, const=self.const)
        raise ValueError("quadrature: 'atlas_big' | 'intervals'")


class RosselandOpacity:
    """kappa_R(T, P_gas) of a gas with ODF line opacity: v2 EOS, ContinuousOpacity at the quadrature frequencies
    + OpacityDistribution.sampled, then rosseland_mean."""

    def __init__(self, odf: OpacityDistribution, config: RosselandConfig = RosselandConfig()):
        self.odf, self.config = odf, config
        self.quadrature = config.frequencies(odf.intervals).to(odf.log_kappa.device)
        self.samples = self.quadrature.samples(odf.fractions)
        self.eos = config.equation_of_state()
        self.continuum = config.continuous_opacity()

    def populations(self, T, P, comp: Composition) -> Populations:
        T = torch.as_tensor(T, dtype=torch.float64)
        P = torch.as_tensor(P, dtype=torch.float64).expand(T.shape)
        return self.eos.solve(Atmosphere(rhox=torch.zeros_like(T), T=T, P=P), comp)

    def __call__(self, T, P, comp: Composition, vturb=None, table_index=None) -> torch.Tensor:
        """kappa_R (..., L) cm^2 g^-1 at T (..., L) K, P_gas (..., L); vturb (cm/s) and table_index as
        OpacityDistribution.log_kappa_at."""
        return self.from_populations(self.populations(T, P, comp), vturb, table_index)

    def from_populations(self, pops: Populations, vturb=None, table_index=None, odf=None) -> torch.Tensor:
        """kappa_R (..., L) from equilibrium populations (no second EOS solve)."""
        odf = odf or self.odf
        s = self.samples
        cont = self.continuum(pops, self.quadrature.nu.to(pops.T.device))
        ext = cont.total[..., s.frequency] + odf.sampled(pops.T, pops.P, s, vturb, table_index)
        return rosseland_mean(ext, s, pops.T, normalization=self.config.normalization, const=self.config.const)

    def log_derivatives(self, T, P, comp: Composition, vturb=None, table_index=None):
        """(dln kappa_R/dln T, dln kappa_R/dln P) at constant P, T, (..., L), by forward-mode differentiation
        through the EOS (implicit function theorem). Not defined for the lagged lowering (kappa_R non-local)."""
        if self.config.eos.get("lagged_lowering"):
            raise ValueError("log_derivatives: the lagged Debye lowering makes kappa_R depend on other layers")
        lt = torch.log(torch.as_tensor(T, dtype=torch.float64))
        lp = torch.log(torch.as_tensor(P, dtype=torch.float64)).expand(lt.shape)
        f = lambda a, b: torch.log(self(torch.exp(a), torch.exp(b), comp, vturb, table_index))  # noqa: E731
        _, dt = torch.func.jvp(f, (lt, lp), (torch.ones_like(lt), torch.zeros_like(lp)))
        _, dp = torch.func.jvp(f, (lt, lp), (torch.zeros_like(lt), torch.ones_like(lp)))
        return dt, dp

    def tabulate(self, comp: Composition, log_t=ODF_LOG_T, log_p=KAPPA9_LOG_P, vturbs=None, rows: int = 6,
                 title: str = " ROSSELAND MASS ABSORPTION COEFFICIENTS") -> RosselandTable:
        """RosselandTable on (log_t x log_p) for the ODF's vturbs (default all). The EOS runs on (P, T) with T
        last, so a lagged lowering chains along T as in KAPPA9's columns; populations and continuum are shared
        by all vturbs. rows: pressures per evaluation block (memory)."""
        vturbs = tuple(self.odf.vturb_kms if vturbs is None else vturbs)
        dev = self.odf.log_kappa.device
        lt = torch.as_tensor(np.asarray(log_t, dtype=np.float64), device=dev)
        lp = torch.as_tensor(np.asarray(log_p, dtype=np.float64), device=dev)
        odfs = [self.odf.select_vturb(v) for v in vturbs]
        s = self.samples
        out = torch.empty(len(vturbs), len(lt), len(lp), dtype=torch.float64, device=dev)
        for r0 in range(0, len(lp), rows):
            p = lp[r0:r0 + rows]
            T = torch.pow(10.0, lt)[None, :].expand(len(p), len(lt)).contiguous()
            P = torch.pow(10.0, p)[:, None].expand(len(p), len(lt)).contiguous()
            pops = self.populations(T, P, comp)
            cont = self.continuum(pops, self.quadrature.nu.to(dev)).total[..., s.frequency]
            for j, odf in enumerate(odfs):
                kr = rosseland_mean(cont + odf.sampled(pops.T, pops.P, s), s, pops.T,
                                    normalization=self.config.normalization, const=self.config.const)
                out[j, :, r0:r0 + rows] = torch.log10(kr).T
            del pops, cont
        return RosselandTable(np.asarray(log_t, dtype=np.float64), np.asarray(log_p, dtype=np.float64), vturbs,
                              out, title)


__all__ = ["rosseland_mean", "map1", "KAPPA9_LOG_P", "RosselandConfig", "RosselandOpacity", "RosselandTable"]
