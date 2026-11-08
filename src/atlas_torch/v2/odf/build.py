"""Opacity distribution functions of a composition: for each of the 57 temperatures, the 0 km/s line spectrum on
the DFSYNTHE grid at the 25 pressures, floored at 10^-3 of the continuum, broadened for each microturbulence and
reduced to 12-step distributions per BIG and LITTLE interval (Castelli 2005, MSAIS 8, 34; Kurucz 2005, MSAIS 8,
14 and 76; dfsynthe.for MAIN). Per-temperature results go to a work directory, so a build can be resumed."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from ..composition import Composition
from ..constants import CODATA2018, KURUCZ, PhysicalConstants
from ..opacity.odf import ODF_LOG_T, IntervalSet, OpacityDistribution
from .distribution import distribution_steps
from .gas import odf_gas
from .grid import DF_GRID, interval_index_ranges, segments
from .lines import DfLineSource, nlte_lines
from .spectrum import LineSpectrum, SpectrumOptions
from .turbulence import broaden, turbulence_kernel

VTURBS = (0, 1, 2, 4, 8)


def _xnfdf_processes() -> tuple:
    from ..opacity.continuum import PROCESSES
    return tuple(p for p in PROCESSES if p not in ("CH", "OH", "H2-H2 CIA", "H2-He CIA"))


XNFDF_PROCESSES = _xnfdf_processes()


def _ext() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


@dataclass(frozen=True)
class OdfConfig:
    """Physics choices of the ODF build. OdfConfig() = v2 physics; OdfConfig.kurucz() = XNFDF + DFSYNTHE's.

    const, eos, molecules_path, continuum, continua_path   gas and continuum (gas.odf_gas)
    processes          ContinuousOpacity processes of the gas continuum (None: all; XNFDF_PROCESSES: XNFDF's KAPP,
                       which computes ACOOL but leaves CH, OH and the H2 CIA out of ACONT, xnfdf.for 3714-3716)
    molecular_ions     populations of the molecular ions (False: stock XNFDF's output)
    spectrum           SpectrumOptions of the line spectrum
    turbulence         'kurucz' (VTTAB) | 'exact' kernel (turbulence.turbulence_kernel)
    floor              'before': BUFFER = max(lines, 1e-3 continuum) is broadened (MAIN loop 401);
                       'after': the line spectrum is broadened, then floored
    steps              'kurucz' (DFINTERVALS NSTEPS) | 'exact' step boundaries (distribution.distribution_steps)
    nodes              'exact' | 'deck' (T, P) nodes (grid.OdfGrid)
    temperature_limits read each packed list only in DFSYNTHE's temperature range (lines.FILE_T_LIMITS)
    vturbs             microturbulences, km/s
    segment_points     grid points per block (memory)"""
    const: PhysicalConstants = CODATA2018
    eos: dict = field(default_factory=dict)
    molecules_path: str | None = None
    continuum: object = None
    continua_path: str | None = None
    processes: tuple | None = None
    molecular_ions: bool = True
    spectrum: SpectrumOptions = field(default_factory=SpectrumOptions)
    turbulence: str = "exact"
    floor: str = "after"
    steps: str = "exact"
    nodes: str = "exact"
    temperature_limits: bool = False
    vturbs: tuple = VTURBS
    segment_points: int = 65536

    @classmethod
    def kurucz(cls, **changes) -> "OdfConfig":
        from ..opacity.continuum import KuruczCompat
        base = cls(const=KURUCZ, eos=dict(network="kurucz", debye="kurucz", lagged_lowering=True),
                   molecules_path=str(_ext() / "BasicATLAS" / "data" / "dfsynthe_files" / "molecules.dat"),
                   continuum=KuruczCompat.all(), processes=XNFDF_PROCESSES, molecular_ions=False,
                   spectrum=SpectrumOptions.kurucz(),
                   turbulence="kurucz", floor="before", steps="kurucz", nodes="deck", temperature_limits=True)
        return replace(base, **changes)

    def describe(self) -> dict:
        c, s = self.continuum, self.spectrum
        lo = s.lines
        return {"const": self.const.name, "eos": dict(self.eos), "molecules_path": self.molecules_path,
                "continuum": {} if c is None else {k: v for k, v in vars(c).items() if v},
                "processes": None if self.processes is None else list(self.processes),
                "molecular_ions": self.molecular_ions,
                "spectrum": {"voigt": lo.voigt, "cutoff_rule": lo.cutoff_rule, "cutoff": lo.cutoff,
                             "max_wing_points": lo.max_wing_points, "weak_line_doppler_cap": s.weak_line_doppler_cap,
                             "molecular_doppler_cap": s.molecular_doppler_cap, "cap_slots": s.cap_slots,
                             "xlinop_window": s.xlinop_window, "nlte_wings": s.nlte_wings,
                             "lyman_alpha": s.lyman_alpha, "hydrogen_sampling": s.hydrogen_sampling},
                "turbulence": self.turbulence, "floor": self.floor, "steps": self.steps, "nodes": self.nodes,
                "temperature_limits": self.temperature_limits, "vturbs": list(self.vturbs)}

    def equation_of_state(self):
        from ..eos import ChemicalEquilibrium
        from ..io.molecules_dat import read_molecules_dat
        rec = read_molecules_dat(self.molecules_path) if self.molecules_path else None
        return ChemicalEquilibrium(rec, const=self.const, **self.eos)

    def continuous_opacity(self):
        from ..opacity.continuum import ContinuousOpacity, KuruczCompat
        return ContinuousOpacity(self.processes, const=self.const, compat=self.continuum or KuruczCompat())


def temperature_steps(gas, config: OdfConfig, source: DfLineSource | None = None, nlte=None,
                      stats: dict | None = None, blocks=None) -> dict:
    """{'big': (V, 328, L, 12), 'little': (V, 1212, L, 12)} step means of log10 kappa for one OdfGas row (on the
    gas's device; results on the CPU). blocks: the grid.segments() to compute (default all; the other intervals
    are left NaN)."""
    dev = gas.T.device
    spec = LineSpectrum(gas, source, nlte, config.spectrum, config.const)
    kernels = [turbulence_kernel(v, DF_GRID.resolution, config.turbulence).to(dev) for v in config.vturbs]
    halo = max(len(k) for k in kernels) // 2
    L = gas.T.shape[0]
    sets = {"big": IntervalSet.big(), "little": IntervalSet.little()}
    ranges = {k: interval_index_ranges(v) for k, v in sets.items()}
    out = {k: torch.full((len(config.vturbs), len(v), L, 12), float("nan"), dtype=torch.float64)
           for k, v in sets.items()}
    cut = config.spectrum.lines.cutoff
    for seg in segments(config.segment_points) if blocks is None else blocks:
        lo, hi = seg.begin - halo, seg.end + halo
        a, b = max(lo, 0), min(hi, DF_GRID.length - 1)
        lines, cont = spec(a, b)
        x = torch.nn.functional.pad(lines, (a - lo, hi - b))
        floor = torch.nn.functional.pad(cont, (a - lo, hi - b)) * cut
        if config.floor == "before":
            x = torch.maximum(x, floor)
        for iv, ker in enumerate(kernels):
            h = len(ker) // 2
            y = broaden(x[:, halo - h:x.shape[1] - (halo - h)], ker)
            if config.floor == "after":
                y = torch.maximum(y, floor[:, halo:floor.shape[1] - halo])
            ly = torch.log10(y)
            for kind, (i0, i1) in (("big", seg.big), ("little", seg.little)):
                beg, end = ranges[kind]
                idx = torch.repeat_interleave(torch.arange(i1 - i0, device=dev),
                                              torch.as_tensor(end[i0:i1] - beg[i0:i1] + 1, device=dev))
                out[kind][iv, i0:i1] = distribution_steps(ly, idx, i1 - i0, config.steps).transpose(0, 1).cpu()
        del lines, cont, x, floor
    if stats is not None:
        stats.update(spec.counts)
    return out


def build_odf(comp: Composition, config: OdfConfig = OdfConfig(), temperatures=None, gas=None, workdir=None,
              verbose: bool = False, device=None) -> dict:
    """ODF tables of `comp`, computed on `device`. temperatures: indices (0..56) into ODF_LOG_T (default all;
    computed in the given order). Each temperature's steps are saved to workdir/t{index:02d}.pt (skipped when
    present). Returns {'big': OpacityDistribution (all vturbs), 'stats': {...}} when all 57 temperatures are
    available, else {'parts': {index: steps}, 'stats': ...}; LITTLE tables are assembled with assemble().
    Gradients: temperature_steps is differentiable, but a whole temperature keeps every block's graph (several
    GB); differentiate block by block (temperature_steps(..., blocks=[segment]))."""
    idx = list(range(len(ODF_LOG_T))) if temperatures is None else [int(i) for i in temperatures]
    wd = None if workdir is None else Path(workdir)
    if wd is not None:
        wd.mkdir(parents=True, exist_ok=True)
    todo = [i for i in idx if wd is None or not (wd / f"t{i:02d}.pt").exists()]
    rows = gas if gas is not None else (odf_gas(comp, config, ODF_LOG_T[todo], config.nodes, device) if todo else [])
    source = DfLineSource(temperature_limits=config.temperature_limits)
    nlte = nlte_lines()
    parts, stats = {}, {}
    for k, i in enumerate(todo):
        t0 = time.time()
        st = {}
        steps = temperature_steps(rows[k], config, source, nlte, st)
        st["seconds"] = round(time.time() - t0, 1)
        stats[i] = st
        if wd is not None:
            torch.save(dict(steps, stats=st, vturbs=list(config.vturbs)), wd / f"t{i:02d}.pt")
        else:
            parts[i] = steps
        if verbose:
            print(f"T index {i} ({10 ** ODF_LOG_T[i]:.0f} K): {st}", flush=True)
    if wd is not None and all((wd / f"t{i:02d}.pt").exists() for i in range(len(ODF_LOG_T))):
        return {"big": assemble(wd, "big", config.vturbs), "stats": stats}
    return {"parts": parts, "stats": stats}


def assemble(workdir, kind: str = "big", vturbs=VTURBS) -> OpacityDistribution:
    """OpacityDistribution of the given vturbs from the per-temperature parts (V, I, 57, 25, 12); one vturb at a
    time for LITTLE (166 MB each in float64)."""
    wd = Path(workdir)
    iv = IntervalSet.big() if kind == "big" else IntervalSet.little()
    out = torch.empty(len(vturbs), len(iv), len(ODF_LOG_T), 25, 12, dtype=torch.float64)
    for i in range(len(ODF_LOG_T)):
        d = torch.load(wd / f"t{i:02d}.pt")
        all_v = list(d.get("vturbs", VTURBS))
        for j, v in enumerate(vturbs):
            out[j, :, i] = d[kind][all_v.index(v)]
    return OpacityDistribution(iv, out, tuple(vturbs))


__all__ = ["OdfConfig", "build_odf", "assemble", "temperature_steps", "VTURBS", "XNFDF_PROCESSES"]
