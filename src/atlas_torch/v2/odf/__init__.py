"""Opacity distribution functions (DFSYNTHE physics; Castelli 2005, MSAIS 8, 34; Kurucz 2005, MSAIS 8, 14, 76).

    res = build_odf(comp, OdfConfig.kurucz(), workdir=parts)      # 57 temperatures x 25 pressures x 5 vturbs
    odf = res["big"]                                               # atlas_torch.v2.opacity.odf.OpacityDistribution
"""
from .build import VTURBS, XNFDF_PROCESSES, OdfConfig, assemble, build_odf, temperature_steps
from .distribution import distribution_steps
from .gas import OdfGas, odf_gas
from .grid import DF_GRID, OdfGrid, interval_index_ranges, segments, step_bounds
from .lines import DfLineSource, SeriesContinua, nelion_species, nlte_lines
from .spectrum import LineSpectrum, SpectrumOptions
from .turbulence import broaden, turbulence_kernel
from .wings import WingPairs, deposit, first_below

__all__ = ["VTURBS", "XNFDF_PROCESSES", "OdfConfig", "assemble", "build_odf", "temperature_steps",
           "distribution_steps", "OdfGas", "odf_gas", "DF_GRID", "OdfGrid", "interval_index_ranges", "segments",
           "step_bounds", "DfLineSource", "SeriesContinua", "nelion_species", "nlte_lines", "LineSpectrum",
           "SpectrumOptions", "broaden", "turbulence_kernel", "WingPairs", "deposit", "first_below"]
