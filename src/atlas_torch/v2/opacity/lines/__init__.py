"""Line opacity: line lists (LineList, readers of the BasicATLAS lists), SYNTHE's wavelength grid, and the LTE
line absorption coefficient per gram, shape (..., layers, wavelengths)."""
from .grid import LogWavelengthGrid
from .linelist import LineKind, LineList
from .opacity import (GasState, LineOpacity, LineOpacityOptions, doppler_width, interpolate_edge_table, line_strength,
                      perturber_density, series_limits)
from .readers import (LineSelection, read_basicatlas_lines, read_gfall, read_gfall_kurucz, read_h2o,
                      read_molecular_asc)

KURUCZ_OPTIONS = dict(line_center="grid", doppler_mass="species", voigt="kurucz", max_wing_points=10000,
                      helium_ii="skip", cutoff_rule="kurucz")

__all__ = ["LogWavelengthGrid", "LineKind", "LineList", "GasState", "LineOpacity", "LineOpacityOptions",
           "interpolate_edge_table", "LineSelection", "read_basicatlas_lines", "read_gfall", "read_gfall_kurucz",
           "read_h2o", "read_molecular_asc", "KURUCZ_OPTIONS", "doppler_width", "line_strength", "perturber_density",
           "series_limits"]
