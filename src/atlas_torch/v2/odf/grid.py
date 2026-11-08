"""The ODF wavelength grid and its partition into intervals and steps (dfsynthe.for MAIN, DFINTERVALS).

DFSYNTHE samples the line opacity at R = 500000 on lambda_k = (1 + 1/R)^k nm, 8.97666 - 10000 nm (3507859
points); interval i holds the points edges[i] < lambda <= edges[i+1]; its sorted values are split into 12 steps of
widths STEP_FRACTIONS (Castelli 2005, MSAIS 8, 34, p. 35)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..opacity.lines.grid import LogWavelengthGrid
from ..opacity.odf import ODF_LOG_P, ODF_LOG_T, STEP_FRACTIONS, IntervalSet

DF_RESOLUTION = 500000.0
DF_WL_BEGIN = 8.97666
DF_LENGTH = 3507859            # dfsynthe.for MAIN: LENGTH=3507859 (last point 9999.99 nm)


def _first_index(wl_begin: float, resolution: float) -> int:
    """MAIN: IXWLBEG = ln(WLBEG)/ln(ratio), +1 if (1+1/R)^IXWLBEG < WLBEG."""
    lr = math.log(1.0 + 1.0 / resolution)
    i = int(math.log(wl_begin) / lr)
    return i + 1 if math.exp(i * lr) < wl_begin else i


DF_GRID = LogWavelengthGrid(DF_RESOLUTION, _first_index(DF_WL_BEGIN, DF_RESOLUTION), DF_LENGTH)


@dataclass(frozen=True, eq=False)
class OdfGrid:
    """(log T, log P) nodes of an ODF table. nodes('exact'): 10**ODF_LOG_T, 10**ODF_LOG_P; nodes('deck'): the
    values on BasicATLAS's xnfdf.com cards (T rounded to 1 K, P to 4 significant digits)."""
    log_t: np.ndarray = field(default_factory=lambda: ODF_LOG_T)
    log_p: np.ndarray = field(default_factory=lambda: ODF_LOG_P)

    def nodes(self, kind: str = "exact") -> tuple[np.ndarray, np.ndarray]:
        t, p = 10.0 ** np.asarray(self.log_t), 10.0 ** np.asarray(self.log_p)
        if kind == "deck":
            return np.round(t), np.array([float(f"{x:.3e}") for x in p])
        if kind != "exact":
            raise ValueError("nodes: 'exact' | 'deck'")
        return t, p


def interval_index_ranges(intervals: IntervalSet, grid: LogWavelengthGrid = DF_GRID) -> tuple[np.ndarray, np.ndarray]:
    """(begin, end) 0-based grid indices (inclusive) of each interval: the points with edges[i] < lambda <=
    edges[i+1] (DFINTERVALS: NEND(i) = INT(ln(edge_{i+1})/ln(ratio)) - NZERO + 1, NBEG(i) = NEND(i-1) + 1)."""
    lr = math.log(1.0 + 1.0 / grid.resolution)
    k = np.floor(np.log(np.asarray(intervals.edges_nm, dtype=np.float64)) / lr).astype(np.int64) - grid.first
    end = k[1:]
    begin = np.r_[max(k[0] + 1, 0), end[:-1] + 1]
    return begin, np.minimum(end, grid.length - 1)


def step_bounds(counts, kind: str = "exact", fractions=STEP_FRACTIONS) -> np.ndarray:
    """Step boundaries (n_intervals, 13) in sorted position, 0 .. count: 'exact' count x cumulative fraction
    (fractional); 'kurucz' DFINTERVALS NSTEPS - 1 = (NN*k)/60 with NN = count - 1 (integer division; the
    largest point of each interval is left out)."""
    n = np.asarray(counts, dtype=np.int64)
    cum = np.r_[0.0, np.cumsum(np.asarray(fractions, dtype=np.float64))]
    if kind == "exact":
        return n[:, None] * cum[None, :]
    if kind != "kurucz":
        raise ValueError("kind: 'exact' | 'kurucz'")
    k60 = np.rint(cum * 60).astype(np.int64)
    b = ((n[:, None] - 1) * k60[None, :]) // 60
    b[:, -1] = n - 1
    return b.astype(np.float64)


@dataclass(frozen=True)
class Segment:
    """Grid points [begin, end] (inclusive, 0-based) holding whole BIG and LITTLE intervals big[0]:big[1] and
    little[0]:little[1] (python slices)."""
    begin: int
    end: int
    big: tuple
    little: tuple


def segments(max_points: int = 65536, grid: LogWavelengthGrid = DF_GRID) -> list[Segment]:
    """Consecutive blocks of the grid, each a union of whole BIG and whole LITTLE intervals, of about max_points
    (blocks are cut only where a BIG and a LITTLE interval end at the same point)."""
    bb, be = interval_index_ranges(IntervalSet.big(), grid)
    lb, le = interval_index_ranges(IntervalSet.little(), grid)
    cuts = np.intersect1d(be, le)
    out, start, ib, il, j = [], int(bb[0]), 0, 0, 0
    while j < len(cuts):
        k = max(int(np.searchsorted(cuts, start + max_points - 1, side="right")) - 1, j)
        c = int(cuts[k])
        jb, jl = int(np.searchsorted(be, c, side="right")), int(np.searchsorted(le, c, side="right"))
        out.append(Segment(start, c, (ib, jb), (il, jl)))
        start, ib, il, j = c + 1, jb, jl, k + 1
    return out


__all__ = ["DF_GRID", "DF_RESOLUTION", "DF_LENGTH", "OdfGrid", "interval_index_ranges", "step_bounds", "Segment",
           "segments"]
