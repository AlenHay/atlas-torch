"""SYNTHE work files as opacity tables: the XNFPELSYN continuum (fort.10) and the SYNTHE line opacity (fort.9),
used as stand-in opacities and to validate the transfer alone against stock spectra."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

KW = 99


@dataclass(frozen=True)
class ContinuumTable:
    """log10 continuum absorption and scattering (cm^2 g^-1) at the edge wavelengths and the midpoints between
    them: log_abs[e, k, j], k = 0, 1, 2 at edge[e], (edge[e] + edge[e+1])/2, edge[e+1]; layer j."""
    edges_nm: np.ndarray
    log_abs: np.ndarray
    log_scat: np.ndarray

    def at(self, wavelength_nm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(absorption, scattering) (W, L) by quadratic interpolation in log10 between the three points of the
        enclosing edge interval (spectrv.for, labels 2004-2006)."""
        dev, dt = wavelength_nm.device, wavelength_nm.dtype
        edges = torch.as_tensor(self.edges_nm, dtype=dt, device=dev)
        e = (torch.searchsorted(edges, wavelength_nm.contiguous(), right=True) - 1).clamp(0, len(edges) - 2)
        a, b = edges[e], edges[e + 1]
        m = 0.5 * (a + b)
        w = wavelength_nm
        lag = torch.stack([(w - m) * (w - b) / ((a - m) * (a - b)), (w - a) * (w - b) / ((m - a) * (m - b)),
                           (w - a) * (w - m) / ((b - a) * (b - m))], -1)       # (W, 3)
        f = lambda t: torch.pow(10.0, torch.einsum("wk,wkj->wj", lag, torch.as_tensor(t, dtype=dt, device=dev)[e]))
        return f(self.log_abs), f(self.log_scat)


def read_xnfpelsyn_continuum(path: str | Path) -> ContinuumTable:
    """The continuum of fort.10 (xnfpelsyn_dat.read_xnfpelsyn_dat) as a ContinuumTable."""
    from .xnfpelsyn_dat import read_xnfpelsyn_dat
    d = read_xnfpelsyn_dat(path)
    shape = lambda q: np.ascontiguousarray(q.reshape(q.shape[0], -1, 3).transpose(1, 2, 0))
    return ContinuumTable(d.wl_edge, shape(d.log_cont_abs), shape(d.log_cont_scat))


@dataclass(frozen=True)
class LineTable:
    """fort.9 of SYNTHE: line opacity (cm^2 g^-1, stimulated emission included) on the SYNTHE wavelength grid
    wavelength_n = exp((ix0 + n) ln(1 + 1/resolution)), n = 0..length-1 (vacuum if vacuum else air)."""
    path: Path
    wlbeg: float
    resolution: float
    wlend: float
    length: int
    n_layers: int
    turbv: float
    vacuum: bool
    edges_nm: np.ndarray
    offset: int
    stride: int

    def opacity(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """ASYNTH (stop - start, n_layers) as float64."""
        stop = self.length if stop is None else stop
        raw = np.memmap(self.path, dtype=np.uint8, mode="r")
        try:
            a = np.ndarray((self.length, self.n_layers), dtype="<f4", buffer=raw, offset=self.offset + 4,
                           strides=(self.stride, 4))
            return np.array(a[start:stop], dtype=np.float64)
        finally:
            del raw


def read_synthe_lines(path: str | Path) -> LineTable:
    """fort.9: (WLBEG, RESOLU, WLEND, LENGTH, NRHOX, LINOUT, TURBV, IFVAC), (NEDGE, edges), ASYNTH(99) x LENGTH."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    (n,) = struct.unpack_from("<i", head, 0)
    wlbeg, resolu, wlend, length, nrhox, _linout, turbv, ifvac = struct.unpack_from("<dddiiifi", head, 4)
    off = n + 8
    (n2,) = struct.unpack_from("<i", head, off)
    (nedge,) = struct.unpack_from("<i", head, off + 4)
    edges = np.frombuffer(head, dtype="<f8", count=3 * nedge, offset=off + 8).reshape(nedge, 3)[:, 1].copy()
    off += n2 + 8
    (n3,) = struct.unpack_from("<i", head, off)
    if n3 < 4 * KW:
        raise ValueError("ASYNTH record shorter than 99 REAL*4")
    return LineTable(path, wlbeg, resolu, wlend, length, nrhox, float(turbv), ifvac == 1, edges, off, n3 + 8)

