"""A continuous opacity sampled at XNFPELSYN's continuum frequencies and interpolated between them, as SYNTHE and
SPECTRV use it (xnfpelsyn.for loop 18; spectrv.for labels 2004-2006)."""
from __future__ import annotations

import numpy as np
import torch

from ...eos.equilibrium import Populations
from ...io.continua_dat import frequency_set, read_continua_dat
from .. import Opacity

C_NM = 2.99792458e17


class EdgeSampled:
    """continuum(pops, nu) evaluated at the lower edge, the mid wavelength and the upper edge of each continua.dat
    interval covering nu, then log10 kappa interpolated by the parabola through the three points (in wavelength),
    separately for absorption and scattering. continuum: a provider (pops, nu) -> Opacity; edges:
    read_continua_dat() by default."""

    def __init__(self, continuum, edges: dict | None = None):
        e = read_continua_dat() if edges is None else edges
        self.continuum = continuum
        self.edges_nm = np.abs(e["wavelength_nm"])
        self.freqset = frequency_set(e)

    def __call__(self, pops: Populations, nu) -> Opacity:
        nu = torch.as_tensor(nu, dtype=torch.float64, device=pops.T.device)
        wl = C_NM / nu
        edges = torch.as_tensor(self.edges_nm, dtype=wl.dtype, device=wl.device)
        cell = (torch.searchsorted(edges, wl.contiguous(), right=True) - 1).clamp(0, len(edges) - 2)
        i0, i1 = int(cell.min()), int(cell.max())
        fs = torch.as_tensor(self.freqset[3 * i0:3 * i1 + 3], dtype=nu.dtype, device=nu.device)
        op = self.continuum(pops, fs)
        a, b = edges[cell], edges[cell + 1]
        m = 0.5 * (a + b)
        lag = torch.stack([(wl - m) * (wl - b) / ((a - m) * (a - b)), (wl - a) * (wl - b) / ((m - a) * (m - b)),
                           (wl - a) * (wl - m) / ((b - a) * (b - m))], -1)                     # (F, 3)
        col = 3 * (cell - i0)[:, None] + torch.arange(3, device=nu.device)                    # (F, 3)

        def interp(k):
            lk = torch.log10(k)
            out = torch.zeros(lk.shape[:-1] + nu.shape, dtype=lk.dtype, device=lk.device)
            for j in range(3):
                out = out + lag[:, j] * lk[..., col[:, j]]
            return torch.pow(10.0, out)
        return Opacity(interp(op.absorption), interp(op.scattering))
