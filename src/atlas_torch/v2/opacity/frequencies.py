"""Frequency quadratures for frequency-integrated quantities with ODF line opacity: integral over nu of f(nu) ~
sum_i w_i sum_s omega_s f(nu_i, kappa_is), steps s of the ODF interval containing nu_i with fractions omega_s
(Kurucz 1970 sec. 2.10; Castelli 2005, MSAIS 8, 34, p. 34).

    q = FrequencyQuadrature.atlas_big()        # the 337 frequencies of ATLAS9's BIG set
    s = q.samples()                            # 12 samples per frequency with an ODF interval
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...tables import table
from ..constants import CODATA2018, PhysicalConstants, decimal32
from .odf import STEP_FRACTIONS, IntervalSet

# ATLAS9 BLOCKR DATA WBIG (atlas9mem.for): wavelengths (nm) of the BIG frequency set, 9.09 nm .. 160 um
WBIG_NM = decimal32(np.concatenate([table("atlas9mem.for", "blockr", n) for n in ("wbig1", "wbig2", "wbig3")]))
# KAPPA9 FRESET(89) (kappa9.for MAIN): lower edge of its first frequency's weight, Hz
KAPPA9_LOWER_EDGE_HZ = 3.364687e16


@dataclass(frozen=True, eq=False)
class SampleSet:
    """Quadrature samples: nu (K,) Hz, weight (K,) Hz = w_i omega_s, frequency (K,) index i, interval (K,) ODF
    interval (-1: continuum only), step (K,) ODF step. 12 samples per frequency with an interval, 1 without,
    frequency-major."""
    nu: torch.Tensor
    weight: torch.Tensor
    frequency: torch.Tensor
    interval: torch.Tensor
    step: torch.Tensor

    def __len__(self) -> int:
        return self.nu.shape[0]

    def chunks(self, n: int) -> list[slice]:
        """Slices of about n samples that never split the steps of one frequency."""
        f = self.frequency.cpu().numpy()
        starts = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
        out, a = [], 0
        for s in starts[1:]:
            if s - a >= n:
                out.append(slice(a, int(s)))
                a = int(s)
        out.append(slice(a, len(f)))
        return out

    def to(self, device) -> "SampleSet":
        return SampleSet(*(getattr(self, k).to(device) for k in ("nu", "weight", "frequency", "interval", "step")))


@dataclass(frozen=True, eq=False)
class FrequencyQuadrature:
    """Nodes nu (F,) Hz with weights (F,) Hz and their ODF interval (F,) (-1 = continuum only)."""
    nu: torch.Tensor
    weight: torch.Tensor
    interval: torch.Tensor

    @classmethod
    def atlas_big(cls, intervals: IntervalSet | None = None, lower_edge_nm: float = 8.97666,
                  const: PhysicalConstants = CODATA2018) -> "FrequencyQuadrature":
        """ATLAS9's 337 BIG frequencies nu_i = c/WBIG_i with the weights of READIN labels 2210-2212:
        (nu_{i-1} - nu_{i+1})/2, the first (c/lower_edge - nu_2)/2, the last nu_336/2; the interval of each node
        as LINOP finds it (last interval beyond 10 um). KAPPA9 uses lower_edge_nm = c/KAPPA9_LOWER_EDGE_HZ."""
        iv = intervals or IntervalSet.big()
        c_nm = const.c * 1e7
        nu = torch.as_tensor(c_nm / WBIG_NM, dtype=torch.float64)
        w = torch.empty_like(nu)
        w[1:-1] = (nu[:-2] - nu[2:]) / 2.0
        w[0] = (c_nm / lower_edge_nm - nu[1]) / 2.0
        w[-1] = nu[-2] / 2.0
        return cls(nu, w, iv.interval_of(torch.as_tensor(WBIG_NM)))

    @classmethod
    def intervals(cls, intervals: IntervalSet, uv_to_nm: float | None = None, ir_to_nm: float | None = None,
                  n_extension: int = 20, const: PhysicalConstants = CODATA2018) -> "FrequencyQuadrature":
        """One node per interval at its frequency midpoint, weighted by its own width c/lambda_lo - c/lambda_hi,
        plus n_extension log-spaced continuum nodes shortward to uv_to_nm (interval -1) and longward to ir_to_nm
        (the last interval's ODF, as LINOP extends it)."""
        c_nm = const.c * 1e7
        edges = [np.asarray(intervals.edges_nm, dtype=np.float64)]
        if uv_to_nm is not None:
            edges.insert(0, np.geomspace(uv_to_nm, edges[0][0], n_extension + 1)[:-1])
        if ir_to_nm is not None:
            edges.append(np.geomspace(edges[-1][-1], ir_to_nm, n_extension + 1)[1:])
        e = np.concatenate(edges)
        f = c_nm / e
        nu = torch.as_tensor(0.5 * (f[:-1] + f[1:]))
        w = torch.as_tensor(f[:-1] - f[1:])
        iv = intervals.interval_of(torch.as_tensor(0.5 * (e[:-1] + e[1:])), extend_last=True)
        return cls(nu, w, iv)

    def samples(self, fractions=STEP_FRACTIONS) -> SampleSet:
        om = torch.as_tensor(np.asarray(fractions), dtype=torch.float64, device=self.nu.device)
        ns = om.shape[0]
        has = self.interval >= 0
        count = torch.where(has, ns, 1)
        freq = torch.repeat_interleave(torch.arange(len(self.nu), device=self.nu.device), count)
        first = torch.cumsum(count, 0) - count
        step = torch.arange(len(freq), device=freq.device) - first[freq]
        lines = has[freq]
        weight = self.weight[freq] * torch.where(lines, om[step.clamp(max=ns - 1)], torch.ones_like(om[:1]))
        return SampleSet(self.nu[freq], weight, freq, self.interval[freq], torch.where(lines, step, 0))

    def to(self, device) -> "FrequencyQuadrature":
        return FrequencyQuadrature(self.nu.to(device), self.weight.to(device), self.interval.to(device))


__all__ = ["FrequencyQuadrature", "SampleSet", "WBIG_NM", "KAPPA9_LOWER_EDGE_HZ"]
