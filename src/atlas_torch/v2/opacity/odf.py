"""Line opacity as opacity distribution functions (ODFs): in each wavelength interval the line opacity is replaced
by its distribution (a monotonic function of the fraction of the interval), represented by a step function on 12
sub-intervals (Kurucz 1970 sec. 5.15, eqs. 5.23-5.24; Castelli 2005, MSAIS 8, 34, p. 35). Tabulated in log10 kappa
(cm^2 g^-1) on 57 log T x 25 log P_gas, per microturbulence; ATLAS9 LINOP interpolates it bilinearly in dex.

    odf = OpacityDistribution.read(".../odf/solar", "big", vturbs=(2,))
    kappa = odf.kappa(T, P)                    # (..., L, 328, 12) cm^2 g^-1
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ...tables import table
from ..constants import decimal32

# LINOP DATA TABT (atlas9mem.for), log10 T; = log10 of XNFDF's 57 temperatures 1995 K .. 199526 K
ODF_LOG_T = decimal32(table("atlas9mem.for", "linop", "tabt"))
# LINOP DATA TABP, log10 P_gas (dyn cm^-2): -4 .. 8 in steps of 0.5
ODF_LOG_P = np.arange(25) * 0.5 - 4.0
# width of each step as a fraction of the interval, in stored (increasing-opacity) order: DFINTERVALS NSTEPS
# (1/10 x 9, 1/20, 1/30, 1/60); LINOP weight WT(13-n) of stored step n
STEP_FRACTIONS = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 2, 1], dtype=np.float64) / 60.0


@lru_cache(maxsize=None)
def _dfintervals(kind: str) -> np.ndarray:
    names = ("biga", "bigb", "bigc") if kind == "big" else ("lita", "litb", "litc", "litd", "lite", "litf", "litg")
    return decimal32(np.concatenate([table("dfsynthe.for", "dfintervals", n) for n in names]))


@dataclass(frozen=True, eq=False)
class IntervalSet:
    """Wavelength intervals of an ODF: edges_nm (n+1,) increasing, vacuum nm."""
    name: str
    edges_nm: np.ndarray

    @classmethod
    def big(cls) -> "IntervalSet":
        """The 328 BIG intervals, 8.97666 - 10000 nm (dfsynthe.for DFINTERVALS WAVEBIG; = atlas9mem.for BLOCKBIG
        WAVEDF(1..328,2), where WAVEDF(329,2) = 1e7)."""
        return cls("big", _dfintervals("big"))

    @classmethod
    def little(cls) -> "IntervalSet":
        """The 1212 LITTLE intervals (DFINTERVALS WAVELIT)."""
        return cls("little", _dfintervals("little"))

    def __len__(self) -> int:
        return len(self.edges_nm) - 1

    def interval_of(self, wl_nm, extend_last: bool = True) -> torch.Tensor:
        """0-based interval of each wavelength (edges[i] <= wl < edges[i+1]); -1 shortward of edges[0].
        extend_last: wavelengths beyond edges[-1] belong to the last interval (LINOP WLEND(328) = 1e6 nm), else -1."""
        wl = torch.as_tensor(wl_nm, dtype=torch.float64)
        e = torch.as_tensor(self.edges_nm, dtype=torch.float64, device=wl.device)
        i = torch.searchsorted(e, wl.contiguous(), right=True) - 1
        n = len(self)
        if extend_last:
            i = torch.where(i >= n, torch.full_like(i, n - 1), i)
        return torch.where(i >= n, torch.full_like(i, -1), i)


def _cell(x: torch.Tensor, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower node index i (nodes[i] <= x < nodes[i+1], 0..n-2) and linear weight of node i+1, x clamped to the
    nodes (LINOP labels 7-19)."""
    x = torch.minimum(torch.maximum(x, nodes[0]), nodes[-1])
    i = (torch.searchsorted(nodes, x.detach().contiguous(), right=True) - 1).clamp(0, len(nodes) - 2)
    return i, (x - nodes[i]) / (nodes[i + 1] - nodes[i])


@dataclass(frozen=True, eq=False)
class OpacityDistribution:
    """ODF table: log_kappa (*B, V, I, 57, 25, 12), log10 cm^2 g^-1, steps in increasing opacity with widths
    `fractions`; V microturbulences vturb_kms (km/s); B optional axis of distinct tables (e.g. mixtures)."""
    intervals: IntervalSet
    log_kappa: torch.Tensor
    vturb_kms: tuple
    log_t: np.ndarray = field(default_factory=lambda: ODF_LOG_T)
    log_p: np.ndarray = field(default_factory=lambda: ODF_LOG_P)
    fractions: np.ndarray = field(default_factory=lambda: STEP_FRACTIONS)

    # ------------------------------------------------------------------ files

    @staticmethod
    def file_name(kind: str, vturb_kms, prefix: str = "p00") -> str:
        return f"{prefix}{'big' if kind == 'big' else 'lit'}{int(vturb_kms)}.bdf"

    @classmethod
    def read(cls, directory, kind: str = "big", vturbs=(0, 1, 2, 4, 8), prefix: str = "p00",
             device=None) -> "OpacityDistribution":
        """p00big{v}.bdf / p00lit{v}.bdf of the requested vturbs (BIG: 45 MB per vturb in float64)."""
        from ..io.odf_bdf import read_bdf
        iv = IntervalSet.big() if kind == "big" else IntervalSet.little()
        steps = [torch.from_numpy(read_bdf(Path(directory) / cls.file_name(kind, v, prefix)).astype(np.float64))
                 for v in vturbs]
        lk = torch.stack(steps).to(device) / 1000.0
        if lk.shape[1] != len(iv):
            raise ValueError(f"{kind} ODF has {lk.shape[1]} intervals, expected {len(iv)}")
        return cls(iv, lk, tuple(vturbs))

    def write(self, directory, prefix: str = "p00") -> None:
        """One file per vturb, 1000 log10 kappa rounded to the nearest integer (int16)."""
        from ..io.odf_bdf import write_bdf
        if self.log_kappa.dim() != 5:
            raise ValueError("write() takes a table without a B axis")
        Path(directory).mkdir(parents=True, exist_ok=True)
        kind = "big" if self.intervals.name == "big" else "little"
        for j, v in enumerate(self.vturb_kms):
            write_bdf(Path(directory) / self.file_name(kind, v, prefix), quantize(self.log_kappa[j]))

    # ------------------------------------------------------------------ containers

    def select_vturb(self, v_kms) -> "OpacityDistribution":
        j = [float(v) for v in self.vturb_kms].index(float(v_kms))
        return replace(self, log_kappa=self.log_kappa[..., j:j + 1, :, :, :, :], vturb_kms=(self.vturb_kms[j],))

    @staticmethod
    def stack(tables: list["OpacityDistribution"]) -> "OpacityDistribution":
        """Tables with the same intervals and vturbs along a new leading B axis."""
        return replace(tables[0], log_kappa=torch.stack([t.log_kappa for t in tables]))

    def to(self, device) -> "OpacityDistribution":
        return replace(self, log_kappa=self.log_kappa.to(device))

    def in_domain(self, T, P) -> torch.Tensor:
        """T, P_gas inside the tabulated (log T, log P) rectangle (outside, log_kappa_at clamps)."""
        lt, lp = torch.log10(torch.as_tensor(T)), torch.log10(torch.as_tensor(P))
        return (lt >= self.log_t[0]) & (lt <= self.log_t[-1]) & (lp >= self.log_p[0]) & (lp <= self.log_p[-1])

    # ------------------------------------------------------------------ evaluation

    def _vturb_weights(self, vturb, shape, device) -> torch.Tensor:
        """(N, L, V) linear interpolation weights in vturb (cm/s, clamped to the table)."""
        nv = len(self.vturb_kms)
        if vturb is None:
            if nv != 1:
                raise ValueError("several vturb tables: pass vturb or select_vturb() first")
            return torch.ones(shape + (1,), dtype=torch.float64, device=device)
        v = vturb
        if nv == 1:
            return torch.ones(shape + (1,), dtype=torch.float64, device=device)
        nodes = torch.as_tensor([float(x) * 1e5 for x in self.vturb_kms], dtype=torch.float64, device=device)
        j, t = _cell(v, nodes)
        w = torch.zeros(shape + (nv,), dtype=torch.float64, device=device)
        w = w.scatter(-1, j[..., None], (1.0 - t)[..., None])
        return w.scatter_add(-1, (j + 1)[..., None], t[..., None])

    def log_kappa_at(self, T, P, vturb=None, table_index=None) -> torch.Tensor:
        """log10 kappa (..., L, I, 12) at T (..., L) K and P_gas (..., L) dyn cm^-2: bilinear in (log10 T, log10 P)
        clamped to the table (LINOP; atlas9mem.for), linear in vturb (cm/s, clamped) between the V tables when
        vturb is given; table_index (...) selects the B table per model."""
        T = torch.as_tensor(T, dtype=torch.float64, device=self.log_kappa.device)
        P = torch.as_tensor(P, dtype=torch.float64, device=T.device).expand(T.shape)
        lk = self.log_kappa if self.log_kappa.dim() == 6 else self.log_kappa[None]
        batch, L = T.shape[:-1], T.shape[-1]
        tf, pf = T.reshape(-1, L), P.reshape(-1, L)
        n = tf.shape[0]
        if table_index is None:
            if lk.shape[0] != 1:
                raise ValueError("a stacked table needs table_index")
            b = torch.zeros(n, 1, dtype=torch.long, device=T.device)
        else:
            b = torch.as_tensor(table_index, device=T.device).expand(batch).reshape(n, 1)
        lt = torch.as_tensor(self.log_t, dtype=torch.float64, device=T.device)
        lp = torch.as_tensor(self.log_p, dtype=torch.float64, device=T.device)
        it, x = _cell(torch.log10(tf), lt)
        ip, y = _cell(torch.log10(pf), lp)
        if vturb is not None:
            vturb = torch.as_tensor(vturb, dtype=torch.float64, device=T.device).expand(T.shape).reshape(n, L)
        w = self._vturb_weights(vturb, (n, L), T.device)
        out = None
        for dt, dp, c in ((0, 0, (1 - x) * (1 - y)), (0, 1, (1 - x) * y), (1, 0, x * (1 - y)), (1, 1, x * y)):
            corner = lk[b, :, :, it + dt, ip + dp]                                   # (N, L, V, I, 12)
            term = torch.einsum("nlv,nlvis->nlis", w * c[..., None], corner)
            out = term if out is None else out + term
        return out.reshape(batch + (L,) + out.shape[-2:])

    def kappa(self, T, P, vturb=None, table_index=None) -> torch.Tensor:
        """10**log_kappa_at, cm^2 g^-1 (..., L, I, 12)."""
        return torch.pow(10.0, self.log_kappa_at(T, P, vturb, table_index))

    def sampled(self, T, P, samples, vturb=None, table_index=None) -> torch.Tensor:
        """Line opacity at the samples of a SampleSet (opacity.frequencies): (..., L, K) cm^2 g^-1, step
        samples.step of interval samples.interval; 0 where samples.interval == -1."""
        k = self.kappa(T, P, vturb, table_index)
        ns = k.shape[-1]
        flat = k.reshape(k.shape[:-2] + (-1,))
        iv = samples.interval.to(flat.device)
        idx = iv.clamp(min=0) * ns + samples.step.to(flat.device)
        out = flat[..., idx]
        return torch.where(iv >= 0, out, torch.zeros((), dtype=out.dtype, device=out.device))


def quantize(log_kappa: torch.Tensor) -> np.ndarray:
    """1000 log10 kappa as int16, nearest integer (the .bdf storage; Kurucz 2005, MSAIS 8, 76)."""
    q = torch.round(log_kappa.detach() * 1000.0).clamp(-32768, 32767)
    return q.cpu().numpy().astype(np.int16)


__all__ = ["ODF_LOG_T", "ODF_LOG_P", "STEP_FRACTIONS", "IntervalSet", "OpacityDistribution", "quantize"]
