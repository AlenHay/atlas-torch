"""Native-math mode for the differentiable path.

The exact (default) path of the port reproduces the Fortran's arithmetic: glibc libm element by element through
Python/ctypes (`atlas9.physics.libm.EXACT`, `synthe.opacity._core.EXACT`), correctly rounded sqrt through numpy,
REAL*4 emulation (PFIRON runs in float32). Those calls leave autograd (numpy round trips) or round to float32.

`native_math()` is a context manager that switches the *existing opt-in flags* of those modules to native
torch kernels (`EXACT = False`: torch.exp/log/log10/pow/sqrt, which differ from glibc by <= 1 ulp) and, for the
duration of the block, replaces the REAL*4 PFIRON interpolation used by the SYNTHE-variant PFSAHA with the same
formulas in float64 (`pfiron_stage1_f64`, `pfiron_stage2_f64`; table cells chosen as in REAL*4) and lets
SahaPlan keep those values in float64 (`_TorchF64Shim`). Nothing is edited on disk; on exit every flag
and attribute is restored, so the exact path is untouched outside the block.

Forward-mode AD (torch.autograd.forward_ad) runs through these torch kernels; see atlas_torch.diff.__init__ for
why forward mode is the primary mode.
"""
from __future__ import annotations

import contextlib
import threading

import numpy as np
import torch

_LOCK = threading.RLock()
_DEPTH = [0]
_SAVED: list = []


def _pfi():
    from ..atlas9.physics import pfsaha as a9
    return a9


def pfiron_stage1_f64(nelem: int, ion: int, tlog8: torch.Tensor, variant: str = "atlas9") -> torch.Tensor:
    """PFIRON's temperature interpolation (same formulas as atlas9.physics.pfsaha.pfiron_stage1) in float64.
    The table cell (IT) is chosen exactly as the REAL*4 routine chooses it (index decisions from float32)."""
    a9 = _pfi()
    pftab, _, _ = a9._pfiron_tables(variant)
    c = {k: float(v) for k, v in a9._PFI.items()}
    c32 = {k: torch.tensor(v, dtype=torch.float32, device=tlog8.device) for k, v in a9._PFI.items()}
    tl32 = tlog8.detach().to(torch.float32)
    it_hi = torch.clamp(torch.trunc((tl32 - c32["4.0"]) / c32[".05"] + c32["31."]).to(torch.int64), max=56)
    it_lo = torch.clamp(torch.trunc((tl32 - c32["3.32"]) / c32[".02"] + c32["2."]).to(torch.int64), min=2)
    it_mid = torch.trunc((tl32 - c32["3.7"]) / c32[".03"] + c32["21."]).to(torch.int64)
    hi, lo = tl32 > c32["4."], tl32 < c32["3.7"]
    tl = tlog8.to(torch.float64)
    f_hi = (tl - (it_hi - 31).to(torch.float64) * c[".05"] - c["4."]) / c[".05"]
    f_lo = (tl - (it_lo - 2).to(torch.float64) * c[".02"] - c["3.32"]) / c[".02"]
    f_mid = (tl - (it_mid - 21).to(torch.float64) * c[".03"] - c["3.7"]) / c[".03"]
    it = torch.where(hi, it_hi, torch.where(lo, it_lo, it_mid))
    f = torch.where(hi, f_hi, torch.where(lo, f_lo, f_mid))
    tab = torch.from_numpy(np.ascontiguousarray(pftab[:, :, ion - 1, nelem - 20]).astype(np.float64)).to(tl.device)
    return tab[:, it - 1].T * f[:, None] + (1.0 - f)[:, None] * tab[:, it - 2].T          # (n, 7)


def pfiron_stage2_f64(a: torch.Tensor, potlow8: torch.Tensor, variant: str = "atlas9") -> torch.Tensor:
    """PFIRON's lowering interpolation (atlas9.physics.pfsaha.pfiron_stage2) in float64 (LOW chosen in float32
    as the REAL*4 routine does)."""
    a9 = _pfi()
    _, potlo, potlolog = a9._pfiron_tables(variant)
    pl32 = potlow8.detach().to(torch.float32)
    potlo_t = torch.from_numpy(potlo).to(pl32.device)
    less = pl32[..., None] < potlo_t
    anyless = less.any(-1)
    first = torch.where(anyless, less.to(torch.int64).argmax(-1) + 1, torch.zeros_like(pl32, dtype=torch.int64))
    direct = (first == 1) | (first == 0)
    low = torch.where(first == 0, torch.full_like(first, 7), first)
    lowm = torch.clamp(low - 1, min=1)
    a = a.to(torch.float64)
    a_low = torch.gather(a, -1, (low - 1)[..., None])[..., 0]
    a_lowm = torch.gather(a, -1, (lowm - 1)[..., None])[..., 0]
    pl = potlow8.to(torch.float64)
    plog = torch.from_numpy(potlolog.astype(np.float64)).to(pl.device)[lowm - 1]
    safe = torch.where(direct, torch.ones_like(pl), pl)                  # log10 only where taken
    pp = (torch.log10(safe) - plog) / float(a9._PFI[".30103"])
    return torch.where(direct, a_low, pp * a_low + (1.0 - pp) * a_lowm)


class _TorchF64Shim:
    """Stand-in for the `torch` module inside atlas_torch.synthe.pfsaha: identical except that
    torch.ones(..., dtype=torch.float32) (SahaPlan's PFIRON stage-1 buffer, the module's only float32 tensor)
    is allocated in float64, so the float64 PFIRON values are not rounded to REAL*4."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def ones(*a, **kw):
        if kw.get("dtype") is torch.float32:
            kw["dtype"] = torch.float64
        return torch.ones(*a, **kw)


_SHIM = _TorchF64Shim()


@contextlib.contextmanager
def native_math():
    """Switch the port's libm emulation to native torch float64 kernels (re-entrant)."""
    from ..atlas9.physics import libm as a9libm
    from ..synthe import pfsaha as spf
    from ..synthe.opacity import _core
    with _LOCK:
        if _DEPTH[0] == 0:
            _SAVED.append((a9libm.EXACT, _core.EXACT, spf.pfiron_stage1, spf.pfiron_stage2, spf.torch))
            a9libm.EXACT = False
            _core.EXACT = False
            spf.pfiron_stage1 = pfiron_stage1_f64
            spf.pfiron_stage2 = pfiron_stage2_f64
            spf.torch = _SHIM
        _DEPTH[0] += 1
    try:
        yield
    finally:
        with _LOCK:
            _DEPTH[0] -= 1
            if _DEPTH[0] == 0:
                (a9libm.EXACT, _core.EXACT, spf.pfiron_stage1, spf.pfiron_stage2, spf.torch) = _SAVED.pop()


def straight_through(table_value: torch.Tensor, smooth: torch.Tensor) -> torch.Tensor:
    """Value of the Fortran's tabulated function, derivative of the smooth function it tabulates.

    The exact path evaluates exp(-x), exp(-v**2), H1(v) ... from REAL*4 tables at a rounded argument
    (FASTEX, H0TAB/H1TAB/H2TAB): piecewise constant, derivative 0 almost everywhere. Returns
    table_value (detached) + (smooth - smooth.detach()): the primal is the table value, the tangent / gradient
    is that of `smooth`."""
    return table_value.detach() + (smooth - smooth.detach())
