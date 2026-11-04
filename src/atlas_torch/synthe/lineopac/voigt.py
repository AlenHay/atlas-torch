"""VOIGT(V,A) of synthe.for (REAL*4, tabulated H0/H1/H2 from TABVOIGT), vectorised.

Evaluation order and single-precision constants follow the Fortran literally; all arithmetic is float32
(python float constants act as REAL*4 literals in float32 torch ops).  Each branch is evaluated only on
the elements that take it.  `h` is the flat COMMON /H1TAB/ [H0TAB(2001), H1TAB(2001), H2TAB(2001)].
"""
from __future__ import annotations

import torch

F = torch.float32


def _tab(v, h):
    n = h.numel()
    iv = torch.trunc(v * 200.0 + 1.5)
    iv = torch.nan_to_num(iv, nan=1.0, posinf=1e9, neginf=-1e9).clamp(-1e9, 1e9).to(torch.int64)
    return h[(iv - 1).clamp(0, n - 1)], h[(iv + 2000).clamp(0, n - 1)], h[(iv + 4001).clamp(0, n - 1)]


def _small(v, a, h):
    h0, h1, h2 = _tab(v, h)
    near = (h2 * a + h1) * a + h0
    far = (0.5642 * a) / (v * v)
    return torch.where(v > 10.0, far, near)


def _mid(v, a, h):
    h0, h1, h2 = _tab(v, h)
    vv = v * v
    hh1 = h1 + h0 * 1.12838
    hh2 = h2 + hh1 * 1.12838 - h0
    hh3 = (1.0 - h2) * .37613 - hh1 * .66667 * vv + hh2 * 1.12838
    hh4 = (3.0 * hh3 - hh1) * .37613 + h0 * .66667 * vv * vv
    poly = ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0)
    fac = ((-.122727278 * a + .532770573) * a - .96284325) * a + .979895032
    return poly * fac


def _far(v, a):
    aa = a * a
    vv = v * v
    u = (aa + vv) * 1.4142
    vo = a * .79788 / u
    aau = aa / u
    vvu = vv / u
    uu = u * u
    corr = (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa) / uu + 1.0
    return torch.where(a > 100.0, vo, corr * vo)


def voigt(v: torch.Tensor, a: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    v, a = torch.broadcast_tensors(v.to(F), a.to(F))
    shape = v.shape
    v = v.reshape(-1)
    a = a.reshape(-1)
    small = a < 0.2
    if bool(small.all()):
        return _small(v, a, h).reshape(shape)
    out = torch.empty_like(v)
    mid = ~small & (a <= 1.4) & ~(a + v > 3.2)
    far = ~small & ~mid
    for m, fn in ((small, lambda vv, aa: _small(vv, aa, h)), (mid, lambda vv, aa: _mid(vv, aa, h)), (far, _far)):
        idx = torch.nonzero(m).squeeze(1)
        if idx.numel():
            out[idx] = fn(v[idx], a[idx])
    return out.reshape(shape)
