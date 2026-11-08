"""Line wings on the DFSYNTHE grid: (layer, line) pairs, their wing extents and the deposition of their profiles
on a block of grid points (dfsynthe.for MAIN loops 1321-1326 for the packed lists, XLINOP labels 200-214 for the
NLTE list: a wing is added out to and including its first point below 10^-3 of the local continuum)."""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .grid import DF_GRID

LADDER_LINEAR = 16          # first ladder offsets 0..15, then geometric
LADDER_RATIO = 1.1


@dataclass(frozen=True)
class WingPairs:
    """One entry per (layer, line) pair, every tensor (P,): layer; center (DF_GRID index); wl0 (nm, the line's
    wavelength); k0 (cm^2 g^-1, centre opacity with H(a, 0) = 1, including stimulated emission); a (damping);
    dop (Delta nu_D / nu); red0, red (red-wing offsets red0..red, the centre is offset 0); blue (offsets 1..blue);
    wcon, wtail (nm; series taper (XLINOP NCON > 0), 0 for none)."""
    layer: torch.Tensor
    center: torch.Tensor
    wl0: torch.Tensor
    k0: torch.Tensor
    a: torch.Tensor
    dop: torch.Tensor
    red0: torch.Tensor
    red: torch.Tensor
    blue: torch.Tensor
    wcon: torch.Tensor
    wtail: torch.Tensor

    def __len__(self) -> int:
        return self.layer.shape[0]

    def select(self, idx) -> "WingPairs":
        return WingPairs(*(getattr(self, f.name)[idx] for f in fields(self)))

    def replace(self, **kw) -> "WingPairs":
        return WingPairs(**{f.name: kw.get(f.name, getattr(self, f.name)) for f in fields(self)})


def grid_wavelength(index: torch.Tensor) -> torch.Tensor:
    """lambda (nm) of DF_GRID points (0-based index)."""
    return torch.exp((DF_GRID.first + index.to(torch.float64)) * DF_GRID.step)


def ragged(n0: torch.Tensor, n1: torch.Tensor, budget: int):
    """Ragged integer ranges n0[i]..n1[i] (inclusive) as blocks (item index, n) of at most `budget` points."""
    cnt = (n1 - n0 + 1).clamp(min=0).long()
    csum = torch.cumsum(cnt, 0)
    total = int(csum[-1]) if len(csum) else 0
    start = 0
    while start < total:
        i0 = int(torch.searchsorted(csum, torch.tensor(start, device=csum.device), right=True))
        i1 = int(torch.searchsorted(csum, torch.tensor(start + budget, device=csum.device), right=True))
        i1 = max(i1, i0 + 1)
        c = cnt[i0:i1]
        item = torch.repeat_interleave(torch.arange(i0, i1, device=c.device), c)
        first = torch.cumsum(c, 0) - c
        off = torch.arange(item.shape[0], device=c.device) - first[item - i0]
        yield item, n0[item].long() + off
        start = int(csum[i1 - 1])


def deposit(pairs: WingPairs, lo: int, hi: int, out: torch.Tensor, value, budget: int) -> int:
    """out (L, hi - lo + 1) += value(pairs, item, index) at the grid points index in [lo, hi] of each pair's wings
    (red: center + red0..red, blue: center - 1..blue). Returns the number of points evaluated."""
    W, count = out.shape[1], 0
    c = pairs.center
    for side in (1, -1):
        if side > 0:
            n0 = torch.maximum(pairs.red0, lo - c)
            n1 = torch.minimum(pairs.red, hi - c)
        else:
            n0 = torch.maximum(torch.ones_like(c), c - hi)
            n1 = torch.minimum(pairs.blue, c - lo)
        keep = (n1 >= n0).nonzero().squeeze(1)
        for item, n in ragged(n0[keep], n1[keep], budget):
            it = keep[item]
            idx = c[it] + side * n
            out.view(-1).index_add_(0, pairs.layer[it] * W + (idx - lo), value(pairs, it, idx))
            count += int(n.shape[0])
    return count


def _ladder(n_max: int) -> list[int]:
    out, x = list(range(LADDER_LINEAR)), float(LADDER_LINEAR)
    while out[-1] < n_max:
        out.append(min(int(x), n_max))
        x *= LADDER_RATIO
    return out


@torch.no_grad()
def first_below(ratio, n_min: torch.Tensor, n_max: torch.Tensor) -> torch.Tensor:
    """For each item i the smallest n in [n_min[i], n_max[i]] with ratio(items, n) < 1, n_max where there is none
    (n_min > n_max: n_min - 1, nothing to deposit). ratio(items (k,), n (k,)) -> (k,) is evaluated on a geometric
    ladder of offsets from n_min, then the crossing is bisected between the last ladder point above 1 and the first
    below (assumes no second crossing between ladder points; the wings decrease monotonically, the continuum is
    smooth between edges)."""
    out = n_max.clone()
    empty = n_min > n_max
    out[empty] = n_min[empty] - 1
    todo = (~empty).nonzero().squeeze(1)
    lo_b = torch.zeros_like(n_min)
    hi_b = torch.zeros_like(n_min)
    found = torch.zeros_like(empty)
    prev = n_min - 1
    span = int((n_max - n_min).clamp(min=0).max()) if len(n_min) else 0
    for off in _ladder(span):
        if not len(todo):
            break
        n = torch.minimum(n_min[todo] + off, n_max[todo])
        below = ratio(todo, n) < 1.0
        hit = todo[below]
        lo_b[hit], hi_b[hit], found[hit] = prev[hit], n[below], True
        prev[todo] = n
        todo = todo[~below & (n < n_max[todo])]
    todo = (found & ((hi_b - lo_b) > 1)).nonzero().squeeze(1)
    while len(todo):
        mid = (lo_b[todo] + hi_b[todo]) // 2
        below = ratio(todo, mid) < 1.0
        hi_b[todo[below]] = mid[below]
        lo_b[todo[~below]] = mid[~below]
        todo = todo[(hi_b[todo] - lo_b[todo]) > 1]
    out[found] = hi_b[found]
    return out


def first_below_from(ratio, guess: torch.Tensor, n_min: torch.Tensor, n_max: torch.Tensor) -> torch.Tensor:
    """first_below with the ladder started at 85 % of an estimate `guess` of the crossing; items already below
    there are searched again from n_min."""
    start = torch.maximum(n_min, torch.minimum(torch.floor(0.85 * guess).long(), n_max))
    out = first_below(ratio, start, n_max)
    redo = ((out == start) & (start > n_min)).nonzero().squeeze(1)
    if len(redo):
        out[redo] = first_below(lambda it, n: ratio(redo[it], n), n_min[redo], start[redo])
    return out


__all__ = ["WingPairs", "deposit", "first_below", "first_below_from", "ragged", "grid_wavelength"]
