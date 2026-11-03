"""Interpolation utilities shared by all variants (MAP1 is identical in atlas7v/atlas9mem/kappa9/xnfdf
up to array-size PARAMETERs; check results/fmap/variants.md before reusing in another variant)."""
from __future__ import annotations

import torch


def map1(xold: torch.Tensor, fold: torch.Tensor, xnew: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Port of FUNCTION MAP1(XOLD,FOLD,NOLD,XNEW,FNEW,NNEW): Kurucz blended-parabola interpolation.

    xold must be increasing, xnew increasing (as the Fortran assumes). Returns (fnew, MAP1 value = LL-1).
    Vectorised: the Fortran's sequential scan is replaced by searchsorted; the reuse of the previous
    interval's forward parabola as this interval's backward parabola (CBAC=CFOR) is operation-for-operation
    identical to recomputing it, so results are bit-identical.
    Works on the last dimension; leading batch dims allowed for fold (xold/xnew 1-D).
    """
    nold = xold.shape[-1]
    # L (1-based) = first index >=2 with XNEW < XOLD(L), else NOLD+1
    L = torch.searchsorted(xold, xnew, right=True) + 1     # 1-based index of first xold > xnew
    L = torch.clamp(L, min=2)
    lin = (L == 2) | (L == 3) | (L > nold)
    Lc = torch.clamp(L, max=nold)
    i = Lc - 1                                              # 0-based L
    x = xold
    f = fold

    def g(t, k):  # gather along last dim with clamped indices
        k = torch.clamp(k, 0, nold - 1)
        return t[..., k]

    xl, xl1, xl2, xl3 = g(x, i), g(x, i - 1), g(x, i - 2), g(x, i + 1)
    fl, fl1, fl2, fl3 = g(f, i), g(f, i - 1), g(f, i - 2), g(f, i + 1)
    # backward parabola through L-2, L-1, L   (label 21)
    d = (fl1 - fl2) / (xl1 - xl2)
    cbac = fl / ((xl - xl1) * (xl - xl2)) + (fl2 / (xl - xl2) - fl1 / (xl - xl1)) / (xl1 - xl2)
    bbac = d - (xl1 + xl2) * cbac
    abac = fl2 - xl2 * d + xl1 * xl2 * cbac
    # forward parabola through L-1, L, L+1   (label 25)
    d2 = (fl - fl1) / (xl - xl1)
    cfor = fl3 / ((xl3 - xl) * (xl3 - xl1)) + (fl1 / (xl3 - xl1) - fl / (xl3 - xl)) / (xl - xl1)
    bfor = d2 - (xl + xl1) * cfor
    afor = fl1 - xl1 * d2 + xl * xl1 * cfor
    wt = torch.where(torch.abs(cfor) != 0, torch.abs(cfor) / (torch.abs(cfor) + torch.abs(cbac)), torch.zeros_like(cfor))
    a = afor + wt * (abac - afor)
    b = bfor + wt * (bbac - bfor)
    c = cfor + wt * (cbac - cfor)
    last = (L == nold)  # label 22: backward parabola only
    a = torch.where(last, abac, a); b = torch.where(last, bbac, b); c = torch.where(last, cbac, c)
    # linear (label 30)
    bl = (fl - fl1) / (xl - xl1)
    al = fl - xl * bl
    a = torch.where(lin, al, a); b = torch.where(lin, bl, b); c = torch.where(lin, torch.zeros_like(c), c)
    fnew = a + (b + c * xnew) * xnew
    ll = int(L[-1].item()) if L.numel() else 0
    return fnew, min(ll, nold) - 1 if ll > nold else ll - 1
