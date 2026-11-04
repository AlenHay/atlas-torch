"""DERIV, INTEG, PARCOE, EXPI and a batched MAP1 (atlas9mem.for), vectorised over the last dimension.

All functions accept leading batch dimensions (e.g. [model, layer] or [frequency, layer]); the Fortran's
sequential loops are reproduced operation-for-operation:
  * PARCOE's smoothing loop reads A(J+1),B(J+1),C(J+1) before they are themselves smoothed, so it is a
    vectorised expression of the unsmoothed coefficients;
  * INTEG's running sum FINT(I+1)=FINT(I)+... is a torch.cumsum in float64 (sequential accumulation on CPU,
    hence bit-identical to the Fortran loop; on GPU the scan order may differ in the last ulp).
"""
from __future__ import annotations

import torch

from . import data, fmath


def deriv(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """SUBROUTINE DERIV(X,F,DFDX,N): derivative with the Kurucz arctangent-averaging of the two one-sided slopes."""
    n = x.shape[-1]
    x = x.expand_as(f) if x.shape != f.shape else x
    out = torch.empty_like(f)
    out[..., 0] = (f[..., 1] - f[..., 0]) / (x[..., 1] - x[..., 0])
    out[..., n - 1] = (f[..., n - 1] - f[..., n - 2]) / (x[..., n - 1] - x[..., n - 2])
    if n == 2:
        return out
    s = torch.abs(x[..., 1] - x[..., 0]) / (x[..., 1] - x[..., 0])
    s = s.unsqueeze(-1)
    fm, f0, fp = f[..., :-2], f[..., 1:-1], f[..., 2:]
    xm, x0, xp = x[..., :-2], x[..., 1:-1], x[..., 2:]
    scale = torch.maximum(torch.maximum(torch.abs(fm), torch.abs(f0)), torch.abs(fp)) / torch.abs(x0)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    d1 = (fp - f0) / (xp - x0) / scale
    d = (f0 - fm) / (x0 - xm) / scale
    tan1 = d1 / (s * fmath.sqrt(1.0 + d1 * d1) + 1.0)
    tan = d / (s * fmath.sqrt(1.0 + d * d) + 1.0)
    out[..., 1:-1] = (tan1 + tan) / (1.0 - tan1 * tan) * scale
    return out


def parcoe(f: torch.Tensor, x: torch.Tensor):
    """SUBROUTINE PARCOE(F,X,A,B,C,N) -> (A, B, C) blended parabola coefficients. Requires N >= 4 (or N == 2)."""
    n = f.shape[-1]
    x = x.expand_as(f) if x.shape != f.shape else x
    a = torch.zeros_like(f); b = torch.zeros_like(f); c = torch.zeros_like(f)
    b[..., 0] = (f[..., 1] - f[..., 0]) / (x[..., 1] - x[..., 0])
    a[..., 0] = f[..., 0] - x[..., 0] * b[..., 0]
    b[..., n - 1] = (f[..., n - 1] - f[..., n - 2]) / (x[..., n - 1] - x[..., n - 2])
    a[..., n - 1] = f[..., n - 1] - x[..., n - 1] * b[..., n - 1]
    if n == 2:
        return a, b, c
    assert n >= 4, "PARCOE indexes F(4) for N >= 3"
    # loop 1, J=2..N-1 (0-based j=1..n-2): J1=J-1
    fj1, fj, fjp = f[..., :-2], f[..., 1:-1], f[..., 2:]
    xj1, xj, xjp = x[..., :-2], x[..., 1:-1], x[..., 2:]
    d = (fj - fj1) / (xj - xj1)
    cc = fjp / ((xjp - xj) * (xjp - xj1)) - fj / ((xj - xj1) * (xjp - xj)) + fj1 / ((xj - xj1) * (xjp - xj1))
    c[..., 1:-1] = cc
    b[..., 1:-1] = d - (xj + xj1) * cc
    a[..., 1:-1] = fj1 - xj1 * d + xj * xj1 * cc
    # C(2)=C(3)=0 with linear pieces
    for j in (1, 2):
        c[..., j] = 0.0
        b[..., j] = (f[..., j + 1] - f[..., j]) / (x[..., j + 1] - x[..., j])
        a[..., j] = f[..., j] - x[..., j] * b[..., j]
    # loop 2, J=2..N-1: blend with J+1 (unsmoothed values of J+1)
    a0, b0, c0 = a.clone(), b.clone(), c.clone()
    cj, cn = c0[..., 1:-1], c0[..., 2:]
    wt = torch.abs(cn) / (torch.abs(cn) + torch.abs(cj))
    upd = cj != 0
    a[..., 1:-1] = torch.where(upd, a0[..., 2:] + wt * (a0[..., 1:-1] - a0[..., 2:]), a0[..., 1:-1])
    b[..., 1:-1] = torch.where(upd, b0[..., 2:] + wt * (b0[..., 1:-1] - b0[..., 2:]), b0[..., 1:-1])
    c[..., 1:-1] = torch.where(upd, c0[..., 2:] + wt * (c0[..., 1:-1] - c0[..., 2:]), c0[..., 1:-1])
    a[..., n - 2] = a[..., n - 1]
    b[..., n - 2] = b[..., n - 1]
    c[..., n - 2] = c[..., n - 1]
    return a, b, c


def integ(x: torch.Tensor, f: torch.Tensor, start) -> torch.Tensor:
    """SUBROUTINE INTEG(X,F,FINT,N,START): integral of the PARCOE parabolas, FINT(1)=START."""
    x = x.expand_as(f) if x.shape != f.shape else x
    a, b, c = parcoe(f, x)
    x0, x1 = x[..., :-1], x[..., 1:]
    a, b, c = a[..., :-1], b[..., :-1], c[..., :-1]
    inc = (a + b / 2.0 * (x1 + x0) + c / 3.0 * ((x1 + x0) * x1 + x0 * x0)) * (x1 - x0)
    lead = f.shape[:-1]
    st = torch.as_tensor(start, dtype=f.dtype, device=f.device)
    if st.dim() == len(lead):
        st = st.unsqueeze(-1)
    st = st.expand(lead + (1,))
    return torch.cumsum(torch.cat([st, inc], dim=-1), dim=-1)


def expi(n: int, x: torch.Tensor) -> torch.Tensor:
    """FUNCTION EXPI(N,X): exponential integral E_n(x), x > 0 (Cody & Thacher). The Fortran's one-entry cache
    (X.EQ.X1 -> reuse) returns the same values and is therefore not modelled."""
    k = data.expi_coeffs()
    ex = fmath.exp(-x)
    xs = torch.where(x > 0, x, torch.ones_like(x))   # keep the unused branches finite
    e = k
    r10 = (ex + ex * (e["e0"] + (e["e1"] + (e["e2"] + (e["e3"] + (e["e4"] + (e["e5"] + fmath.rdiv(e["e6"], xs)) / xs) / xs)
                                                     / xs) / xs) / xs)
           / (xs + e["f1"] + (e["f2"] + (e["f3"] + (e["f4"] + (e["f5"] + fmath.rdiv(e["f6"], xs)) / xs) / xs) / xs) / xs)) / xs
    r20 = ex * (e["c6"] + (e["c5"] + (e["c4"] + (e["c3"] + (e["c2"] + (e["c1"] + e["c0"] * xs) * xs) * xs) * xs)
                           * xs) * xs) / \
        (e["d6"] + (e["d5"] + (e["d4"] + (e["d3"] + (e["d2"] + (e["d1"] + xs) * xs) * xs) * xs) * xs) * xs)
    r30 = (e["a0"] + (e["a1"] + (e["a2"] + (e["a3"] + (e["a4"] + e["a5"] * xs) * xs) * xs) * xs) * xs) / \
        (e["b0"] + (e["b1"] + (e["b2"] + (e["b3"] + (e["b4"] + xs) * xs) * xs) * xs) * xs) - fmath.log(xs)
    ex1 = torch.where(x > 4.0, r10, torch.where(x > 1.0, r20, torch.where(x > 0.0, r30, torch.zeros_like(x))))
    r = ex1
    for i in range(1, n):
        r = (ex - x * r) / float(i)
    return r


def map1b(xold: torch.Tensor, fold: torch.Tensor, xnew: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FUNCTION MAP1 batched over leading dims: xold[..., nold], fold[..., nold], xnew[..., nnew] (each row
    increasing). Same arithmetic as atlas_torch.interp.map1. Returns (fnew, MAP1 value per row = LL-1)."""
    nold = xold.shape[-1]
    lead = torch.broadcast_shapes(xold.shape[:-1], fold.shape[:-1], xnew.shape[:-1])
    xo = xold.expand(lead + xold.shape[-1:]).contiguous()
    fo = fold.expand(lead + fold.shape[-1:])
    xn = xnew.expand(lead + xnew.shape[-1:]).contiguous()
    L = torch.searchsorted(xo, xn, right=True) + 1
    L = torch.clamp(L, min=2)
    lin = (L == 2) | (L == 3) | (L > nold)
    i = torch.clamp(L, max=nold) - 1

    def g(t, k):
        return torch.gather(t, -1, torch.clamp(k, 0, nold - 1))

    xl, xl1, xl2, xl3 = g(xo, i), g(xo, i - 1), g(xo, i - 2), g(xo, i + 1)
    fl, fl1, fl2, fl3 = g(fo, i), g(fo, i - 1), g(fo, i - 2), g(fo, i + 1)
    d = (fl1 - fl2) / (xl1 - xl2)
    cbac = fl / ((xl - xl1) * (xl - xl2)) + (fl2 / (xl - xl2) - fl1 / (xl - xl1)) / (xl1 - xl2)
    bbac = d - (xl1 + xl2) * cbac
    abac = fl2 - xl2 * d + xl1 * xl2 * cbac
    d2 = (fl - fl1) / (xl - xl1)
    cfor = fl3 / ((xl3 - xl) * (xl3 - xl1)) + (fl1 / (xl3 - xl1) - fl / (xl3 - xl)) / (xl - xl1)
    bfor = d2 - (xl + xl1) * cfor
    afor = fl1 - xl1 * d2 + xl * xl1 * cfor
    wt = torch.where(torch.abs(cfor) != 0, torch.abs(cfor) / (torch.abs(cfor) + torch.abs(cbac)), torch.zeros_like(cfor))
    a = afor + wt * (abac - afor)
    b = bfor + wt * (bbac - bfor)
    c = cfor + wt * (cbac - cfor)
    last = L == nold
    a = torch.where(last, abac, a); b = torch.where(last, bbac, b); c = torch.where(last, cbac, c)
    bl = (fl - fl1) / (xl - xl1)
    al = fl - xl * bl
    a = torch.where(lin, al, a); b = torch.where(lin, bl, b); c = torch.where(lin, torch.zeros_like(c), c)
    fnew = a + (b + c * xn) * xn
    ll = torch.clamp(L[..., -1], max=nold)
    return fnew, ll - 1


def map1_shared(xold: torch.Tensor, fold: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
    """MAP1 of many curves on one shared abscissa at one point each: xold[nold], fold[..., R, nold] (leading
    dims broadcast against xnew's), xnew[...] -> fnew[..., R]. Same arithmetic as map1b (NNEW=1 calls)."""
    nold = xold.shape[-1]
    L = torch.clamp(torch.searchsorted(xold, xnew.reshape(-1).contiguous(), right=True) + 1, min=2).reshape(xnew.shape)
    lin = ((L == 2) | (L == 3) | (L > nold)).unsqueeze(-1)
    last = (L == nold).unsqueeze(-1)
    i = torch.clamp(L, max=nold) - 1
    xn = xnew.unsqueeze(-1)

    def gx(k):
        return xold[torch.clamp(k, 0, nold - 1)].unsqueeze(-1)

    def gf(k):
        k = torch.clamp(k, 0, nold - 1)
        idx = k.reshape(k.shape + (1, 1)).expand(k.shape + (fold.shape[-2], 1))
        ff = fold.expand(k.shape + fold.shape[-2:])
        return torch.gather(ff, -1, idx)[..., 0]

    xl, xl1, xl2, xl3 = gx(i), gx(i - 1), gx(i - 2), gx(i + 1)
    fl, fl1, fl2, fl3 = gf(i), gf(i - 1), gf(i - 2), gf(i + 1)
    d = (fl1 - fl2) / (xl1 - xl2)
    cbac = fl / ((xl - xl1) * (xl - xl2)) + (fl2 / (xl - xl2) - fl1 / (xl - xl1)) / (xl1 - xl2)
    bbac = d - (xl1 + xl2) * cbac
    abac = fl2 - xl2 * d + xl1 * xl2 * cbac
    d2 = (fl - fl1) / (xl - xl1)
    cfor = fl3 / ((xl3 - xl) * (xl3 - xl1)) + (fl1 / (xl3 - xl1) - fl / (xl3 - xl)) / (xl - xl1)
    bfor = d2 - (xl + xl1) * cfor
    afor = fl1 - xl1 * d2 + xl * xl1 * cfor
    wt = torch.where(torch.abs(cfor) != 0, torch.abs(cfor) / (torch.abs(cfor) + torch.abs(cbac)), torch.zeros_like(cfor))
    a = afor + wt * (abac - afor)
    b = bfor + wt * (bbac - bfor)
    c = cfor + wt * (cbac - cfor)
    a = torch.where(last, abac, a); b = torch.where(last, bbac, b); c = torch.where(last, cbac, c)
    bl = (fl - fl1) / (xl - xl1)
    al = fl - xl * bl
    a = torch.where(lin, al, a); b = torch.where(lin, bl, b); c = torch.where(lin, torch.zeros_like(c), c)
    return a + (b + c * xn) * xn
