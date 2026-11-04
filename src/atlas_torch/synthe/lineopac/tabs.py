"""SYNTHE's run-time tables: COMMON /EXTAB/ (EXTAB, EXTABF, E1TAB from EXPI) and COMMON /H1TAB/
(TABVOIGT: H0TAB, H1TAB, H2TAB), plus the FASTEX statement function.  All REAL*4.

`mode="stock"` reproduces the gfortran -O3 binary (vectorised loops call libmvec, see libm32);
`mode="scalar"` reproduces the -O0 oracle library (scalar libm everywhere).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ...tables import table
from . import libm32 as L

SRC = "synthe.patched.for"
F32 = np.float32


def map1_f32(xold, fold, xnew):
    """synthe.for MAP1 (REAL*4), scalar transcription (used on small tables only)."""
    xold = np.asarray(xold, F32)
    fold = np.asarray(fold, F32)
    xnew = np.asarray(xnew, F32)
    nold = len(xold)
    fnew = np.zeros(len(xnew), F32)
    L_ = 2
    LL = 0
    A = B = C = F32(0)
    afor = bfor = cfor = abac = bbac = cbac = F32(0)
    for k in range(len(xnew)):
        xk = xnew[k]
        while True:  # label 10
            if xk < xold[L_ - 1]:
                go = 20
                break
            L_ += 1
            if L_ > nold:
                go = 30
                break
        if go == 20:
            if L_ == LL:
                fnew[k] = A + (B + C * xk) * xk
                continue
            if L_ == 2:
                go = 30
        if go == 20:
            L1 = L_ - 1
            fl = lambda i: fold[i - 1]  # noqa: E731
            xo = lambda i: xold[i - 1]  # noqa: E731
            if L_ > LL + 1 or L_ == 3:
                L2 = L_ - 2
                D = (fl(L1) - fl(L2)) / (xo(L1) - xo(L2))
                cbac = fl(L_) / ((xo(L_) - xo(L1)) * (xo(L_) - xo(L2))) + \
                    (fl(L2) / (xo(L_) - xo(L2)) - fl(L1) / (xo(L_) - xo(L1))) / (xo(L1) - xo(L2))
                bbac = D - (xo(L1) + xo(L2)) * cbac
                abac = fl(L2) - xo(L2) * D + xo(L1) * xo(L2) * cbac
                if L_ >= nold:
                    C, B, A = cbac, bbac, abac
                    LL = L_
                    fnew[k] = A + (B + C * xk) * xk
                    continue
            else:
                cbac, bbac, abac = cfor, bfor, afor
                if L_ == nold:
                    C, B, A = cbac, bbac, abac
                    LL = L_
                    fnew[k] = A + (B + C * xk) * xk
                    continue
            D = (fl(L_) - fl(L1)) / (xo(L_) - xo(L1))
            cfor = fl(L_ + 1) / ((xo(L_ + 1) - xo(L_)) * (xo(L_ + 1) - xo(L1))) + \
                (fl(L1) / (xo(L_ + 1) - xo(L1)) - fl(L_) / (xo(L_ + 1) - xo(L_))) / (xo(L_) - xo(L1))
            bfor = D - (xo(L_) + xo(L1)) * cfor
            afor = fl(L1) - xo(L1) * D + xo(L_) * xo(L1) * cfor
            wt = F32(0)
            if abs(cfor) != 0:
                wt = abs(cfor) / (abs(cfor) + abs(cbac))
            A = afor + wt * (abac - afor)
            B = bfor + wt * (bbac - bfor)
            C = cfor + wt * (cbac - cfor)
            LL = L_
            fnew[k] = A + (B + C * xk) * xk
            continue
        # label 30
        if L_ != LL:
            L_ = min(nold, L_)
            C = F32(0)
            B = (fold[L_ - 1] - fold[L_ - 2]) / (xold[L_ - 1] - xold[L_ - 2])
            A = fold[L_ - 1] - xold[L_ - 1] * B
            LL = L_
        fnew[k] = A + (B + C * xk) * xk
    return fnew


def expi1(x: np.ndarray) -> np.ndarray:
    """EXPI(1,X) (REAL*4), elementwise (X1/EX caching is irrelevant for distinct arguments)."""
    g = lambda v: F32(table(SRC, "expi", v))  # noqa: E731
    A0, A1, A2, B0, B1 = (g(v) for v in ("a0", "a1", "a2", "b0", "b1"))
    C0, C1, C2, D1, D2 = (g(v) for v in ("c0", "c1", "c2", "d1", "d2"))
    E0, E1, F1 = (g(v) for v in ("e0", "e1", "f1"))
    x = np.asarray(x, F32)
    ex = L.sexpf(torch.from_numpy(-x)).numpy()
    lg = L.slogf(torch.from_numpy(np.where(x > 0, x, F32(1)))).numpy()
    with np.errstate(all="ignore"):
        r10 = (ex + ex * (E0 + E1 / x) / (x + F1)) / x
        r20 = ex * (C2 + (C1 + C0 * x) * x) / (D2 + (D1 + x) * x)
        r30 = (A0 + (A1 + A2 * x) * x) / (B0 + (B1 + x) * x) - lg
    out = np.where(x > 4, r10, np.where(x > 1, r20, np.where(x > 0, r30, F32(0))))
    return out.astype(F32)


@lru_cache(maxsize=None)
def extab_common(mode: str = "stock") -> np.ndarray:
    """COMMON /EXTAB/ as one float32 vector [EXTAB(1001), EXTABF(1001), E1TAB(2000)]."""
    i = np.arange(1001, dtype=F32)
    a = -i
    b = -(i * F32(.001))
    if mode == "stock":
        fv, fs = L.vexpf, L.sexpf
    else:
        fv = fs = L.sexpf
    # gcc peels iteration I=1 (EXP(0.)=1 folded) and vectorises the remaining 1000 = 250*4 iterations
    ext = np.concatenate([[F32(1)], L.loopvec(fv, fs, torch.from_numpy(a[1:])).numpy()])
    extf = np.concatenate([[F32(1)], L.loopvec(fv, fs, torch.from_numpy(b[1:])).numpy()])
    e1 = expi1(np.arange(1, 2001, dtype=F32) * F32(.01))
    return np.concatenate([ext, extf, e1]).astype(F32)


@lru_cache(maxsize=None)
def h1tab_common(mode: str = "stock", vsteps: float = 200.0, n: int = 2001) -> np.ndarray:
    """COMMON /H1TAB/ after CALL TABVOIGT(200., 2001): [H0TAB, H1TAB, H2TAB] float32."""
    vs = F32(vsteps)
    x = np.arange(n, dtype=F32) / vs
    h1 = map1_f32(table(SRC, "tabvoigt", "tabvi"), table(SRC, "tabvoigt", "tabh1"), x)
    vv = x * x
    if mode == "stock":
        h0 = L.loopvec(L.vexpf, L.sexpf, torch.from_numpy(-vv)).numpy()
    else:
        h0 = L.sexpf(torch.from_numpy(-vv)).numpy()
    h2 = h0 - (vv + vv) * h0
    out = np.zeros(3 * 2001, F32)
    out[0:n] = h0
    out[2001:2001 + n] = h1
    out[4002:4002 + n] = h2
    return out


def fastex(x: torch.Tensor, extab: torch.Tensor) -> torch.Tensor:
    """FASTEX(X)=EXTAB(IFIX(X)+1)*EXTABF(IFIX((X-FLOAT(IFIX(X)))*1000.+1.5)), X REAL*4 >= 0.

    `extab` is the flat COMMON /EXTAB/ (out-of-range EXTAB indices run into EXTABF/E1TAB as in memory)."""
    x = x.to(torch.float32)
    ix = torch.trunc(x)
    i1 = ix.to(torch.int64)                       # EXTAB(i1+1) -> flat index i1
    fr = (x - ix) * 1000.0
    fr = fr + 1.5
    i2 = torch.trunc(fr).to(torch.int64)          # EXTABF(i2) -> flat index 1001 + i2 - 1
    n = extab.numel()
    return extab[i1.clamp(0, n - 1)] * extab[(1000 + i2).clamp(0, n - 1)]
