"""DFSYNTHE set-up tables (MAIN preamble, TABVOIGT, MAP1, EXPI, VTTAB, DFINTERVALS), REAL*4 exact.

DATA tables are read from the Fortran source via atlas_torch.tables (never transcribed).  These run once per
program, on CPU with numpy float32 scalars/arrays (IEEE single, same as the Fortran), then move to the device.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...tables import table
from .f32 import Flavor, lf, lit


def _np(fl, fn, *a):
    """Apply flavour transcendental `fn` to numpy float32/64 arrays."""
    t = [torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) else x for x in a]
    return getattr(fl, fn)(*t).numpy()

SRC = "dfsynthe.for"
f4 = np.float32

# ---------------------------------------------------------------- MAIN constants (/NLINES/)
LENBUFF = 3510000
MAXPROF = 10000
LENGTH = 3507859
RESOLU = 500000.0
RATIO = 1.0 + 1.0 / RESOLU
RATIOLG = float(np.log(RATIO))
WLBEG = lf("8.97666")          # WLBEG=8.97666 : single literal stored in a REAL*8
WLEND = 10000.0
IXWLBEG = int(np.log(WLBEG) / RATIOLG)
WBEGIN = float(np.exp(IXWLBEG * RATIOLG))
if WBEGIN < WLBEG:
    IXWLBEG += 1
    WBEGIN = float(np.exp(IXWLBEG * RATIOLG))
CUTOFF = lit(".001")
VSTEPS = f4(200.0)
C17 = lf("2.99792458E17")      # single literal 2.99792458E17 (in REAL*8 or REAL*4 context)


def map1_f32(xold, fold, xnew):
    """MAP1 (dfsynthe.for, implicit REAL*4), exact port incl. its LL / forward-backward weighting logic."""
    xold = np.asarray(xold, f4); fold = np.asarray(fold, f4); xnew = np.asarray(xnew, f4)
    nold, nnew = len(xold), len(xnew)
    X = lambda i: xold[i - 1]
    F = lambda i: fold[i - 1]
    fnew = np.zeros(nnew, f4)
    L, LL = 2, 0
    a = b = c = abac = bbac = cbac = afor = bfor = cfor = f4(0)
    for k in range(nnew):
        xk = xnew[k]
        while True:                                   # label 10
            if xk < X(L):
                go = 20
                break
            L += 1
            if L > nold:
                go = 30
                break
        if go == 20:
            if L == LL:
                pass
            elif L == 2:
                go = 30
            else:
                L1 = L - 1
                if L > LL + 1 or L == 3:              # label 21
                    L2 = L - 2
                    d = (F(L1) - F(L2)) / (X(L1) - X(L2))
                    cbac = F(L) / ((X(L) - X(L1)) * (X(L) - X(L2))) + \
                        (F(L2) / (X(L) - X(L2)) - F(L1) / (X(L) - X(L1))) / (X(L1) - X(L2))
                    bbac = d - (X(L1) + X(L2)) * cbac
                    abac = F(L2) - X(L2) * d + X(L1) * X(L2) * cbac
                    to22 = not (L < nold)
                else:
                    cbac, bbac, abac = cfor, bfor, afor
                    to22 = (L == nold)
                if to22:                              # label 22
                    c, b, a = cbac, bbac, abac
                    LL = L
                else:                                 # label 25
                    d = (F(L) - F(L1)) / (X(L) - X(L1))
                    cfor = F(L + 1) / ((X(L + 1) - X(L)) * (X(L + 1) - X(L1))) + \
                        (F(L1) / (X(L + 1) - X(L1)) - F(L) / (X(L + 1) - X(L))) / (X(L) - X(L1))
                    bfor = d - (X(L) + X(L1)) * cfor
                    afor = F(L1) - X(L1) * d + X(L) * X(L1) * cfor
                    wt = f4(0)
                    if abs(cfor) != 0:
                        wt = abs(cfor) / (abs(cfor) + abs(cbac))
                    a = afor + wt * (abac - afor)
                    b = bfor + wt * (bbac - bfor)
                    c = cfor + wt * (cbac - cfor)
                    LL = L
        if go == 30 and L != LL:                      # label 30
            L = min(nold, L)
            c = f4(0)
            b = (F(L) - F(L - 1)) / (X(L) - X(L - 1))
            a = F(L) - X(L) * b
            LL = L
        fnew[k] = a + (b + c * xk) * xk               # label 50
    return fnew, LL - 1


def expi_f32(n: int, x, fl: Flavor | None = None):
    """EXPI(N,X) (REAL*4, Cody-Thacher), vectorised over x (the X1 cache is value-transparent)."""
    g = lambda v: table(SRC, "expi", v)[0] if np.ndim(table(SRC, "expi", v)) else table(SRC, "expi", v)
    A0, A1, A2, B0, B1 = (f4(g(v)) for v in ("a0", "a1", "a2", "b0", "b1"))
    C0, C1, C2, D1, D2 = (f4(g(v)) for v in ("c0", "c1", "c2", "d1", "d2"))
    E0, E1, F1 = (f4(g(v)) for v in ("e0", "e1", "f1"))
    fl = fl or Flavor()
    x = np.asarray(x, f4)
    fast = getattr(fl, "fastmath", False)
    lg = lambda v: _np(fl, "logf", v)  # noqa: E731
    ex = None
    if fast and fl.libm is None:        # stock: the scalar EXP/ALOG of EXPI are libimf expf/logf (imf.py)
        from . import imf
        ex = imf.expf((-x).astype(f4))
        if imf.logf(np.ones(1, f4)) is not None:
            lg = imf.logf  # noqa: E731
    if ex is None:
        ex = _np(fl, "expf", (-x).astype(f4))
    with np.errstate(all="ignore"):
        if fast:    # stock (expi_): R=1./X, EX1=((((E1*R)+E0)*EX)/(F1+X)+EX)*R
            r = (f4(1) / x).astype(f4)
            r10 = ((((((E1 * r).astype(f4) + E0).astype(f4) * ex).astype(f4) / (F1 + x).astype(f4)).astype(f4)
                    + ex).astype(f4) * r).astype(f4)
        else:
            r10 = (ex + ex * (E0 + E1 / x) / (x + F1)) / x
        r20 = ex * (C2 + (C1 + C0 * x) * x) / (D2 + (D1 + x) * x)
        r30 = (A0 + (A1 + A2 * x) * x) / (B0 + (B1 + x) * x) - lg(np.where(x > 0, x, f4(1)).astype(f4))
    e = np.where(x > 4, r10, np.where(x > 1, r20, np.where(x > 0, r30, f4(0)))).astype(f4)
    for i in range(1, n):
        e = ((ex - x * e) / f4(i)).astype(f4)
    return e


@dataclass
class Tables:
    h0tab: np.ndarray      # [2001]
    h1tab: np.ndarray
    h2tab: np.ndarray
    atab: np.ndarray       # [281, 1001]  (ATAB(IA,IV) -> [IA-1, IV-1])
    extab: np.ndarray      # [10001]
    e1tab: np.ndarray      # [2000]
    freq1000: np.ndarray   # [4000]
    vtprof: np.ndarray     # [120, 20]
    vtcenter: np.ndarray   # [20]
    nvt: np.ndarray        # [20]
    nbeg: np.ndarray       # [1540]
    nend: np.ndarray
    nsteps: np.ndarray     # [13, 1540]
    nn: np.ndarray


def map1_f32_fast(xold, fold, xnew):
    """MAP1 as compiled into the stock dfsynthe.exe (ifx -fp-model fast, map1_): the divisions of the label 21/25
    coefficients become rcpps + Newton steps (rcpps.py) or one reciprocal, and ABAC is regrouped:
      21: QA=F(L2)/(X(L)-X(L2)), QB=F(L1)/(X(L)-X(L1))          [rcpps+NR]
          CBAC=F(L)/((X(L)-X(L1))*(X(L)-X(L2))) + (QA-QB)/(X(L1)-X(L2))   [both rcpps+NR]
          D=(F(L1)-F(L2))/(X(L1)-X(L2)), BBAC=D-(X(L2)+X(L1))*CBAC, ABAC=(CBAC*X(L1)-D)*X(L2)+F(L2)
      25: R=1/(X(L)-X(L1)), D=(F(L)-F(L1))*R, QC=F(L1)/(X(L+1)-X(L1)), QD=F(L)/(X(L+1)-X(L))   [rcpps+NR]
          CFOR=(QC-QD)*R + F(L+1)/((X(L+1)-X(L))*(X(L+1)-X(L1))), BFOR=D-(X(L1)+X(L))*CFOR,
          AFOR=(X(L1)*X(L))*CFOR+(F(L1)-D*X(L1)); A=(ABAC-AFOR)*WT+AFOR (B, C alike)
    Used for TABVOIGT's H1TAB (checked against the table dumped from dfsynthe.exe)."""
    from .rcpps import div_nr_np
    xold = np.asarray(xold, f4); fold = np.asarray(fold, f4); xnew = np.asarray(xnew, f4)
    nold, nnew = len(xold), len(xnew)
    X = lambda i: xold[i - 1]  # noqa: E731
    F = lambda i: fold[i - 1]  # noqa: E731
    nr = lambda a, b: f4(div_nr_np(np.array([a], f4), np.array([b], f4))[0])  # noqa: E731
    fnew = np.zeros(nnew, f4)
    L, LL = 2, 0
    a = b = c = abac = bbac = cbac = afor = bfor = cfor = f4(0)
    for k in range(nnew):
        xk = xnew[k]
        while True:
            if xk < X(L):
                go = 20
                break
            L += 1
            if L > nold:
                go = 30
                break
        if go == 20:
            if L == LL:
                pass
            elif L == 2:
                go = 30
            else:
                L1 = L - 1
                if L > LL + 1 or L == 3:
                    L2 = L - 2
                    qa, qb = nr(F(L2), X(L) - X(L2)), nr(F(L1), X(L) - X(L1))
                    t1 = nr(qa - qb, X(L1) - X(L2))
                    t2 = nr(F(L), (X(L) - X(L1)) * (X(L) - X(L2)))
                    d = (F(L1) - F(L2)) / (X(L1) - X(L2))
                    cbac = t2 + t1
                    bbac = d - (X(L2) + X(L1)) * cbac
                    abac = (cbac * X(L1) - d) * X(L2) + F(L2)
                    to22 = not (L < nold)
                else:
                    cbac, bbac, abac = cfor, bfor, afor
                    to22 = (L == nold)
                if to22:
                    c, b, a = cbac, bbac, abac
                    LL = L
                else:
                    r = f4(1) / (X(L) - X(L1))
                    d = (F(L) - F(L1)) * r
                    qc, qd = nr(F(L1), X(L + 1) - X(L1)), nr(F(L), X(L + 1) - X(L))
                    cfor = (qc - qd) * r + F(L + 1) / ((X(L + 1) - X(L)) * (X(L + 1) - X(L1)))
                    bfor = d - (X(L1) + X(L)) * cfor
                    afor = (X(L1) * X(L)) * cfor + (F(L1) - d * X(L1))
                    wt = f4(0)
                    if abs(cfor) != 0:
                        wt = abs(cfor) / (abs(cbac) + abs(cfor))
                    a = (abac - afor) * wt + afor
                    b = (bbac - bfor) * wt + bfor
                    c = (cbac - cfor) * wt + cfor
                    LL = L
        if go == 30 and L != LL:
            L = min(nold, L)
            c = f4(0)
            b = (F(L) - F(L - 1)) / (X(L) - X(L - 1))
            a = F(L) - X(L) * b
            LL = L
        fnew[k] = (c * xk + b) * xk + a
    return fnew, LL - 1


def _atab_fast(h0, h1, h2):
    """TABVOIGT's ATAB loop as vectorised in the stock dfsynthe.exe (tabvoigt_): A=(IA-1)*.005, the A polynomial
    with -.122727278/200 and .79788/200 folded, IV=1..1000 four at a time with both branches evaluated
      HH2=(H2-H0)+HH1*1.12838, HH3=((H2*(-.37613)+.37613)+HH2*1.12838)+(VV*(-.66667))*HH1,
      HH4=((VV*VV)*.66667)*H0+(HH3*3-HH1)*.37613,
      R=1/U by rcpps + Newton ((1-U*r)*r+r), AAU=AA*R, VVU=VV*R,
      PB=(R*(A*.79788))*((3*R)*((AAU*(-1/3)+VVU)+((VVU*VVU)*5+(VVU*(-10)+AAU)*AAU)*R)+1),
    and IV=1001 (V=5: always the asymptotic branch) scalar with U=AA*1.4142+35.355 and R=1./U."""
    from .rcpps import recip_nr_np
    atab = np.zeros((281, 1001), f4)
    fa = np.arange(40, 281).astype(f4)[:, None]               # FLOAT(IA-1), IA=41..281
    A = (fa * f4(0.005)).astype(f4)
    AA = (A * A).astype(f4)
    k1 = (lit("-.122727278") * f4(0.005)).astype(f4)
    k5 = (lit(".79788") * f4(0.005)).astype(f4)
    polya = (((((fa * k1).astype(f4) + lit(".532770573")).astype(f4) * A).astype(f4) + lit("-.96284325")).astype(f4)
             * A).astype(f4) + lit(".979895032")
    polya = polya.astype(f4)
    a79 = (fa * k5).astype(f4)
    V = (np.arange(1000).astype(f4) * f4(0.005)).astype(f4)[None, :]
    VV = (V * V).astype(f4)
    H0, H1, H2 = h0[None, :1000], h1[None, :1000], h2[None, :1000]
    k = lit("1.12838")
    c37, c66 = lit(".37613"), lit(".66667")
    hh1 = (H0 * k + H1).astype(f4)
    hh2 = ((H2 - H0).astype(f4) + (hh1 * k).astype(f4)).astype(f4)
    hh3 = ((((H2 * (-c37)).astype(f4) + c37).astype(f4) + (hh2 * k).astype(f4)).astype(f4)
           + ((VV * (-c66)).astype(f4) * hh1).astype(f4)).astype(f4)
    hh4 = ((((VV * VV).astype(f4) * c66).astype(f4) * H0).astype(f4)
           + (((hh3 * f4(3)).astype(f4) - hh1).astype(f4) * c37).astype(f4)).astype(f4)
    hor = hh4
    for t in (hh3, hh2, hh1, H0):
        hor = ((hor * A).astype(f4) + t).astype(f4)
    pa = (hor * polya).astype(f4)
    m3 = f4(-(f4(1) / f4(3)))
    with np.errstate(all="ignore"):
        U = ((VV + AA).astype(f4) * lit("1.4142")).astype(f4)
        R = recip_nr_np(U)
        aau = (AA * R).astype(f4)
        vvu = (VV * R).astype(f4)
        t = (((vvu * f4(-10)).astype(f4) + aau).astype(f4) * aau).astype(f4)
        p1 = (((vvu * vvu).astype(f4) * f4(5)).astype(f4) + t).astype(f4)
        x = (((aau * m3).astype(f4) + vvu).astype(f4) + (p1 * R).astype(f4)).astype(f4)
        p = ((((f4(3) * R).astype(f4)) * x).astype(f4) + f4(1)).astype(f4)
        pb = (((R * a79).astype(f4)) * p).astype(f4)
    poly = ~(f4(3.2) < (A + V).astype(f4))
    atab[40:, :1000] = np.where(poly, pa, pb)
    # IV=1001 (scalar remainder)
    c25 = (f4(25) * lit("1.4142")).astype(f4)
    A1, AA1, a791 = A[:, 0], AA[:, 0], a79[:, 0]
    U1 = ((AA1 * lit("1.4142")).astype(f4) + c25).astype(f4)
    R1 = (f4(1) / U1).astype(f4)
    aau1, vvu1 = (AA1 * R1).astype(f4), (f4(25) * R1).astype(f4)
    t1 = (((vvu1 * f4(-10)).astype(f4) + aau1).astype(f4) * aau1).astype(f4)
    s1 = ((aau1 * m3).astype(f4) + vvu1).astype(f4)
    p11 = (((vvu1 * vvu1).astype(f4) * f4(5)).astype(f4) + t1).astype(f4)
    x1 = (s1 + (p11 * R1).astype(f4)).astype(f4)
    pp1 = (((x1 * f4(3)).astype(f4) * R1).astype(f4) + f4(1)).astype(f4)
    atab[40:, 1000] = (((a791 * R1).astype(f4)) * pp1).astype(f4)
    return atab


def tabvoigt(vsteps=VSTEPS, n=2001, fl: Flavor | None = None):
    fl = fl or Flavor()
    tabvi = table(SRC, "tabvoigt", "tabvi").astype(f4)
    tabh1 = table(SRC, "tabvoigt", "tabh1").astype(f4)
    i = np.arange(n, dtype=f4)
    fast = getattr(fl, "fastmath", False)
    # stock (ifx -fp-model fast): FLOAT(I-1)/VSTEPS as FLOAT(I-1)*(1./VSTEPS); EXP(-VV) through SVML (svml.py)
    v = (i * f4(f4(1.0) / f4(vsteps))).astype(f4) if fast else (i / f4(vsteps)).astype(f4)
    h1, _ = (map1_f32_fast if fast else map1_f32)(tabvi, tabh1, v)
    vv = (v * v).astype(f4)
    h0 = None
    if fast:
        from . import svml
        h0 = svml.expf((-vv).astype(f4))
    if h0 is None:
        h0 = _np(fl, "expf", (-vv).astype(f4))
    h2 = (h0 - (vv + vv) * h0).astype(f4)
    if fast:
        return h0, h1, h2, _atab_fast(h0, h1, h2)
    atab = np.zeros((281, 1001), f4)
    ia = np.arange(41, 282)
    a = ((ia - 1).astype(f4) / f4(200.0)).astype(f4)[:, None]
    aa = a * a
    iv = np.arange(1, 1002)
    v = ((iv - 1).astype(f4) / f4(200.0)).astype(f4)[None, :]
    vv = v * v
    H0, H1, H2 = h0[None, :1001], h1[None, :1001], h2[None, :1001]
    k1, k2, k3, k4 = lit("1.12838"), lit(".37613"), lit(".66667"), lit("3.")
    hh1 = H1 + H0 * k1
    hh2 = H2 + hh1 * k1 - H0
    hh3 = (f4(1) - H2) * k2 - hh1 * k3 * vv + hh2 * k1
    hh4 = (k4 * hh3 - hh1) * k2 + H0 * k3 * vv * vv
    pa = ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + H0) * \
        (((lit("-.122727278") * a + lit(".532770573")) * a - lit(".96284325")) * a + lit(".979895032"))
    with np.errstate(all="ignore"):
        u = (aa + vv) * lit("1.4142")
        aau = aa / u
        vvu = vv / u
        pb = a * lit(".79788") / u * \
            ((((aau - f4(10) * vvu) * aau + f4(5) * vvu * vvu) / u + vvu - aau / f4(3)) * f4(3) / u + f4(1))
    atab[40:] = np.where(a + v > lit("3.2"), pb, pa).astype(f4)
    return h0, h1, h2, atab


def vttab(fl: Flavor | None = None):
    """VTTAB.  The source computes X=I/V*2.99792458E5/500000., VTPROF=VTPROF/SUM.  The stock (ifx -fp-model fast)
    binary evaluates X=I*((1./V)*C) with C=2.99792458E5/500000. folded (.rodata 0x3F197E66) and normalises with
    VTPROF*(1./SUM) (dfsynthe.exe vttab_: divss 1/V, mulss C, ...; divss 1/SUM, mulps): Flavor.fastmath."""
    fl = fl or Flavor()
    fast = getattr(fl, "fastmath", False)
    vtprof = np.zeros((120, 20), f4)
    vtcenter = np.zeros(20, f4)
    nvt = np.zeros(20, np.int64)
    c = lit("2.99792458E5")
    cfold = f4(c / f4(500000.0))
    if fast:
        assert cfold.view(np.uint32) == 0x3F197E66
    for iv in range(1, 21):
        v = f4(iv)
        rv = f4(f4(f4(1.0) / v) * cfold)
        s = f4(1.0)
        last = 121
        for i in range(1, 121):
            x = f4(f4(i) * rv) if fast else f4(f4(f4(f4(i) / v) * c) / f4(500000.0))
            p = f4(_np(fl, "expf", np.array([-f4(x * x)], f4))[0])
            vtprof[i - 1, iv - 1] = p
            s = f4(s + f4(p * f4(2.0)))
            if p < lit("1.E-5"):
                last = i
                break
        nvt[iv - 1] = last
        if fast:
            vtprof[:last, iv - 1] = (vtprof[:last, iv - 1] * f4(f4(1.0) / s)).astype(f4)
        else:
            vtprof[:last, iv - 1] = (vtprof[:last, iv - 1] / s).astype(f4)
        vtcenter[iv - 1] = f4(f4(1.0) / s)
    return vtprof, vtcenter, nvt


def dfintervals():
    big = np.concatenate([table(SRC, "dfintervals", v) for v in ("biga", "bigb", "bigc")]).astype(np.float64)
    lit_ = np.concatenate([table(SRC, "dfintervals", v) for v in
                           ("lita", "litb", "litc", "litd", "lite", "litf", "litg")]).astype(np.float64)
    lr = np.log(1.0 + 1.0 / 500000.0)

    def part(w):
        n = len(w) - 1
        end = np.trunc(np.log(w[1:]) / lr).astype(np.int64)
        nzero = int(np.log(w[0]) / lr) + 1
        end = end - nzero + 1
        beg = np.empty(n, np.int64)
        beg[1:] = end[:-1] + 1
        beg[0] = 1
        return beg, end, end - beg
    bb, be, bn = part(big)
    lb, le, ln = part(lit_)
    nbeg = np.concatenate([bb, lb]); nend = np.concatenate([be, le]); nn = np.concatenate([bn, ln])
    nsteps = np.empty((13, 1540), np.int64)
    nsteps[0] = 1
    for k, m in enumerate((6, 12, 18, 24, 30, 36, 42, 48, 54, 57, 59), start=1):
        nsteps[k] = (nn * m) // 60 + 1
    nsteps[12] = nn + 1
    return nbeg, nend, nsteps, nn


_CACHE: dict = {}


def build_tables(fl: Flavor | None = None) -> Tables:
    fl = fl or Flavor()
    key = (fl.name, type(fl.libm).__name__, getattr(fl, "fastmath", False))
    if key in _CACHE:
        return _CACHE[key]
    h0, h1, h2, atab = tabvoigt(fl=fl)
    i = np.arange(1, 10002)
    extab = None
    if getattr(fl, "fastmath", False):
        # stock (MAIN DO 3456): EXP((I-1)*(-.01)) by __svml_expf4 for I=1..10000 and __svml_expf1 for I=10001
        from . import svml
        xe = ((i - 1).astype(f4) * f4(-lit(".01"))).astype(f4)
        v4, v1 = svml.expf4(xe[:10000]), svml.expf(xe[10000:])
        if v4 is not None and v1 is not None:
            extab = np.concatenate([v4, v1]).astype(f4)
    if extab is None:
        extab = _np(fl, "expf", (-((i - 1).astype(f4) * lit(".01"))).astype(f4))
    e1tab = expi_f32(1, (np.arange(1, 2001).astype(f4) * lit(".01")).astype(f4), fl)
    freq1000 = np.zeros(4000, f4)
    i = np.arange(1, 3511)
    freq1000[:3510] = (C17 / WLBEG / fl.powi8_np(RATIO, i * 1000 - 999)).astype(f4)
    vtprof, vtcenter, nvt = vttab(fl)
    nbeg, nend, nsteps, nn = dfintervals()
    _CACHE[key] = Tables(h0, h1, h2, atab, extab, e1tab, freq1000, vtprof, vtcenter, nvt, nbeg, nend, nsteps, nn)
    return _CACHE[key]


def device_tables(dev, fl: Flavor | None = None) -> dict:
    t = build_tables(fl)
    return {k: torch.from_numpy(np.ascontiguousarray(getattr(t, k))).to(dev) for k in
            ("h0tab", "h1tab", "h2tab", "atab", "extab", "e1tab", "freq1000", "vtprof", "vtcenter", "nvt")}
