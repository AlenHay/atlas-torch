"""JOSH and its helpers (atlas7v.for, SYNTHE variant): PARCOE, INTEG, DERIV, row-batched MAP1, W.

JOSH solves the transfer equation for one frequency on the fixed optical-depth grid XTAU8 (51 points) with
the BLOCKJ/BLOCKH Lambda-operator matrices. Here it is batched over any number of independent frequencies
("rows", leading batch dimension B; layers J are the last dimension of the batched arrays).

Precision quirks reproduced exactly:
  * The scattering solution (Gauss-Seidel on the 51-point grid) and the surface-flux quadrature run in
    REAL*4 (XS, XSBAR, XALPHA, DIAG, DELXS, ERRORX, XH, COEFJ/COEFH, CH/CK are REAL*4). The port keeps
    them float32 and accumulates every dot product sequentially in the Fortran order (M = 1..51), so the
    rounding is identical (torch float32 elementwise ops are IEEE single, no FMA contraction).
    XS(K) = MAX(XS(K)+DELXS, 1.D-38) is a REAL*8 MAX of the widened single sum, then narrowed.
  * Every single-precision literal compared against REAL*8 (.03, .001, .00001) is f32()-widened.
  * Sequential REAL*8 recurrences (INTEG's running integral, the SURFI sums) use torch.cumsum, which on
    CPU accumulates float64 strictly left to right (checked bitwise against the Fortran in the tests).
  * EXTAU (SAVEd) is computed on the first IFSURF=2 call from ANGLE; it depends on ANGLE only, so the
    port recomputes it (valid as long as ANGLE does not change within a run, true for SPECTRV).

Branches: IFSCAT in {0,1}, IFSURF in {1,2} (the SPECTRV paths), including the TAUNU(1) > XTAU8(51)
"MAXJ=1" differential branch (label 401) used at very optically thick frequencies. IFSURF=0 (J and H at
every depth) is not used by SPECTRV and raises NotImplementedError.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from ..fnum import f32
from ..state import State
from ..tables import table

SRC = "atlas7v.for"
NXTAU = 51
F32 = torch.float32
F64 = torch.float64

_ERRX = f32(0.00001)      # IF(ERRORX.GT..00001)   REAL*4 compare
_ERR401 = f32(0.00001)    # IF(ERROR.LT..00001)    REAL*8 ERROR vs widened literal
_D03 = f32(0.03)          # IF(D.LE..03)
_D001 = f32(0.001)        # IF(D.GT..001)


# ----------------------------------------------------------------------------------------------- tables

def _tables_many(keys):
    """table() for many (unit, var) keys. kurucz_data.table() re-reads its whole npz cache per call
    (~0.1 s each); when that module is available load its table dict once instead (same values)."""
    try:
        import os
        from pathlib import Path
        from .. import kurucz_data as kd
        src = Path(os.environ.get("ATLAS_EXT", str(Path.home() / "atlas-ext"))) / "builds" / "src" / SRC
        allt = kd.load_tables(src)
        return [allt[(u, v)] for u, v in keys]
    except (ImportError, AttributeError, FileNotFoundError, KeyError):
        return [table(SRC, u, v) for u, v in keys]


@lru_cache(maxsize=None)
def _np_tables():
    keys = ([("blockj", f"cj{i}") for i in range(1, 74)] + [("blockh", f"ch{i}") for i in range(1, 74)]
            + [("josh", "ch"), ("josh", "ck"), ("josh", "xtau8")])
    t = _tables_many(keys)
    cj = np.concatenate(t[:73]).astype(np.float32)       # BLOCKJ: CJ1..CJ73 EQUIVALENCEd onto /MATXJ/
    chm = np.concatenate(t[73:146]).astype(np.float32)
    return {
        "coefj": cj.reshape(NXTAU, NXTAU, order="F"),       # COMMON /MATXJ/ (BLOCKJ DATA, REAL*4)
        "coefh": chm.reshape(NXTAU, NXTAU, order="F"),      # COMMON /MATXH/ (BLOCKH DATA, REAL*4)
        "ch": np.asarray(t[146], dtype=np.float32),
        "ck": np.asarray(t[147], dtype=np.float32),
        "xtau8": np.asarray(t[148], dtype=np.float64),
    }


_TCACHE: dict = {}


def josh_tables(device=None) -> dict[str, torch.Tensor]:
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dev not in _TCACHE:
        _TCACHE[dev] = {k: torch.as_tensor(v).to(dev) for k, v in _np_tables().items()}
    return _TCACHE[dev]


def cdiv(t: torch.Tensor, c: float) -> torch.Tensor:
    """t / c with an IEEE division. (On CUDA, `tensor / python_scalar` is lowered to a multiplication by
    the reciprocal; a 0-d tensor divisor on the same device keeps the true division.)"""
    return torch.div(t, torch.tensor(c, dtype=t.dtype, device=t.device))


def rdiv(c: float, t: torch.Tensor) -> torch.Tensor:
    """c / t with an IEEE division (`python_scalar / tensor` is reciprocal(t)*c in PyTorch: two roundings)."""
    return torch.div(torch.full_like(t, c), t)


def sqrt_rn(t: torch.Tensor) -> torch.Tensor:
    """Correctly rounded SQRT (gfortran emits sqrtsd). torch.sqrt on CPU float64 is not always correctly
    rounded (vectorised ATen path); numpy's np.sqrt is the hardware instruction. CUDA's sqrt is IEEE."""
    if t.device.type == "cpu":
        return torch.from_numpy(np.sqrt(t.detach().numpy()))
    return torch.sqrt(t)


_GEXP = np.frompyfunc(math.exp, 1, 1)


def exp_glibc(t: torch.Tensor) -> torch.Tensor:
    """EXP as the Fortran's scalar libm call (Python's math.exp is glibc exp); torch.exp differs by an ulp
    on ~0.5% of arguments. ~0.1 us/element on CPU."""
    return torch.from_numpy(_GEXP(t.detach().cpu().numpy()).astype(np.float64)).to(t.device)


def seqsum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Left-to-right sum (Fortran DO-loop accumulation) of a float64 tensor along `dim`."""
    if x.shape[dim] == 0:
        return x.sum(dim)
    return torch.cumsum(x, dim).select(dim, -1)


# ----------------------------------------------------------------------------------------------- PARCOE / INTEG / DERIV

def parcoe(f: torch.Tensor, x: torch.Tensor):
    """SUBROUTINE PARCOE(F,X,A,B,C,N): blended piecewise-parabola coefficients along the last dim.

    f, x broadcastable (..., N), N >= 4 (the Fortran reads F(4) unconditionally once N > 2). Returns (A,B,C).
    """
    f, x = torch.broadcast_tensors(f, x)
    n = f.shape[-1]
    a = torch.zeros_like(f); b = torch.zeros_like(f); c = torch.zeros_like(f)
    b[..., 0] = (f[..., 1] - f[..., 0]) / (x[..., 1] - x[..., 0])
    a[..., 0] = f[..., 0] - x[..., 0] * b[..., 0]
    b[..., n - 1] = (f[..., n - 1] - f[..., n - 2]) / (x[..., n - 1] - x[..., n - 2])
    a[..., n - 1] = f[..., n - 1] - x[..., n - 1] * b[..., n - 1]
    if n == 2:
        return a, b, c
    fj, fj1, fjp = f[..., 1:-1], f[..., :-2], f[..., 2:]
    xj, xj1, xjp = x[..., 1:-1], x[..., :-2], x[..., 2:]
    d = (fj - fj1) / (xj - xj1)
    cc = fjp / ((xjp - xj) * (xjp - xj1)) - fj / ((xj - xj1) * (xjp - xj)) + fj1 / ((xj - xj1) * (xjp - xj1))
    c[..., 1:-1] = cc
    b[..., 1:-1] = d - (xj + xj1) * cc
    a[..., 1:-1] = fj1 - xj1 * d + xj * xj1 * cc
    c[..., 1] = 0.0
    b[..., 1] = (f[..., 2] - f[..., 1]) / (x[..., 2] - x[..., 1])
    a[..., 1] = f[..., 1] - x[..., 1] * b[..., 1]
    c[..., 2] = 0.0
    b[..., 2] = (f[..., 3] - f[..., 2]) / (x[..., 3] - x[..., 2])
    a[..., 2] = f[..., 2] - x[..., 2] * b[..., 2]
    # DO 2: blend with the next interval's parabola; every J reads the not-yet-blended J+1 values
    cj, cn = c[..., 1:-1], c[..., 2:]
    aj, an = a[..., 1:-1], a[..., 2:]
    bj, bn = b[..., 1:-1], b[..., 2:]
    blend = cj != 0.0
    acn = torch.abs(cn)
    wt = acn / (acn + torch.abs(cj))
    na = torch.where(blend, an + wt * (aj - an), aj)
    nb = torch.where(blend, bn + wt * (bj - bn), bj)
    nc = torch.where(blend, cn + wt * (cj - cn), cj)
    a = a.clone(); b = b.clone(); c = c.clone()
    a[..., 1:-1] = na; b[..., 1:-1] = nb; c[..., 1:-1] = nc
    a[..., n - 2] = a[..., n - 1]
    b[..., n - 2] = b[..., n - 1]
    c[..., n - 2] = c[..., n - 1]
    return a, b, c


def integ(x: torch.Tensor, f: torch.Tensor, start, return_coef: bool = False):
    """SUBROUTINE INTEG(X,F,FINT,N,START): FINT(1)=START, running integral of the PARCOE parabolas."""
    x, f = torch.broadcast_tensors(x, f)
    a, b, c = parcoe(f, x)
    x0, x1 = x[..., :-1], x[..., 1:]
    a0, b0, c0 = a[..., :-1], b[..., :-1], c[..., :-1]
    term = (a0 + cdiv(b0, 2.0) * (x1 + x0) + cdiv(c0, 3.0) * ((x1 + x0) * x1 + x0 * x0)) * (x1 - x0)
    start = torch.as_tensor(start, dtype=f.dtype, device=f.device).expand(f.shape[:-1])
    fint = torch.cumsum(torch.cat([start.unsqueeze(-1), term], dim=-1), dim=-1)
    return (fint, (a, b, c)) if return_coef else fint


def deriv(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """SUBROUTINE DERIV(X,F,DFDX,N): derivative by averaging the angles of adjacent secants."""
    x, f = torch.broadcast_tensors(x, f)
    n = f.shape[-1]
    dfdx = torch.empty_like(f)
    dfdx[..., 0] = (f[..., 1] - f[..., 0]) / (x[..., 1] - x[..., 0])
    dfdx[..., n - 1] = (f[..., n - 1] - f[..., n - 2]) / (x[..., n - 1] - x[..., n - 2])
    if n == 2:
        return dfdx
    s = (torch.abs(x[..., 1] - x[..., 0]) / (x[..., 1] - x[..., 0])).unsqueeze(-1)
    fm, fj, fp = f[..., :-2], f[..., 1:-1], f[..., 2:]
    xm, xj, xp = x[..., :-2], x[..., 1:-1], x[..., 2:]
    scale = torch.maximum(torch.maximum(torch.abs(fm), torch.abs(fj)), torch.abs(fp)) / torch.abs(xj)
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    d1 = (fp - fj) / (xp - xj) / scale
    d = (fj - fm) / (xj - xm) / scale
    tan1 = d1 / (s * sqrt_rn(1.0 + d1 * d1) + 1.0)
    tan = d / (s * sqrt_rn(1.0 + d * d) + 1.0)
    dfdx[..., 1:-1] = (tan1 + tan) / (1.0 - tan1 * tan) * scale
    return dfdx


def map1_rows(xold: torch.Tensor, fold: torch.Tensor, xnew: torch.Tensor):
    """FUNCTION MAP1 batched over rows with a per-row XOLD (atlas_torch.interp.map1 takes one XOLD).

    xold: (B, NOLD); fold: (B, NOLD) or (K, B, NOLD) (K functions on the same grid); xnew: (NNEW,) or
    (B, NNEW) increasing. Returns (fnew, MAP1) with MAP1 = LL-1 per row. Same operation sequence as the Fortran (CBAC=CFOR reuse is identical to
    recomputing the backward parabola, see interp.map1).
    """
    B, nold = xold.shape
    xn = xnew.expand(B, xnew.shape[-1]).contiguous()
    # L (1-based) = first index >= 2 with XNEW < XOLD(L), else NOLD+1 (searchsorted on XOLD(2:NOLD))
    L = torch.searchsorted(xold[:, 1:].contiguous(), xn, right=True) + 2
    # The Fortran scan never moves L backwards (L starts at 2 for every call and at MIN(L,NOLD) for the
    # next XNEW). For a non-monotonic XOLD (TAUNU can decrease where ABTOT jumps by orders of magnitude
    # between layers and the parabolic INTEG overshoots) that differs from a global search: replay it.
    nm = (xold[:, 2:] < xold[:, 1:-1]).any(dim=1)
    if bool(nm.any()):
        rows = nm.nonzero().squeeze(1)
        xo, xr = xold[rows], xn[rows]
        lidx = torch.arange(1, nold + 1, device=xold.device).unsqueeze(0)       # 1-based l
        start = torch.full((rows.numel(),), 2, dtype=torch.int64, device=xold.device)
        Ls = torch.empty((rows.numel(), xn.shape[1]), dtype=torch.int64, device=xold.device)
        for k in range(xn.shape[1]):
            ok = (lidx >= start.unsqueeze(1)) & (xr[:, k:k + 1] < xo)
            lk = torch.where(ok.any(dim=1), ok.to(torch.int8).argmax(dim=1) + 1, torch.full_like(start, nold + 1))
            Ls[:, k] = lk
            start = torch.clamp(lk, max=nold)
        L = L.clone()
        L[rows] = Ls
    lin = (L == 2) | (L == 3) | (L > nold)
    last = L == nold
    i = torch.clamp(L, max=nold) - 1
    idx = [torch.clamp(i + o, 0, nold - 1) for o in (0, -1, -2, 1)]
    xl, xl1, xl2, xl3 = (torch.gather(xold, 1, k) for k in idx)
    # XOLD-only subexpressions, shared by every FOLD (identical operations, so identical rounding)
    x01, x02, x12, x31, x30 = xl - xl1, xl - xl2, xl1 - xl2, xl3 - xl1, xl3 - xl
    q012, q301 = x01 * x02, x30 * x31
    s12, p12, s01, p01 = xl1 + xl2, xl1 * xl2, xl + xl1, xl * xl1
    f3 = fold.dim() == 3
    fo = fold if f3 else fold.unsqueeze(0)
    K = fo.shape[0]
    fl, fl1, fl2, fl3 = (torch.gather(fo, 2, k.unsqueeze(0).expand(K, -1, -1)) for k in idx)
    d = (fl1 - fl2) / x12
    cbac = fl / q012 + (fl2 / x02 - fl1 / x01) / x12
    bbac = d - s12 * cbac
    abac = fl2 - xl2 * d + p12 * cbac
    d2 = (fl - fl1) / x01                     # = the linear slope BL of label 30 as well
    cfor = fl3 / q301 + (fl1 / x31 - fl / x30) / x01
    bfor = d2 - s01 * cfor
    afor = fl1 - xl1 * d2 + p01 * cfor
    acf = torch.abs(cfor)
    wt = torch.where(acf != 0, acf / (acf + torch.abs(cbac)), torch.zeros_like(cfor))
    a = torch.where(last, abac, afor + wt * (abac - afor))
    b = torch.where(last, bbac, bfor + wt * (bbac - bfor))
    c = torch.where(last, cbac, cfor + wt * (cbac - cfor))
    al = fl - xl * d2
    a = torch.where(lin, al, a); b = torch.where(lin, d2, b); c = torch.where(lin, torch.zeros_like(c), c)
    fnew = a + (b + c * xn) * xn
    if not f3:
        fnew = fnew[0]
    ll = torch.clamp(L[:, -1], max=nold) if xn.shape[1] else torch.zeros(B, dtype=torch.int64)
    return fnew, ll - 1


# ----------------------------------------------------------------------------------------------- JOSH kernels

def _gauss_seidel(xs, xsbar, xalpha, diag, coefj):
    """DO 34 sweeps (<= 51) of DO 33 K=51..1, REAL*4, per-row convergence. xs etc. are (51, B) float32.

    Returns (xs, nsweeps[B]). Rows that converge leave the active set after the sweep that converged.
    """
    n, B = xs.shape
    nsweep = torch.zeros(B, dtype=torch.int64, device=xs.device)
    active = torch.arange(B, device=xs.device)
    xs = xs.clone()
    lo = torch.tensor(1e-38, dtype=F64, device=xs.device)
    thr = torch.tensor(_ERRX, dtype=F32, device=xs.device)
    for sweep in range(1, NXTAU + 1):
        x = xs[:, active]
        xa, xb, dg = xalpha[:, active], xsbar[:, active], diag[:, active]
        nb = x.shape[1]
        iferr = torch.zeros(nb, dtype=torch.bool, device=xs.device)
        # lower part (M <= K) of every row uses this sweep's starting XS: accumulate all rows at once
        P = torch.zeros((n, nb), dtype=F32, device=xs.device)
        for m in range(n):
            P[m:] += coefj[m:, m:m + 1] * x[m:m + 1]
        tmp = torch.empty(nb, dtype=F32, device=xs.device)
        for k in range(n - 1, -1, -1):
            d = P[k]
            ck = coefj[k]
            for m in range(k + 1, n):           # upper part: already-updated XS(M), M > K, in order
                torch.mul(x[m], ck[m], out=tmp)
                d = d + tmp
            d = (d * xa[k] + xb[k] - x[k]) / dg[k]
            iferr |= torch.abs(d / x[k]) > thr
            x[k] = torch.maximum((x[k] + d).to(F64), lo).to(F32)
        xs[:, active] = x
        nsweep[active] = sweep
        active = active[iferr]
        if active.numel() == 0:
            break
    return xs, nsweep


def _dot32(w: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """XH = 0; DO M: XH = XH + W(M)*XS(M) in REAL*4. w (51,), xs (51, B) -> (B,) float32."""
    acc = torch.zeros(xs.shape[1], dtype=F32, device=xs.device)
    for m in range(xs.shape[0]):
        acc = acc + w[m] * xs[m]
    return acc


def _surfi_grid(xs8, xtau8, angle):
    """Label 670: SURFI(MU) from the parabolic fit of XS8 on XTAU8 (EXTAU = exp(-XTAU8/ANGLE))."""
    a, b, c = parcoe(xs8, xtau8)
    n1 = NXTAU - 1
    ctwo = c * 2.0
    b2ct = b + ctwo * xtau8
    b2ct1 = b[:, :n1] + ctwo[:, :n1] * xtau8[1:]
    out = []
    for ang in angle.tolist():
        tangle = cdiv(xtau8, ang)
        extau = torch.where(tangle < 300.0, exp_glibc(-tangle), torch.zeros_like(tangle))
        t1 = extau[:n1] * (xs8[:, :n1] + (b2ct[:, :n1] + ctwo[:, :n1] * ang) * ang)
        t2 = extau[1:] * (xs8[:, 1:] + (b2ct1 + ctwo[:, :n1] * ang) * ang)
        zero = (extau[:n1] == 0.0).nonzero()
        jstop = int(zero[0]) if zero.numel() else n1
        terms = torch.stack([t1[:, :jstop], -t2[:, :jstop]], dim=-1).reshape(xs8.shape[0], 2 * jstop)
        s = seqsum(terms) if jstop else torch.zeros(xs8.shape[0], dtype=F64, device=xs8.device)
        if jstop == n1:
            s = s + extau[n1] * (xs8[:, n1] + (b2ct[:, n1] + ctwo[:, n1] * ang) * ang)
        out.append(s)
    return torch.stack(out, dim=-1)


def _surfi_depth(snu, taunu, angle):
    """Label 70: SURFI(MU) by integrating the parabolic fit of SNU(TAUNU) (series for thin intervals)."""
    a, b, c = parcoe(snu, taunu)
    B, n = snu.shape
    n1 = n - 1
    ctwo = c * 2.0
    b2ct = b + ctwo * taunu
    b2ct1 = b[:, :n1] + ctwo[:, :n1] * taunu[:, 1:]
    s0, s1 = snu[:, :n1], snu[:, 1:]
    bj, cj = b2ct[:, :n1], ctwo[:, :n1]
    out = []
    for ang in angle.tolist():
        tangle = cdiv(taunu[:, 1:], ang)
        new = exp_glibc(-tangle)
        old = torch.cat([torch.ones_like(new[:, :1]), new[:, :-1]], dim=1)
        d = tangle - cdiv(taunu[:, :n1], ang)
        thick = ~(d <= _D03)
        ta = old * (s0 + (bj + cj * ang) * ang)
        tb = new * (s1 + (b2ct1 + cj * ang) * ang)
        dd = cdiv(d, 9.0) + 1.0                 # ((((D/9.+1.)*D/8.+1.)*D/7.+1.)*D/6.+1.)*D/5.+1.
        for k in (8.0, 7.0, 6.0, 5.0):
            dd = cdiv(dd * d, k) + 1.0
        dd = torch.where(d > _D001, dd, torch.ones_like(d))
        inner = s0 + (bj + cj * ang) * ang
        ser = new * (s0 + cdiv((s0 + bj * ang + cdiv(inner * (cdiv(dd * d, 4.0) + 1.0) * d, 3.0)) * d, 2.0)) * d
        first = torch.where(thick, ta, ser)
        second = torch.where(thick, -tb, torch.zeros_like(tb))
        csum = torch.cumsum(torch.stack([first, second], dim=-1).reshape(B, 2 * n1), dim=1)
        stop = thick & ~(tangle < 300.0)
        has = stop.any(dim=1)
        jstop = torch.where(has, stop.to(torch.int8).argmax(dim=1), torch.full((B,), n1 - 1, device=snu.device))
        s_stop = csum.gather(1, (2 * jstop + 1).unsqueeze(1)).squeeze(1)
        oldn = new[:, -1]
        s_end = csum[:, -1] + oldn * (snu[:, n1] + (b2ct[:, n1] + ctwo[:, n1] * ang) * ang)
        out.append(torch.where(has, s_stop, s_end))
    return torch.stack(out, dim=-1)


def _lambda_401(taunu, snubar, alpha):
    """Labels 401-45 with MAXJ=1: iterate S = (1-alpha) Sbar + alpha (S + d/dtau(d/dtau S /3)) by DERIV."""
    snu = snubar.clone()
    hnu = torch.zeros_like(snu); jmins = torch.zeros_like(snu); jnu = torch.zeros_like(snu)
    act = torch.ones(snu.shape[0], dtype=torch.bool, device=snu.device)
    niter = torch.zeros(snu.shape[0], dtype=torch.int64, device=snu.device)
    for it in range(1, NXTAU + 1):
        h = cdiv(deriv(taunu, snu), 3.0)
        jm = deriv(taunu, h)
        jn = jm + snu
        snew = (1.0 - alpha) * snubar + alpha * jn
        err = seqsum(torch.abs(snew - snu) / snew)
        a = act.unsqueeze(1)
        hnu = torch.where(a, h, hnu); jmins = torch.where(a, jm, jmins); jnu = torch.where(a, jn, jnu)
        snu = torch.where(a, snew, snu)
        niter = torch.where(act, torch.full_like(niter, it), niter)
        act = act & ~(err < _ERR401)
        if not bool(act.any()):
            break
    return {"snu": snu, "hnu": hnu, "jmins": jmins, "jnu": jnu, "niter": niter}


def josh_batch(acont, scont, aline, sline, sigmac, sigmal, rhox, ifscat: int, ifsurf: int,
               angle=None, detail: bool = False) -> dict[str, torch.Tensor]:
    """SUBROUTINE JOSH(IFSCAT,IFSURF) for B independent frequencies.

    acont..sigmal: (B, NRHOX) float64 (COMMON /OPTOT/ for each frequency); rhox: (NRHOX,) or (B, NRHOX);
    angle: (NMU,) cosines (COMMON /MUS/) for IFSURF=2. Returns "hnu1" (B,) = HNU(1) for IFSURF=1,
    "surfi" (B, NMU) for IFSURF=2, plus ABTOT/ALPHA/TAUNU (and the REAL*4 internals if detail=True).
    """
    if ifsurf not in (1, 2):
        raise NotImplementedError("JOSH with IFSURF=0 (J and H at all depths) is not on the SPECTRV path")
    dev = acont.device
    T = josh_tables(dev)
    xtau8 = T["xtau8"]
    B, n = acont.shape
    rhox = rhox.expand(B, n)
    abtot = acont + aline + sigmac + sigmal
    alpha = (sigmac + sigmal) / abtot
    snubar = (acont * scont + aline * sline) / (acont + aline)
    taunu = integ(rhox, abtot, abtot[:, 0] * rhox[:, 0])
    out = {"abtot": abtot, "alpha": alpha, "taunu": taunu, "snubar": snubar}
    hnu1 = torch.zeros(B, dtype=F64, device=dev)
    angle_t = None
    if ifsurf == 2:
        angle_t = torch.as_tensor(angle, dtype=F64, device=dev).reshape(-1)
        surfi = torch.zeros((B, angle_t.numel()), dtype=F64, device=dev)

    if ifscat == 0:
        snu = snubar
        out["snu"] = snu
        if ifsurf == 2:
            out["surfi"] = _surfi_depth(snu, taunu, angle_t)
            return out
        xs8, _ = map1_rows(taunu, snu, xtau8)
        xs = xs8.to(F32).T.contiguous()
        out["hnu1"] = _dot32(T["ch"], xs).to(F64)
        if detail:
            out.update(xs=xs.T, xs8=xs8)
        return out

    thick = taunu[:, 0] > xtau8[NXTAU - 1]              # MAXJ=1 -> label 401
    rg = (~thick).nonzero().squeeze(1)
    rt = thick.nonzero().squeeze(1)
    if rg.numel():
        tn, sb, al = taunu[rg], snubar[rg], alpha[rg]
        both, _ = map1_rows(tn, torch.stack([sb, al]), xtau8)
        xsbar8, xalpha8 = both[0], both[1]
        lo = torch.tensor(1e-38, dtype=F64, device=dev)
        xalpha = torch.clamp(xalpha8, 0.0, 1.0).to(F32)
        xsbar = torch.maximum(xsbar8, lo).to(F32)
        low = xtau8.unsqueeze(0) < tn[:, :1]
        xsbar = torch.where(low, torch.maximum(sb[:, :1], lo).to(F32), xsbar)
        xalpha = torch.where(low, torch.clamp(al[:, :1], 0.0, 1.0).to(F32), xalpha)
        xsbar, xalpha = xsbar.T.contiguous(), xalpha.T.contiguous()
        xs0 = xsbar.clone()
        cjj = torch.diagonal(T["coefj"]).unsqueeze(1)
        diag = 1.0 - xalpha * cjj
        xsbar_s = (1.0 - xalpha) * xsbar
        xs, nsweep = _gauss_seidel(xs0, xsbar_s, xalpha, diag, T["coefj"])
        if ifsurf == 1:
            hnu1[rg] = _dot32(T["ch"], xs).to(F64)
        else:
            surfi[rg] = _surfi_grid(xs.T.to(F64), xtau8, angle_t)
        if detail:
            out.update(rows_grid=rg, xs=xs.T, xsbar=xsbar_s.T, xalpha=xalpha.T, diag=diag.T,
                       xsbar8=xsbar8, xalpha8=xalpha8, nsweep=nsweep)
    if rt.numel():
        r = _lambda_401(taunu[rt], snubar[rt], alpha[rt])
        if ifsurf == 1:
            hnu1[rt] = r["hnu"][:, 0]
        else:
            surfi[rt] = _surfi_depth(r["snu"], taunu[rt], angle_t)
        out["rows_thick"] = rt
        out["thick"] = r
    if ifsurf == 1:
        out["hnu1"] = hnu1
    else:
        out["surfi"] = surfi
    return out


def josh(s: State, ifscat: int, ifsurf: int) -> dict[str, torch.Tensor]:
    """State-level JOSH for one frequency: returns the COMMON entries it writes (Fortran shapes)."""
    n = s.iscalar("rhox.nrhox")
    g = lambda k: s[k][:n].to(F64).unsqueeze(0)
    nmu = s.iscalar("mus.nmu")
    r = josh_batch(g("optot.acont"), g("optot.scont"), g("optot.aline"), g("optot.sline"), g("optot.sigmac"),
                   g("optot.sigmal"), s["rhox.rhox"][:n].to(F64), ifscat, ifsurf,
                   angle=s["mus.angle"][:nmu], detail=True)
    ch = {}
    for key, name in (("abtot.abtot", "abtot"), ("abtot.alpha", "alpha"), ("taushj.taunu", "taunu")):
        v = s[key].clone(); v[:n] = r[name][0]; ch[key] = v
    if ifscat == 0 or "thick" in r:
        v = s["taushj.snu"].clone()
        v[:n] = r["snu"][0] if ifscat == 0 else r["thick"]["snu"][0]
        ch["taushj.snu"] = v
    if "thick" in r:
        th = r["thick"]
        for key, name in (("taushj.hnu", "hnu"), ("taushj.jmins", "jmins"), ("taushj.jnu", "jnu")):
            v = s[key].clone(); v[:n] = th[name][0]; ch[key] = v
        if ifsurf == 1:
            v = s["pzero.knu"].clone(); v[0] = cdiv(th["jnu"][0, 0], 3.0); ch["pzero.knu"] = v
    elif ifsurf == 1:
        v = s["taushj.hnu"].clone(); v[0] = r["hnu1"][0]; ch["taushj.hnu"] = v
    if ifsurf == 2:
        v = s["mus.surfi"].clone(); v[:nmu] = r["surfi"][0]; ch["mus.surfi"] = v
    ch["_detail"] = r
    return ch


# ----------------------------------------------------------------------------------------------- W

def _e12_4(v: float) -> str:
    """One item of FORMAT 1PE12.4 (gfortran): d.ddddE+ee, 3-digit exponents drop the 'E'."""
    if v != v:
        return f"{'NaN':>12}"
    if v in (float("inf"), float("-inf")):
        return f"{('-' if v < 0 else '') + 'Infinity':>12}"
    m, e = f"{v:.4e}".split("e")
    ei = int(e)
    if v == 0.0:
        ei = 0
    es = f"E{ei:+03d}" if abs(ei) <= 99 else f"{ei:+04d}"
    return f"{m + es:>12}"


def w(label: str, b, n: int | None = None) -> str:
    """SUBROUTINE W(A,B,N): WRITE(6,'(1H0,A6,1P10E12.4/(7X,10E12.4))') A,(B(I),I=1,N). Returns the lines."""
    vals = [float(x) for x in (b.tolist() if hasattr(b, "tolist") else b)]
    if n is not None:
        vals = vals[:n]
    lab = (label + " " * 6)[:6]
    lines = ["0" + lab + "".join(_e12_4(v) for v in vals[:10])]
    for i in range(10, len(vals), 10):
        lines.append(" " * 7 + "".join(_e12_4(v) for v in vals[i:i + 10]))
    if len(vals) == 10:      # the '/' after the 10th item is still processed: an empty record (7X writes nothing)
        lines.append("")
    return "\n".join(lines) + "\n"
