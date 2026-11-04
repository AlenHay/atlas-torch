"""JOSH (atlas9mem.for): monochromatic transfer with coherent scattering, batched over independent problems
(frequencies x opacity-sampling steps) -- leading dimension B, layers last.

Algorithm (IFSCAT=1, IFSURF=0, the ATLAS9 iteration case; IFSCAT=0 is the same with ALPHA=0):
  ABTOT, ALPHA, SNUBAR; TAUNU = INTEG(RHOX, ABTOT);
  interpolate SNUBAR, ALPHA onto the fixed 51-point XTAU grid (MAP1) -> REAL*4 XSBAR, XALPHA;
  Gauss-Seidel sweeps K=51..1 with the REAL*4 matrix COEFJ (<= 2*NXTAU sweeps, until no |dS/S| > 1e-5);
  map S back to TAUNU(1..MAXJ); Lambda iteration with DERIV below XTAU(51)=20 (MAXJ..NRHOX);
  JMINS, HNU on the XTAU grid from COEFJ/COEFH, KNU(1) from CK; JNU = JMINS + SNU.

Precision is the Fortran's: XS, XSBAR, XALPHA, DIAG, DELXS, XJS, XH, XK are REAL*4 and every dot product is a
sequential REAL*4 accumulation (reproduced exactly with an in-place float32 loop; numpy on CPU for speed).
Quirks: MAXJ = MAP1(...) = LL-1 <= NRHOX-1; TAUNU(1)>20 -> MAXJ=1 (no matrix solve, KNU(1)=JNU(1)/3);
the Lambda loop stops at the first SNEW==0.
"""
from __future__ import annotations

import numpy as np
import torch

from ..fnum import f32
from . import data
from .numerics import deriv, integ, map1b

_TOL = np.float32(f32(0.00001))


def _gs_numpy(xs, xsbar, xalpha, diag, coefj, active, nsweep):
    """Gauss-Seidel on float32 numpy arrays [NX, B] (in place on xs). Returns xs."""
    nx = coefj.shape[0]
    act = active.copy()
    prods = np.empty_like(xs)
    acc = np.empty(xs.shape[1], dtype=np.float32)
    for _ in range(nsweep):
        if not act.any():
            break
        iferr = np.zeros_like(act)
        for k in range(nx - 1, -1, -1):
            np.multiply(coefj[k][:, None], xs, out=prods)
            acc[:] = 0
            for m in range(nx):
                acc += prods[m]
            delxs = (acc * xalpha[k] + xsbar[k] - xs[k]) / diag[k]
            with np.errstate(all="ignore"):
                errorx = np.abs(delxs / xs[k])
            iferr |= errorx > _TOL
            new = np.maximum((xs[k] + delxs).astype(np.float64), 1e-38).astype(np.float32)
            xs[k] = np.where(act, new, xs[k])
        act &= iferr
    return xs


def _matvec_seq(coef, xs, init=None):
    """Y(L) = init(L) + sum_M COEF(L,M)*XS(M), sequential REAL*4 accumulation over M. coef [NL,NM], xs [NM,B]."""
    nl, nm = coef.shape
    acc = np.zeros((nl, xs.shape[1]), dtype=np.float32) if init is None else init.copy()
    for m in range(nm):
        acc += coef[:, m][:, None] * xs[m][None, :]
    return acc


def josh(acont, scont, aline, sline, sigmac, sigmal, rhox, ifscat: int = 1, ifsurf: int = 0) -> dict:
    """All opacity inputs [B, n] float64; rhox [n]. Returns dict of [B, n] tensors abtot, alpha, taunu, snu,
    hnu, jnu, jmins and knu1 [B]."""
    if ifsurf != 0:
        raise NotImplementedError("JOSH with IFSURF=1/2 (surface flux/intensity runs) is not ported")
    T = data.josh_tables()
    xtau8 = torch.from_numpy(T["xtau8"]).to(acont.device)
    nx = xtau8.shape[0]
    B, n = acont.shape
    abtot = ((acont + aline) + sigmac) + sigmal
    alpha = (sigmac + sigmal) / abtot
    snubar = (acont * scont + aline * sline) / (acont + aline)
    taunu = integ(rhox, abtot, abtot[:, 0] * rhox[0])
    if ifscat == 0:
        alpha = torch.zeros_like(alpha)
    deep = taunu[:, 0] > xtau8[-1]
    xsbar8, maxj = map1b(taunu, snubar, xtau8)
    xalpha8, _ = map1b(taunu, alpha, xtau8)
    maxj = torch.where(deep, torch.ones_like(maxj), maxj)
    # REAL*4 conversions (MAX in double, then stored)
    xalpha = torch.clamp(xalpha8, min=0.0)
    xsbar = torch.maximum(xsbar8, torch.tensor(1e-38, dtype=torch.float64))
    below = xtau8.unsqueeze(0) < taunu[:, :1]
    xsbar = torch.where(below, torch.maximum(snubar[:, :1], torch.tensor(1e-38, dtype=torch.float64)), xsbar)
    xalpha = torch.where(below, torch.clamp(alpha[:, :1], min=0.0), xalpha)
    xalpha = xalpha.to(torch.float32).cpu().numpy().T.copy()          # [NX, B]
    xsbar = xsbar.to(torch.float32).cpu().numpy().T.copy()
    coefj = T["coefj"]
    xs = xsbar.copy()
    diag = (np.float32(1.0) - xalpha * np.diag(coefj)[:, None]).astype(np.float32)
    xsbar = ((np.float32(1.0) - xalpha) * xsbar).astype(np.float32)
    xs = _gs_numpy(xs, xsbar, xalpha, diag, coefj, (~deep).cpu().numpy(), 2 * nx)
    dev = acont.device
    xs8 = torch.from_numpy(xs.T.astype(np.float64)).to(dev)            # [B, NX]
    # S on TAUNU(1..MAXJ)
    snu_m, _ = map1b(xtau8, xs8, taunu)
    jidx = torch.arange(n, device=dev).unsqueeze(0)                     # 0-based J-1
    mj = maxj.unsqueeze(1)                                              # MAXJ (1-based count)
    snu = torch.where(jidx < mj, snu_m, snubar)
    # label 401: SNU(J)=SNUBAR(J) for J>=MAXJ1 (MAXJ1=MAXJ+1, or 1 if MAXJ==1)
    maxj1 = torch.where(maxj == 1, torch.ones_like(maxj), maxj + 1)
    snu = torch.where(jidx >= (maxj1 - 1).unsqueeze(1), snubar, snu)
    m0 = torch.clamp(maxj - 1, min=1) - 1                              # 0-based start M
    hnu = torch.zeros_like(snu)
    jmins = torch.zeros_like(snu)
    jnu = torch.zeros_like(snu)
    running = torch.ones(B, dtype=torch.bool, device=dev)
    tail = jidx >= (maxj1 - 1).unsqueeze(1)                             # J >= MAXJ1
    for _ in range(nx):
        if not running.any():
            break
        h = _deriv_from(taunu, snu, m0) / 3.0
        jm = _deriv_from(taunu, h, maxj - 1)
        jn = jm + snu
        snew = (1.0 - alpha) * snubar + alpha * jn
        zero = tail & (snew == 0)
        first0 = torch.where(zero.any(1), torch.argmax(zero.to(torch.int8), dim=1), torch.full_like(maxj, n))
        upto = jidx < first0.unsqueeze(1)                               # J processed before a SNEW==0
        term = torch.where(tail & upto, torch.abs(snew - snu) / snew, torch.zeros_like(snew))
        error = torch.cumsum(term, dim=1)[:, -1]
        r = running.unsqueeze(1)
        hnu = torch.where(r & (jidx >= m0.unsqueeze(1)), h, hnu)
        jmins = torch.where(r & (jidx >= (maxj - 1).unsqueeze(1)), jm, jmins)
        jnu = torch.where(r & tail & (jidx <= first0.unsqueeze(1)), jn, jnu)
        snu = torch.where(r & tail & upto, snew, snu)
        stop = zero.any(1) | (error < f32(0.00001))
        running = running & ~stop
    # label 50 (MAXJ > 1): JMINS, HNU on the XTAU grid -> TAUNU(1..MAXJ); KNU(1)
    coefh = T["coefh"]
    xjs = _matvec_seq(coefj, xs, init=-xs)
    xh = _matvec_seq(coefh, xs)
    xk = _matvec_seq(T["ck"][None, :], xs)[0]
    xjs8 = torch.from_numpy(xjs.T.astype(np.float64)).to(dev)
    xh8 = torch.from_numpy(xh.T.astype(np.float64)).to(dev)
    jm_m, _ = map1b(xtau8, xjs8, taunu)
    h_m, _ = map1b(xtau8, xh8, taunu)
    low = (jidx < mj) & (maxj != 1).unsqueeze(1)
    jmins = torch.where(low, jm_m, jmins)
    hnu = torch.where(low, h_m, hnu)
    jnu = torch.where(low, jmins + snu, jnu)
    knu1 = torch.where(maxj == 1, jnu[:, 0] / 3.0, torch.from_numpy(xk.astype(np.float64)).to(dev))
    return {"abtot": abtot, "alpha": alpha, "taunu": taunu, "snu": snu, "hnu": hnu, "jnu": jnu,
            "jmins": jmins, "knu1": knu1, "maxj": maxj}


def _deriv_from(x: torch.Tensor, f: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """DERIV(X(M),F(M),DFDX(M),N-M+1) for a per-row 0-based start M; entries before M are unspecified."""
    out = deriv(x, f)
    B, n = f.shape
    rows = torch.arange(B, device=f.device)
    st = torch.clamp(start, max=n - 2)
    one = (f[rows, st + 1] - f[rows, st]) / (x[rows, st + 1] - x[rows, st])
    out[rows, st] = one
    # interior points use S = sign(X(M+1)-X(M)) of the sub-array; recompute where it differs from the full one
    s_full = torch.sign(x[:, 1] - x[:, 0])
    s_sub = torch.sign(x[rows, st + 1] - x[rows, st])
    if bool((s_full != s_sub).any()):
        raise NotImplementedError("non-monotonic TAUNU in DERIV sub-array")
    if n - 1 - st.min() < 2:
        pass
    return out
