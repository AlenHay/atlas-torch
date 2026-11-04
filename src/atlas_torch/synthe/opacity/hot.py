"""HOTOP and ELECOP (atlas7v.for).

HOTOP: free-free of C,N,O,Ne,Mg,Si,S,Fe I-V (COULFF-weighted sum of Z^2 n_ion) + 60 bound-free edges of
C, N, O, Ne ions. ITEMP cache (ITEMP1, XNFP(kw,21), XNFSUMQQ(kw,5)) explicit as save.hotop.*; when ITEMP
changes the populations come from a `pops(s, code, mode)` callback (POPS is ported elsewhere).
"""
from __future__ import annotations

import numpy as np
import torch

from ...state import State
from ._core import Ctx, exp, itemp_tensor, powi, put_rows, rdiv, save_get, save_itemp1, sqrt, table
from .hydrogen import coulff_k

HOT_XNFP = [(6.03, 0), (7.04, 4), (8.05, 9), (10.05, 15)]          # POPS(code,11,XNFP(1,col+1))


def hot_saves(s: State):
    return (save_itemp1(s, "hotop"), save_get(s, "hotop", "xnfp", (99, 21)),
            save_get(s, "hotop", "xnfsumqq", (99, 5)))


def hotop_cache(c: Ctx, itemp1: int, xnfp: torch.Tensor, xnfsumqq: torch.Tensor, pops=None):
    if c.itemp() == itemp1:
        return itemp1, xnfp, xnfsumqq
    if pops is None:
        raise RuntimeError("HOTOP: ITEMP changed, populations needed: pass pops(s, code, mode) "
                           "or provide save.hotop.* with save.hotop.itemp1 = ITEMP")
    n = c.n
    xnfp = xnfp.clone()
    for code, col in HOT_XNFP:
        p = torch.as_tensor(pops(c.s, code, 11), dtype=torch.float64).reshape(99, -1)
        k = min(p.shape[1], 21 - col)
        xnfp[:n, col:col + k] = p[:n, :k]
    q = xnfsumqq.clone()
    q[:n] = 0.0
    for code in table("hotop", "code8"):
        xnf = torch.as_tensor(pops(c.s, float(code), 12), dtype=torch.float64).reshape(99, -1)
        for iz in range(1, 6):
            q[:n, iz - 1] = q[:n, iz - 1] + float(iz * iz) * xnf[:n, iz]
    return c.itemp(), xnfp, q


_A = None


def _a():
    global _A
    if _A is None:
        _A = np.asarray(table("hotop", "a"), dtype=np.float64)
    return _A


def hotop_k(c: Ctx, xnfp: torch.Tensor, xnfsumqq: torch.Tensor) -> torch.Tensor:
    """SUBROUTINE HOTOP after its ITEMP block -> AHOT [NF,n]."""
    n = c.n
    q = xnfsumqq[:n].T.reshape(5, 1, n)
    free = coulff_k(c, 1) * q[0]
    for k in range(2, 6):
        free = free + coulff_k(c, k) * q[k - 1]
    ahot = free * 3.6919e8 / powi(c.freq, 3) * c.lay("state.xne") / sqrt(c.lay("temp.t"))
    A = _a()
    tkev = c.lay("temp.tkev")
    num = int(table("hotop", "num"))
    for i in range(num):
        a0, a1, a2, a3, a4, a5, a6 = A[7 * i:7 * i + 7]
        on = ~(c.freq < a0)
        if not bool(on.any()):
            continue
        r = rdiv(a0, c.freq)
        xsect = a1 * (a2 + r - a2 * r) * sqrt(powi(r, int(a3)))
        xx = xsect * xnfp[:n, int(a6) - 1].reshape(1, n) * a4
        upd = on & (xx > ahot / 100.0)
        ahot = torch.where(upd, ahot + xx * exp(rdiv(-a5, tkev)), ahot)
    return ahot * c.stim / c.lay("state.rho")


def hotop(s: State, pops=None) -> dict:
    c = Ctx.from_state(s)
    it1, xnfp, q = hotop_cache(c, *hot_saves(s), pops=pops)
    return {"ops.ahot": put_rows(s["ops.ahot"], hotop_k(c, xnfp, q), c.n),
            "save.hotop.itemp1": itemp_tensor(it1), "save.hotop.xnfp": xnfp, "save.hotop.xnfsumqq": q}


def elecop_k(c: Ctx) -> torch.Tensor:
    """SUBROUTINE ELECOP: Thomson scattering SIGEL = .6653D-24*XNE/RHO -> [1,n]."""
    return 0.6653e-24 * c.lay("state.xne") / c.lay("state.rho")


def elecop(s: State) -> dict:
    c = Ctx.from_state(s)
    return {"ops.sigel": put_rows(s["ops.sigel"], elecop_k(c), c.n)}
