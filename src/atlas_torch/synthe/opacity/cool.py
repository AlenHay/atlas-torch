"""COOLOP and its callees (atlas7v.for): C1OP, MG1OP, AL1OP, SI1OP, FE1OP, CHOP, OHOP, H2COLLOP.

Fortran side effects reproduced:
  * H2COLLOP always sets COMMON WAVENO = FREQ/2.99792458D10 (so every routine after COOLOP in KAPP sees
    that value, not xnfpelsyn's 1.E7/WAVE); CHOP and OHOP do the same on a cache miss (FREQ.NE.FREQ1).
  * FE1OP writes BSI1(J,1) into COMMON /FLUX/ FLXRAD (EQUIVALENCE (FLXRAD(1),BFUDGE(1))).
SAVEd caches: H2COLLOP (ITEMP1, XNH2) is explicit. The FREQ1 caches of CHOP/OHOP hold pure functions of
FREQ (CROSSCHT, N), so a hit returns what recomputation gives; only their WAVENO write depends on FREQ1.
"""
from __future__ import annotations

import numpy as np
import torch

from ...state import State
from ._core import (Ctx, acc, chain, exp, f32, itemp_tensor, pow_, powi, put_rows, rdiv, save_get, save_itemp1,
                    scalar1, table)


def _bf(x, g, e, hckt, b1, b2, ehvkt):
    """A = X*g*EXP(-E*HCKT)*(B1-B2*EHVKT) (e=None: the `*1.` ground-level form)."""
    if e is None:
        return x * g * 1.0 * (b1 - b2 * ehvkt)
    return x * g * exp(-e * hckt) * (b1 - b2 * ehvkt)


class _Acc:
    """H/S accumulators of a bound-free chain with departure coefficients B1(J,lvl)/B2(J,1)."""

    def __init__(self, c: Ctx, blk1: str, blk2: str, thresholds):
        self.c = c
        self.b1 = lambda lvl: c.lay(blk1, lvl)                     # noqa: E731
        self.b2 = c.lay(blk2, 1)
        self.hckt = c.lay("temp.hckt")
        self.h = c.full(1.0e-30)
        self.s = c.zeros()
        self.m = chain(c.waveno, thresholds)

    def any(self, k):
        return bool(self.m[k].any())

    def add(self, k, a, lvl):
        c = self.c
        self.h = acc(self.m[k], self.h, a)
        self.s = acc(self.m[k], self.s, a * c.bnu * c.stim / (self.b1(lvl) / self.b2 - c.ehvkt))

    def bf(self, x, g, e, lvl):
        return _bf(x, g, e, self.hckt, self.b1(lvl), self.b2, self.c.ehvkt)


# ============================================================================================== C1OP
_C1_THR = [16886.790, 18251.980, 19487.800, 20118.750, 21140.700, 22006.370, 28880.880, 30489.700, 58601.270,
           69172.400, 69235.820, 69767.350, 80627.760, 80691.180, 90777.000, 90804.000, 90820.420, 90840.420,
           90867.420, 90883.840, 100121.000]


def c1op_k(c: Ctx, sc1_prev):
    """SUBROUTINE C1OP: C I bound-free (Luo & Pradhan / Burke & Taylor fits) -> (AC1, SC1)."""
    ryd = 109732.298
    w = c.waveno
    q = _Acc(c, "bc.bc1", "bc.bc2", _C1_THR)
    zero = torch.zeros_like(w)
    for k, (g, e, lvl) in enumerate([(1.0, 73975.91, 13), (5.0, 72610.72, 12), (9.0, 71374.90, 11),
                                     (3.0, 70743.95, 10), (15.0, 69722.00, 9)]):
        q.add(k, q.bf(zero, g, e, lvl), lvl)
    q.add(5, q.bf(2.1e-18 * pow_(rdiv(22006.370, w), 1.5), 3.0, 68856.33, 8), 8)
    q.add(6, q.bf(1.54e-18 * pow_(rdiv(28880.880, w), f32(1.2)), 3.0, 61981.82, 6), 6)
    q.add(7, q.bf(0.2e-18 * pow_(rdiv(30489.700, w), f32(1.2)), 9.0, 60373.00, 5), 5)
    q.add(8, q.bf(zero, 9.0, 75254.93, 14), 14)
    # P2 1S (3)
    x = pow_(10.0, -16.80 - (w - 69172.400) / 3.0 / ryd)
    eps = (w - 97700.0) * 2.0 / 2743.0
    x = x + (68.0e-18 * eps + 118.0e-18) / (eps * eps + 1.0)
    x = x / 3.0
    a = q.bf(x, 1.0, 21648.02, 3)
    q.add(9, a, 3)
    a = a * 2.0
    q.add(10, a, 3)
    q.add(11, q.bf(16.0e-18 * powi(rdiv(69767.350, w), 3), 15.0, 64088.85, 7), 7)
    # P2 1D (2)
    x = pow_(10.0, -16.80 - (w - 80627.760) / 3.0 / ryd)
    eps = (w - 93917.0) * 2.0 / 9230.0
    x = x + (22.0e-18 * eps + 26.0e-18) / (eps * eps + 1.0)
    eps = (w - 111130.0) * 2.0 / 2743.0
    x = x + (-10.5e-18 * eps + 46.0e-18) / (eps * eps + 1.0)
    x = x / 3.0
    a = q.bf(x, 5.0, 10192.66, 2)
    q.add(12, a, 2)
    a = a * 2.0
    q.add(13, a, 2)
    # P2 3P (1)
    x = pow_(10.0, -16.80 - (w - 90777.000) / 3.0 / ryd)
    x = x / 3.0
    q.add(14, q.bf(x, 5.0, 43.42, 1), 1)
    q.add(15, q.bf(x, 3.0, 16.42, 1), 1)
    q.add(16, q.bf(x, 1.0, None, 1), 1)
    x = x * 2.0
    q.add(17, q.bf(x, 5.0, 43.42, 1), 1)
    q.add(18, q.bf(x, 3.0, 16.42, 1), 1)
    q.add(19, q.bf(x, 1.0, None, 1), 1)
    q.add(20, q.bf(1.0e-18 * powi(rdiv(100121.000, w), 3), 5.0, 33735.20, 4), 4)
    xr = c.lay("bc.xnfpc", 1)
    rho = c.lay("state.rho")
    h = q.h * xr / rho
    s = q.s * xr / rho
    return h, torch.where(h > 0.0, s / h, sc1_prev)


def c1op(s: State) -> dict:
    c = Ctx.from_state(s)
    a, sc = c1op_k(c, c.lay("bc.sc1"))
    return {"bc.ac1": put_rows(s["bc.ac1"], a, c.n), "bc.sc1": put_rows(s["bc.sc1"], sc, c.n)}


# ============================================================================================= MG1OP
_MG1_THR = [13713.986, 13823.223, 15267.955, 18167.687, 20473.617, 26619.756, 39759.842, 39800.556, 39820.615,
            61671.020]


def mg1op_k(c: Ctx, smg1_prev):
    """SUBROUTINE MG1OP: Mg I bound-free -> (AMG1, SMG1)."""
    w = c.waveno
    q = _Acc(c, "bmg.bmg1", "bmg.bmg2", _MG1_THR)
    for k, (thr, a0, p, g, e, lvl) in enumerate([
            (13713.986, 25.0e-18, f32(2.7), 15.0, 47957.034, 8), (13823.223, 33.8e-18, f32(2.8), 9.0, 47847.797, 7),
            (15267.955, 45.0e-18, f32(2.7), 5.0, 46403.065, 6), (18167.687, 0.43e-18, f32(2.6), 1.0, 43503.333, 5),
            (20473.617, 2.1e-18, f32(2.6), 3.0, 41197.043, 4)]):
        q.add(k, q.bf(a0 * pow_(rdiv(thr, w), p), g, e, lvl), lvl)
    r = rdiv(26619.756, w)
    q.add(5, q.bf(16.0e-18 * pow_(r, f32(2.1)) - 7.8e-18 * pow_(r, 9.5), 3.0, 35051.264, 3), 3)
    r = rdiv(39759.842, w)
    x = 20.0e-18 * pow_(r, f32(2.7))
    x = torch.maximum(x, 40.0e-18 * powi(r, 14))
    q.add(6, q.bf(x, 5.0, 21911.178, 2), 2)
    q.add(7, q.bf(x, 3.0, 21870.464, 2), 2)
    q.add(8, q.bf(x, 1.0, 21850.405, 2), 2)
    q.add(9, q.bf(1.1e-18 * powi(rdiv(61671.020, w), 10), 1.0, None, 1), 1)
    smg1 = torch.where(q.h > 0.0, q.s / q.h, smg1_prev)
    return q.h * c.lay("bmg.xnfpmg", 1) / c.lay("state.rho"), smg1


def mg1op(s: State) -> dict:
    c = Ctx.from_state(s)
    a, sm = mg1op_k(c, c.lay("bmg.smg1"))
    return {"bmg.amg1": put_rows(s["bmg.amg1"], a, c.n), "bmg.smg1": put_rows(s["bmg.smg1"], sm, c.n)}


# ============================================================================================= AL1OP
_AL1_THR = [6958.993, 8002.467, 9346.231, 10588.957, 15318.007, 15842.129, 22930.614, 48166.309, 48278.370,
            55903.260]


def al1op_k(c: Ctx, sal1_prev):
    """SUBROUTINE AL1OP: Al I bound-free -> (AAL1, SAL1)."""
    w = c.waveno
    q = _Acc(c, "bal.bal1", "bal.bal2", _AL1_THR)
    q.add(0, q.bf(torch.zeros_like(w), 14.0, 41319.377, 9), 9)
    q.add(1, q.bf(50.0e-18 * powi(rdiv(8002.467, w), 3), 6.0, 40275.903, 8), 8)
    q.add(2, q.bf(50.0e-18 * powi(rdiv(9346.231, w), 3), 10.0, 38932.139, 7), 7)
    q.add(3, q.bf(56.7e-18 * pow_(rdiv(10588.957, w), f32(1.9)), 2.0, 37689.413, 6), 6)
    q.add(4, q.bf(rdiv(14.5e-18 * 15318.007, w), 6.0, 32960.363, 5), 5)
    q.add(5, q.bf(47.0e-18 * pow_(rdiv(15842.129, w), f32(1.83)), 10.0, 32436.241, 4), 4)
    r = rdiv(22930.614, w)
    q.add(6, q.bf(10.0e-18 * (r * r), 2.0, 25347.756, 2), 2)
    x = 65.0e-18 * powi(rdiv(48166.309, w), 5)
    q.add(7, q.bf(x, 4.0, 112.061, 1), 1)
    q.add(8, q.bf(x, 2.0, None, 1), 1)
    r = rdiv(55903.260, w)
    q.add(9, q.bf(10.0e-18 * (r * r), 12.0, 29097.11, 3), 3)
    sal1 = torch.where(q.h > 0.0, q.s / q.h, sal1_prev)
    return q.h * c.lay("bal.xnfpal", 1) / c.lay("state.rho"), sal1


def al1op(s: State) -> dict:
    c = Ctx.from_state(s)
    a, sa = al1op_k(c, c.lay("bal.sal1"))
    return {"bal.aal1": put_rows(s["bal.aal1"], a, c.n), "bal.sal1": put_rows(s["bal.sal1"], sa, c.n)}


# ============================================================================================= SI1OP
_SI1_THR = [16810.969, 17777.641, 18587.546, 18655.039, 24947.216, 26079.180, 50353.180, 50640.630, 59448.700,
            59736.150, 63446.510, 65524.393, 65670.435, 65747.550, 65811.843, 65957.885, 66035.000, 75423.767]


def si1op_k(c: Ctx):
    """SUBROUTINE SI1OP: Si I bound-free (Nahar & Pradhan resonance fits) -> (ASI1, SSI1)."""
    w = c.waveno
    q = _Acc(c, "bsi.bsi1", "bsi.bsi2", _SI1_THR)
    zero = torch.zeros_like(w)
    q.add(0, q.bf(zero, 9.0, 49128.131, 11), 11)
    q.add(1, q.bf(18.0e-18 * powi(rdiv(17777.641, w), 3), 15.0, 48161.459, 10), 10)
    q.add(2, q.bf(zero, 5.0, 47351.554, 9), 9)
    q.add(3, q.bf(zero, 3.0, 47284.061, 8), 8)
    r = rdiv(24947.216, w)
    q.add(4, q.bf(4.09e-18 * (r * r), 3.0, 40991.884, 6), 6)
    r = rdiv(26079.180, w)
    q.add(5, q.bf(1.25e-18 * (r * r), 9.0, 39859.920, 5), 5)
    hckt, ehvkt, b2 = q.hckt, c.ehvkt, q.b2
    # P2 1S
    eps = (w - 70000.0) * 2.0 / 6500.0
    reson1 = (97.0e-18 * eps + 94.0e-18) / (eps * eps + 1.0)
    x = 37.0e-18 * pow_(rdiv(50353.180, w), f32(2.40)) + reson1
    bolt = 1.0 * exp(-15394.370 * hckt) * (q.b1(3) - b2 * ehvkt)
    q.add(6, x * bolt / 3.0, 3)
    x = 37.0e-18 * pow_(rdiv(50640.630, w), f32(2.40)) + reson1
    q.add(7, x * bolt * 2.0 / 3.0, 3)
    # P2 1D
    eps = (w - 78600.0) * 2.0 / 13000.0
    reson1 = (-10.0e-18 * eps + 77.0e-18) / (eps * eps + 1.0)
    x = 24.5e-18 * pow_(rdiv(59448.700, w), f32(1.85)) + reson1
    bolt = 5.0 * exp(-6298.850 * hckt) * (q.b1(2) - b2 * ehvkt)
    q.add(8, x * bolt / 3.0, 2)
    x = 24.5e-18 * pow_(rdiv(59736.150, w), f32(1.85)) + reson1
    q.add(9, x * bolt * 2.0 / 3.0, 1)          # sic: source function with BSI1(J,1)
    q.add(10, q.bf(18.0e-18 * powi(rdiv(63446.510, w), 3), 15.0, 45303.310, 7), 7)
    # P2 3P
    r = rdiv(65524.393, w)
    x = 72.0e-18 * pow_(r, f32(1.90))
    x = torch.where(w > 74000.0, 93.0e-18 * pow_(r, 4.0), x)
    x = x / 3.0
    q.add(11, q.bf(x, 5.0, 223.157, 1), 1)
    q.add(12, q.bf(x, 3.0, 77.115, 1), 1)
    q.add(13, q.bf(x, 1.0, None, 1), 1)
    x = x * 2.0
    q.add(14, q.bf(x, 5.0, 223.157, 1), 1)
    q.add(15, q.bf(x, 3.0, 77.115, 1), 1)
    q.add(16, q.bf(x, 1.0, None, 1), 1)
    q.add(17, q.bf(15.0e-18 * powi(rdiv(75423.767, w), 3), 5.0, 33326.053, 4), 4)
    ssi1 = torch.where(q.h > 0.0, q.s / q.h, c.bnu.expand_as(q.h))
    return q.h * c.lay("bsi.xnfpsi", 1) / c.lay("state.rho"), ssi1


def si1op(s: State) -> dict:
    c = Ctx.from_state(s)
    a, ss = si1op_k(c)
    return {"bsi.asi1": put_rows(s["bsi.asi1"], a, c.n), "bsi.ssi1": put_rows(s["bsi.ssi1"], ss, c.n)}


# ============================================================================================= FE1OP
_FE = None


def _fe_tables():
    global _FE
    if _FE is None:
        _FE = tuple(np.asarray(table("fe1op", k), dtype=np.float64) for k in ("g", "e", "wno"))
    return _FE


def fe1op_k(c: Ctx, sfe1_prev):
    """SUBROUTINE FE1OP: Fe I bound-free (48 pseudo-levels) -> (AFE1, SFE1, BFUDGE=BSI1(.,1) [1,n])."""
    G, E, WNO = _fe_tables()
    w = c.waveno
    hckt = c.lay("temp.hckt")
    bfudge = c.lay("bsi.bsi1", 1)
    afe1 = c.zeros()
    run = ~(w < 21000.0)
    for i in range(48):
        m = run & ~(WNO[i] > w)
        if not bool(m.any()):
            continue
        xsect = rdiv(3.0e-18, 1.0 + powi((WNO[i] + 3000.0 - w) / WNO[i] / f32(0.1), 4))
        afe1 = acc(m, afe1, xsect * G[i] * exp(-E[i] * hckt))
    afe1_f = afe1 * c.stim * c.lay("bfe.xnfpfe", 1) / c.lay("state.rho")
    sfe1_f = c.bnu * c.stim / (bfudge - c.ehvkt)
    return (torch.where(run, afe1_f, afe1), torch.where(run, sfe1_f, sfe1_prev.expand_as(sfe1_f)), bfudge)


def fe1op(s: State) -> dict:
    c = Ctx.from_state(s)
    a, sf, bf = fe1op_k(c, c.lay("bfe.sfe1"))
    return {"bfe.afe1": put_rows(s["bfe.afe1"], a, c.n), "bfe.sfe1": put_rows(s["bfe.sfe1"], sf, c.n),
            "flux.flxrad": put_rows(s["flux.flxrad"], bf, c.n)}


# ====================================================================================== CHOP / OHOP
_MOL = {}


def _mol_tables(unit):
    if unit not in _MOL:
        cross = "crossch" if unit == "chop" else "crossoh"
        part = "partch" if unit == "chop" else "partoh"
        _MOL[unit] = (torch.as_tensor(table(unit, cross), dtype=torch.float64),
                      torch.as_tensor(table(unit, part), dtype=torch.float64))
    return _MOL[unit]


def molop_k(c: Ctx, unit: str, freq1: float = 0.0):
    """FUNCTION CHOP(J) / OHOP(J) for all J: cross-section x partition function of CH / OH.

    Returns (values [NF,n], fresh [NF,1] bool: FREQ.NE.FREQ1 at the J=1 call, the WAVENO the routine
    stores on a fresh call [NF,1]). The frequency sequence is the batch order (FREQ1 of row k is row k-1's
    FREQ; of row 0 the SAVEd `freq1`).
    """
    cross, part = (t.to(c.device) for t in _mol_tables(unit))
    ncol = cross.shape[1]
    freq = c.freq
    prev = torch.cat([torch.full_like(freq[:1], float(freq1)), freq[:-1]], dim=0)
    fresh = freq != prev
    waveno = freq / 2.99792458e10
    if unit == "chop":
        evolt = waveno / 8065.479
        n = torch.trunc(evolt * 10.0).to(torch.int64)
        en = (n.to(torch.float32) * f32(0.1)).to(torch.float64)
        valid = (n >= 20) & (n < 105)
    else:
        evolt = waveno / f32(8065.479)
        n = torch.trunc(evolt * 10.0 - 20.0).to(torch.int64)
        en = ((n.to(torch.float32) * f32(0.1)) + torch.tensor(2.0, dtype=torch.float32)).to(torch.float64)
        valid = (n > 0) & (n < 130)
    nc = torch.clamp(n, 1, ncol - 1)                                   # [NF,1], 1-based N (safe gather)
    lo = cross[:, (nc - 1).reshape(-1)].T                              # [NF,15]  CROSSCH(IT,N)
    hi = cross[:, nc.reshape(-1)].T                                    # CROSSCH(IT,N+1)
    crt = lo + (hi - lo) * (evolt - en) / f32(0.1)                     # CROSSCHT [NF,15]
    t = c.lay("temp.t")
    it = torch.clamp(torch.trunc((t - 1000.0) / 200.0 + 1.0), min=1).to(torch.int64)
    itc = torch.clamp(it, max=part.shape[0] - 1)
    tn = itc.to(torch.float64) * 200.0 + 800.0
    pf = part[itc - 1] + (part[itc] - part[itc - 1]) * (t - tn) / 200.0
    it2 = torch.clamp(torch.trunc((t - 2000.0) / 500.0 + 1.0), min=1).to(torch.int64)
    it2c = torch.clamp(it2, max=14)
    tn2 = it2c.to(torch.float64) * 500.0 + 1500.0
    i2 = it2c.expand(c.nf, c.n)
    x1 = torch.gather(crt, 1, i2 - 1)
    x2 = torch.gather(crt, 1, i2)
    val = exp((x1 + (x2 - x1) * (t - tn2) / 500.0) * 2.30258509299405) * pf
    ok = valid & ~(t >= 9000.0)
    return torch.where(ok, val, torch.zeros_like(val)), fresh, waveno


def _molop(s: State, unit: str) -> tuple[torch.Tensor, dict]:
    c = Ctx.from_state(s)
    f1 = float(save_get(s, unit, "freq1").reshape(-1)[0])
    val, fresh, wn = molop_k(c, unit, f1)
    out = {f"save.{unit}.freq1": scalar1(c.freq.reshape(-1)[-1], s["freq.freq"])}
    if bool(fresh[0, 0]):
        out["freq.waveno"] = scalar1(wn.reshape(-1)[0], s["freq.waveno"])
    return val.reshape(-1), out


def chop(s: State):
    """CHOP(J) for J=1..NRHOX (one call per layer, in order) -> (values [n], State entries written)."""
    return _molop(s, "chop")


def ohop(s: State):
    """OHOP(J) for J=1..NRHOX -> (values [n], State entries written)."""
    return _molop(s, "ohop")


# ========================================================================================== H2COLLOP
_H2C = None


def _h2c_tables():
    global _H2C
    if _H2C is None:
        _H2C = (torch.as_tensor(table("h2collop", "h2h2"), dtype=torch.float64),
                torch.as_tensor(table("h2collop", "h2he"), dtype=torch.float64))
    return _H2C


def h2collop_cache(c: Ctx, itemp1: int, xnh2_s: torch.Tensor):
    if c.itemp() == itemp1:
        return itemp1, xnh2_s
    t = c.lay("temp.t")
    q = c.lay("bhyd.xnfph", 1) * 2.0 * c.lay("bhyd.bhyd", 1)
    poly = (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (1.06206e-18 - 3.08720e-23 * t) * t) * t)
                          * t) * t) * t
    xnh2 = q * q * exp(rdiv(f32(4.478), c.lay("temp.tkev")) - 4.64584e1 + poly - 1.5 * c.lay("temp.tlog"))
    return c.itemp(), put_rows(xnh2_s, xnh2, c.n)


def h2collop_k(c: Ctx, xnh2: torch.Tensor):
    """SUBROUTINE H2COLLOP: H2-H2 and H2-He collision-induced absorption -> (AH2COLL [NF,n], WAVENO [NF,1])."""
    h2h2, h2he = (t.to(c.device) for t in _h2c_tables())
    w = c.freq / 2.99792458e10
    big = w > 20000.0
    nu = torch.trunc(torch.where(big, torch.zeros_like(w), w) / 250.0).to(torch.int64) + 1
    nu = torch.clamp(nu, max=80)
    delnu = (w - 250.0 * (nu - 1).to(torch.float64)) / 250.0
    i = (nu - 1).reshape(-1)
    nu_h2 = h2h2[:, i].T * (1.0 - delnu) + h2h2[:, i + 1].T * delnu        # [NF,7]
    nu_he = h2he[:, i].T * (1.0 - delnu) + h2he[:, i + 1].T * delnu
    t = c.lay("temp.t")
    it = torch.clamp(torch.trunc(t / 1000.0), max=6).clamp(min=1).to(torch.int64)
    delt = (t - 1000.0 * it.to(torch.float64)) / 1000.0
    delt = torch.clamp(torch.clamp(delt, max=1.0), min=0.0)
    ie = it.expand(c.nf, c.n)
    xh2h2 = torch.gather(nu_h2, 1, ie - 1) * (1.0 - delt) + torch.gather(nu_h2, 1, ie) * delt
    xh2he = torch.gather(nu_he, 1, ie - 1) * (1.0 - delt) + torch.gather(nu_he, 1, ie) * delt
    x = xnh2[: c.n].reshape(1, c.n)
    a = (pow_(10.0, xh2he) * c.lay("bhe.xnfhe", 1) + pow_(10.0, xh2h2) * x) * x / c.lay("state.rho") * c.stim
    return torch.where(big, torch.zeros_like(a), a), w


def h2collop(s: State) -> dict:
    c = Ctx.from_state(s)
    it1, xnh2 = h2collop_cache(c, save_itemp1(s, "h2collop"), save_get(s, "h2collop", "xnh2", (99,)))
    a, w = h2collop_k(c, xnh2)
    return {"arg:ah2coll": put_rows(torch.zeros(99, dtype=torch.float64), a, c.n),
            "freq.waveno": scalar1(w.reshape(-1)[0], s["freq.waveno"]),
            "save.h2collop.itemp1": itemp_tensor(it1), "save.h2collop.xnh2": xnh2}


# ============================================================================================ COOLOP
def coolop_k(c: Ctx, prev: dict, saves: dict):
    """SUBROUTINE COOLOP. prev: [.,n] previous SC1/SMG1/SFE1 (written only conditionally);
    saves: {"h2collop": (itemp1, xnh2), "chop": freq1, "ohop": freq1}.
    Returns (species dict of [NF,n] tensors, new saves, WAVENO after the call [NF,1], BFUDGE [1,n])."""
    ac1, sc1 = c1op_k(c, prev["sc1"])
    amg1, smg1 = mg1op_k(c, prev["smg1"])
    aal1, sal1 = al1op_k(c, prev["sal1"])
    asi1, ssi1 = si1op_k(c)
    afe1, sfe1, bfudge = fe1op_k(c, prev["sfe1"])
    it1, xnh2 = h2collop_cache(c, *saves["h2collop"])
    ah2coll, waveno = h2collop_k(c, xnh2)
    c2 = c.with_waveno(waveno)
    ch, _, _ = molop_k(c2, "chop", saves["chop"])
    oh, _, _ = molop_k(c2, "ohop", saves["ohop"])
    acool = (ac1 + amg1 + aal1 + asi1 + afe1 + (ch * c.lay("choh.xnfpch") + oh * c.lay("choh.xnfpoh")) * c.stim
             / c.lay("state.rho") + ah2coll)
    sp = dict(ac1=ac1, sc1=sc1, amg1=amg1, smg1=smg1, aal1=aal1, sal1=sal1, asi1=asi1, ssi1=ssi1, afe1=afe1,
              sfe1=sfe1, acool=acool, ah2coll=ah2coll)
    last = float(c.freq.reshape(-1)[-1])
    return sp, {"h2collop": (it1, xnh2), "chop": last, "ohop": last}, waveno, bfudge


def coolop(s: State) -> dict:
    c = Ctx.from_state(s)
    prev = {k: c.lay(f"{b}.{k}") for b, k in (("bc", "sc1"), ("bmg", "smg1"), ("bal", "sal1"), ("bfe", "sfe1"))}
    saves = {"h2collop": (save_itemp1(s, "h2collop"), save_get(s, "h2collop", "xnh2", (99,))),
             "chop": float(save_get(s, "chop", "freq1").reshape(-1)[0]),
             "ohop": float(save_get(s, "ohop", "freq1").reshape(-1)[0])}
    sp, sv, w, bf = coolop_k(c, prev, saves)
    n = c.n
    out = {f"{b}.{k}": put_rows(s[f"{b}.{k}"], sp[k], n) for b, k in (
        ("bc", "ac1"), ("bc", "sc1"), ("bmg", "amg1"), ("bmg", "smg1"), ("bal", "aal1"), ("bal", "sal1"),
        ("bsi", "asi1"), ("bsi", "ssi1"), ("bfe", "afe1"), ("bfe", "sfe1"), ("ops", "acool"))}
    out["flux.flxrad"] = put_rows(s["flux.flxrad"], bf, n)
    out["freq.waveno"] = scalar1(w.reshape(-1)[-1], s["freq.waveno"])
    out["save.coolop.ah2coll"] = put_rows(save_get(s, "coolop", "ah2coll", (99,)), sp["ah2coll"], n)
    out["save.h2collop.itemp1"] = itemp_tensor(sv["h2collop"][0])
    out["save.h2collop.xnh2"] = sv["h2collop"][1]
    out["save.chop.freq1"] = scalar1(sv["chop"], s["freq.freq"])
    out["save.ohop.freq1"] = scalar1(sv["ohop"], s["freq.freq"])
    return out
