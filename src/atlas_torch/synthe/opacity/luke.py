"""LUKEOP and its callees (atlas7v.for): N1OP, O1OP, MG2OP, SI2OP, CA2OP, SEATON.

ITEMP caches reproduced explicitly (State keys save.<unit>.*):
  LUKEOP (ITEMP1, XNFPN, XNFPO, XNFPMG, XNFPSI, XNFPCA: populations from POPS(code,1)),
  N1OP (ITEMP1, C1130, C1020), MG2OP (ITEMP1, C1169), CA2OP (ITEMP1, C1218, C1420), SI2OP (ITEMP1, NT, DT).
The FREQ1 caches (X853.., X911, X824.., X(6) of SI2OP, ...) are pure functions of FREQ: recomputation
gives the cached values, so they are not carried.
POPS is not ported here: when LUKEOP's ITEMP changes, the populations come from a `pops(s, code, mode)`
callback (the population port), otherwise from the SAVEd arrays in the State.
"""
from __future__ import annotations

import torch

from ...state import State
from ._core import Ctx, exp, f32, itemp_tensor, powi, put_rows, rdiv, save_get, save_itemp1, sqrt, table


def seaton(c: Ctx, freq0: float, xsect: float, power: float, a: float) -> torch.Tensor:
    """FUNCTION SEATON(FREQ0,XSECT,POWER,A) at COMMON FREQ -> [NF,1]."""
    r = rdiv(freq0, c.freq)
    return xsect * (a + (1.0 - a) * r) * sqrt(powi(r, int(2.0 * power + f32(0.01))))


def _gated(c: Ctx, f0: float, val):
    return torch.where(c.freq >= f0, val, torch.zeros_like(c.freq))


def _cache(c: Ctx, itemp1: int, saved: dict, fn):
    if c.itemp() == itemp1:
        return itemp1, saved
    new = fn()
    return c.itemp(), {k: put_rows(saved[k], v, c.n) for k, v in new.items()}


# --------------------------------------------------------------------------------------------- N1OP
def n1op_cache(c: Ctx, itemp1: int, saved: dict):
    def f():
        tkev = c.lay("temp.tkev")
        return {"c1130": 6.0 * exp(rdiv(-f32(3.575), tkev)), "c1020": 10.0 * exp(rdiv(-f32(2.384), tkev))}
    return _cache(c, itemp1, saved, f)


def n1op_k(c: Ctx, cache: dict) -> torch.Tensor:
    """REAL*8 FUNCTION N1OP(J) for all J: N I cross-section x partition function -> [NF,n]."""
    x853 = _gated(c, 3.517915e15, seaton(c, 3.517915e15, 1.142e-17, 2.0, 4.29))
    x1020 = _gated(c, 2.941534e15, seaton(c, 2.941534e15, 4.41e-18, 1.5, 3.85))
    x1130 = _gated(c, 2.653317e15, seaton(c, 2.653317e15, 4.2e-18, 1.5, 4.34))
    n = c.n
    return x853 * 4.0 + x1020 * cache["c1020"][:n].reshape(1, n) + x1130 * cache["c1130"][:n].reshape(1, n)


def o1op_k(c: Ctx) -> torch.Tensor:
    """FUNCTION O1OP(J): O I (Peterson after Peach) -> [NF,1] (independent of J)."""
    return _gated(c, 3.28805e15, seaton(c, 3.28805e15, 2.94e-18, 1.0, 2.66)) * 9.0


def mg2op_cache(c: Ctx, itemp1: int, saved: dict):
    return _cache(c, itemp1, saved, lambda: {"c1169": 6.0 * exp(rdiv(-f32(4.43), c.lay("temp.tkev")))})


def mg2op_k(c: Ctx, cache: dict) -> torch.Tensor:
    """REAL*8 FUNCTION MG2OP(J) -> [NF,n]."""
    x824 = _gated(c, 3.635492e15, seaton(c, 3.635492e15, 1.40e-19, 4.0, 6.7))
    x1169 = _gated(c, 2.564306e15, f32(5.11e-19) * powi(rdiv(2.564306e15, c.freq), 3))
    return x824 * 2.0 + x1169 * cache["c1169"][: c.n].reshape(1, c.n)


def ca2op_cache(c: Ctx, itemp1: int, saved: dict):
    def f():
        tkev = c.lay("temp.tkev")
        return {"c1218": 10.0 * exp(rdiv(-f32(1.697), tkev)), "c1420": 6.0 * exp(rdiv(-f32(3.142), tkev))}
    return _cache(c, itemp1, saved, f)


def ca2op_k(c: Ctx, cache: dict) -> torch.Tensor:
    """FUNCTION CA2OP(J) -> [NF,n]."""
    x1044 = _gated(c, 2.870454e15, 5.4e-20 * powi(rdiv(2.870454e15, c.freq), 3))
    x1218 = _gated(c, 2.460127e15, 1.64e-17 * sqrt(rdiv(2.460127e15, c.freq)))
    x1420 = _gated(c, 2.110779e15, seaton(c, 2.110779e15, 4.13e-18, 3.0, 0.69))
    n = c.n
    return x1044 * 2.0 + x1218 * cache["c1218"][:n].reshape(1, n) + x1420 * cache["c1420"][:n].reshape(1, n)


# -------------------------------------------------------------------------------------------- SI2OP
_SI2 = None


def _si2_tables():
    global _SI2
    if _SI2 is None:
        _SI2 = {k: torch.as_tensor(table("si2op", k), dtype=torch.float64) for k in ("peach", "freqsi", "flog", "tlg")}
    return _SI2


def si2op_cache(c: Ctx, itemp1: int, saved: dict):
    """SI2OP's ITEMP block: NT(K)=MAX0(MIN0(5,INT(T/2000.)-4),1), DT(K) (log-T interpolation weight)."""
    if c.itemp() == itemp1:
        return itemp1, saved
    tlg = _si2_tables()["tlg"].to(c.device)
    t = c.lay("temp.t")
    nt = torch.clamp(torch.clamp(torch.trunc(t / 2000.0).to(torch.int64) - 4, max=5), min=1)
    dt = (c.lay("temp.tlog") - tlg[nt - 1]) / (tlg[nt] - tlg[nt - 1])
    n = c.n
    nts = saved["nt"].clone()
    nts[:n] = nt.reshape(-1).to(nts.dtype)
    return c.itemp(), {"nt": nts, "dt": put_rows(saved["dt"], dt, n)}


def si2op_k(c: Ctx, cache: dict) -> torch.Tensor:
    """FUNCTION SI2OP(J): Si II (Peach) cross-section x partition function -> [NF,n]."""
    T = {k: v.to(c.device) for k, v in _si2_tables().items()}
    nf = c.nf
    N = torch.full((nf, 1), 8, dtype=torch.int64, device=c.device)
    for i in range(7, 0, -1):
        N = torch.where(c.freq > T["freqsi"][i - 1], torch.full_like(N, i), N)
    d = (c.freqlg - T["flog"][N - 1]) / (T["flog"][N] - T["flog"][N - 1])
    N = torch.where(N > 2, 2 * N - 2, N)
    N = torch.where(N == 14, torch.full_like(N, 13), N)
    d1 = 1.0 - d
    peach = T["peach"]                                               # (6,14)
    x = peach[:, N.reshape(-1)].T * d + peach[:, (N - 1).reshape(-1)].T * d1  # [NF,6]
    n = c.n
    nt = cache["nt"][:n].reshape(1, n).to(torch.int64).expand(nf, n)
    dt = cache["dt"][:n].reshape(1, n)
    return exp(torch.gather(x, 1, nt - 1) * (1.0 - dt) + torch.gather(x, 1, nt) * dt) * 6.0


# ------------------------------------------------------------------------------------------- LUKEOP
LUKE_POPS = [("xnfpn", 7.0), ("xnfpo", 8.0), ("xnfpmg", 12.01), ("xnfpsi", 14.01), ("xnfpca", 20.01)]
LUKE_SUBS = {"n1op": ("c1130", "c1020"), "mg2op": ("c1169",), "ca2op": ("c1218", "c1420"), "si2op": ("nt", "dt")}


def luke_saves(s: State) -> dict:
    """The SAVEd state LUKEOP and its functions carry, from the State (zeros = never called)."""
    sv = {"lukeop": (save_itemp1(s, "lukeop"), {k: save_get(s, "lukeop", k, (99,)) for k, _ in LUKE_POPS})}
    for u, vs in LUKE_SUBS.items():
        sv[u] = (save_itemp1(s, u), {v: save_get(s, u, v, (99,), dtype=torch.int64 if v == "nt" else torch.float64)
                                     for v in vs})
    return sv


def lukeop_k(c: Ctx, saves: dict, pops=None):
    """SUBROUTINE LUKEOP -> (ALUKE [NF,n], updated saves)."""
    it1, pop = saves["lukeop"]
    if c.itemp() != it1:
        if pops is None:
            raise RuntimeError("LUKEOP: ITEMP changed, populations needed: pass pops(s, code, mode) "
                               "or provide save.lukeop.* with save.lukeop.itemp1 = ITEMP")
        pop = {k: put_rows(pop[k], torch.as_tensor(pops(c.s, code, 1)).reshape(99, -1)[:, 0], c.n)
               for k, code in LUKE_POPS}
        it1 = c.itemp()
    new = {"lukeop": (it1, pop)}
    new["n1op"] = n1op_cache(c, *saves["n1op"])
    new["mg2op"] = mg2op_cache(c, *saves["mg2op"])
    new["si2op"] = si2op_cache(c, *saves["si2op"])
    new["ca2op"] = ca2op_cache(c, *saves["ca2op"])
    n = c.n
    p = {k: v[:n].reshape(1, n) for k, v in pop.items()}
    aluke = ((n1op_k(c, new["n1op"][1]) * p["xnfpn"] + o1op_k(c) * p["xnfpo"] + mg2op_k(c, new["mg2op"][1])
              * p["xnfpmg"] + si2op_k(c, new["si2op"][1]) * p["xnfpsi"] + ca2op_k(c, new["ca2op"][1]) * p["xnfpca"])
             * c.stim / c.lay("state.rho"))
    return aluke, new


def saves_to_state(new: dict) -> dict:
    out = {}
    for u, (it1, arrs) in new.items():
        out[f"save.{u}.itemp1"] = itemp_tensor(it1)
        for k, v in arrs.items():
            out[f"save.{u}.{k}"] = v
    return out


def lukeop(s: State, pops=None) -> dict:
    c = Ctx.from_state(s)
    aluke, new = lukeop_k(c, luke_saves(s), pops)
    out = {"ops.aluke": put_rows(s["ops.aluke"], aluke, c.n)}
    out.update(saves_to_state(new))
    return out


def _fn_api(unit, cache_fn, k_fn):
    def api(s: State):
        """FUNCTION <unit>(J) for J=1..NRHOX -> (values [n], State entries written)."""
        c = Ctx.from_state(s)
        it1, saved = luke_saves(s)[unit]
        it1, saved = cache_fn(c, it1, saved)
        return k_fn(c, saved).expand(1, c.n).reshape(-1), saves_to_state({unit: (it1, saved)})
    api.__name__ = unit
    return api


n1op = _fn_api("n1op", n1op_cache, n1op_k)
mg2op = _fn_api("mg2op", mg2op_cache, mg2op_k)
ca2op = _fn_api("ca2op", ca2op_cache, ca2op_k)
si2op = _fn_api("si2op", si2op_cache, si2op_k)


def o1op(s: State):
    c = Ctx.from_state(s)
    return o1op_k(c).expand(1, c.n).reshape(-1), {}
