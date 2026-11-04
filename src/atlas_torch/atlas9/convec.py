"""CONVEC (atlas9mem.for): mixing-length convection (+ optional overshooting), vectorised over layers.

Thermodynamic derivatives come from four extra POPS(0.D0,1,XNE) calls (Physics.electrons) at T*1.001,
T*.999, P*1.001, P*.999 (ITEMP incremented before each). The per-layer opacity iteration (DELTAT, up to 30
passes with two ROSSTAB calls each) and the DELTA series run on all layers at once with per-layer masks;
each layer follows exactly the Fortran's sequence of operations and exit tests.

Quirks reproduced:
  * DTDRHX is EQUIVALENCEd with DLTDLP (DERIV writes into DLTDLP before DLTDLP(J) is recomputed from it);
  * the T/TK/TKEV/TLOG and P perturbations are undone by division, not restored: they come back with the
    roundoff of x*1.001/1.001*.999/.999 (REAL*4 literals), which the rest of the iteration then uses;
  * REAL*4 constants: 1.001, .999, .0009995003, .0010005003, 1.001**4-1., .999**4-1. (folded in single
    precision), 12.5664, 5.6697E-5 (8.*5.6697E-5 folded in single), .15, .7, .3, 1.E-6;
  * DELTA from the closed form when DDD>=.5, otherwise the binomial series until TERM<=1E-6;
  * the scratch arrays CONVEC keeps in /OPS/ (EDENS1..RHO4, SAVXNE, ..., DILUT) are not modelled: KAPP
    zeroes /OPS/ before any further use.
"""
from __future__ import annotations

import numpy as np
import torch

from ..fnum import f32
from ..state import State
from . import fmath
from .numerics import deriv, integ, map1b
from .rosstab import Rosstab

F = np.float32
_U, _D = f32(1.001), f32(0.999)
_TL1, _TL2 = f32(0.0009995003), f32(0.0010005003)
_U4 = float(F(F(F(F(1.001) * F(1.001)) * F(F(1.001) * F(1.001))) - F(1.0)))
_D4 = float(F(F(F(F(0.999) * F(0.999)) * F(F(0.999) * F(0.999))) - F(1.0)))
C4PI = f32(12.5664)
_8SIG = float(F(8.0) * F(5.6697e-5))
_P15, _P7, _P3, _E6 = f32(0.15), f32(0.7), f32(0.3), f32(1.0e-6)


def _set(s: State, key: str, n: int, v: torch.Tensor):
    a = s[key].clone(); a[:n] = v; s[key] = a


def convec(s: State, physics) -> dict:
    """Returns the State updates of CONVEC (calls physics.electrons four times)."""
    n = s.iscalar("rhox.nrhox")
    w = State(s)
    g = lambda k: w[k][:n]
    rhox, t0 = g("rhox.rhox"), g("temp.t")
    dtdrhx = deriv(rhox, t0)
    w["edens.ifedns"] = torch.tensor([1])
    dilut = 1.0 - fmath.exp(-g("abross.tauros"))
    savxne, savxna, savrho = g("state.xne").clone(), g("state.xnatom").clone(), g("state.rho").clone()
    pradk = g("pzero.pradk")

    def pops():
        w["temp.itemp"] = w["temp.itemp"] + 1
        w.update(physics.electrons(w))

    _set(w, "temp.tlog", n, g("temp.tlog") + _TL1)
    for k in ("temp.t", "temp.tk", "temp.tkev"):
        _set(w, k, n, g(k) * _U)
    pops()
    edens1 = g("edens.edens") + 3.0 * pradk / g("state.rho") * (1.0 + dilut * _U4)
    rho1 = g("state.rho").clone()
    _set(w, "temp.tlog", n, g("temp.tlog") - _TL1 - _TL2)
    for k in ("temp.t", "temp.tk", "temp.tkev"):
        _set(w, k, n, g(k) / _U * _D)
    pops()
    edens2 = g("edens.edens") + 3.0 * pradk / g("state.rho") * (1.0 + dilut * _D4)
    rho2 = g("state.rho").clone()
    _set(w, "temp.tlog", n, g("temp.tlog") + _TL2)
    for k in ("temp.t", "temp.tk", "temp.tkev"):
        _set(w, k, n, g(k) / _D)
    _set(w, "state.p", n, g("state.p") * _U)
    pops()
    edens3 = g("edens.edens") + 3.0 * pradk / g("state.rho")
    rho3 = g("state.rho").clone()
    _set(w, "state.p", n, g("state.p") / _U * _D)
    pops()
    edens4 = g("edens.edens") + 3.0 * pradk / g("state.rho")
    rho4 = g("state.rho").clone()
    _set(w, "state.xne", n, savxne)
    _set(w, "state.xnatom", n, savxna)
    _set(w, "state.rho", n, savrho)
    _set(w, "state.p", n, g("state.p") / _D)

    t, p, rho = g("temp.t"), g("state.p"), g("state.rho")
    ptotal = g("ptotal.ptotal")
    grav = w.scalar("teff.grav")
    mixlth = w.scalar("conv.mixlth")
    ifconv = w.iscalar("conv.ifconv")
    flux = w.scalar("flux.flux")
    abross = g("abross.abross")
    vturb = g("turbpr.vturb")
    dedt = (edens1 - edens2) / t * 500.0
    drdt = (rho1 - rho2) / t * 500.0
    dedpg = (edens3 - edens4) / p * 500.0
    drdpg = (rho3 - rho4) / p * 500.0
    dpdpg = 1.0
    dpdt = 4.0 * pradk / t * dilut
    dltdlp = ptotal / t / grav * dtdrhx
    heatcv = dedt - dedpg * drdt / drdpg
    heatcp = dedt - dedpg * dpdt / dpdpg - ptotal / (rho * rho) * (drdt - drdpg * dpdt / dpdpg)
    velsnd = torch.where(heatcv <= 0.0, torch.zeros_like(t), fmath.sqrt(heatcp / heatcv * dpdpg / drdpg))
    dlrdlt = t / rho * (drdt - drdpg * dpdt / dpdpg)
    grdadb = -ptotal / rho / t * dlrdlt / heatcp
    hscale = ptotal / rho / grav
    vconv = torch.zeros_like(t)
    flxcnv = torch.zeros_like(t)
    jj = torch.arange(n)
    dele = dltdlp - grdadb
    vco = 0.5 * mixlth * fmath.sqrt(torch.clamp(-0.5 * ptotal / rho * dlrdlt, min=0.0))
    cand = (jj >= 3) & ~(dele < 0.0) & (vco != 0.0) if mixlth != 0.0 else torch.zeros(n, dtype=torch.bool)
    fluxco = 0.5 * rho * heatcp * t * mixlth / C4PI
    R = Rosstab(w)
    idx = torch.nonzero(cand).flatten()
    if idx.numel():
        tc, pc, vc = t[idx], p[idx], vturb[idx]
        rosst = R(tc, pc, vc)
        deltat = torch.zeros_like(tc)
        olddelt = torch.zeros_like(tc)
        de, vcoc, fco = dele[idx], vco[idx], fluxco[idx]
        hs, rh, ab = hscale[idx], rho[idx], abross[idx]
        vconv_c = torch.zeros_like(tc)
        flx_c = torch.zeros_like(tc)
        active = torch.ones_like(tc, dtype=torch.bool)
        its = 30 if ifconv != 0 else 1
        t2 = tc * tc
        t4 = t2 * t2
        for _ in range(its):
            if not bool(active.any()):
                break
            a = active
            dplus = R(tc + deltat, pc, vc) / rosst
            dminus = R(tc - deltat, pc, vc) / rosst
            abconv = fmath.rdiv(2.0, fmath.rdiv(1.0, dplus) + fmath.rdiv(1.0, dminus)) * ab
            d = _8SIG * t4 / (abconv * hs * rh) / (fco * C4PI) / vcoc
            taub = abconv * rh * mixlth * hs
            d = d * (taub * taub) / (2.0 + taub * taub)
            d = d * d / 2.0
            q = de / (d + de)
            ddd = q * q
            closed = ~(ddd < 0.5)
            delta_c = (1.0 - fmath.sqrt(1.0 - ddd)) / ddd
            delta_s = _series(ddd)
            delta = torch.where(closed, delta_c, delta_s)
            delta = delta * (de * de) / (d + de)
            vcv = vcoc * fmath.sqrt(delta)
            fl = torch.clamp(fco * vcv * delta, min=0.0)
            dt = tc * mixlth * delta
            dt = torch.minimum(dt, tc * _P15)
            dt = dt * _P7 + olddelt * _P3
            vconv_c = torch.where(a, vcv, vconv_c)
            flx_c = torch.where(a, fl, flx_c)
            deltat = torch.where(a, dt, deltat)
            done = (dt < olddelt + 0.5) & (dt > olddelt - 0.5)
            olddelt = torch.where(a & ~done, dt, olddelt)
            active = a & ~done
        vconv[idx] = vconv_c
        flxcnv[idx] = flx_c
    flxcnv0 = flxcnv.clone()
    flxcnv1 = w["conv.flxcnv1"][:n].clone()
    overwt = w.scalar("conv.overwt")
    if overwt > 0.0:
        height = g("height.height")
        wtcnv = 0.0
        for v in (flxcnv / flux).tolist():
            wtcnv = max(wtcnv, v)
        wtcnv = min(wtcnv, 1.0) * overwt
        delhgt = torch.minimum(torch.minimum(hscale * 0.5e-5 * wtcnv, height[-1] - height), height - height[0])
        flxcnv0 = flxcnv.clone()
        flxcnv1 = torch.zeros_like(flxcnv)
        cnvint = integ(height, flxcnv, 0.0)
        js = [j for j in range(n // 2 - 1, n - 1) if float(delhgt[j]) != 0.0]
        if js:
            jt = torch.tensor(js)
            pts = torch.stack([height[jt] - delhgt[jt], height[jt] + delhgt[jt]], dim=1)   # [K, 2]
            v, _ = map1b(height, cnvint, pts.reshape(-1, 1))
            v = v.reshape(-1, 2)
            flxcnv1[jt] = flxcnv1[jt] + (v[:, 1] - v[:, 0]) / delhgt[jt] / 2.0
        flxcnv1 = torch.clamp(flxcnv1, min=0.0)
        flxcnv0 = torch.clamp(flxcnv0, min=0.0)
        flxcnv = torch.maximum(flxcnv0, flxcnv1)
    nconv = w.iscalar("conv.nconv")
    flxcnv = flxcnv.clone()
    flxcnv[:nconv] = 0.0
    out = {k: w[k] for k in ("temp.t", "temp.tk", "temp.tkev", "temp.tlog", "temp.itemp", "state.p", "state.xne",
                             "state.xnatom", "state.rho", "edens.edens", "edens.ifedns")}
    for k in w:
        if k not in s or w[k] is not s[k]:
            out.setdefault(k, w[k])
    for k, v in (("dltdlp", dltdlp), ("heatcp", heatcp), ("dlrdlt", dlrdlt), ("velsnd", velsnd),
                 ("grdadb", grdadb), ("hscale", hscale), ("vconv", vconv), ("flxcnv", flxcnv),
                 ("flxcnv0", flxcnv0), ("flxcnv1", flxcnv1)):
        a = s[f"conv.{k}"].clone(); a[:n] = v; out[f"conv.{k}"] = a
    return out


def _series(ddd: torch.Tensor) -> torch.Tensor:
    """Label 24/25: DELTA=.5+sum TERM, TERM=UP/DOWN*DDD*TERM until TERM<=1E-6 (per element)."""
    delta = torch.full_like(ddd, 0.5)
    term = torch.full_like(ddd, 0.5)
    up = torch.full_like(ddd, -1.0)
    down = torch.full_like(ddd, 2.0)
    act = ddd < 0.5
    for _ in range(100000):
        if not bool(act.any()):
            break
        up = torch.where(act, up + 2.0, up)
        down = torch.where(act, down + 2.0, down)
        nt = up / down * ddd * term
        term = torch.where(act, nt, term)
        delta = torch.where(act, delta + nt, delta)
        act = act & (nt > _E6)
    return delta
