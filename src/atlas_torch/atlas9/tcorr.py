"""TCORR (atlas9mem.for): Avrett-Krook temperature correction modified for convection, plus the RHOX
correction keeping TAUROS on the standard log grid.

MODE 1 zeroes the integrals (SAVEd locals RJMINS, RDABH, RDIAGJ -> "save.tcorr.*", and FLXRAD);
MODE 2 increments come from `tcorr_terms` and are accumulated by the caller in Fortran order;
MODE 3 is `tcorr3`.

Quirks reproduced:
  * MODE 2: D=TAUNU(J+1)-TAUNU(J) is not recomputed at J=NRHOX (the previous D is reused);
    .922784335098467, .01 (and 0.01/0.02 for TEFF<=4250) are REAL*4 literals; EXPI(3,D) for D<10.
  * MODE 3: the convective flux is smoothed (.25/.5/.25) and written back into FLXCNV(2..NRHOX);
    DDEL(J) is a SAVEd local, only refreshed where CNVFLX>0 and FLXCNV0>0 (stale elsewhere);
    every layer where the Lambda correction is switched off halves DTLAMB of the five layers above it;
    OLDT1 (SAVEd) damps/accelerates the correction from the second iteration of each READIN batch on;
    smoothing (SMOOTH card), forced monotonicity T(J)<=T(J+1)-1, floor 1995 K;
    the final MAP1 back onto TAUSTD, with DUM5 (ABROSS) < 0 replaced by ABROSS(1) above TAUROS(1).
"""
from __future__ import annotations

import torch

from ..fnum import f32
from ..state import State
from . import fmath
from .numerics import deriv, expi, integ, map1b
from .readin import derived_temperature
from .rosstab import Rosstab
from .ttaup import ttaup

_C23 = f32(0.922784335098467)
_D01 = f32(0.01)
_D02 = f32(0.02)
C4PI = f32(12.5664)


def tcorr1(s: State) -> dict:
    z = torch.zeros(s["flux.flxrad"].shape, dtype=torch.float64)
    n = s.iscalar("rhox.nrhox")
    fr = s["flux.flxrad"].clone(); fr[:n] = 0.0
    out = {"flux.flxrad": fr}
    for k in ("rjmins", "rdabh", "rdiagj"):
        a = s.get(f"save.tcorr.{k}", z).clone(); a[:n] = 0.0
        out[f"save.tcorr.{k}"] = a
    return out


def diag_terms(taunu: torch.Tensor, teff: float) -> torch.Tensor:
    """DIAGJ(J) = TERM1 + TERM2 of TCORR MODE 2 for [..., n] optical depths."""
    n = taunu.shape[-1]
    d = torch.empty_like(taunu)
    d[..., :-1] = taunu[..., 1:] - taunu[..., :-1]
    d[..., -1] = d[..., -2]
    d = torch.maximum(torch.tensor(1e-10, dtype=d.dtype), d)
    small = d <= _D01
    ex = torch.where(d < 10.0, expi(3, torch.where(d < 10.0, d, torch.ones_like(d))), torch.zeros_like(d))
    if not teff > 4250.0:
        ex = torch.where((d > _D01) & (d < _D02), torch.zeros_like(ex), ex)
    big = 0.5 * (d + ex - 0.5) / d
    d2 = d * d
    sm = (_C23 - fmath.log(d)) * d / 4.0 + d2 / 12.0 - d2 * d / 96.0 + d2 * d2 / 720.0
    term2 = torch.where(small, sm, big)
    term1 = torch.zeros_like(term2)
    term1[..., 1:] = term2[..., :-1]
    return term1 + term2


def tcorr_terms(rhox, abtot, hnu, jmins, taunu, alpha, bnu, freq, hkt, t, stim, rcowt, teff, numnu, flux):
    """MODE 2 increments for K contributions [K, n]: rdabh, rjminsnu, flxrad, rdiagjnu."""
    r = rcowt.unsqueeze(-1)
    dabtot = deriv(rhox.expand_as(abtot), abtot)
    rdabh = dabtot / abtot * hnu * r
    rjminsnu = abtot * jmins * r
    flxrad = hnu * r
    diagj = diag_terms(taunu, teff)
    if numnu == 1:
        db = (flux * 16.0) / t.expand_as(abtot)
    else:
        db = bnu * freq.unsqueeze(-1) * hkt / t / stim
    rdiagjnu = abtot * (diagj - 1.0) / (1.0 - alpha * diagj) * (1.0 - alpha) * db * r
    return {"rdabh": rdabh, "rjminsnu": rjminsnu, "flxrad": flxrad, "rdiagjnu": rdiagjnu}


def tcorr3(s: State) -> dict:
    n = s.iscalar("rhox.nrhox")
    g = lambda k: s[k][:n]
    rhox, t, tauros, abross = g("rhox.rhox"), g("temp.t"), g("abross.tauros"), g("abross.abross")
    dltdlp, grdadb, dlrdlt, heatcp, hscale = (g(f"conv.{k}") for k in ("dltdlp", "grdadb", "dlrdlt", "heatcp",
                                                                        "hscale"))
    flxcnv, flxcnv0 = g("conv.flxcnv"), g("conv.flxcnv0")
    ptotal, rho = g("ptotal.ptotal"), g("state.rho")
    flxrad = g("flux.flxrad")
    mixlth = s.scalar("conv.mixlth")
    ifconv = s.iscalar("conv.ifconv")
    flux = s.scalar("flux.flux")
    teff = s.scalar("teff.teff")
    rdabh = s["save.tcorr.rdabh"][:n].clone()
    rjmins = s["save.tcorr.rjmins"][:n]
    rdiagj = s["save.tcorr.rdiagj"][:n]
    ddel = s.get("save.tcorr.ddel", torch.zeros(s["temp.t"].shape)).clone()
    oldt1 = s.get("save.tcorr.oldt1", torch.zeros(s["temp.t"].shape)).clone()

    dtdrhx = deriv(rhox, t)
    ddlt = deriv(rhox, dltdlp)
    dabros = deriv(rhox, abross)
    jj = torch.arange(n)
    cnvflx = flxcnv.clone() if ifconv == 1 else torch.zeros_like(flxcnv)
    cnvflx = torch.where(jj < 2, torch.zeros_like(cnvflx), cnvflx)
    cc = torch.zeros_like(cnvflx)
    cc[1:-1] = 0.25 * cnvflx[:-2] + 0.5 * cnvflx[1:-1] + 0.25 * cnvflx[2:]
    cc[-1] = 0.25 * cnvflx[-3] + 0.25 * cnvflx[-2] + 0.5 * cnvflx[-1]
    cnvflx = torch.cat([cnvflx[:1], cc[1:]])
    flxcnv_new = s["conv.flxcnv"].clone(); flxcnv_new[1:n] = cnvflx[1:]
    # loop 31
    rdabh = rdabh - flxrad * dabros / abross
    conv = (cnvflx > 0) & (flxcnv0 > 0)
    dele = torch.where(conv, dltdlp - grdadb, torch.ones_like(t))
    vco = 0.5 * mixlth * fmath.sqrt(torch.clamp(-0.5 * ptotal / rho * dlrdlt, min=0.0))
    fluxco = 0.5 * rho * heatcp * t * mixlth / C4PI
    t2 = t * t
    dd = 8.0 * 5.6697e-5 * (t2 * t2) / (abross * hscale * rho) / (fluxco * C4PI) / vco
    dd = torch.where((torch.tensor(mixlth > 0.0)) & (vco > 0.0), dd, torch.zeros_like(dd))
    taub = abross * rho * mixlth * hscale
    dd = dd * (taub * taub) / (2.0 + taub * taub)
    dd = dd * dd / 2.0
    dd = torch.where(conv, dd, torch.zeros_like(dd))
    ddel_c = (1.0 + dd / (dd + dele)) / dele
    ddel[:n] = torch.where(conv, ddel_c, ddel[:n])
    cnvfl = torch.where((cnvflx / flxrad > 1.0e-3) & (flxcnv0 / flxrad > 1.0e-3), cnvflx, torch.zeros_like(cnvflx))
    q = 1.0 + dd / (dd + dele)
    codrhx = (rdabh + cnvfl * (dtdrhx / t * (1.0 - 9.0 * dd / (dd + dele)) + 1.5 * ddlt / dele * q)) / \
        (flxrad + cnvflx * 1.5 * dltdlp / dele * q)
    codrhx[0] = 0.0
    codrhx[1] = 0.0
    gg = fmath.exp(integ(rhox, codrhx, 0.0))
    gflux = gg * (flxrad + cnvflx - flux) / (flxrad + cnvflx * 1.5 * dltdlp * ddel[:n])
    dtau = integ(tauros, gflux, 0.0)
    dtau = dtau / gg
    dtau = torch.maximum(-tauros / 5.0, torch.minimum(tauros / 5.0, dtau))
    dtflux = -dtau * dtdrhx / abross
    flxerr = (flxrad + cnvflx - flux) / flux * 100.0
    flxdrv = deriv(tauros, flxerr)
    teff25 = teff / 25.0
    lowc = cnvflx / flxrad < 1.0e-5
    flxdrv = torch.where(lowc, rjmins / abross / flux * 100.0, flxdrv)
    dtlamb = -flxdrv * flux / 100.0 / rdiagj * abross
    off = ~(lowc & (tauros < 1.0))                     # DTLAMB(J)=0 and halve J-1..J-5
    if bool(off[:5].any()):
        raise NotImplementedError("TCORR: Lambda correction switched off within the top 5 layers "
                                  "(the Fortran then writes before DTLAMB(1))")
    dtlamb = torch.where(off, torch.zeros_like(dtlamb), dtlamb)
    dtlamb = torch.maximum(torch.tensor(-teff25), torch.minimum(torch.tensor(teff25), dtlamb))
    cnt = torch.zeros(n, dtype=torch.int64)
    offi = off.to(torch.int64)
    for k in range(1, 6):
        cnt[:n - k] += offi[k:]
    for k in range(1, 6):
        dtlamb = torch.where(cnt >= k, dtlamb / 2.0, dtlamb)
    dtsur = (flux - float(flxrad[0])) / flux * 0.25 * float(t[0])
    dtsur = max(-teff25, min(teff25, dtsur))
    dum = dtflux + dtlamb
    tinteg = integ(tauros, dum, 0.0)
    pts, _ = map1b(tauros, tinteg, torch.tensor([0.1, 2.0]))
    tone, ttwo = float(pts[0]), float(pts[1])
    tav = (ttwo - tone) / 2.0
    if dtsur * tav <= 0.0:
        tav = 0.0
    if abs(tav) > abs(dtsur):
        tav = dtsur
    dtsur = dtsur - tav
    dtsurf = torch.full_like(t, dtsur)
    hratio = cnvflx / (cnvflx + flxrad)
    t1 = dtflux + dtlamb + dtsurf
    # loop 60/61
    iter_ = s.iscalar("iter.iter")
    keep = torch.zeros(n, dtype=torch.bool)
    if ifconv == 1:
        keep |= hratio > 0
        keep |= (jj + 1) >= n // 3
    if iter_ == 1:
        keep = torch.ones_like(keep)
    o = oldt1[:n]
    acc = (~keep) & (o * t1 > 0) & (torch.abs(o) > torch.abs(t1))
    t1 = torch.where(acc, t1 * 1.25, t1)
    dmp = (~keep) & (o * t1 < 0)
    t1 = torch.where(dmp, t1 * 0.5, t1)
    oldt1[:n] = t1
    # RHOX correction for constant TAUROS
    tplus = t + t1
    tau1lg, steplg = s.scalar("steplg.tau1lg"), s.scalar("steplg.steplg")
    taustd = fmath.pow(torch.full((n,), 10.0), torch.tensor([tau1lg + float(j) * steplg for j in range(n)]))
    prad, pturb, vturb = g("rad.prad"), g("turbpr.pturb"), g("turbpr.vturb")
    m, _ = map1b(tauros, torch.stack([t, prad, pturb, tplus]), taustd)
    tnew1, prdnew, pturbnew, tnew2 = m[0], m[1], m[2], m[3]
    R = Rosstab(s)
    grav = s.scalar("teff.grav")
    _, ptot1, _ = ttaup(tnew1, taustd, None, prdnew, pturbnew, vturb, grav, n, 0, R)
    _, ptot2, _ = ttaup(tnew2, taustd, None, prdnew, pturbnew, vturb, grav, n, 0, R)
    ppp = (ptot2 - ptot1) / ptot1
    rrr, _ = map1b(taustd, ppp, tauros)
    drhox = rrr * rhox
    tt = t + t1
    j1, j2 = s.iscalar("tsmooth.j1smooth"), s.iscalar("tsmooth.j2smooth")
    tsm = s["tsmooth.tsmooth"].clone()
    if j1 > 0:
        w1, w2, w3 = s.scalar("tsmooth.wtjm1"), s.scalar("tsmooth.wtj"), s.scalar("tsmooth.wtjp1")
        sl = slice(j1 - 1, j2)
        tsm[sl] = w1 * tt[j1 - 2:j2 - 1] + w2 * tt[sl] + w3 * tt[j1:j2 + 1]
        tt = tt.clone(); tt[sl] = tsm[sl]
    # force monotonicity: T(J)=MIN(T(J),T(J+1)-1.) from the bottom up (exact as min_k(T(k)-(k-J)))
    kf = torch.arange(n, dtype=torch.float64)
    v = tt - kf
    v = torch.flip(torch.cummin(torch.flip(v, [0]), 0).values, [0])
    tt = v + kf
    tt = torch.clamp(tt, min=1995.0)
    s2 = State(s)
    tfull = s["temp.t"].clone(); tfull[:n] = tt
    s2["temp.t"] = tfull
    s2.update(derived_temperature(s2))
    rhox2 = rhox + drhox
    p, xne, prad, vt, bmin, pturb, accrad = (g(k) for k in ("state.p", "state.xne", "rad.prad", "turbpr.vturb",
                                                             "depart.bmin", "turbpr.pturb", "rad.accrad"))
    bhyd = s["depart.bhyd"][:n]
    fold = torch.stack([rhox2, tt, p, xne, abross, prad, vt, bmin, pturb, accrad] + [bhyd[:, i] for i in range(6)])
    mm, _ = map1b(tauros, fold, taustd)
    dum5 = mm[4]
    dum5 = torch.where((tauros[0] > taustd) & (dum5 < 0), abross[0].expand_as(dum5), dum5)
    out = {}

    def put(key, val):
        a = s[key].clone(); a[:n] = val; out[key] = a

    put("rhox.rhox", mm[0])
    put("temp.t", mm[1])
    put("state.p", mm[2])
    put("state.xne", mm[3])
    put("abross.abross", dum5)
    put("rad.prad", mm[5])
    put("pzero.pradk", mm[5] + s.scalar("pzero.pradk0"))
    put("turbpr.vturb", mm[6])
    put("turbpr.pturb", mm[8])
    put("depart.bmin", mm[7])
    bh = s["depart.bhyd"].clone()
    for i in range(6):
        bh[:n, i] = mm[10 + i]
    out["depart.bhyd"] = bh
    put("abross.tauros", taustd)
    put("flux.flxerr", flxerr)
    put("flux.flxdrv", flxdrv)
    out["conv.flxcnv"] = flxcnv_new
    s3 = State(s); s3.update(out)
    out.update(derived_temperature(s3))
    out["save.tcorr.ddel"] = ddel
    out["save.tcorr.oldt1"] = oldt1
    rd = s["save.tcorr.rdabh"].clone(); rd[:n] = rdabh
    out["save.tcorr.rdabh"] = rd
    out["tsmooth.tsmooth"] = tsm
    return out
