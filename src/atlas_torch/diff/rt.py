"""Differentiable SPECTRV: continuum interpolation, Planck function and JOSH (surface flux) in float64.

Formulas of atlas_torch.synthe.spectrv_prog / josh (the exact REAL*4-emulating port), with two deliberate
numerical changes:
  * the scattering solution is the exact solution of JOSH's linear system on the 51-point XTAU8 grid,
    (I - diag(XALPHA) COEFJ) XS = (1 - XALPHA) XSBAR, by a float64 LU solve, instead of the Fortran's REAL*4
    Gauss-Seidel sweeps stopped at |dXS/XS| < 1E-5 (differentiable and smooth; the difference is the
    Fortran's iteration/rounding error, quantified in RESULTS);
  * everything in float64 (COEFJ/COEFH/CH are the REAL*4 table values, widened).
The optically thick branch (TAUNU(1) > XTAU8(51), label 401) iterates DERIV in float64 to convergence (an
explicit iteration; derivatives through its final iterate).
Decisions (which rows are optically thick, MAP1's interval for every XTAU8 point, the surface rows) are
recorded on the first call and replayed afterwards (Spectrv.freeze, reset()): MAP1 is continuous but not
differentiable where a TAUNU node crosses an XTAU8 point, so the frozen function is the smooth piece the
base point lies on. Only IFSURF=1 (SURFACE FLUX, BasicATLAS) is
supported. PARCOE / INTEG / MAP1 / DERIV are float64 transcriptions (copied from synthe.josh so that this
package does not depend on that module's internals).
"""
from __future__ import annotations

import numpy as np
import torch

D = torch.float64
NXTAU = 51


def parcoe(f, x):
    f, x = torch.broadcast_tensors(f, x)
    n = f.shape[-1]
    b0 = (f[..., 1] - f[..., 0]) / (x[..., 1] - x[..., 0])
    a0 = f[..., 0] - x[..., 0] * b0
    bn = (f[..., n - 1] - f[..., n - 2]) / (x[..., n - 1] - x[..., n - 2])
    an = f[..., n - 1] - x[..., n - 1] * bn
    fj, fj1, fjp = f[..., 1:-1], f[..., :-2], f[..., 2:]
    xj, xj1, xjp = x[..., 1:-1], x[..., :-2], x[..., 2:]
    d = (fj - fj1) / (xj - xj1)
    cc = fjp / ((xjp - xj) * (xjp - xj1)) - fj / ((xj - xj1) * (xjp - xj)) + fj1 / ((xj - xj1) * (xjp - xj1))
    bb = d - (xj + xj1) * cc
    aa = fj1 - xj1 * d + xj * xj1 * cc
    # J=2, 3 (1-based) are linear
    b1 = (f[..., 2] - f[..., 1]) / (x[..., 2] - x[..., 1])
    a1 = f[..., 1] - x[..., 1] * b1
    b2 = (f[..., 3] - f[..., 2]) / (x[..., 3] - x[..., 2])
    a2 = f[..., 2] - x[..., 2] * b2
    z = torch.zeros_like(b1)
    a = torch.cat([a0[..., None], a1[..., None], a2[..., None], aa[..., 2:], an[..., None]], -1)
    b = torch.cat([b0[..., None], b1[..., None], b2[..., None], bb[..., 2:], bn[..., None]], -1)
    c = torch.cat([z[..., None], z[..., None], z[..., None], cc[..., 2:], z[..., None]], -1)
    cj, cn = c[..., 1:-1], c[..., 2:]
    aj, an_ = a[..., 1:-1], a[..., 2:]
    bj, bn_ = b[..., 1:-1], b[..., 2:]
    blend = cj != 0.0
    acn = torch.abs(cn)
    den = acn + torch.abs(cj)
    wt = acn / torch.where(blend, den, torch.ones_like(den))
    na = torch.where(blend, an_ + wt * (aj - an_), aj)
    nb = torch.where(blend, bn_ + wt * (bj - bn_), bj)
    nc = torch.where(blend, cn + wt * (cj - cn), cj)
    a = torch.cat([a[..., :1], na[..., :-1], a[..., n - 1:n], a[..., n - 1:n]], -1)
    b = torch.cat([b[..., :1], nb[..., :-1], b[..., n - 1:n], b[..., n - 1:n]], -1)
    c = torch.cat([c[..., :1], nc[..., :-1], c[..., n - 1:n], c[..., n - 1:n]], -1)
    return a, b, c


def integ(x, f, start):
    x, f = torch.broadcast_tensors(x, f)
    a, b, c = parcoe(f, x)
    x0, x1 = x[..., :-1], x[..., 1:]
    term = (a[..., :-1] + b[..., :-1] / 2.0 * (x1 + x0) + c[..., :-1] / 3.0 * ((x1 + x0) * x1 + x0 * x0)) * (x1 - x0)
    return torch.cumsum(torch.cat([start.unsqueeze(-1), term], -1), -1)


def deriv(x, f):
    x, f = torch.broadcast_tensors(x, f)
    d0 = (f[..., 1] - f[..., 0]) / (x[..., 1] - x[..., 0])
    dn = (f[..., -1] - f[..., -2]) / (x[..., -1] - x[..., -2])
    s = (torch.abs(x[..., 1] - x[..., 0]) / (x[..., 1] - x[..., 0])).unsqueeze(-1)
    fm, fj, fp = f[..., :-2], f[..., 1:-1], f[..., 2:]
    xm, xj, xp = x[..., :-2], x[..., 1:-1], x[..., 2:]
    scale = torch.maximum(torch.maximum(torch.abs(fm), torch.abs(fj)), torch.abs(fp)) / torch.abs(xj)
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    d1 = (fp - fj) / (xp - xj) / scale
    d = (fj - fm) / (xj - xm) / scale
    tan1 = d1 / (s * torch.sqrt(1.0 + d1 * d1) + 1.0)
    tan = d / (s * torch.sqrt(1.0 + d * d) + 1.0)
    mid = (tan1 + tan) / (1.0 - tan1 * tan) * scale
    return torch.cat([d0[..., None], mid, dn[..., None]], -1)


def map1_rows(xold, fold, xnew, L=None):
    """MAP1 (blended parabolas) for rows with their own XOLD (B, NOLD); fold (K, B, NOLD); xnew (NNEW,).
    Index decisions (L, detached; same as synthe.josh.map1_rows) are taken here unless given. Returns
    (fnew (K, B, NNEW), L)."""
    B, nold = xold.shape
    xn = xnew.expand(B, xnew.shape[-1]).contiguous()
    with torch.no_grad():
        xo = xold.detach()
        given = L is not None
        if not given:
            L = torch.searchsorted(xo[:, 1:].contiguous(), xn, right=True) + 2
        nm = (xo[:, 2:] < xo[:, 1:-1]).any(dim=1) if not given else torch.zeros(B, dtype=torch.bool)
        if bool(nm.any()):
            rows = nm.nonzero().squeeze(1)
            xr, xx = xo[rows], xn[rows]
            lidx = torch.arange(1, nold + 1, device=xo.device).unsqueeze(0)
            start = torch.full((rows.numel(),), 2, dtype=torch.int64, device=xo.device)
            Ls = torch.empty((rows.numel(), xn.shape[1]), dtype=torch.int64, device=xo.device)
            for k in range(xn.shape[1]):
                ok = (lidx >= start.unsqueeze(1)) & (xx[:, k:k + 1] < xr)
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
    x01, x02, x12, x31, x30 = xl - xl1, xl - xl2, xl1 - xl2, xl3 - xl1, xl3 - xl
    K = fold.shape[0]
    fl, fl1, fl2, fl3 = (torch.gather(fold, 2, k.unsqueeze(0).expand(K, -1, -1)) for k in idx)
    one = torch.ones_like(x01)
    s = lambda v, m: torch.where(m, v, one)        # noqa: E731  safe denominators on masked branches
    par = ~lin
    d = (fl1 - fl2) / s(x12, par)
    cbac = fl / s(x01 * x02, par) + (fl2 / s(x02, par) - fl1 / s(x01, par)) / s(x12, par)
    bbac = d - (xl1 + xl2) * cbac
    abac = fl2 - xl2 * d + xl1 * xl2 * cbac
    d2 = (fl - fl1) / x01
    pf = par & ~last
    cfor = fl3 / s(x30 * x31, pf) + (fl1 / s(x31, pf) - fl / s(x30, pf)) / x01
    bfor = d2 - (xl + xl1) * cfor
    afor = fl1 - xl1 * d2 + xl * xl1 * cfor
    acf = torch.abs(cfor)
    den = acf + torch.abs(cbac)
    wt = torch.where(acf != 0, acf / torch.where(acf != 0, den, torch.ones_like(den)), torch.zeros_like(cfor))
    a = torch.where(last, abac, afor + wt * (abac - afor))
    b = torch.where(last, bbac, bfor + wt * (bbac - bfor))
    c = torch.where(last, cbac, cfor + wt * (cbac - cfor))
    al = fl - xl * d2
    a = torch.where(lin, al, a)
    b = torch.where(lin, d2, b)
    c = torch.where(lin, torch.zeros_like(c), c)
    return a + (b + c * xn) * xn, L


class Josh:
    """JOSH(IFSCAT, IFSURF=1) batched over frequencies, float64."""

    def __init__(self, device=None):
        from ..synthe.josh import josh_tables
        t = josh_tables(device)
        self.coefj = t["coefj"].to(D)
        self.ch = t["ch"].to(D)
        self.xtau8 = t["xtau8"].to(D)

    def flux(self, acont, scont, aline, sline, sigmac, sigmal, rhox, ifscat: int, dec: dict | None = None):
        """HNU(1) (B,) for B frequencies; arrays (B, NRHOX). dec: decisions (thick rows, MAP1 intervals,
        surface mask) to reuse; when an empty dict is passed it is filled with this call's decisions."""
        xtau8 = self.xtau8
        B, n = acont.shape
        rhox = rhox.expand(B, n)
        abtot = acont + aline + sigmac + sigmal
        alpha = (sigmac + sigmal) / abtot
        snubar = (acont * scont + aline * sline) / (acont + aline)
        taunu = integ(rhox, abtot, abtot[:, 0] * rhox[:, 0])
        replay = dec is not None and len(dec) > 0
        if ifscat == 0:
            xs, L = map1_rows(taunu, snubar.unsqueeze(0), xtau8, dec["L"] if replay else None)
            if dec is not None and not replay:
                dec["L"] = L
            return xs[0] @ self.ch
        if replay:
            thick = dec["thick"]
        else:
            thick = (taunu[:, 0] > xtau8[NXTAU - 1]).detach()
        out = torch.zeros(B, dtype=D, device=acont.device)
        rg = (~thick).nonzero().squeeze(1)
        rt = thick.nonzero().squeeze(1)
        if dec is not None and not replay:
            dec["thick"] = thick
        if rg.numel():
            tn, sb, al = taunu[rg], snubar[rg], alpha[rg]
            both, L = map1_rows(tn, torch.stack([sb, al]), xtau8, dec["L"] if replay else None)
            xsbar, xalpha = torch.clamp(both[0], min=1e-38), torch.clamp(both[1], 0.0, 1.0)
            if replay:
                low = dec["low"]
            else:
                low = (xtau8.unsqueeze(0) < tn[:, :1]).detach()
                if dec is not None:
                    dec["L"], dec["low"] = L, low
            xsbar = torch.where(low, torch.clamp(sb[:, :1], min=1e-38), xsbar)
            xalpha = torch.where(low, torch.clamp(al[:, :1], 0.0, 1.0), xalpha)
            eye = torch.eye(NXTAU, dtype=D, device=acont.device)
            A = eye.unsqueeze(0) - xalpha.unsqueeze(2) * self.coefj.unsqueeze(0)
            xs = torch.linalg.solve(A, ((1.0 - xalpha) * xsbar).unsqueeze(2)).squeeze(2)
            out = out.index_put((rg,), xs @ self.ch)
        if rt.numel():
            out = out.index_put((rt,), self._thick(taunu[rt], snubar[rt], alpha[rt]))
        return out

    @staticmethod
    def _thick(taunu, snubar, alpha, maxit: int = NXTAU):
        snu = snubar
        h = None
        for _ in range(maxit):
            h = deriv(taunu, snu) / 3.0
            jn = deriv(taunu, h) + snu
            snew = (1.0 - alpha) * snubar + alpha * jn
            err = (torch.abs(snew - snu) / snew).sum(1)
            snu = snew
            if bool((err.detach() < float(np.float32(0.00001))).all()):
                break
        h = deriv(taunu, snu) / 3.0
        return h[:, 0]


class Spectrv:
    """SPECTRV for SURFACE FLUX on a SYNTHE window: flux and continuum flux per wavelength point."""

    def __init__(self, base, win, device=None):
        self.dev = torch.device(device) if device is not None else torch.device("cpu")
        s = base.state0
        n = base.n
        self.n = n
        self.ifscat = int(s["if.ifscat"].reshape(-1)[0])
        self.ifsurf = int(s["if.ifsurf"].reshape(-1)[0])
        if self.ifsurf != 1:
            raise NotImplementedError("differentiable SPECTRV supports SURFACE FLUX only")
        bh = s["bhyd.bhyd"][:n, 0].to(D)
        bc1, bc2 = s["bc.bc1"][:n, 0].to(D), s["bc.bc2"][:n, 0].to(D)
        bs1, bs2 = s["bsi.bsi1"][:n, 0].to(D), s["bsi.bsi2"][:n, 0].to(D)
        # fort.25 of BasicATLAS: RHOXJ=0, PH1=PC1=PSI1=0 -> FSCAT=0, BFUDGE=1
        self.bfudge = torch.pow(bh, 0.0) * torch.pow(bc1 / bc2, 0.0) * torch.pow(bs1 / bs2, 0.0)
        self.fscat = torch.zeros(n, dtype=D)
        wled = np.abs(base.wl).astype(np.float64)
        nedge = len(wled)
        self.nedge = nedge
        self.wledge = torch.as_tensor(wled, device=self.dev)
        self.halfedge = (self.wledge[:-1] + self.wledge[1:]) * 0.5
        dw = self.wledge[1:] - self.wledge[:-1]
        self.deledge = dw * dw * 0.5
        self.win = win
        wave = win.wave()
        self.wave = torch.as_tensor(wave, dtype=D, device=self.dev)
        self.iedge = torch.searchsorted(self.wledge[1:].contiguous(), self.wave.contiguous(), right=True)
        self.josh = Josh(self.dev)
        self.decisions: dict | None = None      # per chunk: JOSH decisions of the base point (record/replay)
        self.freeze = True

    def reset(self):
        self.decisions = None

    def continuum(self, x, wave, iedge):
        """ACONT, SIGMAC (B, n) by SPECTRV's 3-point interpolation of log10 kappa between edges."""
        n = x.n
        ne1 = self.nedge - 1
        ca = x.contabs[:, :3 * ne1].reshape(n, ne1, 3).permute(1, 2, 0)        # (IEDGE, k, J)
        cs = x.contscat[:, :3 * ne1].reshape(n, ne1, 3).permute(1, 2, 0)
        wl_i, wl_i1 = self.wledge[iedge], self.wledge[iedge + 1]
        he, de = self.halfedge[iedge], self.deledge[iedge]
        c1 = ((wave - he) * (wave - wl_i1) / de)[:, None]
        c2 = ((wl_i - wave) * (wave - wl_i1) * 2.0 / de)[:, None]
        c3 = ((wave - wl_i) * (wave - he) / de)[:, None]
        a, s = ca[iedge], cs[iedge]
        acont = torch.pow(10.0, c1 * a[:, 0] + c2 * a[:, 1] + c3 * a[:, 2])
        sigmac = torch.pow(10.0, c1 * s[:, 0] + c2 * s[:, 1] + c3 * s[:, 2])
        return acont, sigmac

    def __call__(self, x, asynth: torch.Tensor, chunk: int = 4096):
        """(flux, continuum) (LENGTH,) for asynth [LENGTH, n] (line opacity incl. stimulated emission)."""
        flux, cont = [], []
        rhox = x.rhox
        record = self.freeze and self.decisions is None
        if record:
            self.decisions = {}
        for c0 in range(0, self.wave.numel(), chunk):
            sl = slice(c0, c0 + chunk)
            wave, iedge = self.wave[sl], self.iedge[sl]
            acont, sigmac = self.continuum(x, wave, iedge)
            freq = 2.99792458e17 / wave
            ehvkt = torch.exp(-freq[:, None] * x.hkt[None, :])
            stim = 1.0 - ehvkt
            f15 = freq / 1.0e15
            bnu = 1.47439e-02 * (f15 * f15 * f15)[:, None] * ehvkt / stim
            asy = asynth[sl]
            zero = torch.zeros_like(acont)
            aline = asy * (1.0 - self.fscat)
            sline = bnu * stim / (self.bfudge - ehvkt)
            sigmal = asy * self.fscat
            if self.freeze:
                dc = self.decisions.setdefault((c0, chunk, "c"), {})
                dl = self.decisions.setdefault((c0, chunk, "l"), {})
            else:
                dc = dl = None
            fc = self.josh.flux(acont, bnu, zero, bnu, sigmac, zero, rhox, self.ifscat, dc)
            fl = self.josh.flux(acont, bnu, aline, sline, sigmac, sigmal, rhox, 1, dl)
            flux.append(fl)
            cont.append(fc)
        return torch.cat(flux), torch.cat(cont)
