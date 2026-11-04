"""He I line profiles of synthe.for: HE1PROF dispatch, HE4471/HE4026/HE4387/HE4921, GRIEM, DIMITRI and
READBCS (Barnard-Cooper-Smith tables from fort.18 = he1tables.dat), vectorised over (layer, point).

Everything is REAL*4 as in the Fortran.  Quantities that depend only on the layer (widths, shifts, Voigt
A, READBCS's PHILAM) are computed once per (line, layer) instead of at every call: the Fortran recomputes
them identically at every call (READBCS caches them under the key J*LINE; the stale-cache quirk is
reproduced by the caller through `philam_source`).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ...io.fortran import ffloat
from ...tables import table
from . import libm32 as L

from .voigt import voigt

SRC = "synthe.patched.for"
F32 = np.float32
F = torch.float32


def f(x):
    return F32(x)


def _parse_fixed(s: str, widths):
    out, i = [], 0
    for w in widths:
        out.append(s[i:i + w])
        i += w
    return out


def _fval(field: str) -> np.float32:
    field = field.strip()
    return F32(ffloat(field)) if field else F32(0)


def _fread(field: str, d: int) -> np.float32:
    """Fortran Fw.d / Ew.d input: an explicit decimal point overrides d; otherwise d implied decimals."""
    s = field.strip()
    if not s:
        return F32(0)
    if "." in s:
        return F32(ffloat(s))
    m = s.upper().replace("D", "E")
    if "E" in m[1:]:
        k = m.index("E", 1)
        mant, ex = m[:k], m[k + 1:]
    else:
        mant, ex = m, "0"
    return F32(float(mant) / 10 ** d * 10 ** int(ex))


@lru_cache(maxsize=None)
def griem_table():
    rows = [r.decode() for r in table(SRC, "griem", "griem0200")]
    n = 42
    wave = np.zeros(n, F32)
    xnelog = np.zeros(n, F32)
    ttab = np.zeros((4, n), F32)
    width = np.zeros((4, n), F32)
    shift = np.zeros((4, n), F32)
    alpha = np.zeros((4, n), F32)
    for il in range(n):
        h = _parse_fixed(rows[5 * il], [5, 10, 10, 5])
        wave[il] = _fread(h[1], 4)
        xnelog[il] = _fread(h[3], 1)
        for i in range(4):
            d = _parse_fixed(rows[5 * il + 1 + i], [10, 10, 10, 10, 10])
            ttab[i, il] = _fread(d[0], 0)
            width[i, il] = _fread(d[1], 6)
            shift[i, il] = _fread(d[2], 6)
            alpha[i, il] = _fread(d[3], 6)
            shift[i, il] = shift[i, il] / width[i, il]
    return dict(wave=wave, xnelog=xnelog, ttab=ttab, width=width, shift=shift, alpha=alpha)


@lru_cache(maxsize=None)
def dimitri_table():
    rows = [r.decode() for r in table(SRC, "dimitri", "dimitri0200")]
    n = 9
    wave = np.zeros(n, F32)
    xnelog = np.zeros(n, F32)
    a = {k: np.zeros((4, n), F32) for k in ("ttab", "width", "shift", "widthp", "shiftp", "widthhe", "shifthe")}
    for il in range(n):
        h = _parse_fixed(rows[5 * il], [5, 10, 10, 5])
        wave[il] = _fread(h[1], 4)
        xnelog[il] = _fread(h[3], 1)
        for i in range(4):
            d = _parse_fixed(rows[5 * il + 1 + i], [7, 9, 9, 9, 9, 9, 9])
            vals = [_fread(d[0], 0)] + [_fread(x, 2) for x in d[1:]]
            for k, v in zip(("ttab", "width", "shift", "widthp", "shiftp", "widthhe", "shifthe"), vals):
                a[k][i, il] = v
            a["shift"][i, il] = a["shift"][i, il] / a["width"][i, il]
            if a["widthp"][i, il] != 0:
                a["shiftp"][i, il] = a["shiftp"][i, il] / a["widthp"][i, il]
            if a["widthhe"][i, il] != 0:
                a["shifthe"][i, il] = a["shifthe"][i, il] / a["widthhe"][i, il]
    a.update(wave=wave, xnelog=xnelog)
    return a


@lru_cache(maxsize=None)
def bcs_tables(path: str):
    """READBCS first-call read of fort.18 (note: 4921 overwrites 4471's PHIHP/PHIHEP, as in the Fortran)."""
    lines = Path(path).read_text().splitlines()
    pos = 2
    dlam = np.zeros((204, 4), F32)
    phihp = np.zeros((4, 7, 142), F32)
    phihep = np.zeros((4, 7, 142), F32)
    phi4026 = np.zeros((4, 8, 196), F32)
    phi4387 = np.zeros((4, 8, 204), F32)
    w = [1, 5, 8] + [7] * 8

    def rd():
        nonlocal pos
        fl = _parse_fixed(lines[pos].ljust(80), w)
        pos += 1
        return _fread(fl[2], 2), [_fread(x, 3) for x in fl[3:]]
    for il in range(142):
        for ne in range(7):
            dwl, phi = rd()
            phihp[:, ne, il] = phi[:4]
            phihep[:, ne, il] = phi[4:8]
        dlam[il, 0] = dwl - F32(150.)
    pos += 2
    for il in range(196):
        for ne in range(8):
            dwl, phi = rd()
            phi4026[:, ne, il] = phi[:4]
        dlam[il, 1] = dwl - F32(150.)
    pos += 2
    for il in range(204):
        for ne in range(8):
            dwl, phi = rd()
            phi4387[:, ne, il] = phi[:4]
        dlam[il, 2] = dwl - F32(150.)
    pos += 2
    for il in range(142):
        for ne in range(7):
            dwl, phi = rd()
            phihp[:, ne, il] = phi[:4]
            phihep[:, ne, il] = phi[4:8]
        dlam[il, 3] = dwl - F32(150.)
    return dict(dlam=dlam, phihp=phihp, phihep=phihep, phi4026=phi4026, phi4387=phi4387)


def parcoe(fv: np.ndarray, x: np.ndarray):
    """PARCOE(F,X,A,B,C,N) vectorised over leading axes of F (REAL*4); x is 1-D."""
    n = x.shape[0]
    fv = fv.astype(F32)
    sh = fv.shape[:-1] + (n,)
    A = np.zeros(sh, F32)
    B = np.zeros(sh, F32)
    C = np.zeros(sh, F32)
    B[..., 0] = (fv[..., 1] - fv[..., 0]) / (x[1] - x[0])
    A[..., 0] = fv[..., 0] - x[0] * B[..., 0]
    n1 = n - 1
    C[..., n - 1] = 0
    B[..., n - 1] = (fv[..., n - 1] - fv[..., n1 - 1]) / (x[n - 1] - x[n1 - 1])
    A[..., n - 1] = fv[..., n - 1] - x[n - 1] * B[..., n - 1]
    if n == 2:
        return A, B, C
    j = np.arange(1, n1)          # Fortran J = 2..N1 (0-based j)
    j1 = j - 1
    jp = j + 1
    D = (fv[..., j] - fv[..., j1]) / (x[j] - x[j1])
    C[..., j] = fv[..., jp] / ((x[jp] - x[j]) * (x[jp] - x[j1])) - fv[..., j] / ((x[j] - x[j1]) * (x[jp] - x[j])) + \
        fv[..., j1] / ((x[j] - x[j1]) * (x[jp] - x[j1]))
    B[..., j] = D - (x[j] + x[j1]) * C[..., j]
    A[..., j] = fv[..., j1] - x[j1] * D + x[j] * x[j1] * C[..., j]
    C[..., 1] = 0
    B[..., 1] = (fv[..., 2] - fv[..., 1]) / (x[2] - x[1])
    A[..., 1] = fv[..., 1] - x[1] * B[..., 1]
    C[..., 2] = 0
    B[..., 2] = (fv[..., 3] - fv[..., 2]) / (x[3] - x[2])
    A[..., 2] = fv[..., 2] - x[2] * B[..., 2]
    # loop 2: blend with the (still unmodified) J+1 coefficients
    Cj, Cj1 = C[..., 1:n1].copy(), C[..., 2:n1 + 1].copy()
    Aj, Aj1 = A[..., 1:n1].copy(), A[..., 2:n1 + 1].copy()
    Bj, Bj1 = B[..., 1:n1].copy(), B[..., 2:n1 + 1].copy()
    with np.errstate(all="ignore"):
        wt = np.abs(Cj1) / (np.abs(Cj1) + np.abs(Cj))
        nz = Cj != 0
        A[..., 1:n1] = np.where(nz, Aj1 + wt * (Aj - Aj1), Aj)
        B[..., 1:n1] = np.where(nz, Bj1 + wt * (Bj - Bj1), Bj)
        C[..., 1:n1] = np.where(nz, Cj1 + wt * (Cj - Cj1), Cj)
    A[..., n1 - 1] = A[..., n - 1]
    B[..., n1 - 1] = B[..., n - 1]
    C[..., n1 - 1] = C[..., n - 1]
    return A, B, C


def integ(x: np.ndarray, fv: np.ndarray, start=0.0) -> np.ndarray:
    """INTEG(X,F,FINT,N,START): sequential REAL*4 sum (vectorised over leading axes of F)."""
    A, B, C = parcoe(fv, x)
    n = x.shape[0]
    fint = np.full(fv.shape[:-1], F32(start), F32)
    for i in range(n - 1):
        xs = x[i + 1] + x[i]
        term = (A[..., i] + B[..., i] / F32(2) * xs +
                C[..., i] / F32(3) * (xs * x[i + 1] + x[i] * x[i])) * (x[i + 1] - x[i])
        fint = (fint + term).astype(F32)
    return fint


class HeI:
    """He I profile functions for one run (layer data as float32 numpy arrays over the NRHOX layers)."""

    def __init__(self, lay: dict, h1tab: torch.Tensor, fort18: str | None, mode: str = "stock"):
        self.lay = {k: np.asarray(v, F32) for k, v in lay.items()}  # t, xne, xnfph2, xnfhe2
        self.h = h1tab
        self.fort18 = fort18
        self.mode = mode
        self.device = h1tab.device

    # ------------------------------------------------------------ dispatch
    @staticmethod
    def kind(wl4: np.float32) -> str:
        line = int(np.trunc(wl4 + F32(1.)))
        if line == 448:
            return "he4471"
        if line == 403:
            line = int(np.trunc(wl4 + f(.4)))
            if line == 402:
                line = int(np.trunc(wl4 + F32(1.)))
            else:
                return "he4026"
        elif line == 439:
            return "he4387"
        elif line == 493:
            return "he4921"
        if line in (382, 387, 388, 393, 394, 401, 403, 415, 417):
            return "dimitri"
        return "griem"

    # ------------------------------------------------------------ helpers
    def _powf(self, x, y):
        return L.spowf(torch.as_tensor(np.asarray(x, F32)), torch.as_tensor(np.asarray(y, F32))).numpy()

    def _common_tail(self, e, temp, xnfhp, w, d, alf, wl4):
        """XX=XNFHP/E ... SIGMA ... X, WTOT, DTOT (shared by HE4471/4026/4387/4921, GRIEM)."""
        xx = xnfhp / e
        vm1 = f(8.78) * (xx + F32(2) * (F32(1) - xx)) / np.sqrt(temp.astype(np.float64)).astype(F32)
        rhom = F32(1) / self._powf(f(4.19) * e, F32(1.) / F32(3.))
        wl10 = wl4 * F32(10.)
        sigma = f(1.885e14) * w * rhom * vm1 / (wl10 * wl10)
        x = self._powf(alf, F32(8.) / F32(9.)) / self._powf(sigma, F32(1.) / F32(3.))
        wtot = w * (F32(1) + f(1.36) * x)
        wtot = wtot * f(0.1)
        dtot = w * d * (F32(1) + f(2.36) * x / np.abs(d))
        dtot = dtot * f(0.1)
        return wtot, dtot

    def _interp_t(self, temp, ts):
        """IT = first I in 2..4 with TS(I) > TEMP (else 4); X = (TEMP-TS(IT-1))/(TS(IT)-TS(IT-1))."""
        it = np.full(temp.shape, 4)
        for i in (4, 3, 2):
            it = np.where(ts[i - 1] > temp, i, it)
        x = (temp - ts[it - 2]) / (ts[it - 1] - ts[it - 2])
        return it, x.astype(F32)

    # ------------------------------------------------------------ BCS-type lines (4471/4026/4387/4921)
    def bcs_layer(self, name: str, wl4, dopwl):
        """Per-layer constants for HE4471/4026/4387/4921 (analytic branch) and the READBCS flag."""
        lay = self.lay
        e = lay["xne"]
        temp = np.minimum(np.maximum(lay["t"], f(5.0e3)), f(4.0e4))
        g = lambda v: table(SRC, name, v).astype(F32)  # noqa: E731
        ws, ds, alfs, ts = g("ws"), g("ds"), g("alfs"), g("ts")
        den = F32(table(SRC, name, "den"))
        it, x = self._interp_t(temp, ts)
        xx = e / den
        w = xx * (x * ws[it - 1] + (F32(1) - x) * ws[it - 2])
        d = x * ds[it - 1] + (F32(1) - x) * ds[it - 2]
        alf = self._powf(xx, f(0.25)) * (x * alfs[it - 1] + (F32(1) - x) * alfs[it - 2])
        wtot, dtot = self._common_tail(e, temp, lay["xnfph2"], w, d, alf, wl4)
        a = wtot / dopwl
        thr = {"he4471": f(1.0e13), "he4026": f(1.e14), "he4921": f(1.0e13), "he4387": None}[name]
        use_bcs = (e > thr) if thr is not None else np.zeros(e.shape, bool)
        return dict(a=a, dtot=dtot, use_bcs=use_bcs, temp=temp)

    def bcs_profile(self, name, j, wave4, wl4, dopwl_j, lc, philam_j):
        """Profile values at (layer j, wave4) for HE4471/4026/4387/4921.

        lc: bcs_layer() dict (numpy per layer); philam_j: PHILAM rows [.., NDLAM] for READBCS layers
        (any value where unused); dopwl_j: DOPWL per element."""
        dev = self.device
        t = lambda a: torch.as_tensor(np.asarray(a, F32), device=dev)  # noqa: E731
        a = t(lc["a"])[j]
        dtot = t(lc["dtot"])[j]
        wwd = (wave4 - float(wl4)) - dtot
        h = self.h
        if name == "he4471":
            sh = [(f(-.0184), 9.), (f(.0013), 12.), (f(.0010), 4.), (f(.0029), 180.), (f(.0025), None)]
        elif name == "he4026":
            sh = [(f(-.0148), 9.), (f(.0012), 12.), (f(.0011), 4.), (f(.0025), 180.), (f(.0023), None)]
        else:
            sh = None
        if sh is not None:
            acc = None
            for s, dv in sh:
                v = voigt(torch.abs(wwd + float(s)) / dopwl_j, a, h)
                term = v / dv if dv is not None else v * 11.0 / 20.0
                acc = term if acc is None else acc + term
            prof = acc * 1.0
        else:
            prof = voigt(torch.abs(wwd) / dopwl_j, a, h)
        if name == "he4387":
            return prof
        use = torch.as_tensor(lc["use_bcs"], device=dev)[j]
        if bool(use.any()):
            line = {"he4471": 1, "he4026": 2, "he4387": 3, "he4921": 4}[name]
            phihe = self.readbcs_interp(line, (wave4 - float(wl4)), philam_j)
            bcs = 1.772453 * phihe * dopwl_j * 10.0
            prof = torch.where(use, bcs, prof)
        return prof

    # ------------------------------------------------------------ READBCS
    def philam(self, line: int) -> np.ndarray:
        """PHILAM(1..NDLAM) after the setup part of READBCS, for every layer: float32 [nrhox, 204]."""
        tb = bcs_tables(self.fort18)
        lay = self.lay
        nd = int(table(SRC, "readbcs", "ndlam")[line - 1])
        nxne = int(table(SRC, "readbcs", "nxne")[line - 1])
        xne1 = F32(table(SRC, "readbcs", "xne1")[line - 1])
        temp = np.minimum(np.maximum(lay["t"], f(5.0e3)), f(4.0e4))   # TEMP as passed by HE4471/...
        xne = lay["xne"]
        at = L.slog10f(torch.as_tensor(temp)).numpy()
        bt = (at - f(3.698970)) / f(.3010300) + F32(1.)
        it = np.trunc(bt + f(0.00001)).astype(np.int64)
        it = np.maximum(np.minimum(it, 3), 1)
        wt = (bt - it.astype(F32)).astype(F32)
        ap = L.slog10f(torch.as_tensor(xne)).numpy()
        ap = np.maximum(xne1, ap)
        bp = (ap - xne1) / f(0.5) + F32(1.)
        ip = np.trunc(bp + f(0.00001)).astype(np.int64)
        ip = np.maximum(np.minimum(ip, nxne - 1), 1)
        wp = (bp - ip.astype(F32)).astype(F32)
        c1w1w = (F32(1) - wp) * (F32(1) - wt)
        c1ww = (F32(1) - wp) * wt
        cw1w = wp * (F32(1) - wt)
        cww = wp * wt

        def mix(tab):  # tab [4, nne, ndl] -> exponent [nrhox, nd]
            return (c1w1w[:, None] * tab[it - 1, ip - 1, :nd] + c1ww[:, None] * tab[it, ip - 1, :nd] +
                    cw1w[:, None] * tab[it - 1, ip, :nd] + cww[:, None] * tab[it, ip, :nd]).astype(F32)
        pw = lambda ex: L.spowf(torch.full(ex.shape, 10.0), torch.as_tensor(ex)).numpy()  # noqa: E731
        if line in (1, 4):
            xxh = lay["xnfph2"] / xne
            xxhe = lay["xnfhe2"] / xne
            ph = xxh[:, None] * pw(mix(tb["phihp"])) + xxhe[:, None] * pw(mix(tb["phihep"]))
        elif line == 2:
            ph = pw(mix(tb["phi4026"]))
        else:
            ph = pw(mix(tb["phi4387"]))
        ph = ph.astype(F32)
        x = tb["dlam"][:nd, line - 1]
        phinorm = integ(x, ph, 0.0)
        r = (ph / phinorm[:, None]).astype(F32)
        if self.mode == "stock":
            lg = L.loopvec(L.vlog10f, L.slog10f, torch.as_tensor(r)).numpy()
        else:
            lg = L.slog10f(torch.as_tensor(r)).numpy()
        out = np.zeros((len(temp), 204), F32)
        out[:, :nd] = lg
        return out

    def readbcs_interp(self, line: int, dlnm: torch.Tensor, philam_j: torch.Tensor) -> torch.Tensor:
        tb = bcs_tables(self.fort18)
        nd = int(table(SRC, "readbcs", "ndlam")[line - 1])
        dlam = torch.as_tensor(tb["dlam"][:nd, line - 1], device=dlnm.device)
        dl = dlnm * 10.0
        # first I in 2..ND with not(DL > DLAM(I) and I < ND)  ->  I = min(ND, 1 + #{k<ND-1 : DL > DLAM(k)} ...)
        gt = dl.unsqueeze(-1) > dlam[1:nd - 1]                    # I = 2..ND-1
        first_false = torch.where(~gt, torch.arange(2, nd, device=dl.device), torch.full_like(gt, nd, dtype=torch.long))
        i = first_false.min(dim=-1).values                       # 1-based I
        d1 = dlam[i - 1]
        d0 = dlam[i - 2]
        a = (d1 - dl) / (d1 - d0)
        b = (dl - d0) / (d1 - d0)
        p0 = torch.gather(philam_j, -1, (i - 2).unsqueeze(-1)).squeeze(-1)
        p1 = torch.gather(philam_j, -1, (i - 1).unsqueeze(-1)).squeeze(-1)
        ex = a * p0 + b * p1
        return L.spowf(torch.full_like(ex, 10.0), ex)

    # ------------------------------------------------------------ GRIEM / DIMITRI
    def griem_layer(self, wl4, dopwl, gammar, gammas):
        tb = griem_table()
        lay = self.lay
        il = None
        for k in range(42):
            if wl4 > tb["wave"][k] - f(.1) and wl4 < tb["wave"][k] + f(.1):
                il = k
                break
        if il is None:
            a = (F32(gammar) + F32(gammas) * lay["xne"]) / (dopwl / wl4)
            return dict(il=None, a=a.astype(F32), dtot=np.zeros_like(a))
        e = lay["xne"]
        temp = np.minimum(np.maximum(lay["t"], f(5.0e3)), f(8.0e4))
        it, x = self._interp_t(temp, tb["ttab"][:, il])
        xx = e / F32(L.spowf1(10.0, tb["xnelog"][il]))
        wd, sh, al = tb["width"][:, il], tb["shift"][:, il], tb["alpha"][:, il]
        w = xx * (x * wd[it - 1] + (F32(1) - x) * wd[it - 2])
        d = x * sh[it - 1] + (F32(1) - x) * sh[it - 2]
        alf = self._powf(xx, f(0.25)) * (x * al[it - 1] + (F32(1) - x) * al[it - 2])
        wtot, dtot = self._common_tail(e, temp, lay["xnfph2"], w, d, alf, wl4)
        a = wtot / dopwl + F32(gammar) / (dopwl / wl4)
        return dict(il=il, a=a.astype(F32), dtot=dtot.astype(F32))

    def dimitri_layer(self, wl4, dopwl, gammar, gammas):
        tb = dimitri_table()
        lay = self.lay
        il = None
        for k in range(9):
            if wl4 > tb["wave"][k] - f(.1) and wl4 < tb["wave"][k] + f(.1):
                il = k
                break
        if il is None:
            a = (F32(gammar) + F32(gammas) * lay["xne"]) / (dopwl / wl4)
            return dict(il=None, a=a.astype(F32), dtot=np.zeros_like(a))
        e = lay["xne"]
        temp = np.minimum(np.maximum(lay["t"], f(5.0e3)), f(8.0e4))
        it, x = self._interp_t(temp, tb["ttab"][:, il])
        p10 = F32(L.spowf1(10.0, tb["xnelog"][il]))
        xx = e / p10
        xxh = lay["xnfph2"] / p10
        xxhe = lay["xnfhe2"] / p10
        g = lambda k: tb[k][:, il]  # noqa: E731
        lin = lambda arr: x * arr[it - 1] + (F32(1) - x) * arr[it - 2]  # noqa: E731
        w = xx * lin(g("width"))
        wp = xxh * lin(g("widthp"))
        whe = xxhe * lin(g("widthhe"))
        d = lin(g("shift"))
        dp = lin(g("shiftp"))
        dhe = lin(g("shifthe"))
        wtot = w + wp + whe
        wtot = wtot * f(0.1)
        halfw = wtot / F32(2.)
        if il == 7:
            dp = f(0.4) * dp
        dtot = w * d + wp * dp + whe * dhe
        dtot = dtot * f(0.1)
        a = halfw / dopwl + F32(gammar) / (dopwl / wl4)
        return dict(il=il, a=a.astype(F32), dtot=dtot.astype(F32))

    def gd_profile(self, lc, j, wave4, wl4, dopwl_j):
        a = torch.as_tensor(lc["a"], device=self.device)[j]
        dtot = torch.as_tensor(lc["dtot"], device=self.device)[j]
        if lc["il"] is None:
            v = torch.abs(wave4 - float(wl4)) / dopwl_j
        else:
            v = torch.abs((wave4 - float(wl4)) - dtot) / dopwl_j
        return voigt(v, a, self.h)
