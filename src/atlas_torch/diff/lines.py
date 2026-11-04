"""Differentiable SYNTHE line opacity (synthe.for main line loop over fort.12) in float64.

Split into *decisions* and *values*:

* Decisions (`LineSet.decide`, detached, computed once at the base point): which (line, layer) pairs pass
  SYNTHE's two cutoff tests (KAPPA0 < KAPMIN, KAPPA0*BOLTZ < KAPMIN), where each profile core stops (first
  point below KAPMIN within 10 Doppler widths) and how far the Lorentzian far wing runs (MAXSTEP). They are taken
  with the exact path's REAL*4 arithmetic and tables (atlas_torch.synthe.lineopac: FASTEX, H0/H1/H2 tables,
  VOIGT), so the set of contributions is the exact path's (up to the float64 continuum used for KAPMIN).
  These are discrete choices: the function is piecewise smooth and the decisions are held fixed under
  differentiation (a derivative ignores contributions switching on/off at the 1e-4*continuum cutoff).

* Values (`LineSet.opacity`, attached): every contribution recomputed from the differentiable populations
  (XNFPEL/RHO), Doppler widths sqrt(DOPPLE**2 + (TURBV/c)**2), XNE, T, HCKT with the Fortran's formulas in
  float64. The tabulated functions are evaluated as the Fortran does (same table entry, chosen with the same
  REAL*4 index arithmetic), with the derivative of the function the table samples (straight_through):
    FASTEX(x)  -> exp(-x);  H0TAB(v) -> exp(-v^2);  H1TAB, H2TAB -> piecewise-linear interpolation of the tables
  (profile="exact", default: values = the exact path's up to float32 vs float64 rounding).
  profile="smooth" evaluates the smooth functions themselves (no table rounding; deviates from the exact path by
  the table resolution, see RESULTS) -- a function whose finite differences match its derivatives tightly.

ASYNTH(J, point) = BUFFER * (1 - exp(-FREQ*HKT(J))) is returned as a [LENGTH, NRHOX] tensor.

Not differentiated / not supported: lines with ALPHA != 0 (ABO van der Waals) raise NotImplementedError;
velocity fields (DECKJ) are honoured in the decisions (NVSHIFT) and in the values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..synthe.lineopac import fortio
from ..synthe.lineopac.tabs import extab_common, fastex, h1tab_common
from ..synthe.lineopac.voigt import voigt as voigt32
from .native import straight_through

F = torch.float32
D = torch.float64
MAXPROF = 10000
C_KMS = 299792.458


def powi_np(x: float, n) -> np.ndarray:
    """libgcc __powidf2(x, n) elementwise (the Fortran's WBEGIN*RATIO**(NBUFF-1))."""
    n = np.asarray(n, dtype=np.int64)
    m = np.abs(n)
    y = np.where(m % 2 == 1, x, 1.0)
    xx = float(x)
    m = m >> 1
    while np.any(m):
        xx = xx * xx
        y = np.where(m % 2 == 1, y * xx, y)
        m = m >> 1
    return np.where(n < 0, 1.0 / y, y)


@dataclass
class Window:
    """The SYNBEG run parameters (fort.93) and SYNTHE's wavelength grid."""
    wlbeg: float
    wlend: float
    resolu: float
    ratio: float
    ratiolg: float
    length: int
    wbegin: float
    cutoff: float
    turbv: float
    ifvac: int
    nlines: int
    n19: int
    velshift: np.ndarray        # (kw,) DECKJ(1,J) [km/s]

    @classmethod
    def from_fort93(cls, path) -> "Window":
        p = fortio.read_fort93(path)
        ratiolg = p["ratiolg"]
        ix = int(math.log(p["wlbeg"]) / ratiolg)
        wbegin = math.exp(ix * ratiolg)
        if wbegin < p["wlbeg"]:
            ix += 1
            wbegin = math.exp(ix * ratiolg)
        return cls(p["wlbeg"], p["wlend"], p["resolu"], p["ratio"], ratiolg, p["length"], wbegin,
                   float(np.float32(p["cutoff"])), float(np.float32(p["turbv"])), p["ifvac"], p["nlines"],
                   p["n19"], p["deckj"][0].astype(np.float32))

    def wave(self) -> np.ndarray:
        return self.wbegin * powi_np(self.ratio, np.arange(self.length))


# ------------------------------------------------------------------------------------- tables (float64 views)

class Tables:
    def __init__(self, mode: str = "stock", device=None):
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.extab32 = torch.as_tensor(extab_common(mode), device=dev)
        self.h32 = torch.as_tensor(h1tab_common(mode), device=dev)
        self.h = self.h32.to(D)
        self.ni = self.h.numel()
        h = self.h
        self.h1 = h[2001:4002].contiguous()
        self.h2 = h[4002:6003].contiguous()
        self.d1 = torch.cat([self.h1[1:] - self.h1[:-1], torch.zeros(1, dtype=D, device=dev)])
        self.d2 = torch.cat([self.h2[1:] - self.h2[:-1], torch.zeros(1, dtype=D, device=dev)])

    # tabulated values (exact path entries) ---------------------------------------------------------
    def fastex_tab(self, x32: torch.Tensor) -> torch.Tensor:
        return fastex(x32, self.extab32).to(D)

    def tab_index(self, v32: torch.Tensor) -> torch.Tensor:
        """VOIGT's IV = INT(V*200+1.5) (REAL*4), 0-based H0TAB position."""
        iv = torch.trunc(v32 * 200.0 + 1.5)
        iv = torch.nan_to_num(iv, nan=1.0, posinf=1e9, neginf=-1e9).clamp(-1e9, 1e9).to(torch.int64)
        return iv - 1

    def h012(self, i0: torch.Tensor):
        n = self.ni
        h = self.h
        return h[i0.clamp(0, n - 1)], h[(i0 + 2001).clamp(0, n - 1)], h[(i0 + 4002).clamp(0, n - 1)]

    # smooth counterparts -------------------------------------------------------------------------
    def h12_smooth(self, v: torch.Tensor, need_h2: bool = True):
        """Piecewise-linear interpolation of H1TAB, H2TAB at x = 200 v (continuous in v)."""
        x = (v * 200.0).clamp(0.0, 2000.0)
        i = torch.floor(x.detach()).clamp(0, 1999).to(torch.int64)
        fr = x - i.to(D)
        h1 = self.h1[i] + self.d1[i] * fr
        h2 = self.h2[i] + self.d2[i] * fr if need_h2 else None
        return h1, h2

    def h01_tab(self, i0: torch.Tensor):
        i0 = i0.clamp(0, 2000)
        return self.h[i0], self.h1[i0]


def _st(tab, smooth, mode):
    return smooth if mode == "smooth" else straight_through(tab, smooth)


def voigt64(v: torch.Tensor, a: torch.Tensor, tabs: Tables, i0: torch.Tensor | None, mode: str) -> torch.Tensor:
    """VOIGT(V,A) of synthe.for in float64. i0: the exact path's table position (from REAL*4 V), or None."""
    v, a = torch.broadcast_tensors(v, a)
    if i0 is None:
        i0 = tabs.tab_index(v.detach().to(F))
    h0t, h1t, h2t = tabs.h012(i0)
    h1s, h2s = tabs.h12_smooth(v)
    h0 = _st(h0t, torch.exp(-v * v), mode)
    h1 = _st(h1t, h1s, mode)
    h2 = _st(h2t, h2s, mode)
    vv = v * v
    small = a < 0.2
    # A < 0.2
    near = (h2 * a + h1) * a + h0
    vsafe = torch.where(v > 10.0, v, torch.ones_like(v))
    farv = (0.5642 * a) / (vsafe * vsafe)
    r_small = torch.where(v > 10.0, farv, near)
    # 0.2 <= A <= 1.4, A+V <= 3.2
    hh1 = h1 + h0 * 1.12838
    hh2 = h2 + hh1 * 1.12838 - h0
    hh3 = (1.0 - h2) * .37613 - hh1 * .66667 * vv + hh2 * 1.12838
    hh4 = (3.0 * hh3 - hh1) * .37613 + h0 * .66667 * vv * vv
    poly = ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0)
    fac = ((-.122727278 * a + .532770573) * a - .96284325) * a + .979895032
    r_mid = poly * fac
    # far
    aa = a * a
    u = (aa + vv) * 1.4142
    vo = a * .79788 / u
    aau = aa / u
    vvu = vv / u
    uu = u * u
    corr = (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa) / uu + 1.0
    r_far = torch.where(a > 100.0, vo, corr * vo)
    mid = ~small & (a <= 1.4) & ~(a + v > 3.2)
    return torch.where(small, r_small, torch.where(mid, r_mid, r_far))


# --------------------------------------------------------------------------------------------- the line set

@dataclass
class Pairs:
    """Decisions for the (line, layer) pairs that pass the cutoffs (1-D arrays over pairs, torch, detached)."""
    j: torch.Tensor          # layer (0-based)
    line: torch.Tensor       # index into LineSet.lines
    nbs: torch.Tensor        # NBUFF + NVSHIFT(J) (1-based centre point)
    nstep: torch.Tensor      # points on each side
    n10: torch.Tensor        # N10DOP
    small: torch.Tensor      # ADAMP < 0.2
    tabstep: torch.Tensor    # REAL*4 TABSTEP (core table walk)
    dvoigt: torch.Tensor     # REAL*4 DVOIGT


def _cont_log(ablog: np.ndarray, wled: np.ndarray, wave: np.ndarray, i0: np.ndarray) -> np.ndarray:
    """SYNTHE's CONTINUUM interpolation (log10, before 10**): ablog (nr, 3, nedge-1)."""
    half = (wled[:-1] + wled[1:]) * .5
    dele = (wled[1:] - wled[:-1]) ** 2 * .5
    w = wave[None, :]
    a1, a2, a3 = (ablog[:, k, i0] for k in range(3))
    wl0, wl1, hf, de = wled[i0], wled[i0 + 1], half[i0], dele[i0]
    return ((w - hf) * (w - wl1) * a1 + (wl0 - w) * (w - wl1) * 2. * a2 + (w - wl0) * (w - hf) * a3) / de


class LineSet:
    """fort.12 lines of a SYNTHE work directory with their cutoff decisions at a base point."""

    def __init__(self, workdir, win: Window | None = None, device=None, mode: str = "stock"):
        self.wd = Path(workdir)
        self.win = win or Window.from_fort93(self.wd / "fort.93")
        self.dev = torch.device(device) if device is not None else torch.device("cpu")
        self.tabs = Tables(mode, self.dev)
        n = self.win.nlines
        self.lines = fortio.read_fort12(self.wd / "fort.12", n) if n > 0 else np.zeros(0, fortio.FORT12)
        if np.any(self.lines["alpha"] != 0):
            raise NotImplementedError("fort.12 lines with ALPHA != 0 are not supported by the differentiable path")
        lt = lambda k, dt: torch.as_tensor(self.lines[k].astype(dt), device=self.dev)   # noqa: E731
        self.nbuff = lt("nbuff", np.int64)
        self.nel = lt("nelion", np.int64) - 1
        self.congf = lt("congf", np.float32)
        self.elo = lt("elo", np.float32)
        self.gr = lt("gamrf", np.float32)
        self.gs = lt("gamsf", np.float32)
        self.gw = lt("gamwf", np.float32)
        self.pairs: Pairs | None = None

    # ----------------------------------------------------------------- quantities from XNFPELSYN
    def layer_quantities(self, x, vturb):
        """Per-layer arrays SYNTHE derives from fort.10 (+ TURBV), attached. x: diff.xnf.XnfResult."""
        n = x.n
        qx = x.xnfpel.permute(0, 2, 1).reshape(n, -1)              # QXNFPEL(J, NELION) (Fortran (6,139) order)
        qd = x.dopple.permute(0, 2, 1).reshape(n, -1)
        tv = torch.as_tensor(vturb, dtype=D) / C_KMS
        qd2 = torch.sqrt(qd * qd + tv * tv)
        qxnfdop = qx / x.rho[:, None] / qd2
        txn = (x.xnfh + .42 * x.xnfhe[:, 0] + .85 * x.xnfh2) * torch.pow(x.t / 10000., float(np.float32(.3)))
        return dict(qxnfdop=qxnfdop, dop=qd2, txn=txn, xne=x.xne, hckt=x.hckt, hkt=x.hkt, t=x.t)

    def continuum_log(self, x, f10_edges) -> np.ndarray:
        """log10 CONTINUUM [nr, LENGTH] for the cutoff decisions (detached)."""
        wled = np.abs(f10_edges["wl"]).astype(np.float64)
        nedge = len(wled)
        n = x.n
        tot = torch.log10(torch.pow(10.0, x.contabs.detach()) + torch.pow(10.0, x.contscat.detach())).cpu().numpy()
        ablog = tot[:, :3 * (nedge - 1)].reshape(n, nedge - 1, 3).transpose(0, 2, 1)
        wave = self.win.wave()
        iedge = np.maximum(np.searchsorted(wled, wave, side="right"), 1)
        return _cont_log(ablog, wled, wave, iedge - 1)

    # ----------------------------------------------------------------- decisions
    def decide(self, x, f10_edges, vturb=None, line_chunk: int = 50000) -> Pairs:
        win = self.win
        dev = self.dev
        vturb = win.turbv if vturb is None else float(vturb)
        q = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in self.layer_quantities(x, vturb).items()}
        n = x.n
        L = win.length
        cont = torch.as_tensor(np.power(10.0, self.continuum_log(x, f10_edges)).astype(np.float32), device=dev)
        self.cont32 = cont
        qxnfdop = q["qxnfdop"]
        dop32 = q["dop"].to(F)
        xne32, txn32, hckt32 = q["xne"].to(F), q["txn"].to(F), q["hckt"].to(F)
        nvs = torch.as_tensor(np.trunc(win.resolu * win.velshift[:n].astype(np.float64) / C_KMS + .5).astype(np.int64),
                              device=dev)
        self.nvs = nvs
        out = {k: [] for k in Pairs.__dataclass_fields__}
        h32 = self.tabs.h32
        ni = h32.numel()
        for c0 in range(0, len(self.lines), line_chunk):
            sl = slice(c0, c0 + line_chunk)
            nb, nel = self.nbuff[sl], self.nel[sl]
            kmidx = torch.clamp(nb, 1, L) - 1
            kappa0 = (self.congf[sl].to(D)[None, :] * qxnfdop[:, nel]).to(F)
            kapmin = cont[:, kmidx] * win.cutoff
            jj, ii = torch.nonzero(~(kappa0 < kapmin), as_tuple=True)
            if not jj.numel():
                continue
            k0 = kappa0[jj, ii] * fastex(self.elo[sl][ii] * hckt32[jj], self.tabs.extab32)
            km = kapmin[jj, ii]
            ok = ~(k0 < km)
            jj, ii, k0, km = jj[ok], ii[ok], k0[ok], km[ok]
            if not jj.numel():
                continue
            gi = ii + c0
            ne_ = self.nel[gi]
            dop = dop32[jj, ne_]
            adamp = (self.gr[gi] + self.gs[gi] * xne32[jj] + self.gw[gi] * txn32[jj]) / dop
            small = adamp < 0.2
            dopd = dop.to(D) * win.resolu
            n10 = torch.trunc(10.0 * dopd).to(torch.long)
            tabstep = torch.div(torch.full_like(dopd, 200.0), dopd).to(F)
            dvoigt = (torch.div(torch.ones_like(dopd), dop.to(D)) / win.resolu).to(F)
            # core walk (REAL*4, as synthe_prog._profiles): first point below KAPMIN within N10DOP
            npair = jj.numel()
            kmax = int(n10.max())
            exitk = torch.full((npair,), -1, dtype=torch.long, device=dev)
            last = torch.zeros(npair, dtype=F, device=dev)
            tabi = torch.full((npair,), 1.5, dtype=F, device=dev)
            alive = torch.ones(npair, dtype=torch.bool, device=dev)
            for k in range(1, kmax + 1):
                tabi = tabi + tabstep
                it = torch.trunc(tabi).to(torch.long)
                pr = k0 * (h32[(it - 1).clamp(0, ni - 1)] + adamp * h32[(it + 2000).clamp(0, ni - 1)])
                if not bool(small.all()):
                    lr = ~small
                    v = float(k) * dvoigt[lr]
                    pr = pr.clone()
                    pr[lr] = k0[lr] * voigt32(v, adamp[lr], h32)
                valid = k <= n10
                ex = alive & valid & (pr < km)
                exitk = torch.where(ex, torch.full_like(exitk, k), exitk)
                last = torch.where(valid & (k == n10), pr, last)
                alive = alive & ~ex & (n10 > k)
                if not bool(alive.any()):
                    break
            far = exitk < 0
            n10f = n10.to(F)
            xx = torch.where(n10 > 0, last * (n10f * n10f), torch.zeros_like(last))
            ms = torch.sqrt((xx / km).to(D)).to(F) + 1.0
            ms = torch.clamp(torch.trunc(torch.nan_to_num(ms, posinf=1e9)).to(torch.long), max=MAXPROF)
            nstep = torch.where(far, ms, exitk)
            nbs = self.nbuff[gi] + nvs[jj]
            for k, v in (("j", jj), ("line", gi), ("nbs", nbs), ("nstep", nstep), ("n10", n10), ("small", small),
                         ("tabstep", tabstep), ("dvoigt", dvoigt)):
                out[k].append(v)
        cat = lambda v, dt: torch.cat(v) if v else torch.zeros(0, dtype=dt, device=dev)   # noqa: E731
        P = Pairs(cat(out["j"], torch.long), cat(out["line"], torch.long), cat(out["nbs"], torch.long),
                  cat(out["nstep"], torch.long), cat(out["n10"], torch.long), cat(out["small"], torch.bool),
                  cat(out["tabstep"], F), cat(out["dvoigt"], F))
        # sort by the number of core points the values need (min(N10DOP, NSTEP); N10DOP for far wings) so that
        # chunks of the value pass are rectangular with little padding (float64 sums: order is immaterial)
        kneed = torch.where(P.nstep > P.n10, P.n10, P.nstep).clamp(min=1)
        order = torch.argsort(kneed, stable=True)
        self.pairs = Pairs(*(getattr(P, k)[order] for k in Pairs.__dataclass_fields__))
        self.kneed = kneed[order]
        return self.pairs

    # ----------------------------------------------------------------- values
    def buffer(self, x, vturb, mode: str = "exact", pair_chunk: int = 100000, max_contrib: int = 3_000_000):
        """BUFFER [n, LENGTH] (line opacity per gram, before stimulated emission), attached."""
        if self.pairs is None:
            raise RuntimeError("call decide() first")
        P = self.pairs
        win = self.win
        n, L = x.n, win.length
        q = self.layer_quantities(x, vturb)
        buf = torch.zeros(n * L, dtype=D, device=self.dev)
        tabs = self.tabs
        for c0 in range(0, P.j.numel(), pair_chunk):
            sl = slice(c0, c0 + pair_chunk)
            jj, li = P.j[sl], P.line[sl]
            nel = self.nel[li]
            kappa0 = self.congf[li].to(D) * q["qxnfdop"][jj, nel]
            xarg = self.elo[li].to(D) * q["hckt"][jj]
            boltz = _st(tabs.fastex_tab(self.elo[li] * q["hckt"][jj].detach().to(F)), torch.exp(-xarg), mode)
            k0 = kappa0 * boltz
            dop = q["dop"][jj, nel]
            adamp = (self.gr[li].to(D) + self.gs[li].to(D) * q["xne"][jj] + self.gw[li].to(D) * q["txn"][jj]) / dop
            small = P.small[sl]
            zero = torch.zeros_like(adamp)
            kapcen = torch.where(small, k0 * (1.0 - 1.128 * adamp),
                                 k0 * voigt64(zero, adamp, tabs, torch.zeros_like(jj), mode))
            n10 = P.n10[sl]
            kmax = max(int(self.kneed[sl].max()) if n10.numel() else 0, 1)
            # core profile values PROFILE(1..N10DOP) per pair: table positions from the REAL*4 walk (detached)
            with torch.no_grad():
                ts = P.tabstep[sl]
                tabi = torch.full_like(ts, 1.5)
                its = []
                for k in range(kmax):
                    tabi = tabi + ts
                    its.append(torch.trunc(tabi).to(torch.long) - 1)
                it = torch.stack(its, 1)                                      # (np, kmax)
            kk = torch.arange(1, kmax + 1, dtype=D, device=self.dev)
            dv = 1.0 / (dop * win.resolu)                                     # smooth Doppler units per point
            v = kk[None, :] * dv[:, None]
            h0t, h1t = tabs.h01_tab(it)
            h1s, _ = tabs.h12_smooth(v, need_h2=False)
            core = k0[:, None] * (_st(h0t, torch.exp(-v * v), mode) + adamp[:, None] * _st(h1t, h1s, mode))
            lr = torch.nonzero(~small).squeeze(1)
            if lr.numel():
                with torch.no_grad():
                    v32 = kk.to(F)[None, :] * P.dvoigt[sl][lr][:, None]
                pl = k0[lr, None] * voigt64(v[lr], adamp[lr, None], tabs, tabs.tab_index(v32), mode)
                core = core.index_put((lr,), pl)
            n10f = n10.to(D)
            lastv = core.gather(1, (n10 - 1).clamp(min=0, max=core.shape[1] - 1).unsqueeze(1)).squeeze(1)
            xfar = torch.where(n10 > 0, lastv * n10f * n10f, torch.zeros_like(lastv))
            buf = self._accumulate(buf, jj, P.nbs[sl], P.nstep[sl], n10, core, xfar, kapcen, L, max_contrib)
        return buf.reshape(n, L)

    @staticmethod
    def _accumulate(buf, jj, nbs, nstep, n10, core, xfar, kapcen, L, max_contrib):
        lo = torch.clamp(nbs - nstep, min=1)
        hi = torch.clamp(nbs + nstep, max=L)
        cnt = torch.clamp(hi - lo + 1, min=0)
        csum = torch.cumsum(cnt, 0)
        npair = cnt.numel()
        a = 0
        dev = buf.device
        while a < npair:
            base = int(csum[a - 1]) if a > 0 else 0
            b = int(torch.searchsorted(csum, torch.tensor(base + max_contrib, device=dev), right=True))
            b = min(max(b, a + 1), npair)
            c = cnt[a:b]
            t = int(c.sum())
            if t:
                pid = torch.repeat_interleave(torch.arange(b - a, device=dev), c)
                first = torch.cumsum(c, 0) - c
                pnt = torch.arange(t, device=dev) - first[pid] + lo[a:b][pid]
                off = torch.abs(pnt - nbs[a:b][pid])
                incore = off <= n10[a:b][pid]
                cidx = torch.clamp(off - 1, min=0, max=core.shape[1] - 1)
                gp = pid + a
                val_core = core[gp, cidx]
                offf = torch.where(off > 0, off, torch.ones_like(off)).to(core.dtype)
                val_far = xfar[gp] / (offf * offf)
                val = torch.where(incore, val_core, val_far)
                val = torch.where(off == 0, kapcen[gp], val)
                key = jj[gp] * L + (pnt - 1)
                buf = buf.index_add(0, key, val)
            a = b
        return buf

    def asynth(self, x, buf: torch.Tensor) -> torch.Tensor:
        """ASYNTH [LENGTH, n] = BUFFER * (1 - EXP(-FREQ*HKT(J)))."""
        wave = torch.as_tensor(self.win.wave(), dtype=D, device=self.dev)
        freq = 2.99792458e17 / wave
        stim = 1.0 - torch.exp(-freq[None, :] * x.hkt[:, None])
        return (buf * stim).T
