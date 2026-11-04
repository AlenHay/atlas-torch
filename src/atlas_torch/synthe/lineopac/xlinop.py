"""XLINOP of synthe.for: the special lines of fort.19 (hydrogen, He I, autoionizing, merged continua,
lines with full Voigt wings), added to the layer buffers before the ordinary lines of fort.12.

Lines are taken in fort.19 order (each adds at most once per wavelength point, so per-line float32
buffer updates reproduce the Fortran's sequential sums exactly); every line is evaluated for all layers
at once and over chunks of wavelength points, with the Fortran's sequential stop rules (first point
below the cutoff, wavelength limits, ...) applied per layer.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ...tables import table
from .hydrogen import hfnm
from .helium import HeI
from .tabs import fastex
from .voigt import voigt

SRC = "synthe.patched.for"
F32 = np.float32
F = torch.float32
D = torch.float64
C_KMS = 299792.458


def powi(x: float, n):
    """libgcc __powidf2(x, n) (REAL*8 ** INTEGER), elementwise over integer array n."""
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


class LazySeq:
    """A wavelength/frequency sequence along a walk, computed on demand in blocks and cached as a prefix.

    `gen(prev, n)` returns the next n values given the last computed one (None for the first block): the
    recurrence is the Fortran's sequential one (WAVE=WAVE*RATIO, ...), so blocks continue it exactly.  Walks
    usually stop after a few points: only what is visited is computed (the buffer may have 2M points)."""
    __slots__ = ("n", "gen", "vals", "size")

    def __init__(self, n: int, gen):
        self.n = max(int(n), 0)
        self.gen = gen
        self.vals = []
        self.size = 0

    def __len__(self):
        return self.n

    def _extend(self, upto: int):
        while self.size < upto:
            m = min(self.n - self.size, max(4096, self.size))
            prev = self.vals[-1][-1] if self.vals else None
            a = self.gen(prev, m)
            self.vals.append(a)
            self.size += m

    def get(self, a: int, b: int) -> np.ndarray:
        b = min(b, self.n)
        if b <= a:
            return np.zeros(0, np.float64)
        self._extend(b)
        out = []
        pos = 0
        for v in self.vals:
            lo, hi = max(a - pos, 0), min(b - pos, len(v))
            if lo < hi:
                out.append(v[lo:hi])
            pos += len(v)
            if pos >= b:
                break
        return out[0] if len(out) == 1 else np.concatenate(out)

    def first_above(self, x: float):
        """Index of the first value > x, or None (scans the sequence block by block)."""
        pos = 0
        while pos < self.n:
            m = min(self.n - pos, max(65536, pos))
            v = self.get(pos, pos + m)
            h = np.nonzero(v > x)[0]
            if h.size:
                return pos + int(h[0])
            pos += m
        return None

    def nbytes(self) -> int:
        return 8 * self.size


class IRange:
    """1-based buffer indices start, start+step, ... (n values), sliced lazily."""
    __slots__ = ("start", "step", "n")

    def __init__(self, start: int, step: int, n: int):
        self.start, self.step, self.n = int(start), int(step), max(int(n), 0)

    def __len__(self):
        return self.n

    def get(self, a: int, b: int) -> np.ndarray:
        b = min(b, self.n)
        return self.start + self.step * np.arange(a, b, dtype=np.int64)


def _get(x, a: int, b: int) -> np.ndarray:
    return x.get(a, b) if hasattr(x, "get") else x[a:b]


def vacair(w):
    """VACAIR (REAL*8)."""
    w = np.asarray(w, np.float64)
    waven = 1.e7 / w
    return w / (1.0000834213 + 2406030. / (1.30e10 - waven ** 2) + 15997. / (3.89e9 - waven ** 2))


class XLinop:
    def __init__(self, ctx):
        self.c = ctx
        self._seq = {}
        self._bcs_map = {}
        self.first = True
        self.he = None
        self._philam = {}
        self.j0 = 0              # absolute index of this context's first layer (layer blocks)
        self.he_full = None      # callable -> HeI over all layers (PHILAM rows of other layer blocks)

    # ------------------------------------------------------------ wavelength sequences
    SEQ_CACHE_BYTES = 64 << 20

    def _cached(self, key, make):
        sq = self._seq.get(key)
        if sq is None:
            if sum(v.nbytes() for v in self._seq.values()) > self.SEQ_CACHE_BYTES:
                self._seq.clear()
            sq = self._seq[key] = make()
        return sq

    def red_seq(self, start: int) -> LazySeq:
        """WAVE at IBUFF=start..LENGTH: WAVE=WBEGIN*RATIO**(start-1), then WAVE=WAVE*RATIO."""
        c = self.c
        ratio = c.ratio

        def gen(prev, m):
            a = np.empty(m, np.float64)
            a[0] = c.wbegin * float(powi(ratio, start - 1)) if prev is None else prev * ratio
            a[1:] = ratio
            return np.multiply.accumulate(a)
        return self._cached(("r", start), lambda: LazySeq(c.length - start + 1, gen))

    def blue_seq(self, ib0: int) -> LazySeq:
        """WAVE at IBUFF=ib0-1..1: WAVE=WBEGIN*RATIO**(ib0-1), then WAVE=WAVE/RATIO before each point."""
        c = self.c
        ratio = c.ratio

        def gen(prev, m):
            a = np.empty(m + 1, np.float64)
            a[0] = c.wbegin * float(powi(ratio, ib0 - 1)) if prev is None else prev
            a[1:] = ratio
            return np.divide.accumulate(a)[1:]
        return self._cached(("b", ib0), lambda: LazySeq(ib0 - 1, gen))

    # ------------------------------------------------------------ generic sequential wing walk
    def walk(self, lay: torch.Tensor, ibuf, wave, ev, chunk: int = 4096):
        """Visit points ibuf (1-based, in loop order; an array or an IRange) for layers `lay`; wave: array or
        LazySeq; ev(lj, ib0, w) -> (kappa, skip, stop_before, stop_after) with shapes broadcastable to [nl, np].
        Per layer: skip -> no add; stop_before -> stop without adding; else add, then stop if stop_after."""
        c = self.c
        dev = c.device
        alive = lay.clone()
        pos = 0
        n = len(ibuf)
        while alive.numel() and pos < n:
            end_ = min(pos + chunk, n)
            ib = torch.as_tensor(_get(ibuf, pos, end_) - 1, device=dev)
            w = torch.as_tensor(_get(wave, pos, end_), device=dev)
            kap, skip, sb, sa = ev(alive, ib, w)
            npt = ib.numel()
            shape = (alive.numel(), npt)
            kap = kap.expand(shape)
            skip = torch.as_tensor(skip, device=dev).expand(shape)
            sb = torch.as_tensor(sb, device=dev).expand(shape) & ~skip
            sa = torch.as_tensor(sa, device=dev).expand(shape) & ~skip
            ar = torch.arange(npt, device=dev)
            big = torch.full(shape, npt, device=dev, dtype=torch.long)
            fsb = torch.where(sb, ar, big).min(dim=1).values
            fsa = torch.where(sa, ar + 1, big).min(dim=1).values
            end = torch.minimum(fsb, fsa)
            add = (ar.unsqueeze(0) < end.unsqueeze(1)) & ~skip
            rows = alive.unsqueeze(1).expand(shape)[add]
            cols = ib.unsqueeze(0).expand(shape)[add]
            buf = c.buffer
            buf[rows, cols] = buf[rows, cols] + kap[add]
            alive = alive[end >= npt]
            pos += npt
        return alive

    def cc(self, lj, ib):
        """CONTINUUM(IBUFF)*CUTOFF for layers lj x points ib (0-based)."""
        return self.c.cutcont[lj.unsqueeze(1), ib.unsqueeze(0)]

    # ------------------------------------------------------------ main entry
    def __call__(self, lines: np.ndarray):
        c = self.c
        if self.first:
            self._first_call()
        for il, ln in enumerate(lines):
            vel = c.velshift
            # group layers with equal DOPRATIO (one group unless DECKJ velocity fields are used)
            for v in np.unique(vel):
                lay = torch.as_tensor(np.nonzero(vel == v)[0], device=c.device)
                dopratio = 1.0 + float(v) / C_KMS
                self._line(il + 1, ln, lay, dopratio)

    def _first_call(self):
        c = self.c
        self.first = False
        ehyd = np.zeros(100)
        ehyd[1:8] = [82259.105, 97492.302, 102823.893, 105291.651, 106632.160, 107440.444, 107965.051]
        for n in range(9, 101):
            ehyd[n - 1] = 109678.764 - 109677.576 / (n * n)
        self.ehyd = ehyd
        self.alphahyd = 1.e7 / (ehyd[1:] - ehyd[:-1])
        c.out.append("       NMERGE    EMERGE    EMERGEH")
        xne = c.xne_np
        inglis = F32(1600.) / c.spowf_np(xne, F32(2.) / F32(15.))
        nmerge = (inglis - F32(1.5)).astype(F32)
        n2 = (nmerge * nmerge).astype(F32).astype(np.float64)
        self.emerge = 109737.312 / n2
        self.emergeh = 109677.576 / n2
        c.emerge = self.emerge
        for k in range(c.nrhox):
            c.out.append("%3d%s%s%s" % (k + 1, c.ff(nmerge[k], 10, 3), c.ff(self.emerge[k], 10, 3),
                                        c.ff(self.emergeh[k], 10, 3)))

    # ------------------------------------------------------------ one line, layer group
    def _line(self, iline: int, ln, lay: torch.Tensor, dopratio: float):
        typ = int(ln["type"])
        if typ in (2, -5, -6):
            return
        if typ in (0, 3) or typ < -6:
            return self._normal(iline, ln, lay, dopratio)
        if typ in (-1, -2):
            return self._hydrogen(iline, ln, lay, dopratio)
        if typ in (-3, -4):
            return self._helium(iline, ln, lay, dopratio)
        if typ == 1:
            return self._autoion(iline, ln, lay, dopratio)
        return self._merged(iline, ln, lay, dopratio)

    def _kapmin(self, nbuff: int, lay):
        c = self.c
        k = min(max(nbuff, 1), c.length) - 1
        if c.continuum is None:                      # all-layer context without stored continuum
            return c.cont_at(lay, k) * c.cutoff
        return c.continuum[lay, k] * c.cutoff

    def _f15(self, iline, lay, vals, force=False):
        c = self.c
        if force or c.linout >= 0:
            for j, v in zip(lay.tolist(), vals.tolist()):
                j += self.j0
                c.f15.append((j, c.seq_next(j), iline, v))

    # ------------------------------------------------------------ TYPE 0/3: Voigt line
    def _normal(self, iline, ln, lay, dopratio):
        c = self.c
        nel = int(ln["nelion"]) - 1
        nbuff = int(ln["nbuff"])
        wl = float(ln["wl"]) * dopratio
        wl4 = F32(wl)
        kappa0 = float(F32(ln["gf"])) * c.xnfdop[lay, nel]
        kapmin = self._kapmin(nbuff, lay)
        ok = ~(kappa0 < kapmin)
        bolt = fastex(float(F32(ln["elo"])) * c.hckt[lay], c.extab)
        kappa0 = kappa0 * bolt
        ok &= ~(kappa0 < kapmin)
        lay, kappa0 = lay[ok], kappa0[ok]
        if not lay.numel():
            return
        c.mlines[lay] += 1
        ncon = int(ln["ncon"])
        if ncon > 0:
            ctx = table(SRC, "xlinop", "contx")[ncon - 1, int(ln["nelionx"]) - 1]
            em = torch.as_tensor(self.emerge, device=c.device)[lay]
            wcon = 1.e7 / (ctx - em) * dopratio
            wtail = 1.e7 / (ctx - em - 500.) * dopratio
        else:
            wcon = torch.zeros(lay.numel(), dtype=D, device=c.device)
            wtail = wcon.clone()
        if float(ln["alpha"]) != 0.0:
            c.alpha_txnxn(lay, int(ln["nelion"]), F32(ln["alpha"]))
        adamp = (float(F32(ln["gammar"])) + float(F32(ln["gammas"])) * c.xne[lay] +
                 float(F32(ln["gammaw"])) * c.txnxn[lay]) / c.dopple[lay, nel]
        kapcen = kappa0 * voigt(torch.zeros_like(adamp), adamp, c.h1tab)
        self._f15(iline, lay, kapcen)
        dopwl = c.dopple[lay, nel] * float(wl4)
        wcon2, wtail2 = wcon.unsqueeze(1), wtail.unsqueeze(1)
        k0, a2, dw2 = kappa0.unsqueeze(1), adamp.unsqueeze(1), dopwl.unsqueeze(1)

        def kap(lj, w):
            sel = self._sel(lay, lj)
            w4 = w.to(F)
            v = torch.abs(w4 - float(wl4)) / dw2[sel]
            k = k0[sel] * voigt(v, a2[sel], c.h1tab)
            tl = w.unsqueeze(0) < wtail2[sel]
            kt = (k.to(D) * (w.unsqueeze(0) - wcon2[sel]) / (wtail2[sel] - wcon2[sel])).to(F)
            return torch.where(tl, kt, k), w.unsqueeze(0) < wcon2[sel]

        if not wl > c.wlend:
            minred = max(1, nbuff)

            def ev(lj, ib, w):
                k, below = kap(lj, w)
                return k, below, k < self.cc(lj, ib), torch.zeros_like(below)
            self._walk_red(lay, minred, ev)
            if minred == 1 or wl < c.wbegin:
                return
        ib0 = min(c.length + 1, nbuff)

        def evb(lj, ib, w):
            k, below = kap(lj, w)
            return k, below, torch.zeros_like(below), k < self.cc(lj, ib)
        self._walk_blue(lay, ib0, evb)

    @staticmethod
    def _sel(lay, lj):
        """Row indices into per-`lay` arrays for the alive subset lj (lj is a subset of lay, same order)."""
        if lj.numel() == lay.numel():
            return slice(None)
        return torch.searchsorted(lay, lj)

    def _walk_red(self, lay, minred, ev):
        c = self.c
        if minred > c.length:
            return
        self.walk(lay, IRange(minred, 1, c.length - minred + 1), self.red_seq(minred), ev)

    def _walk_blue(self, lay, ib0, ev):
        if ib0 - 1 < 1:
            return
        self.walk(lay, IRange(ib0 - 1, -1, ib0 - 1), self.blue_seq(ib0), ev)

    # ------------------------------------------------------------ TYPE -1/-2: hydrogen
    def _hydrogen(self, iline, ln, lay, dopratio):
        c = self.c
        nbuff = int(ln["nbuff"])
        nblo, nbup = int(ln["nblo"]), int(ln["nbup"])
        typ = int(ln["type"])
        wl = float(ln["wl"]) * dopratio
        kappa0 = float(F32(ln["gf"])) * c.xnfdop[lay, 0]
        kapmin = self._kapmin(nbuff, lay)
        if nbup == 2:
            kapmin = c.continuum[lay, c.length - 1] * c.cutoff
        ok = ~(kappa0 < kapmin)
        bolth = fastex(float(F32(ln["elo"])) * c.hckt[lay], c.extab)
        # DOPPH(J) is set only for layers passing the first test (SAVEd array)
        dp = c.dopple[lay, 0]
        if typ == -2:
            dp = (dp.to(D) / 1.4142).to(F)
        c.dopph[lay[ok]] = dp[ok]
        kappa0 = kappa0 * bolth
        ok &= ~(kappa0 < kapmin)
        lay, kappa0 = lay[ok], kappa0[ok]
        if not lay.numel():
            return
        self._f15(iline, lay, kappa0)
        c.mlines[lay] += 1
        ncon = int(ln["ncon"])
        hp = c.hprof4
        dopph = c.dopph
        k0 = kappa0.unsqueeze(1)
        ehyd = self.ehyd

        def prof(n, m, lj, w, wlc, sel, kk):
            dl = w.unsqueeze(0) - wlc
            return kk[sel] * hp(n, m, lj.unsqueeze(1), dl.expand(lj.numel(), -1), dopph)

        if ncon == 0 or nbup == nblo + 1 or nbup == nblo + 2:
            beta = (ncon != 0 and nbup == nblo + 2)

            def evr(lj, ib, w):
                sel = self._sel(lay, lj)
                k = prof(nblo, nbup, lj, w, wl, sel, k0)
                z = torch.zeros_like(k, dtype=torch.bool)
                if beta:
                    gt = (w > float(self.alphahyd[nblo - 1])).unsqueeze(0).expand_as(z)
                    return k, z, gt | (k < self.cc(lj, ib)), z
                return k, z, k < self.cc(lj, ib), z

            def evb(lj, ib, w):
                sel = self._sel(lay, lj)
                k = prof(nblo, nbup, lj, w, wl, sel, k0)
                z = torch.zeros_like(k, dtype=torch.bool)
                return k, z, z, k < self.cc(lj, ib)

            if wl > c.wlend:
                self._walk_blue(lay, min(c.length + 1, nbuff), evb)
                return
            minred = max(1, nbuff)
            if not beta:
                self._walk_red(lay, minred, evr)
                if minred == 1 or wl < c.wbegin:
                    return
                self._walk_blue(lay, min(c.length + 1, nbuff), evb)
                return
            # beta lines: red wing until ALPHAHYD(NBLO) (label 633 -> blue wing from NBUFF), else 622
            alpha_lim = float(self.alphahyd[nblo - 1])
            hit = self.red_seq(minred).first_above(alpha_lim) if minred <= c.length else None
            # layers that stop on the cutoff before reaching ALPHAHYD go to 622; the others to 633
            reached = self._walk_red_track(lay, minred, evr, hit)
            if hit is not None:
                l633 = lay[reached]
                l622 = lay[~reached]
            else:
                l633 = lay[:0]
                l622 = lay
            if l633.numel() and nbuff >= 1:
                sel_lay = l633
                self._walk_blue_sub(lay, sel_lay, nbuff, evb)
            if l622.numel() and not (minred == 1 or wl < c.wbegin):
                self._walk_blue_sub(lay, l622, min(c.length + 1, nbuff), evb)
            return
        # general case: series with merging into the continuum
        conth = table(SRC, "xlinop", "conth")
        wshift = 1.e7 / (conth[ncon - 1] - 109677.576 / 81.0 ** 2)
        emh = torch.as_tensor(c.emergeh_np, device=c.device)[lay]
        wmerge = 1.e7 / (conth[ncon - 1] - emh)
        wmerge = torch.where(wmerge < 0., torch.full_like(wmerge, wshift + wshift), wmerge)
        wcon = torch.clamp(wmerge, min=wshift)
        wtail = 1.e7 / (1.e7 / wcon - 500.)
        wcon = torch.clamp(wcon, max=wshift + wshift)
        wtail = torch.where(wtail < 0., wcon + wcon, wtail)
        wtail = torch.minimum(wcon + wcon, wtail)
        if c.ifvac == 0:
            wcon = torch.as_tensor(vacair(wcon.cpu().numpy()), device=c.device)
            wtail = torch.as_tensor(vacair(wtail.cpu().numpy()), device=c.device)
        wcon = wcon * dopratio
        if c.ifvac == 0:
            wl = float(vacair(1.e7 / (ehyd[nbup - 1] - ehyd[nblo - 1]))) * dopratio
        wc2, wt2 = wcon.unsqueeze(1), wtail.unsqueeze(1)

        def tail(k, w, sel):
            tl = w.unsqueeze(0) < wt2[sel]
            kt = (k.to(D) * (w.unsqueeze(0) - wc2[sel]) / (wt2[sel] - wc2[sel])).to(F)
            return torch.where(tl, kt, k)

        go_blue = True
        if not wl > c.wlend and not c.wbegin > float(self.alphahyd[nblo - 1]):
            redcut = 1.e7 / (109678.764 - 109677.576 / (nbup - 0.8) ** 2 - ehyd[nblo - 1])
            if c.ifvac == 0:
                redcut = float(vacair(redcut))
            redcut = redcut * dopratio
            wlm1 = 1.e7 / (ehyd[nbup - 2] - ehyd[nblo - 1])
            if c.ifvac == 0:
                wlm1 = float(vacair(wlm1))
            wlm1 = wlm1 * dopratio
            wlm2 = 1.e7 / (ehyd[nbup - 3] - ehyd[nblo - 1])
            if c.ifvac == 0:
                wlm2 = float(vacair(wlm2))
            wlm2 = wlm2 * dopratio
            k0red = ((kappa0 * float(hfnm(nblo, nbup - 2)) / float(hfnm(nblo, nbup))).to(D) /
                     (ehyd[nbup - 3] - ehyd[nblo - 1]) * (ehyd[nbup - 1] - ehyd[nblo - 1])).to(F).unsqueeze(1)

            def evr(lj, ib, w):
                sel = self._sel(lay, lj)
                k = tail(prof(nblo, nbup, lj, w, wl, sel, k0), w, sel)
                below = (w < wc2[sel])
                past = (w > wlm1).unsqueeze(0)
                stop = past.expand_as(k) | (k < self.cc(lj, ib))
                rc = (w > redcut).unsqueeze(0)
                if bool(rc.any()):
                    kr = tail(prof(nblo, nbup - 2, lj, w, wlm2, sel, k0red), w, sel)
                    stop = stop | (rc & (kr >= k))
                # WAVE < WCON skips the point before any other test (incl. the WLMINUS1 exit)
                return k, below.expand_as(k), stop, torch.zeros_like(stop)
            minred = max(1, nbuff)
            self._walk_red(lay, minred, evr)
            if minred == 1 or wl < c.wbegin:
                go_blue = False
        if not go_blue:
            return
        bluecut = 1.e7 / (109678.764 - 109677.576 / (nbup + 0.8) ** 2 - ehyd[nblo - 1])
        if c.ifvac == 0:
            bluecut = float(vacair(bluecut))
        bluecut = bluecut * dopratio
        wlp1 = 1.e7 / (ehyd[nbup] - ehyd[nblo - 1])
        if c.ifvac == 0:
            wlp1 = float(vacair(wlp1))
        wlp1 = wlp1 * dopratio
        wlp2 = 1.e7 / (ehyd[nbup + 1] - ehyd[nblo - 1])
        if c.ifvac == 0:
            wlp2 = float(vacair(wlp2))
        wlp2 = wlp2 * dopratio
        k0blue = ((kappa0 * float(hfnm(nblo, nbup + 2)) / float(hfnm(nblo, nbup))).to(D) /
                  (ehyd[nbup + 1] - ehyd[nblo - 1]) * (ehyd[nbup - 1] - ehyd[nblo - 1])).to(F).unsqueeze(1)

        def evb2(lj, ib, w):
            sel = self._sel(lay, lj)
            k = tail(prof(nblo, nbup, lj, w, wl, sel, k0), w, sel)
            stop_b = (w.unsqueeze(0) < wc2[sel]) | (w < wlp1).unsqueeze(0)
            bc = (w < bluecut).unsqueeze(0)
            if bool(bc.any()):
                kb = tail(prof(nblo, nbup + 2, lj, w, wlp2, sel, k0blue), w, sel)
                stop_b = stop_b | (bc & (kb > k))
            stop_b = stop_b.expand_as(k)
            return k, torch.zeros_like(stop_b), stop_b, k < self.cc(lj, ib)
        self._walk_blue(lay, min(c.length + 1, nbuff), evb2)

    def _walk_red_track(self, lay, minred, ev, k_alpha):
        """Red walk that also reports, per layer of `lay`, whether the walk ended at the ALPHAHYD limit
        (index k_alpha in the red sequence) rather than on the cutoff or the end of the buffer."""
        c = self.c
        if minred > c.length:
            return torch.zeros(lay.numel(), dtype=torch.bool, device=c.device)
        if k_alpha is None:
            self._walk_red(lay, minred, ev)
            return torch.zeros(lay.numel(), dtype=torch.bool, device=c.device)
        reached = torch.zeros(lay.numel(), dtype=torch.bool, device=c.device)
        # walk the points before the limit; layers still alive at the limit reached it
        ib = IRange(minred, 1, k_alpha)
        seq = self.red_seq(minred).get(0, k_alpha)
        alive_after = self._walk_collect(lay, ib, seq, ev)
        reached[torch.searchsorted(lay, alive_after)] = True
        return reached

    def _walk_collect(self, lay, ibuf, wave, ev):
        """walk() returning the layers still alive at the end."""
        if len(ibuf) == 0:
            return lay
        return self.walk(lay, ibuf, wave, ev)

    def _walk_blue_sub(self, lay, sub, ib0, ev):
        """Blue walk for the subset `sub` of `lay` (ev indexes per-`lay` arrays through _sel)."""
        if ib0 - 1 < 1 or not sub.numel():
            return
        self.walk(sub, IRange(ib0 - 1, -1, ib0 - 1), self.blue_seq(ib0), ev)

    # ------------------------------------------------------------ TYPE -3/-4: He I
    def _helium(self, iline, ln, lay, dopratio):
        c = self.c
        if self.he is None:
            self.he = HeI(dict(t=c.t_np, xne=c.xne_np, xnfph2=c.xnfph_np[:, 1], xnfhe2=c.xnfhe_np[:, 1]),
                          c.h1tab, c.fort18, c.mode)
        he = self.he
        typ = int(ln["type"])
        nel = int(ln["nelion"]) - 1
        nbuff = int(ln["nbuff"])
        wl = float(ln["wl"]) * dopratio
        wl4 = F32(wl)
        kappa0 = float(F32(ln["gf"])) * c.xnfpel[lay, nel] / c.dopple[lay, nel]
        if typ == -4:
            kappa0 = kappa0 / float(F32(1.155))
        kapmin = self._kapmin(nbuff, lay)
        ok = ~(kappa0 < kapmin)
        bolt = fastex(float(F32(ln["elo"])) * c.hckt[lay], c.extab)
        kappa0 = kappa0 * bolt
        ok &= ~(kappa0 < kapmin)
        lay, kappa0 = lay[ok], kappa0[ok]
        if not lay.numel():
            return
        dopwl_all = (c.dopple[:, 6] * float(wl4))
        if typ == -4:
            dopwl_all = dopwl_all * float(F32(1.155))
        kind = he.kind(wl4)
        gammar, gammas = F32(ln["gammar"]), F32(ln["gammas"])
        dnp = dopwl_all.cpu().numpy()
        if kind in ("griem", "dimitri"):
            lc = (he.griem_layer if kind == "griem" else he.dimitri_layer)(wl4, dnp, gammar, gammas)

            def pf(lj, w4):
                return he.gd_profile(lc, lj.unsqueeze(-1) if w4.dim() > 1 else lj, w4, wl4,
                                     dopwl_all[lj].unsqueeze(-1) if w4.dim() > 1 else dopwl_all[lj])
            philam_rows = None
        else:
            lc = he.bcs_layer(kind, wl4, dnp)
            line_no = {"he4471": 1, "he4026": 2, "he4387": 3, "he4921": 4}[kind]
            philam_rows = self._bcs_rows(iline, lay, line_no) if kind != "he4387" else None

            def pf(lj, w4):
                two = w4.dim() > 1
                jj = lj.unsqueeze(-1) if two else lj
                pr = None
                if philam_rows is not None:
                    pr = philam_rows[self._sel(lay, lj)]
                    if two:
                        pr = pr.unsqueeze(1).expand(-1, w4.shape[1], -1)
                dw = dopwl_all[lj].unsqueeze(-1) if two else dopwl_all[lj]
                return he.bcs_profile(kind, jj, w4, wl4, dw, lc, pr)
        kapcen = kappa0 * pf(lay, torch.full((lay.numel(),), float(wl4), dtype=F, device=c.device))
        self._f15(iline, lay, kapcen, force=True)
        c.mlines[lay] += 1
        k0 = kappa0.unsqueeze(1)

        def kap(lj, w):
            sel = self._sel(lay, lj)
            w4 = w.to(F).unsqueeze(0).expand(lj.numel(), -1)
            return k0[sel] * pf(lj, w4)

        if not wl > c.wlend:
            minred = max(1, nbuff)

            def evr(lj, ib, w):
                k = kap(lj, w)
                z = torch.zeros_like(k, dtype=torch.bool)
                return k, z, (w > c.wlend).unsqueeze(0) | (k < self.cc(lj, ib)), z
            self._walk_red(lay, minred, evr)
            if nbuff == 1 or wl < c.wlbeg:
                return

        def evb(lj, ib, w):
            k = kap(lj, w)
            z = torch.zeros_like(k, dtype=torch.bool)
            return k, z, (w < c.wlbeg).unsqueeze(0).expand_as(z), k < self.cc(lj, ib)
        self._walk_blue(lay, min(c.length + 1, nbuff), evb)

    def prepare_bcs(self, lines):
        """READBCS caches PHILAM under the key J*LINE (JSAVE).  Calls happen layer by layer, He lines in
        fort.19 order; a call whose key equals the previous one reuses the cached PHILAM even when it
        belongs to another (layer, line).  Precompute the PHILAM source of every (fort.19 line, layer)."""
        c = self.c
        calls = []
        for il, ln in enumerate(lines):
            if int(ln["type"]) not in (-3, -4):
                continue
            lay = torch.arange(c.nrhox, device=c.device)
            ok = self._he_pass(ln, lay)
            for j in range(c.nrhox):
                if not bool(ok[j]):
                    continue
                dopratio = 1.0 + float(c.velshift[j]) / C_KMS
                kind = HeI.kind(F32(float(ln["wl"]) * dopratio))
                line = {"he4471": 1, "he4026": 2, "he4921": 4}.get(kind)
                if line is None:
                    continue
                thr = F32(1e14) if line == 2 else F32(1e13)
                if c.xne_np[j] > thr:
                    calls.append((j, il + 1, line))
        calls.sort(key=lambda t: (t[0], t[1]))
        self._bcs_map = {}
        jsave, src = 0, None
        for j, il, line in calls:
            key = (j + 1) * line
            if key != jsave:
                src = (j, line)
                jsave = key
            self._bcs_map[(il, j)] = src

    def _he_pass(self, ln, lay):
        c = self.c
        typ = int(ln["type"])
        nel = int(ln["nelion"]) - 1
        kappa0 = float(F32(ln["gf"])) * c.xnfpel[lay, nel] / c.dopple[lay, nel]
        if typ == -4:
            kappa0 = kappa0 / float(F32(1.155))
        kapmin = self._kapmin(int(ln["nbuff"]), lay)
        ok = ~(kappa0 < kapmin)
        kappa0 = kappa0 * fastex(float(F32(ln["elo"])) * c.hckt[lay], c.extab)
        return ok & ~(kappa0 < kapmin)

    def _bcs_rows(self, iline, lay, line):
        """PHILAM row used by each layer of `lay` (see prepare_bcs)."""
        c = self.c
        he = self.he
        out = torch.zeros((lay.numel(), 204), dtype=F, device=c.device)
        for k, j in enumerate(lay.tolist()):
            src = self._bcs_map.get((iline, j + self.j0))
            if src is None:
                continue
            jl, ll = src                              # absolute layer of the cached PHILAM
            if ll not in self._philam:
                hf = self.he_full() if self.he_full is not None else he
                self._philam[ll] = torch.as_tensor(hf.philam(ll), device=c.device)
            out[k] = self._philam[ll][jl]
        return out

    # ------------------------------------------------------------ TYPE 1: autoionizing
    def _autoion(self, iline, ln, lay, dopratio):
        c = self.c
        nel = int(ln["nelion"]) - 1
        nbuff = int(ln["nbuff"])
        wl = float(ln["wl"]) * dopratio
        ashore = F32(ln["gammas"])
        bshore = F32(ln["gammaw"])
        g = F32(ln["gf"])
        gammar = F32(ln["gammar"])
        kappa0 = float(bshore * g) * c.xnfpel[lay, nel]
        kapmin = self._kapmin(nbuff, lay)
        ok = ~(kappa0 < kapmin)
        kappa0 = kappa0 * fastex(float(F32(ln["elo"])) * c.hckt[lay], c.extab)
        ok &= ~(kappa0 < kapmin)
        lay, kappa0 = lay[ok], kappa0[ok]
        if not lay.numel():
            return
        self._f15(iline, lay, kappa0)
        c.mlines[lay] += 1
        frelin = F32(2.99792458e17 / wl)
        k0 = kappa0.unsqueeze(1)

        def kap(lj, fr):
            sel = self._sel(lay, lj)
            eps = (2.0 * (fr - float(frelin))) / float(gammar)
            num = float(ashore) * eps + float(bshore)
            return k0[sel] * num / (eps * eps + 1.0) / float(bshore)
        ratio = c.ratio
        if not wl > c.wlend:
            minred = max(1, nbuff)
            if minred <= c.length:
                n = c.length - minred + 1
                x0 = F32(2.99792458e17 / (c.wbegin * float(powi(ratio, minred - 1))))

                def gen_r(prev, m):          # FREQ(point) = x, then x = REAL*4(x/RATIO)
                    out = np.empty(m, np.float64)
                    x = x0 if prev is None else F32(float(F32(prev)) / ratio)
                    for k in range(m):
                        out[k] = x
                        x = F32(float(x) / ratio)
                    return out

                def evr(lj, ib, w):
                    k = kap(lj, w.to(F))
                    z = torch.zeros_like(k, dtype=torch.bool)
                    return k, z, k < self.cc(lj, ib), z
                self.walk(lay, IRange(minred, 1, n), LazySeq(n, gen_r), evr)
            if nbuff == 1 or wl < c.wbegin:
                return
        ib0 = min(c.length + 1, nbuff)
        if ib0 - 1 >= 1:
            n = ib0 - 1
            x0 = F32(2.99792458e17 / (c.wbegin * float(powi(ratio, ib0 - 1))))

            def gen_b(prev, m):              # x = REAL*4(x*RATIO), then FREQ(point) = x
                out = np.empty(m, np.float64)
                x = x0 if prev is None else F32(prev)
                for k in range(m):
                    x = F32(float(x) * ratio)
                    out[k] = x
                return out

            def evb(lj, ib, w):
                k = kap(lj, w.to(F))
                z = torch.zeros_like(k, dtype=torch.bool)
                return k, z, z, k < self.cc(lj, ib)
            self.walk(lay, IRange(ib0 - 1, -1, n), LazySeq(n, gen_b), evb)

    # ------------------------------------------------------------ merged continuum (TYPE >= 4)
    def _merged(self, iline, ln, lay, dopratio):
        c = self.c
        nlast = int(ln["type"])
        nel = int(ln["nelion"])
        wl = float(ln["wl"]) * dopratio
        em = torch.as_tensor(self.emerge if nel != 1 else self.emergeh, device=c.device)[lay]
        ryd = 109677.576 if nel == 1 else 109737.312
        wshift = 1.e7 / (1.e7 / wl - ryd / (nlast * nlast))
        wmerge = 1.e7 / (1.e7 / wl - em)
        wmerge = torch.where(wmerge < 0., torch.full_like(wmerge, wshift + wshift), wmerge)
        wmerge = torch.clamp(wmerge, min=wshift)
        wmerge = torch.clamp(wmerge, max=wshift + wshift)
        wtail = 1.e7 / (1.e7 / wmerge - 500.)
        wtail = torch.where(wtail < 0., wmerge + wmerge, wtail)
        wtail = torch.minimum(wmerge + wmerge, wtail)
        if c.ifvac == 0:
            wmerge = torch.as_tensor(vacair(wmerge.cpu().numpy()) * dopratio, device=c.device)
            wtail = torch.as_tensor(vacair(wtail.cpu().numpy()) * dopratio, device=c.device)
        ixwl = int(math.log(wl) / c.ratiolg)
        edgeblue = F32(math.exp(ixwl * c.ratiolg))
        if float(edgeblue) > wl:
            ixwl -= 1
        nbuff1 = ixwl + 1 - c.ixwlbeg + 1
        lw = torch.as_tensor([math.log(v) for v in wmerge.cpu().tolist()], dtype=D, device=c.device)
        nbuff2 = torch.trunc(lw / c.ratiolg + .5).to(torch.long) - c.ixwlbeg + 1
        lt = torch.as_tensor([math.log(v) for v in wtail.cpu().tolist()], dtype=D, device=c.device)
        nbuff3 = torch.trunc(lt / c.ratiolg + .5).to(torch.long) - c.ixwlbeg + 1
        ok = ~((nbuff1 > c.length) | (nbuff3 < 1))
        if nbuff1 > c.length:
            return
        lay, nbuff2, nbuff3 = lay[ok], nbuff2[ok], nbuff3[ok]
        if not lay.numel():
            return
        dnbuff = (nbuff3 - nbuff2).to(F)
        nb1 = max(nbuff1, 1)
        kappa = float(F32(ln["gf"])) * c.xnfpel[lay, nel - 1] * \
            fastex(float(F32(ln["elo"])) * c.hckt[lay], c.extab)
        last = int(torch.clamp(nbuff3, max=c.length).max())       # no layer adds past its NBUFF3
        step = 65536
        for p0 in range(nb1, last + 1, step):
            ib = torch.arange(p0, min(p0 + step, last + 1), device=c.device)
            inr = ib.unsqueeze(0) <= torch.clamp(nbuff3, max=c.length).unsqueeze(1)
            tl = torch.where(ib.unsqueeze(0) > nbuff2.unsqueeze(1),
                             (nbuff3.unsqueeze(1) - ib.unsqueeze(0)).to(F) / dnbuff.unsqueeze(1),
                             torch.ones((), dtype=F, device=c.device))
            # TAIL keeps its last value: once IBUFF > NBUFF2 it is recomputed at every point (monotone)
            val = kappa.unsqueeze(1) * tl
            rows = lay.unsqueeze(1).expand_as(inr)[inr]
            cols = (ib - 1).unsqueeze(0).expand_as(inr)[inr]
            c.buffer[rows, cols] = c.buffer[rows, cols] + val[inr]
        self._f15(iline, lay, kappa)
        c.mlines[lay] += 1
