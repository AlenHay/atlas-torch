"""SYNTHE (synthe.for as patched by BasicATLAS: + progress.dat), ported to PyTorch.

Reads, in a SYNTHE work directory, fort.93 (run parameters from SYNBEG), fort.10 (xnfpelsyn.dat:
model, continuum opacities, number densities / Doppler widths), fort.12 (ordinary lines), fort.19 (special
lines for XLINOP), fort.18 (he1tables.dat), and writes what synthe.exe writes:

  fort.9   header + one record ASYNTH(kw) per wavelength point (line opacity per layer, REAL*4) + N9
  fort.8   N9 (number of saved lines; 0 for LINOUT<0)
  fort.14  direct-access scratch file left behind: the line-opacity buffer of every layer, LENREC blocks
  fort.15  (ILINE, KAPCEN) records (He I lines always; all lines when LINOUT>=0)
  progress.dat, and stdout (returned, and written to `stdout_path` if given)
and deletes fort.93, fort.12 and fort.13 as the Fortran does (CLOSE STATUS='DELETE').

Numerics follow the Fortran exactly: the program is REAL*4 except wavelengths/energies; every REAL*4
operation is done in float32 in the same order; libm calls reproduce glibc 2.39 + libmvec (see
lineopac/libm32.py), including gfortran's vectorised loops.  The line buffer of a layer is a sequential
REAL*4 sum over lines in file order; we batch over layers and lines and restore that order with a stable
sort of the contributions by (layer, point) followed by rank-wise float32 additions.

LINOUT >= 0 (line-identification output to fort.7/fort.9) is not implemented (BasicATLAS always runs
SYNTHE with LINOUT=-1); the program raises NotImplementedError in that case.

Memory is bounded independently of the window (BasicATLAS batches go up to 2,010,001 points and tens of
millions of fort.12 lines): like the Fortran, which loops over layers (DO 500 J) re-reading fort.12 and keeping
one BUFFER/CONTINUUM, the layers are processed in blocks (`layer_block`, chosen from `mem_budget`) holding
BUFFER and CONTINUUM of the block only; fort.12 is streamed in chunks for every block; each block's buffers go to
the direct-access fort.14 and fort.9 is then written by transposing fort.14 in point chunks (the Fortran's
"DIRECT IO TRANSPOSITION").  Every layer is independent in SYNTHE (XLINOP and the line loop only touch layer
J's arrays; READBCS's JSAVE cache is precomputed over all layers), so the blocking does not change any value.
"""
from __future__ import annotations

import copy
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..tables import table
from ..io.fortran import write_record
from .lineopac import fortio
from .lineopac import libm32 as L
from .lineopac.hydrogen import HProf4, rdiv, sqrtf
from .lineopac.tabs import extab_common, fastex, h1tab_common
from .lineopac.voigt import voigt
from .lineopac.xlinop import XLinop, powi

SRC = "synthe.patched.for"
F32 = np.float32
F = torch.float32
D = torch.float64
KW = 99
LENREC = 8000
MAXBUFF = 2000001 + 10000          # MAXLEN + MAXPROF: COMMON /BUFFER/ BUFFER(MAXBUFF),PROFILE(MAXPROF)
MAXPROF = 10000
C_KMS = 299792.458


def ff(x, w: int, d: int) -> str:
    """Fortran Fw.d output."""
    s = "%*.*f" % (w, d, float(x))
    return "*" * w if len(s) > w else s


@dataclass
class Ctx:
    device: torch.device
    mode: str
    length: int
    nrhox: int
    wbegin: float
    ratio: float
    ratiolg: float
    ixwlbeg: int
    wlbeg: float
    wlend: float
    resolu: float
    cutoff: float
    linout: int
    ifvac: int
    fort18: str | None
    out: list = field(default_factory=list)
    f15: list = field(default_factory=list)
    _seq: dict = field(default_factory=dict)

    ff = staticmethod(ff)

    def seq_next(self, j):
        self._seq[j] = self._seq.get(j, 0) + 1
        return self._seq[j]

    @staticmethod
    def spowf_np(x, y):
        return L.spowf(torch.as_tensor(np.asarray(x, F32)), torch.as_tensor(np.asarray(y, F32))).numpy()

    def alpha_txnxn(self, lay, nelion, alpha):
        """alpha != 0 branch of XLINOP/main loop: TXNXN(J) for these layers (sticky)."""
        t = self.t[lay]
        nelem = int(nelion / 6) + 1
        am = F32(table(SRC, "blocke", "atmass")[nelem - 1])
        v2 = (F32(1.) - F32(alpha)) / F32(2)
        hf = L.spowf(t / 10000.0, torch.full_like(t, float(v2)))
        td = t.to(D).cpu().numpy()
        he = (float(F32(.628)) * L.dpow(2.0991e-4 * td * float(F32(.25) + F32(1.008) / am), float(v2))).astype(F32)
        h2 = (float(F32(1.08)) * L.dpow(2.0991e-4 * td * float(F32(.5) + F32(1.008) / am), float(v2))).astype(F32)
        he = torch.as_tensor(he, device=t.device)
        h2 = torch.as_tensor(h2, device=t.device)
        # synthe.for lines 309/730 run past column 72: fixed form truncates "...+xnfh2(j)*h2factor(j)" to
        # "...+xnfh2(j)*h2fact", an implicit REAL local that is never set (SAVEd, zero): the H2 term is 0.
        h2fact = 0.0
        self.txnxn[lay] = self.xnfh[lay] * hf + self.xnfhe[lay, 0] * he + self.xnfh2[lay] * h2fact
        return hf, he, h2


class Continuum:
    """CONTINUUM(NBUFF) of every layer: quadratic interpolation of log10 kappa_cont between edges
    (REAL*8), stored REAL*4, then 10.**CONTINUUM (DO 2006: a vectorised powf loop over NBUFF=1..LENGTH, i.e.
    libmvec powf for NBUFF <= 4*(LENGTH/4) and scalar powf for the rest in stock mode).  Every value only
    depends on its (layer, point), so any block of layers and points can be evaluated separately."""

    def __init__(self, ctx: Ctx, f10: dict, ablog: np.ndarray):
        n = ctx.length
        self.ctx = ctx
        self.ablog = ablog.astype(F32)               # [nrhox, 3, nedge-1]
        wled = np.abs(f10["wledge"]).astype(np.float64)
        self.wled = wled
        self.half = (wled[:-1] + wled[1:]) * .5
        self.dele = (wled[1:] - wled[:-1]) ** 2 * .5
        wave = ctx.wbegin * powi(ctx.ratio, np.arange(n))
        # the Fortran's sequential edge pointer (IEDGE reset to 1 per layer, advanced while WAVE >= WLEDGE(IEDGE+1))
        if n and np.all(np.diff(wled) > 0) and np.all(np.diff(wave) >= 0):
            iedge = np.maximum(np.searchsorted(wled, wave, side="right"), 1)
            if np.any(iedge >= len(wled)):
                raise IndexError("SYNTHE: wavelength beyond the last continuum edge")
        else:
            iedge = np.empty(n, np.int64)
            ie = 1
            for k in range(n):
                while wave[k] >= wled[ie]:
                    ie += 1
                iedge[k] = ie
        self.i0 = (iedge - 1).astype(np.int64)        # 0-based IEDGE
        self.nv = n // 4 * 4 if ctx.mode == "stock" else 0

    def block(self, j0: int, j1: int, k0: int = 0, k1: int | None = None) -> torch.Tensor:
        """float32 [j1-j0, k1-k0]: layers j0..j1-1, points k0..k1-1 (0-based)."""
        ctx = self.ctx
        k1 = ctx.length if k1 is None else k1
        i0 = self.i0[k0:k1]
        w = (ctx.wbegin * powi(ctx.ratio, np.arange(k0, k1)))[None, :]
        ab = self.ablog[j0:j1]
        a1 = ab[:, 0, i0].astype(np.float64)
        a2 = ab[:, 1, i0].astype(np.float64)
        a3 = ab[:, 2, i0].astype(np.float64)
        wl0, wl1, hf, de = self.wled[i0], self.wled[i0 + 1], self.half[i0], self.dele[i0]
        c = ((w - hf) * (w - wl1) * a1 + (wl0 - w) * (w - wl1) * 2. * a2 + (w - wl0) * (w - hf) * a3) / de
        del a1, a2, a3
        c32 = torch.as_tensor(c.astype(F32), device=ctx.device)
        del c
        ten = torch.full_like(c32, 10.0)
        kv = min(max(self.nv - k0, 0), k1 - k0)       # points of this block inside the vectorised part
        if kv == k1 - k0:
            return L.vpowf(ten, c32)
        if kv == 0:
            return L.spowf(ten, c32)
        return torch.cat([L.vpowf(ten[:, :kv], c32[:, :kv]), L.spowf(ten[:, kv:], c32[:, kv:])], dim=1)

    def layers(self, j0: int, j1: int, step: int | None = None) -> torch.Tensor:
        n = self.ctx.length
        step = step or max(4096, (1 << 21) // max(1, j1 - j0))       # bounded REAL*8 temporaries
        out = torch.empty((j1 - j0, n), dtype=F, device=self.ctx.device)
        for k0 in range(0, n, step):
            out[:, k0:k0 + step] = self.block(j0, j1, k0, min(n, k0 + step))
        return out

    def at(self, lay: torch.Tensor, k: int) -> torch.Tensor:
        """CONTINUUM(k+1) of the layers `lay` (all-layer context: READBCS bookkeeping)."""
        col = self.block(0, self.ctx.nrhox, k, k + 1)[:, 0]
        return col[lay]


# per-layer attributes of Ctx (dimension 0 = layer) sliced for a layer block
_LAYER_ATTRS = ("t", "xne", "hkt", "hckt", "xnfh", "xnfhe", "xnfh2", "xnfpel", "dopple", "qxnfdop", "xnfdop",
                "txnxn", "t_np", "xne_np", "xnfhe_np", "xnfph_np", "velshift", "dopph")


class SyntheProgram:
    """One SYNTHE run.  `run()` returns a dict of outputs (also written to `workdir` unless write=False)."""

    def __init__(self, workdir, *, device=None, mode: str = "stock", line_chunk: int = 200_000,
                 max_contrib: int = 4_000_000, layer_block: int | None = None, mem_budget: int = 384 << 20,
                 keep_arrays: bool | None = None, point_chunk: int = 1 << 16):
        """line_chunk: fort.12 lines per batch (scaled down for wide layer blocks: <= 4M layer x line pairs);
        max_contrib: buffer contributions per ordered scatter-add; layer_block: layers per block (default:
        as many as fit BUFFER+CONTINUUM+CUTCONT of LENGTH points in `mem_budget` bytes); keep_arrays: also
        return BUFFER [nrhox, LENGTH] and ASYNTH [LENGTH, kw] (and keep ctx.continuum) in memory (default:
        when they are small); point_chunk: points per fort.14 -> fort.9 transposition batch."""
        self.wd = Path(workdir)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.mode = mode
        self.line_chunk = line_chunk
        self.max_contrib = max_contrib
        self.layer_block = layer_block
        self.mem_budget = mem_budget
        self.keep_arrays = keep_arrays
        self.point_chunk = point_chunk
        self.j0 = 0
        self.hprof4_full = None

    # ------------------------------------------------------------------ setup
    def _setup(self):
        wd = self.wd
        p = fortio.read_fort93(wd / "fort.93")
        self.p = p
        ratiolg = p["ratiolg"]
        ixwlbeg = int(math.log(p["wlbeg"]) / ratiolg)
        wbegin = math.exp(ixwlbeg * ratiolg)
        if wbegin < p["wlbeg"]:
            ixwlbeg += 1
            wbegin = math.exp(ixwlbeg * ratiolg)
        f18 = wd / "fort.18"
        ctx = Ctx(device=self.device, mode=self.mode, length=p["length"], nrhox=0, wbegin=wbegin,
                  ratio=p["ratio"], ratiolg=ratiolg, ixwlbeg=ixwlbeg, wlbeg=p["wlbeg"], wlend=p["wlend"],
                  resolu=p["resolu"], cutoff=float(F32(p["cutoff"])), linout=p["linout"], ifvac=p["ifvac"],
                  fort18=str(f18) if f18.exists() else None)
        self.ctx = ctx
        ctx.out.append(" NLINES=%20d LENGTH=%20d LINOUT=%10d" % (p["nlines"], p["length"], p["linout"]))
        if p["linout"] >= 0:
            raise NotImplementedError("LINOUT >= 0 (line identification output) is not ported")
        dev = self.device
        ctx.extab = torch.as_tensor(extab_common(self.mode), device=dev)
        ctx.h1tab = torch.as_tensor(h1tab_common(self.mode), device=dev)
        f10 = fortio.read_fort10(wd / "fort.10")
        self.f10 = f10
        nr = f10["nrhox"]
        ctx.nrhox = nr
        ctx.out.append(" TEFF=%10.1f   GRAV=%6.3f   %s" % (f10["teff"], f10["glog"], f10["title"].decode("latin-1")))
        ctx.out.append("%10d" % f10["ncon"])
        st = f10["state"]
        g = lambda k: st[k][:nr].astype(F32)  # noqa: E731
        ctx.t_np, ctx.xne_np = g("t"), g("xne")
        ctx.t = torch.as_tensor(ctx.t_np, device=dev)
        ctx.xne = torch.as_tensor(ctx.xne_np, device=dev)
        ctx.hkt = torch.as_tensor(g("hkt"), device=dev)
        ctx.hckt = torch.as_tensor(g("hckt"), device=dev)
        rho32 = g("rho")
        qrho = st["rho"][:nr]
        xnfh = g("xnfh")
        xnfhe = st["xnfhe"][:nr].astype(F32)
        xnfh2 = g("xnfh2")
        ctx.xnfh = torch.as_tensor(xnfh, device=dev)
        ctx.xnfhe = torch.as_tensor(xnfhe, device=dev)
        ctx.xnfh2 = torch.as_tensor(xnfh2, device=dev)
        ctx.xnfhe_np = xnfhe
        qx = f10["qxnfpel"]
        qd = f10["qdopple"]
        turbv = F32(p["turbv"])
        qd2 = np.sqrt(qd ** 2 + (float(turbv) / 299792.458) ** 2)
        x32 = np.where(qx < 1.e25, qx, 0.0).astype(F32)
        ctx.xnfpel = torch.as_tensor((x32 / rho32[:, None]).astype(F32), device=dev)
        ctx.dopple = torch.as_tensor(qd2.astype(F32), device=dev)
        qxnfdop = qx / qrho[:, None] / qd2
        ctx.qxnfdop = torch.as_tensor(qxnfdop, device=dev)
        ctx.xnfdop = torch.as_tensor(qxnfdop.astype(F32), device=dev)
        ctx.xnfph_np = qx[:, 0:2].astype(F32)
        txn = (xnfh.astype(np.float64) + .42 * xnfhe[:, 0].astype(np.float64) + .85 * xnfh2.astype(np.float64)) * \
            L.dpow(ctx.t_np.astype(np.float64) / 10000., float(F32(.3)))
        ctx.txnxn = torch.as_tensor(txn.astype(F32), device=dev)
        deckj = p["deckj"]
        ctx.velshift = deckj[0, :nr].astype(F32)
        self.nvshift = np.trunc(p["resolu"] * ctx.velshift.astype(np.float64) / 299792.458 + .5).astype(np.int64)
        ablog = f10["qablog"][:, :3 * (f10["nedge"] - 1)].reshape(nr, f10["nedge"] - 1, 3).transpose(0, 2, 1)
        self.cont = Continuum(ctx, f10, ablog)
        ctx.continuum = None                      # per layer block (see _block_ctx); all layers: ctx.cont_at
        ctx.cutcont = None
        ctx.buffer = None
        ctx.cont_at = self.cont.at
        ctx.mlines = torch.zeros(nr, dtype=torch.long, device=dev)
        ctx.dopph = torch.zeros(nr, dtype=F, device=dev)
        ctx.hprof4 = None
        ctx.emergeh_np = None
        ctx.emerge = None
        numrec = (ctx.length + LENREC - 1) // LENREC
        ctx.kprof = max(0, min(MAXPROF, numrec * LENREC - MAXBUFF))   # PROFILE values in fort.14's padding

    def _block_ctx(self, j0: int, j1: int) -> Ctx:
        """The context of layers j0..j1-1: per-layer arrays sliced, BUFFER/CONTINUUM of the block; the
        stdout/fort.15 accumulators are shared with the all-layer context."""
        full = self.full
        sub = copy.copy(full)
        for k in _LAYER_ATTRS:
            setattr(sub, k, getattr(full, k)[j0:j1])
        if full.emergeh_np is not None:
            sub.emergeh_np = full.emergeh_np[j0:j1]
        if full.emerge is not None:
            sub.emerge = full.emerge[j0:j1]
        sub.nrhox = j1 - j0
        sub.mlines = torch.zeros(j1 - j0, dtype=torch.long, device=self.device)
        sub.continuum = self.cont.layers(j0, j1)
        sub.cutcont = sub.continuum * sub.cutoff
        sub.buffer = torch.zeros((j1 - j0, sub.length), dtype=F, device=self.device)
        if sub.kprof > 0:
            sub.prof_val = torch.zeros((j1 - j0, sub.kprof), dtype=F, device=self.device)
            sub.prof_has = torch.zeros((j1 - j0, sub.kprof), dtype=torch.bool, device=self.device)
        if self.hprof4_full is None:
            st4 = dict(t=full.t, xne=full.xne, xnfhe1=full.xnfhe[:, 0], xnfh2=full.xnfh2,
                       xnfph1=torch.as_tensor(full.xnfph_np[:, 0], device=self.device),
                       xnfph2=torch.as_tensor(full.xnfph_np[:, 1], device=self.device))
            self.hprof4_full = HProf4(st4, full.extab, mode=self.mode)
        sub.hprof4 = self.hprof4_full.sliced(j0, j1)
        return sub

    def _blocks(self):
        nr, n = self.full.nrhox, self.full.length
        lb = self.layer_block
        if lb is None:
            lb = max(1, min(nr, self.mem_budget // max(1, 12 * n)))
        lb = max(1, min(nr, int(lb)))
        return [(j0, min(nr, j0 + lb)) for j0 in range(0, nr, lb)]

    # ------------------------------------------------------------------ fort.12 lines
    def _main_lines(self, lines: np.ndarray, alpha_print: dict):
        ctx = self.ctx
        dev = self.device
        nr = ctx.nrhox
        length = ctx.length
        nvs = torch.as_tensor(self.nvshift[self.j0:self.j0 + nr], device=dev)
        n = len(lines)
        step = max(1, min(self.line_chunk, 4_000_000 // max(1, nr)))
        for c0 in range(0, n, step):
            ch = lines[c0:c0 + step]
            nb = torch.as_tensor(ch["nbuff"].astype(np.int64), device=dev)
            nel = torch.as_tensor(ch["nelion"].astype(np.int64) - 1, device=dev)
            congf = torch.as_tensor(ch["congf"].astype(np.float64), device=dev)
            # pass 1 (all layers x lines): KAPPA0 = CONGF*QXNFDOP(NELION) >= KAPMIN
            kmidx = torch.clamp(nb, 1, length) - 1
            kappa0 = (congf.unsqueeze(0) * ctx.qxnfdop[:, nel]).to(F)
            kapmin = ctx.continuum[:, kmidx] * ctx.cutoff
            ok = ~(kappa0 < kapmin)
            jj, ii = torch.nonzero(ok, as_tuple=True)        # layer-major, file order within a layer
            if not jj.numel():
                continue
            k0 = kappa0[jj, ii]
            km = kapmin[jj, ii]
            elo = torch.as_tensor(ch["elo"], device=dev)[ii]
            k0 = k0 * fastex(elo * ctx.hckt[jj], ctx.extab)
            ok2 = ~(k0 < km)
            jj, ii, k0, km = jj[ok2], ii[ok2], k0[ok2], km[ok2]
            if not jj.numel():
                continue
            self._pairs(ch, c0, jj, ii, k0, km, nel, nb, nvs, alpha_print)
            del kappa0, kapmin, ok

    def _pairs(self, ch, c0, jj, ii, k0, km, nel, nb, nvs, alpha_print):
        ctx = self.ctx
        dev = self.device
        length = ctx.length
        ne = nel[ii]
        alpha = torch.as_tensor(ch["alpha"], device=dev)[ii]
        txn = ctx.txnxn[jj]
        if bool((alpha != 0).any()):
            txn = self._alpha_sticky(ch, c0, jj, ii, alpha, nb, alpha_print)
        gr = torch.as_tensor(ch["gamrf"], device=dev)[ii]
        gs = torch.as_tensor(ch["gamsf"], device=dev)[ii]
        gw = torch.as_tensor(ch["gamwf"], device=dev)[ii]
        dop = ctx.dopple[jj, ne]
        adamp = (gr + gs * ctx.xne[jj] + gw * txn) / dop
        nbs = nb[ii] + nvs[jj]
        inwin = (nbs >= 1) & (nbs <= length)
        ctx.mlines += torch.bincount(jj[inwin], minlength=ctx.nrhox)
        small = adamp < 0.2
        kapcen = torch.where(small, k0 * (1.0 - 1.128 * adamp),
                             k0 * voigt(torch.zeros_like(adamp), adamp, ctx.h1tab))
        dopd = dop.to(D) * ctx.resolu
        n10 = torch.trunc(10.0 * dopd).to(torch.long)
        tabstep = rdiv(200.0, dopd).to(F)
        dvoigt = (rdiv(1.0, dop).to(D) / ctx.resolu).to(F)
        # process the pairs in sub-batches bounded by the number of buffer contributions
        npair = jj.numel()
        start = 0
        while start < npair:
            stop = min(npair, start + 200_000)
            sl = slice(start, stop)
            self._profiles(jj[sl], nbs[sl], inwin[sl], k0[sl], km[sl], adamp[sl], small[sl], kapcen[sl], n10[sl],
                           tabstep[sl], dvoigt[sl])
            start = stop

    def _alpha_sticky(self, ch, c0, jj, ii, alpha, nb, alpha_print):
        """TXNXN for every passing pair: the alpha!=0 branch rewrites TXNXN(J), which then holds for the
        following lines of the layer (and is reset at the next layer)."""
        ctx = self.ctx
        dev = self.device
        isa = alpha != 0
        base = ctx.txnxn.clone()
        npair = jj.numel()
        ar = torch.arange(npair, device=dev)
        big = npair + 1
        key = torch.where(isa, jj * big + ar, jj * big - 1)
        cm = torch.cummax(key, dim=0).values
        has = cm >= jj * big
        src = torch.where(has, cm - jj * big, torch.zeros_like(cm))
        # value of TXNXN set by each alpha pair
        vals = torch.zeros(npair, dtype=F, device=dev)
        ia = torch.nonzero(isa).squeeze(1)
        nelion = torch.as_tensor(ch["nelion"], device=dev)[ii]
        for k in ia.tolist():   # alpha lines are rare: per-pair scalar evaluation
            j = int(jj[k])
            lay = torch.tensor([j], device=dev)
            hf, he, h2 = ctx.alpha_txnxn(lay, int(nelion[k]), F32(alpha[k].item()))
            vals[k] = ctx.txnxn[j]
            wavel = ctx.wbegin * float(powi(ctx.ratio, int(nb[ii[k]]) - 1))
            nelem = int(int(nelion[k]) / 6) + 1
            v2 = (F32(1.) - F32(alpha[k].item())) / F32(2)
            ja = j + self.j0                      # absolute layer
            alpha_print.setdefault(ja, []).append("%10.3f%10dalpha=%10.2f%10.2f" % (wavel, nelem, alpha[k].item(), v2))
            if ja == 0:
                alpha_print[ja].append("hfactor%s" % "".join(_e104(x) for x in (hf.item(), he.item(), h2.item())))
        return torch.where(has, vals[src], base[jj])

    def _profiles(self, jj, nbs, inwin, k0, km, adamp, small, kapcen, n10, tabstep, dvoigt):
        """Line profiles of a batch of (layer, line) pairs and their ordered addition to the buffers.

        Core (inside 10 Doppler widths): PROFILE(k) for k=1..N10DOP until the first value < KAPMIN;
        computed column by column only for the pairs still inside their core (TABI is a sequential REAL*4
        sum, so each column needs the previous one)."""
        ctx = self.ctx
        dev = self.device
        h = ctx.h1tab
        ni = h.numel()
        npair = jj.numel()
        kmax = int(n10.max()) if npair else 0
        core = torch.zeros((npair, max(kmax, 1)), dtype=F, device=dev)
        exitk = torch.full((npair,), -1, dtype=torch.long, device=dev)
        rows = torch.arange(npair, device=dev)          # pairs still inside their core
        tabi = torch.full((npair,), 1.5, dtype=F, device=dev)
        k = 0
        while rows.numel() and k < kmax:
            # advance a block of columns for the current rows, then drop the rows that exited
            kb = min(kmax, k + (8 if k < 16 else 32))
            nb_ = kb - k
            n10r = n10[rows]
            sm = small[rows]
            ts = tabstep[rows]
            t = tabi[rows]
            cols = []
            for _ in range(nb_):
                t = t + ts
                cols.append(t)
            tabi[rows] = t
            tb = torch.stack(cols, 1)                                        # [nr, nb_]
            it = torch.trunc(tb).to(torch.long)
            k0r = k0[rows].unsqueeze(1)
            pr = k0r * (h[(it - 1).clamp(0, ni - 1)] + adamp[rows].unsqueeze(1) * h[(it + 2000).clamp(0, ni - 1)])
            if not bool(sm.all()):
                lr = torch.nonzero(~sm).squeeze(1)
                kk = torch.arange(k + 1, kb + 1, device=dev, dtype=F)
                v = kk.unsqueeze(0) * dvoigt[rows[lr]].unsqueeze(1)
                pr[lr] = k0r[lr] * voigt(v, adamp[rows[lr]].unsqueeze(1), h)
            kcol = torch.arange(k + 1, kb + 1, device=dev)
            valid = kcol.unsqueeze(0) <= n10r.unsqueeze(1)
            pr = torch.where(valid, pr, torch.zeros_like(pr))
            core[rows, k:kb] = pr
            ex = (pr < km[rows].unsqueeze(1)) & valid
            anyex = ex.any(1)
            first = torch.where(ex, kcol.unsqueeze(0), torch.full_like(ex, kb + 1, dtype=torch.long)).min(1).values
            exitk[rows[anyex]] = first[anyex]
            keep = ~anyex & (n10r > kb)
            rows = rows[keep]
            k = kb
        far = exitk < 0
        nstep = exitk.clone()
        # far wings: X = PROFILE(N10DOP)*N10DOP**2, MAXSTEP = SQRT(X/KAPMIN)+1
        last = torch.where(n10 > 0, core.gather(1, (n10 - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
                           torch.zeros_like(k0))
        n10f = n10.to(F)
        x = last * (n10f * n10f)
        x = torch.where(n10 > 0, x, torch.zeros_like(x))
        ms = (sqrtf(x / km) + 1.0)
        ms = torch.clamp(torch.trunc(torch.nan_to_num(ms, posinf=1e9)).to(torch.long), max=MAXPROF)
        nstep = torch.where(far, ms, nstep)
        if self.ctx.kprof > 0:
            self._track_profile(jj, core, exitk, n10, nstep, x, far)
        self._accumulate(jj, nbs, inwin, kapcen, core, nstep, n10, far, x)

    def _track_profile(self, jj, core, exitk, n10, nstep, x, far):
        """PROFILE(1..KPROF) as the Fortran leaves it after these pairs (layer-major, file order): each pair
        writes PROFILE(1..M), M = the core exit step, or MAX(N10DOP, MAXSTEP) with the far wings X/STEP**2.
        Only needed when the last fort.14 record of a layer reaches past BUFFER(MAXBUFF) into PROFILE."""
        ctx = self.ctx
        kp = ctx.kprof
        dev = self.device
        npair = jj.numel()
        if not npair:
            return
        m = torch.where(far, torch.maximum(n10, nstep), exitk)
        # suffix maximum of M within each layer segment (pairs are layer-major): flip, offset by segment
        big = 1 << 22
        seg = (ctx.nrhox - 1 - jj).flip(0)
        v = seg * big + m.flip(0)
        cm = torch.cummax(v, 0).values - seg * big                       # max of M from this pair on
        sm_incl = cm.flip(0)
        nxt = torch.zeros_like(sm_incl)
        same = jj[1:] == jj[:-1]
        nxt[:-1] = torch.where(same, sm_incl[1:], torch.zeros_like(sm_incl[1:]))
        rec = (m > nxt) & (nxt < kp)                                     # last writer of PROFILE(nxt+1..m)
        r = torch.nonzero(rec).squeeze(1)
        if not r.numel():
            return
        lo = nxt[r] + 1
        hi = torch.clamp(m[r], max=kp)
        cnt = hi - lo + 1
        rid = torch.repeat_interleave(torch.arange(r.numel(), device=dev), cnt)
        first = torch.cumsum(cnt, 0) - cnt
        k = torch.arange(int(cnt.sum()), device=dev) - first[rid] + lo[rid]          # 1-based steps
        pr = r[rid]
        incore = k <= n10[pr]
        cv = core[pr, torch.clamp(k - 1, min=0, max=core.shape[1] - 1)]
        kf = k.to(F)
        val = torch.where(incore, cv, x[pr] / (kf * kf))
        ctx.prof_val[jj[pr], k - 1] = val
        ctx.prof_has[jj[pr], k - 1] = True

    def _accumulate(self, jj, nbs, inwin, kapcen, core, nstep, n10, far, x):
        """Contributions of the pairs to the buffers, in pair order (= the Fortran's line order within a
        layer).  A pair covers the points NBUFF-NSTEP..NBUFF+NSTEP clipped to 1..LENGTH: PROFILE(|offset|)
        in the wings, KAPCEN at the centre (only reachable when NBUFF is inside the buffer)."""
        ctx = self.ctx
        dev = self.device
        length = ctx.length
        lo = torch.clamp(nbs - nstep, min=1)
        hi = torch.clamp(nbs + nstep, max=length)
        cnt = torch.clamp(hi - lo + 1, min=0)
        csum = torch.cumsum(cnt, 0)
        tot = int(csum[-1]) if csum.numel() else 0
        if tot == 0:
            return
        lim = self.max_contrib
        a = 0
        npair = cnt.numel()
        i32 = torch.int32
        while a < npair:
            base = int(csum[a - 1]) if a > 0 else 0
            b = int(torch.searchsorted(csum, torch.tensor(base + lim, device=dev), right=True))
            b = min(max(b, a + 1), npair)
            sl = slice(a, b)
            c = cnt[sl].to(i32)
            t = int(c.sum())
            if t:
                pid = torch.repeat_interleave(torch.arange(b - a, device=dev, dtype=i32), c)
                first = torch.cumsum(c, 0, dtype=i32) - c
                pnt = torch.arange(t, device=dev, dtype=i32) - first[pid] + lo[sl].to(i32)[pid]
                del first
                off = torch.abs(pnt - nbs[sl].to(i32)[pid])
                incore = off <= n10[sl].to(i32)[pid]
                cidx = torch.clamp(off - 1, min=0, max=core.shape[1] - 1)
                val = core[sl][pid, cidx]
                del cidx
                offf = off.to(F)
                vfar = x[sl][pid] / (offf * offf)
                val = torch.where(incore, val, vfar)
                del vfar, offf, incore
                val = torch.where(off == 0, kapcen[sl][pid], val)
                key = jj[sl].to(i32)[pid] * length + (pnt - 1)
                del pid, pnt, off
                self._ordered_add(key, val)
            a = b

    def _ordered_add(self, key: torch.Tensor, val: torch.Tensor):
        """buffer.view(-1)[key[i]] += val[i] sequentially in i order, exactly (float32)."""
        buf = self.ctx.buffer.view(-1)
        if buf.device.type == "cpu" and _index_add_is_sequential():
            buf.index_add_(0, key, val)        # CPU kernel: serial loop in index order
            return
        ordered_add_sorted(buf, key, val)

    # ------------------------------------------------------------------ run
    def run(self, write: bool = True, delete_inputs: bool = True, stdout_path=None) -> dict:
        self._setup()
        ctx = self.ctx
        self.full = ctx
        p = self.p
        f10 = self.f10
        wd = self.wd
        nr = ctx.nrhox
        length = ctx.length
        n19 = p["n19"]
        nl12 = p["nlines"]
        keep = self.keep_arrays
        if keep is None:
            keep = nr * length * 4 <= (64 << 20) or not write
        lines19 = fortio.read_fort19(wd / "fort.19", n19) if n19 > 0 else np.zeros(0, fortio.FORT19)
        progress = "".join("%12d /%12d\n" % (j + 1, nr) for j in range(nr))
        alpha_print: dict = {}
        xl_full = None
        if n19 > 0:
            xl_full = XLinop(ctx)
            xl_full._first_call()
            ctx.emergeh_np = xl_full.emergeh
            emerge_lines = ctx.out[-(nr + 1):]
            del ctx.out[-(nr + 1):]
            xl_full.prepare_bcs(lines19)
            he_cache = {}

            def he_full():
                if "he" not in he_cache:
                    from .lineopac.helium import HeI
                    he_cache["he"] = HeI(dict(t=ctx.t_np, xne=ctx.xne_np, xnfph2=ctx.xnfph_np[:, 1],
                                              xnfhe2=ctx.xnfhe_np[:, 1]), ctx.h1tab, ctx.fort18, ctx.mode)
                return he_cache["he"]
        numrec = (length + LENREC - 1) // LENREC
        f14 = wd / "fort.14"
        fh14 = None
        if write:
            fh14 = open(f14, "r+b" if f14.exists() else "wb")
        buf_all = np.zeros((nr, length), F32) if keep else None
        prof_state = np.zeros(ctx.kprof, F32)
        cont_all = torch.zeros((nr, length), dtype=F, device=self.device) if keep else None
        try:
            for j0, j1 in self._blocks():
                sub = self._block_ctx(j0, j1)
                self.ctx, self.j0 = sub, j0
                if n19 > 0:
                    xl = XLinop(sub)
                    xl.first = False
                    for k in ("ehyd", "alphahyd"):
                        setattr(xl, k, getattr(xl_full, k))
                    xl.emerge = xl_full.emerge[j0:j1]
                    xl.emergeh = xl_full.emergeh[j0:j1]
                    sub.emerge = xl.emerge
                    xl._bcs_map = xl_full._bcs_map
                    xl._philam = xl_full._philam
                    xl._seq = xl_full._seq
                    xl.j0 = j0
                    xl.he_full = he_full
                    xl(lines19)
                if nl12 > 0:
                    for lines12 in fortio.iter_fort12(wd / "fort.12", nl12):
                        self._main_lines(lines12, alpha_print)
                ctx.mlines[j0:j1] = sub.mlines
                b = sub.buffer.cpu().numpy()
                if fh14 is not None:
                    # the last record of a layer: BUFFER(NBEG..NEND) past LENGTH (zeros up to MAXBUFF, then
                    # PROFILE(1..) as left by the lines of the layers so far: PROFILE is never reset)
                    pads = np.zeros((j1 - j0, numrec * LENREC - length), "<f4")
                    if ctx.kprof > 0:
                        has = sub.prof_has.cpu().numpy()
                        val = sub.prof_val.cpu().numpy()
                        for j in range(j1 - j0):          # carried from layer to layer (KPROF values)
                            prof_state = np.where(has[j], val[j], prof_state)
                            pads[j, -ctx.kprof:] = prof_state
                    for j in range(j0, j1):
                        fh14.seek(j * numrec * LENREC * 4)
                        fh14.write(b[j - j0].astype("<f4", copy=False).tobytes())
                        fh14.write(pads[j - j0].tobytes())
                if keep:
                    buf_all[j0:j1] = b
                    cont_all[j0:j1] = sub.continuum
                del sub, b
                self.ctx = ctx
        finally:
            self.ctx, self.j0 = ctx, 0
            if fh14 is not None:
                fh14.close()
        if keep:
            ctx.buffer = torch.as_tensor(buf_all, device=self.device)
            ctx.continuum = cont_all
            ctx.cutcont = cont_all * ctx.cutoff
        mlines = ctx.mlines.cpu().numpy()
        # stdout of the layer loop
        for j in range(nr):
            ctx.out.append("%5d  VELOCITY SHIFT%s%7d" % (j + 1, ff(ctx.velshift[j], 9, 3), self.nvshift[j]))
            if j == 0 and n19 > 0:
                ctx.out.extend(emerge_lines)
            ctx.out.extend(alpha_print.get(j, []))
            ctx.out.append("%s%10d%10d LINES USED" % (" " * 30, j + 1, mlines[j]))
        ilines = int(mlines.sum())
        ctx.out.append("%10d" % ilines)
        for nrec in range(1, numrec + 1):
            ctx.out.append("%10d" % nrec)
        ctx.out.append("%10d OPACITY VECTORS ON TAPE 9 " % length)
        nlines_tot = nl12 + n19
        ctx.out.append(" NLINES=%20d LINOUT=%10d" % (nlines_tot, p["linout"]))
        n9 = 0
        ctx.out.append(" N9=%10d" % n9)
        stdout = "\n".join(ctx.out) + "\n"
        # fort.9 headers
        h1 = struct.pack("<dddiii", p["wlbeg"], p["resolu"], p["wlend"], length, nr, p["linout"]) + \
            struct.pack("<f", float(F32(p["turbv"]))) + struct.pack("<i", p["ifvac"])
        nedge = f10["nedge"]
        e = np.stack([f10["frqedg"], np.abs(f10["wledge"]), f10["cmedge"]], axis=1).astype("<f8")
        h2 = struct.pack("<i", nedge) + e.tobytes() + f10["idmol"].astype("<f8").tobytes() + \
            f10["momass"].astype("<f8").tobytes()
        f15 = np.zeros(len(ctx.f15), fortio.FORT15)
        if ctx.f15:
            rec = sorted(ctx.f15, key=lambda t: (t[0], t[1]))
            f15["iline"] = [r[2] for r in rec]
            f15["kapcen"] = [r[3] for r in rec]
        asynth = None
        if write:
            # DIRECT IO TRANSPOSITION: fort.14 -> ASYNTH records of fort.9, in point chunks
            with open(wd / "fort.9", "wb") as f9, open(f14, "rb") as r14:
                write_record(f9, h1)
                write_record(f9, h2)
                for k0 in range(0, length, self.point_chunk):
                    k1 = min(length, k0 + self.point_chunk)
                    bt = np.empty((nr, k1 - k0), F32)
                    for j in range(nr):
                        r14.seek((j * numrec * LENREC + k0) * 4)
                        bt[j] = np.frombuffer(r14.read((k1 - k0) * 4), "<f4")
                    a = self._asynth(torch.as_tensor(bt, device=self.device), k0)
                    fortio.write_asynth_records(f9, a)
                    if keep:
                        asynth = a if asynth is None else np.concatenate([asynth, a])
                write_record(f9, struct.pack("<i", n9))
        elif keep:
            asynth = np.concatenate([self._asynth(ctx.buffer[:, k0:k0 + self.point_chunk], k0)
                                     for k0 in range(0, length, self.point_chunk)]) if length else \
                np.zeros((0, KW), F32)
        out = dict(stdout=stdout, asynth=asynth, buffer=buf_all, n9=n9, fort15=f15, progress=progress,
                   header1=h1, header2=h2, mlines=mlines)
        if write:
            fortio.write_int_record(wd / "fort.8", n9)
            if len(f15):
                fortio.write_fort15(wd / "fort.15", f15)
            (wd / "progress.dat").write_text(progress)
            if stdout_path is not None:
                Path(stdout_path).write_text(stdout)
            if delete_inputs:
                for nm in ("fort.93", "fort.12", "fort.13"):
                    try:
                        os.remove(wd / nm)
                    except FileNotFoundError:
                        pass
        return out

    def _asynth(self, buf: torch.Tensor, k0: int) -> np.ndarray:
        """ASYNTH(J) = BUFFER(J)*(1-EXP(-FREQ*HKT(J))) for the points k0.. of `buf` [nrhox, np] (vectorised
        expf over J in groups of 4, scalar expf for the NRHOX mod 4 last layers)."""
        ctx = self.ctx
        n = buf.shape[1]
        wave = ctx.wbegin * powi(ctx.ratio, np.arange(k0, k0 + n))
        freq = torch.as_tensor((2.99792458e17 / wave).astype(F32), device=self.device)
        arg = -(freq.unsqueeze(1) * ctx.hkt.unsqueeze(0))      # [n, nrhox]
        if self.mode == "stock":
            if ctx.nrhox - 1 <= 2:
                ex = L.sexpf(arg)
            else:
                ex = L.loopvec(L.vexpf, L.sexpf, arg, dim=1)
        else:
            ex = L.sexpf(arg)
        a = buf.t() * (1.0 - ex)
        out = np.zeros((n, KW), F32)
        out[:, :ctx.nrhox] = a.cpu().numpy()
        return out


def ordered_add_sorted(buf: torch.Tensor, key: torch.Tensor, val: torch.Tensor) -> None:
    """Device-independent exact sequential scatter-add: stable sort by key, then rank-wise float32
    additions (rank r = r-th contribution of its key in the original order)."""
    ks, perm = torch.sort(key, stable=True)
    vs = val[perm]
    del perm
    uk, cnts = torch.unique_consecutive(ks, return_counts=True)
    starts = torch.cumsum(cnts, 0) - cnts
    order = torch.argsort(cnts, descending=True, stable=True)
    cs = cnts[order]
    st = starts[order]
    kk = uk[order]
    maxc = int(cs[0]) if cs.numel() else 0
    ncnt = torch.searchsorted(-cs, -torch.arange(1, maxc + 1, device=cs.device), right=True).tolist()
    for rnk in range(maxc):
        m = ncnt[rnk]
        kr = kk[:m]
        buf[kr] = buf[kr] + vs[st[:m] + rnk]


_SEQ_OK = None


def _index_add_is_sequential() -> bool:
    """CPU index_add_ on a 1-D float32 tensor is a serial in-order loop (checked once at run time)."""
    global _SEQ_OK
    if _SEQ_OK is None:
        g = torch.Generator().manual_seed(1)
        key = torch.randint(0, 7, (20000,), generator=g)
        val = (torch.rand(20000, generator=g) * 10.0 ** torch.randint(-8, 8, (20000,), generator=g)).float()
        a = torch.zeros(7, dtype=F).index_add_(0, key, val)
        b = torch.zeros(7, dtype=F)
        ordered_add_sorted(b, key, val)
        _SEQ_OK = bool(torch.equal(a, b))
    return _SEQ_OK


def _e104(x: float) -> str:
    """Fortran 1PE10.4."""
    if x == 0:
        return "0.0000E+00".rjust(10)
    s = "%.4E" % x
    return s.rjust(10)


def run_synthe(workdir, **kw) -> dict:
    """Run the SYNTHE port in a SYNTHE work directory (see module docstring)."""
    write = kw.pop("write", True)
    delete_inputs = kw.pop("delete_inputs", True)
    stdout_path = kw.pop("stdout_path", None)
    return SyntheProgram(workdir, **kw).run(write=write, delete_inputs=delete_inputs, stdout_path=stdout_path)
