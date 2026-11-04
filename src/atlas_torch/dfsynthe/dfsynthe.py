"""DFSYNTHE (dfsynthe.for) in torch: opacity distribution functions for one or more of the 57 temperatures.

Pipeline per temperature (Fortran MAIN):
  SELECTLINES (lines/selectlines.py, numpy memmap streaming; TAPE12 streamed twice, never stored)
  per layer J (vectorised over all 25 layers): continuum (lines/continuum.py), XLINOP (lines/xlinop.py),
  TAPE12 line profiles (lines/accumulate.py), BUFFER=MAX(BUFFER,CONTINUUM), DFCALC for VT=0,
  turbulent broadening for VT=1,2,4,8 km/s (`broaden`, exact REAL*4 summation order) + DFCALC.
Outputs the five ODF step files (fort.15,16,17,18,20 = vt0,1,2,4,8) byte-identical in format to the Fortran.

REAL*4 arithmetic is reproduced operation by operation; see lines/f32.py for the compiler-runtime flavours
('stock' = ifx -O2 BasicATLAS build, 'O0' = debug/oracle build) and the libm caveat.

Stock floating-point semantics (defaults): Config.ftz runs the temperature in the flush-to-zero/denormals-are-zero
environment of the ifx build (ftz.py; CPU, one torch thread), and Flavor('stock').fastmath reproduces the code
ifx -fp-model fast generated for the stock dfsynthe.exe where it matters (read from its machine code):
  * HPROF4 label 52 (Lyman alpha H2+ term): one numerator/denominator pair that underflows to 0/0 = NaN at low
    electron density -> the stock binary drops all opacity in the sampled bins of the Lyman-alpha wing
    (up to 3.6-dex ODF rows at cool T; lines/hydrogen.py _lya_h2plus_fast) - the qualitative one;
  * HPROF4 Lyman alpha H2+ cutoff 4000-20000 cm-1 red of the centre: ((XNFPH(J,2)*K)*DOP)*10.**(CUTOFF-14.),
    K=(1/C)*1.77245 folded; the source order's 10.**(CUTOFF-14.)/C is subnormal beyond ~13100 cm-1 and FTZ
    would flush it (Hydrogen._lya_h2plus_cut_fast; 14-41-step ODF jumps at 12589-22387 K otherwise);
  * XLINOP Inglis-Teller merging: NMERGE=XNE**(-2./15.)*1600.-1.5 (SVML powf), EMERGE=(1/NMERGE**2)*109737.312
    with rcpps + Newton in the 4-wide vector loop (XLinop._merge_energies; one ulp of EMERGE decides whether the
    He II Lyman far wings start at WCON in the densest layer: 119 steps at 112202 K otherwise);
  * HPROF4 Doppler core: D=|FREQ-(FINEST+FREQ0)|*(1/DOP), 4-lane sums (Hydrogen._doppler_fast);
  * HPROF4 everything else (Hydrogen._depth_fast, setup, prep, _finish_fast, sofbet(fast=True), vcse1f_fast,
    _hfnm_fast): the per-temperature depth vectors with __svml_powf4, rcpps and rsqrtps (+ Newton) for the 4-wide
    layers and __svml_powf1/divss/sqrtss for the rest; the per-line constants with ifx's folded constants and
    reciprocals; the Lorentz, Stark, CUTOFF-interpolation, SOFBET and VCSE1F regroupings; libimf logf/expf/powf
    for the scalar calls (lines/imf.py);
  * XLINOP: hydrogen-wing interpolation (libimf logf, (NEW-OLD)*.05, __svml_expf4/__svml_expf1), normal lines
    (WTAIL=1.E7/((-500.-EMERGE)+CONTX), VVOIGT=|WAVE-WL|*(1./DOPWL), taper ((WAVE-WCON)*KAPPA)*(1./(WTAIL-WCON)),
    blue WAVE*(1./RATIO), voigt_ asymptotic branch with R=1./U), merged continua (FLOAT(..)*(1./DNBUFF)),
    autoionizing lines (red FREQ*(1./RATIO), EPSIL=(FREQ-FRELIN)*(2/GAMMAR), ((A*E+B)*K0)/((E*E+1)*B));
  * TABVOIGT/MAP1 (tables.map1_f32_fast, _atab_fast: rcpps divisions, regrouped Horner forms), EXTAB with
    __svml_expf4/__svml_expf1, EXPI (E1TAB) with R=1./X and libimf, the continuum edge coefficients with one
    reciprocal of DELEDGE (continuum.edge_coefficients);
  * SELECTLINES: TAPE12(2)=((1.-EXP(..))*CONGF)/FREQ4, TABLOG(..)*(1./FRQ4PI), TABLOG/TABBOLT by
    __svml_powf4/__svml_expf4, libimf expf per line, NLTE EXP((HCKT*(-1.E7))/WL) (lines/selectlines.py);
  * TABVOIGT: (I-1)*(1/VSTEPS) and SVML expf for H0TAB (lines/tables.py, lines/svml.py); VTTAB: I*((1/V)*C),
    VTPROF*(1/SUM);
  * MAIN: XNFPEL*(1/RHO), XNFDOP and the vectorised X/NSTEP**2 through rcpps + Newton step with the measured
    rcpps table of the reference CPU (lines/rcpps.py; rsqrtps likewise); ADAMP>.2 centres and far wings with one
    reciprocal 1/U (LineAccumulator._asym_fast), IV=NSTEP*(DVOIGT*200)+1.5.
  With the Intel runtime (libimf/libsvml, loaded with ctypes) and the C helpers (`make native`), the stock flavour
  reproduces the stock SELECTLINES output, tables, continuum and per-layer opacity buffers bit for bit, and the ODFs
  at 5012, 12589 and 22387 K exactly (tests/stage/test_dfsynthe_stock_state.py, test_dfsynthe_prog.py).  Without
  the Intel runtime the correctly rounded functions are used instead (rounding-level differences).
  Flavor('stock', fastmath=False) reproduces the ifx -fp-model precise build instead (within 1 step in <1e-5).

Memory/speed: TAPE12 is streamed from the line lists (twice: AA pre-pass, accumulation) and never stored.  The
two loops that dominate (TAPE12 line profiles: up to 2.5e11 profile points at 1995 K; XLINOP normal lines) run in
a small C kernel (lines/native/linesum.c, built with gcc, loaded with ctypes) that transcribes the Fortran loop
literally and is bit-identical to the vectorised torch path (tests/oracle/test_dfsynthe_lines.py,
test_dfsynthe_xlinop.py); without the built library the torch path is used (exact, ~20x slower).  Measured on
this machine, 4 temperatures in parallel: 1995 K 293 s, 5012 K ~2 min, full 57-temperature mix 34 min, peak
RSS < 2 GB per worker.  Exact per-bin float32 summation order is guaranteed on CPU.  Optional:
Config.lines_device=torch.device('cuda') runs only the TAPE12 accumulation on an accelerator (same float32
values, atomic adds -> order non-deterministic at the rounding level); everything else stays on the CPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import odf_io
from .lines.accumulate import LineAccumulator
from .lines.continuum import continuum, layer_setup
from .lines.f32 import F32, I64, Flavor
from .lines.selectlines import iter_tape12, select_tape19
from .lines.tables import LENBUFF, LENGTH, build_tables, device_tables
from .lines.xlinop import XLinop

RECLEN = 10000
IVTS = (1, 2, 4, 8)


@dataclass
class Config:
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    flavor: Flavor = field(default_factory=Flavor)
    line_chunk: int = 50_000          # TAPE12 lines per chunk
    max_elements: int = 8_000_000     # profile points generated at once (~0.5 GB of temporaries)
    select_chunk: int = 1_000_000     # line-list entries per memmap chunk (memory)
    dfcalc_rows: int = 2              # spectra converted at once in DFCALC (memory)
    bin_chunk: int = 700_000          # bins per broadening chunk
    ftz: bool = True                  # stock flush-to-zero/denormals-are-zero environment (ftz.py); CPU: 1 thread
    lines_device: torch.device | None = None   # optional device for the TAPE12 line accumulation only (e.g.
    #   torch.device('cuda'): float32 profile values are the same IEEE operations, but the atomic adds make the
    #   per-bin summation order non-deterministic -> rounding-level differences; not the default)
    verbose: bool = False


# ------------------------------------------------------------------------------------------ broadening
def _broaden_native(b0, nvt, vtc, vtp):
    """broaden() through lines/native/odfcalc.c broaden_rows (same REAL*4 terms in the same order), or None."""
    if b0.device.type != "cpu" or b0.dtype != F32 or not b0.is_contiguous() or nvt < 1:
        return None
    from .lines.native import load_extra
    lib = load_extra("odfcalc")
    if lib is None:
        return None
    import ctypes
    out = torch.empty((b0.shape[0], LENGTH), dtype=F32)
    w = vtp.to("cpu", F32).contiguous()
    rc = lib.broaden_rows(ctypes.c_int(b0.shape[0]), ctypes.c_int64(b0.shape[1]), ctypes.c_int64(LENGTH),
                          ctypes.c_int(nvt), ctypes.c_float(float(vtc)), ctypes.c_void_p(w.data_ptr()),
                          ctypes.c_void_p(b0.data_ptr()), ctypes.c_void_p(out.data_ptr()))
    return out if rc == 0 else None


def broaden(b0: torch.Tensor, ivt: int, tabs: dict, bin_chunk: int = 0, native: bool | None = None) -> torch.Tensor:
    """Turbulent broadening of the saved 0 km/s spectrum (MAIN labels 381-420), [NL, LENBUFF] -> [NL, LENGTH].

    The Fortran adds B10000(I)*VTPROF/VTCENTER record by record (10000 points); for a target bin b the terms
    arrive ordered by source s = b+d ascending, except that the centre term (d=0) comes at the start of b's
    record: in record 1 first of all, in record R>1 after the red terms from sources in record R-1.  We sum the
    REAL*4 products in exactly that order (separate IEEE multiply and add, no FMA), in cache-sized bin chunks
    with in-place operations."""
    dev = b0.device
    nvt = int(tabs["nvt"][ivt - 1])
    vtc = tabs["vtcenter"][ivt - 1]
    vtp = tabs["vtprof"][:nvt, ivt - 1]
    nl = b0.shape[0]
    if native is not False:
        r = _broaden_native(b0, nvt, vtc, vtp)
        if r is not None:
            return r
    C = bin_chunk if bin_chunk and bin_chunk < 200_000 else 16384
    out = torch.empty((nl, LENGTH), dtype=F32, device=dev)
    tmp = torch.empty((nl, C), dtype=F32, device=dev)
    terms = [(d, vtp[abs(d) - 1]) for d in list(range(-nvt, 0)) + list(range(1, nvt + 1))]
    head = torch.cat([torch.zeros((nl, nvt), dtype=F32, device=dev), b0[:, :C + nvt]], dim=1)

    def src(lo, hi):                                       # BUFFER(lo+1..hi) (0 below BUFFER(1))
        return b0[:, lo:hi] if lo >= 0 else head[:, nvt + lo:nvt + hi]
    # generic case (record 1, or >= NVT bins into the record): centre first, then d = -NVT..-1, 1..NVT
    for s0 in range(0, LENGTH, C):
        e = min(LENGTH, s0 + C)
        o = out[:, s0:e]
        t = tmp[:, :e - s0]
        torch.mul(src(s0, e), vtc, out=o)
        for d, w in terms:
            torch.mul(src(s0 + d, e + d), w, out=t)
            o.add_(t)
    del tmp, head
    # first NVT bins of records 2..: the centre term comes after the red terms of the previous record
    lst = torch.cat([torch.arange(-nvt, 0, device=dev), torch.arange(1, nvt + 1, device=dev)])
    wts = torch.cat([vtp.flip(0), vtp])
    rs = torch.arange(RECLEN + 1, LENGTH + 1, RECLEN, device=dev)
    b = (rs[:, None] + torch.arange(nvt, device=dev)[None, :]).reshape(-1)
    b = b[b <= LENGTH]
    kc = nvt - (b - ((b - 1) // RECLEN) * RECLEN - 1)
    acc = torch.zeros((nl, b.numel()), dtype=F32, device=dev)
    for t in range(2 * nvt + 1):
        tt = torch.full_like(b, t)
        li = torch.where(tt < kc, tt, tt - 1).clamp(0, 2 * nvt - 1)
        d = torch.where(tt == kc, torch.zeros_like(b), lst[li])
        w = torch.where(tt == kc, vtc, wts[li])
        acc = acc + b0[:, b + d - 1] * w[None, :]
    out[:, b - 1] = acc
    return out


# ------------------------------------------------------------------------------------------ DFCALC
def _partitions(t):
    """BIG (IDF 1..328) and LIT (329..1540) partitions of the bins: (idf0, nbeg, nend) per partition."""
    return [(0, t.nbeg[:328], t.nend[:328]), (328, t.nbeg[328:], t.nend[328:])]


def _rank_sums_sorted(k: torch.Tensor, seg: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Reference path (any K): BISORT semantics = stable sort by the low NBITS bits of K (NBITS = bit length of
    |MAX(K)| per segment), then prefix sums of the sorted values at positions P.  k [R, n] int64 (K+30000)."""
    dev = k.device
    nseg = int(seg[-1]) + 1
    mx = torch.full((k.shape[0], nseg), -2**62, dtype=I64, device=dev).scatter_reduce(
        1, seg[None, :].expand_as(k), k, reduce="amax")
    # BISORT halves MAX (truncating) until 0: number of passes = bit length of |MAX|
    nbits = ((mx.abs()[..., None] >> torch.arange(63, device=dev)) > 0).sum(-1)
    mask = (torch.ones_like(nbits) << nbits) - 1
    key = (k & mask.gather(1, seg[None, :].expand_as(k))) + (seg[None, :] << 40)
    order = torch.sort(key, dim=1, stable=True).indices
    srt = k.gather(1, order) - 30000
    cs = torch.cat([torch.zeros((k.shape[0], 1), dtype=I64, device=dev), torch.cumsum(srt, dim=1)], dim=1)
    return cs[:, P]


def _rank_sums_hist(k: np.ndarray, seg: np.ndarray, segstart: np.ndarray, P: np.ndarray):
    """Same as _rank_sums_sorted for one row when every K >= 0 (then the radix sort is a plain sort by value):
    counting sort per segment (one concatenated histogram, O(n)), prefix sums of sorted values at positions P.
    Returns None when the fast path does not apply (some K < 0, or too wide a value range)."""
    mn = np.minimum.reduceat(k, segstart)
    if mn.min() < 0:
        return None
    mx = np.maximum.reduceat(k, segstart)
    width = mx - mn + 1
    H = int(width.sum())
    if H > 8 * k.size + 1_000_000:
        return None
    off = np.cumsum(width) - width
    idx = off[seg] + (k - mn[seg])
    C = np.cumsum(np.bincount(idx, minlength=H))
    # sums of values per histogram bin: integers < 2**53 in float64 (exact)
    W = np.cumsum(np.bincount(idx, weights=(k - 30000).astype(np.float64), minlength=H))
    q = np.searchsorted(C, P, side="left")
    qm = np.maximum(q - 1, 0)
    Cp = np.where(q > 0, C[qm], 0)
    Wp = np.where(q > 0, W[qm], 0.0).astype(np.int64)
    sq = np.searchsorted(off, q, side="right") - 1                      # segment of histogram bin q
    vq = q - off[sq] + mn[sq] - 30000
    return Wp + (P - Cp) * vq


_DFC_CACHE: dict = {}


def _dfcalc_plan(tabs_np):
    key = id(tabs_np)
    if key not in _DFC_CACHE:
        plan = []
        for idf0, nbeg, nend in _partitions(tabs_np):
            nseg = len(nbeg)
            lo, hi = int(nbeg[0]), int(nend[-1])
            seg = np.repeat(np.arange(nseg), nend - nbeg + 1)
            segstart = (nbeg - lo).astype(np.int64)
            ns = tabs_np.nsteps[:, idf0:idf0 + nseg]                 # (13, nseg), positions 1-based in segment
            a = segstart[None, :] + ns[:-1] - 1                      # first position (0-based) of each step
            bnd = segstart[None, :] + ns[1:] - 1                     # one past last
            n = (ns[1:] - ns[:-1]).reshape(-1)
            plan.append((idf0, nseg, lo, hi, seg, segstart, a.reshape(-1), bnd.reshape(-1), n))
        _DFC_CACHE.clear()
        _DFC_CACHE[key] = (tabs_np, plan)
    return _DFC_CACHE[key][1]


def _dfcalc_native(buf, tabs_np, fl, rows, out) -> bool:
    """DFCALC through lines/native/odfcalc.c (integer-exact, same ISTEPS); False if not built or a row has a
    non-finite LOG10 (then the reference path below runs)."""
    from .lines.native import load_extra
    lib = load_extra("odfcalc")
    if lib is None:
        return False
    import ctypes
    c = lambda a: np.ascontiguousarray(a, dtype=np.int64)  # noqa: E731
    nbeg, nend, nsteps = c(tabs_np.nbeg), c(tabs_np.nend), c(tabs_np.nsteps)
    assert nsteps.shape == (13, 1540)
    P = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
    for r0 in range(0, buf.shape[0], rows):
        v = (fl.log10f(buf[r0:r0 + rows]) * 1000.0).to("cpu", torch.float32).contiguous().numpy()
        o = np.empty((v.shape[0], 1540, 12), dtype=np.int16)
        rc = lib.dfcalc_rows(ctypes.c_int(v.shape[0]), ctypes.c_int64(v.shape[1]), ctypes.c_int64(v.shape[1]),
                             P(v), P(nbeg), P(nend), P(nsteps), P(o))
        if rc != 0:
            return False
        out[r0:r0 + v.shape[0]] = o
    return True


def dfcalc(buf: torch.Tensor, tabs_np, fl: Flavor, rows: int = 5, native: bool | None = None) -> torch.Tensor:
    """DFCALC: buf [B, LENGTH] float32 -> ISTEPS int16 [B, 1540, 12].

    IBUFFER = NINT(1000*LOG10(BUFFER)); per interval BISORT (radix sort of IBUFFER+30000) and integer averages
    over 12 rank ranges.  Only the sums of the sorted values between fixed ranks are needed: computed exactly
    (integer arithmetic) from per-segment histograms when all keys are >= 0, by the sort otherwise."""
    dev = buf.device
    B = buf.shape[0]
    out = np.empty((B, 1540, 12), dtype=np.int16)
    if native is not False and _dfcalc_native(buf, tabs_np, fl, rows, out):
        return torch.from_numpy(out).to(dev)
    plan = _dfcalc_plan(tabs_np)
    for r0 in range(0, B, rows):
        x = buf[r0:r0 + rows]
        v = fl.log10f(x) * 1000.0
        d = v.to(torch.float64)
        ib = (torch.sign(d) * torch.floor(d.abs() + 0.5)).to(I64).cpu().numpy()   # NINT
        for idf0, nseg, lo, hi, seg, segstart, a, bnd, n in plan:
            P = np.concatenate([bnd, a])
            for r in range(ib.shape[0]):
                k = ib[r, lo - 1:hi] + 30000                          # BISORT adds 30000
                S = _rank_sums_hist(k, seg, segstart, P)
                if S is None:
                    S = _rank_sums_sorted(torch.from_numpy(k)[None], torch.from_numpy(seg),
                                          torch.from_numpy(P))[0].numpy()
                isum = S[:a.size] - S[a.size:]
                t = isum + n // 2
                st = np.sign(t) * (np.abs(t) // n)                    # Fortran integer division (truncation)
                out[r0 + r, idf0:idf0 + nseg, :] = st.reshape(12, nseg).T
    return torch.from_numpy(out).to(dev)


# ------------------------------------------------------------------------------------------ one temperature
def _log(cfg, phase, timing, cpu, extra=None):
    if cfg.verbose:
        print(f"  {phase}: wall {timing[phase]:.1f} s, cpu {cpu[phase]:.1f} s" + (f" {extra}" if extra else ""),
              flush=True)


def run_temperature(xnfpdf: str | Path, xnfpdfmax: str | Path, data_dir: str | Path, iteff: int,
                    cfg: Config | None = None, aa_init: float = 0.0) -> dict:
    """ODF steps of temperature `iteff` (1..57): {vt: (idfout int32 [25*1540], isteps int16 [25*1540, 12])}."""
    cfg = cfg or Config()
    from .ftz import scope
    with scope(cfg.ftz and cfg.device.type == "cpu"):
        return _run_temperature(xnfpdf, xnfpdfmax, data_dir, iteff, cfg, aa_init)


def _run_temperature(xnfpdf, xnfpdfmax, data_dir, iteff, cfg, aa_init):
    fl, dev = cfg.flavor, cfg.device
    timing, cpu = {}, {}
    c0 = time.process_time()
    t0 = time.time()
    xt = odf_io.read_xnfpdf(xnfpdf, iteff)
    mx = odf_io.read_xnfpdfmax(xnfpdfmax, iteff)
    # TAPE19 now; TAPE12 is streamed from the line lists twice (AA pre-pass, accumulation), never stored
    counts: dict = {}
    tape19 = select_tape19(data_dir, mx.teff, mx.wledge, mx.xnfdopmax, fl)
    counts["nlte"] = len(tape19)
    timing["selectlines"] = time.time() - t0
    cpu["selectlines"] = time.process_time() - c0
    _log(cfg, "selectlines", timing, cpu, counts)
    c0 = time.process_time()
    t0 = time.time()
    tabs = device_tables(dev, fl)
    tnp = build_tables(fl)
    lay = layer_setup(xt, fl, dev)
    cont = continuum(xt.wledge, lay["ablog"], fl)
    timing["continuum"] = time.time() - t0
    cpu["continuum"] = time.process_time() - c0
    _log(cfg, "continuum", timing, cpu)
    c0 = time.process_time()
    nl = cont.shape[0]
    PAD = LineAccumulator.PAD
    bufpad = torch.zeros(nl * LENBUFF + 2 * PAD, dtype=F32, device=dev)
    buf = bufpad[PAD:PAD + nl * LENBUFF]
    t0 = time.time()
    xl = XLinop(tabs, lay, cont, fl)
    xstats = xl.run(buf, tape19)
    timing["xlinop"] = time.time() - t0
    cpu["xlinop"] = time.process_time() - c0
    _log(cfg, "xlinop", timing, cpu)
    c0 = time.process_time()
    t0 = time.time()
    ldev = cfg.lines_device if cfg.lines_device is not None else dev
    cols = ("nbuff", "congf", "nelion", "gamr", "gams", "gamw")

    def t12():                    # streamed SELECTLINES (int32 columns are widened chunk by chunk)
        for _, t in iter_tape12(data_dir, mx.teff, mx.wledge, mx.xnfdopmax, fl, cfg.select_chunk, counts):
            yield {k: torch.from_numpy(getattr(t, k)).to(ldev) for k in cols}
    if ldev == dev:
        acc = LineAccumulator(tabs, lay["xne"], lay["txnxn"], lay["dopple"], lay["xnfdop"], cont, aa_init=aa_init,
                              max_elements=cfg.max_elements, fastmath=fl.fastmath)
        astats = acc.accumulate(buf, t12, chunk=cfg.line_chunk, padded=bufpad)
    else:   # optional accelerator path: float32 values identical, per-bin order of the atomic adds is not
        lt = device_tables(ldev, fl)
        mv = lambda x: x.to(ldev)  # noqa: E731
        acc = LineAccumulator(lt, mv(lay["xne"]), mv(lay["txnxn"]), mv(lay["dopple"]), mv(lay["xnfdop"]), mv(cont),
                              aa_init=aa_init, max_elements=cfg.max_elements, fastmath=fl.fastmath)
        bp = bufpad.to(ldev)
        astats = acc.accumulate(bp[PAD:PAD + nl * LENBUFF], t12, chunk=cfg.line_chunk, padded=bp)
        bufpad.copy_(bp.to(dev))
        del bp
    timing["lines"] = time.time() - t0
    cpu["lines"] = time.process_time() - c0
    _log(cfg, "lines", timing, cpu)
    c0 = time.process_time()
    t0 = time.time()
    b = buf.view(nl, LENBUFF)
    # BUFFER=MAX(BUFFER,CONTINUUM) is `maxps buffer, continuum`: a NaN bin (stock Lyman alpha, see
    # Hydrogen._lya_h2plus_fast) becomes the continuum
    b[:, :LENGTH] = torch.where(b[:, :LENGTH] > cont, b[:, :LENGTH], cont)
    aa_after, extab_oob = float(acc.aa_after), xl.hyd.extab_oob
    del cont, acc, xl, lay                       # release the continuum (held by XLINOP/accumulator) before broadening
    res = {}
    j = np.arange(1, nl + 1)
    idf = np.arange(1, 1541)
    for ivt in (0,) + IVTS:
        spec = b[:, :LENGTH] if ivt == 0 else broaden(b, ivt, tabs, cfg.bin_chunk)
        st = dfcalc(spec, tnp, fl, cfg.dfcalc_rows)
        idfout = (ivt * 100000000 + idf[None, :] * 10000 + iteff * 100 + j[:, None]).reshape(-1).astype(np.int32)
        res[ivt] = (idfout, st.reshape(-1, 12).cpu().numpy())
        del spec
    timing["broaden+dfcalc"] = time.time() - t0
    cpu["broaden+dfcalc"] = time.process_time() - c0
    c0 = time.process_time()
    res["info"] = dict(counts=counts, timing=timing, cpu=cpu, xlinop=xstats, lines=astats, aa_after=aa_after,
                       extab_oob=extab_oob, teff=mx.teff)
    if cfg.verbose:
        print(f"T={mx.teff:.0f} counts={counts} timing={ {k: round(v, 1) for k, v in timing.items()} }")
    return res


def write_temperature(res: dict, outdir: str | Path, dft: str | None = None) -> dict:
    """Write vt{v}.bin (= fort.15/16/17/18/20) and the dfsortp ascii files like the BasicATLAS script."""
    from .dfsortp import dfsortp_arrays
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for v in odf_io.VTURBS:
        idfout, isteps = res[v]
        stem = f"dfp00t{dft}vt{v}" if dft else f"vt{v}"
        odf_io.write_odf_bin(outdir / f"{stem}.bin", idfout, isteps)
        idf, st = dfsortp_arrays(idfout, isteps)
        (outdir / f"{stem}sortp.asc").write_bytes(odf_io.format_sortp(idf, st))
        paths[v] = outdir / f"{stem}sortp.asc"
    return paths


def run_all(xnfpdf, xnfpdfmax, data_dir, outdir, cfg: Config | None = None, temps=None) -> None:
    """All (or selected 1-based) temperatures, each as an independent DFSYNTHE run (as BasicATLAS does),
    then SEPARATEDF -> p00big{v}.bdf / p00lit{v}.bdf in outdir."""
    from .separatedf import separatedf
    outdir = Path(outdir)
    temps = temps or range(1, 58)
    for it in temps:
        dft = str(int(float(odf_io.DFTS[it - 1])))
        if all((outdir / f"dfp00t{dft}vt{v}sortp.asc").exists() for v in odf_io.VTURBS):
            continue
        res = run_temperature(xnfpdf, xnfpdfmax, data_dir, it, cfg)
        write_temperature(res, outdir, dft)
    if len(list(temps)) == 57:
        for v in odf_io.VTURBS:
            files = [outdir / f"dfp00t{int(float(d))}vt{v}sortp.asc" for d in odf_io.DFTS]
            separatedf(files, outdir / f"p00big{v}.bdf", outdir / f"p00lit{v}.bdf")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="torch DFSYNTHE")
    ap.add_argument("--xnfpdf", required=True)
    ap.add_argument("--xnfpdfmax", required=True)
    ap.add_argument("--data", required=True, help="BasicATLAS data/dfsynthe_files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temps", default="all", help="'all' or comma list of 1-based indices")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--flavor", default="stock", choices=["stock", "O0"])
    a = ap.parse_args(argv)
    cfg = Config(device=torch.device(a.device), flavor=Flavor(a.flavor), verbose=True)
    temps = None if a.temps == "all" else [int(x) for x in a.temps.split(",")]
    run_all(a.xnfpdf, a.xnfpdfmax, a.data, a.out, cfg, temps)


if __name__ == "__main__":
    main()
