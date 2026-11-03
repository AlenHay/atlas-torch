"""DFSYNTHE (dfsynthe.for) in torch: opacity distribution functions for one or more of the 57 temperatures.

Pipeline per temperature (Fortran MAIN):
  SELECTLINES (lines/selectlines.py, numpy memmap streaming)
  per layer J (vectorised over all 25 layers): continuum (lines/continuum.py), XLINOP (lines/xlinop.py),
  TAPE12 line profiles (lines/accumulate.py), BUFFER=MAX(BUFFER,CONTINUUM), DFCALC for VT=0,
  turbulent broadening for VT=1,2,4,8 km/s (`broaden`, exact REAL*4 summation order) + DFCALC.
Outputs the five ODF step files (fort.15,16,17,18,20 = vt0,1,2,4,8) byte-identical in format to the Fortran.

REAL*4 arithmetic is reproduced operation by operation; see lines/f32.py for the compiler-runtime flavours
('stock' = ifx -O2 BasicATLAS build, 'O0' = debug/oracle build) and the libm caveat.  Device-agnostic: pass
`device=torch.device('cuda')`; exact per-bin float32 summation order is guaranteed on CPU (sequential 1-D
index_add_); on GPU atomics make it order-nondeterministic at the rounding level.
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
from .lines.selectlines import selectlines
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
    select_chunk: int = 2_000_000     # line-list entries per memmap chunk
    dfcalc_rows: int = 5              # spectra sorted at once in DFCALC
    bin_chunk: int = 700_000          # bins per broadening chunk
    verbose: bool = False


# ------------------------------------------------------------------------------------------ broadening
def broaden(b0: torch.Tensor, ivt: int, tabs: dict, bin_chunk: int = 0) -> torch.Tensor:
    """Turbulent broadening of the saved 0 km/s spectrum (MAIN labels 381-420), [NL, LENBUFF] -> [NL, LENGTH].

    The Fortran adds B10000(I)*VTPROF/VTCENTER record by record (10000 points); for a target bin b the terms
    arrive ordered by source s = b+d ascending, except that the centre term (d=0) comes at the start of b's
    record: in record 1 first of all, in record R>1 after the red terms from sources in record R-1.  We sum the
    REAL*4 products in exactly that order."""
    dev = b0.device
    nvt = int(tabs["nvt"][ivt - 1])
    vtc = tabs["vtcenter"][ivt - 1]
    vtp = tabs["vtprof"][:nvt, ivt - 1]
    nl = b0.shape[0]
    pad = torch.zeros((nl, nvt), dtype=F32, device=dev)
    src = torch.cat([pad, b0, pad], dim=1)                 # src[:, nvt + s - 1] = BUFFER(s)
    # generic case (record 1, or >= NVT bins into the record): centre first, then d = -NVT..-1, 1..NVT
    out = src[:, nvt:nvt + LENGTH] * vtc
    for d in list(range(-nvt, 0)) + list(range(1, nvt + 1)):
        out = out + src[:, nvt + d:nvt + d + LENGTH] * vtp[abs(d) - 1]
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
        acc = acc + src[:, nvt + b + d - 1] * w[None, :]
    out[:, b - 1] = acc
    return out


# ------------------------------------------------------------------------------------------ DFCALC
def _partitions(t):
    """BIG (IDF 1..328) and LIT (329..1540) partitions of the bins: (idf0, nbeg, nend) per partition."""
    return [(0, t.nbeg[:328], t.nend[:328]), (328, t.nbeg[328:], t.nend[328:])]


def dfcalc(buf: torch.Tensor, tabs_np, fl: Flavor, rows: int = 5) -> torch.Tensor:
    """DFCALC: buf [B, LENGTH] float32 -> ISTEPS int16 [B, 1540, 12]."""
    dev = buf.device
    B = buf.shape[0]
    out = torch.empty((B, 1540, 12), dtype=torch.int16, device=dev)
    nsteps = torch.from_numpy(tabs_np.nsteps).to(dev)            # (13, 1540), positions 1-based in segment
    for r0 in range(0, B, rows):
        x = buf[r0:r0 + rows]
        v = fl.log10f(x) * 1000.0
        d = v.to(torch.float64)
        ib = (torch.sign(d) * torch.floor(d.abs() + 0.5)).to(I64)  # NINT
        for idf0, nbeg, nend in _partitions(tabs_np):
            nseg = len(nbeg)
            lo, hi = int(nbeg[0]), int(nend[-1])
            seg = torch.from_numpy(np.repeat(np.arange(nseg), nend - nbeg + 1)).to(dev)
            k = ib[:, lo - 1:hi] + 30000                             # BISORT adds 30000
            # radix-sort semantics: stable sort by the low NBITS bits, NBITS = bit length of MAX(K) (0: no sort)
            mx = torch.full((k.shape[0], nseg), -2**62, dtype=I64, device=dev).scatter_reduce(
                1, seg[None, :].expand_as(k), k, reduce="amax")
            # BISORT halves MAX (truncating) until 0: number of passes = bit length of |MAX|
            nbits = ((mx.abs()[..., None] >> torch.arange(63, device=dev)) > 0).sum(-1)
            mask = (torch.ones_like(nbits) << nbits) - 1
            key = (k & mask.gather(1, seg[None, :].expand_as(k))) + (seg[None, :] << 40)
            order = torch.sort(key, dim=1, stable=True).indices
            srt = k.gather(1, order) - 30000
            cs = torch.cat([torch.zeros((k.shape[0], 1), dtype=I64, device=dev), torch.cumsum(srt, dim=1)], dim=1)
            segstart = torch.from_numpy(nbeg - lo).to(dev)           # 0-based offset of each segment
            ns = nsteps[:, idf0:idf0 + nseg]                         # (13, nseg)
            a = segstart[None, :] + ns[:-1] - 1                      # first position (0-based) of step
            bnd = segstart[None, :] + ns[1:] - 1                     # one past last
            isum = cs[:, bnd.reshape(-1)] - cs[:, a.reshape(-1)]
            n = (ns[1:] - ns[:-1]).reshape(-1)[None, :]
            st = torch.div(isum + n // 2, n, rounding_mode="trunc")  # Fortran integer division
            out[r0:r0 + rows, idf0:idf0 + nseg, :] = st.reshape(-1, 12, nseg).transpose(1, 2).to(torch.int16)
    return out


# ------------------------------------------------------------------------------------------ one temperature
def _log(cfg, phase, timing, cpu, extra=None):
    if cfg.verbose:
        print(f"  {phase}: wall {timing[phase]:.1f} s, cpu {cpu[phase]:.1f} s" + (f" {extra}" if extra else ""),
              flush=True)


def run_temperature(xnfpdf: str | Path, xnfpdfmax: str | Path, data_dir: str | Path, iteff: int,
                    cfg: Config | None = None, aa_init: float = 0.0) -> dict:
    """ODF steps of temperature `iteff` (1..57): {vt: (idfout int32 [25*1540], isteps int16 [25*1540, 12])}."""
    cfg = cfg or Config()
    fl, dev = cfg.flavor, cfg.device
    timing, cpu = {}, {}
    c0 = time.process_time()
    t0 = time.time()
    xt = odf_io.read_xnfpdf(xnfpdf, iteff)
    mx = odf_io.read_xnfpdfmax(xnfpdfmax, iteff)
    tape12, tape19, counts = selectlines(data_dir, mx.teff, mx.wledge, mx.xnfdopmax, fl, cfg.select_chunk)
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
    buf = torch.zeros(nl * LENBUFF, dtype=F32, device=dev)
    t0 = time.time()
    xl = XLinop(tabs, lay, cont, fl)
    xstats = xl.run(buf, tape19)
    timing["xlinop"] = time.time() - t0
    cpu["xlinop"] = time.process_time() - c0
    _log(cfg, "xlinop", timing, cpu)
    c0 = time.process_time()
    t0 = time.time()
    t12 = {k: torch.from_numpy(getattr(tape12, k).astype(np.int64) if getattr(tape12, k).dtype.kind == "i"
                               else getattr(tape12, k)).to(dev) for k in ("nbuff", "congf", "nelion", "gamr", "gams", "gamw")}
    acc = LineAccumulator(tabs, lay["xne"], lay["txnxn"], lay["dopple"], lay["xnfdop"], cont, aa_init=aa_init,
                          max_elements=cfg.max_elements)
    astats = acc.accumulate(buf, t12, chunk=cfg.line_chunk)
    timing["lines"] = time.time() - t0
    cpu["lines"] = time.process_time() - c0
    _log(cfg, "lines", timing, cpu)
    c0 = time.process_time()
    t0 = time.time()
    b = buf.view(nl, LENBUFF)
    b[:, :LENGTH] = torch.maximum(b[:, :LENGTH], cont)
    del cont
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
    res["info"] = dict(counts=counts, timing=timing, cpu=cpu, xlinop=xstats, lines=astats, aa_after=float(acc.aa_after),
                       extab_oob=xl.hyd.extab_oob, teff=mx.teff)
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
