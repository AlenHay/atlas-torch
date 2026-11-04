"""PROGRAM SPECTRV (spectrv.for as patched by BasicATLAS): emergent flux / intensity spectrum from the
SYNTHE line opacity (fort.9), the XNFPELSYN continuum (fort.10) and the model (READIN state).

    state = <State after CALL READIN(20)>            (atlas_torch.synthe.readin, or a golden trace)
    inp = load_inputs(workdir)                       (fort.25 card, fort.10, fort.9, optional mask.bin)
    res = spectrv(state, inp)                        -> SpectrvResult; res.write("fort.7")
    run_spectrv(workdir, state=...)                  file-level driver (writes fort.7 = spectrum.bin)

Fortran semantics reproduced:
  * wavelength grid WAVE = WBEGIN*RATIO**(NU-1) with REAL**INTEGER evaluated by libgcc __powidf2
    (square-and-multiply), WBEGIN/RATIOLG by libm log/exp (Python math = glibc).
  * continuum: per-edge 3-point parabolic interpolation in log10 of the fort.10 opacities; the edge
    search uses the fort.9 WLEDGE but HALFEDGE/DELEDGE from fort.10 (as the Fortran does).
  * JOSH is called twice per point (continuum with IFSCAT from the model, then line+continuum with
    IFSCAT=1); RESID=HNU/SURF, Q(MU)=RESID*SURF(MU), Q(MU+NMU)=SURF(MU) are written per point.
  * the BasicATLAS patch: if mask.bin exists, byte NU != 0 keeps point NU; masked points (NU>1) still
    consume their fort.9 record but are neither solved nor written (NU=1 is always written).
  * the NLINES block (LINOUT>0 line-centre records) is ported; the N10 block is dead code (N10=0).
All JOSH calls are batched over wavelengths (chunked); see atlas_torch.synthe.josh for the REAL*4 solver.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..fnum import f32
from ..io.fortran import read_records, write_record
from ..state import State
from .josh import cdiv, exp_glibc, josh_batch, rdiv, w as w_format
from .lineopac.libm32 import dloopvec, dvexp, dvpow

F64 = torch.float64
KW = 99
NEDGE_MAX = 377


# ----------------------------------------------------------------------------------------------- formatted I/O helpers

def read_fw_d(field_: str, d: int) -> float:
    """Fortran Fw.d input (BLANK='NULL': blanks ignored; no decimal point -> implied d decimals)."""
    s = field_.replace(" ", "").upper().replace("D", "E")
    if not s:
        return 0.0
    if "." in s:
        return float(s)
    mant, _, ex = s.partition("E")
    if not ex:
        for i in range(1, len(mant)):
            if mant[i] in "+-":
                mant, ex = mant[:i], mant[i:]
                break
    v = int(mant) if mant not in ("", "+", "-") else 0
    return float(f"{v}e{int(ex or 0) - d}")


def fmt_f(x: float, w: int, d: int) -> str:
    s = f"{x:.{d}f}"
    if len(s) > w and (s.startswith("0.") or s.startswith("-0.")):
        s = s.replace("0.", ".", 1)
    return "*" * w if len(s) > w else s.rjust(w)


def fmt_i(n: int, w: int) -> str:
    s = str(int(n))
    return "*" * w if len(s) > w else s.rjust(w)


# ----------------------------------------------------------------------------------------------- inputs

@dataclass
class SpectrvInputs:
    card25: str                         # unit 25: RHOXJ,R1,R101,PH1,PC1,PSI1,PRDDOP,PRDPOW  (8F10.5)
    fort10: Path                        # XNFPELSYN continuum opacities
    fort9: Path                         # SYNTHE output: header, edges, ASYNTH(kw) per point, NLINES, lines
    mask: np.ndarray | None = None      # mask.bin bytes (int8) or None (no file)


def load_inputs(workdir: str | Path, mask: str | Path | None = "mask.bin") -> SpectrvInputs:
    wd = Path(workdir)
    m = None
    if mask is not None:
        mp = Path(mask) if Path(mask).is_absolute() else wd / mask
        if mp.exists():
            m = np.fromfile(mp, dtype=np.int8)
    return SpectrvInputs(card25=(wd / "fort.25").read_text(), fort10=wd / "fort.10", fort9=wd / "fort.9", mask=m)


def _read_card25(text: str) -> list[float]:
    line = (text.splitlines() or [""])[0].ljust(80)
    return [read_fw_d(line[10 * i:10 * i + 10], 5) for i in range(8)]


def _read_fort10(path: Path, nrhox: int):
    """READ(10) as in SPECTRV: edges, continuum frequencies, and per depth QCONTABS/QCONTSCAT (1131 each)."""
    recs = read_records(path)
    nedge = struct.unpack("<i", recs[1][:4])[0]
    e = np.frombuffer(recs[1], dtype="<f8", count=3 * nedge, offset=4).reshape(nedge, 3)
    ncon = struct.unpack("<i", recs[2][:4])[0]
    qa = np.empty((nrhox, 1131)); qs = np.empty((nrhox, 1131))
    for j in range(nrhox):
        qa[j] = np.frombuffer(recs[5 + 4 * j], dtype="<f8", count=1131)
        qs[j] = np.frombuffer(recs[6 + 4 * j], dtype="<f8", count=1131)
    ne1 = nedge - 1
    contabs = qa[:, :3 * ne1].reshape(nrhox, ne1, 3).transpose(1, 2, 0).copy()   # (IEDGE, k, J)
    contscat = qs[:, :3 * ne1].reshape(nrhox, ne1, 3).transpose(1, 2, 0).copy()
    return {"nedge": nedge, "frqedg": e[:, 0].copy(), "wledge": e[:, 1].copy(), "cmedge": e[:, 2].copy(),
            "ncon": ncon, "contabs": contabs, "contscat": contscat}


@dataclass
class Fort9:
    wlbeg: float
    resolu: float
    wlend: float
    length: int
    n: int
    linout: int
    turbv: np.float32
    ifvac: int
    nedge: int
    frqedg: np.ndarray
    wledge: np.ndarray
    cmedge: np.ndarray
    asynth: "Asynth"              # (LENGTH, 99) float32, read on demand
    nlines: int
    lines: list = field(default_factory=list)   # raw 596-byte line records


class Asynth:
    """ASYNTH(1..kw) records of fort.9 as a (LENGTH, kw) float32 array read on demand: indexing maps the file
    only for the duration of the read, so a 2M-point fort.9 (800 MB) is never resident as a whole."""

    def __init__(self, path, offset: int, length: int, stride: int):
        self.path, self.offset, self.length, self.stride = path, offset, length, stride
        self.shape = (length, KW)

    def __getitem__(self, key):
        raw = np.memmap(self.path, dtype=np.uint8, mode="r")
        try:
            a = np.ndarray((self.length, KW), dtype="<f4", buffer=raw, offset=self.offset + 4,
                           strides=(self.stride, 4))
            return np.array(a[key])
        finally:
            del raw

    def __array__(self, dtype=None, copy=None):
        v = self[:, :]
        return v if dtype is None else v.astype(dtype)


def read_fort9(path: Path) -> Fort9:
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    buf = memoryview(raw)

    def rec(off):
        (n,) = struct.unpack_from("<i", buf, off)
        if n < 0:
            raise NotImplementedError("subrecords in fort.9")
        (t,) = struct.unpack_from("<i", buf, off + 4 + n)
        assert t == n, "corrupt record marker in fort.9"
        return off + 4, n, off + 8 + n

    p, n, off = rec(0)
    wlbeg, resolu, wlend, length, nn, linout, turbv, ifvac = struct.unpack_from("<dddiiifi", buf, p)
    p, n, off = rec(off)
    (nedge,) = struct.unpack_from("<i", buf, p)
    e = np.frombuffer(buf, dtype="<f8", count=3 * nedge, offset=p + 4).reshape(nedge, 3)
    (n1,) = struct.unpack_from("<i", buf, off)
    if n1 < 4 * KW:
        raise ValueError("ASYNTH record shorter than kw REAL*4")
    stride = n1 + 8
    heads = np.ndarray((length,), dtype="<i4", buffer=raw, offset=off, strides=(stride,))
    tails = np.ndarray((length,), dtype="<i4", buffer=raw, offset=off + 4 + n1, strides=(stride,))
    for k0 in range(0, length, 1 << 18):         # check the record markers in blocks
        if not (np.all(heads[k0:k0 + (1 << 18)] == n1) and np.all(tails[k0:k0 + (1 << 18)] == n1)):
            raise ValueError("ASYNTH records are not of uniform length")
    asynth = Asynth(path, off, length, stride)
    off += length * stride
    p, n, off = rec(off)
    (nlines,) = struct.unpack_from("<i", buf, p)
    lines = []
    for _ in range(nlines):
        p, n, off = rec(off)
        lines.append(bytes(buf[p:p + n]))
    del buf, heads, tails, raw
    return Fort9(wlbeg, resolu, wlend, length, nn, linout, np.float32(turbv), ifvac, nedge,
                 e[:, 0].copy(), e[:, 1].copy(), e[:, 2].copy(), asynth, nlines, lines)


# ----------------------------------------------------------------------------------------------- numerics

def powi(x: float, m: torch.Tensor) -> torch.Tensor:
    """libgcc __powidf2(x, m) for m >= 0 (gfortran REAL*8**INTEGER), vectorised over m."""
    m = m.to(torch.int64)
    if bool((m < 0).any()):
        raise NotImplementedError("negative exponents")
    y = torch.where(m % 2 == 1, torch.full(m.shape, x, dtype=F64, device=m.device),
                    torch.ones(m.shape, dtype=F64, device=m.device))
    n = m >> 1
    xx = x
    while bool((n > 0).any()):
        xx = xx * xx
        y = torch.where(n % 2 == 1, y * xx, y)
        n = n >> 1
    return y


def powi_scalar(x: float, m: int) -> float:
    n = abs(m)
    y = x if n % 2 else 1.0
    n >>= 1
    while n:
        x = x * x
        if n % 2:
            y = y * x
        n >>= 1
    return 1.0 / y if m < 0 else y


def edge_index(wave: torch.Tensor, wledge: np.ndarray, nedge: int, start: int = 0) -> torch.Tensor:
    """0-based IEDGE after `2004 IF(WAVE.LT.WLEDGE(IEDGE+1))...IEDGE=IEDGE+1` for an increasing wave sequence
    starting at IEDGE=start+1. Exact for any WLEDGE (sequential scan over edges, not over points)."""
    wl = torch.as_tensor(np.asarray(wledge[:NEDGE_MAX], dtype=np.float64), device=wave.device)
    if nedge > 1 and bool((wl[1:nedge] > wl[:nedge - 1]).all()) and start == 0:
        return torch.searchsorted(wl[1:nedge].contiguous(), wave.contiguous(), right=True)
    wn = wl.cpu().numpy()
    w = wave.cpu().numpy()
    out = np.empty(w.shape, dtype=np.int64)
    i, k = start, 0
    while k < len(w):
        while not (w[k] < wn[i + 1]):
            i += 1
        k2 = max(k + int(np.searchsorted(w[k:], wn[i + 1], side="left")), k + 1)  # points below WLEDGE(i+1)
        out[k:k2] = i
        k = k2
    return torch.as_tensor(out, device=wave.device)


_GPOW = np.frompyfunc(math.pow, 2, 1)


def _GEXP1(x):
    return np.vectorize(math.exp, otypes=[np.float64])(x)


def _GPOW1(x, y):
    return np.vectorize(math.pow, otypes=[np.float64])(x, y)


def _exp(x: torch.Tensor, libm: str) -> torch.Tensor:
    """EXP over the layer axis (last): torch's vectorised exp, glibc's scalar exp (what gfortran -O0 calls) when
    libm='glibc', or as the stock gfortran -O3 loop (libmvec _ZGVbN2v_exp on layer pairs, scalar exp for an odd
    last layer) when libm='libmvec'."""
    if libm == "libmvec":
        return torch.from_numpy(dloopvec(dvexp, _GEXP1, x.detach().cpu().numpy())).to(x.device)
    return exp_glibc(x) if libm == "glibc" else torch.exp(x)


def _pow10(x: torch.Tensor, libm: str) -> torch.Tensor:
    """10.**X (REAL*8 ** REAL*8 -> pow). Loop 2006 is not vectorised by gfortran -O3 (scalar pow there too)."""
    if libm in ("glibc", "libmvec"):
        return torch.from_numpy(_GPOW(10.0, x.cpu().numpy()).astype(np.float64)).to(x.device)
    return torch.pow(10.0, x)


def _cont_opacity(wave, iedge, wl9, halfedge, deledge, contabs, contscat, libm="torch"):
    """ACONT, SIGMAC (B, NRHOX) at WAVE by the three-point edge interpolation in log10."""
    wl_i = wl9[iedge]; wl_i1 = wl9[iedge + 1]; he = halfedge[iedge]; de = deledge[iedge]
    c1 = (wave - he) * (wave - wl_i1) / de
    c2 = (wl_i - wave) * (wave - wl_i1) * 2.0 / de
    c3 = (wave - wl_i) * (wave - he) / de
    ca = contabs[iedge]; cs = contscat[iedge]                 # (B, 3, J)
    c1, c2, c3 = c1[:, None], c2[:, None], c3[:, None]
    acont = _pow10(c1 * ca[:, 0] + c2 * ca[:, 1] + c3 * ca[:, 2], libm)
    sigmac = _pow10(c1 * cs[:, 0] + c2 * cs[:, 1] + c3 * cs[:, 2], libm)
    return acont, sigmac


def _planck(wave, hkt, libm="torch"):
    freq = rdiv(2.99792458e17, wave)
    freq15 = cdiv(freq, 1.0e15)
    ehvkt = _exp(-freq[:, None] * hkt[None, :], libm)
    stim = 1.0 - ehvkt
    f3 = freq15 * freq15 * freq15
    bnu = 1.47439e-02 * f3[:, None] * ehvkt / stim
    return ehvkt, stim, bnu


# ----------------------------------------------------------------------------------------------- result

@dataclass
class SpectrvResult:
    teff: float
    glog: float
    title: bytes                 # 74 bytes
    wbegin: float
    deltaw: float
    numnu: int
    ifsurf: int
    nmu: int
    angle: np.ndarray            # (20,)
    nedge: int
    wledge: np.ndarray           # (377,)
    nu: np.ndarray               # written points, 1-based NU
    wave: np.ndarray
    q: np.ndarray                # (len(nu), 2*NMU): Q(MU)=RESID*SURF(MU), Q(MU+NMU)=SURF(MU)
    n910: int
    line_records: list = field(default_factory=list)
    fort17: str = ""
    stdout: str = ""

    @property
    def flux(self):
        return self.q[:, : self.q.shape[1] // 2]

    @property
    def cont(self):
        return self.q[:, self.q.shape[1] // 2:]

    def header_bytes(self) -> bytes:
        return (struct.pack("<dd", self.teff, self.glog) + self.title
                + struct.pack("<ddiii", self.wbegin, self.deltaw, self.numnu, self.ifsurf, self.nmu)
                + np.asarray(self.angle, dtype="<f8").tobytes() + struct.pack("<i", self.nedge)
                + np.asarray(self.wledge, dtype="<f8").tobytes())

    def to_bytes(self) -> bytes:
        import io
        f = io.BytesIO()
        write_record(f, self.header_bytes())
        n = self.q.shape[1] * 8
        body = np.empty((len(self.q), n + 8), dtype=np.uint8)
        body[:, :4] = np.frombuffer(struct.pack("<i", n), dtype=np.uint8)
        body[:, 4:4 + n] = np.ascontiguousarray(self.q, dtype="<f8").view(np.uint8).reshape(len(self.q), n)
        body[:, 4 + n:] = body[:, :4]
        f.write(body.tobytes())
        write_record(f, struct.pack("<i", self.n910))
        for r in self.line_records:
            write_record(f, r)
        return f.getvalue()

    def write(self, path: str | Path, chunk: int = 1 << 16) -> None:
        """fort.7 (= to_bytes()), written in blocks of `chunk` point records."""
        n = self.q.shape[1] * 8
        mark = np.frombuffer(struct.pack("<i", n), dtype=np.uint8)
        with open(path, "wb") as f:
            write_record(f, self.header_bytes())
            for k0 in range(0, len(self.q), chunk):
                q = self.q[k0:k0 + chunk]
                body = np.empty((len(q), n + 8), dtype=np.uint8)
                body[:, :4] = mark
                body[:, 4:4 + n] = np.ascontiguousarray(q, dtype="<f8").view(np.uint8).reshape(len(q), n)
                body[:, 4 + n:] = mark
                f.write(body.tobytes())
            write_record(f, struct.pack("<i", self.n910))
            for r in self.line_records:
                write_record(f, r)


# ----------------------------------------------------------------------------------------------- program

def _t(s: State, key: str) -> torch.Tensor:
    v = s[key]
    return v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v))


def spectrv(state: State, inp: SpectrvInputs, *, chunk: int = 8192, device=None,
            libm: str = "auto") -> SpectrvResult:
    """PROGRAM SPECTRV after CALL READIN(20): `state` holds the COMMONs READIN produced.

    libm: "torch" (torch exp/pow everywhere), "glibc" (glibc scalar exp/pow for every point, as the
    -O0 Fortran; ~35 us/point extra, CPU), "auto" (torch, then glibc for the optically thick points whose
    REAL*8 solver path is ulp-sensitive; bitwise = the gf-O0 Fortran), "libmvec" (as the stock gfortran -O3
    spectrv.exe: EXP(-FREQ*HKT(J)) of loop 20 and the DO 4 BFUDGE/FSCAT powers/exps through glibc libmvec's
    2-lane _ZGVbN2v_exp/_ZGVbN2vv_pow on layer pairs, scalar glibc otherwise; bitwise = stock, which differs
    from -O0 only in the deepest line cores, e.g. 9 points of Ca II H&K at <= 5e-15).
    chunk: points per batch (two JOSH rows per point).
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    s = state
    out = []                                                    # stdout lines after READIN
    rhoxj, r1, r101, ph1, pc1, psi1, prddop, prdpow = _read_card25(inp.card25)
    slope = 100.0 / (r101 - r1)
    origin = 1.5 - r1 * slope
    nrhox = s.iscalar("rhox.nrhox")
    rhox = _t(s, "rhox.rhox")[:nrhox].to(dev, F64)
    hkt = _t(s, "temp.hkt")[:nrhox].to(dev, F64)
    ifscat = s.iscalar("if.ifscat"); ifsurf = s.iscalar("if.ifsurf")
    nmu = s.iscalar("mus.nmu")
    angle20 = _t(s, "mus.angle").to(F64).cpu().numpy().copy()
    # IFCORO only selects CALL CORINT / LINCEN, which are empty stubs in spectrv.for

    # ---- fort.10
    t10 = _read_fort10(inp.fort10, nrhox)
    nedge10 = t10["nedge"]
    out.append(fmt_i(nedge10, 10)); out.append(fmt_i(t10["ncon"], 10))
    wledge = np.zeros(NEDGE_MAX); halfedge = np.zeros(NEDGE_MAX); deledge = np.zeros(NEDGE_MAX)
    wledge[:nedge10] = np.abs(t10["wledge"])
    halfedge[:nedge10 - 1] = (wledge[:nedge10 - 1] + wledge[1:nedge10]) * 0.5
    dw = wledge[1:nedge10] - wledge[:nedge10 - 1]
    deledge[:nedge10 - 1] = dw * dw * 0.5
    out.append(" " * 10 + "TAPE10")

    # ---- depth set-up (DO 4)
    bh = _t(s, "bhyd.bhyd")[:nrhox, 0].to(dev, F64)
    bc1 = _t(s, "bc.bc1")[:nrhox, 0].to(dev, F64); bc2 = _t(s, "bc.bc2")[:nrhox, 0].to(dev, F64)
    bs1 = _t(s, "bsi.bsi1")[:nrhox, 0].to(dev, F64); bs2 = _t(s, "bsi.bsi2")[:nrhox, 0].to(dev, F64)
    if libm == "libmvec":
        # DO 4 as gfortran -O3 vectorises it: libmvec pow/exp on layer pairs (exp on every lane, then masked),
        # scalar pow/exp for an odd last layer
        def bf(f, a, b, c):
            return (f(a, ph1) * f(b, pc1)) * f(c, psi1)
        bargs = [v.cpu().numpy() for v in (bh, bc1 / bc2, bs1 / bs2)]
        bfudge = torch.from_numpy(dloopvec(lambda a, b, c: bf(dvpow, a, b, c), lambda a, b, c: bf(_GPOW1, a, b, c),
                                           *bargs)).to(dev)
    else:
        bfudge = torch.pow(bh, ph1) * torch.pow(bc1 / bc2, pc1) * torch.pow(bs1 / bs2, psi1)
    fscat = torch.zeros(nrhox, dtype=F64, device=dev)
    if rhoxj != 0.0:
        r = cdiv(rhox, rhoxj)
        e = _exp(-r, libm) if libm == "libmvec" else torch.exp(-r)
        fscat = torch.where(r < 100.0, e, torch.zeros_like(r))
    out.append(w_format("FSCAT ", fscat.cpu(), nrhox).rstrip("\n"))
    out.append(w_format("BFUDGE", bfudge.cpu(), nrhox).rstrip("\n"))

    # ---- fort.9 header
    f9 = read_fort9(inp.fort9)
    wledge[:f9.nedge] = f9.wledge
    title = bytearray(np.asarray(s["junk.title"]).astype("S1").tobytes().ljust(74)[:74])
    title[0:1] = b"V" if f9.ifvac == 1 else b"A"
    out.append(" WLBEG=" + fmt_f(f9.wlbeg, 10, 4) + "   RESOLU=" + fmt_f(f9.resolu, 10, 1) + "   WLEND="
               + fmt_f(f9.wlend, 10, 4) + "   LENGTH=" + fmt_i(f9.length, 7) + "   NRHOX=" + fmt_i(f9.n, 2))
    ratio = 1.0 + 1.0 / f9.resolu
    ratiolg = math.log(ratio)
    ixwl = int(math.log(f9.wlbeg) / ratiolg)
    wbegin = math.exp(ixwl * ratiolg)
    if wbegin < f9.wlbeg:
        ixwl += 1
        wbegin = math.exp(ixwl * ratiolg)
    deltaw = f9.resolu
    nulo, nuhi = 1, f9.length
    numnu = f9.length
    wlend = wbegin * powi_scalar(ratio, nuhi - 1)
    out.append(fmt_i(numnu, 10) + fmt_f(deltaw, 13, 4) + fmt_f(wbegin, 13, 4) + fmt_f(wlend, 13, 4))
    if ifsurf == 1:
        out.append(" SURFACE FLUX")
    if ifsurf == 2:
        out.append(" SURFACE INTENSITY" + fmt_i(nmu, 3))
        out.append("".join(fmt_f(a, 6, 3) for a in angle20[:nmu]))

    # ---- mask.bin (BasicATLAS patch)
    nbytes = nuhi - nulo + 1
    if inp.mask is not None:
        if inp.mask.size < nbytes:
            raise EOFError(f"mask.bin has {inp.mask.size} bytes, SPECTRV reads {nbytes}")
        keep = inp.mask[:nbytes] != 0
    else:
        keep = np.ones(nbytes, dtype=bool)
    keep = keep.copy(); keep[0] = True                          # IF(.NOT.KEEP(NU).AND.NU.GT.1)CYCLE
    nu_all = np.arange(1, nbytes + 1)
    nu_keep = nu_all[keep]

    # ---- per-point solve (DO 25), chunked and batched
    wl9 = torch.as_tensor(wledge, device=dev)
    he_t = torch.as_tensor(halfedge, device=dev); de_t = torch.as_tensor(deledge, device=dev)
    ca_t = torch.as_tensor(t10["contabs"], device=dev); cs_t = torch.as_tensor(t10["contscat"], device=dev)
    angle_mu = torch.as_tensor(angle20[:nmu], device=dev)
    nu_t = torch.as_tensor(nu_keep, device=dev)
    wave_all = wbegin * powi(ratio, nu_t - 1)
    iedge_all = edge_index(wave_all, wledge, f9.nedge)
    q = np.empty((len(nu_keep), 2 * nmu))
    nplot = min(abs(f9.linout), nbytes)
    resid_plot = []                                             # RESID(MU=1) for the stdout plot
    def solve(wave, iedge, asy, lm):
        """Both JOSH calls of DO 25 for a batch of points -> SURF(MU), the line+continuum HNU/SURFI, thick."""
        B = wave.shape[0]
        acont, sigmac = _cont_opacity(wave, iedge, wl9, he_t, de_t, ca_t, cs_t, lm)
        ehvkt, stim, bnu = _planck(wave, hkt, lm)
        zero = torch.zeros_like(acont)
        aline = asy * (1.0 - fscat)
        sline = bnu * stim / (bfudge - ehvkt)
        sigmal = asy * fscat
        key = "hnu1" if ifsurf == 1 else "surfi"
        thick = torch.zeros(2 * B, dtype=torch.bool, device=dev)
        if ifscat == 1:
            r = josh_batch(torch.cat([acont, acont]), torch.cat([bnu, bnu]), torch.cat([zero, aline]),
                           torch.cat([bnu, sline]), torch.cat([sigmac, sigmac]), torch.cat([zero, sigmal]),
                           rhox, 1, ifsurf, angle=angle_mu)
            rc, rl = r[key][:B], r[key][B:]
            if "rows_thick" in r:
                thick[r["rows_thick"]] = True
        else:
            c = josh_batch(acont, bnu, zero, bnu, sigmac, zero, rhox, ifscat, ifsurf, angle=angle_mu)
            r = josh_batch(acont, bnu, aline, sline, sigmac, sigmal, rhox, 1, ifsurf, angle=angle_mu)
            rc, rl = c[key], r[key]
            if "rows_thick" in r:
                thick[B + r["rows_thick"]] = True
        if ifsurf == 1:
            rc, rl = rc[:, None].expand(B, nmu), rl[:, None].expand(B, nmu)
        return rc, rl, thick[:B] | thick[B:]

    for c0 in range(0, len(nu_keep), chunk):
        sl = slice(c0, c0 + chunk)
        wave = wave_all[sl]; iedge = iedge_all[sl]
        asy = torch.as_tensor(np.asarray(f9.asynth[nu_keep[sl] - 1, :nrhox]), device=dev).to(F64)
        surf, line, thick = solve(wave, iedge, asy, libm if libm in ("glibc", "libmvec") else "torch")
        if libm == "auto" and bool(thick.any()):
            # optically thick points go through JOSH's REAL*8 DERIV iteration, which (unlike the REAL*4
            # solver) carries input ulps to the output: recompute their opacities with glibc exp/pow
            ix = thick.nonzero().squeeze(1)
            s2, l2, _ = solve(wave[ix], iedge[ix], asy[ix], "glibc")
            surf, line = surf.clone(), line.clone()
            surf[ix] = s2; line[ix] = l2
        resid = line / surf
        q[sl, :nmu] = (resid * surf).cpu().numpy()
        q[sl, nmu:] = surf.cpu().numpy()
        nk = int((nu_t[sl] <= nplot).sum())
        if nk:
            resid_plot.extend(resid[:nk, 0].cpu().tolist())

    # stdout of the point loop: FORMAT(1H1) at NU=1, the residual plot for NU <= |LINOUT|, MU=1
    out.append("1")
    wave_np = wave_all.cpu().numpy()
    for k, res1 in enumerate(resid_plot):
        iresid = int(res1 * 1000.0 + 0.5)
        iplot = max(1, min(101, int(res1 * slope + origin)))
        aplot = [" "] * 101; aplot[iplot - 1] = "X"
        out.append(" " + fmt_i(nu_keep[k], 5) + fmt_f(wave_np[k], 11, 4) + fmt_i(iresid, 7) + "".join(aplot))

    # ---- NLINES block
    if nmu > 10:
        ifsurf_l = 1
    else:
        ifsurf_l = ifsurf
    n10 = 0
    nlines = f9.nlines
    n910 = nlines + n10
    out.append(fmt_i(nlines, 10) + fmt_i(n10, 10) + fmt_i(n910, 10))
    line_records, fort17 = [], ""
    if nlines:
        line_records, fort17 = _lines_block(s, f9, nrhox, rhox, hkt, fscat, bfudge, wl9, he_t, de_t, ca_t,
                                            cs_t, ifsurf_l, angle_mu[:1], dev, chunk)
    hdr_card = "".join(fmt_f(v, 10, 5) for v in (rhoxj, r1, r101, ph1, pc1, psi1, prddop, prdpow))
    return SpectrvResult(
        teff=s.scalar("teff.teff"), glog=s.scalar("teff.glog"), title=bytes(title), wbegin=wbegin,
        deltaw=deltaw, numnu=numnu, ifsurf=ifsurf, nmu=nmu, angle=angle20, nedge=f9.nedge,
        wledge=wledge.copy(), nu=nu_keep, wave=wave_np, q=q, n910=n910, line_records=line_records,
        fort17=fort17, stdout=hdr_card + "\n" + "\n".join(out) + "\n")


# ----------------------------------------------------------------------------------------------- lines (LINOUT > 0)

_LIN = np.dtype([("wl", "<f8"), ("e", "<f8"), ("ep", "<f8"), ("label", "S10"), ("labelp", "S10"),
                 ("other1", "S10"), ("other2", "S10"), ("wlvac", "<f8"), ("center", "<f8"), ("concen", "<f8"),
                 ("nelion", "<i4"), ("gammar", "<f4"), ("gammas", "<f4"), ("gammaw", "<f4"), ("ref", "S4"),
                 ("nblo", "<i4"), ("nbup", "<i4"), ("iso1", "<i4"), ("x1", "<f4"), ("iso2", "<i4"), ("x2", "<f4"),
                 ("gflog", "<f4"), ("xj", "<f4"), ("xjp", "<f4"), ("code", "<f4"), ("elo", "<f4"), ("gf", "<f4"),
                 ("gs", "<f4"), ("gr", "<f4"), ("gw", "<f4"), ("dwl", "<f4"), ("dgflog", "<f4"),
                 ("dgammar", "<f4"), ("dgammas", "<f4"), ("dgammaw", "<f4"), ("extra1", "<f4"),
                 ("extra2", "<f4"), ("extra3", "<f4")])
assert _LIN.itemsize == 200


def _fort17_line(r) -> str:
    f = lambda k: float(r[k])
    a = lambda k: r[k].decode("latin-1").ljust(10)[:10] if k != "ref" else r[k].decode("latin-1").ljust(4)[:4]
    l1 = (fmt_f(f("wl"), 10, 4) + fmt_f(f("gflog"), 7, 3) + fmt_f(f("xj"), 5, 1) + fmt_f(f("e"), 12, 3)
          + fmt_f(f("xjp"), 5, 1) + fmt_f(f("ep"), 12, 3) + fmt_f(f("code"), 9, 2) + a("label") + a("labelp"))
    l2 = (fmt_f(f("wl"), 10, 4) + fmt_i(r["nelion"], 4) + fmt_f(f("gr"), 6, 2) + fmt_f(f("gs"), 6, 2)
          + fmt_f(f("gw"), 6, 2) + a("ref") + fmt_i(r["nblo"], 2) + fmt_i(r["nbup"], 2) + fmt_i(r["iso1"], 3)
          + fmt_f(f("x1"), 7, 3) + fmt_i(r["iso2"], 3) + fmt_f(f("x2"), 7, 3) + a("other1") + a("other2")
          + fmt_f(f("dwl"), 7, 4) + fmt_f(f("dgflog"), 7, 3) + fmt_f(f("dgammar"), 6, 2)
          + fmt_f(f("dgammas"), 6, 2) + fmt_f(f("dgammaw"), 6, 2))
    if f("wl") < 0:
        l2 += fmt_f(f("wl"), 10, 4)
    return l1 + "\n" + l2 + "\n" if f("wl") != 0 else ""


def _lines_block(s, f9, nrhox, rhox, hkt, fscat, bfudge, wl9, he_t, de_t, ca_t, cs_t, ifsurf, angle1, dev,
                 chunk):
    recs = np.frombuffer(b"".join(r[:200] for r in f9.lines), dtype=_LIN).copy()
    alinec = np.stack([np.frombuffer(r, dtype="<f4", count=KW, offset=200) for r in f9.lines])
    wave = torch.as_tensor(recs["wlvac"].astype(np.float64), device=dev)
    # IEDGE: reset to 1 when WAVE < WAVOLD, then the forward scan
    iedge = torch.empty(len(recs), dtype=torch.int64, device=dev)
    w = recs["wlvac"]
    seg = [0] + [k for k in range(1, len(w)) if w[k] < w[k - 1]] + [len(w)]
    for a0, a1 in zip(seg[:-1], seg[1:]):
        iedge[a0:a1] = edge_index(wave[a0:a1], wl9.cpu().numpy(), f9.nedge)   # restart at IEDGE=1
    g = lambda k: (_t(s, k)[:nrhox].to(dev, F64) if k in s else torch.zeros(nrhox, dtype=F64, device=dev))
    ahline, shline = g("bhyd.ahline"), g("bhyd.shline")
    axline, sxline = g("ops.axline"), g("ops.sxline")
    center = np.empty(len(recs)); concen = np.empty(len(recs))
    for c0 in range(0, len(recs), chunk):
        sl = slice(c0, c0 + chunk)
        wv = wave[sl]; B = wv.shape[0]
        acont, sigmac = _cont_opacity(wv, iedge[sl], wl9, he_t, de_t, ca_t, cs_t)
        ehvkt, stim, bnu = _planck(wv, hkt)
        zero = torch.zeros_like(acont)
        alc = torch.as_tensor(alinec[sl, :nrhox], device=dev).to(F64)
        aline = ahline + alc * (1.0 - fscat) + axline
        slinec = bnu * stim / (bfudge - ehvkt)
        sline = torch.where(aline > 0.0, (ahline * shline + alc * slinec * (1.0 - fscat) + axline * sxline) / aline,
                            bnu)
        sigmal = alc * fscat
        r = josh_batch(torch.cat([acont, acont]), torch.cat([bnu, bnu]), torch.cat([zero, aline]),
                       torch.cat([bnu, sline]), torch.cat([sigmac, sigmac]), torch.cat([zero, sigmal]),
                       rhox, 1, ifsurf, angle=angle1)
        v = r["hnu1"] if ifsurf == 1 else r["surfi"][:, 0]
        concen[sl] = v[:B].cpu().numpy(); center[sl] = v[B:].cpu().numpy()
    recs["center"] = center; recs["concen"] = concen
    out_recs = [recs[i:i + 1].tobytes() for i in range(len(recs))]
    fort17 = "".join(_fort17_line(recs[i]) for i in range(len(recs)))
    return out_recs, fort17


# ----------------------------------------------------------------------------------------------- driver

def run_spectrv(workdir: str | Path, state: State | None = None, *, deck: str | Path | None = None,
                readin=None, out: str | Path | None = None, **kw) -> SpectrvResult:
    """File-level SPECTRV: inputs from `workdir` (fort.25, fort.10, fort.9, mask.bin), output fort.7.

    `state` is the State after READIN(20). If omitted, `readin(deck)` is called (default: the port in
    atlas_torch.synthe.readin, which must provide `readin(deck_path) -> State`).
    """
    wd = Path(workdir)
    if state is None:
        if readin is None:
            from . import readin as _rd                      # provided by the READIN port
            readin = _rd.readin
        state = readin(deck)
    res = spectrv(state, load_inputs(wd), **kw)
    res.write(Path(out) if out is not None else wd / "fort.7")
    if res.fort17:
        (wd / "fort.17").write_text(res.fort17)
    return res
