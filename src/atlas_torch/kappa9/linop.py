"""KAPPA9's LINOP: line opacity from an in-memory ODF (BIG frequency set), batched over layers and frequencies.

Fortran (kappa9.for LINOP, the "reads a big DF file into memory" version):
* On the first call the whole ODF (unit 9) is read: 328 wavelength intervals x 57 temperatures records of
  ((KAP(ip,it,istep), istep=1..12), ip=1..25) INTEGER*2 (= 1000 * log10 kappa); WLBEG/WLEND(IWL) are the
  BIG boundaries WAVEDF(IWL,2), WAVEDF(IWL+1,2) with WLEND(328) = 1.E6 (REAL*8).
* Whenever ITEMP changes, per layer: TL = clip(TLOG/TENLOG, 3.30, 5.3), PLOG = clip(log10 P, -4, 8), first
  grid index with TL < TABT(IT) (IT = 2..57, else 57), same for P on LINOP's own 25-point TABP; bilinear
  weights CO1..CO4 = (1-X)(1-Y) TENLOG .001 etc. stored as REAL*4; FSCAT = exp(-TAUROS/TAUSCAT) if TAUSCAT > 0;
  the step weights WT(1..12) = 1./60., 2./60., 3./60., 6./60. ... are REAL*4 quotients; IWL restarts at 1.
* Each call: WAVE = 2.997925D17/FREQ; return without lines (NSTEPS = 1, STEPWT = 1) if WAVE < WLBEG(1);
  advance IWL while WAVE >= WLEND(IWL) (IWL never decreases while ITEMP is unchanged); NSTEPS = 12,
  STEPWT = WT(N), N12 = 13 - N and
      A = EXP(CO1*KAP(IP-1,IT-1) + CO2*KAP(IP,IT-1) + CO3*KAP(IP-1,IT) + CO4*KAP(IP,IT))     (all REAL*4)
      SIGLIN = FSCAT*A,  ALINES = (1-FSCAT)*A.
  REAL*4 * INTEGER*2 is evaluated in single precision, the sum left to right in single precision, EXP in
  single precision (here: exp of the float32 argument in double, rounded to float32 = correctly rounded expf).
"""
from __future__ import annotations

import numpy as np
import torch

from ..fnum import f32
from ..io.formats import read_bdf
from ..tables import table

SRC = "kappa9.for"
TENLOG = 2.30258509299405


def glibc_expf(x: torch.Tensor) -> torch.Tensor:
    """The platform libm's expf element by element (what the -O0 oracle library calls; slow, for tests)."""
    import ctypes
    import ctypes.util
    f = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6").expf
    f.restype, f.argtypes = ctypes.c_float, [ctypes.c_float]
    u = np.frompyfunc(f, 1, 1)
    r = np.asarray(u(x.detach().cpu().numpy()), dtype=np.float32)
    return torch.from_numpy(r.reshape(tuple(x.shape))).to(x.device)


def odf_kap(path) -> torch.Tensor:
    """ODF file -> int16 tensor [328 (IWL), 57 (IT), 25 (IP), 12 (ISTEP)]."""
    return torch.from_numpy(np.ascontiguousarray(read_bdf(path, nt=57)))


def wl_bounds(wavedf: np.ndarray, nwl: int = 328) -> tuple[np.ndarray, np.ndarray]:
    wlbeg = np.asarray(wavedf)[:nwl, 1].astype(np.float64).copy()
    wlend = np.asarray(wavedf)[1:nwl + 1, 1].astype(np.float64).copy()
    wlend[nwl - 1] = 1000000.0
    return wlbeg, wlend


def step_weights() -> torch.Tensor:
    """WT(1..12) (REAL*8 array assigned single-precision quotients)."""
    num = [1, 2, 3] + [6] * 9
    return torch.tensor([f32(np.float32(n) / np.float32(60.0)) for n in num], dtype=torch.float64)


def setup(tlog: torch.Tensor, p: torch.Tensor, tauros: torch.Tensor | None = None, tauscat: float = 0.0) -> dict:
    """Per-layer interpolation setup (the ITEMP-change branch). Layers on dim 0 (any batch shape)."""
    tabt = torch.as_tensor(table(SRC, "LINOP", "TABT"), dtype=torch.float64, device=tlog.device)
    tabp = torch.as_tensor(table(SRC, "LINOP", "TABP"), dtype=torch.float64, device=tlog.device)
    tl = torch.clamp(torch.clamp(tlog / TENLOG, min=3.30), max=5.3)       # DMIN1(DMAX1(.,3.30D0),5.3D0)
    plog = torch.clamp(torch.clamp(torch.log10(p), min=-4.0), max=8.0)
    # first IT in 2..57 with TL < TABT(IT), else 57 (Fortran index)
    it = 2 + (tl.unsqueeze(-1) >= tabt[1:]).to(torch.int64).cumprod(-1).sum(-1)
    it = torch.clamp(it, max=57)
    ip = 2 + (plog.unsqueeze(-1) >= tabp[1:]).to(torch.int64).cumprod(-1).sum(-1)
    ip = torch.clamp(ip, max=25)
    x = (tl - tabt[it - 2]) / (tabt[it - 1] - tabt[it - 2])
    y = (plog - tabp[ip - 2]) / (tabp[ip - 1] - tabp[ip - 2])
    c = f32(0.001)
    co1 = ((1.0 - x) * (1.0 - y) * TENLOG * c).to(torch.float32)
    co2 = ((1.0 - x) * y * TENLOG * c).to(torch.float32)
    co3 = (x * (1.0 - y) * TENLOG * c).to(torch.float32)
    co4 = (x * y * TENLOG * c).to(torch.float32)
    fscat = torch.zeros_like(tlog)
    if tauscat > 0.0:
        if tauros is None:
            raise ValueError("TAUSCAT > 0 needs TAUROS")
        fscat = torch.exp(-tauros / tauscat)
    return {"it": it, "ip": ip, "tl": tl, "plog": plog, "x": x, "y": y,
            "co1": co1, "co2": co2, "co3": co3, "co4": co4, "fscat": fscat}


def iwl_sequence(freqs, wlbeg: np.ndarray, wlend: np.ndarray) -> list[int]:
    """Fortran IWL (1-based) for a sequence of LINOP calls at frequencies `freqs` with unchanged ITEMP
    (IWL starts at 1, only advances); 0 marks WAVE < WLBEG(1) (no line opacity, NSTEPS = 1)."""
    out, iwl = [], 1
    for fr in freqs:
        wave = 2.997925e17 / float(fr)
        if wave < wlbeg[0]:
            out.append(0)
            continue
        while not wave < wlend[iwl - 1]:
            iwl += 1
        out.append(iwl)
    return out


def line_opacity(kap: torch.Tensor, iwl: torch.Tensor, co: dict, expf=None, return_arg: bool = False):
    """A (float32) for frequencies with IWL (1-based tensor [F]) and all 12 steps N=1..12 (dim 1),
    layers from `co` (setup) on the last dim: returns [F, 12, J]. Frequencies with iwl == 0 give A = 0.
    expf: single-precision EXP implementation (float32 tensor -> float32 tensor); default: correctly rounded
    (exp of the float32 argument in double, rounded once). return_arg: also return the float32 exponent."""
    it0 = co["it"] - 1          # 0-based IT
    ip0 = co["ip"] - 1
    valid = iwl > 0
    w = (torch.clamp(iwl, min=1) - 1).view(-1, 1, 1)
    n12 = torch.arange(11, -1, -1, device=kap.device).view(1, 12, 1)     # N12 = 13-N -> 0-based 12-N
    def k(itx, ipx):
        return kap[w, itx.view(1, 1, -1), ipx.view(1, 1, -1), n12].to(torch.float32)
    s = co["co1"] * k(it0 - 1, ip0 - 1)
    s = s + co["co2"] * k(it0 - 1, ip0)
    s = s + co["co3"] * k(it0, ip0 - 1)
    s = s + co["co4"] * k(it0, ip0)
    a = expf(s) if expf is not None else torch.exp(s.to(torch.float64)).to(torch.float32)
    a = torch.where(valid.view(-1, 1, 1), a, torch.zeros((), dtype=torch.float32))
    return (a, s) if return_arg else a
