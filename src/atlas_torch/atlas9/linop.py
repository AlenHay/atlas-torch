"""LINOP (atlas9mem.for, the in-memory big-ODF version) and the ODF reader for unit 9.

The ODF (p00big*.bdf) has, for each of the 328 BIG wavelength intervals and each of the 57 log T, one record
of INTEGER*2 KAP(ip=1..25, istep=1..12) (1000 * log10 kappa). LINOP interpolates bilinearly in (log T, log P)
-- coefficients CO1..CO4 are REAL*4 including the factor TENLOG*.001 -- and evaluates the REAL*4 expression
EXP(CO1*K11+CO2*K21+CO3*K12+CO4*K22) (glibc expf). Sub-step N uses ODF step 13-N with weight WT(N)=n/60
(REAL*4 constants 1./60., 2./60., 3./60., 6./60. widened).

Quirks reproduced:
  * coefficients (IT, IP, CO*, FSCAT) are recomputed only when ITEMP changes, the interval search IWL only
    moves forward from its saved value (reset to 1 with the coefficients): `interval_index` replays that.
  * WAVE < WLBEG(1): LINOP returns without setting NSTEPS/STEPWT/ALINES (main keeps KAPP's NSTEPS=1,
    STEPWT=1 and ALINES=0).
  * The T/P clamps (TL in [3.30, 5.3], log P in [-4, 8]) use DMIN1/DMAX1 with the REAL*4 table values.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ..fnum import f32
from ..io.formats import read_bdf
from . import data, fmath

WT = np.array([np.float32(k) / np.float32(60.0) for k in (1, 2, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6)],
              dtype=np.float32).astype(np.float64)
TENLOG = 2.30258509299405
_P001 = f32(0.001)


@lru_cache(maxsize=4)
def read_odf(path: str) -> torch.Tensor:
    """KAPWL as int16 tensor [iwl=328, it=57, ip=25, istep=12] (index 0 = Fortran 1)."""
    a = read_bdf(path)
    assert a.shape[1:] == (57, 25, 12), a.shape
    return torch.from_numpy(np.ascontiguousarray(a[:328]))


def wl_bounds() -> tuple[np.ndarray, np.ndarray]:
    """WLBEG(IWL)=WAVEDF(IWL,2), WLEND(IWL)=WAVEDF(IWL+1,2), WLEND(328)=1e6 (nm)."""
    w = data.wavedf()[:, 1]
    beg = w[:328].copy()
    end = w[1:329].copy()
    end[327] = 1000000.0
    return beg, end


def interval_index(waves: list[float], iwl0: int = 1) -> list[int | None]:
    """IWL (1-based) for each successive LINOP call wavelength within one ITEMP (forward-only search from the
    saved IWL); None where WAVE < WLBEG(1) (early return)."""
    beg, end = wl_bounds()
    out = []
    iwl = iwl0
    for w in waves:
        if w < beg[0]:
            out.append(None)
            continue
        while not (w < end[iwl - 1]):
            iwl += 1
        out.append(iwl)
    return out


def coefficients(tlog: torch.Tensor, p: torch.Tensor, tauros: torch.Tensor, tauscat: float):
    """Per-layer (IT, IP) (1-based upper indices), REAL*4 CO1..CO4 and FSCAT (LINOP labels 7..19)."""
    t = data.linop_tables()
    tabt = torch.from_numpy(t["tabt"]).to(tlog.device)
    tabp = torch.from_numpy(t["tabp"]).to(tlog.device)
    tenlog = float(t["tenlog"])
    tl = torch.minimum(torch.maximum(tlog / tenlog, torch.tensor(3.30, dtype=tlog.dtype)), torch.tensor(5.3, dtype=tlog.dtype))
    # the DO loop starts at IT=2 and stops at the first TL < TABT(IT); otherwise IT=57
    it = _first_less(tl, tabt, 2, 57)
    plog = torch.minimum(torch.maximum(fmath.log10(p), torch.tensor(-4.0, dtype=p.dtype)), torch.tensor(8.0, dtype=p.dtype))
    ip = _first_less(plog, tabp, 2, 25)
    x = (tl - tabt[it - 2]) / (tabt[it - 1] - tabt[it - 2])
    y = (plog - tabp[ip - 2]) / (tabp[ip - 1] - tabp[ip - 2])
    co1 = ((1.0 - x) * (1.0 - y) * tenlog * _P001).to(torch.float32)
    co2 = ((1.0 - x) * y * tenlog * _P001).to(torch.float32)
    co3 = (x * (1.0 - y) * tenlog * _P001).to(torch.float32)
    co4 = (x * y * tenlog * _P001).to(torch.float32)
    fscat = torch.zeros_like(tlog)
    if tauscat > 0.0:
        fscat = fmath.exp(-tauros / tauscat)
    return it, ip, (co1, co2, co3, co4), fscat


def _first_less(v: torch.Tensor, tab: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """DO I=lo,hi: IF(V.LT.TAB(I)) exit; I=hi if none -> 1-based I."""
    lt = v.unsqueeze(-1) < tab[lo - 1:hi].unsqueeze(0)
    any_ = lt.any(-1)
    first = torch.argmax(lt.to(torch.int8), dim=-1) + lo
    return torch.where(any_, first, torch.full_like(first, hi))


def line_opacity(kapwl: torch.Tensor, iwl: torch.Tensor, it, ip, co, fscat):
    """ALINES, SIGLIN for all 12 steps of the given intervals: iwl [F] (1-based) -> [F, 12, n] float64 each.

    Step N uses ODF column N12=13-N. REAL*4 arithmetic: ((CO1*K11+CO2*K21)+CO3*K12)+CO4*K22, then expf."""
    co1, co2, co3, co4 = co
    k = kapwl[iwl - 1]                                  # [F, it, ip, istep]
    itm, ipm = it - 2, ip - 2                           # 0-based IT-1, IP-1
    def g(a, b):
        return k[:, a, b, :].to(torch.float32)          # [F, n, 12]
    k11, k21, k12, k22 = g(itm, ipm), g(itm, ipm + 1), g(itm + 1, ipm), g(itm + 1, ipm + 1)
    s = co1[None, :, None] * k11
    s = s + co2[None, :, None] * k21
    s = s + co3[None, :, None] * k12
    s = s + co4[None, :, None] * k22
    a = fmath.expf(s).to(torch.float64)                 # [F, n, istep]
    a = a.flip(-1).permute(0, 2, 1)                     # step N <- istep 13-N ; [F, 12(N), n]
    siglin = fscat[None, None, :] * a
    alines = (1.0 - fscat)[None, None, :] * a
    return alines, siglin
