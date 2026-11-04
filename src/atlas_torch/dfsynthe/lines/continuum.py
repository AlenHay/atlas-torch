"""Per-layer set-up of DFSYNTHE MAIN (labels 186-221): continuum on the 500000-resolution grid, XNFPEL/RHO,
XNFDOP, TXNXN.  Vectorised over layers (dimension 0) and wavelength bins (chunked)."""
from __future__ import annotations

import numpy as np
import torch

from .f32 import F32, F64, Flavor, lit
from .tables import CUTOFF, LENGTH, RATIO, WBEGIN

_WAVE_CACHE: dict = {}


def wave_grid(fl: Flavor) -> np.ndarray:
    """WAVE(NBUFF) = WBEGIN*RATIO**(NBUFF-1), NBUFF=1..LENGTH (REAL*8 ** INTEGER as compiled)."""
    if fl.name not in _WAVE_CACHE:
        _WAVE_CACHE[fl.name] = WBEGIN * fl.powi8_np(RATIO, np.arange(LENGTH))
    return _WAVE_CACHE[fl.name]


def edge_coefficients(wledge: np.ndarray, ablog: torch.Tensor, fastmath: bool = False):
    """AEDGE/BEDGE/CEDGE [NL, NEDGE-1] (REAL*8) from ABLOG [NL, 377, 3] (REAL*4)."""
    w = torch.as_tensor(np.abs(wledge), dtype=F64, device=ablog.device)
    nedge = w.numel()
    half = (w[:-1] + w[1:]) * 0.5
    dl = w[1:] - w[:-1]
    dele = (dl * dl) * 0.5
    a8 = ablog[:, : nedge - 1, :].to(F64)
    a1, a2, a3 = a8[..., 0], a8[..., 1], a8[..., 2]
    w0, w1 = w[:-1], w[1:]
    if fastmath:
        # stock (MAIN loop 2002): one reciprocal R=1./DELEDGE and the regrouped sums
        #   A=((a1-2a2)+a3)*R, B=((w0+w1)*(2a2)-((h+w1)*a1+(h+w0)*a3))*R, C=((w1*a1+w0*a3)*h-(w1*w0)*(2a2))*R
        r = torch.div(torch.ones_like(dele), dele)
        a22 = a2 + a2
        A = ((a1 - a22) + a3) * r
        B = ((w0 + w1) * a22 - ((half + w1) * a1 + (half + w0) * a3)) * r
        C = ((w1 * a1 + w0 * a3) * half - (w1 * w0) * a22) * r
    else:
        A = ((a1 - 2.0 * a2) + a3) / dele
        B = ((-((half + w1) * a1) + ((w0 + w1) * 2.0) * a2) - (w0 + half) * a3) / dele
        C = (((half * w1) * a1 - ((w0 * w1) * 2.0) * a2) + (w0 * half) * a3) / dele
    ok = (dele != 0)[None, :]
    z = torch.zeros_like(A)
    return torch.where(ok, A, z), torch.where(ok, B, z), torch.where(ok, C, z)


def continuum(wledge: np.ndarray, ablog: torch.Tensor, fl: Flavor, chunk: int = 400_000) -> torch.Tensor:
    """CONTINUUM [NL, LENGTH] float32 = CUTOFF*10.**((A*WAVE+B)*WAVE+C)."""
    dev = ablog.device
    wave = wave_grid(fl)
    w = np.abs(wledge)
    if np.any(np.diff(w) < 0):
        raise ValueError("edges not sorted")
    iedge = np.maximum.accumulate(np.searchsorted(w[1:], wave, side="right"))   # 0-based IEDGE
    if iedge.max() >= len(w) - 1:
        raise ValueError("wavelength beyond last edge")
    A, B, C = edge_coefficients(wledge, ablog, getattr(fl, "fastmath", False))
    nl = ablog.shape[0]
    out = torch.empty((nl, LENGTH), dtype=F32, device=dev)
    cut = float(CUTOFF)
    for s in range(0, LENGTH, chunk):
        e = min(LENGTH, s + chunk)
        wv = torch.from_numpy(wave[s:e]).to(dev)
        ie = torch.from_numpy(iedge[s:e]).to(dev)
        x = (A[:, ie] * wv + B[:, ie]) * wv + C[:, ie]
        out[:, s:e] = (cut * fl.pow(torch.full_like(x, 10.0), x)).to(F32)
    return out


def layer_setup(xt, fl: Flavor, dev) -> dict:
    """XNFPEL/RHO, XNFDOP, DOPPLE, TXNXN, XNE, XNFPH, ... for all layers (REAL*4)."""
    st = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in xt.state.items()}
    xnfpel_raw = torch.from_numpy(xt.xnfpel).to(dev)
    dopple = torch.from_numpy(xt.dopple).to(dev)
    rho = st["rho"][:, None]
    if getattr(fl, "fastmath", False):     # stock (label 205): XNFPEL*(1./RHO(J)) (divss, mulps)
        xnfpel = xnfpel_raw * (torch.ones_like(rho) / rho)
    else:
        xnfpel = xnfpel_raw / rho
    if getattr(fl, "fastmath", False):     # stock: vectorised XNFPEL/DOPPLE = rcpps + Newton step (rcpps.py)
        from .rcpps import div_nr
        xnfdop = div_nr(xnfpel, dopple)
    else:
        xnfdop = xnfpel / dopple
    t = st["t"]
    txnxn = ((st["xnfh"] + float(lit(".42")) * st["xnfhe"][:, 0]) + float(lit(".85")) * st["xnfh2"]) * \
        fl.powf(t / 10000.0, torch.tensor(float(lit(".3")), dtype=F32))
    return dict(st=st, xnfpel=xnfpel, xnfdop=xnfdop, dopple=dopple, txnxn=txnxn, xne=st["xne"],
                xnfph=xnfpel_raw[:, 0:2].clone(), xnfphe=xnfpel_raw[:, 6:9].clone(),
                ablog=torch.from_numpy(xt.ablog).to(dev))
