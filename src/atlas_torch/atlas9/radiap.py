"""RADIAP (atlas9mem.for): radiation pressure / energy density integrals.

MODE 1 zeroes the sums (H is a SAVEd local: "save.radiap.h"); MODE 2 contributions are accumulated by the
caller in Fortran order (see main.accumulate); MODE 3 normalises and integrates PRAD.
Quirks: 12.5664 is a REAL*4 literal; KNU(1) of the last JOSH call is used for every contribution of a
frequency, including the continuum remainder after the opacity-sampling loop stops early.
"""
from __future__ import annotations

import torch

from ..fnum import f32
from ..state import State
from .numerics import integ

C4PI = f32(12.5664)


def radiap_terms(jnu, hnu, abtot, knu1, rcowt):
    """MODE 2 increments: RADEN += JNU*RCOWT, H += HNU*RCOWT, ACCRAD += ABTOT*HNU*RCOWT, PRADK0 += KNU(1)*RCOWT."""
    r = rcowt.unsqueeze(-1)
    return {"raden": jnu * r, "h": hnu * r, "accrad": abtot * hnu * r, "pradk0": knu1 * rcowt}


def radiap3(s: State) -> dict:
    n = s.iscalar("rhox.nrhox")
    flux = s.scalar("flux.flux")
    h = s["save.radiap.h"][:n]
    raden = s["pzero.raden"].clone(); accrad = s["rad.accrad"].clone()
    raden[:n] = raden[:n] * C4PI / 2.997925e10
    acc = accrad[:n] * C4PI / 2.997925e10
    acc = torch.where(h / flux > 1.0, acc * flux / h, acc)
    accrad[:n] = acc
    pradk0 = s.scalar("pzero.pradk0") * C4PI / 2.997925e10
    if float(h[0]) / flux > 1.0:
        pradk0 = pradk0 * flux / float(h[0])
    rx = s["rhox.rhox"][:n]
    prad = s["rad.prad"].clone(); prad[:n] = integ(rx, acc, acc[0] * rx[0])
    pradk = s["pzero.pradk"].clone(); pradk[:n] = prad[:n] + pradk0
    return {"pzero.raden": raden, "rad.accrad": accrad, "pzero.pradk0": torch.tensor([pradk0]), "rad.prad": prad,
            "pzero.pradk": pradk}
