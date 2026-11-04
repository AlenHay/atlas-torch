"""PUTOUT (atlas9mem.for as patched by BasicATLAS).

MODE 1/2/3/4 bookkeeping per frequency (surface flux sums) is done by the main loop; this module writes:
  * MODE 4 lines for IFPRNT=2 (unit 6, per frequency) and IFPNCH>=2 (FLUX cards on unit 7);
  * MODE 5: the FLUX line, the patched iteration table between ' START TABLE' / ' END TABLE' (columns
    RHOX, TEMP, PRESSURE, XNE, RHO, ABROSS, HEIGHT, TAUROS, FLXCNV, PRADK, FLXERR, FLXDRV), the punched model
    (RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB,FLXRAD,VCONV,VELSND + PRADK + BEGIN card + '=========='), and the patch's
    stops: 'INVALID STRUCTURE' (NaN/negative checks, before the table), 'REACHED GOLD TOLERANCES'
    (ITER multiple of 15, max|FLXERR|<1 and max|FLXDRV|<10) and 'MODEL DIVERGED' (max|FLXDRV|>1e5).

Quirks reproduced: AMAXERR/AMAXDER are only recomputed when IFPRNT(ITER)>0 (SAVEd otherwise, as with
-fno-automatic); FLXRAD=FLUX-FLXCNV is set only when IFCORR=0 (and only inside the printed branch); the
INVALID STRUCTURE test is also only done when printing; FLXCNVratio is computed but not printed.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..state import State
from . import fformat as ff
from .errors import FortranExit

HEAD1 = ("0" + " " * 36 + "ELECTRON" + " " * 13 + "ROSSELAND" + " " * 4 + "HEIGHT" + " " * 3 + "ROSSELAND"
         + " " * 13 + "RADIATION" + " " * 8 + "PER CENT FLUX").ljust(132)
HEAD2 = (" " * 8 + "RHOX" + " " * 6 + "TEMP" + " " * 4 + "PRESSURE" + " " * 4 + "NUMBER" + " " * 4 + "DENSITY"
         + " " * 6 + "MEAN" + " " * 7 + "(KM)" + " " * 6 + "DEPTH" + " " * 4 + "CONV FLUX  PRESSURE" + " " * 8
         + "ERROR" + " " * 4 + "DERIV").ljust(132)


def _str(s: State, key: str) -> str:
    a = np.asarray(s[key]).reshape(-1)
    return "".join((b.decode("latin-1") if b else "\0") for b in a) if a.dtype.itemsize == 1 else \
        a[0].decode("latin-1").ljust(a.dtype.itemsize, "\0")


def mode5(s: State, stdout: list, punch: list) -> dict:
    """PUTOUT(5). Appends to stdout/punch; returns State updates; raises FortranExit for the patch's STOPs."""
    it = s.iscalar("iter.iter")
    n = s.iscalar("rhox.nrhox")
    ifprnt = int(s["iter.ifprnt"][it - 1])
    ifpnch = int(s["iter.ifpnch"][it - 1])
    out = {}
    g = lambda k: s[k][:n].tolist()
    amaxerr = s.scalar("save.putout.amaxerr") if "save.putout.amaxerr" in s else 0.0
    amaxder = s.scalar("save.putout.amaxder") if "save.putout.amaxder" in s else 0.0
    flux = s.scalar("flux.flux")
    if ifprnt != 0:
        stdout.append("0" + " " * 108 + "FLUX" + ff.E(flux, 12, 4, 1))
        t, ferr, fdrv, rx, rho, p = g("temp.t"), g("flux.flxerr"), g("flux.flxdrv"), g("rhox.rhox"), \
            g("state.rho"), g("state.p")
        for j in range(n):
            nan = lambda v: v != v
            if (nan(t[j]) or nan(ferr[j]) or nan(fdrv[j]) or nan(rx[j]) or ferr[j] > 1e100 or fdrv[j] > 1e100
                    or t[j] < 0 or rho[j] < 0 or rx[j] < 0 or p[j] < 0 or nan(p[j]) or nan(rho[j])):
                stdout.append(" INVALID STRUCTURE")
                raise FortranExit("INVALID STRUCTURE", 0)
        stdout.append(" START TABLE")
        title = _str(s, "junk.title")
        stdout.append(" TEFF" + ff.F(s.scalar("teff.teff"), 8, 0) + "   LOG G" + ff.F(s.scalar("teff.glog"), 9, 5)
                      + ff.X(10) + title + ff.X(2) + "ITERATION" + ff.I(it, 3))
        if s.iscalar("if.ifcorr") == 0:
            fr = s["flux.flxrad"].clone()
            fr[:n] = flux - s["conv.flxcnv"][:n]
            out["flux.flxrad"] = fr
        stdout.append(HEAD1)
        stdout.append(HEAD2)
        xne, ab, h, tr, fc, pk = g("state.xne"), g("abross.abross"), g("height.height"), g("abross.tauros"), \
            g("conv.flxcnv"), g("pzero.pradk")
        for j in range(n):
            stdout.append(ff.I(j + 1, 4) + " " + ff.E(rx[j], 17, 5, 1) + " " + ff.E(t[j], 17, 5, 0) + " "
                          + "".join(ff.E(v, 17, 5, 1) for v in (p[j], xne[j], rho[j], ab[j], h[j], tr[j], fc[j],
                                                                 pk[j]))
                          + " " + ff.E(ferr[j], 15, 3, 0) + " " + ff.E(fdrv[j], 15, 3, 0))
        stdout.append(" END TABLE")
        amaxerr = 0.0
        amaxder = 0.0
        for j in range(n):
            if abs(ferr[j]) > amaxerr:
                amaxerr = abs(ferr[j])
            if abs(fdrv[j]) > amaxder:
                amaxder = abs(fdrv[j])
    out["save.putout.amaxerr"] = torch.tensor([amaxerr])
    out["save.putout.amaxder"] = torch.tensor([amaxder])
    if ifpnch == 0:
        return out
    s2 = State(s); s2.update(out)
    punch.extend(punch_model(s2))
    if it % 15 == 0 and amaxder < 10 and amaxerr < 1:
        stdout.append(" REACHED GOLD TOLERANCES")
        raise FortranExit("REACHED GOLD TOLERANCES", 0)
    if it % 15 == 0 and amaxder > 100000:
        stdout.append(" MODEL DIVERGED")
        raise FortranExit("MODEL DIVERGED", 0)
    return out


def punch_model(s: State) -> list[str]:
    n = s.iscalar("rhox.nrhox")
    it = s.iscalar("iter.iter")
    L = []
    a = "ON " if s.iscalar("conv.ifconv") == 1 else "OFF"
    b = "ON " if s.iscalar("turbpr.ifturb") == 1 else "OFF"
    L.append("TEFF " + ff.F(s.scalar("teff.teff"), 7, 0) + "  GRAVITY" + ff.F(s.scalar("teff.glog"), 8, 5) + " "
             + _str(s, "junk.wlte"))
    L.append("TITLE " + _str(s, "junk.title"))
    L.append(" OPACITY IFOP" + "".join(ff.I(int(v), 2) for v in s["ifop.ifop"]))
    L.append(" CONVECTION " + a + ff.F(s.scalar("conv.mixlth"), 6, 2) + " TURBULENCE " + b
             + "".join(ff.F(s.scalar(f"turbpr.{k}"), 6, 2) for k in ("trbfdg", "trbpow", "trbsnd", "trbcon")))
    ab = s["elem.abund"].tolist()
    L.append("ABUNDANCE SCALE " + ff.F(s.scalar("junk.xscale"), 9, 5) + " ABUNDANCE CHANGE"
             + ff.I(1, 2) + ff.F(ab[0], 8, 5) + ff.I(2, 2) + ff.F(ab[1], 8, 5))
    for i0 in range(2, 99, 6):
        L.append(" ABUNDANCE CHANGE" + "".join(ff.I(iz + 1, 3) + ff.F(ab[iz], 7, 2) for iz in range(i0, min(i0 + 6, 99))))
    L.append("READ DECK6" + ff.I(n, 3) + " RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB," + " " * 9 + "FLXRAD,VCONV,VELSND")
    cols = [s[k][:n].tolist() for k in ("rhox.rhox", "temp.t", "state.p", "state.xne", "abross.abross", "rad.accrad",
                                         "turbpr.vturb", "flux.flxrad", "conv.vconv", "conv.velsnd")]
    for j in range(n):
        L.append(ff.E(cols[0][j], 15, 8, 1) + ff.F(cols[1][j], 9, 1) + "".join(ff.E(c[j], 10, 3, 1) for c in cols[2:]))
    L.append("PRADK" + ff.E(s.scalar("pzero.pradk0"), 11, 4, 1))
    if s.iscalar("depart.nlteon") != 0:
        L.append("READ DEPARTURE COEFFICIENTS" + ff.I(n, 3) + " RHOX  BHYD 1-6  BMIN")
        bh, bm, rx = s["depart.bhyd"], s["depart.bmin"], s["rhox.rhox"]
        for j in range(n):
            L.append(ff.E(float(rx[j]), 11, 4, 1) + "".join(ff.F(float(bh[j, i]), 9, 4) for i in range(6))
                     + ff.F(float(bm[j]), 9, 4))
    L.append("BEGIN" + " " * 20 + "ITERATION " + ff.I(it, 3) + " COMPLETED")
    L.append("==========")
    return L


def mode4_lines(nu: int, freq: float, hsurf: float, contin: float, nsteps: int, taunu_c: torch.Tensor,
                ifprnt: int, ifpnch: int, nuhi: int, ifsurf: int):
    """PUTOUT(4) output for one frequency: (stdout lines, punch lines). HSURF/CONTIN as summed by MODE 2/3."""
    so, pu = [], []
    wave = 2.997925e17 / freq
    if nsteps == 1:
        contin = hsurf
    resid = None
    if contin > 0.0:
        resid = hsurf / contin
    if hsurf <= 0.0:
        hsurf = 1.0e-50
    if ifprnt > 1:
        tn = taunu_c.tolist()
        jtau1 = len(tn) + 1
        for j, v in enumerate(tn):
            if v > 1.0:
                jtau1 = j + 1
                break
        tauend = math.log10(tn[-1])
        if nsteps > 1:
            jtau1, tauend = 0, 0.0
        if ifsurf in (0, 1):
            hlam = hsurf * freq / wave
            hnulg = math.log10(hsurf)
            hlamlg = math.log10(hlam)
            so.append(" ")
            so.append(ff.I(nu, 6) + ff.F(wave, 10, 3) + ff.E(hlam, 13, 4, 1) + ff.F(hlamlg, 12, 5)
                      + ff.F(-2.5 * hlamlg, 10, 3) + ff.E(freq, 20, 6, 1) + ff.E(hsurf, 13, 4, 1)
                      + ff.F(hnulg, 12, 5) + ff.F(-2.5 * hnulg, 10, 3) + ff.F(resid if resid is not None else 0.0, 9, 5)
                      + ff.I(jtau1, 6) + ff.F(tauend, 6, 2) + ff.I(nu, 5))
    if ifpnch >= 2 and ifsurf <= 1:
        pu.append("FLUX" + ff.I(nu, 5) + ff.F(wave, 9, 2) + ff.E(freq, 20, 6, 1) + ff.E(hsurf, 13, 4, 1)
                  + ff.E(contin, 13, 4, 1) + ff.F(resid if resid is not None else 0.0, 10, 5))
        if nu == nuhi:
            pu.append("FLUX")
    return so, pu
