"""xnfpelsyn.dat (fort.10 of the SYNTHE chain, written by xnfpelsyn.for): model state, continuous opacities at
the continuum edges and n/U, Doppler widths per species slot, per layer."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...io.fortran import read_records

KW = 99
MW = 139


@dataclass(frozen=True)
class XnfpelsynData:
    """wl_edge (nedge,) nm (|edge|, sorted); log_cont_total/abs/scat (L, 3 (nedge-1)) log10 kappa per gram at
    each edge, the interval midpoint and the next edge (xnfpelsyn.for loop 129: ABTOT = ACONT + SIGMAC);
    state: per-layer arrays T, P, XNE, XNATOM, RHO, RHOX, VTURB, XNFH, XNFHE (L, 2), XNFH2 (number densities);
    xnfpel, dopple: (L, 6, 139) n/U (not divided by rho) and Doppler width / c of each XNFPEL slot
    (v2.io.synthe_layout.xnfpel layout); idmol, momass: molecule code and mass of slots 40..139."""
    wl_edge: np.ndarray
    log_cont_total: np.ndarray
    log_cont_abs: np.ndarray
    log_cont_scat: np.ndarray
    state: dict
    xnfpel: np.ndarray
    dopple: np.ndarray
    idmol: np.ndarray
    momass: np.ndarray
    teff: float
    glog: float


def read_xnfpelsyn_dat(path: str | Path) -> XnfpelsynData:
    recs = read_records(path)
    nrhox = struct.unpack_from("<i", recs[0], 0)[0]
    teff, glog = struct.unpack_from("<2d", recs[0], 4)
    nedge = struct.unpack_from("<i", recs[1], 0)[0]
    e = np.frombuffer(recs[1], dtype="<f8", count=3 * nedge, offset=4).reshape(nedge, 3)
    idmol = np.frombuffer(recs[1], dtype="<f8", count=100, offset=4 + 24 * nedge).copy()
    momass = np.frombuffer(recs[1], dtype="<f8", count=100, offset=4 + 24 * nedge + 800).copy()
    q = np.frombuffer(recs[3], dtype="<f8")
    names = ("T", "TKEV", "TK", "HKT", "TLOG", "HCKT", "P", "XNE", "XNATOM", "RHO", "RHOX", "VTURB", "XNFH")
    st = {n: q[i * KW:i * KW + nrhox].copy() for i, n in enumerate(names)}
    o = len(names) * KW
    st["XNFHE"] = q[o:o + 2 * KW].reshape(2, KW).T[:nrhox].copy()
    st["XNFH2"] = q[o + 2 * KW:o + 2 * KW + nrhox].copy()
    lay = recs[4:]
    n3 = 3 * (nedge - 1)
    tot = np.stack([np.frombuffer(lay[4 * j], dtype="<f8")[:n3] for j in range(nrhox)])
    ab = np.stack([np.frombuffer(lay[4 * j + 1], dtype="<f8")[:n3] for j in range(nrhox)])
    sc = np.stack([np.frombuffer(lay[4 * j + 2], dtype="<f8")[:n3] for j in range(nrhox)])
    xd = np.stack([np.frombuffer(lay[4 * j + 3], dtype="<f8") for j in range(nrhox)])
    xnfpel = xd[:, :6 * MW].reshape(nrhox, MW, 6).transpose(0, 2, 1).copy()
    dopple = xd[:, 6 * MW:12 * MW].reshape(nrhox, MW, 6).transpose(0, 2, 1).copy()
    return XnfpelsynData(np.abs(e[:, 1]), tot, ab, sc, st, xnfpel, dopple, idmol, momass, teff, glog)
