"""I/O of the xnfdf program: Fortran unformatted sequential files (ifx: 4-byte little-endian record
markers) fort.10 = xnfpdf.dat and fort.11 = xnfpdfmax.dat, and readers used for comparisons.

Record layout per model (written by PROGRAM XNFDF for every READIN/BEGIN):
  fort.10: (NT int4, TEFF r8, GLOG r8, TITLE 74*r8)
           (IN int4, (FRQEDG,WLEDGE,CMEDGE)(IN) r8, IDMOL(60) r8, MOMASS(60) r8)
           (NUMNU int4, FREQSET(NUMNU) r8)
           (T4,TKEV4,TK4,HKT4,TLOG4,HCKT4,P4,XNE4,XNATOM4,RHO4,RHOX4,VTURB4,XNFH4 (25) r4,
            XNFHE4(25,2) r4, XNFH24(25) r4)
           NT x (XNFPEL4(6,99) r4, DOPPLE4(6,99) r4, CONTINALL4(1131) r4)
  fort.11: the same first three records (without IDMOL/MOMASS) then XNFDOPMAX(377,6,99) r4.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MARK = "<i4"


class FortranWriter:
    """Unformatted sequential writer with 4-byte record markers."""

    def __init__(self, path):
        self.fh = open(path, "wb")

    def record(self, *items):
        parts = []
        for it in items:
            if isinstance(it, (bytes, bytearray)):
                parts.append(bytes(it))
            else:
                a = np.asarray(it)
                parts.append(np.ascontiguousarray(a.reshape(-1, order="F") if a.ndim > 1 else a).tobytes())
        body = b"".join(parts)
        n = struct.pack("<i", len(body))
        self.fh.write(n + body + n)

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_records(path) -> list[bytes]:
    b = Path(path).read_bytes()
    out, off = [], 0
    while off < len(b):
        (n,) = struct.unpack_from("<i", b, off)
        out.append(b[off + 4: off + 4 + n])
        (n2,) = struct.unpack_from("<i", b, off + 4 + n)
        if n2 != n:
            raise ValueError(f"bad record marker at {off}")
        off += 8 + n
    return out


@dataclass
class XnfModel:
    nt: int
    teff: float
    glog: float
    title: bytes
    nedge: int
    edges: np.ndarray          # (IN, 3) FRQEDG, WLEDGE, CMEDGE
    freqset: np.ndarray        # (NUMNU,)
    idmol: np.ndarray | None = None
    momass: np.ndarray | None = None
    state: dict = field(default_factory=dict)   # r4 arrays (T4 .. XNFH24)
    xnfpel: np.ndarray | None = None   # (NT, 99, 6) as Fortran (6,99) transposed? kept Fortran: (NT, 6, 99)
    dopple: np.ndarray | None = None   # (NT, 6, 99)
    continall: np.ndarray | None = None  # (NT, 1131)
    xnfdopmax: np.ndarray | None = None  # (377, 6, 99)


STATE_NAMES = ("t", "tkev", "tk", "hkt", "tlog", "hckt", "p", "xne", "xnatom", "rho", "rhox", "vturb", "xnfh")


def _head(recs, i):
    r = recs[i]
    nt, teff, glog = struct.unpack_from("<idd", r, 0)
    return nt, teff, glog, r[20:]


def read_xnfpdf(path) -> list[XnfModel]:
    recs = read_records(path)
    models, i = [], 0
    while i < len(recs):
        nt, teff, glog, title = _head(recs, i)
        r = recs[i + 1]
        (nin,) = struct.unpack_from("<i", r, 0)
        e = np.frombuffer(r, "<f8", nin * 3, 4).reshape(nin, 3)
        idm = np.frombuffer(r, "<f8", 60, 4 + nin * 24)
        mom = np.frombuffer(r, "<f8", 60, 4 + nin * 24 + 480)
        r = recs[i + 2]
        (nnu,) = struct.unpack_from("<i", r, 0)
        fs = np.frombuffer(r, "<f8", nnu, 4)
        st = np.frombuffer(recs[i + 3], "<f4")
        state = {}
        for k, nm in enumerate(STATE_NAMES):
            state[nm] = st[k * 25:(k + 1) * 25]
        state["xnfhe"] = st[13 * 25:15 * 25].reshape(25, 2, order="F")
        state["xnfh2"] = st[15 * 25:16 * 25]
        lay = [np.frombuffer(recs[i + 4 + j], "<f4") for j in range(nt)]
        xp = np.stack([l[:594].reshape(6, 99, order="F") for l in lay])
        dp = np.stack([l[594:1188].reshape(6, 99, order="F") for l in lay])
        ca = np.stack([l[1188:] for l in lay])
        models.append(XnfModel(nt, teff, glog, title, nin, e, fs, idm, mom, state, xp, dp, ca))
        i += 4 + nt
    return models


def read_xnfpdfmax(path) -> list[XnfModel]:
    recs = read_records(path)
    models = []
    for i in range(0, len(recs), 4):
        nt, teff, glog, title = _head(recs, i)
        r = recs[i + 1]
        (nin,) = struct.unpack_from("<i", r, 0)
        e = np.frombuffer(r, "<f8", nin * 3, 4).reshape(nin, 3)
        r = recs[i + 2]
        (nnu,) = struct.unpack_from("<i", r, 0)
        fs = np.frombuffer(r, "<f8", nnu, 4)
        mx = np.frombuffer(recs[i + 3], "<f4").reshape(377, 6, 99, order="F")
        models.append(XnfModel(nt, teff, glog, title, nin, e, fs, xnfdopmax=mx))
    return models


def _relerr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    both0 = (a == b)
    d = np.abs(a - b) / np.maximum(np.abs(a), np.abs(b))
    d[both0] = 0.0
    return d


def compare_files(pdf_a, pdf_b, max_a=None, max_b=None) -> dict[str, dict]:
    """Per-array statistics between two xnfdf outputs: max relative error, number of differing r4
    values, max ulp distance (r4 arrays)."""
    out = {}

    def acc(name, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        d = _relerr(x, y)
        st = out.setdefault(name, {"n": 0, "ndiff": 0, "max_rel": 0.0, "max_ulp": 0, "where": None})
        st["n"] += x.size
        nd = int(np.count_nonzero(x != y))
        st["ndiff"] += nd
        if nd:
            k = int(np.argmax(d))
            if d.reshape(-1)[k] > st["max_rel"]:
                st["max_rel"] = float(d.reshape(-1)[k])
                st["where"] = (float(x.reshape(-1)[k]), float(y.reshape(-1)[k]))
            if x.dtype == np.float32:
                xi = x.view(np.int32).astype(np.int64)
                yi = y.view(np.int32).astype(np.int64)
                xi = np.where(xi < 0, -(xi & 0x7FFFFFFF), xi)
                yi = np.where(yi < 0, -(yi & 0x7FFFFFFF), yi)
                st["max_ulp"] = max(st["max_ulp"], int(np.max(np.abs(xi - yi))))
    A = read_xnfpdf(pdf_a)
    B = read_xnfpdf(pdf_b)
    assert len(A) == len(B), (len(A), len(B))
    for a, b in zip(A, B):
        acc("header", np.array([a.nt, a.teff, a.glog]), np.array([b.nt, b.teff, b.glog]))
        acc("title", np.frombuffer(a.title, np.uint8), np.frombuffer(b.title, np.uint8))
        acc("edges", a.edges, b.edges)
        acc("freqset", a.freqset, b.freqset)
        acc("idmol+momass", np.r_[a.idmol, a.momass], np.r_[b.idmol, b.momass])
        for k in a.state:
            acc("state." + k, a.state[k], b.state[k])
        acc("xnfpel4", a.xnfpel, b.xnfpel)
        acc("dopple4", a.dopple, b.dopple)
        acc("continall4", a.continall, b.continall)
    if max_a is not None:
        A = read_xnfpdfmax(max_a)
        B = read_xnfpdfmax(max_b)
        assert len(A) == len(B)
        for a, b in zip(A, B):
            acc("max:edges", a.edges, b.edges)
            acc("max:freqset", a.freqset, b.freqset)
            acc("xnfdopmax", a.xnfdopmax, b.xnfdopmax)
    return out


def format_report(stats: dict) -> str:
    lines = [f"{'array':18s} {'n':>10s} {'ndiff':>9s} {'max_rel':>10s} {'max_ulp':>8s}  where(a,b)"]
    for k, s in stats.items():
        w = "" if s["where"] is None else f"{s['where'][0]:.6e} {s['where'][1]:.6e}"
        lines.append(f"{k:18s} {s['n']:10d} {s['ndiff']:9d} {s['max_rel']:10.3e} {s['max_ulp']:8d}  {w}")
    return "\n".join(lines)
