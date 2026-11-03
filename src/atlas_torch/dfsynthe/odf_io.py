"""File formats of the DFSYNTHE chain (dfsynthe.for / dfsortp.for / separatedf.for, ifx conventions).

* xnfpdf.dat (unit 10, written by XNFDF, sequential unformatted): per temperature 3 header records
  (NRHOX,TEFF,GLOG,TITLE | NEDGE,(FREQEDGE,WLEDGE,CMEDGE)*NEDGE,IDMOL,MOMASS | NCON,CONFRQ), one state record
  (T,TKEV,TK,HKT,TLOG,HCKT,P,XNE,XNATOM,RHO,RHOX,VTURB,XNFH,XNFHE(kw,2),XNFH2, all REAL*4(25)) and NRHOX layer
  records (XNFPEL(594),DOPPLE(594),ABLOG(3,377), REAL*4).
* xnfpdfmax.dat (unit 22): per temperature (NT,TEFF,GLOG,...), (NEDGE,edges), (NUMNU,FREQSET), XNFDOPMAX(377,594).
* line lists *df.bin: RECORDTYPE='FIXED', RECL=8000 words -> flat stream of 16-byte entries
  (INTEGER*4 IWL=NBUFF, INTEGER*2 NELION,IELO,IGFLOG,IGR,IGS,IGW); the DO-loop counts of SELECTLINES say how
  many entries are real (the tail of the last record is padding).
* nltelinesdf.bin: fixed 64-byte records WL(R8),ELO(R4),GNNNGGG(11 x I4/R4) (+8 bytes padding).
* fort.15/16/17/18/20 (ODF steps of one temperature): fixed 32-byte records, IDFOUT(7) INTEGER*4 where
  IDFOUT(2:7) hold ISTEPS(12) INTEGER*2, then 4 zero bytes.  Order: layer J outer, IDF=1..1540 inner.
* dfsortp output (formatted): '(1x,i10,1x,12i8)' per (nu, ip), nu outer.
* separatedf output p00big/p00lit (.bdf): sequential unformatted, one record of 25*12 INTEGER*2 per (inu, it).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.fortran import iter_records

KW = 25
NFREQ = 1540
NBIG = 328

LINE_DTYPE = np.dtype([("iwl", "<i4"), ("nelion", "<i2"), ("ielo", "<i2"), ("igflog", "<i2"),
                       ("igr", "<i2"), ("igs", "<i2"), ("igw", "<i2")])
NLTE_DTYPE = np.dtype([("wl", "<f8"), ("elo", "<f4"), ("gf", "<f4"), ("nblo", "<i4"), ("nbup", "<i4"),
                       ("nelion", "<i4"), ("type", "<i4"), ("ncon", "<i4"), ("nelionx", "<i4"),
                       ("gammar", "<f4"), ("gammas", "<f4"), ("gammaw", "<f4"), ("nbuff", "<i4"),
                       ("pad", "<i4", 2)])
ODF_RECORD_DTYPE = np.dtype([("idfout", "<i4"), ("isteps", "<i2", 12), ("pad", "<i4")])

# (unit, file name, DO-loop count in SELECTLINES)
LINE_FILES = {
    "low": (11, "lowlinesdf.bin", 31108567),
    "hi": (21, "highlinesdf.bin", 10057574),
    "diat": (31, "diatomicsdf.bin", 7771848),
    "tio": (41, "tiolinesdf.bin", 36979284),
    "h2o": (43, "h2olinesdf.bin", 48999843),
}
NLTE_FILE = "nltelinesdf.bin"
VTURBS = (0, 1, 2, 4, 8)
VT_UNITS = {0: 15, 1: 16, 2: 17, 4: 18, 8: 20}

DFTS = ("1995.", "2089.", "2188.", "2291.", "2399.", "2512.", "2630.", "2754.", "2884.", "3020.", "3162.",
        "3311.", "3467.", "3631.", "3802.", "3981.", "4169.", "4365.", "4571.", "4786.", "5012.", "5370.",
        "5754.", "6166.", "6607.", "7079.", "7586.", "8128.", "8710.", "9333.", "10000.", "11220.", "12589.",
        "14125.", "15849.", "17783.", "19953.", "22387.", "25119.", "28184.", "31623.", "35481.", "39811.",
        "44668.", "50119.", "56234.", "63096.", "70795.", "79433.", "89125.", "100000.", "112202.", "125893.",
        "141254.", "158489.", "177828.", "199526.")


def open_linelist(path: str | Path, count: int | None = None) -> np.memmap:
    """Memory-map a repacked line list (no data read until sliced)."""
    m = np.memmap(path, dtype=LINE_DTYPE, mode="r")
    return m if count is None else m[:count]


def read_nlte(path: str | Path) -> np.ndarray:
    return np.fromfile(path, dtype=NLTE_DTYPE)


# ------------------------------------------------------------------------------------------ xnfpdf

@dataclass
class XnfpdfTemperature:
    nrhox: int
    teff: float
    glog: float
    nedge: int
    freqedge: np.ndarray   # f8 [nedge]
    wledge: np.ndarray     # f8 [nedge] (signed as stored)
    cmedge: np.ndarray
    state: dict            # name -> f4 [25] (xnfhe -> [25,2])
    xnfpel: np.ndarray     # f4 [nrhox, 594]
    dopple: np.ndarray     # f4 [nrhox, 594]
    ablog: np.ndarray      # f4 [nrhox, 377, 3]  (ABLOG(3,377) Fortran -> [377,3])


_STATE_NAMES = ("t", "tkev", "tk", "hkt", "tlog", "hckt", "p", "xne", "xnatom", "rho", "rhox", "vturb",
                "xnfh", "xnfhe", "xnfh2")


def _records(path):
    with open(path, "rb") as f:
        yield from iter_records(f)


def read_xnfpdf(path: str | Path, iteff: int) -> XnfpdfTemperature:
    """Block of temperature `iteff` (1-based) of xnfpdf.dat."""
    nrhox = struct.unpack("<i", next(_records(path))[:4])[0]
    per = 3 + 1 + nrhox
    it = _records(path)
    for _ in range((iteff - 1) * per):
        next(it)
    r0, r1, _r2, rs = next(it), next(it), next(it), next(it)
    nrhox, teff, glog = struct.unpack("<idd", r0[:20])
    nedge = struct.unpack("<i", r1[:4])[0]
    e = np.frombuffer(r1, dtype="<f8", count=3 * nedge, offset=4).reshape(nedge, 3)
    s = np.frombuffer(rs, dtype="<f4")
    st, k = {}, 0
    for nm in _STATE_NAMES:
        n = KW * (2 if nm == "xnfhe" else 1)
        a = s[k:k + n]
        st[nm] = a.reshape(2, KW).T.copy() if nm == "xnfhe" else a.copy()
        k += n
    xnfpel, dopple, ablog = [], [], []
    for _ in range(nrhox):
        r = np.frombuffer(next(it), dtype="<f4")
        xnfpel.append(r[:594]); dopple.append(r[594:1188]); ablog.append(r[1188:].reshape(377, 3))
    return XnfpdfTemperature(nrhox, teff, glog, nedge, e[:, 0].copy(), e[:, 1].copy(), e[:, 2].copy(), st,
                             np.stack(xnfpel), np.stack(dopple), np.stack(ablog))


@dataclass
class XnfpdfmaxTemperature:
    nt: int
    teff: float
    glog: float
    nedge: int
    wledge: np.ndarray
    xnfdopmax: np.ndarray  # f4 [377, 594] (Fortran XNFDOPMAX(377,594): [nu, nelion])


def read_xnfpdfmax(path: str | Path, iteff: int) -> XnfpdfmaxTemperature:
    it = _records(path)
    for _ in range((iteff - 1) * 4):
        next(it)
    r0, r1, _r2, r3 = next(it), next(it), next(it), next(it)
    nt, teff, glog = struct.unpack("<idd", r0[:20])
    nedge = struct.unpack("<i", r1[:4])[0]
    e = np.frombuffer(r1, dtype="<f8", count=3 * nedge, offset=4).reshape(nedge, 3)
    x = np.frombuffer(r3, dtype="<f4").reshape(594, 377).T.copy()
    return XnfpdfmaxTemperature(nt, teff, glog, nedge, e[:, 1].copy(), x)


# ------------------------------------------------------------------------------------------ ODF records

def odf_records(idfout: np.ndarray, isteps: np.ndarray) -> np.ndarray:
    rec = np.zeros(len(idfout), dtype=ODF_RECORD_DTYPE)
    rec["idfout"] = idfout
    rec["isteps"] = isteps
    return rec


def write_odf_bin(path: str | Path, idfout: np.ndarray, isteps: np.ndarray) -> None:
    odf_records(idfout, isteps).tofile(path)


def read_odf_bin(path: str | Path) -> np.ndarray:
    return np.fromfile(path, dtype=ODF_RECORD_DTYPE)


# ------------------------------------------------------------------------------------------ sortp ascii

def format_sortp(idfout: np.ndarray, isteps: np.ndarray) -> bytes:
    """'(1x,i10,1x,12i8)' lines; idfout [N], isteps [N,12]."""
    rows = np.concatenate([idfout.reshape(-1, 1).astype(np.int64), isteps.astype(np.int64)], axis=1)
    fmt = " %10d " + "%8d" * 12 + "\n"
    return "".join(fmt % tuple(r) for r in rows.tolist()).encode()


def parse_sortp(data: bytes | str | Path) -> tuple[np.ndarray, np.ndarray]:
    """List-directed read of a sortp file -> (idfout [N], isteps [N,12] int16)."""
    if isinstance(data, (str, Path)) and Path(data).exists():
        data = Path(data).read_bytes()
    a = np.array(bytes(data).split(), dtype=np.int64).reshape(-1, 13)
    return a[:, 0], a[:, 1:].astype(np.int16)


# ------------------------------------------------------------------------------------------ .bdf

def write_seq_records(path: str | Path, payloads: np.ndarray) -> None:
    """Sequential unformatted file of equal-length records: payloads [nrec, nbytes] uint8."""
    payloads = np.ascontiguousarray(payloads)
    n = payloads.shape[1]
    out = np.empty((payloads.shape[0], n + 8), dtype=np.uint8)
    m = np.frombuffer(struct.pack("<i", n), dtype=np.uint8)
    out[:, :4] = m
    out[:, 4:4 + n] = payloads
    out[:, 4 + n:] = m
    out.tofile(path)
