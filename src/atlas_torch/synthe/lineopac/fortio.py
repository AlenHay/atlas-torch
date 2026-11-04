"""Readers/writers for the unformatted files SYNTHE reads and writes (gfortran sequential records with
4-byte markers; fort.14 is a direct-access file without markers)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ...io.fortran import read_records, write_record

KW = 99
MW6 = 139 * 6

FORT12 = np.dtype([("nbuff", "<i4"), ("congf", "<f4"), ("nelion", "<i4"), ("elo", "<f4"),
                   ("gamrf", "<f4"), ("gamsf", "<f4"), ("gamwf", "<f4"), ("alpha", "<f4")])
FORT19 = np.dtype([("wl", "<f8"), ("elo", "<f4"), ("gf", "<f4"), ("nblo", "<i4"), ("nbup", "<i4"),
                   ("nelion", "<i4"), ("type", "<i4"), ("ncon", "<i4"), ("nelionx", "<i4"),
                   ("gammar", "<f4"), ("gammas", "<f4"), ("gammaw", "<f4"), ("alpha", "<f4"),
                   ("nbuff", "<i4")])
FORT15 = np.dtype([("iline", "<i4"), ("kapcen", "<f4")])


def _fixed_records(path: str | Path, dtype: np.dtype, nrec: int | None = None) -> np.ndarray:
    """Sequential file of equal-length records -> structured array (reads only the leading dtype.itemsize
    bytes of each record, as a Fortran READ with a shorter list does)."""
    raw = Path(path).read_bytes()
    if not raw:
        return np.zeros(0, dtype)
    (n,) = struct.unpack("<i", raw[:4])
    rec = np.dtype([("m1", "<i4"), ("d", f"V{n}"), ("m2", "<i4")])
    if len(raw) % rec.itemsize:
        recs = read_records(path)
        return np.frombuffer(b"".join(r[:dtype.itemsize] for r in recs), dtype=dtype).copy()
    a = np.frombuffer(raw, dtype=rec)
    if not ((a["m1"] == n).all() and (a["m2"] == n).all()):
        recs = read_records(path)
        return np.frombuffer(b"".join(r[:dtype.itemsize] for r in recs), dtype=dtype).copy()
    body = np.frombuffer(a["d"].tobytes(), dtype=np.uint8).reshape(len(a), n)[:, :dtype.itemsize]
    out = np.frombuffer(np.ascontiguousarray(body).tobytes(), dtype=dtype).copy()
    return out if nrec is None else out[:nrec]


def read_fort12(path, nlines: int | None = None) -> np.ndarray:
    return _fixed_records(path, FORT12, nlines)


def iter_fort12(path, nlines: int, chunk: int = 1 << 20):
    """The first `nlines` fort.12 records in blocks of <= `chunk` records (structured FORT12 arrays), read
    from the file block by block (fort.12 of a wide window has tens of millions of records)."""
    path = Path(path)
    size = path.stat().st_size
    if nlines <= 0 or size == 0:
        return
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<i", fh.read(4))
    rec = np.dtype([("m1", "<i4"), ("d", f"V{n}"), ("m2", "<i4")])
    if n < FORT12.itemsize or size % rec.itemsize:
        a = read_fort12(path, nlines)          # non-uniform records: whole-file reader
        for c0 in range(0, len(a), chunk):
            yield a[c0:c0 + chunk]
        return
    nrec = min(nlines, size // rec.itemsize)
    with open(path, "rb") as fh:
        for c0 in range(0, nrec, chunk):
            m = min(chunk, nrec - c0)
            fh.seek(c0 * rec.itemsize)
            a = np.frombuffer(fh.read(m * rec.itemsize), dtype=rec)
            if not ((a["m1"] == n).all() and (a["m2"] == n).all()):
                raise ValueError("fort.12: corrupt record markers")
            body = np.frombuffer(a["d"].tobytes(), dtype=np.uint8).reshape(m, n)[:, :FORT12.itemsize]
            yield np.frombuffer(bytearray(np.ascontiguousarray(body)), dtype=FORT12)


def write_asynth_records(f, asynth: np.ndarray) -> None:
    """Append WRITE(9)ASYNTH records (float32 [n, kw]) to the open file `f`."""
    asynth = np.ascontiguousarray(asynth, dtype="<f4")
    n = 4 * KW
    rec = np.empty(asynth.shape[0], dtype=np.dtype([("m1", "<i4"), ("d", "<f4", (KW,)), ("m2", "<i4")]))
    rec["m1"] = n
    rec["m2"] = n
    rec["d"] = asynth
    f.write(rec.tobytes())


def read_fort19(path, n19: int | None = None) -> np.ndarray:
    return _fixed_records(path, FORT19, n19)


def read_fort93(path) -> dict:
    (r,) = read_records(path)
    o = 0

    def take(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, r, o)
        o += struct.calcsize("<" + fmt)
        return v
    nlines, length, ifvac, ifnlte, n19 = take("5i")
    (turbv,) = take("f")
    deckj = np.frombuffer(r, dtype="<f4", count=7 * KW, offset=o).reshape(KW, 7).T.copy()  # (7, kw) Fortran
    o += 7 * KW * 4
    (ifpred,) = take("i")
    wlbeg, wlend, resolu, ratio, ratiolg = take("5d")
    (cutoff,) = take("f")
    (linout,) = take("i")
    return dict(nlines=nlines, length=length, ifvac=ifvac, ifnlte=ifnlte, n19=n19, turbv=turbv, deckj=deckj,
                ifpred=ifpred, wlbeg=wlbeg, wlend=wlend, resolu=resolu, ratio=ratio, ratiolg=ratiolg,
                cutoff=cutoff, linout=linout)


def read_fort10(path) -> dict:
    """xnfpelsyn.dat: header records + per layer (QABLOG, 2 continuum records, QXNFPEL+QDOPPLE)."""
    recs = read_records(path)
    r0 = recs[0]
    nrhox = struct.unpack_from("<i", r0, 0)[0]
    teff, glog = struct.unpack_from("<2d", r0, 4)
    title = r0[20:94]
    r1 = recs[1]
    nedge = struct.unpack_from("<i", r1, 0)[0]
    e = np.frombuffer(r1, dtype="<f8", count=3 * nedge, offset=4).reshape(nedge, 3)
    idmol = np.frombuffer(r1, dtype="<f8", count=100, offset=4 + 24 * nedge)
    momass = np.frombuffer(r1, dtype="<f8", count=100, offset=4 + 24 * nedge + 800)
    r2 = recs[2]
    ncon = struct.unpack_from("<i", r2, 0)[0]
    confrq = np.frombuffer(r2, dtype="<f8", count=ncon, offset=4)
    q = np.frombuffer(recs[3], dtype="<f8")
    names = ["t", "tkev", "tk", "hkt", "tlog", "hckt", "p", "xne", "xnatom", "rho", "rhox", "vturb", "xnfh"]
    st = {nm: q[i * KW:(i + 1) * KW].copy() for i, nm in enumerate(names)}
    o = len(names) * KW
    st["xnfhe"] = q[o:o + 2 * KW].reshape(2, KW).T.copy()   # (kw, 2) Fortran order
    st["xnfh2"] = q[o + 2 * KW:o + 3 * KW].copy()
    lay = recs[4:]
    ablog = np.stack([np.frombuffer(lay[4 * j], dtype="<f8") for j in range(nrhox)])
    xd = np.stack([np.frombuffer(lay[4 * j + 3], dtype="<f8") for j in range(nrhox)])
    return dict(nrhox=nrhox, teff=teff, glog=glog, title=title, nedge=nedge, rec1=r1, frqedg=e[:, 0].copy(),
                wledge=e[:, 1].copy(), cmedge=e[:, 2].copy(), idmol=idmol.copy(), momass=momass.copy(),
                ncon=ncon, confrq=confrq.copy(), state=st, qablog=ablog, qxnfpel=xd[:, :MW6].copy(),
                qdopple=xd[:, MW6:2 * MW6].copy())


def write_fort9(path, header1: bytes, header2: bytes, asynth: np.ndarray, n9: int) -> None:
    """asynth: float32 [length, kw]."""
    asynth = np.ascontiguousarray(asynth, dtype="<f4")
    length = asynth.shape[0]
    n = 4 * KW
    rec = np.empty(length, dtype=np.dtype([("m1", "<i4"), ("d", "<f4", (KW,)), ("m2", "<i4")]))
    rec["m1"] = n
    rec["m2"] = n
    rec["d"] = asynth
    with open(path, "wb") as f:
        write_record(f, header1)
        write_record(f, header2)
        f.write(rec.tobytes())
        write_record(f, struct.pack("<i", n9))


def write_int_record(path, value: int) -> None:
    with open(path, "wb") as f:
        write_record(f, struct.pack("<i", value))


def write_fort15(path, recs: np.ndarray) -> None:
    rec = np.empty(len(recs), dtype=np.dtype([("m1", "<i4"), ("d", FORT15), ("m2", "<i4")]))
    rec["m1"] = 8
    rec["m2"] = 8
    rec["d"] = recs
    Path(path).write_bytes(rec.tobytes())
