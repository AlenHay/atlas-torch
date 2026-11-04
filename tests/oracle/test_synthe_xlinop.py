"""Oracle test (synthe.for): XLINOP, all line types, vs the -O0 library.

A SYNTHE work directory is built from the 380-420 nm reference inputs with NLINES=0 (no fort.12 lines) and
a fort.19 holding the real special lines of that window plus synthetic ones exercising every XLINOP branch
the reference windows do not: Voigt-wing lines (TYPE 0, with and without a merging continuum NCON, with
alpha != 0 van der Waals scaling, TYPE 3 = PRD), autoionizing (TYPE 1), merged continua (TYPE >= 4, H and
metals), coronal and He II (skipped), deuterium (TYPE -2), Balmer/Paschen alpha and beta lines (620/630
branches) and a 3He line (TYPE -4).  The port (mode="scalar") runs the whole program; the oracle's XLINOP is
called layer by layer exactly like SYNTHE's main loop, with the same COMMONs, and each layer's BUFFER, the
MLINES count and the fort.15 records are compared bit for bit.
"""
import ctypes
import math
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from atlas_torch.io.fortran import read_records
from atlas_torch.synthe.lineopac import fortio, libm32
from atlas_torch.synthe.synthe_prog import SyntheProgram

F32 = np.float32
EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
SRC = EXT / "work" / "sun_380_420_stock" / "pre_synthe"
HE1 = EXT / "BasicATLAS" / "data" / "synthe_files" / "he1tables.dat"


def _rec(path, payloads):
    with open(path, "wb") as f:
        for p in payloads:
            m = struct.pack("<i", len(p))
            f.write(m + p + m)


def _line(wl, elo, gf, nblo, nbup, nelion, typ, ncon, nelionx, gr, gs, gw, alpha, ratiolg, ixwlbeg, lim=1):
    ixwl = int(math.log(wl) / ratiolg + .5)
    nbuff = ixwl - ixwlbeg + 1
    return struct.pack("<dffiiiiiiffffii", wl, elo, gf, nblo, nbup, nelion, typ, ncon, nelionx, gr, gs, gw,
                       alpha, nbuff, lim)


def _cgf(gf, wl):
    return float(F32(.026538 / 1.77245 * gf / (2.99792458e17 / wl)))


def _gam(g, wl):
    return float(F32(10 ** g / 12.5664 / (2.99792458e17 / wl)))


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    for p in (SRC, HE1):
        if not p.exists():
            pytest.skip(f"missing {p}")
    wd = tmp_path_factory.mktemp("xlinop")
    for f in ("fort.10", "fort.20", "fort.14"):
        shutil.copy(SRC / f, wd / f)
        os.chmod(wd / f, 0o644)
    shutil.copy(HE1, wd / "fort.18")
    p = fortio.read_fort93(SRC / "fort.93")
    ixwlbeg = int(math.log(p["wlbeg"]) / p["ratiolg"])
    if math.exp(ixwlbeg * p["ratiolg"]) < p["wlbeg"]:
        ixwlbeg += 1
    real = read_records(SRC / "fort.19")
    rl, ix = p["ratiolg"], ixwlbeg
    syn = [
        # Ca II K, H: Voigt wings over the whole window (TYPE 0), and with a merging continuum (NCON=1)
        _line(393.4777, 0., _cgf(10 ** .135, 393.4777), 0, 0, 116, 0, 0, 0, _gam(8.2, 393.48), _gam(-5.5, 393.48),
              _gam(-7.8, 393.48), 0., rl, ix),
        _line(396.9591, 0., _cgf(10 ** -.18, 396.96), 0, 0, 116, 0, 1, 16, _gam(8.2, 396.96), _gam(-5.5, 396.96),
              _gam(-7.8, 396.96), 0., rl, ix),
        # alpha != 0 (ABO van der Waals), Fe I line, and a PRD line (TYPE 3) after it (sticky TXNXN)
        _line(404.5813, 7000., _cgf(10 ** .28, 404.58), 0, 0, 157, 0, 0, 0, _gam(8.0, 404.58), _gam(-6.0, 404.58),
              _gam(-7.5, 404.58), .25, rl, ix),
        _line(406.3594, 12000., _cgf(10 ** .06, 406.36), 0, 0, 157, 3, 0, 0, _gam(8.0, 406.36), _gam(-6.0, 406.36),
              _gam(-7.5, 406.36), 0., rl, ix),
        # line outside the window (red side, blue wing only) and one below WBEGIN (red wing only)
        _line(421.5, 5000., _cgf(10 ** .5, 421.5), 0, 0, 115, 0, 0, 0, _gam(8.0, 421.5), _gam(-5.0, 421.5),
              _gam(-7.0, 421.5), 0., rl, ix),
        _line(379.5, 5000., _cgf(10 ** .5, 379.5), 0, 0, 115, 0, 0, 0, _gam(8.0, 379.5), _gam(-5.0, 379.5),
              _gam(-7.0, 379.5), 0., rl, ix),
        # autoionizing line (GF=G, GAMMAS=ASHORE, GAMMAW=BSHORE, GAMMAR=width in Hz)
        _line(400.0, 20000., 2.0, 0, 0, 116, 1, 0, 0, 3.e12, 1.5, 2.e-17, 0., rl, ix),
        # merged continua: metal (NELION 79 = Mg I, NLAST 12) and hydrogen (NELION 1, NLAST 9)
        _line(390.0, 30000., 1.e-18, 0, 0, 67, 12, 0, 0, 0., 0., 0., 0., rl, ix),
        _line(383.0, 82259.1, 3.e-19, 0, 0, 1, 9, 0, 0, 0., 0., 0., 0., rl, ix),
        # coronal and He II lines are skipped
        _line(395.0, 0., 1.e-15, 0, 0, 157, 2, 0, 0, 1., 0., 0., 0., rl, ix),
        _line(405.0, 0., 1.e-15, 0, 0, 8, -6, 0, 0, 1., 0., 0., 0., rl, ix),
        # deuterium (TYPE -2) H delta, H alpha-type (NBUP=NBLO+1, Paschen alpha far red) and a beta line
        _line(410.1745, 82259.1, 3.6e-18, 2, 6, 1, -2, 2, 1, 6.7e-8, 8.5e-21, 1.2e-23, 0., rl, ix),
        _line(656.4614, 82259.1, 1.0e-16, 2, 3, 1, -1, 2, 1, 6.7e-8, 8.5e-21, 1.2e-23, 0., rl, ix),
        _line(486.2710, 82259.1, 2.3e-17, 2, 4, 1, -1, 2, 1, 7.8e-8, 1.3e-21, 7.5e-24, 0., rl, ix),
        # 3He line of 4026 (TYPE -4)
        _line(402.7347, 169087.02, 8.58e-18, 4, 23, 7, -4, 4, 2, 1.77e-9, 7.4e-20, 9.1e-24, 0., rl, ix),
    ]
    lines = list(real) + syn
    _rec(wd / "fort.19", lines)
    # fort.93: no fort.12 lines, our N19
    deck = np.zeros((99, 7), "<f4")
    deck[:, :] = p["deckj"].T
    _rec(wd / "fort.93", [struct.pack("<5if", 0, p["length"], p["ifvac"], p["ifnlte"], len(lines), p["turbv"]) +
                          deck.tobytes() + struct.pack("<i5dfi", p["ifpred"], p["wlbeg"], p["wlend"], p["resolu"],
                                                       p["ratio"], p["ratiolg"], p["cutoff"], p["linout"])])
    open(wd / "fort.12", "wb").close()
    prog = SyntheProgram(wd, mode="scalar")
    old = libm32.LOG10F
    libm32.LOG10F = "host"
    try:
        res = prog.run(write=False)
    finally:
        libm32.LOG10F = old
    return dict(wd=wd, prog=prog, res=res, nlines=len(lines), p=p, ixwlbeg=ixwlbeg)


@pytest.mark.stage   # the `case` fixture runs the whole SYNTHE program on a real fort.19 (5-30 s)
def test_xlinop_vs_fortran(oracle, case, monkeypatch):
    wd, prog, res = case["wd"], case["prog"], case["res"]
    ctx = prog.ctx
    p = case["p"]
    nr, length = ctx.nrhox, ctx.length
    od = wd / "oracle"
    od.mkdir(exist_ok=True)
    shutil.copy(wd / "fort.19", od / "fort.19")
    shutil.copy(wd / "fort.18", od / "fort.18")
    monkeypatch.chdir(od)
    o = oracle("synthe")
    ext = ctx.extab.numpy()
    e = o.common("extab")
    e["extab"][:] = ext[:1001]
    e["extabf"][:] = ext[1001:2002]
    e["e1tab"][:] = ext[2002:]
    o.call("tabvoigt", ctypes.c_float(200.0), 2001)
    nl = o.common("nlines")
    for k, v in dict(wlbeg=p["wlbeg"], wlend=p["wlend"], resolu=p["resolu"], ratio=p["ratio"],
                     ratiolg=p["ratiolg"], wbegin=ctx.wbegin, length=length, ixwlbeg=case["ixwlbeg"]).items():
        nl[k][...] = v
    o.set_common("rhox", nrhox=nr)
    f10 = prog.f10
    s = f10["state"]
    tt = o.common("temp")
    tt["t"][:] = s["t"]
    tt["hckt"][:] = s["hckt"]
    tt["itemp"][0] = 1
    o.set_common("state", xne=s["xne"])
    o.common("bhe")["xnfhe"][:, :] = s["xnfhe"].astype(F32)
    bh = o.common("bhyd")
    xph = np.zeros((99, 2), F32)
    xph[:nr] = f10["qxnfpel"][:, :2]
    bh["xnfph"][:, :] = xph
    bh["xnfh"][:] = s["xnfh"]
    tx = o.common("txnxn")
    tx["xnfh2"][:] = s["xnfh2"]
    xd = o.common("xnfdop")
    cont = o.common("contin")["continuum"]
    buf = o.common("buffer")["buffer"]
    d = lambda a: a[:nr].astype(F32).astype(np.float64)  # noqa: E731  (REAL*4 values in a REAL*8 expression)
    txn0 = (d(s["xnfh"]) + .42 * d(s["xnfhe"][:, 0]) + .85 * d(s["xnfh2"])) * \
        libm32.dpow(d(s["t"]) / 10000., float(F32(.3)))
    for j in range(nr):
        xd["xnfpel"][:] = ctx.xnfpel[j].numpy()
        xd["dopple"][:] = ctx.dopple[j].numpy()
        xd["xnfdop"][:] = ctx.xnfdop[j].numpy()
        cont[:length] = ctx.continuum[j].numpy()
        buf[:] = 0
        nl["mlines"][...] = 0
        tx["txnxn"][j] = F32(txn0[j])
        o.call("xlinop", j + 1, case["nlines"], ctypes.c_float(ctx.cutoff), ctypes.c_float(ctx.velshift[j]),
               ctx.ifvac, ctx.linout)
        got = res["buffer"][j]
        ref = buf[:length]
        bad = got.view(np.uint32) != ref.view(np.uint32)
        assert not bad.any(), (j, np.nonzero(bad)[0][:5], got[bad][:5], ref[bad][:5])
        assert int(np.ravel(nl["mlines"])[0]) == int(res["mlines"][j]), j
    # fort.15: (ILINE, KAPCEN) of the He I lines, written by the oracle in the same order (unit 15 is
    # still open in the oracle: FLUSH it through libgfortran before reading)
    ctypes.CDLL("libgfortran.so.5")._gfortran_flush_i4(ctypes.byref(ctypes.c_int(15)))
    r15 = np.fromfile(od / "fort.15", dtype=[("m1", "<i4"), ("d", fortio.FORT15), ("m2", "<i4")])["d"]
    assert np.array_equal(r15["iline"], res["fort15"]["iline"])
    assert np.array_equal(r15["kapcen"].view(np.uint32), res["fort15"]["kapcen"].view(np.uint32))
