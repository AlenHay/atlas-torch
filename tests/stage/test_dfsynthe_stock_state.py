"""Stage test: intermediate state of the stock dfsynthe.exe vs Flavor('stock') at 5012 K (ITEFF=21).

The stock executable (ifx -O2, -fp-model fast) runs once under gdb; breakpoints after SELECTLINES, after XLINOP of
layers 1, 16 and 25 and after their TAPE12 lines dump (addresses of this binary, see the objdump notes in
lines/selectlines.py, tables.py, hydrogen.py, xlinop.py):
  * the scratch files TAPE12 (fort.12) and TAPE19 (fort.19) SELECTLINES writes, and its TABLOG / TABBOLT;
  * COMMON /H1TAB/ (H0TAB, H1TAB, H2TAB, ATAB), /EXTAB/ (EXTAB, E1TAB) and HPROF4's SAVEd depth vectors;
  * CONTINUUM and BUFFER after XLINOP and after the TAPE12 lines of the three layers.
All must be bit-identical to the port's (stock flavour with the Intel runtime: libimf / SVML through ctypes).
Before these forms were emulated, TAPE12 differed in 1.17e6 of 2.2e6 CONGF values, ATAB in 59 % of its entries
and BUFFER in 2-14 % of the bins, and one 5012 K MLINES count (tests/stage/test_dfsynthe_printouts.py) was off by
one line.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe import odf_io
from atlas_torch.dfsynthe.dfsynthe import Config
from atlas_torch.dfsynthe.ftz import scope
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines import imf, svml
from atlas_torch.dfsynthe.lines.accumulate import LineAccumulator
from atlas_torch.dfsynthe.lines.continuum import continuum, layer_setup
from atlas_torch.dfsynthe.lines.hydrogen import Hydrogen
from atlas_torch.dfsynthe.lines.selectlines import iter_tape12, select_consts, select_tape19, selectlines
from atlas_torch.dfsynthe.lines.tables import LENBUFF, LENGTH, build_tables, device_tables
from atlas_torch.dfsynthe.lines.xlinop import XLinop

pytestmark = pytest.mark.stage

STOCK_EXE = S.EXT / "BasicATLAS" / "bin" / "dfsynthe.exe"
ITEFF = 21
LAYERS = (1, 16, 25)
# dfsynthe.exe addresses
AFTER_SELECTLINES, AFTER_XLINOP, AFTER_LINES = 0x4066B0, 0x406C19, 0x4077B0
J_ADDR, BUFFER, CONTIN = 0x4BD0DC, 0x1409080, 0x5CA9A0
TABLOG, TABBOLT, H1TAB, EXTAB, HDEPTH = 0x4EE880, 0x4CE880, 0x21805A0, 0x5660A0, 0x4CD660
HDEPTH_NAMES = {0x4CD660: "t3nh2", 0x4CD6E0: "t3nhe", 0x4CD760: "c2d", 0x4CD7E0: "c1d", 0x4CD860: "y1s",
                0x4CD8E0: "y1b", 0x4CD960: "gcon2", 0x4CD9E0: "gcon1", 0x4CDA60: "fo", 0x4CDAE0: "pp"}


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.int32)


@pytest.fixture(scope="module")
def stock(tmp_path_factory):
    gdb = shutil.which("gdb")
    lib = S.ifx_libdir()
    if gdb is None or lib is None or not STOCK_EXE.exists() or not (S.STOCK / "xnfpdf.dat").exists():
        pytest.skip("needs gdb, the Intel runtime, the stock dfsynthe.exe and the stock solar run")
    if not (svml.available() and imf.logf(np.ones(1, np.float32)) is not None):
        pytest.skip("Intel runtime not loadable through ctypes")
    wd = tmp_path_factory.mktemp("stock_state")
    out = wd / "dump"
    out.mkdir()
    for name, src in (("fort.10", S.STOCK / "xnfpdf.dat"), ("fort.22", S.STOCK / "xnfpdfmax.dat")):
        (wd / name).symlink_to(src)
    for unit, name, _ in odf_io.LINE_FILES.values():
        (wd / f"fort.{unit}").symlink_to(S.DATA / name)
    (wd / "fort.51").symlink_to(S.DATA / odf_io.NLTE_FILE)
    (wd / "sw.txt").write_text("0" * (ITEFF - 1) + "1" + "0" * (57 - ITEFF) + "\n")
    n_ab = 4 * (3 * 2001 + 281 * 1001)
    cmd = ["set debuginfod enabled off", "set language c", "set pagination off",
           f"break *{AFTER_SELECTLINES:#x}", "commands 1", "silent",
           f"shell cp fort.12 {out}/tape12.bin; cp fort.19 {out}/tape19.bin",
           f"dump binary memory {out}/tablog.bin {TABLOG:#x} {TABLOG + 131072:#x}",
           f"dump binary memory {out}/tabbolt.bin {TABBOLT:#x} {TABBOLT + 131072:#x}",
           "continue", "end",
           f"break *{AFTER_XLINOP:#x}", "commands 2", "silent", f"set $j = *(int*){J_ADDR:#x}",
           "if $j == 1",
           f"  dump binary memory {out}/h1tab.bin {H1TAB:#x} {H1TAB + n_ab:#x}",
           f"  dump binary memory {out}/extab.bin {EXTAB:#x} {EXTAB + 4 * 12001:#x}",
           f"  dump binary memory {out}/hdepth.bin {HDEPTH:#x} {HDEPTH + 0x500:#x}",
           "end"]
    for j in LAYERS:
        cmd += [f"if $j == {j}", f"  dump binary memory {out}/xl{j:02d}.bin {BUFFER:#x} {BUFFER + 4 * LENGTH:#x}",
                "end"]
    cmd += ["continue", "end", f"break *{AFTER_LINES:#x}", "commands 3", "silent",
            f"set $j = *(int*){J_ADDR:#x}"]
    for j in LAYERS:
        cmd += [f"if $j == {j}", f"  dump binary memory {out}/ln{j:02d}.bin {BUFFER:#x} {BUFFER + 4 * LENGTH:#x}",
                f"  dump binary memory {out}/ct{j:02d}.bin {CONTIN:#x} {CONTIN + 4 * LENGTH:#x}", "end"]
    cmd += [f"if $j == {max(LAYERS)}", "  kill", "  quit", "end", "continue", "end", "run < sw.txt > out.txt"]
    (wd / "g.cmd").write_text("\n".join(cmd) + "\n")
    subprocess.run([gdb, "-q", "-batch", "-x", "g.cmd", str(STOCK_EXE)], cwd=wd, capture_output=True, timeout=1800,
                   env={"LD_LIBRARY_PATH": lib, "PATH": "/usr/bin:/bin", "SHELL": "/bin/sh"})
    need = ["tape12.bin", "tape19.bin", "h1tab.bin", "extab.bin"] + [f"ln{j:02d}.bin" for j in LAYERS]
    if not all((out / f).exists() for f in need):
        pytest.skip("gdb run of the stock executable did not produce its dumps")
    return out


@pytest.fixture(scope="module")
def port():
    fl = Config().flavor
    torch.set_num_threads(1)
    xt = odf_io.read_xnfpdf(S.STOCK / "xnfpdf.dat", ITEFF)
    mx = odf_io.read_xnfpdfmax(S.STOCK / "xnfpdfmax.dat", ITEFF)
    dev = torch.device("cpu")
    with scope(True):
        consts = select_consts(mx.teff, mx.wledge, fl)
        t12, t19, _ = selectlines(S.DATA, mx.teff, mx.wledge, mx.xnfdopmax, fl)
        tnp = build_tables(fl)
        tabs = device_tables(dev, fl)
        lay = layer_setup(xt, fl, dev)
        cont = continuum(xt.wledge, lay["ablog"], fl)
        hyd = Hydrogen(lay["st"], lay["xnfph"], tabs["extab"], tabs["h0tab"], fl)
        nl = cont.shape[0]
        pad = LineAccumulator.PAD
        bufpad = torch.zeros(nl * LENBUFF + 2 * pad, dtype=torch.float32)
        buf = bufpad[pad:pad + nl * LENBUFF]
        XLinop(tabs, lay, cont, fl).run(buf, select_tape19(S.DATA, mx.teff, mx.wledge, mx.xnfdopmax, fl))
        xl = {j: buf.view(nl, LENBUFF)[j - 1, :LENGTH].clone() for j in LAYERS}
        cols = ("nbuff", "congf", "nelion", "gamr", "gams", "gamw")

        def gen():
            for _, t in iter_tape12(S.DATA, mx.teff, mx.wledge, mx.xnfdopmax, fl):
                yield {k: torch.from_numpy(getattr(t, k)) for k in cols}
        LineAccumulator(tabs, lay["xne"], lay["txnxn"], lay["dopple"], lay["xnfdop"], cont,
                        fastmath=fl.fastmath).accumulate(buf, gen, padded=bufpad)
        ln = {j: buf.view(nl, LENBUFF)[j - 1, :LENGTH].clone() for j in LAYERS}
    return dict(consts=consts, t12=t12, t19=t19, tnp=tnp, cont=cont, hyd=hyd, xl=xl, ln=ln)


def test_selectlines_tapes(stock, port):
    raw = np.fromfile(stock / "tape12.bin", np.uint8)
    rec = 8 + 4 * 6 * 10000                                  # sequential unformatted records with 4-byte markers
    body = raw.reshape(-1, rec)[:, 4:-4].copy().view(np.int32).reshape(-1, 6)
    t12 = port["t12"]
    n = len(t12)
    assert body.shape[0] >= n
    for k, (name, dt) in enumerate((("nbuff", np.int32), ("congf", np.float32), ("nelion", np.int32),
                                    ("gamr", np.float32), ("gams", np.float32), ("gamw", np.float32))):
        assert np.array_equal(getattr(t12, name).astype(dt).view(np.int32), body[:n, k]), name
    r19 = np.fromfile(stock / "tape19.bin", np.uint8).reshape(-1, 56)          # RECL=14 words: WL, GNNNGGG(11)
    t19 = port["t19"]
    assert len(t19) == len(r19)
    assert np.array_equal(t19["wl"], r19[:, :8].copy().view(np.float64).ravel())
    g = r19[:, 8:52].copy().view(np.int32)
    assert np.array_equal(t19["gf"].view(np.int32), g[:, 0]) and np.array_equal(t19["nbuff"], g[:, 10])
    c = port["consts"]
    assert np.array_equal(_bits(c.tablog), np.fromfile(stock / "tablog.bin", np.int32))
    assert np.array_equal(_bits(c.tabbolt), np.fromfile(stock / "tabbolt.bin", np.int32))


def test_tables(stock, port):
    t = port["tnp"]
    s = np.fromfile(stock / "h1tab.bin", np.int32)
    assert np.array_equal(_bits(t.h0tab), s[:2001])
    assert np.array_equal(_bits(t.h1tab), s[2001:4002])
    assert np.array_equal(_bits(t.h2tab), s[4002:6003])
    assert np.array_equal(_bits(t.atab), s[6003:].reshape(1001, 281).T)          # ATAB(281,1001), column-major
    e = np.fromfile(stock / "extab.bin", np.int32)
    assert np.array_equal(_bits(t.extab), e[:10001]) and np.array_equal(_bits(t.e1tab), e[10001:])


def test_hprof4_depth_vectors(stock, port):
    d = np.fromfile(stock / "hdepth.bin", np.int32).reshape(-1, 32)[:, :25]
    hyd = port["hyd"]
    for addr, name in HDEPTH_NAMES.items():
        assert np.array_equal(_bits(getattr(hyd, name).numpy()), d[(addr - HDEPTH) // 128]), name


@pytest.mark.parametrize("j", LAYERS)
def test_buffers(stock, port, j):
    for key, val in (("ct", port["cont"][j - 1]), ("xl", port["xl"][j]), ("ln", port["ln"][j])):
        s = np.fromfile(stock / f"{key}{j:02d}.bin", np.int32)
        p = _bits(val.numpy())
        assert int((p != s).sum()) == 0, (key, j, np.nonzero(p != s)[0][:8])
