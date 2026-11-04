"""Stock (ifx -fp-model fast) semantics of HPROF4's Lyman-alpha H2+ term (label 52) and of the NaN handling in
XLINOP / MAIN (Hydrogen._lya_h2plus_fast).

Root cause of the stock-vs-precise ODF differences (up to 3.6 dex rows at cool temperatures): the stock binary
evaluates CUTOFF4000/PRQSP4000*PRQS/FO*DBETA*1.77245*DOP as one numerator/denominator pair; at low electron
density both underflow, FTZ flushes them and 0/0 = NaN; the NaN Lyman-alpha wing then replaces the opacity of
every sampled bin by the continuum.  Checked against an instrumented stock-flags build (dfsynthe.exe prints NaN
for HPROF4 in exactly these samples) and the stock machine code (dfsynthe.exe 0x40c8ab..0x40c900).

1. the folded constant equals the stock binary's .rodata word;
2. ITEFF=6 (2512 K) of the solar run: fastmath gives NaN outside the Doppler core within 4000 cm-1 of the
   line centre in layers 1-7 and a finite value from layer 8 on (as the stock binary); precise: finite;
3. where finite, fastmath and precise agree to rounding;
4. NaN semantics of MAX(KAPPA,1.E-20) (maxss) and MAX(BUFFER,CONTINUUM) (maxps) as used by the port.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe import odf_io
from atlas_torch.dfsynthe.ftz import scope
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines.continuum import layer_setup
from atlas_torch.dfsynthe.lines.f32 import F32, F64, Flavor
from atlas_torch.dfsynthe.lines.hydrogen import Hydrogen
from atlas_torch.dfsynthe.lines.tables import device_tables

STOCK_EXE = S.EXT / "BasicATLAS" / "bin" / "dfsynthe.exe"


def test_c52_constant_in_stock_binary():
    c = Hydrogen._c52()
    assert c.view(np.uint32) == 0x04D60B42
    if STOCK_EXE.exists():
        data = STOCK_EXE.read_bytes()
        assert np.float32(c).tobytes() in data                      # 4891a0: 42 0b d6 04
        assert np.float32(1.1991699e14).tobytes() in data            # 48919c: 4000*2.99792458E10 folded


@pytest.fixture(scope="module")
def lya():
    if not (S.STOCK / "xnfpdf.dat").exists():
        pytest.skip("stock solar run missing")
    xt = odf_io.read_xnfpdf(S.STOCK / "xnfpdf.dat", 6)
    out = {}
    with scope(True):
        for fast in (True, False):
            fl = Flavor("stock", fastmath=fast)
            lay = layer_setup(xt, fl, torch.device("cpu"))
            tabs = device_tables(torch.device("cpu"), fl)
            hyd = Hydrogen(lay["st"], lay["xnfph"], tabs["extab"], tabs["h0tab"], fl)
            s = hyd.setup(1, 2)
            wl = 121.56708975
            # samples just outside the Doppler core on both sides and within 4000 cm-1 (label 52)
            dl = torch.tensor([-0.02, -0.005, 0.005, 0.0049, 0.02, 0.1], dtype=F64)
            vals = []
            for j in range(12):
                n = dl.numel()
                jt = torch.full((n,), j)
                q = hyd.prep(s, jt, torch.full((n,), wl, dtype=F64), dl, lay["dopple"][j, 0].expand(n))
                assert not bool(q["ifcore"].any())
                pp = hyd.pp[jt]
                vals.append(hyd.finish(q, pp, pp))
            out[fast] = torch.stack(vals)
    return out


def test_lya_nan_layers(lya):
    fast, precise = lya[True], lya[False]
    assert torch.isfinite(precise).all()
    assert torch.isnan(fast[:7]).all()                 # layers 1-7: 0/0 in the stock code
    assert torch.isfinite(fast[7:]).all()              # layer 8+: denominator stays normal


def test_lya_finite_agree(lya):
    fast, precise = lya[True][7:], lya[False][7:]
    rel = ((fast - precise).abs() / precise.abs()).max()
    assert float(rel) < 1e-5


def test_nan_semantics():
    kap = torch.tensor([float("nan"), 1e-30, 2.0], dtype=F32)
    floor = 1e-20
    got = torch.where(kap > floor, kap, torch.full_like(kap, floor))        # maxss kappa, const
    assert got.tolist()[0] == pytest.approx(1e-20) and got[2] == 2.0
    buf = torch.tensor([float("nan"), 0.5, 3.0], dtype=F32)
    cont = torch.tensor([1.0, 1.0, 1.0], dtype=F32)
    assert torch.where(buf > cont, buf, cont).tolist() == [1.0, 1.0, 3.0]   # maxps buffer, continuum
    assert not bool((kap < cont)[0])                                       # NaN never terminates a wing


def test_vttab_matches_stock_printout(tmp_path):
    """Stock dfsynthe.exe prints VTTAB (FORMAT I3,' KM/S',10F10.7/(8X,10F10.7)) before any temperature; with
    every temperature switch off it stops right after.  Flavor('stock') (fast-math VTTAB: I*((1./V)*C),
    VTPROF*(1./SUM)) reproduces all 131 printed lines; the source formula (fastmath=False) does not."""
    import re
    import subprocess
    from atlas_torch.dfsynthe.lines.tables import vttab
    lib = S.ifx_libdir()
    if not STOCK_EXE.exists() or lib is None or not (S.STOCK / "xnfpdf.dat").exists():
        pytest.skip("stock dfsynthe.exe / Intel runtime / stock run missing")
    for name, src in (("fort.10", S.STOCK / "xnfpdf.dat"), ("fort.22", S.STOCK / "xnfpdfmax.dat")):
        (tmp_path / name).symlink_to(src)
    for unit, name, _ in odf_io.LINE_FILES.values():
        (tmp_path / f"fort.{unit}").symlink_to(S.DATA / name)
    (tmp_path / "fort.51").symlink_to(S.DATA / odf_io.NLTE_FILE)
    out = subprocess.run([str(STOCK_EXE)], input="0" * 57 + "\n", text=True, cwd=tmp_path, capture_output=True,
                         env={"LD_LIBRARY_PATH": lib, "PATH": "/usr/bin:/bin"}, timeout=300).stdout.splitlines()
    i = next(k for k, ln in enumerate(out) if re.match(r"^\s+1 KM/S", ln))
    printed = []
    while i < len(out) and (re.match(r"^\s*\d+ KM/S", out[i]) or (out[i].startswith(" " * 8) and out[i].strip())):
        printed.append(out[i].rstrip())
        i += 1

    def lines(fl):
        vp, vc, nvt = vttab(fl)
        res = []
        for iv in range(1, 21):
            vals = [float(vc[iv - 1])] + [float(x) for x in vp[:nvt[iv - 1], iv - 1]]
            res.append("%3d KM/S" % iv + "".join("%10.7f" % x for x in vals[:10]))
            res += [" " * 8 + "".join("%10.7f" % x for x in vals[k:k + 10]) for k in range(10, len(vals), 10)]
        return res
    assert len(printed) == 131
    assert lines(Flavor("stock")) == printed
    assert lines(Flavor("stock", fastmath=False)) != printed
