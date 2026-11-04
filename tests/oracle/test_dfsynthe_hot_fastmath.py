"""Stock (ifx -fp-model fast) forms that matter at hot temperatures (ODF jumps of 14-41 steps at 12589-22387 K
and 119 steps at 112202 K before they were emulated; tests/stage/test_dfsynthe_prog.py checks the ODFs).

1. HPROF4 Lyman-alpha quasi-H2+ cutoff, 4000-20000 cm-1 red of the centre (Hydrogen._lya_h2plus_cut_fast): the
   source order forms 10.**(CUTOFF-14.)/2.99792458E10, which is subnormal for CUTOFF < -13.45 and flushes to 0
   under FTZ; the stock binary computes ((XNFPH(J,2)*K)*DOP)*10.**(CUTOFF-14.) and keeps the term.
2. XLINOP Inglis-Teller merging (XLinop._merge_energies): NMERGE = XNE**(-2./15.)*1600.-1.5 through SVML powf,
   EMERGE = (1./NMERGE**2)*109737.312 with rcpps + Newton in the 4-wide vector loop, divss for the remainder.
"""
import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe import odf_io
from atlas_torch.dfsynthe.ftz import scope
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines import svml
from atlas_torch.dfsynthe.lines.continuum import continuum, layer_setup
from atlas_torch.dfsynthe.lines.f32 import F32, Flavor, lit
from atlas_torch.dfsynthe.lines.hydrogen import C10, Hydrogen, _tab
from atlas_torch.dfsynthe.lines.rcpps import rcpps
from atlas_torch.dfsynthe.lines.tables import device_tables
from atlas_torch.dfsynthe.lines.xlinop import XLinop

STOCK_EXE = S.EXT / "BasicATLAS" / "bin" / "dfsynthe.exe"
RODATA_VMA, RODATA_OFF = 0x489000, 0x89000            # objdump -h dfsynthe.exe


def _rodata(addr: int) -> np.float32:
    data = STOCK_EXE.read_bytes()
    return np.frombuffer(data[addr - RODATA_VMA + RODATA_OFF:][:4], np.float32)[0]


@pytest.mark.skipif(not STOCK_EXE.exists(), reason="stock dfsynthe.exe missing")
def test_constants_in_stock_binary():
    assert Hydrogen._kh2plus().view(np.uint32) == 0x2E82030B
    assert _rodata(0x489198) == Hydrogen._kh2plus()                    # used at 0x40c9b0 / 0x40e0c9 / 0x40fa90
    f4 = np.float32
    vals = (-(f4(2.0) / f4(15.0)), 1600.0, -1.5, 1.0, lit("109737.312"), lit("109677.576"))
    scalar = (0x48905C, 0x489060, 0x489064, 0x48900C, 0x489068, 0x48906C)      # loop 0x408935..0x4089cf
    vector = (0x489480, 0x489490, 0x4894A0, 0x4894B0, 0x4894C0, 0x4894D0)      # loop 0x408859..0x408920
    for a, b, v in zip(scalar, vector, vals):
        assert _rodata(a) == f4(v) and _rodata(b) == f4(v), (hex(a), hex(b))


@pytest.fixture(scope="module")
def hot():
    if not (S.STOCK / "xnfpdf.dat").exists():
        pytest.skip("stock solar run missing")
    xt = odf_io.read_xnfpdf(S.STOCK / "xnfpdf.dat", 33)                # 12589 K
    fl = Flavor("stock")
    with scope(True):
        lay = layer_setup(xt, fl, torch.device("cpu"))
        tabs = device_tables(torch.device("cpu"), fl)
        hyd = Hydrogen(lay["st"], lay["xnfph"], tabs["extab"], tabs["h0tab"], fl)
    return dict(xt=xt, lay=lay, tabs=tabs, hyd=hyd, fl=fl)


def test_h2plus_cutoff_survives_ftz(hot):
    hyd, fl = hot["hyd"], hot["fl"]
    ch = torch.from_numpy(_tab("hprof4", "cutoffh2plus").astype(np.float32))
    cut = ch[:25].clone()                                              # tabulated, 15000-12600 cm-1 from the centre
    j = torch.full((cut.numel(),), 16)                                 # layer 17 (dense; 15-step jumps before)
    dop = torch.full_like(cut, 2.5e10)

    def source():
        c = (fl.powf(torch.full_like(cut, 10.0), cut - 14.0) / torch.tensor(float(C10), dtype=F32)) * hyd.xnfph[j, 1]
        return (c * float(lit("1.77245"))) * dop
    with scope(True):
        fast = hyd._lya_h2plus_cut_fast(cut, dop, j)
        src_ftz = source()
    with scope(False):
        src_ieee = source()
    tiny = (cut - 14.0) < np.log10(np.finfo(np.float32).tiny * float(C10))
    assert bool(tiny.any()) and bool((src_ftz[tiny] == 0).all())     # the old (source-order) port lost these
    assert bool((fast > 0).all())
    ulp = (fast.double() - src_ieee.double()).abs() / torch.finfo(F32).eps / src_ieee.double()
    assert float(ulp[~tiny].max()) <= 4                                # same value up to rounding where normal
    assert float(((fast.double() - src_ieee.double()).abs() / src_ieee.double())[tiny].max()) < 1e-3


@pytest.mark.skipif(not svml.available(), reason="Intel SVML runtime missing")
def test_svml_powf_not_correctly_rounded():
    x = np.float32(10.0) ** np.linspace(5, 20, 2001, dtype=np.float32)
    y = np.float32(-(np.float32(2.0) / np.float32(15.0)))
    p = svml.powf(x, y)
    cr = (x.astype(np.float64) ** np.float64(y)).astype(np.float32)
    d = np.abs(p.view(np.int32).astype(np.int64) - cr.view(np.int32).astype(np.int64))
    assert d.max() <= 1 and d.any()                                    # 1-ulp differences exist


@pytest.mark.skipif(not svml.available(), reason="Intel SVML runtime missing")
def test_merge_energies_stock_form(hot):
    xt, lay, tabs = hot["xt"], hot["lay"], hot["tabs"]
    with scope(True):
        cont = continuum(xt.wledge, lay["ablog"], hot["fl"])
        e_fast, eh_fast = XLinop(tabs, lay, cont, Flavor("stock"))._merge_energies(Flavor("stock"))
        e_src, _ = XLinop(tabs, lay, cont, Flavor("stock", fastmath=False))._merge_energies(
            Flavor("stock", fastmath=False))
    xne = lay["st"]["xne"]
    nl = xne.shape[0]
    pw = torch.from_numpy(svml.powf(xne.numpy(), -(np.float32(2.0) / np.float32(15.0))))
    nm = pw * 1600.0 + float(lit("-1.5"))
    x2 = nm * nm
    nv = nl & ~3
    r = rcpps(x2[:nv])
    inv = torch.cat([(1.0 - x2[:nv] * r) * r + r, 1.0 / x2[nv:]])
    assert torch.equal(e_fast, (inv * float(lit("109737.312"))).double())
    assert torch.equal(eh_fast, (inv * float(lit("109677.576"))).double())
    rel = ((e_fast - e_src).abs() / e_src).max()
    assert 0 < float(rel) < 1e-6                                       # rounding-level, but not identical
