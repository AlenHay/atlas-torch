"""XLINOP (dfsynthe.for) vs the Fortran.

1. oracle: the library's own XLINOP (in a spawned process: the Intel runtime keeps unit 19 connected
   process-wide) reads a TAPE19 (fort.19) of chosen NLTE lines for layers J=1,2,3 in turn
   (SAVE state, e.g. SOFBET's P cache, carried between the calls as in the program); the port processes the same
   lines for the same layers; buffers must be bit-identical.  Subsets cover every line type (normal with/without
   continuum-merging tails, merged continua, autoionizing, hydrogen alpha/beta/general paths with HPROF4m/p).
2. golden: the full solar TAPE19 for all 25 layers vs the traced XLINOP-exit buffers of layers 1-3 of the debug
   executable (Intel libimf -> intel_flavor), bit for bit.
"""
import os

import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe import odf_io
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines import tables as T
from atlas_torch.dfsynthe.lines.continuum import continuum, layer_setup
from atlas_torch.dfsynthe.lines.selectlines import selectlines
from atlas_torch.dfsynthe.lines.xlinop import XLinop

pytestmark = pytest.mark.skipif(not (S.available() and S.DATA.exists() and S.STOCK.exists()),
                                reason="needs libdfsynthe.so, Intel runtime, line lists and the stock solar run")
STAGE = "odf_solar/dfsynthe"
NL = 3
SCR = "/tmp/claude-1000" if os.path.isdir("/tmp/claude-1000") else None


@pytest.fixture(scope="module")
def solar():
    fl = S.oracle_flavor()
    xt = odf_io.read_xnfpdf(S.STOCK / "xnfpdf.dat", 21)
    mx = odf_io.read_xnfpdfmax(S.STOCK / "xnfpdfmax.dat", 21)
    _, t19, _ = selectlines(S.DATA, mx.teff, mx.wledge, mx.xnfdopmax, fl)
    lay = layer_setup(xt, fl, torch.device("cpu"))
    cont = continuum(xt.wledge, lay["ablog"][:NL], fl)
    return fl, xt, t19, lay, cont


def _subset(t19, kind):
    typ = t19["type"]
    h = np.nonzero(typ == -1)[0]
    pick = {
        "normal": np.nonzero(typ == 0)[0][:300],
        "merged_auto": np.nonzero((typ > 3) | (typ == 1))[0],
        "h_alpha_beta": h[(t19["nbup"][h] - t19["nblo"][h] <= 2) | (t19["ncon"][h] == 0)][:40],
        "h_general": h[(t19["nbup"][h] - t19["nblo"][h] > 2) & (t19["ncon"][h] > 0)][::12],
        "he_edge": np.arange(10, 40),
    }[kind]
    return np.sort(pick)


def _oracle_xlinop(xt, lay, cont, sub, fl):
    return S.oracle_xlinop(dict(state=xt.state, txnxn=lay["txnxn"].numpy(), xnfpel=lay["xnfpel"].numpy(),
                                dopple=lay["dopple"].numpy(), xnfdop=lay["xnfdop"].numpy(),
                                xnfph=lay["xnfph"].numpy(), cont=cont.numpy(), extab=T.build_tables(fl).extab,
                                tape19=sub, scratch=SCR))


@pytest.mark.stage   # module fixture runs SELECTLINES+CONTINUUM with the ctypes-libm flavor (~12 s); 2-4 s per case (oracle in a spawned process)
@pytest.mark.parametrize("kind", ["normal", "merged_auto", "he_edge", "h_alpha_beta", "h_general"])
def test_xlinop_vs_fortran(solar, kind):
    fl, xt, t19, lay, cont = solar
    sub = t19[_subset(t19, kind)]
    ref = _oracle_xlinop(xt, lay, cont, sub, fl)
    tabs = T.device_tables(torch.device("cpu"), fl)
    L = {k: (v[:NL] if torch.is_tensor(v) else v) for k, v in lay.items()}
    L["st"] = {k: v[:NL] for k, v in lay["st"].items()}
    xl = XLinop(tabs, L, cont, fl)
    buf = torch.zeros(NL * T.LENBUFF, dtype=torch.float32)
    xl.run(buf, sub)
    got = buf.view(NL, -1).numpy()
    for j in range(NL):
        assert (ref[j] != 0).sum() > 0
        assert np.array_equal(got[j].view(np.int32), ref[j].view(np.int32)), (kind, j + 1)


@pytest.mark.stage
def test_xlinop_golden(golden):
    fl = S.intel_flavor()
    xt = odf_io.read_xnfpdf(S.STOCK / "xnfpdf.dat", 21)
    mx = odf_io.read_xnfpdfmax(S.STOCK / "xnfpdfmax.dat", 21)
    _, t19, _ = selectlines(S.DATA, mx.teff, mx.wledge, mx.xnfdopmax, fl)
    lay = layer_setup(xt, fl, torch.device("cpu"))
    cont = continuum(xt.wledge, lay["ablog"], fl)
    xl = XLinop(T.device_tables(torch.device("cpu"), fl), lay, cont, fl)
    buf = torch.zeros(25 * T.LENBUFF, dtype=torch.float32)
    xl.run(buf, t19)
    b = buf.view(25, -1).numpy()
    for hit in range(3):
        g = golden(STAGE, "xlinop", hit, "exit")
        j = int(g["arg:j"][0]) - 1
        assert np.array_equal(b[j].view(np.int32), g["buffer.buffer"].view(np.int32)), j + 1


@pytest.mark.parametrize("kind", ["normal", "merged_auto"])
def test_xlinop_batching_modes_identical(solar, kind):
    """Weak-line batches, tentative strong-line batches (budget overflow -> per-line streaming) and one line per
    batch give bit-identical buffers (per-bin REAL*4 order is the line order in every mode)."""
    fl, xt, t19, lay, cont = solar
    sub = t19[_subset(t19, kind)]
    tabs = T.device_tables(torch.device("cpu"), fl)
    L = {k: (v[:NL] if torch.is_tensor(v) else v) for k, v in lay.items()}
    L["st"] = {k: v[:NL] for k, v in lay["st"].items()}
    out = []
    for budget, chunk in ((None, 200), (1, 200), (None, 1)):
        xl = XLinop(tabs, L, cont, fl)
        if budget is not None:
            xl.strong_budget = budget
        buf = torch.zeros(NL * T.LENBUFF, dtype=torch.float32)
        xl.run(buf, sub, chunk_lines=chunk)
        out.append(buf)
    assert bool((out[0] != 0).any())
    for b in out[1:]:
        assert torch.equal(out[0].view(torch.int32), b.view(torch.int32))
