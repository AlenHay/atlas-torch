"""DFSYNTHE XLINOP hydrogen lines: the C transcription (lines/native/hline.c, XLinop._h_native) == the torch path
(XLinop._h_wing / Hydrogen), bit for bit, for the stock flavour with and without the fast-math forms.

1. line subsets (alpha/beta and general paths incl. HPROF4m/p and the SOFBET carry-in iteration), 3 layers;
2. stage: one hot temperature, all 25 layers, the whole TAPE19 (buffers and the hydrogen sample counts);
3. stage: the kernel's LOG/EXP/10.** of REAL*4 arguments ((float)glibc((double)x)) equal the torch port's
   torch.f(x.double()).float() on a dense sample of all float32 bit patterns, with and without FTZ/DAZ
   (an exhaustive run over all 2**32 patterns found no difference either: tools/bench notes in the M7 report).
"""
import ctypes

import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe import odf_io
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines import tables as T
from atlas_torch.dfsynthe.lines.continuum import continuum, layer_setup
from atlas_torch.dfsynthe.lines.f32 import Flavor
from atlas_torch.dfsynthe.lines.native import load_extra
from atlas_torch.dfsynthe.lines.selectlines import select_tape19
from atlas_torch.dfsynthe.lines.xlinop import XLinop

pytestmark = pytest.mark.skipif(not (S.DATA.exists() and S.STOCK.exists()) or load_extra("hline") is None,
                                reason="needs the line lists, the stock solar XNFDF run and `make native`")
NL = 3


def _setup(it, fastmath, nl=None):
    from atlas_torch.dfsynthe.ftz import scope
    fl = Flavor("stock", fastmath=fastmath)
    xt = odf_io.read_xnfpdf(S.STOCK / "xnfpdf.dat", it)
    mx = odf_io.read_xnfpdfmax(S.STOCK / "xnfpdfmax.dat", it)
    with scope(True):
        t19 = select_tape19(S.DATA, mx.teff, mx.wledge, mx.xnfdopmax, fl)
        lay = layer_setup(xt, fl, torch.device("cpu"))
        cont = continuum(xt.wledge, lay["ablog"], fl)
    if nl is not None:
        L = {k: (v[:nl] if torch.is_tensor(v) else v) for k, v in lay.items()}
        L["st"] = {k: v[:nl] for k, v in lay["st"].items()}
        lay, cont = L, cont[:nl].contiguous()
    return fl, t19, lay, cont


def _run(fl, t19, lay, cont, native):
    from atlas_torch.dfsynthe.ftz import scope
    tabs = T.device_tables(torch.device("cpu"), fl)
    xl = XLinop(tabs, lay, cont, fl)
    xl.use_native_h = native
    buf = torch.zeros(cont.shape[0] * T.LENBUFF, dtype=torch.float32)
    with scope(True):
        st = xl.run(buf, t19)
    return buf, dict(st), xl.hyd.extab_oob


@pytest.mark.parametrize("fastmath", [True, False])
@pytest.mark.parametrize("kind", ["alpha_beta", "general"])
def test_h_native_equals_torch_subset(kind, fastmath):
    fl, t19, lay, cont = _setup(21, fastmath, NL)
    typ = t19["type"]
    h = np.nonzero((typ == -1) | (typ == -2))[0]
    d = t19["nbup"][h] - t19["nblo"][h]
    sel = h[(d <= 2) | (t19["ncon"][h] == 0)][:60] if kind == "alpha_beta" else h[(d > 2) & (t19["ncon"][h] > 0)][::6]
    sub = t19[np.sort(sel)]
    a, sa, _ = _run(fl, sub, lay, cont, True)
    b, sb, _ = _run(fl, sub, lay, cont, False)
    assert bool((a != 0).any())
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    assert sa["h_samples"] == sb["h_samples"] and sa["contribs"] == sb["contribs"]


@pytest.mark.stage
def test_h_native_equals_torch_full_temperature():
    """22387 K (index 38): all layers, the whole TAPE19 (hydrogen wings dominate there)."""
    fl, t19, lay, cont = _setup(38, True)
    a, sa, _ = _run(fl, t19, lay, cont, True)
    b, sb, _ = _run(fl, t19, lay, cont, False)
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    assert sa["h_samples"] == sb["h_samples"] and sa["contribs"] == sb["contribs"]
    assert sa["h_reruns"] == sb["h_reruns"]


@pytest.mark.stage
@pytest.mark.parametrize("ftz", [False, True])
def test_transcendentals_equal_torch(ftz):
    from atlas_torch.dfsynthe.ftz import scope
    lib = load_extra("hline")
    rng = np.random.default_rng(7)
    bits = np.concatenate([np.arange(0, 1 << 32, 97, dtype=np.uint64),
                           rng.integers(0, 1 << 32, 4_000_000, dtype=np.uint64)]).astype(np.uint32)
    x = bits.view(np.float32)
    with scope(ftz):
        for which, f in ((0, torch.log), (1, torch.exp), (2, lambda t: torch.pow(torch.full_like(t, 10.0), t))):
            out = np.empty_like(x)
            lib.hl_transcendental(ctypes.c_int(which), x.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(x.size),
                                  out.ctypes.data_as(ctypes.c_void_p))
            ref = f(torch.from_numpy(x).double()).float().numpy()
            same = (out.view(np.uint32) == ref.view(np.uint32)) | (np.isnan(out) & np.isnan(ref))
            assert same.all(), (which, x[~same][:5])
