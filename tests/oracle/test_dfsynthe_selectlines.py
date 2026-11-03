"""SELECTLINES (dfsynthe.for) vs the golden trace of the debug executable (5012 K, solar): line counts, the
TABLOG/TABBOLT/IWLTAB tables and the last TAPE12 record, bit for bit (the debug executable binds expf/powf to
Intel libimf -> intel_flavor).  With the default float64-rounded transcendentals the counts agree too and the
values within a few ulps (< 1% of them differ).
"""
import numpy as np
import pytest

from atlas_torch.dfsynthe import odf_io
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines.f32 import Flavor
from atlas_torch.dfsynthe.lines.selectlines import select_consts, selectlines

pytestmark = pytest.mark.skipif(not (S.available() and S.DATA.exists() and S.STOCK.exists()),
                                reason="needs libdfsynthe.so, Intel runtime, line lists and the stock solar run")
STAGE = "odf_solar/dfsynthe"


def _bits(a):
    return np.asarray(a, np.float32).view(np.int32)


@pytest.fixture(scope="module")
def maxdat():
    return odf_io.read_xnfpdfmax(S.STOCK / "xnfpdfmax.dat", 21)


@pytest.mark.parametrize("flavor", ["intel", "default"])
def test_selectlines_golden(golden, maxdat, flavor):
    g = golden(STAGE, "selectlines", 0, "exit")
    fl = S.intel_flavor() if flavor == "intel" else Flavor("O0")
    c = select_consts(maxdat.teff, maxdat.wledge, fl)
    assert np.array_equal(c.iwltab, g["static:iwltab"][: maxdat.nedge])
    assert c.hckt == g["static:hckt"][0] and c.hkt == g["static:hkt"][0]
    t12, t19, counts = selectlines(S.DATA, maxdat.teff, maxdat.wledge, maxdat.xnfdopmax, fl)
    assert len(t12) == int(g["arg:n12"][0]) and len(t19) == int(g["arg:n19"][0])
    L = int(g["static:l"][0]) - 1                      # lines in the final (partial) TAPE12 record
    it, ft = g["tape1112.itape12"], g["tape1112.tape12"]
    n = len(t12)
    assert np.array_equal(it[0, :L], t12.nbuff[n - L:]) and np.array_equal(it[2, :L], t12.nelion[n - L:])
    assert it[2, L] == 0                               # terminator
    for k, name in ((1, "congf"), (3, "gamr"), (4, "gams"), (5, "gamw")):
        d = np.abs(_bits(ft[k, :L]).astype(np.int64) - _bits(getattr(t12, name)[n - L:]))
        if flavor == "intel":
            assert d.max() == 0, name
            assert np.array_equal(_bits(c.tablog), _bits(g["static:tablog"]))
            assert np.array_equal(_bits(c.tabbolt), _bits(g["static:tabbolt"]))
        else:                                          # chained expf/powf ulps (correctly rounded vs libimf)
            assert d.max() <= 8 and (d > 0).mean() < 0.01, name
