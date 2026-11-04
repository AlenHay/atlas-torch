"""ATLAS9 DATA tables and initial COMMON image vs the compiled library (bit-for-bit)."""
import numpy as np
import torch

from atlas_torch.atlas9 import data
from atlas_torch.atlas9.commons import initial_state, layout


def test_initial_state_matches_library_image(oracle):
    o = oracle("atlas9")
    s = initial_state()
    for blk, members in layout().items():
        c = o.common(blk)
        for name, shape, dt in members:
            a = s[f"{blk}.{name}"]
            a = a.numpy() if isinstance(a, torch.Tensor) else a
            e = c[name]
            assert a.shape == e.shape, (blk, name)
            if a.dtype.kind == "S":
                assert np.array_equal(a, e), (blk, name)
            else:
                assert np.array_equal(a.astype(e.dtype), e), (blk, name)


def test_named_tables(oracle):
    o = oracle("atlas9")
    jt = data.josh_tables()
    assert np.array_equal(jt["coefj"].ravel(order="F"), o.common("matxj")["cj"])
    assert np.array_equal(jt["coefh"].ravel(order="F"), o.common("matxh")["ch"])
    sj = o.saved("josh")
    for k in ("ch", "ck", "xtau8"):
        assert np.array_equal(jt[k], sj[k]), k
    lt, sl = data.linop_tables(), o.saved("linop")
    assert np.array_equal(lt["tabt"], sl["tabt"]) and np.array_equal(lt["tabp"], sl["tabp"])
    assert lt["tenlog"] == sl["tenlog"][0]
    e, se = data.expi_coeffs(), o.saved("expi")
    assert all(e[k] == se[k][0] for k in e)
    assert np.array_equal(data.readin_ifop1(), o.saved("readin")["ifop1"])
    assert np.array_equal(data.wavedf()[:, 1], o.common("biglit")["wavebig"])
