"""Oracle tests for SUBROUTINE LINTER(XOLD,YOLD,NOLD,XNEW,YNEW,NNEW) (atlas7v.for): linear interpolation /
extrapolation with a forward-only walk. Golden: the traced HMINOP calls; property: random tables."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.hydrogen import linter
from atlas_torch.testing.compare import assert_close


def _fortran(o, xold, yold, xnew):
    ynew = np.zeros(len(xnew), order="F")
    o.call("linter", np.asfortranarray(xold), np.asfortranarray(yold), len(xold), np.asfortranarray(xnew), ynew,
           len(xnew))
    return ynew


def test_linter_golden(golden, oracle):
    """The traced calls are HMINOP's (WFFLOG, FFLOG(1,ITHETA), 22, WAVELOG); the trace holds the first elements
    of the arrays only, so the tables come from HMINOP's DATA (as HMINOP builds them)."""
    from atlas_torch.synthe.opacity.hydrogen import _hminop_tables
    t = _hminop_tables()
    o = oracle("atlas7v")
    for hit in range(3):
        d, e = golden(T.STAGE, "linter", hit, "entry"), golden(T.STAGE, "linter", hit, "exit")
        x = float(d["arg:xnew"][0])
        assert float(d["arg:xold"][0]) == float(t["wfflog"][0])
        y = linter(t["wfflog"], t["fflog"][:, hit].reshape(1, -1), torch.tensor([[x]]))
        assert_close(y.reshape(1), e["arg:ynew"], rtol=0.0, name=f"linter[{hit}]")
        assert_close(y.reshape(1), _fortran(o, t["wfflog"].numpy(), t["fflog"][:, hit].numpy(), [x]), rtol=0.0)


@settings(max_examples=150, deadline=None)
@given(k=st.integers(2, 30), m=st.integers(1, 40), seed=st.integers(0, 2**32 - 1), sorted_new=st.booleans())
def test_linter_vs_fortran(oracle, k, m, seed, sorted_new):
    rng = np.random.default_rng(seed)
    xold = np.sort(rng.uniform(-5, 5, k))
    if np.any(np.diff(xold) == 0):
        return
    yold = rng.normal(size=k)
    xnew = rng.uniform(-7, 7, m)
    o = oracle("atlas7v")
    if sorted_new:          # one NNEW=m call == m independent calls when XNEW increases
        xnew = np.sort(xnew)
        ref = _fortran(o, xold, yold, xnew)
    else:                   # NNEW=1 calls (the way HMINOP calls it)
        ref = np.array([_fortran(o, xold, yold, [v])[0] for v in xnew])
    got = linter(torch.from_numpy(xold), torch.from_numpy(yold).reshape(1, -1), torch.from_numpy(xnew).reshape(1, -1))
    assert_close(got.reshape(-1), ref, rtol=0.0, name="linter")
