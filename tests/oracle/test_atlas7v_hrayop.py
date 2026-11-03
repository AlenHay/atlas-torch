"""Oracle tests for HRAYOP and MAP1 (SYNTHE variant, atlas7v.for). Template for every routine port:
  1. golden: State from the solar trace entry -> port -> compare with trace exit (bitwise-ish)
  2. property: random inputs -> port vs the Fortran routine itself through the ctypes oracle
"""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.interp import map1
from atlas_torch.state import State
from atlas_torch.synthe.hrayop import hrayop
from atlas_torch.testing.compare import assert_close

STAGE = "sun_500_510/xnfpelsyn"


def test_hrayop_golden(golden):
    for hit in range(3):
        s = State.from_trace(golden(STAGE, "hrayop", hit, "entry"))
        exp = golden(STAGE, "hrayop", hit, "exit")
        out = hrayop(s)
        assert_close(out["bhyd.sigh"], exp["bhyd.sigh"], rtol=0, name="sigh")


@settings(max_examples=60, deadline=None)
@given(lf=st.floats(min_value=12.0, max_value=16.5))
def test_hrayop_vs_fortran(oracle, golden, lf):
    o = oracle("atlas7v")
    s = State.from_trace(golden(STAGE, "hrayop", 0, "entry"))
    for blk in ("bhyd", "state", "rhox"):
        o.set_common(blk, **{k.split(".")[1]: v.numpy() for k, v in s.items() if k.startswith(blk + ".")})
    freq = 10.0 ** lf
    o.set_common("freq", freq=freq)
    o.call("hrayop")
    s = s.updated({"freq.freq": torch.tensor([freq])})
    assert_close(hrayop(s)["bhyd.sigh"], o.common("bhyd")["sigh"], rtol=0, name=f"sigh@{freq:.6e}")


@settings(max_examples=100, deadline=None)
@given(n=st.integers(4, 40), m=st.integers(1, 60), seed=st.integers(0, 2**31 - 1))
def test_map1_vs_fortran(oracle, n, m, seed):
    rng = np.random.default_rng(seed)
    xold = np.asfortranarray(np.sort(rng.uniform(0, 10, n)))
    if np.any(np.diff(xold) == 0):
        return
    fold = np.asfortranarray(rng.normal(size=n))
    xnew = np.asfortranarray(np.sort(rng.uniform(-1, 11, m)))
    fnew = np.zeros(m, order="F")
    o = oracle("atlas7v")
    r, _ = o.call("map1", xold, fold, n, xnew, fnew, m, restype=__import__("ctypes").c_int)
    got, ll = map1(torch.from_numpy(xold), torch.from_numpy(fold), torch.from_numpy(xnew))
    assert_close(got, fnew, rtol=0, atol=0, name="map1")
    assert ll == r
