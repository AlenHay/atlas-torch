"""Oracle tests (synthe.for): TABVOIGT/MAP1 tables, EXPI, VOIGT, FASTEX tables vs the -O0 library.

The oracle library is built -O0, so it calls the scalar libm everywhere: the port is run with
mode="scalar" here (mode="stock" reproduces the -O3 binary's libmvec loops; covered by the stage test).
"""
import ctypes

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.lineopac.tabs import expi1, extab_common, h1tab_common, map1_f32
from atlas_torch.synthe.lineopac.voigt import voigt

F32 = np.float32


def _bits_equal(a, b):
    a = np.asarray(a, F32)
    b = np.asarray(b, F32)
    return np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.fixture(scope="module")
def syn(oracle):
    o = oracle("synthe")
    o.call("tabvoigt", ctypes.c_float(200.0), 2001)
    return o


def test_tabvoigt(syn):
    h = h1tab_common("scalar")
    c = syn.common("h1tab")
    assert _bits_equal(h[:2001], c["h0tab"])
    assert _bits_equal(h[2001:4002], c["h1tab"])
    assert _bits_equal(h[4002:], c["h2tab"])


def test_expi_e1tab(syn):
    x = np.arange(1, 2001, dtype=F32) * F32(.01)
    ref = [syn.call("expi", 1, ctypes.c_float(v), restype=ctypes.c_float)[0] for v in x]
    assert _bits_equal(expi1(x), ref)
    # EXTAB/EXTABF (scalar expf) are plain expf of the table arguments
    e = extab_common("scalar")
    assert _bits_equal(e[2002:], ref)


@settings(max_examples=50, deadline=None)
@given(n=st.integers(4, 60), m=st.integers(1, 80), seed=st.integers(0, 2**31 - 1))
def test_map1_f32(syn, n, m, seed):
    rng = np.random.default_rng(seed)
    xold = np.sort(rng.uniform(0, 10, n)).astype(F32)
    if np.any(np.diff(xold) == 0):
        return
    fold = rng.normal(size=n).astype(F32)
    xnew = np.sort(rng.uniform(-1, 11, m)).astype(F32)
    fnew = np.zeros(m, F32)
    syn.call("map1", np.asfortranarray(xold), np.asfortranarray(fold), n, np.asfortranarray(xnew), fnew, m)
    assert _bits_equal(map1_f32(xold, fold, xnew), fnew)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_voigt(syn, seed):
    rng = np.random.default_rng(seed)
    n = 400
    v = np.concatenate([rng.uniform(0, 12, n), rng.uniform(0, 3.5, n), rng.uniform(0, 100, n)]).astype(F32)
    a = np.concatenate([10 ** rng.uniform(-5, 2.5, n), rng.uniform(0.15, 1.5, n), 10 ** rng.uniform(-3, 3, n)]).astype(F32)
    ref = [syn.call("voigt", ctypes.c_float(x), ctypes.c_float(y), restype=ctypes.c_float)[0] for x, y in zip(v, a)]
    h = torch.as_tensor(h1tab_common("scalar"))
    assert _bits_equal(voigt(torch.from_numpy(v), torch.from_numpy(a), h).numpy(), ref)
