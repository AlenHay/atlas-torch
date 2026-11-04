"""Oracle tests (synthe.for): hydrogen profiles HPROF4 (+ SOFBET, VCSE1F, HFNM) vs the -O0 library.

Layer data: the solar model of the 380-420 nm reference run (fort.10); HPROF4's depth vectors are set up
by the routine itself (ITEMP=1), the port uses mode="scalar" (the -O0 library calls scalar powf).
"""
import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.lineopac import hydrogen as H
from atlas_torch.synthe.lineopac.fortio import read_fort10
from atlas_torch.synthe.lineopac.tabs import extab_common

F32 = np.float32
EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
F10 = EXT / "work" / "sun_380_420_stock" / "pre_synthe" / "fort.10"


def _bits(a):
    return np.asarray(a, F32).view(np.uint32)


@pytest.fixture(scope="module")
def setup(oracle):
    if not F10.exists():
        pytest.skip(f"missing {F10}")
    o = oracle("synthe")
    f10 = read_fort10(F10)
    nr = f10["nrhox"]
    st_ = f10["state"]
    ext = extab_common("scalar")
    e = o.common("extab")
    e["extab"][:] = ext[:1001]
    e["extabf"][:] = ext[1001:2002]
    e["e1tab"][:] = ext[2002:]
    tt = o.common("temp")
    tt["t"][:] = st_["t"]
    tt["itemp"][0] = 1
    o.set_common("state", xne=st_["xne"])
    o.set_common("rhox", nrhox=nr)
    o.common("bhe")["xnfhe"][:, :] = st_["xnfhe"].astype(F32)
    o.set_common("txnxn", xnfh2=st_["xnfh2"])
    xph = np.zeros((99, 2), F32)
    xph[:nr, 0] = f10["qxnfpel"][:, 0]
    xph[:nr, 1] = f10["qxnfpel"][:, 1]
    o.common("bhyd")["xnfph"][:, :] = xph
    dopph = np.zeros(99, F32)
    dopph[:nr] = np.sqrt(f10["qdopple"][:, 0] ** 2 + (1.5 / 299792.458) ** 2)
    stt = dict(t=torch.from_numpy(st_["t"][:nr].astype(F32)), xne=torch.from_numpy(st_["xne"][:nr].astype(F32)),
               xnfhe1=torch.from_numpy(st_["xnfhe"][:nr, 0].astype(F32)),
               xnfh2=torch.from_numpy(st_["xnfh2"][:nr].astype(F32)),
               xnfph1=torch.from_numpy(xph[:nr, 0]), xnfph2=torch.from_numpy(xph[:nr, 1]))
    hp = H.HProf4(stt, torch.from_numpy(ext), mode="scalar")
    return o, hp, dopph, nr, ext


def test_hfnm(setup):
    o = setup[0]
    for n in range(1, 9):
        for m in range(1, 101):
            assert F32(o.call("hfnm", n, m, restype=ctypes.c_float)[0]) == H.hfnm(n, m), (n, m)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_vcse1f(setup, seed):
    o, ext = setup[0], setup[4]
    rng = np.random.default_rng(seed)
    x = np.concatenate([10 ** rng.uniform(-5, 1.7, 300), [0, -1, .01, 1, 30, 31]]).astype(F32)
    ref = [o.call("vcse1f", ctypes.c_float(t), restype=ctypes.c_float)[0] for t in x]
    assert np.array_equal(_bits(H.vcse1f(torch.from_numpy(x), torch.from_numpy(ext)).numpy()), _bits(ref))


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), nm=st.sampled_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (2, 10), (4, 9), (5, 7)]))
def test_sofbet(setup, seed, nm):
    o = setup[0]
    n, m = nm
    rng = np.random.default_rng(seed)
    b = np.concatenate([10 ** rng.uniform(-2, 3.3, 300), [1, 25.12, 500, 8, 10]]).astype(F32)
    p = rng.uniform(0, 1.3, len(b)).astype(F32)
    ref = [o.call("sofbet", ctypes.c_float(x), ctypes.c_float(y), n, m, restype=ctypes.c_float)[0] for x, y in zip(b, p)]
    got = H.Sofbet("cpu")(torch.from_numpy(b), torch.from_numpy(p), n, m).numpy()
    assert np.array_equal(_bits(got), _bits(ref))


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**31 - 1),
       nm=st.sampled_from([(2, 6), (2, 8), (2, 12), (2, 30), (2, 99), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),
                           (3, 4), (3, 10), (4, 5), (5, 9)]))
def test_hprof4(setup, seed, nm):
    o, hp, dopph, nr, _ = setup
    n, m = nm
    rng = np.random.default_rng(seed)
    k = 150
    j = rng.integers(0, nr, k)
    dw = np.sign(rng.uniform(-1, 1, k)) * 10 ** rng.uniform(-6, 1.7, k)
    ref = [o.call("hprof4", n, m, int(jj + 1), float(d), dopph, restype=ctypes.c_float)[0] for jj, d in zip(j, dw)]
    got = hp(n, m, torch.from_numpy(j), torch.from_numpy(dw), torch.from_numpy(dopph)).numpy()
    bad = _bits(got) != _bits(ref)
    assert not bad.any(), (n, m, j[bad][:3], dw[bad][:3], np.asarray(ref)[bad][:3], got[bad][:3])
