"""Oracle tests (synthe.for): He I profiles (HE1PROF dispatch, HE4471/HE4026/HE4387/HE4921, GRIEM, DIMITRI,
READBCS with fort.18) and PARCOE/INTEG vs the -O0 library.

The oracle runs with this host's libm; its log10f may differ from glibc 2.39's (which the stock reference
runs used and the port reproduces by default), so these tests switch the port to the host log10f.
"""
import ctypes
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.lineopac import helium as HE
from atlas_torch.synthe.lineopac import libm32
from atlas_torch.synthe.lineopac.fortio import read_fort10
from atlas_torch.synthe.lineopac.tabs import h1tab_common

F32 = np.float32
EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
F10 = EXT / "work" / "sun_380_420_stock" / "pre_synthe" / "fort.10"
HE1 = EXT / "BasicATLAS" / "data" / "synthe_files" / "he1tables.dat"


def _bits(a):
    return np.asarray(a, F32).view(np.uint32)


@pytest.fixture(scope="module")
def setup(oracle, tmp_path_factory):
    for p in (F10, HE1):
        if not p.exists():
            pytest.skip(f"missing {p}")
    d = tmp_path_factory.mktemp("he")
    shutil.copy(HE1, d / "fort.18")
    o = oracle("synthe")
    o.call("tabvoigt", ctypes.c_float(200.0), 2001)
    f10 = read_fort10(F10)
    nr = f10["nrhox"]
    s = f10["state"]
    o.common("temp")["t"][:] = s["t"]
    o.set_common("state", xne=s["xne"])
    o.common("bhe")["xnfhe"][:, :] = s["xnfhe"].astype(F32)
    xph = np.zeros((99, 2), F32)
    xph[:nr] = f10["qxnfpel"][:, :2]
    o.common("bhyd")["xnfph"][:, :] = xph
    dop7 = np.sqrt(f10["qdopple"][:, 6] ** 2 + (1.5 / 299792.458) ** 2).astype(F32)
    lay = dict(t=s["t"][:nr].astype(F32), xne=s["xne"][:nr].astype(F32), xnfph2=xph[:nr, 1],
               xnfhe2=s["xnfhe"][:nr, 1].astype(F32))
    he = HE.HeI(lay, torch.as_tensor(h1tab_common("scalar")), str(d / "fort.18"), mode="scalar")
    return dict(o=o, he=he, nr=nr, dop7=dop7, dir=d)


@pytest.fixture(autouse=True)
def host_log10f(monkeypatch):
    monkeypatch.setattr(libm32, "LOG10F", "host")


def _waves(rng, wl4, k, span=0.6):
    return (wl4 + rng.uniform(-span, span, k)).astype(F32)


@pytest.mark.parametrize("name,wl", [("he4026", 402.6184), ("he4471", 447.1479), ("he4921", 492.1931),
                                     ("he4387", 438.7929)])
def test_bcs_lines(setup, monkeypatch, name, wl):
    o, he, nr, dop7 = setup["o"], setup["he"], setup["nr"], setup["dop7"]
    monkeypatch.chdir(setup["dir"])      # READBCS reads fort.18 from the working directory
    wl4 = F32(wl)
    assert he.kind(wl4) == name
    dopwl = (dop7 * wl4).astype(F32)
    lc = he.bcs_layer(name, wl4, dopwl)
    line = {"he4471": 1, "he4026": 2, "he4387": 3, "he4921": 4}[name]
    phil = torch.as_tensor(he.philam(line)) if name != "he4387" else None
    rng = np.random.default_rng(line)
    for j in range(nr):                   # one J per call sequence -> the J*LINE cache key never collides
        w = np.concatenate([[wl4], _waves(rng, wl4, 60)]).astype(F32)
        ref = [o.call(name, j + 1, ctypes.c_float(x), ctypes.c_float(wl4), ctypes.c_float(dopwl[j]),
                      restype=ctypes.c_float)[0] for x in w]
        jt = torch.full((len(w),), j, dtype=torch.long)
        pr = phil[jt] if phil is not None else None
        got = he.bcs_profile(name, jt, torch.from_numpy(w), wl4, torch.full((len(w),), float(dopwl[j]),
                             dtype=torch.float32), lc, pr).numpy()
        assert np.array_equal(_bits(got), _bits(ref)), (name, j)


@pytest.mark.parametrize("kind,wl", [("griem", 388.9750), ("griem", 501.5678), ("griem", 410.0),
                                     ("dimitri", 381.9624), ("dimitri", 414.3759), ("dimitri", 416.8972)])
def test_griem_dimitri(setup, kind, wl):
    o, he, nr, dop7 = setup["o"], setup["he"], setup["nr"], setup["dop7"]
    wl4 = F32(wl)
    dopwl = (dop7 * wl4).astype(F32)
    gr, gs = F32(1.5e-8), F32(3e-20)
    lc = (he.griem_layer if kind == "griem" else he.dimitri_layer)(wl4, dopwl, gr, gs)
    rng = np.random.default_rng(int(wl))
    j = rng.integers(0, nr, 300)
    w = _waves(rng, wl4, 300)
    ref = [o.call(kind, int(jj + 1), ctypes.c_float(x), ctypes.c_float(wl4), ctypes.c_float(dopwl[jj]),
                  ctypes.c_float(gr), ctypes.c_float(gs), restype=ctypes.c_float)[0] for jj, x in zip(j, w)]
    jt = torch.from_numpy(j)
    got = he.gd_profile(lc, jt, torch.from_numpy(w), wl4, torch.from_numpy(dopwl)[jt]).numpy()
    assert np.array_equal(_bits(got), _bits(ref))


@pytest.mark.parametrize("wl", [381.96, 386.75, 387.18, 388.97, 392.65, 393.59, 400.93, 401.04, 402.40, 402.62,
                                402.73, 414.38, 416.90, 438.79, 447.15, 492.19, 501.57, 587.56])
def test_he1prof_dispatch(setup, monkeypatch, wl):
    """HE1PROF(J,WAVE,WL,...) = the function chosen by HeI.kind, evaluated at the line centre."""
    o, he, nr, dop7 = setup["o"], setup["he"], setup["nr"], setup["dop7"]
    monkeypatch.chdir(setup["dir"])
    wl4 = F32(wl)
    kind = he.kind(wl4)
    dopwl = (dop7 * wl4).astype(F32)
    gr, gs = F32(1.5e-8), F32(3e-20)
    j = 30
    ref = o.call("he1prof", j + 1, ctypes.c_float(wl4 + F32(0.003)), ctypes.c_float(wl4), ctypes.c_float(dopwl[j]),
                 ctypes.c_float(gr), ctypes.c_float(gs), restype=ctypes.c_float)[0]
    jt = torch.tensor([j])
    w = torch.tensor([wl4 + F32(0.003)], dtype=torch.float32)
    if kind in ("griem", "dimitri"):
        lc = (he.griem_layer if kind == "griem" else he.dimitri_layer)(wl4, dopwl, gr, gs)
        got = he.gd_profile(lc, jt, w, wl4, torch.from_numpy(dopwl)[jt])
    else:
        lc = he.bcs_layer(kind, wl4, dopwl)
        line = {"he4471": 1, "he4026": 2, "he4387": 3, "he4921": 4}[kind]
        pr = torch.as_tensor(he.philam(line))[jt] if kind != "he4387" else None
        got = he.bcs_profile(kind, jt, w, wl4, torch.from_numpy(dopwl)[jt], lc, pr)
    assert _bits(got.numpy())[0] == _bits([ref])[0], kind


@settings(max_examples=40, deadline=None)
@given(n=st.integers(4, 220), seed=st.integers(0, 2**31 - 1))
def test_integ_parcoe(oracle, n, seed):
    o = oracle("synthe")
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-150, 150, n)).astype(F32)
    if np.any(np.diff(x) == 0):
        return
    fv = (10 ** rng.uniform(-3, 0, n)).astype(F32)
    fint = ctypes.c_float(0.0)
    o.call("integ", np.asfortranarray(x), np.asfortranarray(fv), fint, n, ctypes.c_float(0.0))
    got = HE.integ(x, fv[None, :], 0.0)[0]
    assert _bits([got])[0] == _bits([fint.value])[0]
    a = np.zeros(n, F32); b = np.zeros(n, F32); c = np.zeros(n, F32)
    o.call("parcoe", np.asfortranarray(fv), np.asfortranarray(x), a, b, c, n)
    A, B, C = HE.parcoe(fv[None, :], x)
    for u, v in ((A[0], a), (B[0], b), (C[0], c)):
        assert np.array_equal(_bits(u), _bits(v))
