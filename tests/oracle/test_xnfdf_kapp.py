"""Oracle tests for KAPP(N=1) and the continuum sources of xnfdf.for (HOP, H2PLOP, HMINOP, HRAYOP, HE1OP,
HE2OP, HEMIOP, HERAOP, COOLOP's C1OP/MG1OP/AL1OP/SI1OP/FE1OP, LUKEOP with N1OP/O1OP/MG2OP/SI2OP/CA2OP,
HOTOP, ELECOP, H2RAOP) and the scalar helpers XKARSAS, COULFF, SEATON.

1. golden: KAPP hits 0..2 of the traced solar run (model 1, the first three frequencies): ACONT at exit.
2. hypothesis: models of the solar grid (populations from the port, pushed into the oracle COMMONs),
   random frequency over the DFSYNTHE range (and every edge of continua.dat) -> Fortran KAPP vs port:
   ACONT, SIGMAC and every source array; XKARSAS/COULFF/SEATON on random arguments.
Max relative error observed: ~1e-15 (libm exp/log/pow ulps).
"""
import ctypes
import math

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.dfsynthe.xnf import oracle_support as osup
from atlas_torch.dfsynthe.xnf.kapp import Freq, coulff, kapp, seaton, xkarsas
from atlas_torch.dfsynthe.xnf.readin import Readin, read_edges
from atlas_torch.dfsynthe.xnfdf import group_state
from atlas_torch.testing.compare import assert_close

pytestmark = pytest.mark.skipif(not osup.available(), reason="libxnfdf.so or the Intel Fortran runtime missing")
RUN = osup.ext_dir() / "traces" / "odf_solar" / "xw"
STAGE = "odf_solar/xnfdf"
SEL = (0, 5, 14, 23, 33, 44, 56)          # T = 1995 K ... 200000 K
RTOL = 1e-13
SOURCES = {"bhyd": ("ahyd", "ah2p", "ahmin", "sigh", "sigh2"), "bhe": ("ahe1", "ahe2", "ahemin", "sighe"),
           "bc": ("ac1",), "bmg": ("amg1",), "bal": ("aal1",), "bsi": ("asi1",), "bfe": ("afe1",),
           "ops": ("aluke", "ahot", "sigel")}


@pytest.fixture(scope="module")
def grid():
    r = Readin((RUN / "stdin.txt").read_text(), (RUN / "fort.2").read_text())
    models = [r.read(1) for _ in range(57)]
    ms = [models[i] for i in SEL]
    g = group_state(ms)
    frq, wl, cm, fs = read_edges((RUN / "fort.17").read_text())
    return ms, g, fs


def test_kapp_golden(golden, grid):
    ms, g, fs = grid
    for hit in range(3):
        ex = golden(STAGE, "kapp", hit, "exit")
        freq = float(ex["freq.freq"].reshape(-1)[0])
        assert freq == fs[hit]
        acont, sigmac = kapp(freq, g["ki"], g["c"])
        assert_close(acont[0], ex["optot.acont"][:25], rtol=RTOL, name=f"acont nu={hit + 1}")


def _fortran_kapp(m, g, b, freq):
    o = osup.new_oracle()
    osup.push_kapp_state(o, m, g, b, itemp=3)
    osup.push_freq(o, freq, g["hkt"][b].numpy())
    o.call("kapp", 1, ctypes.c_int(0), ctypes.c_double(0.))
    return o


def _check(o, g, b, freq):
    acont, sigmac, parts = kapp(freq, g["ki"], g["c"], parts=True)
    ot = o.common("optot")
    for blk, names in SOURCES.items():
        c = o.common(blk)
        for k in names:
            assert_close(parts[k][b], c[k][:25], rtol=RTOL, atol=0, name=f"{k} freq={freq:.6e}")
    assert_close(acont[b], ot["acont"][:25], rtol=RTOL, name=f"acont freq={freq:.6e}")
    assert_close(sigmac[b], ot["sigmac"][:25], rtol=RTOL, name=f"sigmac freq={freq:.6e}")


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(b=st.integers(0, len(SEL) - 1), lf=st.floats(math.log10(1.4e12), math.log10(6.0e16)))
def test_kapp_vs_fortran(grid, b, lf):
    ms, g, fs = grid
    freq = 10 ** lf
    _check(_fortran_kapp(ms[b], g, b, freq), g, b, freq)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(b=st.integers(0, len(SEL) - 1), k=st.integers(0, 857))
def test_kapp_at_freqset_vs_fortran(grid, b, k):
    """The DFSYNTHE frequencies themselves (just below/above every edge: thresholds are exercised)."""
    ms, g, fs = grid
    freq = float(fs[k])
    _check(_fortran_kapp(ms[b], g, b, freq), g, b, freq)


@pytest.fixture(scope="module")
def ora():
    return osup.new_oracle()


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(lf=st.floats(12.0, 17.5), n=st.integers(1, 20), l=st.integers(0, 5), z2=st.sampled_from([1.0, 4.0, 9.0]))
def test_xkarsas_vs_fortran(ora, lf, n, l, z2):
    freq = 10 ** lf
    l = n if n > 6 else min(l, n)           # L>=N (shell total) or 0<=L<N (subshell) for N<=6
    ref, _ = ora.call("xkarsas", float(freq), float(z2), n, l, restype=ctypes.c_double)
    assert xkarsas(freq, z2, n, l) == ref


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(lf=st.floats(12.0, 17.0), t=st.lists(st.floats(1500., 300000.), min_size=6, max_size=6), nz=st.integers(1, 6))
def test_coulff_vs_fortran(ora, lf, t, nz):
    t = np.array(t)
    tlog = np.log(t)
    freq = 10 ** lf
    ora.set_common("temp", t=t, tlog=tlog)
    ora.set_common("freq", freq=freq, freqlg=math.log(freq))

    class Lay:
        pass
    lay = Lay()
    lay.tlog = torch.from_numpy(tlog)
    got = coulff(lay, math.log(freq), nz).numpy()
    ref = [ora.call("coulff", j + 1, nz, restype=ctypes.c_double)[0] for j in range(6)]
    assert_close(got, np.array(ref), rtol=1e-15, name="coulff")


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(f0=st.floats(1e15, 5e15), lf=st.floats(15.0, 17.0), xs=st.floats(1e-20, 1e-16), power=st.sampled_from(
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.5]), a=st.floats(-1.0, 8.0))
def test_seaton_vs_fortran(ora, f0, lf, xs, power, a):
    freq = 10 ** lf
    ora.set_common("freq", freq=freq)
    ref, _ = ora.call("seaton", float(f0), float(xs), float(power), float(a), restype=ctypes.c_double)
    assert seaton(f0, xs, power, a, freq) == ref
