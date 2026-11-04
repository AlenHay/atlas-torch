"""XKARSAS (REAL*4 Karzas-Latter cross-sections), COULBF1S, COULX, COULFF, SEATON vs the Fortran (bitwise)."""
import ctypes

import numpy as np
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics import opacity as O

D = ctypes.c_double


@settings(max_examples=400, deadline=None)
@given(lf=st.floats(12.5, 17.5), z=st.floats(0.3, 30.0), n=st.integers(1, 20), l=st.integers(0, 6))
def test_xkarsas_vs_fortran(lf, z, n, l):
    if n > 6 or l >= n:
        l = n if (l >= n or n > 6) else l
    f = 10.0 ** lf
    assert O.xkarsas(f, z, n, l) == T.make_oracle("atlas9").call("xkarsas", f, z, n, l, restype=D)[0]


@settings(max_examples=200, deadline=None)
@given(lf=st.floats(14.0, 17.5), z=st.floats(1.0, 3.0), n=st.integers(1, 9))
def test_coulx_coulbf1s_vs_fortran(lf, z, n):
    o = T.make_oracle("atlas9")
    f = 10.0 ** lf
    assert O.coulbf1s(f, z) == o.call("coulbf1s", f, z, restype=D)[0]
    assert O.coulx(n, f, z) == o.call("coulx", n, f, z, restype=D)[0]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 2**31 - 1), lf=st.floats(12.5, 16.5), nz=st.integers(1, 5))
def test_coulff_vs_fortran(golden, seed, lf, nz):
    s = T.perturbed_state(T.opacity_state(golden, "hop", 0), seed, lf)
    o = T.make_oracle("atlas9")
    T.load_state(o, s)
    n = s.iscalar("rhox.nrhox")
    got = O.coulff(s, O.Freq.from_state(s, n), n, nz).numpy()
    exp = np.array([o.call("coulff", j, nz, restype=D)[0] for j in range(1, n + 1)])
    assert T.max_rel(got, exp) == 0.0


@settings(max_examples=200, deadline=None)
@given(lf=st.floats(14.0, 17.0), lf0=st.floats(14.0, 16.0), p=st.floats(0.5, 4.0), a=st.floats(0.5, 7.0))
def test_seaton_vs_fortran(lf, lf0, p, a):
    o = T.make_oracle("atlas9")
    f, f0 = 10.0 ** lf, 10.0 ** lf0
    o.set_common("freq", freq=f)
    assert O.seaton(f, f0, 1e-18, p, a) == o.call("seaton", f0, 1e-18, p, a, restype=D)[0]
