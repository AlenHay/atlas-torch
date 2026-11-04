"""Oracle tests for SUBROUTINE SOLVIT(A,N,B,IPIVOT) (atlas7v.for; identical in xnfdf.for): Gauss-Jordan
elimination with full pivoting.
  golden:   the traced SOLVIT hits are NMOLEC's first three Newton systems (layer 1). The trace only keeps
            A(1), B(1), IPIVOT(1) (A is declared A(1)), so the full systems are rebuilt by the NMOLEC port
            (hook) and checked against the traced entry elements; SOLVIT's results against the exit elements.
  property: random (badly scaled, sparse) matrices and NMOLEC-like systems vs the Fortran, bitwise.
"""
import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.pops import nmolec, solvit
from test_atlas7v_pfsaha import STAGE, note, solar_state


def test_solvit_golden(golden):
    systems = []

    def hook(j, it, deq, eq):
        if len(systems) < 3:
            systems.append((j, it, deq, eq))
    s = solar_state(golden)
    from atlas_torch.state import State
    s = s.updated({k: v for k, v in State.from_trace(golden(STAGE, "nmolec", 0, "entry")).items()})
    nmolec(s, hook=hook)
    for hit, (j, it, deq, eq) in enumerate(systems):
        e = golden(STAGE, "solvit", hit, "entry")
        x = golden(STAGE, "solvit", hit, "exit")
        assert (j, it) == (1, hit + 1)
        assert int(e["arg:n"][0]) == deq.shape[0]
        assert deq[0, 0] == e["arg:a"][0] and eq[0] == e["arg:b"][0]
        a, b, ipiv = solvit(deq, eq)
        assert a[0, 0] == x["arg:a"][0], hit
        assert b[0] == x["arg:b"][0], hit
        assert ipiv[0] == x["arg:ipivot"][0], hit


def _oracle(o, a, b):
    n = b.shape[0]
    fa, fb, piv = np.asfortranarray(a.copy()), b.copy(), np.zeros(n, np.int32)
    o.call("solvit", fa, n, fb, piv)
    return fa, fb, piv


@settings(max_examples=40, deadline=None)
@given(n=st.integers(1, 30), seed=st.integers(0, 2**31 - 1), zeros=st.floats(0.0, 0.6))
def test_solvit_vs_fortran(oracle, n, seed, zeros):
    o = oracle("atlas7v")
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) * 10.0 ** rng.uniform(-8, 8, (n, n))
    a[rng.random((n, n)) < zeros] = 0.0
    a += np.diag(10.0 ** rng.uniform(-3, 3, n))
    if n > 2:
        a[0, :] = a[1, :] * 2.0          # ties in |A| exercise the first-maximum pivot rule
        a[0, 0] += 1.0
    b = rng.normal(size=n) * 10.0 ** rng.uniform(-5, 5, n)
    fa, fb, piv = _oracle(o, a, b)
    ga, gb, gp = solvit(a, b)
    note("solvit", gb, fb)
    assert np.array_equal(gb, fb)
    assert np.array_equal(ga, fa)
    assert np.array_equal(gp, piv)


def test_solvit_nmolec_systems(golden, oracle):
    """Every 7th Newton system of the solar NMOLEC run, port vs Fortran SOLVIT."""
    o = oracle("atlas7v")
    systems = []
    k = [0]

    def hook(j, it, deq, eq):
        k[0] += 1
        if k[0] % 7 == 1:
            systems.append((deq, eq))
    from atlas_torch.state import State
    s = solar_state(golden).updated(State.from_trace(golden(STAGE, "nmolec", 0, "entry")))
    nmolec(s, hook=hook)
    for deq, eq in systems:
        fa, fb, piv = _oracle(o, deq, eq)
        ga, gb, gp = solvit(deq, eq)
        note("solvit", gb, fb)
        assert np.array_equal(gb, fb) and np.array_equal(ga, fa) and np.array_equal(gp, piv)
