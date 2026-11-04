"""Oracle tests for FUNCTION PFGROUND(NELION,T) (atlas7v.for): ground-term partition functions, a computed
GO TO over closed-form expressions read from the source with their exact literal kinds (D0 constants here,
unlike the atlas9mem copy). The IF(NELION.EQ.431/491/501) branches are dead code in this variant (they sit
behind an always-taken computed GO TO), so O I / F I / F II fall through to PFGROUND=1.
  golden:   the traced PFGROUND hits (the trace stores the arguments only, so the value is checked against
            the Fortran function on the traced arguments)
  property: every NELION 1..600 on random temperatures 1000..200000 K vs the Fortran, bitwise.
"""
import ctypes

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.pfsaha import pfground
from atlas_torch.testing.compare import assert_close
from test_atlas7v_pfsaha import STAGE, note


def _oracle(o, nelion, t):
    r, _ = o.call("pfground", int(nelion), float(t), restype=ctypes.c_double)
    return r


def test_pfground_golden(golden, oracle):
    o = oracle("atlas7v")
    for hit in range(3):
        e = golden(STAGE, "pfground", hit, "entry")
        nel, t = int(e["arg:nelion"][0]), float(e["arg:t"][0])
        got = pfground(nel, torch.tensor([t]))
        assert_close(got, [_oracle(o, nel, t)], rtol=0, name=f"pfground hit {hit}")


def test_pfground_dead_branches(oracle):
    o = oracle("atlas7v")
    for nel in (114, 431, 491, 501, 150, 337, 600):
        assert pfground(nel, 3000.0) == 1.0 == _oracle(o, nel, 3000.0)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_pfground_vs_fortran(oracle, seed):
    o = oracle("atlas7v")
    rng = np.random.default_rng(seed)
    t = 10.0 ** rng.uniform(3.0, 5.3, 8)
    tt = torch.from_numpy(t)
    for nel in range(1, 601):
        got = pfground(nel, tt)
        exp = np.array([_oracle(o, nel, x) for x in t])
        note("pfground", got, exp)
        assert_close(got, exp, rtol=0, name=f"pfground {nel}")
