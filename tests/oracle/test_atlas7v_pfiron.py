"""Oracle tests for PFIRON (atlas7v.for; identical source in atlas9mem.for, shared tensor port):
REAL*4 interpolation of the iron-group partition-function table in log T and the ionisation-potential
lowering.
  golden:   the traced PFIRON hits of the solar xnfpelsyn run (args at entry -> PF at exit)
  property: random NELEM 20..28, ION 1..10, log T in [3.2, 5.6], lowering 10..1e5 cm^-1 vs the Fortran.
"""
import ctypes

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.pfsaha import pfiron
from atlas_torch.testing.compare import assert_close
from test_atlas7v_pfsaha import STAGE, note


def _oracle_pf(o, nelem, ion, tlog8, potlow8):
    pf = ctypes.c_double(0.0)
    o.call("pfiron", int(nelem), int(ion), float(tlog8), float(potlow8), pf)
    return pf.value


def test_pfiron_golden(golden):
    for hit in range(3):
        e = golden(STAGE, "pfiron", hit, "entry")
        x = golden(STAGE, "pfiron", hit, "exit")
        got = pfiron(int(e["arg:nelem"][0]), int(e["arg:ion"][0]), torch.tensor(e["arg:tlog8"]),
                     torch.tensor(e["arg:potlow8"]))
        note("pfiron", got, x["arg:pf"])
        assert_close(got, x["arg:pf"], rtol=0, name=f"pfiron hit {hit}")


@settings(max_examples=60, deadline=None)
@given(nelem=st.integers(20, 28), ion=st.integers(1, 10), seed=st.integers(0, 2**31 - 1))
def test_pfiron_vs_fortran(oracle, nelem, ion, seed):
    o = oracle("atlas7v")
    rng = np.random.default_rng(seed)
    tl = rng.uniform(3.2, 5.6, 40)
    pl = 10.0 ** rng.uniform(1.0, 5.0, 40)
    tl[:3] = (3.7, 4.0, 3.32)                     # branch boundaries of the T interpolation
    pl[:3] = (500.0, 1000.0, 32000.0)             # lowering-table nodes
    got = pfiron(nelem, ion, torch.from_numpy(tl), torch.from_numpy(pl))
    exp = np.array([_oracle_pf(o, nelem, ion, a, b) for a, b in zip(tl, pl)])
    note("pfiron", got, exp)
    assert_close(got, exp, rtol=0, name=f"pfiron {nelem} {ion}")
