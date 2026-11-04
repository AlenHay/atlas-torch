"""Oracle tests for FUNCTION COULFF(J,NZ) (atlas7v.for): free-free Gaunt factor table interpolation, for all
layers J and charges NZ=1..6, on golden and perturbed states."""
import ctypes

import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity._core import Ctx
from atlas_torch.synthe.opacity.hydrogen import coulff_k
from atlas_torch.testing.compare import assert_close


def _cmp(oracle, s, nz, name):
    o = T.run_oracle(oracle, s)
    n = s.iscalar("rhox.nrhox")
    ref = np.array([o.call("coulff", j, nz, restype=ctypes.c_double)[0] for j in range(1, n + 1)])
    assert_close(coulff_k(Ctx.from_state(s), nz)[0], ref, rtol=T.RTOL, name=name)


def test_coulff_golden(golden, oracle):
    for hit in range(3):
        d = golden(T.STAGE, "coulff", hit, "entry")
        s, _, _ = T.golden_state(golden, "coulff", hit)
        _cmp(oracle, T.base_state(golden).updated({k: v for k, v in s.items() if k.startswith(("freq.", "temp."))}),
             int(d["arg:nz"][0]), f"coulff[{hit}]")


@settings(max_examples=60, deadline=None)
@given(lf=st.floats(12.0, 17.0), seed=st.integers(0, 2**32 - 1), nz=st.integers(1, 6),
       tscale=st.sampled_from([(0.8, 1.25), (0.2, 0.5), (2.0, 8.0)]))
def test_coulff_vs_fortran(oracle, golden, lf, seed, nz, tscale):
    _cmp(oracle, T.random_case(golden, seed, 10.0**lf, tscale=tscale), nz, f"coulff@{10**lf:.4e},{nz}")
