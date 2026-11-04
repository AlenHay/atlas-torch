"""Oracle tests for LUKEOP (atlas7v.for): (N1OP n_N + O1OP n_O + MG2OP n_Mg+ + SI2OP n_Si+ + CA2OP n_Ca+)
* STIM/RHO. POPS is not ported here: the populations are LUKEOP's SAVEd arrays (ITEMP1 = ITEMP), or come
from a `pops` callback when ITEMP changes (checked against the cached path and the Fortran)."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity.luke import LUKE_POPS, LUKE_SUBS, lukeop

KEYS = ["ops.aluke", "save.lukeop.itemp1"] + [f"save.lukeop.{k}" for k, _ in LUKE_POPS] + \
    [f"save.{u}.{v}" for u, vs in LUKE_SUBS.items() for v in ("itemp1",) + vs]
POPS = {k: ((99,), 1e11) for k, _ in LUKE_POPS}


def test_lukeop_golden(golden):
    T.golden_check(golden, "lukeop", lukeop, KEYS)


@settings(max_examples=60, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1), sub_cached=st.booleans())
def test_lukeop_vs_fortran(oracle, golden, lf, seed, sub_cached):
    rng = np.random.default_rng(seed + 1)
    s = T.stale_saves(T.random_case(golden, seed, 10.0**lf), "lukeop", POPS, rng)
    units = ["lukeop"]
    if sub_cached:
        s = T.stale_saves(s, "n1op", {"c1130": ((99,), 1e-2), "c1020": ((99,), 1e-1)}, rng)
        s = T.stale_saves(s, "si2op", {"nt": ((99,), "int"), "dt": ((99,), "unit")}, rng)
        units += ["n1op", "si2op"]
    o = T.run_oracle(oracle, s, "lukeop", saves=units)
    T.check(lukeop(s), T.oracle_ref(o, "lukeop"), KEYS, name=f"lukeop@{10**lf:.6e} ")


def test_lukeop_pops_callback(golden):
    """ITEMP changed: populations through pops(s, code, mode=1) == the same populations SAVEd."""
    s = T.stale_saves(T.random_case(golden, 7, 3e15), "lukeop", POPS, np.random.default_rng(8))
    table = {code: s[f"save.lukeop.{k}"] for k, code in LUKE_POPS}
    calls = []

    def pops(st_, code, mode):
        calls.append((code, mode))
        return table[code]
    fresh = s.updated({"save.lukeop.itemp1": torch.tensor([0])})
    got, ref = lukeop(fresh, pops=pops), lukeop(s)
    assert calls == [(code, 1) for _, code in LUKE_POPS]
    T.check(got, ref, KEYS, rtol=0.0)
