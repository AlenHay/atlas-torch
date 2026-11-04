"""Oracle tests for HOTOP (atlas7v.for): free-free of C,N,O,Ne,Mg,Si,S,Fe I-V plus 60 bound-free edges of
C, N, O, Ne ions. Populations: HOTOP's SAVEd XNFP/XNFSUMQQ (ITEMP1 = ITEMP), or a `pops` callback when ITEMP
changes (POPS itself is ported elsewhere)."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity._core import table
from atlas_torch.synthe.opacity.hot import HOT_XNFP, hotop

KEYS = ["ops.ahot", "save.hotop.itemp1", "save.hotop.xnfp", "save.hotop.xnfsumqq"]
POPS = {"xnfp": ((99, 21), 1e9), "xnfsumqq": ((99, 5), 1e10)}


def test_hotop_golden(golden):
    T.golden_check(golden, "hotop", hotop, KEYS)


@settings(max_examples=60, deadline=None)
@given(lf=st.floats(14.5, 16.7), seed=st.integers(0, 2**32 - 1), tscale=st.sampled_from([(0.8, 1.25), (2.0, 4.0)]))
def test_hotop_vs_fortran(oracle, golden, lf, seed, tscale):
    """FREQ over the 60 edges (4.1e15 .. 3.8e16 Hz); tscale (2,4): hot layers where the edges matter."""
    s = T.stale_saves(T.random_case(golden, seed, 10.0**lf, tscale=tscale), "hotop", POPS,
                      np.random.default_rng(seed + 1))
    o = T.run_oracle(oracle, s, "hotop", saves=["hotop"])
    T.check(hotop(s), T.oracle_ref(o, "hotop"), KEYS, name=f"hotop@{10**lf:.6e} ")


def test_hotop_pops_callback(oracle, golden):
    """ITEMP changed: XNFP from POPS(code,11) placed at columns 1,5,10,16; XNFSUMQQ(J,IZ) = sum over the 8
    CODE8 elements of IZ**2 * XNF(J,IZ+1) from POPS(code,12). Compared with the Fortran given the same
    populations in its SAVEd arrays."""
    rng = np.random.default_rng(3)
    s = T.random_case(golden, 3, 2e16, tscale=(2.0, 4.0))
    mode11 = {code: torch.from_numpy(10.0 ** rng.uniform(5, 12, (99, w)))
              for (code, _), w in zip(HOT_XNFP, (4, 5, 6, 6))}
    mode12 = {float(code): torch.from_numpy(10.0 ** rng.uniform(5, 12, (99, 6))) for code in table("hotop", "code8")}
    calls = []

    def pops(st_, code, mode):
        calls.append((round(code, 2), mode))
        return mode11[code] if mode == 11 else mode12[round(code, 2)]
    got = hotop(s, pops=pops)
    assert calls[:4] == [(c, 11) for c, _ in HOT_XNFP] and all(m == 12 for _, m in calls[4:]) and len(calls) == 12
    n = s.iscalar("rhox.nrhox")
    xnfp = np.zeros((99, 21))
    for (code, col), w in zip(HOT_XNFP, (4, 5, 6, 6)):
        xnfp[:n, col:col + w] = mode11[code][:n].numpy()
    q = np.zeros((99, 5))
    for code in table("hotop", "code8"):
        for iz in range(1, 6):
            q[:n, iz - 1] += iz**2 * mode12[float(code)][:n, iz].numpy()
    ref = s.updated({"save.hotop.itemp1": torch.tensor([s.iscalar("temp.itemp")]),
                     "save.hotop.xnfp": torch.from_numpy(xnfp), "save.hotop.xnfsumqq": torch.from_numpy(q)})
    o = T.run_oracle(oracle, ref, "hotop", saves=["hotop"])
    T.check(got, T.oracle_ref(o, "hotop"), KEYS, name="hotop(pops) ")
