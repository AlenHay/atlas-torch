"""Oracle tests for FUNCTION CHOP(J) (atlas7v.for): CH photodissociation cross-section x partition function.
KAPP calls it for J=1..NRHOX in order; the J=1 call on a new FREQ refreshes the FREQ1 cache and writes
COMMON WAVENO = FREQ/2.99792458D10. Also guards the CROSSCH DATA table (blank-free `DATAC10/` statements)
against the compiled library's initial values."""
import ctypes

import numpy as np
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.opacity import _testing as T
from atlas_torch.synthe.opacity._core import table
from atlas_torch.synthe.opacity.cool import chop
from atlas_torch.testing.compare import assert_close

UNIT = "chop"
FN = chop


def test_table_matches_compiled():
    from atlas_torch.testing.so_tables import so_table
    for v in ("crossch", "partch"):
        assert np.array_equal(table(UNIT, v), so_table("atlas7v.for", UNIT, v)), v


def _fortran(o, n):
    return np.array([o.call(UNIT, j, restype=ctypes.c_double)[0] for j in range(1, n + 1)])


def test_golden(golden):
    for hit in range(3):
        s, e, d = T.golden_state(golden, UNIT, hit)
        _, out = FN(s)
        T.check(out, e, [k for k in out if k in e], name=f"{UNIT}[{hit}] ")
        assert int(d["static:n"][0]) == int(np.trunc(float(s.scalar("freq.freq")) / 2.99792458e10 /
                                                     (8065.479 if UNIT == "chop" else np.float32(8065.479))
                                                     * 10.0 - (0 if UNIT == "chop" else 20)))


@settings(max_examples=80, deadline=None)
@given(lf=st.floats(14.6, 15.6), seed=st.integers(0, 2**32 - 1), repeat=st.booleans(),
       tscale=st.sampled_from([(0.8, 1.25), (0.3, 0.6), (1.2, 1.6)]))
def test_vs_fortran(oracle, golden, lf, seed, repeat, tscale):
    """FREQ over the tabulated range (and a bit beyond); repeat=True: FREQ1 already equals FREQ (cache hit,
    WAVENO untouched)."""
    freq = 10.0**lf
    s = T.random_case(golden, seed, freq, tscale=tscale)
    if repeat:
        s[f"save.{UNIT}.freq1"] = s["freq.freq"].clone()
    o = T.run_oracle(oracle, s)
    if repeat:
        o.call(UNIT, 1, restype=ctypes.c_double)        # fill the Fortran cache at this FREQ
        T.push(o, s, ["freq"])                           # restore xnfpelsyn's WAVENO
    ref = _fortran(o, s.iscalar("rhox.nrhox"))
    val, out = FN(s)
    assert_close(val, ref, rtol=T.RTOL, name=f"{UNIT}@{freq:.6e}")
    T.check(out, T.oracle_ref(o, UNIT), list(out), name=f"{UNIT}@{freq:.6e} ")
    assert ("freq.waveno" in out) == (not repeat)
