"""PARCOE (atlas7v.for): golden (the INTEG call inside JOSH, traced) + property vs the Fortran routine."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.josh import parcoe
from atlas_torch.testing.compare import assert_close

STAGE = "sun_500_510/spectrv"


def test_parcoe_golden(golden):
    # trace order: josh_0 -> integ_1 -> parcoe_1 ; josh_1 -> integ_2 -> parcoe_2 (integ_0 is READIN's)
    for jh, ih in ((0, 1), (1, 2)):
        ent, ext = golden(STAGE, "josh", jh, "entry"), golden(STAGE, "josh", jh, "exit")
        it = golden(STAGE, "integ", ih, "exit")
        n = int(ent["rhox.nrhox"][0])
        x = torch.as_tensor(ent["rhox.rhox"][:n])
        f = torch.as_tensor(ext["abtot.abtot"][:n])
        a, b, c = parcoe(f, x)
        for nm, v in (("a", a), ("b", b), ("c", c)):
            assert_close(v, it[f"static:{nm}"][:n], rtol=0, atol=0, name=f"parcoe.{nm}[josh {jh}]")


def _fortran(o, f, x):
    n = len(f)
    a, b, c = (np.zeros(n, order="F") for _ in range(3))
    o.call("parcoe", np.asfortranarray(f), np.asfortranarray(x), a, b, c, n)
    return a, b, c


@settings(max_examples=80, deadline=None)
@given(n=st.integers(4, 99), seed=st.integers(0, 2**31 - 1), mono=st.booleans(), lg=st.booleans())
def test_parcoe_vs_fortran(oracle, n, seed, mono, lg):
    rng = np.random.default_rng(seed)
    o = oracle("atlas7v")
    rows = []
    for _ in range(4):                     # batched over rows, each row vs one Fortran call
        x = np.cumsum(rng.uniform(0.01, 1.0, n)) if mono else rng.uniform(-5, 5, n)
        if lg:
            x = 10.0 ** (x / x.max() * 6 - 4)
        f = rng.normal(size=n) * 10.0 ** rng.uniform(-3, 3, n)
        f[rng.random(n) < 0.1] = 0.0
        rows.append((x, f))
    X = torch.as_tensor(np.stack([r[0] for r in rows])); F = torch.as_tensor(np.stack([r[1] for r in rows]))
    A, B, C = parcoe(F, X)
    for i, (x, f) in enumerate(rows):
        ra, rb, rc = _fortran(o, f, x)
        with np.errstate(all="ignore"):
            for nm, got, exp in (("a", A[i], ra), ("b", B[i], rb), ("c", C[i], rc)):
                assert_close(got, exp, rtol=0, atol=0, name=f"parcoe.{nm} n={n}")
