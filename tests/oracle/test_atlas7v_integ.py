"""INTEG (atlas7v.for): golden (TAUNU of the traced JOSH calls) + property vs the Fortran routine."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.synthe.josh import integ
from atlas_torch.testing.compare import assert_close

STAGE = "sun_500_510/spectrv"


def test_integ_golden(golden):
    for jh, ih in ((0, 1), (1, 2)):
        ent, ext = golden(STAGE, "josh", jh, "entry"), golden(STAGE, "josh", jh, "exit")
        it = golden(STAGE, "integ", ih, "exit")
        n = int(ent["rhox.nrhox"][0])
        x = torch.as_tensor(ent["rhox.rhox"][:n])
        f = torch.as_tensor(ext["abtot.abtot"][:n])
        fint, (a, b, c) = integ(x, f, f[0] * x[0], return_coef=True)
        assert_close(fint, ext["taushj.taunu"][:n], rtol=0, atol=0, name=f"taunu[josh {jh}]")
        assert int(it["static:n1"][0]) == n - 1
        for nm, v in (("a", a), ("b", b), ("c", c)):
            assert_close(v, it[f"static:{nm}"][:n], rtol=0, atol=0, name=f"integ.{nm}")


@settings(max_examples=80, deadline=None)
@given(n=st.integers(4, 99), seed=st.integers(0, 2**31 - 1), start=st.floats(-10, 10), wild=st.booleans())
def test_integ_vs_fortran(oracle, n, seed, start, wild):
    rng = np.random.default_rng(seed)
    o = oracle("atlas7v")
    xs, fs = [], []
    for _ in range(4):
        x = np.cumsum(10.0 ** rng.uniform(-5, 0, n))
        f = 10.0 ** rng.uniform(-6, 6 if wild else 0, n)      # wild: non-monotonic integrals
        xs.append(x); fs.append(f)
    X, F = torch.as_tensor(np.stack(xs)), torch.as_tensor(np.stack(fs))
    got = integ(X, F, torch.full((4,), start))
    for i in range(4):
        fint = np.zeros(n, order="F")
        o.call("integ", np.asfortranarray(xs[i]), np.asfortranarray(fs[i]), fint, n, float(start))
        assert_close(got[i], fint, rtol=0, atol=0, name=f"integ n={n}")
