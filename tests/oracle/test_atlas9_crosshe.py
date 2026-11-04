"""CROSSHE and HE12S1S/HE12S3S/HE12P1P/HE12P3P (He I photoionisation helpers of HE1OP) vs the Fortran."""
import ctypes

from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics import opacity as O


@settings(max_examples=500, deadline=None)
@given(lf=st.floats(14.85, 17.5))
def test_he_helpers_vs_fortran(lf):
    o = T.make_oracle("atlas9")
    f = 10.0 ** lf
    for name in ("crosshe", "he12s1s", "he12s3s", "he12p1p", "he12p3p"):
        assert getattr(O, name)(f) == o.call(name, f, restype=ctypes.c_double)[0], name
