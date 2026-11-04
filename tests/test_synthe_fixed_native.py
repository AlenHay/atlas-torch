"""synthe/lines/_fixed.py: native fcol/icol (synthe/native/fixedcol.c) == the numpy reference, bit for bit, on random
fixed-width fields (digits, blanks, points, signs, exponent letters and junk -> scalar fallback)."""
import numpy as np
import pytest

from atlas_torch.synthe.lines import _fixed as X


def _fields(rng, n, w):
    alphabet = np.frombuffer(b"0123456789     ..--+E*", np.uint8)
    a = rng.choice(alphabet, size=(n, w))
    clean = rng.random(n) < 0.7                             # mostly well-formed numbers
    for i in np.flatnonzero(clean):
        s = f"{rng.uniform(-1e4, 1e4):.{rng.integers(0, 6)}f}"
        if rng.random() < 0.2:
            s = s.replace(".", "")
        s = s.rjust(w)[-w:] if rng.random() < 0.8 else s.ljust(w)[:w]
        a[i] = np.frombuffer(s.encode(), np.uint8)
    return np.ascontiguousarray(a)


@pytest.mark.skipif(X._native() is None, reason="fixedcol kernel not built (make native)")
@pytest.mark.parametrize("w,d", [(10, 4), (7, 3), (11, 3), (4, 0), (20, 2)])
def test_fcol_icol_native_equal_reference(w, d):
    rng = np.random.default_rng(w * 31 + d)
    a = _fields(rng, 20000, w + 3)
    lo, hi = 1, 1 + w

    def ok(fn, row):                                         # rows the scalar Fortran reader accepts
        try:
            fn(bytes(row[lo:hi]).decode("latin1"))
            return True
        except Exception:                                    # noqa: BLE001
            return False
    af = np.ascontiguousarray(a[[ok(lambda t: X.fread(t, d), r) for r in a]])
    ai = np.ascontiguousarray(a[[ok(X.iread, r) for r in a]])
    assert len(af) > 10000 and len(ai) > 1000
    v, r = X.fcol(af, lo, hi, d), X.fcol_ref(af, lo, hi, d)
    assert np.array_equal(v.view(np.int64), r.view(np.int64))
    assert np.array_equal(X.icol(ai, lo, hi), X.icol_ref(ai, lo, hi))
