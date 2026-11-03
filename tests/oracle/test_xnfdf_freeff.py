"""Oracle tests for the free-format card readers of xnfdf.for (FREEFF, IWORDF, XFREEF) and the
continuum-edge reader of PROGRAM XNFDF.

1. golden: edges + FREQSET of the traced solar run (traces/odf_solar/xw/fort.10 = ifx -g -O0 output);
   XFREEF golden hits (card buffer at exit).
2. hypothesis: random cards -> port FREEFF / IWORDF vs the Fortran functions (value, NUMCOL/LETCOL,
   IFFAIL, WORD) through the ctypes oracle.
"""
import ctypes

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.dfsynthe.xnf import oracle_support as osup
from atlas_torch.dfsynthe.xnf.freeform import Free, card_from_line, freeff, iwordf
from atlas_torch.dfsynthe.xnf.readin import read_edges
from atlas_torch.dfsynthe.xnf.xnfio import read_records

pytestmark = pytest.mark.skipif(not osup.available(), reason="libxnfdf.so or the Intel Fortran runtime missing")
RUN = osup.ext_dir() / "traces" / "odf_solar" / "xw"


def test_edges_golden():
    frq, wl, cm, fs = read_edges((RUN / "fort.17").read_text())
    recs = read_records(RUN / "fort.10")
    nin = int(np.frombuffer(recs[1], "<i4", 1)[0])
    e = np.frombuffer(recs[1], "<f8", 3 * nin, 4).reshape(nin, 3)
    nnu = int(np.frombuffer(recs[2], "<i4", 1)[0])
    f = np.frombuffer(recs[2], "<f8", nnu, 4)
    assert nin == len(frq) and nnu == len(fs)
    assert np.array_equal(e[:, 0], frq) and np.array_equal(e[:, 1], wl) and np.array_equal(e[:, 2], cm)
    assert np.array_equal(f, fs)


def test_xfreef_golden(golden):
    """XFREEF hits 0..2: the card buffer read from fort.17 is the first data line(s) of continua.dat."""
    lines = (RUN / "fort.17").read_text().splitlines()
    for hit in range(3):
        ex = golden("odf_solar/xnfdf", "xfreef", hit, "exit")
        raw = np.ascontiguousarray(ex["arg:card"]).tobytes()
        txt = "".join(raw[8 * i:8 * i + 1].decode("latin-1") for i in range(80))
        assert txt.rstrip() in [l[:80].rstrip() for l in lines[:3]]


@pytest.fixture(scope="module")
def ora():
    return osup.new_oracle()


def _fortran(ora, fn, line, numcol, letcol, more, maxpow):
    card = osup.card_array(card_from_line(line))
    ora.set_common("free", numcol=numcol, letcol=letcol, last=81, more=more, iffail=0, maxpow=maxpow)
    r, _ = ora.call(fn, card, restype=ctypes.c_double if fn == "freeff" else ctypes.c_int)
    fr = ora.common("free")
    word = "".join(np.asarray(w).tobytes()[:1].decode("latin-1") for w in fr["word"])
    return r, int(fr["numcol"][0]), int(fr["letcol"][0]), int(fr["iffail"][0]), word


_tok = st.one_of(
    st.from_regex(r"-?[0-9]{1,7}(\.[0-9]{0,6})?([EeD][-+]?[0-9]{1,2})?", fullmatch=True),
    st.from_regex(r"-?\.[0-9]{1,5}", fullmatch=True),
    st.from_regex(r"[A-Z]{1,9}[0-9]?", fullmatch=True),
    st.sampled_from(["SAME", "E", "1.E25", "1354.044E25,", "-110.E25,", "0", "1-05", "3.+2", ",,", "=", "1E"]))
_card = st.lists(_tok, min_size=0, max_size=10).map(lambda t: " ".join(t)[:80]) | \
    st.lists(_tok, min_size=1, max_size=12).map(lambda t: ",".join(t)[:80])


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(line=_card, numcol=st.integers(1, 81), maxpow=st.sampled_from([38, 99]))
def test_freeff_vs_fortran(ora, line, numcol, maxpow):
    fr = Free(numcol=numcol, letcol=1, last=81, more=1, maxpow=maxpow)
    got = freeff(card_from_line(line), fr)
    r, nc, lc, iff, _ = _fortran(ora, "freeff", line, numcol, 1, 1, maxpow)
    assert (got, fr.numcol, fr.iffail) == (r, nc, iff), line


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(line=_card, letcol=st.integers(1, 81))
def test_iwordf_vs_fortran(ora, line, letcol):
    fr = Free(numcol=1, letcol=letcol, last=81, more=1, maxpow=38)
    got = iwordf(card_from_line(line), fr)
    r, nc, lc, iff, word = _fortran(ora, "iwordf", line, 1, letcol, 1, 38)
    assert (got, fr.letcol, fr.iffail, "".join(fr.word)) == (r, lc, iff, word), line
