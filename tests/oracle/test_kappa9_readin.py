"""Oracle tests for KAPPA9's READIN (+ READMOL) and its card parser FREEFF / IWORDF (kappa9.for).

1. READIN on the BasicATLAS solar / alpha decks: every COMMON member the port produces equals the Fortran's
   (READIN(1) run in a subprocess with unit 5 = the deck, unit 2 = molecules.dat), bit for bit.
2. hypothesis: perturbed decks (ABUNDANCE SCALE / CHANGE values, TEFF, GRAVITY, VTURB) -> same comparison.
3. hypothesis: random cards -> FREEFF / IWORDF results and /FREE/ cursor state vs the Fortran functions.
"""
import ctypes
import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.kappa9.readin import Cards, readin
from atlas_torch.testing import kappa9_oracle_physics as k9o

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
WORK = EXT / "work"
MOL = EXT / "traces" / "odf_solar" / "kw" / "fort.2"

pytestmark = pytest.mark.skipif(not k9o.available(), reason="libkappa9.so or the Intel Fortran runtime missing")


def _deck(mix, v=2):
    p = WORK / f"kappa9_{mix}_stock" / f"stdin_v{v}.txt"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p.read_text()


def _compare(rd, F, ulps=None):
    """Bitwise, except keys in `ulps` (key -> max ulp distance)."""
    ulps = ulps or {}
    bad = []
    for k, v in rd.items():
        a = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        if a.dtype.kind == "S":
            continue
        assert k in F, k
        b = F[k].reshape(a.shape)
        if k in ulps:
            d = np.abs(a.astype(np.float64).view(np.int64) - b.astype(np.float64).view(np.int64))
            if d.max() > ulps[k]:
                bad.append(f"{k}: {int(d.max())} ulp")
            continue
        if not np.array_equal(a.astype(np.float64), b.astype(np.float64)):
            d = np.flatnonzero(a.reshape(-1).astype(np.float64) != b.reshape(-1).astype(np.float64))
            bad.append(f"{k}: {d.size} differ, e.g. [{d[0]}] port {a.reshape(-1)[d[0]]!r} fortran {b.reshape(-1)[d[0]]!r}")
    assert not bad, "\n".join(bad)


def test_readin_golden(golden):
    """Trace of the stock run's READIN(1) (ifx -O2 build, Intel libimf/SVML EXP): every recorded COMMON member
    the port produces, bitwise except XABUND = EXP10(ABUND)*XSCALE (1 ulp: libimf vs the platform libm exp;
    the -O0 oracle library agrees with the port bitwise, see test_readin_vs_fortran)."""
    ex = golden("odf_solar/kappa9", "readin", 0, "exit")
    rd = readin(_deck("solar"), MOL)
    common = [k for k in ex if not k.startswith(("arg:", "static:")) and k in rd]
    assert len(common) >= 10, common
    _compare({k: rd[k] for k in common}, ex, ulps={"xabund.xabund": 1})


@pytest.mark.parametrize("mix", ["solar", pytest.param("alpha", marks=pytest.mark.stage)])   # alpha: ~2 s, stage tier
def test_readin_vs_fortran(mix):
    txt = _deck(mix)
    _compare(readin(txt, MOL), k9o.fortran_readin(txt, MOL))


@pytest.mark.stage   # ~20 s (whole READIN deck per example); fast tier: test_readin_golden, test_readin_vs_fortran
@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(xscale=st.floats(0.001, 30.0), dab=st.lists(st.floats(-1.5, 1.5), min_size=6, max_size=6),
       teff=st.integers(3000, 50000), logg=st.floats(0.0, 5.0), v=st.sampled_from([0, 1, 2, 4, 8]))
def test_readin_perturbed_vs_fortran(xscale, dab, teff, logg, v):
    txt = _deck("solar")
    txt = re.sub(r"ABUNDANCE SCALE +[0-9.]+", f"ABUNDANCE SCALE {xscale:9.6f}", txt)
    txt = re.sub(r"TEFF +[0-9.]+ +GRAVITY +[0-9.]+", f"TEFF {teff:7.0f}.  GRAVITY {logg:7.5f}", txt)
    txt = re.sub(r"VTURB .*", f"VTURB {v}.0E5", txt)
    lines = txt.splitlines()
    for i, ln in enumerate(lines):             # perturb the first value of six ABUNDANCE CHANGE cards
        if ln.startswith(" ABUNDANCE CHANGE") and dab:
            m = re.match(r"( ABUNDANCE CHANGE +\d+ +)(-?[0-9.]+)(.*)", ln)
            val = float(m.group(2)) + dab.pop()
            lines[i] = f"{m.group(1)}{val:6.2f}{m.group(3)}"
    txt = "\n".join(lines) + "\n"
    _compare(readin(txt, MOL), k9o.fortran_readin(txt, MOL))


# ---------------------------------------------------------------- FREEFF / IWORDF

@pytest.fixture(scope="module")
def ko():
    k9o.preload_ifx_runtime()
    from atlas_torch.testing.fortran_oracle import Oracle
    return Oracle(k9o.LIB)


def _card_array(text):
    s = text.ljust(80)[:80] + " "
    return np.asfortranarray(np.array([ord(c) | 0x20202000 for c in s], dtype=np.int32))


def _setup(ko, numcol, letcol):
    ko.set_common("free", numcol=numcol, letcol=letcol, last=81, more=1, iffail=0, maxpow=38)


_tok = st.one_of(
    st.from_regex(r"-?[0-9]{1,9}(\.[0-9]{0,8})?([E][+-]?[0-9]{1,2})?", fullmatch=True),
    st.from_regex(r"-?\.[0-9]{1,6}", fullmatch=True),
    st.from_regex(r"[A-Z]{1,10}[0-9]?", fullmatch=True),
    st.sampled_from([",", "=", "-", "+", ".", "E", "1.E", "2E5", "1.5-3", "7.+2", "--4", "3.2.1"]))


@settings(max_examples=300, deadline=None)
@given(toks=st.lists(_tok, min_size=1, max_size=8), seps=st.lists(st.sampled_from([" ", "  ", ",", " , "]), min_size=8, max_size=8),
       start=st.integers(1, 12), nread=st.integers(1, 6))
def test_freeff_vs_fortran(ko, toks, seps, start, nread):
    text = "".join(t + s for t, s in zip(toks, seps))[:80]
    card = _card_array(text)
    cd = Cards(text)
    cd.read()
    cd.numcol, cd.letcol, cd.more = start, start, 1
    _setup(ko, start, start)
    for _ in range(nread):
        r, _ = ko.call("freeff", card, restype=ctypes.c_double)
        v = cd.freeff()
        fr = ko.common("free")
        assert (v, cd.numcol, cd.iffail) == (r, int(fr["numcol"].reshape(-1)[0]), int(fr["iffail"].reshape(-1)[0])), text
        assert np.float64(v).tobytes() == np.float64(r).tobytes(), (text, v, r)


@settings(max_examples=300, deadline=None)
@given(toks=st.lists(_tok, min_size=1, max_size=8), seps=st.lists(st.sampled_from([" ", "  ", ",", "=", " = "]), min_size=8, max_size=8),
       start=st.integers(1, 12), nread=st.integers(1, 6))
def test_iwordf_vs_fortran(ko, toks, seps, start, nread):
    text = "".join(t + s for t, s in zip(toks, seps))[:80]
    card = _card_array(text)
    cd = Cards(text)
    cd.read()
    cd.numcol, cd.letcol, cd.more = start, start, 1
    _setup(ko, start, start)
    for _ in range(nread):
        r, _ = ko.call("iwordf", card, restype=ctypes.c_int)
        m = cd.iwordf()
        fr = ko.common("free")
        word = "".join(chr(int(x) & 0xFF) for x in fr["word"].view(np.int32))
        assert (m, cd.letcol, cd.iffail, "".join(cd.word)) == (
            r, int(fr["letcol"].reshape(-1)[0]), int(fr["iffail"].reshape(-1)[0]), word), text
