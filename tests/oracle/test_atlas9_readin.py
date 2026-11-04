"""READIN (atlas9mem.for, BasicATLAS cards): golden trace entry -> exit for the traced solar run (control cards
READ KAPPA / READ PUNCH / MOLECULES / READ MOLECULES / FREQUENCIES BIG / VTURB / CONVECTION OVER / TITLE /
SCALE (-> TTAUP) / ABUNDANCE ... / ITERATIONS / PRINT / PUNCH / BEGIN) and its unit-6 printout; the card
scanner FREEFF/IWORDF vs the Fortran functions on random cards."""
from pathlib import Path

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9.cards import Free, StdinReader
from atlas_torch.atlas9.commons import initial_state
from atlas_torch.atlas9.readin import IO, readin

STAGE = "sun_atlas9/atlas9"
# COMMON members that are ROSSTAB scratch (its PKAP cache), not READIN results
_SCRATCH = {"tabtp.pkap", "tabtp.pkapv"}


def _run(ext):
    w = Path(ext) / "traces/sun_atlas9/w"
    io = IO(stdin=StdinReader((w / "stdin.txt").read_text()), files={1: w / "fort.1", 2: w / "fort.2", 3: w / "fort.3"})
    return readin(initial_state(), io), io


def test_readin_golden(golden, ext):
    e, x = golden(STAGE, "readin", 0, "entry"), golden(STAGE, "readin", 0, "exit")
    s0 = initial_state()
    for k, v in e.items():                     # the traced entry is the program-start image
        if k in s0 and not k.startswith(("arg:", "static:")):
            a = s0[k].numpy() if isinstance(s0[k], torch.Tensor) else s0[k]
            if a.dtype.kind == "S":
                assert np.array_equal(a, v), k
            else:
                assert np.array_equal(a.astype(v.dtype), v, equal_nan=v.dtype.kind == "f"), k
    s, io = _run(ext)
    checked = 0
    for k, v in x.items():
        if k not in s or k in _SCRATCH:
            continue
        a = s[k].numpy() if isinstance(s[k], torch.Tensor) else np.asarray(s[k])
        if a.dtype.kind == "S":
            assert [b.rstrip(b"\0") for b in a.ravel()] == [b.rstrip(b"\0") for b in v.ravel()], k
        else:
            assert np.array_equal(a.astype(v.dtype), v, equal_nan=v.dtype.kind == "f"), k
        checked += 1
    assert checked > 100


def test_readin_printout(ext):
    """READIN's unit-6 output = the first part of the Fortran's stdout (before the first iteration table)."""
    _, io = _run(ext)
    ref = (Path(ext) / "traces/sun_atlas9/atlas9/gdb.log").read_text().split("\n")
    i0 = next(i for i, l in enumerate(ref) if l.startswith(" ROSSELAND MASS"))
    i1 = next(i for i, l in enumerate(ref) if l.startswith("0" + " " * 108 + "FLUX"))
    assert io.stdout == ref[i0:i1]


_cards = st.lists(st.one_of(
    st.from_regex(r"-?[0-9]{1,6}\.?[0-9]{0,6}", fullmatch=True),
    st.from_regex(r"-?[0-9]{1,3}\.[0-9]{1,4}E[-+]?[0-9]{1,2}", fullmatch=True),
    st.from_regex(r"[A-Z]{1,9}", fullmatch=True),
    st.sampled_from(["READ", "PUNCH", "2.0E+5", "1.0E-5", ".5", "5770.", "-6.875", "E5", "1E5", "3.E", ",", "="])),
    min_size=1, max_size=8)


@settings(max_examples=200, deadline=None)
@given(toks=_cards, seps=st.lists(st.sampled_from([" ", "  ", ",", " , "]), min_size=8, max_size=8))
def test_card_scanner_vs_fortran(oracle, toks, seps):
    line = "".join(t + seps[i] for i, t in enumerate(toks))[:80]
    o = oracle("atlas9")
    card = np.array([c.encode() for c in line.ljust(80)] + [b" "], dtype="S1")
    fr = Free([])
    fr.more = 1
    c = Free.make_card(line)
    for which in ("freeff", "iwordf", "freeff", "freeff", "iwordf"):
        o.set_common("free", numcol=fr.numcol, letcol=fr.letcol, last=81, more=1, iffail=0, maxpow=38)
        if fr.numcol == 0:
            fr.numcol = 1
            o.set_common("free", numcol=1)
        if fr.letcol == 0:
            fr.letcol = 1
            o.set_common("free", letcol=1)
        if which == "freeff":
            r, _ = o.call("freeff", card, restype=__import__("ctypes").c_double)
            mine = fr.freeff(c)
        else:
            r, _ = o.call("iwordf", card, restype=__import__("ctypes").c_int)
            mine = fr.iwordf(c)
        cm = o.common("free")
        assert mine == r, (which, line)
        assert (fr.numcol, fr.letcol, fr.iffail) == (int(cm["numcol"][0]), int(cm["letcol"][0]), int(cm["iffail"][0]))
