"""Oracle tests for READIN (atlas7v.for) and its free-format helpers FREEFF / IWORDF / FREEFR (+ XFREEF of
xnfpelsyn.for).

  golden:   initial_state() == the COMMON image at READIN entry (BLOCKR/BLOCKE DATA); readin() on the
            BasicATLAS deck (scratch_sun/output_synthe.out) == every COMMON member at READIN exit (+ MIAC);
            the traced FREEFF / IWORDF / XFREEF hits.
  property: READIN on randomly modified decks (TEFF/GRAVITY/abundances/model values perturbed, D exponents,
            optional cards: LTE/NLTE, SCATTERING, CONVECTION OFF, TURBULENCE ON, OPACITY ON/OFF/IFOP,
            SURFACE INTENSITY with FREEFR continuation lines, CORRECTION, PRESSURE, WAVELENGTH, CALL,
            READ FREQUENCIES, ABUNDANCE TABLE, READ DECK instead of DECK6, CHANGE RHOX) vs the Fortran
            READIN run in a subprocess with the deck on stdin and molecules.dat as fort.2; FREEFF / IWORDF on
            random cards vs the Fortran, bitwise.
"""
import ctypes
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.synthe.freeform import CardStream, Free, FortranExit, freeff, iwordf, xfreef
from atlas_torch.synthe.readin import ALIASES, default_molecules_path, initial_state, readin
from test_atlas7v_pfsaha import STAGE

DECK = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "scratch_sun" / "output_synthe.out"
SO = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "oracle" / "libatlas7v.so"


def _np(v):
    return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _same(a, e) -> bool:
    a, e = _np(a).reshape(-1), np.asarray(e).reshape(-1)
    if e.dtype.kind == "S":
        return np.array_equal(a.astype(e.dtype), e)
    return np.array_equal(a.astype(np.float64), e.astype(np.float64))


def _compare(s, exp: dict, name: str, subset: bool = False):
    """All COMMON members of `exp` (a trace) -- or, subset=True (an oracle dump, whose blocks also hold
    members READIN never declares), the members both have -- are equal."""
    keys = [k for k in exp if not k.startswith(("arg:", "static:")) and (not subset or k in s)]
    assert len(keys) > (100 if subset else 5)
    bad = [k for k in keys if not _same(s[k], exp[k])]
    assert not bad, f"{name}: differing COMMON members {bad[:10]}"


# ------------------------------------------------------------------------------------------------ golden

def test_initial_state_matches_trace(golden):
    e = golden(STAGE, "readin", 0, "entry")
    _compare(initial_state(), e, "initial COMMON image")


def test_readin_golden(golden, ext):
    s = readin(DECK.read_text())
    x = golden(STAGE, "readin", 0, "exit")
    _compare(s, x, "READIN exit")
    assert int(s["save.readin.miac"][0]) == int(x["static:miac"][0])
    rm = golden(STAGE, "readmol", 0, "exit")
    _compare(s, rm, "READMOL COMMONs after READIN")
    for a, k in ALIASES.items():
        assert _same(s[a], _np(s[k]))


def _card(e):
    return [" "] + [c.decode("latin-1") or "\0" for c in e["arg:card"]] + [" ", " "]


def test_freeff_iwordf_golden(golden):
    for r, fn in (("freeff", freeff), ("iwordf", iwordf)):
        for hit in range(3):
            e, x = golden(STAGE, r, hit, "entry"), golden(STAGE, r, hit, "exit")
            fr = Free.from_state({k: v for k, v in e.items()})
            fn(_card(e), fr)
            for k in Free.KEYS:
                assert getattr(fr, k) == int(x["free." + k][0]), (r, hit, k)
            if r == "iwordf":
                assert [c.encode() for c in fr.word] == [c or b"\0" for c in x["free.word"]] or \
                    "".join(fr.word).strip() == b"".join(x["free.word"]).decode().strip()


def test_xfreef_golden(golden):
    """XFREEF hits read the continuum edges from unit 17 (the traced run's fort.17)."""
    fort17 = (Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "traces" / "sun_500_510" / "xw"
              / "fort.17").read_text(encoding="latin-1")
    unit = CardStream(fort17)
    card = None
    for hit in range(3):
        e, x = golden(STAGE, "xfreef", hit, "entry"), golden(STAGE, "xfreef", hit, "exit")
        c = [" "] + [ch.decode("latin-1") or " " for ch in e["arg:card"]] + [" "]
        if card is not None:
            assert c[1:82] == card[1:82]
        fr = Free.from_state(dict(e))
        xfreef(c, fr, unit)
        card = c
        for k in Free.KEYS:
            assert getattr(fr, k) == int(x["free." + k][0]), (hit, k)


# ---------------------------------------------------------------------------------------- FREEFF / IWORDF

_ORACLE = {}
_TOKS = ["1.0E30", "1.5D25", "-3.25", "12345678901234567890", ".5", "1.E-37", "7.e+12", "2.3E", "E5", "4.",
         "1,2", "+5", "-", "1.2.3", "9.99999999999E-300", "1.0E-38", "123.456E-12", "1.0D+5", "0.1E39",
         "ABC", "EDGE", "x", "READ,DECK6", "3E", ".E5", "X=Y", "1A"]


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(words=st.lists(st.sampled_from(_TOKS), min_size=1, max_size=10), maxpow=st.sampled_from([38, 99]),
       tail=st.sampled_from([" ", "\0", "7", "E", "A"]))
def test_freeff_iwordf_vs_fortran(oracle, words, maxpow, tail):
    o = _ORACLE.setdefault("o", oracle("atlas7v"))     # FREEFF/IWORDF keep no state between calls
    f = o.lib.freeff_
    f.restype, f.argtypes = ctypes.c_double, [ctypes.c_char_p, ctypes.c_long]
    g = o.lib.iwordf_
    g.restype, g.argtypes = ctypes.c_int, [ctypes.c_char_p, ctypes.c_long]
    s = " ".join(words)[:80].ljust(80) + tail + " "          # CARD(81), CARD(82) as READIN sees them
    raw = s.encode("latin-1")
    for fn, cfn, start in ((freeff, f, "numcol"), (iwordf, g, "letcol")):
        fr = Free(numcol=1, letcol=1, last=81, more=1, maxpow=maxpow)
        o.set_common("free", numcol=1, letcol=1, last=81, more=1, iffail=0, maxpow=maxpow)
        card = [" "] + list(s)
        for _ in range(12):
            v = cfn(raw, 80)
            vp = fn(card, fr)
            cm = o.common("free")
            assert v == vp, (s, v, vp)
            assert int(cm[start][0]) == getattr(fr, start) and int(cm["iffail"][0]) == fr.iffail
            if fn is iwordf:
                assert b"".join(cm["word"]).decode("latin-1") == "".join(fr.word)
            if fr.iffail:
                break


# ----------------------------------------------------------------------------------- READIN vs Fortran

_RUNNER = r"""
import ctypes, pickle, sys, numpy as np
from atlas_torch.testing import fortran_oracle
layout = pickle.load(open(sys.argv[5], "rb"))        # DWARF layout parsed once by the parent process
fortran_oracle.read_layout = lambda path: layout
o = fortran_oracle.Oracle(sys.argv[1])
# a gfortran main program calls _gfortran_set_options; without it libgfortran rejects READIN's legacy
# FORMATs (missing commas) at run time. Options: warn_std, allow_std (all), pedantic, backtrace, sign_zero,
# bounds_check, fpe_summary.
opts = (ctypes.c_int * 7)(0, -1, 0, 1, 1, 0, 0)
ctypes.CDLL("libgfortran.so.5")._gfortran_set_options(7, opts)
o.call("readin", int(sys.argv[3]))
out = {}
for blk in sys.argv[4].split(","):
    for k, v in o.common(blk).items():
        out[blk + "." + k] = np.array(v)
out["static:miac"] = np.array(o.saved("readin")["miac"])
np.savez(sys.argv[2], **out)
"""


def oracle_readin(deck: str, mol: Path, mode: int = 20):
    """Fortran READIN(MODE) on `deck` (stdin) with fort.2 -> mol, in a subprocess. None if it EXITed."""
    blocks = sorted({k.split(".")[0] for k in initial_state() if "." in k and not k.startswith("save.")})
    from atlas_torch.testing.fortran_oracle import read_layout
    with tempfile.TemporaryDirectory() as td:
        lay = Path(td, "layout.pkl")
        lay.write_bytes(pickle.dumps(read_layout(str(SO.resolve()))))
        Path(td, "fort.2").symlink_to(mol)
        Path(td, "deck").write_text(deck)
        out = Path(td, "out.npz")
        with open(Path(td, "deck")) as fin:
            r = subprocess.run([sys.executable, "-c", _RUNNER, str(SO), str(out), str(mode), ",".join(blocks), str(lay)],
                               stdin=fin, cwd=td, capture_output=True, timeout=300)
        if not out.exists():
            return None
        with np.load(out) as d:
            return {k: d[k] for k in d.files}


def _fmt(x, w=9):
    return f"{x:{w}.3E}"


def make_deck(seed: int) -> str:
    rng = np.random.default_rng(seed)
    lines = DECK.read_text().split("\n")
    out = []
    deck6 = rng.random() < 0.7
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("TEFF"):
            teff = rng.uniform(3500, 30000)
            g = rng.choice([f"{rng.uniform(0, 5):.5f}", f"{10 ** rng.uniform(1.2, 5):.1f}", "4.44D0"])
            ln = f"TEFF {teff:8.1f}  GRAVITY {g}"
        elif ln.startswith("ABUNDANCE SCALE"):
            ln = f"ABUNDANCE SCALE {rng.uniform(0.1, 3):9.5f} ABUNDANCE CHANGE 1 {rng.uniform(0.8, 0.95):.5f} " \
                 f"2 {rng.uniform(0.05, 0.15):.5f}"
        elif ln.startswith(" ABUNDANCE CHANGE") and rng.random() < 0.5:
            parts = ln.split()
            vals = parts[2:]
            for k in range(1, len(vals), 2):
                vals[k] = f"{float(vals[k]) + rng.uniform(-0.5, 0.5):.2f}"
            ln = " ABUNDANCE CHANGE " + " ".join(f"{vals[k]:>2} {vals[k + 1]:>6}" for k in range(0, len(vals), 2))
        elif ln.startswith("READ DECK6"):
            n = int(ln.split()[2])
            if not deck6:
                ln = ln.replace("DECK6", "DECK ")
            out.append(ln)
            rhox = []
            for j in range(n):
                f = lines[i + 1 + j].split()
                vals = [float(v) for v in f]
                vals[1] *= rng.uniform(0.9, 1.1)
                vals[2] *= 10 ** rng.uniform(-0.3, 0.3)
                vals[3] *= 10 ** rng.uniform(-0.3, 0.3)
                vals[6] = rng.uniform(0, 2e5) if rng.random() < 0.3 else vals[6]
                rhox.append(vals[0])
                out.append(f" {vals[0]:.8E} {vals[1]:8.1f} " + " ".join(_fmt(v) for v in vals[2:]))
            i += 1 + n
            if deck6:
                out.append(f"PRADK {rng.uniform(0.5, 3):.4E}")
            i += 1                                       # the original PRADK card (dropped for READ DECK)
            if rng.random() < 0.3:                       # CHANGE RHOX onto a subset of the model's RHOX
                sel = sorted(rng.choice(np.arange(1, n - 1), size=40, replace=False))
                out.append("CHANGE RHOX 40")
                vals = [f"{rhox[k] * rng.uniform(0.999, 1.001):.6E}" for k in sel]
                for k in range(0, 40, 6):
                    out.append(" " + " ".join(vals[k:k + 6]))
            continue
        out.append(ln)
        if ln.startswith("MOLECULES ON"):
            extra = []
            if rng.random() < 0.3:
                extra.append(rng.choice(["LTE", "NLTE"]))
            if rng.random() < 0.3:
                extra.append("SCATTERING " + rng.choice(["ON", "OFF"]))
            if rng.random() < 0.3:
                extra.append("OPACITY OFF HMINUS HRAY\nOPACITY ON COOL LUKE")
            if rng.random() < 0.3:
                extra.append("SURFACE INTENSITY 7 1.0 .9 .8 .7\n .5 .3,\n .1")
            if rng.random() < 0.3:
                extra.append("CORRECTION ON PRESSURE ON CALL")
            if rng.random() < 0.3:
                extra.append(f"WAVELENGTH {rng.uniform(300, 500):.1f} {rng.uniform(0.01, 1):.3f} 600.")
            if rng.random() < 0.3:
                extra.append("READ FREQUENCIES 3 1 3 TESTNU\n 1 5000. 1.\n 2 1.2D15 2.\n 3 2.2E26 1.")
            if rng.random() < 0.2:
                extra.append("TURBULENCE ON 1.5 2.0 0.5 0.1")
            out.extend(extra)
        if ln.startswith(" CONVECTION") and rng.random() < 0.3:
            out[-1] = " CONVECTION OFF TURBULENCE OFF"
        if ln.startswith(" ABUNDANCE CHANGE 99") and rng.random() < 0.3:
            out.append("ABUNDANCE TABLE")
            out.append(f"{'':7}{rng.uniform(0.85, 0.95):10.6f}{'':10}{rng.uniform(0.05, 0.12):10.6f}")
            recs = [f"{'':7}{rng.uniform(-12, -3):7.3f}{rng.uniform(-0.3, 0.3):6.3f}" for _ in range(97)]
            for k in range(0, 97, 5):
                out.append("".join(recs[k:k + 5]))
        i += 1
    return "\n".join(out)


def _check_deck(deck: str, name: str):
    mol = default_molecules_path()
    exp = oracle_readin(deck, mol)
    try:
        s = readin(deck, mol)
    except FortranExit:
        assert exp is None, f"{name}: port EXITed, Fortran did not"
        return
    assert exp is not None, f"{name}: Fortran EXITed, port did not"
    _compare(s, exp, name, subset=True)
    assert int(s["save.readin.miac"][0]) == int(exp["static:miac"][0])


def test_readin_basicatlas_deck_vs_fortran():
    _check_deck(DECK.read_text(), "output_synthe.out")


@pytest.mark.stage   # ~3 s (whole random decks); fast tier: test_readin_golden, test_readin_basicatlas_deck_vs_fortran, FREEFF/IWORDF/XFREEF checks
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2**31 - 1))
def test_readin_vs_fortran(seed):
    _check_deck(make_deck(seed), f"deck seed={seed}")
