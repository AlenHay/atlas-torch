"""Oracle tests for READIN / READMOL (xnfdf.for) and the BEGIN-block state.

1. golden: READIN hits 0..2 of the traced solar run (COMMON first members recorded by the tracer:
   ABUND, RHOX, T, P, TEFF, TITLE, VTURB, ACCRAD, FRESET, WORD) vs the port's first three models.
2. READMOL vs the Fortran READMOL (molecules.dat of the traced run): /IFEQUA/ and /XNMOL/ CODE.
3. hypothesis: random control decks (TEFF/GRAVITY, OPACITY IFOP, ABUNDANCE SCALE/CHANGE, TITLE, MOLECULES,
   CONVECTION, TURBULENCE, READ DECK6 with random number formats, log RHOX, ACCRAD -> INTEG) fed to the
   Fortran READIN on unit 5 (fd 0 redirected) vs the port: every COMMON member READIN/BEGIN sets.
"""
import os
import tempfile
from contextlib import contextmanager

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.dfsynthe.xnf import oracle_support as osup
from atlas_torch.dfsynthe.xnf.readin import Readin, readmol

pytestmark = pytest.mark.skipif(not osup.available(), reason="libxnfdf.so or the Intel Fortran runtime missing")
RUN = osup.ext_dir() / "traces" / "odf_solar" / "xw"
STAGE = "odf_solar/xnfdf"


def _chars(arr):
    raw = np.ascontiguousarray(arr).tobytes()
    return "".join(raw[8 * i:8 * i + 1].decode("latin-1") for i in range(len(raw) // 8))


def test_readin_golden(golden):
    r = Readin((RUN / "stdin.txt").read_text(), (RUN / "fort.2").read_text())
    for hit in range(3):
        m = r.read(1)
        ex = golden(STAGE, "readin", hit, "exit")
        n = m.nrhox
        assert np.array_equal(ex["elem.abund"], m.abund)
        assert np.array_equal(ex["rhox.rhox"][:n], m.rhox[:n])
        assert np.array_equal(ex["temp.t"][:n], m.t[:n])
        assert np.array_equal(ex["state.p"][:n], m.p[:n])
        assert np.array_equal(ex["turbpr.vturb"][:n], m.vturb[:n])
        assert np.array_equal(ex["rad.accrad"][:n], m.accrad[:n])
        assert float(ex["teff.teff"].reshape(-1)[0]) == m.teff
        assert _chars(ex["junk.title"]) == "".join(m.title)
        assert float(ex["freset.freset"][0]) == m.freset[0]
        assert _chars(ex["free.word"]).rstrip() == "".join(r.free.word).rstrip()


def test_readmol_vs_fortran():
    o = osup.new_oracle()
    cwd = os.getcwd()
    try:
        os.chdir(RUN)
        o.call("readmol")
    finally:
        os.chdir(cwd)
    mol = readmol((RUN / "fort.2").read_text())
    ie, xm = o.common("ifequa"), o.common("xnmol")
    assert int(ie["nequa"][0]) == mol.nequa and int(xm["nummol"][0]) == mol.nummol
    assert int(ie["nloc"][0]) == len(mol.kcomps)
    assert np.array_equal(xm["codemol"][:mol.nummol], mol.code)
    assert np.array_equal(ie["equil"][:, :mol.nummol], mol.equil)
    assert np.array_equal(ie["kcomps"][:len(mol.kcomps)], mol.kcomps)
    assert np.array_equal(ie["locj"][:mol.nummol + 1], mol.locj)
    assert np.array_equal(ie["idequa"][:mol.nequa], mol.idequa[1:mol.nequa + 1])
    assert np.array_equal(ie["ifequa"], mol.ifequa[1:102])


@contextmanager
def unit5(text):
    """Make `text` the Fortran's unit 5 (stdin) for the duration of the block."""
    saved = os.dup(0)
    fd, path = tempfile.mkstemp(suffix=".stdin")
    try:
        os.write(fd, text.encode("latin-1"))
        os.close(fd)
        f = os.open(path, os.O_RDONLY)
        os.dup2(f, 0)
        os.close(f)
        yield
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.unlink(path)


def _num(fmt, v):
    return {"e": f"{v:.3E}", "f": f"{v:.4f}", "g": f"{v:.6g}", "int": f"{int(round(v))}"}[fmt]


@st.composite
def decks(draw):
    teff = draw(st.floats(2000., 60000.))
    g = draw(st.floats(0.5, 5.0))
    lines = ["MOLECULES " + draw(st.sampled_from(["ON", "OFF"])), "ITERATIONS 1 PRINT 0 PUNCH 0",
             "READ FREQUENCIES 1 1 1 TEST", "1 1 1",
             f"TEFF {teff:8.0f}.  GRAVITY {g:7.5f} " + draw(st.sampled_from(["LTE", "NLTE"])),
             "TITLE " + draw(st.from_regex(r"[A-Z0-9 ,.]{0,60}", fullmatch=True)),
             " OPACITY IFOP " + " ".join(str(draw(st.integers(0, 1))) for _ in range(20)),
             " CONVECTION " + draw(st.sampled_from(["ON   1.25", "OFF"])) + " TURBULENCE " +
             draw(st.sampled_from(["OFF  0.00  0.00  0.00  0.00", "ON 1.0 0.5 0.2 0.1"]))]
    scale = draw(st.sampled_from(["1.000000", "0.100000", "3.162278", "1."]))
    lines.append(f"ABUNDANCE SCALE  {scale} ABUNDANCE CHANGE 1 {draw(st.floats(.85, .95)):.5f} 2 "
                 f"{draw(st.floats(.04, .12)):.5f}")
    zs = draw(st.lists(st.integers(3, 99), min_size=0, max_size=12, unique=True))
    for k in range(0, len(zs), 6):
        lines.append(" ABUNDANCE CHANGE " + " ".join(f"{z:2d} {draw(st.floats(-12., -3.)):6.2f}" for z in zs[k:k + 6]))
    n = draw(st.integers(4, 25))
    logr = draw(st.booleans())
    fmt = draw(st.sampled_from(["e", "f", "g"]))
    lines.append(f"READ DECK6 {n} RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB")
    lr = np.sort(np.array(draw(st.lists(st.floats(-4., 4.), min_size=n, max_size=n, unique=True))))
    for j in range(n):
        rh = f"{lr[j]:.4f}" if logr else _num(fmt, 10 ** lr[j])
        t = draw(st.floats(1800., 200000.))
        p = 10 ** draw(st.floats(-4., 8.))
        xne = draw(st.sampled_from([0., 10 ** draw(st.floats(5., 16.))]))
        acc = draw(st.floats(0., 10.))
        vt = draw(st.sampled_from([0., draw(st.floats(0., 3e5))]))
        lines.append(f"  {rh}  {t:9.1f}  {_num(fmt, p)}  {xne:.3E} {draw(st.floats(0, 1e3)):.3E} {acc:.3E} {vt:.3E}")
    lines.append(f" {draw(st.floats(0., 10.)):.3f}")
    lines.append("BEGIN")
    return "\n".join(lines) + "\n"


def _eq(a, b):
    return np.array_equal(np.asarray(a, np.float64), np.asarray(b, np.float64), equal_nan=True)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(deck=decks())
def test_readin_vs_fortran(deck):
    o = osup.new_oracle()
    with unit5(deck):
        o.call("readin", 1)
    m = Readin(deck).read(1)
    n = m.nrhox
    e, t, s = o.common("elem"), o.common("temp"), o.common("state")
    assert int(o.common("rhox")["nrhox"][0]) == n
    for k in ("abund", "xabund"):
        assert _eq(e[k], getattr(m, k)), k
    assert float(e["wtmole"][0]) == m.wtmole
    for k in ("t", "tk", "hkt", "hckt", "tkev", "tlog"):
        assert _eq(t[k][:n], getattr(m, k)[:n]), k
    for k in ("p", "xne", "xnatom", "rho"):
        assert _eq(s[k][:n], getattr(m, k)[:n]), k
    assert _eq(o.common("rhox")["rhox"][:n], m.rhox[:n])
    assert _eq(o.common("rad")["prad"][:n], m.prad[:n])
    assert _eq(o.common("pzero")["pradk"][:n], m.pradk[:n])
    tp = o.common("turbpr")
    assert _eq(tp["vturb"][:n], m.vturb[:n]) and _eq(tp["pturb"][:n], m.pturb[:n])
    assert [float(tp[k][0]) for k in ("trbfdg", "trbpow", "trbsnd", "trbcon")] == [m.trbfdg, m.trbpow, m.trbsnd, m.trbcon]
    f = o.common("if")
    assert list(f["ifop"]) == m.ifop and int(f["ifmol"][0]) == m.ifmol and int(f["nlteon"][0]) == m.nlteon
    te = o.common("teff")
    assert (float(te["teff"][0]), float(te["grav"][0]), float(te["glog"][0])) == (m.teff, m.grav, m.glog)
    assert _chars(o.common("junk")["title"]) == "".join(m.title)
    assert float(o.common("junk")["xscale"][0]) == m.xscale
    assert float(o.common("conv")["mixlth"][0]) == m.mixlth
