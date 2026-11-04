"""Stage test: the whole PROGRAM XNFDF (DFSYNTHE) port vs the Fortran.

For each mix with a stock run (runs/stock/odf/<mix>/: solar, alpha) the port writes xnfpdf.dat
(fort.10) and xnfpdfmax.dat (fort.11) from the stock stdin (heredoc of xnfdf.com) and the same
molecules.dat / continua.dat, and is compared with
  1. the ifx -O0 build of the same source (the oracle semantics): byte-identical files required
     (solar: the traced run traces/odf_solar/xw = ifx -g -O0; other mixes: builds/ifx-O0/bin/xnfdf.exe is
     run in a scratch dir when the Intel runtime is available);
     with ftz=False (IEEE gradual underflow), and the ifx -O0 -ftz build (builds/ifx-O0-ftz) with ftz=True
     (the default: stock's flush-to-zero/denormals-are-zero environment, dfsynthe/ftz.py);
  2. the stock build (ifx -O2, fp-model fast, flush-to-zero): max relative error per array is reported;
     the port must stay within the Fortran's own O0-vs-O2 spread: normal REAL*4 values within 2e-6
     relative (fast-math reassociation + NMOLEC convergence: measured 1.4e-6 for xnfpel4/xnfdopmax),
     log10 continua within 1e-6 dex (2.3e-6 relative in the opacity; measured 4.8e-7 = the populations'
     spread), and values that stock flushes to zero (single-precision
     denormals, FTZ) are counted separately: none may remain with ftz=True (13586 xnfpel4 / 92651 xnfdopmax
     values for solar with ftz=False).
Run: pytest -m stage tests/stage/test_xnfdf_prog.py -s  (prints the error tables).
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe.xnf import oracle_support as osup
from atlas_torch.dfsynthe.xnfdf import run
from atlas_torch.dfsynthe.xnf.xnfio import read_xnfpdf, read_xnfpdfmax

pytestmark = pytest.mark.stage   # whole-program / big-data tier (tests/stage/)

EXT = osup.ext_dir()
TINY = np.float32(1.1754944e-38)


def _stock(mix):
    d = EXT / "runs" / "stock" / "odf" / mix
    if not (d / "xnfpdf.dat").exists() or not (d / "xnfdf.com").exists():
        pytest.skip(f"no stock xnfdf run for {mix}")
    com = (d / "xnfdf.com").read_text()
    stdin = com.split("<<EOF>xnfdf.out\n", 1)[1].split("\nEOF", 1)[0] + "\n"
    cps = dict((Path(dst).name, Path(src)) for src, dst in re.findall(r"^cp\s+(\S+)\s+(fort\.\d+)\s*$", com, re.M))
    return d, stdin, cps


def _stats(name, x, y, out):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    s = out.setdefault(name, dict(n=0, ndiff=0, ftz=0, nrm_diff=0, max_rel=0.0, max_abs=0.0, max_ulp=0))
    s["n"] += x.size
    diff = x != y
    s["ndiff"] += int(diff.sum())
    if x.dtype.kind == "f":
        ftz = diff & (((x == 0) & (np.abs(y) < TINY)) | ((y == 0) & (np.abs(x) < TINY)))
        s["ftz"] += int(ftz.sum())
        nrm = diff & ~ftz & (np.abs(x) >= TINY) & (np.abs(y) >= TINY)
        s["nrm_diff"] += int(nrm.sum())
        if nrm.any():
            xa, ya = x[nrm].astype(np.float64), y[nrm].astype(np.float64)
            s["max_rel"] = max(s["max_rel"], float(np.max(np.abs(xa - ya) / np.maximum(np.abs(xa), np.abs(ya)))))
            s["max_abs"] = max(s["max_abs"], float(np.max(np.abs(xa - ya))))
            if x.dtype == np.float32:
                xi = x[nrm].view(np.int32).astype(np.int64)
                yi = y[nrm].view(np.int32).astype(np.int64)
                xi = np.where(xi < 0, -(xi & 0x7FFFFFFF), xi)
                yi = np.where(yi < 0, -(yi & 0x7FFFFFFF), yi)
                s["max_ulp"] = max(s["max_ulp"], int(np.max(np.abs(xi - yi))))


def compare(port10, port11, ref10, ref11):
    A, B = read_xnfpdf(port10), read_xnfpdf(ref10)
    MA, MB = read_xnfpdfmax(port11), read_xnfpdfmax(ref11)
    assert len(A) == len(B) == len(MA) == len(MB)
    out = {}
    for a, b in zip(A, B):
        assert (a.nt, a.teff, a.glog, a.title, a.nedge) == (b.nt, b.teff, b.glog, b.title, b.nedge)
        assert np.array_equal(a.idmol, b.idmol) and np.array_equal(a.momass, b.momass)
        _stats("edges", a.edges, b.edges, out)
        _stats("freqset", a.freqset, b.freqset, out)
        for k in a.state:
            _stats("state." + k, a.state[k], b.state[k], out)
        _stats("xnfpel4", a.xnfpel, b.xnfpel, out)
        _stats("dopple4", a.dopple, b.dopple, out)
        _stats("continall4", a.continall, b.continall, out)
    for a, b in zip(MA, MB):
        _stats("xnfdopmax", a.xnfdopmax, b.xnfdopmax, out)
    return out


def report(title, out):
    lines = [title, f"{'array':14s} {'n':>9s} {'ndiff':>8s} {'ftz':>7s} {'nrm_diff':>8s} {'max_rel':>10s} "
                    f"{'max_abs':>10s} {'max_ulp':>8s}"]
    for k, s in out.items():
        lines.append(f"{k:14s} {s['n']:9d} {s['ndiff']:8d} {s['ftz']:7d} {s['nrm_diff']:8d} {s['max_rel']:10.3e} "
                     f"{s['max_abs']:10.3e} {s['max_ulp']:8d}")
    return "\n".join(lines)


def _o0_reference(mix, d, stdin, cps, tmp, variant="ifx-O0"):
    """(fort.10, fort.11) written by the ifx -O0 (or -O0 -ftz) build of xnfdf.for for this mix, or None."""
    if mix == "solar" and variant == "ifx-O0":
        xw = EXT / "traces" / "odf_solar" / "xw"
        if (xw / "fort.10").exists() and (xw / "stdin.txt").read_text() == stdin:
            return xw / "fort.10", xw / "fort.11"
    exe = EXT / "builds" / variant / "bin" / "xnfdf.exe"
    lib = osup.ifx_lib_dir()
    if not exe.exists() or lib is None:
        return None
    w = tmp / variant
    w.mkdir()
    for dst, src in cps.items():
        shutil.copy(src, w / dst)
    env = dict(os.environ, LD_LIBRARY_PATH=lib)
    subprocess.run([str(exe)], input=stdin.encode(), cwd=w, env=env, stdout=subprocess.DEVNULL, check=True,
                   timeout=900)
    return w / "fort.10", w / "fort.11"


@pytest.fixture
def big_tmp():
    """xnfpdf*.dat (66 MB) and the Fortran reference run dir go to $ATLAS_EXT/scratch, not the /tmp tmpfs."""
    import tempfile
    base = EXT / "scratch"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="test_xnfdf_prog_", dir=base))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.stage
@pytest.mark.parametrize("ftz", [True, False], ids=["ftz", "ieee"])
@pytest.mark.parametrize("mix", ["solar", "alpha"])
def test_xnfdf_prog(mix, ftz, big_tmp):
    tmp_path = big_tmp
    torch.set_default_dtype(torch.float64)
    d, stdin, cps = _stock(mix)
    p10, p11 = tmp_path / "xnfpdf.dat", tmp_path / "xnfpdfmax.dat"
    run(stdin, continua_text=cps["fort.17"].read_text(), molecules_text=cps["fort.2"].read_text(),
        out10=p10, out11=p11, ftz=ftz)
    assert p10.stat().st_size == (d / "xnfpdf.dat").stat().st_size
    assert p11.stat().st_size == (d / "xnfpdfmax.dat").stat().st_size

    st = compare(p10, p11, d / "xnfpdf.dat", d / "xnfpdfmax.dat")
    print("\n" + report(f"== {mix}: port vs stock (ifx -O2, fp-model fast)", st))
    for k, s in st.items():
        if k == "continall4":
            assert s["max_abs"] <= 1e-6, (k, s)       # log10(kappa): 1e-6 dex = 2.3e-6 relative opacity
        else:
            assert s["max_rel"] <= 2e-6, (k, s)
        if ftz:
            assert s["ftz"] == 0, (k, s)                 # nothing left that stock flushes and we do not
    assert st["dopple4"]["ndiff"] == 0
    if not ftz:
        assert st["xnfpel4"]["ftz"] > 0                  # the IEEE run keeps the subnormals stock flushes

    variant = "ifx-O0-ftz" if ftz else "ifx-O0"
    ref = _o0_reference(mix, d, stdin, cps, tmp_path, variant)
    if ref is None:
        pytest.skip(f"no {variant} reference output for byte comparison")
    st0 = compare(p10, p11, *ref)
    print(report(f"== {mix}: port vs {variant}", st0))
    assert p10.read_bytes() == Path(ref[0]).read_bytes()
    assert p11.read_bytes() == Path(ref[1]).read_bytes()
