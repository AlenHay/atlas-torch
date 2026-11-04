"""Stage test: the standard-output printouts of the DFSYNTHE chain written by the torch runners, vs stock.

BasicATLAS parses two of these printouts: meta()/meta_dfsynthe() reads the READIN abundance listing and 0XSCALE
of xnfdf.out; validate_run() the same of xnfdf.out and kapm40k2.out (compat: atlas_torch.compat.atlas).  Those
parts must be byte-identical to the Fortran's; the rest of the printouts is reproduced too and must agree line
by line in structure (every non-numeric character, every field width) with numbers differing by at most one
unit in the last printed digit (two in the 16th significant digit of XNFDF's 1PE25.15 frequencies): the stock
build is ifx -O2 fast-math, the port computes the -O0 values, so a value next to a rounding boundary of the
printed digits can print differently.

  xnfdf.out      solar, alpha: torch runner (compat.torch_programs 'xnfdf') on the stock stdin.
  kapm40k2.out   solar, alpha: torch runner (kappa9_prog.kappa9_program) for vturb 8 (the last of the five
                 BasicATLAS runs, whose printout is the one left in the ODF directory) on the stock ODF.
  dfp00t5012.out DFSYNTHE at 5012 K from the stock xnfpdf.dat: the reference is made by running the stock
                 dfsynthe.exe in a scratch directory (~1 min); the port's printout (printouts.dfsynthe_stdout with
                 selectlines_report: SELECTLINES trace and counts, per-layer MLINES) must match it.
  meta()         a directory assembled from the torch printouts (plus the stock command files) gives the same
                 meta() as the stock directory and passes validate_run() (and Settings().check_ODF for solar).
Run: pytest -m stage tests/stage/test_dfsynthe_printouts.py -s (prints the per-file difference counts).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.stage

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
STOCK = EXT / "runs" / "stock" / "odf"
BASIC = EXT / "BasicATLAS"
DFDATA = BASIC / "data" / "dfsynthe_files"
MIXES = ("solar", "alpha")

_NUM = re.compile(r"-?\d*\.\d+(?:E[+-]\d+|[+-]\d{3})?|-?\d+")


def _stock(mix: str) -> Path:
    d = STOCK / mix
    if not (d / "xnfdf.out").exists() or not (d / "kapm40k2.out").exists():
        pytest.skip(f"no stock ODF run with printouts for {mix}")
    return d


def _units(a: str, b: str) -> float:
    """|a-b| in units of the last printed digit (same-format numeric fields)."""
    def parse(t):
        m = re.match(r"(-?)(\d*)\.(\d+)(?:E([+-]\d+)|([+-]\d{3}))?$", t)
        if not m:
            return int(t), 0
        sign, ip, fp, e1, e2 = m.groups()
        return int((ip or "0") + fp) * (-1 if sign else 1), int(e1 or e2 or 0) - len(fp)
    (va, ea), (vb, eb) = parse(a), parse(b)
    lo = min(ea, eb)
    return abs(va * 10 ** (ea - lo) - vb * 10 ** (eb - lo)) / 10 ** (max(ea, eb) - lo)


def compare(port: str, ref: str) -> dict:
    """Line-by-line comparison: structural mismatches (line count, non-numeric text, field widths) and the
    numeric differences in units of the last printed digit."""
    a, b = port.split("\n"), ref.split("\n")
    res = dict(lines=len(b), nlines_port=len(a), structure=[], ndiff=0, worst=0.0, diff_lines=[])
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        res["ndiff"] += 1
        res["diff_lines"].append(i)
        tx, ty = _NUM.findall(x), _NUM.findall(y)
        if len(x) != len(y) or _NUM.sub("#", x) != _NUM.sub("#", y) or len(tx) != len(ty) or \
                any(len(p) != len(q) for p, q in zip(tx, ty)):
            res["structure"].append((i + 1, x, y))
            continue
        res["worst"] = max([res["worst"]] + [_units(p, q) for p, q in zip(tx, ty) if p != q])
    return res


def _abund(path: Path):
    from atlas_torch.compat.atlas import parse_atlas_abundances
    return parse_atlas_abundances(str(path), classic_style=False, lookbehind=1, params=["0XSCALE"])


def _begin_blocks(text: str) -> list[str]:
    """The READIN BEGIN listing lines BasicATLAS parses, plus the rest of FORMAT 1512..1561: from ' TEFF' to
    the FREQID/WBEGIN line of every model."""
    out, on = [], False
    for ln in text.split("\n"):
        if ln.startswith(" TEFF  "):
            on = True
        if on:
            out.append(ln)
        if on and (ln.startswith("0FREQID") or ln.startswith("0WBEGIN")):
            on = False
    return out


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def xnfdf_runs(tmp_path_factory):
    from atlas_torch.compat.torch_programs import PROGRAMS
    cache = {}

    def get(mix):
        if mix not in cache:
            d = _stock(mix)
            com = (d / "xnfdf.com").read_text()
            stdin = com.split("<<EOF>xnfdf.out\n", 1)[1].split("\nEOF", 1)[0] + "\n"
            cps = dict((Path(dst).name, Path(src))
                       for src, dst in re.findall(r"^cp\s+(\S+)\s+(fort\.\d+)\s*$", com, re.M))
            wd = tmp_path_factory.mktemp(f"xnfdf_{mix}")
            out = PROGRAMS["xnfdf"](wd, stdin, {2: cps["fort.2"], 17: cps["fort.17"]})
            (wd / "xnfdf.out").write_text(out, encoding="latin-1")
            cache[mix] = wd / "xnfdf.out"
        return cache[mix]
    return get


@pytest.fixture(scope="module")
def kappa9_runs(tmp_path_factory):
    from atlas_torch.kappa9 import kappa9_prog as K
    cache = {}

    def get(mix):
        if mix not in cache:
            d = _stock(mix)
            stdin = K.control_from_com(d / "kappa9v8.com")
            wd = tmp_path_factory.mktemp(f"kappa9_{mix}")
            out = K.kappa9_program(wd, stdin, {9: d / "p00big8.bdf", 2: K.default_molecules()})
            (wd / "kapm40k2.out").write_text(out, encoding="latin-1")
            cache[mix] = wd / "kapm40k2.out"
        return cache[mix]
    return get


# ------------------------------------------------------------------------------------------------ tests
@pytest.mark.parametrize("mix", MIXES)
def test_xnfdf_printout(mix, xnfdf_runs):
    port_p = xnfdf_runs(mix)
    port, ref = port_p.read_text(encoding="latin-1"), (STOCK / mix / "xnfdf.out").read_text(encoding="latin-1")
    r = compare(port, ref)
    print(f"\nxnfdf.out {mix}: {r['lines']} lines, {r['ndiff']} differ (max {r['worst']:g} unit(s) of the last "
          f"printed digit), structural mismatches {len(r['structure'])}")
    assert r["nlines_port"] == r["lines"]
    assert not r["structure"], r["structure"][:5]
    assert r["worst"] <= 2
    assert r["ndiff"] <= 0.03 * r["lines"]
    # what BasicATLAS reads: byte-identical
    assert _begin_blocks(port) == _begin_blocks(ref)
    assert len(_begin_blocks(ref)) == 57 * 19
    assert _abund(port_p) == _abund(STOCK / mix / "xnfdf.out")


@pytest.mark.parametrize("mix", MIXES)
def test_kappa9_printout(mix, kappa9_runs):
    port_p = kappa9_runs(mix)
    port, ref = port_p.read_text(encoding="latin-1"), (STOCK / mix / "kapm40k2.out").read_text(encoding="latin-1")
    r = compare(port, ref)
    print(f"\nkapm40k2.out {mix}: {r['lines']} lines, {r['ndiff']} differ (max {r['worst']:g} unit(s) of the last "
          f"printed digit), structural mismatches {len(r['structure'])}")
    assert r["nlines_port"] == r["lines"]
    assert not r["structure"], r["structure"][:5]
    assert r["worst"] <= 1
    assert r["ndiff"] <= 50
    # READMOL / READIN / deck / type* / TABT section and every KTAB line byte-identical; only NMOLEC tables differ
    head = ref.index("      DATA TABT/")
    assert port[:head] == ref[:head]
    lines = ref.split("\n")
    assert all("," not in lines[i] for i in r["diff_lines"])         # KTAB blocks (FORMAT 77) identical
    assert _begin_blocks(port) == _begin_blocks(ref)
    assert _abund(port_p) == _abund(STOCK / mix / "kapm40k2.out")


def _stock_dfsynthe_printout(tmp: Path, iteff: int) -> Path:
    exe = BASIC / "bin" / "dfsynthe.exe"
    ref = tmp / "stock" / f"dfp{iteff}.out"
    if not exe.exists():
        pytest.skip("stock dfsynthe.exe missing")
    wd = ref.parent
    wd.mkdir(parents=True, exist_ok=True)
    links = {10: STOCK / "solar" / "xnfpdf.dat", 22: STOCK / "solar" / "xnfpdfmax.dat", 11: DFDATA / "lowlinesdf.bin",
             21: DFDATA / "highlinesdf.bin", 31: DFDATA / "diatomicsdf.bin", 41: DFDATA / "tiolinesdf.bin",
             43: DFDATA / "h2olinesdf.bin", 51: DFDATA / "nltelinesdf.bin"}
    for u, p in links.items():
        os.symlink(p, wd / f"fort.{u}")
    sw = "0" * (iteff - 1) + "1" + "0" * (57 - iteff)
    with open(ref, "wb") as out:
        r = subprocess.run([str(exe)], input=(sw + "\n").encode(), stdout=out, stderr=subprocess.PIPE, cwd=wd,
                           timeout=1800)
    if r.returncode != 0:
        pytest.skip(f"stock dfsynthe.exe failed to run here (rc {r.returncode}): {r.stderr[-300:]!r}")
    for f in wd.glob("fort.*"):
        f.unlink()
    return ref


def test_dfsynthe_printout(tmp_path):
    import torch
    from atlas_torch.dfsynthe import printouts as P
    from atlas_torch.dfsynthe.ftz import scope
    from atlas_torch.dfsynthe.lines.f32 import Flavor
    _stock("solar")
    iteff = 21                                                  # 5012 K
    ref = _stock_dfsynthe_printout(tmp_path, iteff).read_text(encoding="latin-1")
    xp, xm = STOCK / "solar" / "xnfpdf.dat", STOCK / "solar" / "xnfpdfmax.dat"
    torch.set_num_threads(1)
    with scope(True):
        rep = P.selectlines_report(xp, xm, DFDATA, iteff, Flavor("stock", fastmath=True))
    port = P.dfsynthe_stdout(ref.split("\n")[0][1:], xp, rep)
    r = compare(port, ref)
    print(f"\ndfp00t5012.out: {r['lines']} lines, {r['ndiff']} differ (max {r['worst']:g} unit(s)), "
          f"structural mismatches {len(r['structure'])}")
    assert r["nlines_port"] == r["lines"]
    assert not r["structure"], r["structure"][:5]
    # all differences are in the VTTAB table (the port's REAL*4 VTPROF vs stock's fast-math VTTAB, see
    # printouts.vttab_lines); intervals, SELECTLINES trace and counts, per-layer MLINES are identical
    lines = ref.split("\n")
    vt0 = next(i for i, ln in enumerate(lines) if ln.startswith("  1 KM/S"))
    vt1 = next(i for i, ln in enumerate(lines) if ln.startswith(" TEFF="))
    assert all(vt0 <= i < vt1 for i in r["diff_lines"])
    assert r["worst"] <= 1
    assert any("SPECTRUM COMPLETED WITH" in ln for ln in port.split("\n"))


@pytest.mark.parametrize("mix", MIXES)
def test_meta_of_torch_odf_dir(mix, xnfdf_runs, kappa9_runs, tmp_path):
    """compat meta()/validate_run() on an ODF directory whose printouts come from the torch runners."""
    from atlas_torch.compat.atlas import meta, validate_run
    from atlas_torch.compat.settings import Settings
    d = _stock(mix)
    od = tmp_path / "odf"
    od.mkdir()
    for f in d.iterdir():
        if f.suffix == ".com":
            shutil.copy(f, od / f.name)
    shutil.copy(xnfdf_runs(mix), od / "xnfdf.out")
    shutil.copy(kappa9_runs(mix), od / "kapm40k2.out")
    m_port, m_stock = meta(str(od)), meta(str(d))
    assert m_port == m_stock
    assert m_port["type"] == "DFSYNTHE"
    assert validate_run(str(od), silent=True) is True
    if mix == "solar":
        Settings().check_ODF(m_port)
