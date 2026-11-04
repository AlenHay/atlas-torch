"""Stage test: PROGRAM SPECTRV (BasicATLAS-patched) from the stage-isolated inputs of the reference runs.

Inputs: $ATLAS_EXT/work/<window>_stock/pre_spectrv (fort.9 from SYNTHE, fort.10 = xnfpelsyn.dat, ...) and
the fort.25 card. The post-READIN State comes from the golden trace of the 500-510 nm run (READIN reads
only the model deck, the same output_synthe.out for both windows) until atlas_torch.synthe.readin lands.

Reference outputs: spectrum.bin of the stock run (gfortran -O3) and, for the bitwise check, the gf-O0
build run here on the same inputs. Where they differ (380-420 nm, 15 Ca II H/K core points at <= 7.3e-15)
the -O3 build vectorises EXP with glibc libmvec (_ZGVbN2v_exp), which is not bitwise scalar exp.
"""
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.state import State
from atlas_torch.synthe import spectrv_prog as sp

pytestmark = pytest.mark.stage   # whole-program / big-data tier (tests/stage/)
# Quick unit checks of SPECTRV pieces (W, READ_FW_D, POWI, EDGE_INDEX) live in tests/oracle/test_spectrv_units.py.

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
WINDOWS = ["sun_500_510", "sun_380_420"]
DECK = EXT / "scratch_sun" / "output_synthe.out"
GF_O0 = EXT / "builds" / "gf-O0" / "bin" / "spectrv.exe"
HDR = 3306                                   # header record of fort.7 incl. markers (NEDGE array of 377)


def _need(p):
    if not Path(p).exists():
        pytest.skip(f"missing {p}")
    return Path(p)


@pytest.fixture(scope="module")
def state():
    p = _need(EXT / "traces" / "sun_500_510" / "spectrv" / "readin_0_exit.npz")
    with np.load(p) as d:
        return State.from_trace({k: d[k] for k in d.files})


def _workdir(tmp: Path, win: str) -> Path:
    src = _need(EXT / "work" / f"{win}_stock")
    tmp.mkdir(parents=True, exist_ok=True)
    for f in (src / "pre_spectrv").glob("fort.*"):
        shutil.copy(f, tmp / f.name)
        os.chmod(tmp / f.name, 0o644)
    shutil.copy(src / "fort.25", tmp / "fort.25")
    return tmp


def _run_fortran(wd: Path, exe: Path = GF_O0) -> float:
    _need(exe); _need(DECK)
    t = time.perf_counter()
    with open(DECK) as fi, open(wd / "spectrv.out", "w") as fo:
        subprocess.run([str(exe)], stdin=fi, stdout=fo, cwd=wd, check=True)
    return time.perf_counter() - t


def _points(b: bytes, n: int) -> np.ndarray:
    return np.frombuffer(b[HDR:HDR + 24 * n], dtype=[("m", "<i4"), ("q", "<f8", 2), ("t", "<i4")])["q"]


def _rel(a, b):
    return float(np.max(np.abs(a - b) / np.abs(b), initial=0.0))


@pytest.fixture(scope="module")
def runs(state, tmp_path_factory):
    """Port + gf-O0 Fortran on each window (cached for the module)."""
    out = {}
    for win in WINDOWS:
        wd = _workdir(tmp_path_factory.mktemp(win), win)
        t_f = _run_fortran(wd)
        t = time.perf_counter()
        res = sp.spectrv(state, sp.load_inputs(wd))
        t_p = time.perf_counter() - t
        out[win] = dict(wd=wd, res=res, bytes=res.to_bytes(), t_port=t_p, t_fortran=t_f)
    return out


@pytest.mark.parametrize("win", WINDOWS)
def test_spectrum_vs_stock(runs, win, record_property):
    r = runs[win]
    ref_path = EXT / "work" / f"{win}_stock" / "spectrum.bin"
    ref = ref_path.read_bytes()
    got = r["bytes"]
    assert len(got) == len(ref)
    assert got[:HDR] == ref[:HDR], "header record differs"
    assert got[-12:] == ref[-12:], "N910 record differs"
    n = r["res"].q.shape[0]
    a, e = _points(got, n), _points(ref, n)
    ef, ec = _rel(a[:, 0], e[:, 0]), _rel(a[:, 1], e[:, 1])
    nbad = int((a != e).any(axis=1).sum())
    record_property("flux_max_rel", ef); record_property("cont_max_rel", ec); record_property("npts_differ", nbad)
    print(f"\n{win}: {n} points, flux max rel {ef:.3e}, cont max rel {ec:.3e}, {nbad} points not bitwise; "
          f"port {r['t_port']:.2f} s, Fortran gf-O0 {r['t_fortran']:.2f} s")
    assert ef <= 1e-13 and ec <= 1e-13
    if win == "sun_500_510":
        assert got == ref                     # -O0 and -O3 agree here: the port is bitwise


@pytest.mark.parametrize("win", WINDOWS)
def test_spectrum_bitwise_vs_gfortran_O0(runs, win):
    r = runs[win]
    assert r["bytes"] == (r["wd"] / "fort.7").read_bytes()


@pytest.mark.parametrize("win", WINDOWS)
def test_stdout_after_readin(runs, win):
    """Everything SPECTRV itself prints (card echo; then from the TAPE10 block to the end, W included)."""
    ref = (EXT / "work" / f"{win}_stock" / "spectrv.out").read_text().splitlines()
    mine = runs[win]["res"].stdout.splitlines()
    i = max(k for k, line in enumerate(ref) if line.strip() == "TAPE10") - 2
    assert ref[0] == mine[0]
    assert ref[i:] == mine[1:]


def test_chunking_and_libm_modes_invariant(runs, state):
    r = runs["sun_500_510"]
    a = sp.spectrv(state, sp.load_inputs(r["wd"]), chunk=1000, libm="torch").to_bytes()
    assert a == r["bytes"]


def test_mask_bin(state, tmp_path):
    """BasicATLAS patch: mask.bin bytes != 0 keep NU; masked NU>1 are skipped (not written), NU=1 always."""
    wd = _workdir(tmp_path / "m", "sun_500_510")
    rng = np.random.default_rng(3)
    n = 11882
    m = (rng.random(n) < 0.3).astype(np.int8)
    m[0] = 0; m[5] = -3; m[n - 1] = 1
    m.tofile(wd / "mask.bin")
    _run_fortran(wd)
    res = sp.spectrv(state, sp.load_inputs(wd))
    assert res.nu[0] == 1 and len(res.nu) == 1 + int((m[1:] != 0).sum())
    assert res.to_bytes() == (wd / "fort.7").read_bytes()
    ref = (wd / "spectrv.out").read_text().splitlines()
    i = max(k for k, line in enumerate(ref) if line.strip() == "TAPE10") - 2
    assert ref[i:] == res.stdout.splitlines()[1:]


def test_mask_bin_too_short(state, tmp_path):
    wd = _workdir(tmp_path / "s", "sun_500_510")
    np.ones(100, np.int8).tofile(wd / "mask.bin")
    with pytest.raises(EOFError):
        sp.spectrv(state, sp.load_inputs(wd))


def _synthetic_lines(rng, nl):
    waves = np.r_[np.linspace(300, 900, 25), np.linspace(500, 510, 10)[::-1], [393.37, 396.85, 1200., 250., 656.28]]
    recs = []
    for k in range(nl):
        wl = float(waves[k]) * (-1 if k % 7 == 3 else 1) * (0 if k == 11 else 1)
        b = struct.pack("<ddd", wl, 1234.567 * k, 20000.0 + k)
        b += (f"LAB{k:<7}"[:10] + f"LP{k:<8}"[:10] + "OTH1      " + "OTH2      ").encode()
        b += struct.pack("<ddd", float(waves[k]), 0.0, 0.0)
        b += struct.pack("<ifff", 2601 + k, 8.1, -6.2, -7.5) + b"REF1"
        b += struct.pack("<iii", 3, 4, 1) + struct.pack("<f", -0.12) + struct.pack("<i", 2) + struct.pack("<f", -1.5)
        b += np.array([-1.234 + 0.1 * k, 2.5, 1.5, 26.01, 0, 0, 0, 0, 0, 0.01, -0.05, 0.1, 0.2, 0.3, 0, 0, 0],
                      dtype="<f4").tobytes()
        al = np.zeros(99, np.float32)
        al[:72] = np.float32(10.0 ** rng.uniform(-6, 2 + 4 * (k % 5 == 0), 72)) * (k % 9 != 4)
        recs.append(b + al.tobytes())
    return recs


def test_nlines_block(state, tmp_path):
    """LINOUT>0 line-centre records (fort.7 tail, CENTER/CONCEN) and the fort.17 listing, synthetic lines
    (wild per-layer ALINEC: non-monotonic TAUNU, thick cores, WL<0, WL=0, IEDGE resets) vs the Fortran."""
    wd = _workdir(tmp_path / "l", "sun_500_510")
    raw = (wd / "fort.9").read_bytes()
    assert struct.unpack("<iii", raw[-12:]) == (4, 0, 4)
    recs = _synthetic_lines(np.random.default_rng(3), 40)
    out = bytearray(raw[:-12]) + struct.pack("<iii", 4, len(recs), 4)
    for b in recs:
        out += struct.pack("<i", len(b)) + b + struct.pack("<i", len(b))
    (wd / "fort.9").write_bytes(bytes(out))
    _run_fortran(wd)
    res = sp.spectrv(state, sp.load_inputs(wd))
    assert res.n910 == 40
    assert res.to_bytes() == (wd / "fort.7").read_bytes()
    assert res.fort17 == (wd / "fort.17").read_text()


@pytest.mark.parametrize("win", WINDOWS)
def test_libmvec_mode_bitwise_vs_stock(runs, state, win):
    """libm="libmvec" (loop 20's EXP through libmvec _ZGVbN2v_exp on layer pairs, as the stock -O3 build)
    reproduces the stock spectrum.bin byte for byte, including the Ca II H/K cores where -O0 differs."""
    ref = _need(EXT / "work" / f"{win}_stock" / "spectrum.bin").read_bytes()
    got = sp.spectrv(state, sp.load_inputs(runs[win]["wd"]), libm="libmvec").to_bytes()
    n = (len(ref) - HDR - 12) // 24
    print(f"\n{win}: libmvec mode vs stock: {int((_points(got, n) != _points(ref, n)).sum())} values differ; "
          f"auto (-O0) mode vs stock: {int((_points(runs[win]['bytes'], n) != _points(ref, n)).sum())}")
    assert got == ref


@pytest.mark.parametrize("libm", ["libmvec", "glibc"])
@pytest.mark.parametrize("win", WINDOWS)
def test_native_point_loop_equals_torch(runs, state, win, libm):
    """The C point loop (atlas_torch.synthe.native spectrv.c, IFSURF=1) == the torch JOSH path, byte for byte,
    for both libm modes it implements, single- and multi-threaded, with a mask."""
    from atlas_torch.synthe.native import load
    if load("spectrv") is None:
        pytest.skip("spectrv kernel not built (make native)")
    inp = sp.load_inputs(runs[win]["wd"])
    ref = sp.spectrv(state, inp, libm=libm, native=False).to_bytes()
    nt = torch.get_num_threads()
    try:
        for t in (1, 4):
            torch.set_num_threads(t)
            assert sp.spectrv(state, inp, libm=libm, native=True).to_bytes() == ref
    finally:
        torch.set_num_threads(nt)
    m = (np.random.default_rng(5).random(inp.fort9.stat().st_size) < 0.4).astype(np.int8)
    inp.mask = m
    assert sp.spectrv(state, inp, libm=libm, native=True).to_bytes() == \
        sp.spectrv(state, inp, libm=libm, native=False).to_bytes()
