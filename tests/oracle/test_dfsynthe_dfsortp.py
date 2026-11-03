"""DFSORTP and SEPARATEDF ports, byte-exact against the stock Fortran executables and the stock solar ODFs.

dfsortp: the port turns an ODF step file (fort.15-type records) into exactly the bytes of dfsortp.exe's fort.2
(fixed 32-byte records, '(1x,i10,1x,12i8)').
separatedf: the 57 per-temperature sortp files are rebuilt from the stock p00big{v}/p00lit{v} (separatedf ignores
the IDFOUT labels); the port must reproduce the stock .bdf files byte for byte, and the stock separatedf.exe run on
the same rebuilt files must too (which validates the rebuilt inputs).
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from atlas_torch.dfsynthe import odf_io
from atlas_torch.dfsynthe.dfsortp import dfsortp, dfsortp_arrays
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.separatedf import separatedf
from atlas_torch.io.formats import read_bdf

BIN = S.EXT / "BasicATLAS" / "bin"
SCR = "/tmp/claude-1000" if os.path.isdir("/tmp/claude-1000") else None


def _run(exe, cwd):
    env = dict(os.environ)
    d = S.ifx_libdir()
    if d is None or not (BIN / exe).exists():
        pytest.skip("stock executables / Intel runtime missing")
    env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run([str(BIN / exe)], cwd=cwd, env=env, check=True, stdout=subprocess.DEVNULL)


def _check_dfsortp(src: Path, td: Path):
    shutil.copy(src, td / "fort.1")
    _run("dfsortp.exe", td)
    dfsortp(src, td / "port.asc")
    assert (td / "port.asc").read_bytes() == (td / "fort.2").read_bytes()


@pytest.mark.parametrize("unit", [15, 16, 20])
def test_dfsortp_on_debug_run(unit):
    src = S.TRACE_RUN / f"fort.{unit}"
    if not src.exists():
        pytest.skip("debug run outputs missing")
    with tempfile.TemporaryDirectory(dir=SCR) as td:
        _check_dfsortp(src, Path(td))


@settings(max_examples=5, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), lo=st.integers(-32768, 0), hi=st.integers(1, 32767))
def test_dfsortp_random(seed, lo, hi):
    rng = np.random.default_rng(seed)
    idf = rng.integers(0, 2**31 - 1, 25 * 1540).astype(np.int32)
    steps = rng.integers(lo, hi, (25 * 1540, 12)).astype(np.int16)
    with tempfile.TemporaryDirectory(dir=SCR) as td:
        td = Path(td)
        odf_io.write_odf_bin(td / "in.bin", idf, steps)
        _check_dfsortp(td / "in.bin", td)


def test_separatedf_stock():
    if not (S.STOCK / "p00big0.bdf").exists():
        pytest.skip("stock ODFs missing")
    for v in (0, 4):
        big = read_bdf(S.STOCK / f"p00big{v}.bdf")             # [328, 57, 25, 12]
        lit = read_bdf(S.STOCK / f"p00lit{v}.bdf")
        steps = np.concatenate([big, lit])                       # [1540, 57, 25, 12]
        with tempfile.TemporaryDirectory(dir=SCR) as td:
            td = Path(td)
            files = []
            for it in range(57):
                idf = (v * 100000000 + np.arange(1, 1541)[:, None] * 10000 + (it + 1) * 100
                       + np.arange(1, 26)[None, :]).reshape(-1)
                p = td / f"fort.{it + 10}"
                p.write_bytes(odf_io.format_sortp(idf, steps[:, it].reshape(-1, 12)))
                files.append(p)
            separatedf(files, td / "big.bdf", td / "lit.bdf")
            assert (td / "big.bdf").read_bytes() == (S.STOCK / f"p00big{v}.bdf").read_bytes()
            assert (td / "lit.bdf").read_bytes() == (S.STOCK / f"p00lit{v}.bdf").read_bytes()
            _run("separatedf.exe", td)
            assert (td / "fort.2").read_bytes() == (S.STOCK / f"p00big{v}.bdf").read_bytes()
            assert (td / "fort.3").read_bytes() == (S.STOCK / f"p00lit{v}.bdf").read_bytes()


def test_dfsortp_separatedf_roundtrip():
    """dfsortp_arrays inverts the (nu, ip) <-> (ip, nu) record order used by DFCALC."""
    rng = np.random.default_rng(3)
    idf = np.arange(25 * 1540, dtype=np.int32)
    st_ = rng.integers(-5000, 5000, (25 * 1540, 12)).astype(np.int16)
    a, b = dfsortp_arrays(idf, st_)
    assert np.array_equal(b.reshape(1540, 25, 12).transpose(1, 0, 2).reshape(-1, 12), st_)
    assert np.array_equal(a.reshape(1540, 25).T.reshape(-1), idf)
