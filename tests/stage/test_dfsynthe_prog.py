"""Stage tests for the torch DFSYNTHE program (atlas_torch.dfsynthe.dfsynthe), one temperature (5012 K, ITEFF=21,
solar xnfpdf.dat / xnfpdfmax.dat of the stock BasicATLAS run).

1. stock flavour (FTZ environment + the ifx -fp-model fast Lyman-alpha term, see test_dfsynthe_lya_fastmath.py)
   + default (correctly rounded) transcendentals vs the stock ODF (rows it=21 of p00big{v}/p00lit{v}, ifx -O2
   fp-model fast): max |difference| in integer steps and the fraction of differing entries per velocity.
   Measured: max 1 step, 5.4e-5 of entries (ifx-O3 vs stock: max 1, 2e-7).  Without the fast-math emulation the
   port reproduces ifx -fp-model precise(-ftz) instead, which differs from stock by up to 18 steps here (the
   Lyman-alpha NaN wing of the stock binary).
1b. Flavor('stock', fastmath=False) vs the ifx-precise-ftz build: max 1 step (measured 1 step in 2e-6).
1c. hot temperatures 12589 K (ITEFF=33) and 112202 K (52) vs stock: max 1 step, measured 2-8e-5 of entries differ.
   Before the two hot-temperature stock forms were emulated (test_dfsynthe_hot_fastmath.py) the port gave 14-15
   steps at 12589 K (the H2+ cutoff flushed by FTZ: the ifx-precise-ftz result) and 119 at 112202 K (EMERGE
   rounding: He II Lyman far wings in layer 25), like the ifx -O0/precise builds.
2. O0 flavour + Intel libimf (the debug executable's semantics) vs the debug run's own fort.15/16/17/18/20:
   bit-identical ODF records (marked `full`: ~10 min, mostly ctypes libm calls).
3. the five output files (vt*.bin) are byte-compatible records (32-byte, zero padded) and dfsortp of them equals
   the stock dfsortp format.
"""
import os
import time

import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe import odf_io
from atlas_torch.dfsynthe.dfsynthe import Config, run_temperature, write_temperature
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines.f32 import Flavor
from atlas_torch.io.formats import read_bdf

pytestmark = [pytest.mark.stage,
              pytest.mark.skipif(not (S.DATA.exists() and (S.STOCK / "p00big0.bdf").exists()),
                                 reason="needs the BasicATLAS line lists and the stock solar ODF run")]
ITEFF = 21
SPREAD_MAX = 30          # integer steps vs the -O0 debug run (Fortran builds differ by up to 18)
SPREAD_FRAC = 3e-3       # fraction of entries (Fortran builds: up to 1.1e-3)
STOCK_MAX = 2            # vs stock (measured 1 with the fast-math emulations)
STOCK_FRAC = 2e-4        # (measured 5.4e-5)
PRECISE = S.EXT / "runs" / "ifx-precise-ftz" / "odf" / "solar"


def _stock_rows(v, d=S.STOCK):
    big = read_bdf(d / f"p00big{v}.bdf")[:, ITEFF - 1]
    lit = read_bdf(d / f"p00lit{v}.bdf")[:, ITEFF - 1]
    return np.concatenate([big, lit])                       # [1540, 25, 12]


def _as_rows(isteps):
    return isteps.reshape(25, 1540, 12).transpose(1, 0, 2)


@pytest.fixture(scope="module")
def stock_run(tmp_path_factory):
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    t0 = time.time()
    res = run_temperature(S.STOCK / "xnfpdf.dat", S.STOCK / "xnfpdfmax.dat", S.DATA, ITEFF,
                          Config(flavor=Flavor("stock")))
    res["info"]["wall"] = time.time() - t0
    return res


def test_vs_stock_odf(stock_run):
    info = stock_run["info"]
    print(f"\nwall {info['wall']:.0f} s, timing {info['timing']}, counts {info['counts']}")
    for v in odf_io.VTURBS:
        d = np.abs(_as_rows(stock_run[v][1]).astype(int) - _stock_rows(v).astype(int))
        print(f"vt={v}: max |diff| {d.max()} steps, fraction differing {(d > 0).mean():.2e}")
        assert d.max() <= STOCK_MAX and (d > 0).mean() <= STOCK_FRAC, v


@pytest.mark.parametrize("iteff", [33, 52])
def test_hot_vs_stock_odf(iteff):
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    res = run_temperature(S.STOCK / "xnfpdf.dat", S.STOCK / "xnfpdfmax.dat", S.DATA, iteff,
                          Config(flavor=Flavor("stock")))
    for v in odf_io.VTURBS:
        big = read_bdf(S.STOCK / f"p00big{v}.bdf")[:, iteff - 1]
        lit = read_bdf(S.STOCK / f"p00lit{v}.bdf")[:, iteff - 1]
        d = np.abs(_as_rows(res[v][1]).astype(int) - np.concatenate([big, lit]).astype(int))
        print(f"ITEFF={iteff} vt={v}: max |diff| {d.max()} steps, fraction differing {(d > 0).mean():.2e}")
        assert d.max() <= 1 and (d > 0).mean() <= STOCK_FRAC, v


@pytest.mark.skipif(not (PRECISE / "p00lit8.bdf").exists(), reason="ifx-precise-ftz solar run missing")
def test_precise_flavor_vs_ifx_precise():
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    res = run_temperature(S.STOCK / "xnfpdf.dat", S.STOCK / "xnfpdfmax.dat", S.DATA, ITEFF,
                          Config(flavor=Flavor("stock", fastmath=False)))
    for v in odf_io.VTURBS:
        d = np.abs(_as_rows(res[v][1]).astype(int) - _stock_rows(v, PRECISE).astype(int))
        print(f"precise flavour vs ifx-precise-ftz vt={v}: max {d.max()}, fraction {(d > 0).mean():.2e}")
        assert d.max() <= 1 and (d > 0).mean() <= 1e-5, v


def test_vs_debug_run(stock_run):
    """Same run vs the -O0 debug executable's output (another member of the Fortran's spread)."""
    if not (S.TRACE_RUN / "fort.15").exists():
        pytest.skip("debug run outputs missing")
    for v in odf_io.VTURBS:
        ref = odf_io.read_odf_bin(S.TRACE_RUN / f"fort.{odf_io.VT_UNITS[v]}")
        assert np.array_equal(ref["idfout"], stock_run[v][0])
        d = np.abs(ref["isteps"].astype(int) - stock_run[v][1].astype(int))
        print(f"vs debug vt={v}: max |diff| {d.max()}, fraction {(d > 0).mean():.2e}")
        assert d.max() <= SPREAD_MAX and (d > 0).mean() <= SPREAD_FRAC


def test_output_files(stock_run, tmp_path):
    paths = write_temperature(stock_run, tmp_path, "5012")
    for v in odf_io.VTURBS:
        b = (tmp_path / f"dfp00t5012vt{v}.bin").read_bytes()
        assert len(b) == 32 * 25 * 1540
        rec = np.frombuffer(b, odf_io.ODF_RECORD_DTYPE)
        assert np.all(rec["pad"] == 0)
        lines = paths[v].read_text().splitlines()
        assert len(lines) == 25 * 1540 and all(len(x) == 108 for x in lines[:100])


@pytest.mark.full
@pytest.mark.skipif(not S.available(), reason="Intel runtime missing")
def test_exact_vs_debug_executable():
    """O0 flavour + Intel libimf reproduces the debug (ifx -O0) executable's ODF files bit for bit."""
    if not (S.TRACE_RUN / "fort.15").exists():
        pytest.skip("debug run outputs missing")
    res = run_temperature(S.STOCK / "xnfpdf.dat", S.STOCK / "xnfpdfmax.dat", S.DATA, ITEFF,
                          Config(flavor=S.intel_flavor()))
    for v in odf_io.VTURBS:
        ref = odf_io.read_odf_bin(S.TRACE_RUN / f"fort.{odf_io.VT_UNITS[v]}")
        d = np.abs(ref["isteps"].astype(int) - res[v][1].astype(int))
        print(f"exact-mode vt={v}: max |diff| {d.max()}, differing {(d > 0).sum()}")
        assert d.max() == 0, v
