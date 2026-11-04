"""Units of the stock (ifx -fp-model fast) emulation used by Flavor('stock').fastmath:
rcpps table + Newton division (lines/rcpps.py), the far-wing formula with one reciprocal
(LineAccumulator._asym_fast), TABVOIGT's reciprocal/SVML H tables.  The end-to-end check against the stock binary
is tests/stage/test_dfsynthe_prog.py (5012 K: within 1 step of stock in ~5e-5 of the ODF entries)."""
import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe.lines import rcpps, svml
from atlas_torch.dfsynthe.lines.accumulate import LineAccumulator
from atlas_torch.dfsynthe.lines.f32 import Flavor
from atlas_torch.dfsynthe.lines.tables import tabvoigt


def test_rcpps_table_scaling():
    t = rcpps._table_np()
    assert t.size == 4096
    one = torch.tensor([1.0, 2.0, 0.5, -4.0, 3.0e10], dtype=torch.float32)
    r = rcpps.rcpps(one)
    assert r.view(torch.int32)[0] == int(t[0])                      # input 1.0 -> table entry 0
    assert float(r[1]) == float(r[0]) / 2 and float(r[2]) == float(r[0]) * 2 and float(r[3]) == -float(r[0]) / 4
    x = torch.from_numpy((10.0 ** np.random.default_rng(0).uniform(-30, 30, 100000)).astype(np.float32))
    rel = (rcpps.rcpps(x).double() * x.double() - 1).abs().max()
    assert float(rel) < 1.5 * 2.0 ** -12                             # rcpps accuracy bound


def test_div_nr_within_two_ulp():
    rng = np.random.default_rng(1)
    a = torch.from_numpy((10.0 ** rng.uniform(-10, 10, 200000)).astype(np.float32))
    b = torch.from_numpy((10.0 ** rng.uniform(-10, 10, 200000)).astype(np.float32))
    q, e = rcpps.div_nr(a, b), a / b
    ulp = (q.view(torch.int32).long() - e.view(torch.int32).long()).abs()
    assert int(ulp.max()) <= 2 and 0 < int((ulp > 0).sum()) < 100000      # one Newton step: not always IEEE


def test_asym_fast_close_to_source_formula():
    tabs = {"h0tab": torch.zeros(1)}
    acc = LineAccumulator.__new__(LineAccumulator)
    LineAccumulator.__init__(acc, tabs, torch.zeros(1), torch.zeros(1), torch.ones(1, 1), torch.ones(1, 1),
                             torch.ones(1, 10), fastmath=True)
    rng = np.random.default_rng(2)
    profc = torch.from_numpy(rng.uniform(1, 100, (1, 300)).astype(np.float32))
    aa = torch.from_numpy(rng.uniform(2, 1e4, (1, 300)).astype(np.float32))
    kdv = torch.arange(1, 2001, dtype=torch.float32)[:, None] * torch.from_numpy(
        rng.uniform(1e-3, 1e-1, (1, 300)).astype(np.float32))
    f, s = acc._asym_fast(profc, aa, kdv.clone()), acc._asym(profc, aa, kdv.clone())
    rel = ((f.double() - s.double()) / s.double()).abs().max()
    assert float(rel) < 1e-5 and bool((f != s).any())


def test_tabvoigt_fast_tables():
    h0f, h1f, h2f, _ = tabvoigt(fl=Flavor("stock"))
    h0p, h1p, h2p, _ = tabvoigt(fl=Flavor("stock", fastmath=False))
    assert h0f[0] == 1.0 and bool(np.all(np.diff(h0f) <= 0))
    assert np.max(np.abs(h0f.astype(np.float64) - h0p) / h0p) < 1e-4
    if svml.available():
        assert int((h0f != h0p).sum()) > 500                          # reciprocal VV + SVML: ~790 of 2001 differ
