"""ODF construction pieces (v2/odf): grid and intervals, distribution steps, turbulent broadening, line decoding,
against the exact port's DFSYNTHE tables and SELECTLINES, and analytic results."""
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.v2.odf.distribution import dfcalc_round, distribution_steps
from atlas_torch.v2.odf.grid import DF_GRID, interval_index_ranges, segments, step_bounds
from atlas_torch.v2.odf.turbulence import broaden, turbulence_kernel
from atlas_torch.v2.opacity.odf import STEP_FRACTIONS, IntervalSet

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


@pytest.fixture(scope="module")
def dftables():
    from atlas_torch.dfsynthe.lines.tables import build_tables
    return build_tables()


def test_grid_and_intervals(dftables):
    """DF_GRID = MAIN's grid; interval index ranges = DFINTERVALS NBEG/NEND, except at three edges (17.25088,
    18.24438, 20.8 nm) where DFINTERVALS' REAL*4 DATA literals move the boundary by one point; with those
    float32 edges the ranges are identical. NSTEPS = step_bounds('kurucz') + 1."""
    from atlas_torch.tables import table
    assert (DF_GRID.first, DF_GRID.length) == (1097316, 3507859)
    nb, ne = np.asarray(dftables.nbeg), np.asarray(dftables.nend)
    bb, be = interval_index_ranges(IntervalSet.big())
    lb, le = interval_index_ranges(IntervalSet.little())
    ours = np.r_[be, le] + 1
    diff = np.flatnonzero(ours != ne)
    assert len(diff) == 6 and np.abs(ours[diff] - ne[diff]).max() == 1
    f32 = lambda names: np.concatenate([table("dfsynthe.for", "dfintervals", n) for n in names])  # noqa: E731
    b32 = interval_index_ranges(IntervalSet("big", f32(("biga", "bigb", "bigc"))))
    l32 = interval_index_ranges(IntervalSet("little", f32(("lita", "litb", "litc", "litd", "lite", "litf", "litg"))))
    assert np.array_equal(np.r_[b32[0], l32[0]] + 1, nb) and np.array_equal(np.r_[b32[1], l32[1]] + 1, ne)
    counts = np.r_[b32[1] - b32[0] + 1, l32[1] - l32[0] + 1]
    assert np.array_equal(step_bounds(counts, "kurucz").T.astype(np.int64) + 1, np.asarray(dftables.nsteps))


def test_segments_cover_each_interval_once():
    seg = segments()
    cb, cl = np.zeros(328, int), np.zeros(1212, int)
    for a, b in zip(seg[:-1], seg[1:]):
        assert b.begin == a.end + 1
    for s in seg:
        cb[s.big[0]:s.big[1]] += 1
        cl[s.little[0]:s.little[1]] += 1
        assert s.end - s.begin + 1 <= 65536
    assert cb.min() == cb.max() == 1 and cl.min() == cl.max() == 1
    assert seg[0].begin == 0 and seg[-1].end == DF_GRID.length - 1


def _reference_steps(values, fractions):
    """Mean of the step function of the sorted values over each step's fraction range (weights by overlap)."""
    y = np.sort(values)
    n = len(y)
    edges = np.r_[0.0, np.cumsum(fractions)] * n
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        w = np.clip(np.minimum(np.arange(1, n + 1), b) - np.maximum(np.arange(n), a), 0, None)
        out.append((w * y).sum() / (b - a))
    return np.array(out)


def test_distribution_steps_exact():
    g = torch.Generator().manual_seed(3)
    counts = [37, 120, 61]
    iv = torch.repeat_interleave(torch.arange(3), torch.tensor(counts))
    x = torch.randn(4, len(iv), generator=g, dtype=torch.float64) * 3 - 5
    s = distribution_steps(x, iv, 3, "exact")
    for r in range(4):
        for i in range(3):
            ref = _reference_steps(x[r, iv == i].numpy(), STEP_FRACTIONS)
            assert np.allclose(s[r, i].numpy(), ref, atol=1e-10, rtol=0)
    lin = torch.arange(600, dtype=torch.float64)[None] / 600.0          # uniform: step mean = mid-fraction
    m = distribution_steps(lin, torch.zeros(600, dtype=torch.long), 1, "exact")[0, 0].numpy()
    cf = np.r_[0.0, np.cumsum(STEP_FRACTIONS)]
    assert np.allclose(m, 0.5 * (cf[1:] + cf[:-1]) - 0.5 / 600.0, atol=1e-12)


def test_distribution_steps_kurucz_dfcalc():
    """'kurucz' bounds on integer input give DFCALC's step sums; DFCALC's value is (ISUM + N/2)/N truncated."""
    g = torch.Generator().manual_seed(4)
    counts = [1229 * 10 + 7, 333, 61]
    iv = torch.repeat_interleave(torch.arange(3), torch.tensor(counts))
    k = torch.randint(-6000, 3000, (2, len(iv)), generator=g).to(torch.float64)
    s = distribution_steps(k, iv, 3, "kurucz")
    for r in range(2):
        for i, n in enumerate(counts):
            y = np.sort(k[r, iv == i].numpy().astype(np.int64))
            nn = n - 1
            ns = [1] + [(nn * c) // 60 + 1 for c in (6, 12, 18, 24, 30, 36, 42, 48, 54, 57, 59)] + [nn + 1]
            sums = [y[ns[j] - 1:ns[j + 1] - 1].sum() for j in range(12)]
            cnt = np.diff(ns)
            assert np.allclose(s[r, i].numpy() * cnt, sums, atol=1e-6)
            fort = dfcalc_round(np.array(sums), cnt)
            assert np.abs(fort - s[r, i].numpy()).max() < 1.5


def test_distribution_steps_gradcheck():
    g = torch.Generator().manual_seed(5)
    iv = torch.repeat_interleave(torch.arange(2), torch.tensor([13, 29]))
    x = torch.randn(2, 42, generator=g, dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(lambda a: distribution_steps(a, iv, 2, "exact"), (x,))


def test_turbulence_kernel_vttab(dftables):
    for v in (1, 2, 4, 8):
        k = turbulence_kernel(v, kind="kurucz")
        n = len(k) // 2
        assert n == int(dftables.nvt[v - 1])
        assert abs(float(k[n]) - float(dftables.vtcenter[v - 1])) < 1e-7
        assert np.abs(k[n + 1:].numpy() - np.asarray(dftables.vtprof)[:n, v - 1]).max() < 1e-7


def test_broadening_gaussian_and_integral():
    """A Gaussian line of Doppler velocity 3 km/s broadened by xi = 2 km/s is the Gaussian of sqrt(9 + 4) km/s
    (to 1e-3 in the core, sampling); the convolution conserves the integral."""
    step = 2.99792458e5 * math.log1p(1.0 / DF_GRID.resolution)
    n = torch.arange(-2000, 2001, dtype=torch.float64)
    gauss = lambda w: torch.exp(-(n * step / w) ** 2) / (math.sqrt(math.pi) * w)    # noqa: E731
    k = turbulence_kernel(2.0, kind="exact", floor=1e-12)
    h = len(k) // 2
    y = broaden(gauss(3.0)[None], k)[0]
    ref = gauss(math.sqrt(13.0))[h:-h]
    core = ref > 0.1 * ref.max()
    assert float(((y - ref).abs() / ref)[core].max()) < 1e-3
    x = torch.rand(3, 500, dtype=torch.float64)
    xp = torch.nn.functional.pad(x, (2 * h, 2 * h))
    assert torch.allclose(broaden(xp, k).sum(-1), x.sum(-1), rtol=1e-12)


def test_decoded_lines_vs_selectlines():
    """Packed lowlines at 5012 K: gf e^(-E hc/kT)(1 - e^(-h nu/kT)) (pi e^2/mc)/sqrt(pi)/nu and Gamma/(4 pi nu)
    against SELECTLINES' TAPE12: REAL*4 TABLOG (its exponent (I-16384)*.001 is rounded to 5e-7, i.e. 1e-6 in
    E_lo, times E hc/kT up to ~50: < 1e-4) and REAL*4 Gamma/(4 pi nu) (< 3e-5); Kurucz's 1.77245 and 12.5664 are
    2e-6 off sqrt(pi), 4 pi."""
    from atlas_torch.dfsynthe.lines.f32 import Flavor
    from atlas_torch.dfsynthe.lines.selectlines import iter_select_file, select_consts
    from atlas_torch.v2.constants import KURUCZ
    from atlas_torch.v2.io.dfsynthe_lines import default_dir, open_df_list
    from atlas_torch.v2.odf.lines import CGF_KURUCZ, decode
    p = default_dir() / "lowlinesdf.bin"
    if not p.exists():
        pytest.skip(f"missing {p}")
    m = np.asarray(open_df_list(p)[1_000_000:1_020_000])
    d0 = decode(m)
    m = m[(d0.log_gf - d0.e_lower * KURUCZ.hc_k / 5012.0 / math.log(10.0)).numpy() > -25.0]   # REAL*4 range
    edges = np.array([8.0, 1e5])
    c = select_consts(5012.0, edges, Flavor("O0"))
    t12 = next(iter_select_file(m, len(m), c, np.full((2, 594), 3e38, np.float32), Flavor("O0"), chunk=len(m)))
    assert len(t12) == len(m)
    d = decode(m)
    T = 5012.0
    nu = 2.99792458e17 / d.wavelength
    hc_kt = KURUCZ.hc_k / T
    s = CGF_KURUCZ * 10.0 ** d.log_gf * torch.exp(-d.e_lower * hc_kt) * -torch.expm1(-KURUCZ.h * nu / (KURUCZ.k * T)) / nu
    normal = t12.congf > 1e-36                                          # REAL*4 subnormals lose precision
    assert float(np.abs(s.numpy() / t12.congf - 1)[normal].max()) < 1e-4
    for name, ref in (("gamma_rad", t12.gamr), ("gamma_stark", t12.gams), ("gamma_vdw", t12.gamw)):
        g = getattr(d, name) / (4 * math.pi * nu)
        assert float(np.abs(g.numpy() / ref - 1).max()) < 3e-5, name
    assert np.array_equal(d.center.numpy() + 1, t12.nbuff) and np.array_equal(d.slot.numpy() + 1, t12.nelion)


def test_index_range_bisect(tmp_path):
    """index_range on a memory map = np.searchsorted on the column (duplicates, ends, values outside)."""
    from atlas_torch.v2.io.dfsynthe_lines import LINE_DTYPE, index_range
    g = np.random.default_rng(3)
    rec = np.zeros(5000, dtype=LINE_DTYPE)
    rec["iwl"] = np.sort(g.integers(1, 2000, 5000))
    rec.tofile(tmp_path / "x.bin")
    m = np.memmap(tmp_path / "x.bin", dtype=LINE_DTYPE, mode="r")
    for lo, hi in ((1, 1), (0, 5000), (17, 17), (500, 1500), (1999, 3000), (3000, 4000)):
        s = index_range(m, lo, hi)
        ref = slice(int(np.searchsorted(rec["iwl"], lo)), int(np.searchsorted(rec["iwl"], hi, side="right")))
        assert (s.start, s.stop) == (ref.start, ref.stop)


def test_first_below_and_deposit():
    """first_below = brute-force first crossing of a decreasing ratio (with empty ranges and caps); deposit = the
    dense sum of the wings over a block."""
    from atlas_torch.v2.odf.wings import WingPairs, deposit, first_below
    g = torch.Generator().manual_seed(5)
    k = 300
    scale = 10.0 ** (6 * torch.rand(k, generator=g, dtype=torch.float64))
    n_min = torch.randint(0, 5, (k,), generator=g)
    n_max = n_min + torch.randint(-2, 200000, (k,), generator=g)
    ratio = lambda it, n: scale[it] / (1.0 + n.double()) ** 2                  # noqa: E731
    got = first_below(ratio, n_min, n_max)
    ref = torch.ceil(torch.sqrt(scale) - 1.0 + 1e-12).long().clamp(min=0)
    ref = torch.where(ratio(torch.arange(k), ref) < 1.0, ref, ref + 1)
    ref = torch.maximum(ref, n_min).minimum(n_max)
    ref = torch.where(n_min > n_max, n_min - 1, ref)
    assert torch.equal(got, ref)
    P, L, lo, hi = 40, 3, 100, 400
    c = torch.randint(0, 500, (P,), generator=g)
    z = torch.zeros(P, dtype=torch.float64)
    p = WingPairs(torch.randint(0, L, (P,), generator=g), c, z, torch.rand(P, generator=g, dtype=torch.float64), z,
                  z, torch.zeros(P, dtype=torch.long), torch.randint(0, 300, (P,), generator=g),
                  torch.randint(0, 300, (P,), generator=g), z, z)
    value = lambda q, it, idx: q.k0[it] / (1.0 + (idx - q.center[it]).abs().double())   # noqa: E731
    out = torch.zeros(L, hi - lo + 1, dtype=torch.float64)
    deposit(p, lo, hi, out, value, budget=257)
    dense = torch.zeros_like(out)
    for i in range(P):
        for n in range(-int(p.blue[i]), int(p.red[i]) + 1):
            j = int(c[i]) + n
            if lo <= j <= hi:
                dense[p.layer[i], j - lo] += float(p.k0[i]) / (1 + abs(n))
    assert torch.allclose(out, dense, rtol=1e-14, atol=0)


def test_lyman_alpha_profile():
    """Lyman alpha (HPROF4 without the quasi-molecular wings): positive, normalised core, the radiative wing only
    in HRAYOP's gap, nothing from the resonance wing beyond 4000 cm^-1 to the red."""
    from atlas_torch.v2.opacity.lines.hydrogen import LYA_RADIATIVE_HZ, HydrogenLine, line_wavelength
    T, ne, nh, nhe = (torch.tensor([[x]], dtype=torch.float64) for x in (9000.0, 1e14, 1e16, 1e15))
    t43, ne16 = (T / 1e4) ** 0.3, ne ** (1 / 6)
    fo = ne16 ** 4 * 1.25e-9
    lay = dict(n_e=ne, fo=fo, pp=ne16 * 0.08989 / torch.sqrt(T), y1b=2.0 / (1.0 + 0.012 / T * torch.sqrt(ne / T)),
               y1s=t43 / ne16, t3nhe=t43 * nhe, t3nh2=t43 * 0.0, c1d=fo * 78940.0 / T, c2d=fo ** 2 / 5.96e-23 / ne,
               gcon1=0.2 + 0.09 * torch.sqrt(T / 1e4) / (1.0 + ne / 1e13), gcon2=0.2 / (1.0 + ne / 1e15),
               n_h1_ground=2.0 * nh / 2.0)
    hl = HydrogenLine(1, 2)
    wl0 = line_wavelength(1, 2)
    dop = torch.sqrt(2 * 1.380649e-16 * T / 1.6735e-24) / 2.99792458e10
    dl = torch.linspace(-30.0, 80.0, 20001, dtype=torch.float64)[None, :]
    phi = hl.profile(dl, lay, dop)
    assert bool((phi > 0).all()) and bool(torch.isfinite(phi).all())
    nu = 2.99792458e17 / (wl0 + dl)
    area = torch.trapezoid(phi[0].flip(0), nu[0].flip(0))
    assert 0.9 < float(area) < 1.1
    dense = dict(lay, n_h1_ground=10.0 * lay["n_h1_ground"])
    at = lambda dk: torch.tensor([[1e7 / (1e7 / wl0 - dk) - wl0]], dtype=torch.float64)   # noqa: E731
    assert float(hl.profile(at(3000.0), dense, dop)) > 5.0 * float(hl.profile(at(3000.0), lay, dop))
    assert math.isclose(float(hl.profile(at(5000.0), dense, dop)), float(hl.profile(at(5000.0), lay, dop)),
                        rel_tol=1e-9)
    assert LYA_RADIATIVE_HZ[0] < 2.99792458e17 / wl0 < LYA_RADIATIVE_HZ[1]
