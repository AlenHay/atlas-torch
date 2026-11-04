"""v2 line opacity: Voigt function, line-list readers against the SYNTHE readers' output (fort.12/fort.19 of the
stock 500-510 nm work dir), profile normalisation, layer independence and gradients."""
import math

import numpy as np
import pytest
import torch

from atlas_torch.v2.opacity.lines import (GasState, LineKind, LineList, LineOpacity, LineOpacityOptions,
                                          LogWavelengthGrid, read_gfall_kurucz, read_h2o,
                                          read_molecular_asc)
from atlas_torch.v2.opacity.lines.hydrogen import oscillator_strength
from atlas_torch.v2.opacity.lines.readers import synthe_files
from atlas_torch.v2.opacity.lines.voigt import voigt, voigt_kurucz

from _lines_oracle import workdir

WD = workdir("500_510")


def _fort12():
    from atlas_torch.synthe.lineopac.fortio import read_fort12, read_fort19, read_fort93
    if not (WD / "pre_synthe" / "fort.12").exists():
        pytest.skip(f"missing {WD}")
    f93 = read_fort93(WD / "pre_synthe" / "fort.93")
    return f93, read_fort12(WD / "pre_synthe" / "fort.12", f93["nlines"]), read_fort19(WD / "pre_synthe" / "fort.19")


def _as_fort12(ll: LineList, grid: LogWavelengthGrid, const=0.026538 / 1.77245):
    nu = 2.99792458e17 / ll.wavelength.numpy()
    f = 12.5664 * nu
    nb = np.floor(np.log(ll.wavelength.numpy()) / grid.step + 0.5).astype(np.int64) - grid.first + 1
    g = lambda k: 10.0 ** getattr(ll, k).numpy() / f
    return dict(nbuff=nb, congf=const * 10.0 ** (ll.log_gf + ll.log_isotope).numpy() / nu, elo=ll.e_lower.numpy(),
                gamrf=g("log_gamma_rad"), gamsf=g("log_gamma_stark"), gamwf=g("log_gamma_vdw"))


def test_voigt_matches_faddeeva():
    from scipy.special import wofz
    a = np.concatenate([[1e-4, 1e-3], np.logspace(-3, 2.5, 45)])[:, None]
    v = np.concatenate([np.linspace(0.0, 12.0, 241), np.logspace(1.1, 4.0, 40)])[None, :]
    h = voigt(torch.tensor(a), torch.tensor(v)).numpy()
    assert np.abs(h / wofz(v + 1j * a).real - 1.0).max() < 1.2e-4


def test_kurucz_h1_is_harris_table():
    """H1(v) of voigt_kurucz (a -> 0 slope) against Harris (1948), synthe.for TABVOIGT DATA TABH1."""
    from atlas_torch.tables import table
    vi = table("synthe.for", "tabvoigt", "tabvi").astype(np.float64)
    h1 = table("synthe.for", "tabvoigt", "tabh1").astype(np.float64)
    vt = torch.tensor(vi)
    slope = (voigt_kurucz(torch.full_like(vt, 1e-3), vt) - voigt_kurucz(torch.zeros_like(vt), vt)) / 1e-3
    assert np.abs(slope.numpy() - h1).max() < 2e-4


def test_hydrogen_f_values():
    """Kurucz's HFNM fit against the exact hydrogenic absorption oscillator strengths (NIST ASD, Wiese & Fuhr 2009)."""
    exact = {(1, 2): 0.4162, (1, 3): 0.07910, (2, 3): 0.6407, (2, 4): 0.1193, (2, 5): 0.04467, (3, 4): 0.8421,
             (4, 5): 1.038, (2, 6): 0.02209, (2, 10): 0.003851}
    for (n, m), f in exact.items():
        assert abs(oscillator_strength(n, m) / f - 1.0) < 0.03, (n, m)


def test_grid_is_synbeg_grid():
    f93, _, _ = _fort12()
    g = LogWavelengthGrid.window(f93["wlbeg"], f93["wlend"], f93["resolu"])
    assert g.length == f93["length"]


def test_gfall_reader_reproduces_rgfalllinesnew():
    """Atomic lines in file order = fort.12 atomic block (NBUFF, CGF, E_lo, damping constants / 4 pi nu)."""
    f93, f12, f19 = _fort12()
    grid = LogWavelengthGrid.window(f93["wlbeg"], f93["wlend"], f93["resolu"])
    ll = read_gfall_kurucz(synthe_files() / "BasicATLAS.dat", f93["wlbeg"], f93["wlend"])
    plain = ll.select(ll.kind == int(LineKind.ATOMIC))
    hyd = ll.select(ll.kind == int(LineKind.HYDROGEN))
    ref = f12[:len(plain)]
    mine = _as_fort12(plain, grid)
    assert np.array_equal(mine["nbuff"], ref["nbuff"])
    for k in ("congf", "elo", "gamrf", "gamsf", "gamwf"):
        np.testing.assert_allclose(mine[k], ref[k].astype(np.float64), rtol=2e-5, err_msg=k)
    np.testing.assert_allclose(hyd.wavelength.numpy(), f19["wl"], rtol=1e-12)
    np.testing.assert_array_equal(hyd.n_upper.numpy(), f19["nbup"])


def test_molecular_and_h2o_readers():
    """c2ax.asc and h2ofastfix.bin blocks of fort.12, in order (Kurucz isotopic fractions; H2O CGF uses 0.01502)."""
    f93, f12, _ = _fort12()
    grid = LogWavelengthGrid.window(f93["wlbeg"], f93["wlend"], f93["resolu"])
    c2 = read_molecular_asc(synthe_files() / "c2ax.asc", f93["wlbeg"] - 0.1, f93["wlend"] + 1.0, True, True)
    counts = {}
    for name in ("chmasseron_corrected", "mgh_exomol", "nh", "ohupdate", "sihax", "h2", "h2xx", "hdxx", "c2ax"):
        txt = (WD / f"rmol_{name}.out").read_text()
        counts[name] = int(txt.split("LINES ADDED")[0].split()[-1])
    nat = int((WD / "rgfalllinesnew.out").read_text().split("LINES ADDED TO TAPE 12")[0].split()[-1])
    off = nat + sum(v for k, v in counts.items() if k != "c2ax")
    ref = f12[off:off + len(c2)]
    assert len(c2) == counts["c2ax"]
    mine = _as_fort12(c2, grid)
    assert np.array_equal(mine["nbuff"], ref["nbuff"])
    for k in ("congf", "elo", "gamrf", "gamsf", "gamwf"):
        np.testing.assert_allclose(mine[k], ref[k].astype(np.float64), rtol=2e-5, err_msg=k)
    h2o = read_h2o(None, f93["wlbeg"] - 1.0, f93["wlend"] + 1.0)
    ref = f12[-len(h2o):]
    mine = _as_fort12(h2o, grid, const=0.01502)
    assert np.array_equal(mine["nbuff"], ref["nbuff"])
    np.testing.assert_allclose(mine["congf"], ref["congf"].astype(np.float64), rtol=5e-6)
    np.testing.assert_allclose(mine["elo"], ref["elo"].astype(np.float64), rtol=1e-7)


# ------------------------------------------------------------------------------------------ synthetic gas

def _gas(T, n_layers=None):
    T = torch.as_tensor(T, dtype=torch.float64)
    M = T.shape[0]
    nou = torch.zeros(M, 99, 10, dtype=torch.float64)
    nou[:, 25, 0] = 1e10                                        # Fe I n/U
    nou[:, 0, 0] = 5e15                                         # H I n/U
    one = torch.ones(M, dtype=torch.float64)
    return GasState(T=T, n_e=1e14 * one, rho=3e-7 * one, n_h1=1e16 * one,
                    n_he1=1e15 * one, n_h2=1e12 * one, vturb=0 * one, n_over_u_atom=nou,
                    n_over_u_mol=torch.zeros(M, 1, dtype=torch.float64), mol_codes=np.array([101.0]))


def _fe_lines(wl):
    n = len(wl)
    return LineList.from_numpy(wavelength=np.asarray(wl, float), species=np.full(n, 26.0), log_gf=np.full(n, -1.0),
                               e_lower=np.full(n, 8000.0), log_gamma_rad=np.full(n, 8.0),
                               log_gamma_stark=np.full(n, -6.0), log_gamma_vdw=np.full(n, -7.5),
                               mass=np.full(n, 55.845), kind=np.zeros(n))


def test_profile_normalisation():
    """sum kappa d(nu) over the grid = (pi e^2/mc) gf (n/U)/rho exp(-E/kT) (1 - exp(-h nu/kT)) (K&A 1981 p. 4),
    for a line in the middle of a window much wider than its wings."""
    grid = LogWavelengthGrid.window(499.0, 501.0, 600000.0)
    gas = _gas([5000.0])
    op = LineOpacity(_fe_lines([500.0]), grid, LineOpacityOptions(cutoff=1e-14))
    k = op(gas, torch.ones(1, grid.length, dtype=torch.float64))[0]
    nu = 2.99792458e17 / grid.wavelengths()
    integral = float((k[:-1] * (nu[:-1] - nu[1:])).sum())
    c = op.opt.const
    s = 0.026540 * 0.1 * 1e10 / 3e-7 * math.exp(-8000.0 * c.hc_k / 5000.0) * -math.expm1(-c.hc_k * 1e7 / 500.0 / 5e3)
    # the Lorentz wings beyond the window (|dnu| > 1 nm) carry a/(pi v_edge) of the line
    assert abs(integral / s - 1.0) < 2e-3


def test_layers_independent_and_batched():
    grid = LogWavelengthGrid.window(499.9, 500.1, 600000.0)
    ll = _fe_lines([499.95, 500.0, 500.02])
    op = LineOpacity(ll, grid, LineOpacityOptions())
    cont = torch.full((4, grid.length), 1e-1, dtype=torch.float64)
    T = [4000.0, 5000.0, 6000.0, 7000.0]
    both = op(_gas(T), cont)
    one = torch.cat([op(_gas(T[:2]), cont[:2]), op(_gas(T[2:]), cont[2:])])
    torch.testing.assert_close(both, one, rtol=1e-14, atol=0.0)
    g = _gas(T)
    gb = GasState(**{**g.__dict__, "batch_shape": (2, 2)})
    assert op(gb, cont).shape == (2, 2, grid.length)


def test_gradient_wrt_temperature():
    grid = LogWavelengthGrid.window(499.98, 500.02, 600000.0)
    op = LineOpacity(_fe_lines([500.0]), grid, LineOpacityOptions(cutoff=1e-12))
    cont = torch.ones(1, grid.length, dtype=torch.float64)
    T = torch.tensor([5000.0], dtype=torch.float64, requires_grad=True)
    f = lambda t: op(_gas(t), cont).sum()
    (g,) = torch.autograd.grad(f(T), T)
    h = 0.01
    fd = (f(T.detach() + h) - f(T.detach() - h)) / (2 * h)
    assert torch.isfinite(g).all() and abs(float(g) / float(fd) - 1.0) < 1e-5


def test_series_ramp_and_fano():
    """Merged-series weight (0 below lambda_con, 1 above lambda_tail) and the Fano profile of autoionizing lines."""
    from atlas_torch.v2.opacity.lines.opacity import merging_wavenumber, series_ramp
    from atlas_torch.tables import table
    ne = torch.tensor([1e14], dtype=torch.float64)
    lim = float(table("synthe.for", "xlinop", "contx")[0, 3])                 # C I, first edge
    em = float(merging_wavenumber(ne, False))
    wcon, wtail = 1e7 / (lim - em), 1e7 / (lim - em - 500.0)
    wl = torch.tensor([[[wcon - 0.01, 0.5 * (wcon + wtail), wtail + 0.01]]], dtype=torch.float64)
    r = series_ramp(wl, torch.tensor([6.0], dtype=torch.float64), torch.tensor([1]), ne)
    torch.testing.assert_close(r[0, 0], torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
    grid = LogWavelengthGrid.window(199.0, 201.0, 100000.0)
    ll = LineList.from_numpy(wavelength=np.array([200.0]), species=np.array([26.0]), log_gf=np.array([0.0]),
                             log_gamma_rad=np.array([12.0]), log_gamma_stark=np.array([-0.5]),
                             log_gamma_vdw=np.array([-17.0]), mass=np.array([55.845]),
                             kind=np.array([int(LineKind.AUTOIONIZING)]))
    gas = _gas([5000.0])
    k = LineOpacity(ll, grid, LineOpacityOptions(cutoff=1e-30))(gas, torch.ones(1, grid.length, dtype=torch.float64),
                                                              stimulated=False)[0]
    nu = 2.99792458e17 / grid.wavelengths()
    eps = 2.0 * (nu - 2.99792458e17 / 200.0) / 1e12
    A, B = 10 ** -0.5, 1e-17
    expect = 1e10 / 3e-7 * (A * eps + B) / (eps ** 2 + 1.0)
    torch.testing.assert_close(k, torch.where(expect >= 1e-30, expect, torch.zeros_like(expect)), rtol=1e-12, atol=0)
