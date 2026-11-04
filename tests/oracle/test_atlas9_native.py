"""The optional native helpers of the ATLAS9 port (atlas9/native: vlibm.c, nmolec.c) against their pure
Python/torch fallbacks: bit-identical results (golden NMOLEC states, IFEDNS=0 and 1, and perturbed T/P; the
batched glibc calls on random and special arguments)."""
import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9 import fmath, native
from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics import libm
from atlas_torch.atlas9.physics.molec import nmolec

pytestmark = pytest.mark.skipif(native.load() is None, reason="native helpers not built (make native)")


def _same(a, b):
    if isinstance(a, torch.Tensor):
        a, b = a.numpy(), b.numpy()
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind != "f":
        return np.array_equal(a, b)
    iv = np.int64 if a.dtype == np.float64 else np.int32
    return a.shape == b.shape and np.array_equal(a.view(iv), b.view(iv))


def test_vlibm_matches_scalar_glibc():
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.uniform(-800, 800, 20000), rng.uniform(-30, 30, 20000),
                        [0.0, -0.0, np.inf, -np.inf, np.nan, 1e-320, 710.0, -746.0]])
    p = np.concatenate([10.0 ** rng.uniform(-300, 300, 20000), [0.0, -0.0, np.inf, np.nan, 5e-324, 1.0, 2.0, -1.0]])
    for mod in (libm, fmath):
        for name, arr in (("exp", x), ("log", p), ("log10", p)):
            t = torch.from_numpy(arr.copy())
            a = getattr(mod, name)(t)
            with native.disabled():
                b = getattr(mod, name)(t)
            assert _same(a, b), (mod.__name__, name)
        y = torch.from_numpy(rng.uniform(-3, 3, p.size))
        a = mod.pow(torch.from_numpy(p.copy()), y)
        with native.disabled():
            b = mod.pow(torch.from_numpy(p.copy()), y)
        assert _same(a, b), (mod.__name__, "pow")
    for name in ("exp", "log", "log10"):
        t = torch.from_numpy((np.abs(x) if name != "exp" else x / 10).astype(np.float32))
        a = getattr(libm, name)(t)
        with native.disabled():
            b = getattr(libm, name)(t)
        assert _same(a, b), ("float32", name)


def _nmolec_both(s, mode):
    a = nmolec(s, mode)
    with native.disabled():
        b = nmolec(s, mode)
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert _same(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


@pytest.mark.parametrize("hit", [0, 1, 2])
def test_nmolec_native_golden(golden, hit):
    s = T.nmolec_state(golden, hit)
    _nmolec_both(s, int(golden(T.TRACE_STAGE, "nmolec", hit, "entry")["arg:mode"][0]))


@settings(max_examples=4, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(hit=st.sampled_from([0, 1]), ft=st.floats(0.85, 1.2), fp=st.floats(0.3, 3.0), seed=st.integers(0, 2**31 - 1))
def test_nmolec_native_perturbed(golden, hit, ft, fp, seed):
    s = T.nmolec_state(golden, hit)
    rng = np.random.default_rng(seed)
    w = torch.from_numpy(np.exp(rng.normal(0, 0.01, 99)))
    T_ = s["temp.t"] * ft * w
    TK = 1.38054e-16 * T_
    s.update({"temp.t": T_, "temp.tk": TK, "temp.tkev": 8.6171e-5 * T_, "temp.hckt": 6.6256e-27 / TK * 2.99792458e10,
              "temp.hkt": 6.6256e-27 / TK, "temp.tlog": torch.log(T_), "state.p": s["state.p"] * fp})
    _nmolec_both(s, 1)


@settings(max_examples=5, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_josh_native_gauss_seidel(seed):
    """JOSH on random (scattering-dominated and absorbing) opacity problems, native vs numpy sweeps."""
    from atlas_torch.atlas9.josh import josh
    rng = np.random.default_rng(seed)
    B, n = 64, 72
    rhox = torch.from_numpy(np.logspace(-4, 2.5, n))
    lr = torch.log(rhox)

    def smooth(lo, hi, slope):        # smooth positive profiles in depth (as real opacities)
        c = torch.from_numpy(rng.uniform(lo, hi, (B, 1)))
        p = torch.from_numpy(rng.uniform(0, slope, (B, 1)))
        return torch.exp(c * 2.302585 + p * lr)
    acont, aline = smooth(-3, 1, 0.6), smooth(-4, 2, 0.4) * torch.from_numpy(rng.uniform(0, 1, (B, 1)) > 0.2)
    sigmac, sigmal = smooth(-3, 2, 0.2), smooth(-6, -2, 0.1)
    s_ = smooth(-8, -3, 0.3)
    a = josh(acont, s_, aline, s_ * 1.1, sigmac, sigmal, rhox, 1, 0)
    with native.disabled():
        b = josh(acont, s_, aline, s_ * 1.1, sigmac, sigmal, rhox, 1, 0)
    for k in a:
        assert _same(a[k], b[k]), k


def _rosstab(ext):
    from pathlib import Path
    from atlas_torch.atlas9.commons import initial_state
    from atlas_torch.atlas9.rosstab import Rosstab, read_kappa_ros
    _, tab = read_kappa_ros(Path(ext) / "BasicATLAS/data/solar_ODF/kappa.ros")
    s = initial_state(); s["tabtp.tabkap"] = tab; s["tabtp.iftabk"] = torch.tensor([1])
    return Rosstab(s)


def test_rosstab_point_native(ext):
    from atlas_torch.atlas9.rosstab import LN10_F32
    R = _rosstab(ext)
    lib = native.load()
    tabt, nt, tabp, np_, tabv, tk = R.native_args()
    work = np.empty(6 * np_)
    rng = np.random.default_rng(3)
    T_ = np.concatenate([10.0 ** rng.uniform(3.2, 5.5, 300), [1500.0, 2e5, float(10 ** tabt[0]), float(10 ** tabt[-1])]])
    P_ = np.concatenate([10.0 ** rng.uniform(-6, 9, 300), [1e-12, 1e12, 1.0, float(10 ** tabp[0])]])
    V_ = np.concatenate([rng.choice([0.0, 1e5, 2e5, 4e5, 8e5, 3.3e5], 300), [0.0, 8e5, 1e6, -1.0]])
    got = np.array([lib.rosstab_point(tabt.ctypes.data, nt, tabp.ctypes.data, np_, tabv.ctypes.data, tk.ctypes.data,
                                      float(LN10_F32), t, p, v, work.ctypes.data) for t, p, v in zip(T_, P_, V_)])
    ref = R(torch.from_numpy(T_), torch.from_numpy(P_), torch.from_numpy(V_)).numpy()
    assert _same(got, ref)


@settings(max_examples=10, deadline=None)
@given(teff=st.floats(4000.0, 9000.0), logg=st.floats(2.5, 5.0), prad0=st.sampled_from([0.0, 0.05, 1.0]),
       vt=st.sampled_from([0.0, 1e5, 2e5]), iross=st.sampled_from([0, 1]))
def test_ttaup_native(ext, teff, logg, prad0, vt, iross):
    from atlas_torch.atlas9.ttaup import ttaup
    R = _rosstab(ext)
    n = 72
    tau = 10 ** (-6.875 + 0.125 * np.arange(n))
    t = teff * (0.75 * (0.710 + tau - 0.1331 * np.exp(-3.4488 * tau))) ** 0.25
    a = [torch.from_numpy(x) for x in (t, tau, np.full(n, 3e-4), prad0 * (1 + np.linspace(0, 0.3, n)), np.zeros(n),
                                       np.full(n, vt))]
    got = ttaup(*a, 10.0 ** logg, n, iross, R)
    with native.disabled():
        ref = ttaup(*a, 10.0 ** logg, n, iross, R)
    for g, r in zip(got, ref):
        assert _same(g, r)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_map1b_native(seed):
    """numerics.map1b: native rows vs the torch path (shared and per-row abscissae, points outside the table,
    exact knots)."""
    from atlas_torch.atlas9.numerics import map1b
    rng = np.random.default_rng(seed)
    B, nold, nnew = 37, 51, 72
    xo = torch.from_numpy(np.sort(rng.uniform(-3, 3, (B, nold)), axis=1))
    fo = torch.from_numpy(rng.normal(0, 1, (B, nold)).cumsum(1))
    xn = torch.from_numpy(np.sort(rng.uniform(-4, 4, (B, nnew)), axis=1))
    xn[:, 5] = xo[:, 7]
    shared = torch.from_numpy(np.sort(rng.uniform(-3, 3, nold)))
    for args in ((xo, fo, xn), (shared, fo, xn), (xo, fo, xn[0]), (shared, fo[0], xn[0])):
        a = map1b(*args)
        with native.disabled():
            b = map1b(*args)
        assert _same(a[0], b[0]) and _same(a[1], b[1])
