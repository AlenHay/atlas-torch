"""Oracle tests for NMOLEC, SOLVIT, NELECT and POPS (xnfdf.for).

1. golden (traced solar run, model 1, T=1995 K): NMOLEC exit (XNZ after partition-function scaling,
   XN and EQUILJ of the last layer) and POPS hits 0..2 (XNFH, XNFHE, XNFPH after NMOLEC).
2. hypothesis:
   - NMOLEC on batches of models of the solar grid with perturbed T and P -> port (batched over models,
     per-model Newton convergence masks) vs the Fortran NMOLEC run model by model;
   - SOLVIT on random well-conditioned systems (bit-exact);
   - POPS (molecular path) for random element/ion/molecule codes and modes, given the solved state;
   - NELECT (IFMOL=0 path) on perturbed solar models.
Max relative error observed vs the oracle (ctypes oracle binds glibc libm; port uses torch.exp):
XNE/XNATOM/RHO 2e-16, XNMOL/XNFPMOL 6.7e-11 (trace ions, e.g. Ti ions at 28184 K, amplify exp ulps
through the Newton stop criterion |dX/X|<1e-3); bit-identical (0) with the oracle's libm patched into
the port (test_nmolec_exact_oracle_libm).
"""
import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.dfsynthe.xnf import oracle_support as osup
from atlas_torch.dfsynthe.xnf.nmolec import Pops, nelect, nmolec, solvit
from atlas_torch.dfsynthe.xnf.pfsaha import Layers, _tabs
from atlas_torch.dfsynthe.xnf.readin import Readin
from atlas_torch.dfsynthe.xnfdf import populations
from atlas_torch.testing.compare import assert_close

pytestmark = pytest.mark.skipif(not osup.available(), reason="libxnfdf.so or the Intel Fortran runtime missing")
RUN = osup.ext_dir() / "traces" / "odf_solar" / "xw"
STAGE = "odf_solar/xnfdf"
NJ = 25


@pytest.fixture(scope="module")
def models():
    r = Readin((RUN / "stdin.txt").read_text(), (RUN / "fort.2").read_text())
    return [r.read(1) for _ in range(57)]


def _batch(ms):
    st_ = lambda k: torch.from_numpy(np.stack([getattr(m, k)[:NJ] for m in ms]))
    b = {k: torch.from_numpy(np.stack([m.b[k][:NJ] for m in ms])) for k in ms[0].b}
    L = Layers(st_("t"), st_("tkev"), st_("tk"), st_("hckt"), st_("tlog"), st_("p"), st_("xne"), b, 0)
    return L, torch.from_numpy(np.stack([m.xabund for m in ms])), torch.tensor([m.wtmole for m in ms])


def test_nmolec_golden(golden, models):
    m = models[0]
    L, xab, wt = _batch([m])
    r = nmolec(L, m.mol, xab, m.atmass, wt)
    ex = golden(STAGE, "nmolec", 0, "exit")
    ne = m.mol.nequa
    xnz = ex["static:xnz"].reshape(99, 30, order="F")[:NJ, :ne]
    assert_close(r["xnz"][0], xnz, rtol=1e-12, name="xnz")
    assert_close(r["equilj"][0, NJ - 1], ex["static:equilj"][:m.mol.nummol], rtol=1e-12, name="equilj(J=25)")
    for hit, (code, mode, n) in enumerate([(1.00, 12, 1), (2.01, 12, 2), (1.01, 11, 2)]):
        exp = golden(STAGE, "pops", hit, "exit")
        assert float(exp["arg:code"][0]) == code and int(exp["arg:mode"][0]) == mode
        num = exp["arg:number"].reshape(99, 10, order="F")
        pops = Pops(L.with_(xne=r["xne"]), xab, r["xnatom"], 1, m.mol, r["xnmol"], r["xnfpmol"])
        d = pops(code, mode)
        for ion in range(1, n + 1):
            assert_close(d[ion][0], num[:NJ, ion - 1], rtol=1e-12, name=f"pops {code} ion {ion}")


def _push_and_nmolec(o, m, itemp):
    osup.push_model(o, m, itemp=itemp)
    osup.push_molecules(o, m.mol)
    o.call("nmolec")
    xm, s = o.common("xnmol"), o.common("state")
    nm = m.mol.nummol
    return dict(xnmol=xm["xnmol"][:NJ, :nm].copy(), xnfpmol=xm["xnfpmol"][:NJ, :nm].copy(),
                xne=s["xne"][:NJ].copy(), xnatom=s["xnatom"][:NJ].copy(), rho=s["rho"][:NJ].copy())


def _perturb(m, ft, lp):
    m = m.copy()
    n = m.nrhox
    m.t[:n] = m.t[:n] * ft
    m.p[:n] = m.p[:n] * 10 ** lp
    m.tk[:n] = 1.38054e-16 * m.t[:n]
    m.hkt[:n] = 6.6256e-27 / m.tk[:n]
    m.hckt[:n] = m.hkt[:n] * 2.99792458e10
    m.tkev[:n] = 8.6171e-5 * m.t[:n]
    m.tlog[:n] = np.log(m.t[:n])
    return m


@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(idx=st.lists(st.integers(0, 56), min_size=2, max_size=4, unique=True), ft=st.floats(0.93, 1.07),
       lp=st.floats(-0.3, 0.3))
def test_nmolec_vs_fortran(models, idx, ft, lp):
    ms = [_perturb(models[i], ft, lp) for i in idx]
    L, xab, wt = _batch(ms)
    r = nmolec(L, ms[0].mol, xab, ms[0].atmass, wt)
    o = osup.new_oracle()
    for b, m in enumerate(ms):
        ref = _push_and_nmolec(o, m, itemp=b + 1)
        for k in ("xne", "xnatom", "rho"):
            assert_close(r[k][b], ref[k], rtol=1e-13, name=f"{k} model {idx[b]}")
        # The Newton loop stops once every |EQ/XN| <= 1e-3; with exp differing in the last bit (glibc in the
        # oracle vs torch here), a case near that switch can take one more/fewer quadratic step, so trace
        # species legitimately differ by up to ~(1e-3)^2. The logic itself is checked bit-exact with the oracle's
        # libm in test_nmolec_exact_oracle_libm.
        for k in ("xnmol", "xnfpmol"):
            assert_close(r[k][b], ref[k], rtol=2e-6, name=f"{k} model {idx[b]}")


def test_nmolec_exact_oracle_libm(models):
    """With the oracle's libm the batched port is bit-identical to the Fortran (incl. the ill-conditioned
    model 40, T=28184 K, whose Ti ions amplify exp ulps to 7e-11 otherwise)."""
    ms = [models[0], models[39]]
    L, xab, wt = _batch(ms)
    with osup.oracle_libm():
        r = nmolec(L, ms[0].mol, xab, ms[0].atmass, wt)
    o = osup.new_oracle()
    for b, m in enumerate(ms):
        ref = _push_and_nmolec(o, m, itemp=b + 1)
        for k in ref:
            assert np.array_equal(r[k][b].numpy(), ref[k]), k


@settings(max_examples=100, deadline=None)
@given(n=st.integers(2, 20), seed=st.integers(0, 2 ** 31 - 1))
def test_solvit_vs_fortran(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) * 10 ** rng.uniform(-3, 3, size=(n, n)) + np.diag(rng.uniform(1, 5, n))
    b = rng.normal(size=n)
    o = osup.new_oracle()
    af = np.asfortranarray(a.copy())
    bf = np.asfortranarray(b.copy())
    o.call("solvit", af, n, bf, np.zeros(n, dtype=np.int32, order="F"))
    got = solvit(torch.from_numpy(a[None].copy()), torch.from_numpy(b[None].copy()))[0].numpy()
    assert np.array_equal(got, bf)


def nions_of(iz):
    lz = _tabs()["locz"]
    return 3 if iz > 28 else 6 if iz in (6, 7) else 10 if 20 <= iz < 29 else lz[iz] - lz[iz - 1]


CODES = [1.00, 2.01, 1.01, 2.02, 5.00, 6.01, 8.00, 11.00, 12.01, 13.01, 14.01, 19.00, 20.01, 26.00, 106.00, 108.00,
         7.0, 8.0, 6.03, 7.04, 8.05, 10.05, 26.09, 28.09, 20.09, 4.03, 3.03, 17.04, 18.04, 29.02, 56.02, 99.02,
         101.0, 106.01, 607.0, 10108.0, 60808.0, 822.0, 999.0]


@pytest.fixture(scope="module")
def solved(models):
    ms = [models[i] for i in (0, 9, 18, 30, 45, 56)]
    L, pops, r, xab = populations(ms, NJ)
    return ms, L, pops, r, xab


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(code=st.sampled_from(CODES) | st.integers(1, 99).flatmap(
    lambda z: st.integers(0, min(5, nions_of(z) - 1)).map(lambda k: z + 0.01 * k)),
    mode=st.sampled_from([1, 2, 3, 11, 12, 13]), b=st.integers(0, 5))
def test_pops_vs_fortran(solved, code, mode, b):
    ms, L, pops, r, xab = solved
    m = ms[b]
    o = osup.new_oracle()
    osup.push_model(o, m, itemp=7)
    osup.push_molecules(o, m.mol)
    o.saved("pops")["itemp1"][...] = 7                  # NMOLEC already done for ITEMP=7
    o.set_common("state", xne=r["xne"][b].numpy(), xnatom=r["xnatom"][b].numpy(), rho=r["rho"][b].numpy())
    o.set_common("xnmol", xnmol=r["xnmol"][b].numpy(), xnfpmol=r["xnfpmol"][b].numpy())
    num = np.zeros((99, 10), order="F")
    o.call("pops", float(code), mode, num)
    d = pops(float(code), mode)
    for ion, v in d.items():
        assert_close(v[b], num[:NJ, ion - 1], rtol=1e-14, name=f"pops({code},{mode}) ion {ion}")


@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(idx=st.lists(st.integers(0, 56), min_size=1, max_size=3, unique=True), ft=st.floats(0.95, 1.05))
def test_nelect_vs_fortran(models, idx, ft):
    ms = [_perturb(models[i], ft, 0.0) for i in idx]
    for m in ms:
        m.ifmol = 0
    L, xab, wt = _batch(ms)
    r = nelect(L, xab, wt)
    o = osup.new_oracle()
    for b, m in enumerate(ms):
        osup.push_model(o, m, itemp=b + 1)
        o.call("nelect")
        s = o.common("state")
        for k in ("xne", "xnatom", "rho"):
            assert_close(r[k][b], s[k][:NJ], rtol=1e-14, name=f"nelect {k} model {idx[b]}")
