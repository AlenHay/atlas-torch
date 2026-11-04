"""Oracle tests for PFSAHA, PFIRON and PFGROUND (xnfdf.for).

1. golden: PFSAHA hits 0..2 of the traced solar run (the first calls of NMOLEC: layer 1 of model 1,
   XNE = the NMOLEC starting guess, NLTEON=-1): ANSWER and the SAVEd intermediates at exit.
2. hypothesis: random (T, P, XNE) layers, departure coefficients, element, NION, MODE, NLTEON
   -> port vs Fortran PFSAHA; PFIRON (REAL*4, bit-exact) and PFGROUND on random arguments.
Max relative error observed: PFSAHA 6e-16 (libm exp ulps; 0 with the oracle's own libm), PFIRON 0.
"""
import ctypes

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, example, given, settings, strategies as st

from atlas_torch.dfsynthe.xnf import oracle_support as osup
from atlas_torch.dfsynthe.xnf.fdata import pfground
from atlas_torch.dfsynthe.xnf.pfsaha import Layers, _tabs, pfiron, pfsaha
from atlas_torch.dfsynthe.xnf.readin import B_ARRAYS, Readin
from atlas_torch.testing.compare import assert_close

pytestmark = pytest.mark.skipif(not osup.available(), reason="libxnfdf.so or the Intel Fortran runtime missing")
RUN = osup.ext_dir() / "traces" / "odf_solar" / "xw"
STAGE = "odf_solar/xnfdf"
RTOL = 1e-13


def nions_of(iz):
    lz = _tabs()["locz"]
    if iz > 28:
        return 3
    if iz in (6, 7):
        return 6
    if 20 <= iz < 29:
        return 10
    return lz[iz] - lz[iz - 1]


def _layers(m, nj, xne):
    t = torch.from_numpy(m.t[:nj].copy())
    return Layers(t, torch.from_numpy(m.tkev[:nj].copy()), torch.from_numpy(m.tk[:nj].copy()),
                  torch.from_numpy(m.hckt[:nj].copy()), torch.from_numpy(m.tlog[:nj].copy()),
                  torch.from_numpy(m.p[:nj].copy()), torch.as_tensor(xne, dtype=torch.float64),
                  {k: torch.from_numpy(v[:nj].copy()) for k, v in m.b.items()}, 0)


def test_pfsaha_golden(golden):
    m = Readin((RUN / "stdin.txt").read_text(), (RUN / "fort.2").read_text()).read(1)
    xntot = m.p[0] / m.tk[0]
    x = xntot / 2. / 10.                       # NMOLEC: XN(1)=XNTOT/2, XNE(1)=XN(1)/10
    L = _layers(m, 1, np.array([x]))
    for hit in range(3):
        en = golden(STAGE, "pfsaha", hit, "entry")
        ex = golden(STAGE, "pfsaha", hit, "exit")
        j, iz, nion, mode = (int(en[f"arg:{k}"].reshape(-1)[0]) for k in ("j", "iz", "nion", "mode"))
        assert j == 1 and mode == 3
        got = pfsaha(L.with_(nlteon=-1), iz, nion, mode)[0].numpy()
        k = got.shape[-1]
        assert_close(got, ex["arg:answer"].reshape(99, 31, order="F")[0, :k] if ex["arg:answer"].ndim == 1
                     else ex["arg:answer"][0, :k], rtol=RTOL, name=f"pfsaha hit {hit} iz={iz}")
        charge = 2. * x                          # EXCESS = 2*XNE - P/TK < 0
        potlow = min(1.0, 1.44e-7 / np.sqrt(m.tk[0] / 2.8965e-18 / charge))
        assert float(ex["static:potlow"].reshape(-1)[0]) == potlow
        assert float(ex["static:charge"].reshape(-1)[0]) == charge


@pytest.fixture(scope="module")
def ora():
    return osup.new_oracle()


@pytest.fixture(scope="module")
def ora_imf():
    """libxnfdf.so bound to libimf like the stock ifx program (Oracle deepbind=True; see fortran_oracle).

    PFIRON computes LOG10(POTLOW) in REAL*4. The stock xnfdf.exe binds log10f to libimf (LD_DEBUG=bindings);
    the plain oracle binds it to glibc 2.39, whose log10f is 1 ulp off in ~3% of arguments (e.g. 5766.0947),
    and PFIRON then differs by 2 float ulps. The port (float32(log10 in double)) equals libimf's log10f.
    """
    from atlas_torch.testing.fortran_oracle import Oracle
    osup.ensure_ifx_runtime()
    return Oracle(osup.oracle_path(), deepbind=True)


NJ = 6


@st.composite
def layer_sets(draw):
    t = np.exp(np.array(draw(st.lists(st.floats(np.log(1800.), np.log(250000.)), min_size=NJ, max_size=NJ))))
    p = 10 ** np.array(draw(st.lists(st.floats(-4., 8.), min_size=NJ, max_size=NJ)))
    tk = 1.38054e-16 * t
    xne = p / tk * 10 ** np.array(draw(st.lists(st.floats(-9., -0.05), min_size=NJ, max_size=NJ)))
    nlte = draw(st.booleans())
    b = {}
    for name, nlev in list(B_ARRAYS.items()) + [("bmin", None)]:
        shape = (99,) if nlev is None else (99, nlev)
        b[name] = np.asfortranarray(np.exp(draw(st.floats(-0.5, 0.5))) * np.ones(shape)) if nlte else np.ones(shape)
    return t, p, xne, b


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(lay=layer_sets(), iz=st.integers(1, 99), mode=st.sampled_from([1, 2, 3, 4, 11, 12, 13]),
       nlteon=st.sampled_from([0, -1, 1]), data=st.data())
def test_pfsaha_vs_fortran(ora, lay, iz, mode, nlteon, data):
    t, p, xne, b = lay
    nions = nions_of(iz)
    nion = 1 if mode == 4 else data.draw(st.integers(1, nions))
    tk = 1.38054e-16 * t
    hckt = 6.6256e-27 / tk * 2.99792458e10
    ora.set_common("temp", t=t, tkev=8.6171e-5 * t, tk=tk, hckt=hckt, tlog=np.log(t))
    ora.set_common("state", p=p, xne=xne)
    ora.set_common("if", nlteon=nlteon)
    for name, arr in b.items():
        ora.set_common(osup._BLOCK_OF[name], **{name: arr})
    L = Layers(*(torch.from_numpy(np.ascontiguousarray(a)) for a in (t, 8.6171e-5 * t, tk, hckt, np.log(t), p, xne)),
               {k: torch.from_numpy(np.ascontiguousarray(v[:NJ])) for k, v in b.items()}, nlteon)
    got = pfsaha(L, iz, nion, mode).numpy()
    ans = np.zeros((99, 31), order="F")
    for j in range(NJ):
        ora.call("pfsaha", j + 1, iz, nion, mode, ans)
    assert_close(got, ans[:NJ, :got.shape[-1]], rtol=RTOL, name=f"pfsaha iz={iz} nion={nion} mode={mode}")


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(nelem=st.integers(20, 28), ion=st.integers(1, 10),
       tl=st.lists(st.floats(3.2, 5.6), min_size=8, max_size=8),
       pl=st.lists(st.floats(1.5, 5.0), min_size=8, max_size=8))
# LOG10(POTLOW) in REAL*4: glibc 2.39 log10f(5766.0947) is 1 ulp off libimf's (the stock ifx program's);
# failed against the glibc-bound oracle; see ora_imf
@example(nelem=20, ion=1, tl=[4.0] * 8, pl=[2.0] * 7 + [3.760881766617063])
def test_pfiron_vs_fortran(ora_imf, nelem, ion, tl, pl):
    tl = np.array(tl)
    pot = 10 ** np.array(pl)
    got = pfiron(nelem, ion, torch.from_numpy(tl), torch.from_numpy(pot)).numpy()
    ref = []
    for a, b in zip(tl, pot):
        _, outs = ora_imf.call("pfiron", nelem, ion, float(a), float(b), ctypes.c_double(0.))
        ref.append(outs[-1])
    assert np.array_equal(got, np.array(ref))


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(nelion=st.integers(1, 400), t=st.lists(st.floats(1000., 300000.), min_size=5, max_size=5))
def test_pfground_vs_fortran(ora, nelion, t):
    got = pfground(nelion, torch.tensor(t, dtype=torch.float64)).numpy()
    ref = [ora.call("pfground", nelion, float(v), restype=ctypes.c_double)[0] for v in t]
    assert_close(got, np.array(ref), rtol=1e-14, name=f"pfground {nelion}")
