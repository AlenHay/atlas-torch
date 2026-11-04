"""Oracle tests for PFSAHA (+ PFIRON, PFGROUND, IONPOTS data) of the ATLAS9 variant (atlas9mem.patched.for).
golden: the solar trace's first PFSAHA calls; property: random layers (T 3000..50000 K, P, XNE), every element
class (special-cased light ions, NNN table, iron group via PFIRON, heavy elements), every MODE, vs the Fortran.
"""
import numpy as np
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics._data import potion
from atlas_torch.atlas9.physics.pfsaha import Layers, _nions, pfsaha
from atlas_torch.state import State


def test_pfsaha_golden(golden):
    for hit in range(3):
        e = golden(T.TRACE_STAGE, "pfsaha", hit, "entry")
        x = golden(T.TRACE_STAGE, "pfsaha", hit, "exit")
        j, iz, nion, mode = (int(e["arg:" + k][0]) for k in ("j", "iz", "nion", "mode"))
        L = Layers.from_state(State.from_trace(e), 99).index(slice(j - 1, j))
        r = pfsaha(L, iz, nion, mode, e["potion.potion"])
        ans = x["arg:answer"].reshape(99, 31, order="F")
        exp = ans[j - 1, :nion] if mode >= 10 else ans[j - 1, :1]
        got = r[0].numpy() if mode >= 10 else r[:1].numpy()
        T.compare({"answer": got}, {"answer": exp}, T.golden_tol(), f"pfsaha hit {hit}: ")


def random_layers(seed, n=T.KW):
    rng = np.random.default_rng(seed)
    t = np.exp(rng.uniform(np.log(3000.0), np.log(50000.0), n))
    tk = 1.38054e-16 * t
    p = 10.0 ** rng.uniform(-2, 6, n)
    xne = p / tk * 10.0 ** rng.uniform(-8, -0.31, n)
    bhyd = np.ones((T.KW, 6))
    return dict(t=t, tk=tk, tkev=8.6171e-5 * t, hckt=6.6256e-27 * 2.99792458e10 / tk, tlog=np.log(t), p=p, xne=xne,
                bhyd=bhyd)


def load_layers(o, d, variant):
    o.set_common("temp", t=d["t"], tk=d["tk"], tkev=d["tkev"], hckt=d["hckt"], tlog=d["tlog"])
    o.set_common("state", p=d["p"], xne=d["xne"])
    o.set_common("depart", bhyd=d["bhyd"], nlteon=0)
    o.set_common("potion", potion=potion(variant))


def check_vs_fortran(variant, seed, iz, nion_frac, mode, rtol, nj=12):
    o = T.make_oracle(variant)
    d = random_layers(seed)
    load_layers(o, d, variant)
    nions = _nions(iz, variant)
    nion = 1 + int(nion_frac * min(nions, 6 if mode == 5 else nions)) % min(nions, 6 if mode == 5 else nions)
    L = Layers(*[torch.from_numpy(d[k]) for k in ("t", "tk", "tkev", "hckt", "tlog", "xne", "p")],
               torch.from_numpy(d["bhyd"]), 0)
    r = pfsaha(L, iz, nion, mode, potion(variant), variant)
    worst = 0.0
    for j in np.random.default_rng(seed + 1).choice(T.KW, nj, replace=False) + 1:
        ans = np.zeros((99, 31), order="F")
        o.call("pfsaha", int(j), iz, nion, mode, ans)
        if mode == 5:
            pf, eion = r
            got = np.concatenate([pf[j - 1].numpy(), eion[j - 1].numpy()])
            exp = np.concatenate([ans[:nion, 0], ans[6:7 + nion, 0]])
        elif mode >= 10:
            got, exp = r[j - 1].numpy(), ans[j - 1, :nion]
        else:
            got, exp = np.atleast_1d(r[j - 1].numpy()), ans[j - 1, :1]
        T.compare({"answer": got}, {"answer": exp}, rtol, f"pfsaha[{variant}] iz={iz} nion={nion} mode={mode} j={j}: ")
        worst = max(worst, T.max_rel(got, exp))
    return worst


ELEMENTS = list(range(1, 41)) + [41, 50, 56, 57, 63, 70, 82, 90, 92, 99]


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 2**31 - 1), iz=st.sampled_from(ELEMENTS), nf=st.floats(0, 0.999),
       mode=st.sampled_from([1, 2, 3, 4, 5, 11, 12, 13]))
def test_pfsaha_vs_fortran(seed, iz, nf, mode):
    check_vs_fortran("atlas9", seed, iz, nf, mode, rtol=0.0)


@settings(max_examples=60, deadline=None)
@given(nelem=st.integers(20, 28), ion=st.integers(1, 10), tlog=st.floats(3.3, 4.75),
       lpot=st.floats(0.5, 5.0))
def test_pfiron_vs_fortran(nelem, ion, tlog, lpot):
    """PFIRON(NELEM,ION,TLOG8,POTLOW8,PF): REAL*4 Fe-group partition-function tables."""
    import ctypes
    from atlas_torch.atlas9.physics.pfsaha import pfiron
    o = T.make_oracle("atlas9")
    pf = ctypes.c_double(0.0)
    o.call("pfiron", nelem, ion, float(tlog), float(10.0 ** lpot), pf)
    got = pfiron(nelem, ion, torch.tensor([tlog], dtype=torch.float64), torch.tensor([10.0 ** lpot], dtype=torch.float64))
    assert got.item() == pf.value


@settings(max_examples=40, deadline=None)
@given(t=st.floats(1000.0, 60000.0))
def test_pfground_vs_fortran(t):
    """FUNCTION PFGROUND(NELION,T) for every NELION (incl. the fall-through quirks 43->431, 54, 66)."""
    import ctypes
    from atlas_torch.atlas9.physics.pfsaha import pfground
    o = T.make_oracle("atlas9")
    tt = torch.tensor([t], dtype=torch.float64)
    for nel in range(1, 400):
        exp = o.call("pfground", nel, float(t), restype=ctypes.c_double)[0]
        assert pfground(nel, tt).item() == exp, nel
