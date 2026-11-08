"""Stage tests of the v2 structure modules on the exact ATLAS9 port's iteration States (stock sun and t9000g40):
the frequency integrals with the port's continuum as a stand-in (isolates transfer, ODF sampling, truncation,
quadrature, Rosseland mean) and with v2's ATLAS9-compatible continuum; convection; the temperature correction fed
the port's own moments."""
import numpy as np
import pytest
import torch

import _atlas9_oracle as O

pytestmark = pytest.mark.stage
torch.set_num_threads(1)
MODELS = [("sun", 1), ("sun", 2), ("t9000g40", 1)]


@pytest.fixture(scope="module")
def runs():
    import _structure_stages as S
    out = {}
    for model in sorted({m for m, _ in MODELS}):
        if not (O.stock_dir(model) / "atlas_control.com").exists():
            pytest.skip(f"missing stock run {model}")
        its = tuple(i for m, i in MODELS if m == model)
        out[model] = (S.states(model, its), S.solver(model))
    return out


def _domain(s):
    import _structure_stages as S
    lt = np.log10(S._v(s, "abross.tauros").numpy()[0])
    return (lt >= -4) & (lt <= 1)


@pytest.mark.parametrize("model,it", MODELS)
def test_radiation_moments(runs, model, it):
    import _structure_stages as S
    st, sv = runs[model]
    s0, s1, s2 = (st[(it, h)] for h in ("frequencies", "ross3", "convec"))
    pr = S.port_radiation(s1, s2)
    F = float(sv.const.sigma * sv.batch.teff[0] ** 4 / (4 * np.pi))
    r = S.radiation(sv, s0, S.PortContinuum(s0))
    assert S.rel(r.kappa_ross, pr["kappa_ross"]) < 1e-4
    assert S.rel(r.lambda_denominator, pr["lambda_denominator"]) < 1e-4
    # the same fixed XTAU depths as JOSH (grid="xtau"); the operators differ (v2's Hermite integration against
    # Kurucz's REAL*4 COEFJ / COEFH tables)
    assert float((r.flux_rad - pr["flux_rad"]).abs().max()) / F < 1e-3
    fd = lambda x: x / pr["kappa_ross"] / F * 100.0                                # FLXDRV of radiative layers, %
    assert float((fd(r.j_minus_s) - fd(pr["j_minus_s"])).abs().max()) < 20.0         # top layer: -143 vs -129 (sun)
    dom = _domain(s2)
    assert float((fd(r.j_minus_s) - fd(pr["j_minus_s"]))[:, dom].abs().max()) < 0.5
    assert S.rel(r.accel_rad, pr["accel_rad_raw"], dom) < 2e-3
    assert S.rel(r.dkappa_flux, pr["dkappa_flux"], dom) < 2e-2
    rv = S.radiation(sv, s0)
    assert S.rel(rv.kappa_ross, pr["kappa_ross"], dom) < 0.023                   # continuum compat, ABROSS gate


@pytest.mark.parametrize("model,it", MODELS)
def test_convection(runs, model, it):
    import _structure_stages as S
    st, sv = runs[model]
    s2, s3 = st[(it, "convec")], st[(it, "tcorr3")]
    td, grad, cv = S.convection(sv, s2)
    g = lambda k: S._v(s3, k)                                                       # noqa: E731
    for v2, key, tol in ((td.heat_cp, "conv.heatcp", 3e-3), (td.grad_ad, "conv.grdadb", 3e-3),
                         (grad, "conv.dltdlp", 1e-10), (td.sound_speed, "conv.velsnd", 1e-3)):
        assert S.rel(v2, g(key)) < tol, key
    F = float(5.6697e-5 * sv.batch.teff[0] ** 4 / 12.5664)
    assert float((cv.flux - g("conv.flxcnv")).abs().max()) / F < 1e-2               # FLXCNV/F gate


@pytest.mark.parametrize("model,it", MODELS)
def test_temperature_correction(runs, model, it):
    import _structure_stages as S
    st, sv = runs[model]
    s3, s4 = st[(it, "tcorr3")], st[(it, "end")]
    c = S.correction(sv, s3)
    e = lambda k: S._v(s4, k)                                                       # noqa: E731
    assert float((c.flux_error - e("flux.flxerr")).abs().max()) < 1e-4             # percent
    assert float((c.flux_derivative - e("flux.flxdrv")).abs().max()) < 1e-3
    if it == 1:                                                                    # no acceleration in iteration 1
        assert float((c.dT - e("save.tcorr.oldt1")).abs().max()) < 1e-3            # K


@pytest.mark.parametrize("model,tol", [("sun", 2e-2), ("t9000g40", 0.1)])
def test_atlas9_continuum_compat(runs, model, tol):
    """KuruczCompat.atlas9() against the port's KAPP(0) on the port's populations: H I (HOP), C I, Mg I, Al I
    (COOLOP) reproduced; Si I (SI1OP), He I (HE1OP), H Rayleigh (HRAYOP) and HOTOP keep atlas7v's forms (8 % of
    the 0.3-2 um extinction in the deepest layers of t9000g40)."""
    from atlas_torch.v2.eos import MoleculeSet
    from atlas_torch.v2.constants import KURUCZ
    from atlas_torch.v2.opacity.continuum import ContinuousOpacity, KuruczCompat
    st, _ = runs[model]
    s = st[(1, "frequencies")]
    orc, ops = O.populations(s), O.continuum(s)
    pops = O.v2_populations(orc, MoleculeSet.default())
    op = ContinuousOpacity(const=KURUCZ, compat=KuruczCompat.atlas9())(pops, torch.as_tensor(ops["freq"]), parts=True)
    from atlas_torch.atlas9.kapp import OPS
    tot = sum(ops[k] for k in OPS if k.startswith(("a", "sig")) and k not in ("alines", "axline", "axcont", "ahline",
                                                                               "siglin", "sigxl", "sigx"))
    h = (op.absorbers["H I bf"] + op.absorbers["H I ff"]).numpy().T
    assert float(np.max(np.abs(h - ops["ahyd"]) / tot)) < 1e-4
    wl = 2.99792458e17 / ops["freq"]
    vis = (wl > 300.0) & (wl < 2000.0)
    v = (op.absorption + op.scattering).numpy().T
    assert float(np.max(np.abs(v[vis] / tot[vis] - 1))) < tol
