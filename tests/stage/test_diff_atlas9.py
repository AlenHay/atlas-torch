"""Stage: implicit differentiation of the converged ATLAS9 solar model (atlas_torch.diff.atlas9_ift) and the
chained d(spectrum)/d(Teff, log g, [M/H]) (atlas_torch.diff.chain).

dx*/dtheta comes from ChainDiff.prepare (Picard linear-response IFT, Jacobian-free; ~240 iterations of the port),
the reference models from 40 further iterations of G at theta +/- h (~280 iterations); both are cached in
$ATLAS_EXT/scratch/m8/cache/sun/ (delete to recompute: 30-90 min depending on the machine and code version).
Validation:
  1. dx*/dtheta (IFT) vs central differences of fully re-converged models (40 iterations of the same map, no
     gold-tolerance stop) at Teff +/- 5 K, log g +/- 0.005, [M/H] +/- 0.005, for T, ln P, ln RHOX, ln XNE.
     Thresholds = measured agreement (RESULTS, M8) with ~2x margin; T for log g is loose because its response
     is small and dominated by the slowly converging layers at the convective boundary (layers 55-62).
  2. the chained spectrum derivative vs central differences of the differentiable spectrum of those models.
Run: pytest -m stage tests/stage/test_diff_atlas9.py -s -p no:cacheprovider
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.stage

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
MODEL = EXT / "runs" / "stock" / "atlas" / "sun"
STEPS = {"teff": 5.0, "logg": 0.005, "mh": 0.005}


def _need(p):
    if not Path(p).exists():
        pytest.skip(f"missing {p}")
    return p


@pytest.fixture(scope="module")
def chain():
    from atlas_torch.diff import ChainDiff
    torch.set_num_threads(4)
    # profile="smooth": with the default straight-through tables the *finite differences* of the frozen-decision
    # function would keep the Voigt-table positions (set by the Doppler width) fixed while T changes
    return ChainDiff(_need(MODEL), (500.0, 510.0), synthe_kw={"profile": "smooth"}, polish=True).prepare(log=print)


@pytest.fixture(scope="module")
def reconverged(chain):
    out = {}
    for k, th in enumerate(("teff", "logg", "mh")):
        for sg in (1, -1):
            t = chain.fp.theta0.copy()
            t[k] += sg * STEPS[th]
            out[(th, sg)] = chain.reconverge(t, log=print)
    return out


def _rel_stats(ift, fd):
    e = np.abs(ift - fd)
    return float(e.max() / np.abs(fd).max()), float(np.sqrt((e ** 2).mean() / (fd ** 2).mean()))


@pytest.mark.parametrize("th", ["teff", "logg", "mh"])
def test_structure_ift_vs_reconverged(chain, reconverged, th):
    n = chain.fp.n
    k = ("teff", "logg", "mh").index(th)
    xp, xm = reconverged[(th, 1)], reconverged[(th, -1)]
    fd = (xp - xm) / (2 * STEPS[th])
    ift = chain.dx[:, k]
    x0 = chain.xstar
    res = {}
    for i, nm in enumerate(("RHOX", "T", "P", "XNE")):
        sl = slice(i * n, (i + 1) * n)
        if nm == "T":
            res[nm] = _rel_stats(ift[sl], fd[sl])
        else:                                     # logarithmic derivatives
            res[nm] = _rel_stats(ift[sl] / x0[sl], fd[sl] / x0[sl])
    print(th, {k: f"max {v[0]:.2e} rms {v[1]:.2e}" for k, v in res.items()})
    limits = {"teff": {"T": 0.05, "P": 0.01, "RHOX": 0.02, "XNE": 0.06},
              "logg": {"T": 0.4, "P": 0.01, "RHOX": 0.01, "XNE": 0.03},
              "mh": {"T": 0.1, "P": 0.01, "RHOX": 0.01, "XNE": 0.03}}[th]
    for nm, (mx, rms) in res.items():
        assert rms < limits[nm], (th, nm, rms)


@pytest.mark.parametrize("th", ["teff", "logg", "mh"])
def test_chained_spectrum_derivative(chain, reconverged, th):
    d = chain.spectrum_derivatives((th,))[th]["norm"]
    sp = chain.spectrum_at(reconverged[(th, 1)], mh=STEPS[th] if th == "mh" else 0.0)["norm"]
    sm = chain.spectrum_at(reconverged[(th, -1)], mh=-STEPS[th] if th == "mh" else 0.0)["norm"]
    fd = (sp - sm) / (2 * STEPS[th])
    mx, rms = _rel_stats(d.numpy(), fd.numpy())
    print(th, f"spectrum (IFT tangent): max {mx:.2e} rms {rms:.2e}")
    # composition check: the spectrum JVP along the FD structure tangent of the re-converged models reproduces
    # their spectrum difference (isolates the SYNTHE part from the accuracy of dx*/dtheta)
    n = chain.fp.n
    xfd = (reconverged[(th, 1)] - reconverged[(th, -1)]) / (2 * STEPS[th])
    tg = {nm: torch.as_tensor(xfd[i * n:(i + 1) * n], dtype=torch.float64)
          for i, nm in enumerate(("RHOX", "T", "P", "XNE"))}
    if th == "mh":
        tg["mh"] = 1.0
    r = chain.synthe().jvp(chain.structure_params(chain.xstar), tg)
    mx2, rms2 = _rel_stats(r["dnorm"].numpy(), fd.numpy())
    print(th, f"spectrum (FD structure tangent): max {mx2:.2e} rms {rms2:.2e}")
    assert rms2 < 1e-2
    # end to end: limited by the structure tangent (T at the convective boundary), see RESULTS (M8)
    assert rms < {"teff": 0.1, "logg": 0.12, "mh": 0.03}[th]
