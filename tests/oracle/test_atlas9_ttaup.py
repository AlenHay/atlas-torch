"""TTAUP (atlas9mem.for): hydrostatic integration on a log tau grid vs the Fortran subroutine.

(The golden trace records only first elements of TTAUP's array arguments; its full golden coverage is the
READIN test, whose SCALE card integrates the BasicATLAS restart model with TTAUP.)"""
from pathlib import Path

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9.commons import initial_state
from atlas_torch.atlas9.rosstab import Rosstab, read_kappa_ros
from atlas_torch.atlas9.ttaup import ttaup


@settings(max_examples=25, deadline=None)
@given(teff=st.floats(4000.0, 9000.0), logg=st.floats(2.5, 5.0), prad0=st.sampled_from([0.0, 0.05, 1.0]),
       vt=st.sampled_from([0.0, 1e5, 2e5]), iross=st.sampled_from([0, 1]))
def test_ttaup_vs_fortran(oracle, ext, teff, logg, prad0, vt, iross):
    _, tab = read_kappa_ros(Path(ext) / "BasicATLAS/data/solar_ODF/kappa.ros")
    s = initial_state(); s["tabtp.tabkap"] = tab; s["tabtp.iftabk"] = torch.tensor([1])
    o = oracle("atlas9")
    o.set_common("tabtp", tabkap=tab.numpy(), iftabk=1)
    n = 72
    tau = 10 ** (-6.875 + 0.125 * np.arange(n))
    t = teff * (0.75 * (0.710 + tau - 0.1331 * np.exp(-3.4488 * tau))) ** 0.25
    z = lambda a=None: np.asfortranarray(np.r_[a, np.zeros(99 - n)] if a is not None else np.zeros(99))
    tf, tauf, prad, pturb, vturb = z(t), z(tau), z(prad0 * (1 + np.linspace(0, 0.3, n))), z(), z(np.full(n, vt))
    ab0 = z(np.full(n, 3e-4))
    ab, pt, p = ab0.copy(), z(), z()
    grav = 10.0 ** logg
    o.call("ttaup", tf, tauf, ab, pt, p, prad, pturb, vturb, grav, n, iross)
    A, PT, P = ttaup(torch.from_numpy(tf), torch.from_numpy(tauf), torch.from_numpy(ab0), torch.from_numpy(prad),
                     torch.from_numpy(pturb), torch.from_numpy(vturb), grav, n, iross, Rosstab(s))
    assert np.array_equal(A.numpy(), ab[:n]) and np.array_equal(PT.numpy(), pt[:n]) and np.array_equal(P.numpy(), p[:n])
