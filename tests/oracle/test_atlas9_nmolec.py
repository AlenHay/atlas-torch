"""Oracle tests for NMOLEC (+ SOLVIT, PFSAHA inside it) of the ATLAS9 variant: molecular/ionisation
equilibrium Newton solve, layer recurrence, IFEDNS=1 internal-energy branch with its T perturbations.
golden: the three traced calls (iteration 1 and the first two CONVEC calls); bitwise under the trace's
glibc. On another glibc an ulp in exp can flip the 1E-4 convergence test in one layer (seen: hit 1, layer 71
on glibc 2.43, 7.5e-8 relative), hence 1e-6 there. property: perturbed T and P vs the Fortran, bitwise.
"""
import numpy as np
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics.molec import nmolec


def test_nmolec_golden(golden):
    tol = T.golden_tol(smooth=False)
    for hit in range(3):
        s = T.nmolec_state(golden, hit)
        x = golden(T.TRACE_STAGE, "nmolec", hit, "exit")
        out = nmolec(s, int(golden(T.TRACE_STAGE, "nmolec", hit, "entry")["arg:mode"][0]))
        exp = {k: x[k] for k in T.NMOLEC_OUT if k in x}
        T.compare_nmolec(out, exp, s.iscalar("rhox.nrhox"), tol, f"nmolec hit {hit}: ")


@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(hit=st.sampled_from([0, 1]), ft=st.floats(0.85, 1.2), fp=st.floats(0.3, 3.0), seed=st.integers(0, 2**31 - 1))
def test_nmolec_vs_fortran(golden, hit, ft, fp, seed):
    s = T.nmolec_state(golden, hit)
    rng = np.random.default_rng(seed)
    w = torch.from_numpy(np.exp(rng.normal(0, 0.01, 99)))
    T_ = s["temp.t"] * ft * w
    TK = 1.38054e-16 * T_
    s.update({"temp.t": T_, "temp.tk": TK, "temp.tkev": 8.6171e-5 * T_, "temp.hckt": 6.6256e-27 / TK * 2.99792458e10,
              "temp.hkt": 6.6256e-27 / TK, "temp.tlog": torch.log(T_), "state.p": s["state.p"] * fp})
    out = nmolec(s, 1)
    exp = T.oracle_nmolec(T.make_oracle("atlas9"), s, 1)
    T.compare_nmolec(out, exp, s.iscalar("rhox.nrhox"), 0.0, f"nmolec hit={hit} ft={ft} fp={fp}: ")


def test_solvit_vs_fortran():
    import ctypes
    from atlas_torch.atlas9.physics.molec import solvit
    o = T.make_oracle("atlas9")
    rng = np.random.default_rng(3)
    for n in (2, 3, 5, 9, 18, 25):
        for _ in range(20):
            a = rng.normal(size=(n, n)) * 10.0 ** rng.uniform(-5, 5, (n, n))
            a[rng.random((n, n)) < 0.3] = 0.0
            a += np.diag(10.0 ** rng.uniform(-3, 3, n))
            b = rng.normal(size=n)
            fa, fb, piv = np.asfortranarray(a.copy()), b.copy(), np.zeros(n, np.int32)
            o.call("solvit", fa, n, fb, piv)
            assert np.array_equal(solvit(a, b), fb)


def test_readmol_matches_trace(golden, ext):
    """READMOL on BasicATLAS' molecules.dat reproduces COMMON /IFEQUA/ of the traced run."""
    from atlas_torch.atlas9.physics.molec import readmol
    e = golden(T.TRACE_STAGE, "nmolec", 0, "entry")
    r = readmol(ext / "BasicATLAS" / "data" / "atlas_files" / "molecules.dat")
    for k, v in r.items():
        assert np.array_equal(np.asarray(v).reshape(-1), e[k].reshape(-1).astype(np.asarray(v).dtype)), k
