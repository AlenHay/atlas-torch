"""Frequency-batched continuum (physics/opacity_batch.py) vs the per-frequency KAPP (physics/opacity.py, itself
tested against the Fortran): every /OPS/ array must be bit-identical, for the golden states and perturbed layers
(T 3000..50000 K), both variants, on a dense log-frequency grid plus every frequency branch threshold of the
sources (edges are where a batched mask could disagree with the scalar IF)."""
import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics import opacity as O
from atlas_torch.atlas9.physics.api import _with_defaults
from atlas_torch.atlas9.physics.opacity_batch import FreqBatch, continuum as continuum_batch

EDGES = [4.05933e13, 1.82365e14, 2.055e14, 1.31522e14, 1.443e15, 2.0761e15, 2.110779e15, 2.4196e15, 2.460127e15,
         2.564306e15, 2.653317e15, 2.7254e15, 2.870454e15, 2.941534e15, 3.28805e15, 3.517915e15, 3.635492e15,
         5.945209e15, 1.25408e16, 21000.0 * 2.99792458e10, 20000.0 * 2.99792458e10]


def _freqs(seed):
    rng = np.random.default_rng(seed)
    f = list(np.logspace(12.5, 16.6, 160)) + list(10.0 ** rng.uniform(12.5, 16.6, 40))
    hefreq = O._tab("atlas9", "he1op", "hefreq")
    A = O._eq("atlas9", "hotop", "a")
    for e in EDGES + [float(x) for x in hefreq] + [float(A[7 * i]) for i in range(60)]:
        f += [e, float(np.nextafter(e, 0)), float(np.nextafter(e, np.inf))]
    return [float(x) for x in f]


def _check(s, freqs, variant):
    s = _with_defaults(s, variant)
    n = s.iscalar("rhox.nrhox")
    fqs = [O.Freq.derive(s, n, f) for f in freqs]
    fb = FreqBatch(freqs, [q.freqlg for q in fqs], torch.stack([q.ehvkt for q in fqs]),
                   torch.stack([q.stim for q in fqs]), torch.stack([q.bnu for q in fqs]))
    got = continuum_batch(s, fb, variant)
    for i, q in enumerate(fqs):
        ref = O.kapp(s, q, variant)
        for k in O.OPS:
            r = ref["ops." + k][:n]
            g = got[k][i] if k in got else torch.zeros(n, dtype=torch.float64)
            assert torch.equal(g.view(torch.int64), r.view(torch.int64)), (variant, freqs[i], k)


@pytest.mark.parametrize("hit", [0, 1, 2])
def test_batched_continuum_golden(golden, hit):
    _check(T.opacity_state(golden, "kapp", hit), _freqs(hit), "atlas9")


@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 2**31 - 1), variant=st.sampled_from(["atlas9", "kappa9"]))
def test_batched_continuum_perturbed(golden, seed, variant):
    s = T.perturbed_state(T.opacity_state(golden, "kapp", 0), seed)
    _check(s, _freqs(seed)[::3], variant)
