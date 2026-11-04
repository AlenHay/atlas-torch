"""Oracle tests for KAPP (atlas7v.for, SYNTHE variant) on the xnfpelsyn path (IFOP 1..13, N=1).

1. golden: the three traced KAPP calls of the solar run, chaining the port's SAVE state from call to call
   (the populations LUKEOP/HOTOP get from POPS at the first call are taken from their exit traces).
2. single calls vs the Fortran on perturbed states, with fresh caches and with caches warmed by a previous
   call at another frequency (ITEMP caches reused, CHOP/OHOP FREQ1 different).
3. kapp_batch over many frequencies (xnfpelsyn's loop, including a repeated frequency) vs the Fortran KAPP
   called in sequence after xnfpelsyn's /FREQ/ setup.
4. native torch libm (EXACT = False, the device-agnostic path) stays within 1e-12 of the Fortran.
"""
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.state import State
from atlas_torch.synthe.kapp import OPTOT, ZEROED, kapp, kapp_batch
from atlas_torch.synthe.opacity import _core
from atlas_torch.synthe.opacity import _testing as T

ARRAYS = [f"{b}.{v}" for b, vs in ZEROED.items() for v in vs] + [f"optot.{k}" for k in OPTOT]
SAVES = ["save.hminop.itemp1", "save.hminop.xhmin", "save.hminop.theta", "save.h2raop.xnh2", "save.h2collop.xnh2",
         "save.coolop.ah2coll", "save.chop.freq1", "save.ohop.freq1", "save.n1op.c1130", "save.si2op.dt",
         "save.si2op.nt", "save.lukeop.xnfpo", "save.hotop.xnfsumqq"]
KEYS = ARRAYS + ["flux.flxrad", "freq.waveno"] + SAVES
NRHOX = 72


def _populations(golden):
    """LUKEOP/HOTOP SAVEd populations after their first (POPS) call in the solar run."""
    pop = {}
    for u in ("lukeop", "hotop"):
        e = T.trace_state(golden(T.STAGE, u, 0, "exit"), u)
        pop.update({k: v for k, v in e.items() if k.startswith(f"save.{u}.")})
    return pop


def test_kapp_golden(golden):
    saves = _populations(golden)
    for hit in range(3):
        s = T.base_state(golden, hit)
        s.update(saves)
        out = kapp(s)
        e = State.from_trace(golden(T.STAGE, "kapp", hit, "exit"))
        T.check(out, e, [k for k in out if k in e], name=f"kapp[{hit}] ")
        saves.update({k: v for k, v in out.items() if k.startswith("save.")})


def _case(golden, seed, freq, tscale=(0.8, 1.25)):
    rng = np.random.default_rng(seed + 1)
    s = T.random_case(golden, seed, freq, tscale=tscale)
    s.update(_populations(golden))
    return T.stale_saves(T.stale_saves(s, "lukeop", {k: ((99,), 1e11) for k in
                                                     ("xnfpn", "xnfpo", "xnfpmg", "xnfpsi", "xnfpca")}, rng),
                         "hotop", {"xnfp": ((99, 21), 1e9), "xnfsumqq": ((99, 5), 1e10)}, rng)


@settings(max_examples=40, deadline=None)
@given(lf=st.floats(12.5, 16.5), seed=st.integers(0, 2**32 - 1), warm=st.booleans(),
       tscale=st.sampled_from([(0.8, 1.25), (0.4, 0.7), (1.5, 3.0)]))
def test_kapp_vs_fortran(oracle, golden, lf, seed, warm, tscale):
    s = _case(golden, seed, 10.0**lf, tscale)
    o = T.run_oracle(oracle, s, saves=["lukeop", "hotop"])
    if warm:                                     # a previous KAPP call at another frequency
        s0 = T.with_freq(s, 10.0 ** (lf - 0.37))
        T.push(o, s0, ["freq"])
        o.call("kapp", 1, 0, 0.0)
        T.push(o, s, ["freq", "flux"])
        s = s.updated({k: v for k, v in kapp(s0).items() if k.startswith("save.")})
    o.call("kapp", 1, 0, 0.0)
    T.check(kapp(s), T.oracle_ref(o, "kapp"), KEYS, name=f"kapp@{10**lf:.6e} warm={warm} ")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kapp_batch_vs_fortran_sequence(oracle, golden, seed):
    rng = np.random.default_rng(seed)
    s = _case(golden, 100 + seed, 1e15)
    freqs = np.sort(10.0 ** rng.uniform(12.5, 16.5, 40))[::-1].copy()
    freqs = np.insert(freqs, 20, freqs[20])      # a repeated frequency (CHOP/OHOP FREQ1 cache hit)
    out = kapp_batch(s, freqs)
    o = T.run_oracle(oracle, s, saves=["lukeop", "hotop"])
    for i, f in enumerate(freqs):
        T.push(o, T.with_freq(s, float(f)), ["freq"])
        o.call("kapp", 1, 0, 0.0)
        ref = T.oracle_ref(o, "kapp")
        T.check({k: out[k][i] for k in ARRAYS}, lambda k: ref(k)[:NRHOX], ARRAYS, name=f"batch[{i}] ")
        T.check({"freq.waveno": out["freq.waveno"][i:i + 1]}, ref, ["freq.waveno"], name=f"batch[{i}] ")
        for k in ("freq.ehvkt", "freq.stim", "freq.bnu"):
            T.check({k: out[k][i]}, lambda k: ref(k)[:NRHOX], [k], name=f"batch[{i}] ")
    T.check(out, T.oracle_ref(o, "kapp"), [k for k in SAVES if k != "save.coolop.ah2coll"] + ["flux.flxrad"],
            name="batch saves ")


def test_kapp_native_libm(oracle, golden):
    """Device-agnostic path (torch exp/log/pow/sqrt): agreement to ~1e-13 (ulp differences only)."""
    s = _case(golden, 5, 1e15)
    freqs = np.sort(10.0 ** np.random.default_rng(5).uniform(12.5, 16.5, 25))[::-1].copy()
    old = _core.EXACT
    _core.EXACT = False
    try:
        out = kapp_batch(s, freqs)
    finally:
        _core.EXACT = old
    o = T.run_oracle(oracle, s, saves=["lukeop", "hotop"])
    worst = 0.0
    for i, f in enumerate(freqs):
        T.push(o, T.with_freq(s, float(f)), ["freq"])
        o.call("kapp", 1, 0, 0.0)
        for k in ("optot.acont", "optot.scont", "optot.sigmac"):
            b, v = k.split(".")
            worst = max(worst, T.rel(out[k][i], o.common(b)[v][:NRHOX]))
    assert worst < 1e-12, worst
