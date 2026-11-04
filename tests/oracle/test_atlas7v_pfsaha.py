"""Oracle tests for PFSAHA (atlas7v.for, SYNTHE variant): partition functions / Saha fractions.

  golden:   the traced PFSAHA hits of the solar xnfpelsyn run (entry State -> port -> ANSWER(J,*) at exit)
  property: perturbed layers (T*U(0.8,1.25), P*10^U(-0.5,0.5), XNE*10^U(-0.5,0.5), random departure
            coefficients, NLTEON in {-1,0,1}), a cool (~3500 K at the top) and a hot (~20000 K at the bottom)
            model, every element / mode / NION drawn at random -> port vs the Fortran PFSAHA, bitwise.

This module also holds the model/oracle helpers shared by the other test_atlas7v_* population tests.
"""
import numpy as np
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.state import State
from atlas_torch.synthe.pfsaha import Layers, nions_of, pfsaha, potion
from atlas_torch.testing.compare import assert_close, err_stats

STAGE = "sun_500_510/xnfpelsyn"
B_BLOCKS = ("bhyd", "bhe", "bc", "bmg", "bal", "bsi", "bca", "bo", "bna", "bb", "bk")
B_ARRAYS = {"bhyd": ("bhyd",), "bhe": ("bhe1", "bhe2"), "bc": ("bc1", "bc2"), "bmg": ("bmg1", "bmg2"),
            "bal": ("bal1",), "bsi": ("bsi1", "bsi2"), "bca": ("bca1", "bca2"), "bo": ("bo1",), "bna": ("bna1",),
            "bb": ("bb1",), "bk": ("bk1",)}
MODEL_BLOCKS = ("temp", "state", "rhox", "if", "xabund", "elem", "ifequa", "xnmol", "iter") + B_BLOCKS
SETTINGS = dict(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])

MAXREL = {}      # routine -> max relative error seen (printed by the tests with -s)


def note(name, got, exp):
    e = err_stats(got, exp)["max_rel"]
    MAXREL[name] = max(MAXREL.get(name, 0.0), e)
    return e


def solar_state(golden) -> State:
    """COMMON image of the traced xnfpelsyn run right after READIN (incl. READMOL), as the first POPS sees it."""
    d = dict(golden(STAGE, "readin", 0, "exit"))
    d.update(golden(STAGE, "readmol", 0, "exit"))
    s = State.from_trace(d)
    s["xnmol.codemol"] = s["xnmol.code"]
    s["potion.potion"] = torch.from_numpy(potion().copy())
    s["temp.itemp"] = torch.tensor([1])
    s["if.ifpres"] = torch.tensor([1])
    s["iter.iter"] = torch.tensor([1])
    s["iter.numits"] = torch.tensor([2])          # ITER<NUMITS: the Fortran skips its printout
    return s


def with_temperature(s: State, t: torch.Tensor, p=None, xne=None) -> State:
    """Replace T (and optionally P, XNE) of the first NRHOX layers; derived /TEMP/ quantities as in READIN's
    BEGIN block."""
    n = s.iscalar("rhox.nrhox")
    out = State(s)

    def put(key, v):
        a = s[key].clone()
        a[:n] = v
        out[key] = a
    tk = 1.38054e-16 * t
    hkt = torch.tensor(6.6256e-27, dtype=torch.float64) / tk
    put("temp.t", t)
    put("temp.tk", tk)
    put("temp.hkt", hkt)
    put("temp.hckt", hkt * 2.99792458e10)
    put("temp.tkev", 8.6171e-5 * t)
    put("temp.tlog", torch.log(t))
    if p is not None:
        put("state.p", p)
    if xne is not None:
        put("state.xne", xne)
    return out


def model(golden, kind: str, seed: int, nlte: bool = False) -> State:
    """solar | perturbed | cool | hot variants of the traced solar model (see module docstring)."""
    s = solar_state(golden)
    n = s.iscalar("rhox.nrhox")
    rng = np.random.default_rng(seed)
    t0, p0, x0 = s["temp.t"][:n], s["state.p"][:n], s["state.xne"][:n]
    if kind == "solar":
        t, p, x = t0, p0, x0
    elif kind == "perturbed":
        t = t0 * torch.from_numpy(rng.uniform(0.8, 1.25, n))
        p = p0 * torch.from_numpy(10.0 ** rng.uniform(-0.5, 0.5, n))
        x = x0 * torch.from_numpy(10.0 ** rng.uniform(-0.5, 0.5, n))
    elif kind == "cool":          # ~3500 K upper layers: molecules matter
        f = 3500.0 / float(t0[0])
        t = t0 * f * torch.from_numpy(rng.uniform(0.98, 1.02, n))
        p = p0 * torch.from_numpy(10.0 ** rng.uniform(-0.2, 0.2, n))
        x = x0 * 0.3
    elif kind == "hot":           # ~20000 K at the bottom
        t = t0 * (20000.0 / float(t0[-1])) * torch.from_numpy(rng.uniform(0.98, 1.02, n))
        p = p0 * torch.from_numpy(10.0 ** rng.uniform(-0.2, 0.2, n))
        x = x0 * 30.0
    else:
        raise ValueError(kind)
    s = with_temperature(s, t.clone(), p.clone(), x.clone())
    if nlte:
        for blk, names in B_ARRAYS.items():
            for nm in names:
                k = f"{blk}.{nm}"
                s[k] = torch.from_numpy(rng.uniform(0.5, 2.0, tuple(s[k].shape)))
    return s


def load_oracle(o, s: State, blocks=MODEL_BLOCKS):
    """Copy the State's COMMON members into the oracle library."""
    for blk in blocks:
        cm = o.common(blk)
        vals = {}
        for k, v in s.items():
            b, _, var = k.partition(".")
            if b == blk and var in cm and isinstance(v, torch.Tensor):
                vals[var] = v.numpy()
        o.set_common(blk, **vals)


def oracle_pfsaha(o, n: int, iz: int, nion: int, mode: int) -> np.ndarray:
    ans = np.zeros((99, 31), order="F")
    for j in range(1, n + 1):
        o.call("pfsaha", j, iz, nion, mode, ans)
    return ans


def _cols(ans, n, nion, mode):
    return ans[:n, :nion] if mode > 10 and mode != 14 else ans[:n, 0]


# ------------------------------------------------------------------------------------------------ golden

def test_pfsaha_golden(golden):
    for hit in range(3):
        e = golden(STAGE, "pfsaha", hit, "entry")
        x = golden(STAGE, "pfsaha", hit, "exit")
        s = State.from_trace(e)
        j, iz, nion, mode = (int(e[f"arg:{k}"][0]) for k in ("j", "iz", "nion", "mode"))
        assert np.array_equal(e["potion.potion"], potion())
        L = Layers.from_state(s, n=j)
        got = pfsaha(L, iz, nion, mode)
        exp = x["arg:answer"].reshape(99, 31, order="F")
        if mode > 10 and mode != 14:
            assert_close(got[j - 1], exp[j - 1, :nion], rtol=0, name=f"pfsaha hit {hit}")
        else:
            assert_close(got[j - 1], exp[j - 1, 0], rtol=0, name=f"pfsaha hit {hit}")
        note("pfsaha", got[j - 1], exp[j - 1, :nion] if mode > 10 else exp[j - 1, 0])


# ---------------------------------------------------------------------------------------------- property

@settings(max_examples=40, **SETTINGS)
@given(kind=st.sampled_from(["perturbed", "cool", "hot"]), seed=st.integers(0, 2**31 - 1),
       nlteon=st.sampled_from([-1, 0, 1]), nlte=st.booleans(),
       izs=st.lists(st.integers(1, 99), min_size=4, max_size=8, unique=True))
def test_pfsaha_vs_fortran(oracle, golden, kind, seed, nlteon, nlte, izs):
    s = model(golden, kind, seed, nlte=nlte)
    s["if.nlteon"] = torch.tensor([nlteon])
    n = s.iscalar("rhox.nrhox")
    o = oracle("atlas7v")
    load_oracle(o, s, ("temp", "state", "if") + B_BLOCKS)
    L = Layers.from_state(s)
    rng = np.random.default_rng(seed)
    for iz in sorted(set(izs) | {1, 2, 6, 8, 12, 14, 26}):
        nion = int(rng.integers(1, nions_of(iz) + 1))
        for mode in (1, 2, 3, 4, 11, 12, 13):
            got = pfsaha(L, iz, nion, mode)
            exp = _cols(oracle_pfsaha(o, n, iz, nion, mode), n, nion, mode)
            note("pfsaha", got, exp)
            assert_close(got, exp, rtol=0, name=f"pfsaha {kind} iz={iz} nion={nion} mode={mode} nlteon={nlteon}")
