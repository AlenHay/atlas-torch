"""Test helpers for the atlas7v opacity oracle tests (golden traces + ctypes oracle).

* `trace_state(golden_dict, *units)`: State from a trace snapshot, with the snapshot's SAVEd locals
  ("static:<var>") loaded as "save.<unit>.<var>" in their Fortran shapes (taken from the oracle's DWARF).
* `perturbed(s, rng, freq)`: physically plausible variation of a solar State (T, XNE, populations,
  departure coefficients) at frequency `freq`, with /TEMP/ and /FREQ/ made consistent.
* `push(o, s, blocks)` / `push_saves(o, s, unit)`: copy State entries into the oracle's COMMONs/SAVEs.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ...state import State
from ._core import Ctx

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
STAGE = "sun_500_510/xnfpelsyn"

POP_BLOCKS = {
    "bhyd": ["xnfph", "xnfh"], "bhe": ["xnfphe", "xnfhe"], "bc": ["xnfpc"], "bmg": ["xnfpmg"],
    "bal": ["xnfpal"], "bsi": ["xnfpsi"], "bfe": ["xnfpfe"], "bca": ["xnfpca"], "bo": ["xnfpo"],
    "bb": ["xnfpb"], "bk": ["xnfpk"], "bna": ["xnfpna"], "choh": ["xnfpch", "xnfpoh"],
}
DEPARTURE = {"bhyd": ["bhyd", "bmin"], "bhe": ["bhe1", "bhe2"], "bc": ["bc1", "bc2"], "bmg": ["bmg1", "bmg2"],
             "bal": ["bal1", "bal2"], "bsi": ["bsi1", "bsi2"], "bfe": ["bfe1"], "bca": ["bca1", "bca2"],
             "bo": ["bo1", "bo2"]}


@lru_cache(maxsize=None)
def _layout():
    from ...testing.fortran_oracle import read_layout
    return read_layout(str(EXT / "oracle" / "libatlas7v.so"))


def static_shape(unit: str, var: str):
    return _layout()[1][unit.lower()].statics[var.lower()].shape


def trace_state(d: dict, *units: str) -> State:
    s = State.from_trace(d)
    for unit in units:
        for k, v in d.items():
            if k.startswith("static:"):
                var = k.split(":", 1)[1]
                try:
                    shp = static_shape(unit, var)
                except KeyError:
                    continue
                a = np.asarray(v)
                a = a.reshape(shp, order="F") if shp else a.reshape(1)
                s[f"save.{unit}.{var}"] = torch.from_numpy(np.ascontiguousarray(
                    a.astype(np.float64) if a.dtype.kind == "f" else a.astype(np.int64)))
    return s


def with_freq(s: State, freq: float) -> State:
    """/FREQ/ for `freq` as xnfpelsyn sets it (layers > NRHOX keep their values)."""
    c = Ctx.from_freqs(s, [freq])
    n = c.n
    out = {"freq.freq": torch.tensor([freq]), "freq.freqlg": c.freqlg.reshape(1),
           "freq.waveno": c.waveno.reshape(1)}
    for k, v in (("ehvkt", c.ehvkt), ("stim", c.stim), ("bnu", c.bnu)):
        a = s[f"freq.{k}"].clone()
        a[:n] = v[0]
        out[f"freq.{k}"] = a
    return s.updated(out)


def perturbed(s: State, rng: np.random.Generator, freq: float | None = None, tscale=(0.8, 1.25),
              pop_dex=0.3, b_jitter=0.05) -> State:
    n = s.iscalar("rhox.nrhox")
    ch = {}
    t = s["temp.t"].clone()
    t[:n] = t[:n] * rng.uniform(*tscale) * (1.0 + 0.02 * torch.from_numpy(rng.standard_normal(n)))
    ch["temp.t"] = t
    ch["temp.tkev"] = 8.6171e-5 * t
    ch["temp.tk"] = 1.38054e-16 * t
    ch["temp.hkt"] = torch.where(t > 0, 6.6256e-27 / torch.where(t > 0, ch["temp.tk"], torch.ones_like(t)),
                                 torch.zeros_like(t))
    ch["temp.tlog"] = torch.where(t > 0, torch.log(torch.where(t > 0, t, torch.ones_like(t))), torch.zeros_like(t))
    ch["temp.hckt"] = ch["temp.hkt"] * 2.99792458e10
    ch["state.xne"] = s["state.xne"] * 10.0 ** rng.uniform(-pop_dex, pop_dex)
    ch["state.rho"] = s["state.rho"] * 10.0 ** rng.uniform(-0.2, 0.2)
    for blk, vs in POP_BLOCKS.items():
        for v in vs:
            k = f"{blk}.{v}"
            if k in s:
                ch[k] = s[k] * torch.from_numpy(10.0 ** rng.uniform(-pop_dex, pop_dex, s[k].shape[1:] or (1,)))
    for blk, vs in DEPARTURE.items():
        for v in vs:
            k = f"{blk}.{v}"
            if k in s:
                ch[k] = s[k] * torch.from_numpy(rng.uniform(1.0 - b_jitter, 1.0 + b_jitter, s[k].shape))
    s2 = s.updated(ch)
    return with_freq(s2, float(s.scalar("freq.freq")) if freq is None else freq)


def push(o, s: State, blocks):
    for blk in blocks:
        vals = {k.split(".", 1)[1]: v.numpy() for k, v in s.items() if k.startswith(blk + ".")}
        o.set_common(blk, **vals)


def push_saves(o, s: State, unit: str):
    views = o.saved(unit)
    pre = f"save.{unit}."
    for k, v in s.items():
        if k.startswith(pre):
            var = k[len(pre):]
            dst = views[var]
            src = np.asarray(v.numpy(), dtype=dst.dtype)
            if dst.ndim == 1 and dst.shape == (1,):
                dst[0] = src.reshape(-1)[0]
            else:
                dst[...] = src.reshape(dst.shape)


ALL_BLOCKS = ["bal", "bb", "bc", "bca", "bfe", "bhe", "bhyd", "bk", "bmg", "bna", "bo", "bsi", "choh", "flux",
              "freq", "if", "ops", "optot", "rhox", "state", "temp"]


def base_state(golden, hit: int = 0) -> State:
    """Full solar State at KAPP entry (all COMMONs) + /CHOH/ and /FLUX/ from the COOLOP/FE1OP traces."""
    s = State.from_trace(golden(STAGE, "kapp", hit, "entry"))
    for r in ("coolop", "fe1op"):
        d = State.from_trace(golden(STAGE, r, hit, "entry"))
        for k, v in d.items():
            if k not in s:
                s[k] = v
    return s


def rel(a, b) -> float:
    a = np.asarray(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a, dtype=np.float64)
    b = np.asarray(b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b, dtype=np.float64)
    d = np.abs(a - b)
    m = np.maximum(np.abs(a), np.abs(b))
    r = np.where(d == 0, 0.0, d / np.where(m > 0, m, 1.0))
    return float(r.max(initial=0.0))


# ------------------------------------------------------------------------------- generic test drivers
RTOL = 1e-15     # the port is bit-exact with EXACT libm; this only absorbs nothing-but-ulp platform noise

#: sub-unit traces whose COMMONs/SAVEs a routine's golden snapshot needs (the routine itself does not
#: declare those COMMONs, or its callees keep their own SAVEs)
GOLDEN_MERGE = {
    "coolop": ("h2collop", "c1op", "fe1op"),
    "lukeop": ("n1op", "mg2op", "ca2op", "si2op"),
}


def golden_state(golden, unit: str, hit: int):
    """(entry State incl. save.<unit>.* and merged sub-unit state, exit State, exit dict)."""
    d_in, d_out = golden(STAGE, unit, hit, "entry"), golden(STAGE, unit, hit, "exit")
    s, e = trace_state(d_in, unit), trace_state(d_out, unit)
    for u in GOLDEN_MERGE.get(unit, ()):
        su = trace_state(golden(STAGE, u, hit, "entry"), u)
        for k, v in su.items():
            if k.startswith("save.") or k not in s:
                s[k] = v
        eu = trace_state(golden(STAGE, u, hit, "exit"), u)
        for k, v in eu.items():          # callee SAVEs, and COMMONs the routine itself does not declare
            if (k.startswith(f"save.{u}.") or not k.startswith("save.")) and k not in e:
                e[k] = v
    if unit == "coolop":
        # CHOP/OHOP run once per layer (their trace hits are J=1,2,3 of the first call): FREQ1 at COOLOP
        # entry is the previous frequency, at exit the current one
        prev = golden(STAGE, "coolop", hit - 1, "entry")["freq.freq"] if hit > 0 else np.zeros(1)
        for u in ("chop", "ohop"):
            s[f"save.{u}.freq1"] = torch.from_numpy(np.asarray(prev, dtype=np.float64).reshape(1))
            e[f"save.{u}.freq1"] = e["freq.freq"].clone()
    if unit in ("lukeop", "hotop") and save_itemp_of(s, unit) != s.iscalar("temp.itemp"):
        # first call refreshes the populations through POPS (not ported here): take them from the exit
        for k, v in e.items():
            if k.startswith(f"save.{unit}.") and not k.endswith(".itemp1"):
                s[k] = v
        s[f"save.{unit}.itemp1"] = torch.tensor([s.iscalar("temp.itemp")])
    return s, e, d_out


def save_itemp_of(s: State, unit: str) -> int:
    k = f"save.{unit}.itemp1"
    return int(s[k].reshape(-1)[0]) if k in s else 0


def check(out: dict, ref, keys, rtol=RTOL, name=""):
    """Compare out[k] with ref (State or callable k -> array); returns {k: max rel err}."""
    from ...testing.compare import assert_close
    errs = {}
    for k in keys:
        exp = ref(k) if callable(ref) else ref[k]
        got = out[k]
        exp = np.asarray(exp.numpy() if isinstance(exp, torch.Tensor) else exp)
        got = got.reshape(exp.shape) if isinstance(got, torch.Tensor) else np.asarray(got).reshape(exp.shape)
        errs[k] = rel(got, exp)
        assert_close(got, exp, rtol=rtol, name=f"{name}{k}")
    return errs


def golden_check(golden, unit: str, fn, keys, hits=range(3), **kw):
    errs = {}
    for hit in hits:
        s, e, d = golden_state(golden, unit, hit)
        out = fn(s, **kw)
        ref = lambda k: d[k] if k.startswith("arg:") else e[k]  # noqa: E731
        for k, v in check(out, ref, keys, name=f"{unit}[{hit}] ").items():
            errs[k] = max(errs.get(k, 0.0), v)
    return errs


def oracle_ref(o, unit: str):
    """k -> the oracle's value for State key k (COMMON block.var or save.<unit>.var)."""
    def get(k):
        if k.startswith("save."):
            _, u, v = k.split(".")
            return o.saved(u)[v]
        b, v = k.split(".")
        return o.common(b)[v]
    return get


def random_case(golden, seed: int, freq: float, **kw) -> State:
    rng = np.random.default_rng(seed)
    return perturbed(base_state(golden), rng, freq, **kw)


def run_oracle(oracle, s: State, unit: str | None = None, saves=(), args=()):
    o = oracle("atlas7v")
    push(o, s, ALL_BLOCKS)
    for u in saves:
        push_saves(o, s, u)
    if unit is not None:
        o.call(unit, *args)
    return o


def stale_saves(s: State, unit: str, arrays: dict, rng: np.random.Generator) -> State:
    """Mark the ITEMP cache of `unit` valid (ITEMP1=ITEMP) and fill its arrays with arbitrary values, so a
    test can tell whether the cached arrays (not the current T, ...) are used."""
    ch = {f"save.{unit}.itemp1": torch.tensor([s.iscalar("temp.itemp")])}
    for var, (shape, kind) in arrays.items():
        if kind == "int":
            ch[f"save.{unit}.{var}"] = torch.from_numpy(rng.integers(1, 6, shape)).to(torch.int64)
        elif kind == "unit":
            ch[f"save.{unit}.{var}"] = torch.from_numpy(rng.uniform(0.0, 1.0, shape))
        else:
            base = kind
            ch[f"save.{unit}.{var}"] = base * torch.from_numpy(10.0 ** rng.uniform(-0.5, 0.5, shape))
    return s.updated(ch)
