"""Test-only helpers for the ATLAS9/KAPPA9 physics oracle tests (never imported by the port itself).

- oracle(variant): Oracle over $ATLAS_EXT/oracle/libatlas9.so | libkappa9.so. The ifx-built KAPPA9 library
  needs the Intel runtime (libifport, libifcoremt, libimf, libsvml, libintlc); they are preloaded
  RTLD_GLOBAL from $INTEL_LIB, /opt/intel/oneapi/compiler/*/lib or the atlas-dev container image.
- Canonical COMMON layouts: routines declare /OPS/, /XNFP/ with dummy names (D1(kw,19), XNFP(kw,21) ...).
  The State always uses the main program's member names; `canonical_from_trace` rebuilds them from a golden
  trace using the traced addresses, `canonical_from_oracle` from the oracle's memory.
"""
from __future__ import annotations

import ctypes
import glob
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ...state import State

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
KW = 99

# main-program declarations (atlas9mem.for = kappa9.for): block -> [(name, ncols)] of REAL*8 (kw, ncols)
CANON = {
    "ops": [(n, 1) for n in ("ahyd ah2p ahmin sigh ahe1 ahe2 ahemin sighe acool aluke ahot sigel sigh2 ahline "
                             "alines siglin axline sigxl axcont sigx shyd shmin shline sxline sxcont").split()],
    "xnfp": [("xnfpc", 4), ("xnfpn", 5), ("xnfpo", 6), ("xnfpne", 6), ("xnfpal", 1), ("xnfpmg", 2),
             ("xnfpsi", 2), ("xnfpca", 2), ("xnfpfe", 1), ("xnfpch", 1), ("xnfpoh", 1)],
    "xnf": [("xnfc", 6), ("xnfn", 6), ("xnfo", 6), ("xnfne", 6), ("xnfmg", 6), ("xnfsi", 6), ("xnfs", 6),
            ("xnffe", 5)],
    "ions": [("xnfph", 2), ("xnfphe", 3), ("xnfh", 2), ("xnfhe", 3)],
}


# members declared (kw,1): keep the 2-D Fortran shape
_TWO_D_1 = {"xnfp.xnfpal", "xnfp.xnfpfe"}


def _split(block: str, flat: np.ndarray) -> dict[str, np.ndarray]:
    out, o = {}, 0
    for name, nc in CANON[block]:
        a = flat[o:o + KW * nc].reshape((KW, nc), order="F")
        key = f"{block}.{name}"
        out[key] = a[:, 0].copy() if nc == 1 and key not in _TWO_D_1 else a.copy()
        o += KW * nc
    return out


def block_size(block: str) -> int:
    return sum(nc for _, nc in CANON[block]) * KW


# ------------------------------------------------------------------------------------------- traces

@lru_cache(maxsize=4)
def _index(stage_dir: str):
    idx = {}
    with open(Path(stage_dir) / "index.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            idx.setdefault((r["routine"], r["phase"], r["hit"]), []).append(r)
    return idx


@lru_cache(maxsize=None)
def _canonical_base(stage_dir: str, block: str) -> int:
    """Address of the block's first member, from any traced routine that declares it by its canonical name."""
    first = CANON[block][0][0]
    for rows in _index(stage_dir).values():
        for r in rows:
            if r["name"] == first:
                return r["addr"]
    raise KeyError(f"no traced routine declares {block}.{first}")


def canonical_from_trace(d: dict, stage_dir: str, routine: str, hit: int, phase: str) -> dict:
    """Add canonical "ops.*", "xnfp.*", "xnf.*" entries to a golden-trace dict using traced addresses."""
    d = dict(d)
    rows = _index(str(stage_dir)).get((routine, phase, hit), [])
    byname = {r["name"]: r for r in rows}
    for block in ("ops", "xnfp", "xnf"):
        keys = [k for k in d if k.startswith(block + ".") and not k.startswith(block + "._")]
        if not keys or all(k in d for k in (f"{block}.{n}" for n, _ in CANON[block])):
            continue
        mem = [(byname[k.split(".", 1)[1]]["addr"], d[k]) for k in keys if k.split(".", 1)[1] in byname]
        if not mem:
            continue
        base = _canonical_base(str(stage_dir), block)
        flat = np.full(block_size(block), np.nan)
        for a, v in sorted(mem, key=lambda t: t[0]):
            o = (a - base) // 8
            v = np.asarray(v, dtype=np.float64).reshape(-1, order="F")
            flat[o:o + v.size] = v[: max(0, min(v.size, flat.size - o))]
        for k in keys:
            d.pop(k)
        d.update(_split(block, flat))
    return d


def golden_state(golden, stage: str, routine: str, hit: int, phase: str) -> tuple[State, dict]:
    raw = golden(stage, routine, hit, phase)
    d = canonical_from_trace(raw, EXT / "traces" / stage, routine, hit, phase)
    return State.from_trace(d), d


# ------------------------------------------------------------------------------------------- oracles

def _intel_dirs():
    env = os.environ.get("INTEL_LIB")
    cands = [env] if env else []
    cands += sorted(glob.glob("/opt/intel/oneapi/compiler/*/lib"))
    cands += sorted(glob.glob(str(Path.home() / ".local/share/containers/storage/overlay/*/diff/opt/intel/oneapi/compiler/*/lib")))
    return [c for c in cands if c and Path(c, "libifport.so.5").exists()]


_PRELOADED = []


def preload_intel():
    if _PRELOADED:
        return True
    dirs = _intel_dirs()
    if not dirs:
        return False
    d = dirs[-1]
    for lib in ("libintlc.so.5", "libimf.so", "libsvml.so", "libirng.so", "libifcoremt.so.5", "libifport.so.5"):
        p = Path(d, lib)
        if p.exists():
            _PRELOADED.append(ctypes.CDLL(str(p), mode=os.RTLD_LAZY | os.RTLD_GLOBAL))
    return True


def oracle_path(variant: str) -> Path:
    return EXT / "oracle" / ("libatlas9.so" if variant == "atlas9" else "libkappa9.so")


_ORACLES: dict = {}


def _cleanup():
    import shutil
    for o in _ORACLES.values():
        if getattr(o, "_tmp", None):
            shutil.rmtree(o._tmp, ignore_errors=True)
            o._tmp = None
    _ORACLES.clear()


import atexit as _atexit
_atexit.register(_cleanup)


def make_oracle(variant: str, fresh: bool = False):
    """Oracle over the variant's library. One instance per variant is reused (each instance holds a private
    copy of the .so in $TMPDIR; fresh=True reloads it to reset all SAVEd state)."""
    from ...testing.fortran_oracle import Oracle
    if variant == "kappa9" and not preload_intel():
        raise RuntimeError("Intel runtime libraries not found (set INTEL_LIB)")
    o = _ORACLES.get(variant)
    if o is None:
        o = _ORACLES[variant] = Oracle(oracle_path(variant))
    elif fresh:
        o.reset()
    return o


def _block_base(o, block: str) -> int:
    lay = o.commons_layout[block]
    return min(v.addr for v in lay.values())


def raw_block(o, block: str) -> np.ndarray:
    n = block_size(block)
    buf = (ctypes.c_double * n).from_address(o.base + _block_base(o, block))
    return np.frombuffer(buf, dtype=np.float64, count=n)


def canonical_from_oracle(o, block: str) -> dict[str, np.ndarray]:
    return _split(block, raw_block(o, block).copy())


def set_canonical(o, block: str, values: dict):
    flat = raw_block(o, block)
    off = 0
    for name, nc in CANON[block]:
        k = f"{block}.{name}"
        if k in values:
            v = values[k]
            v = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            flat[off:off + KW * nc] = np.asarray(v, dtype=np.float64).reshape(-1, order="F")
        off += KW * nc


def load_state(o, s: dict, blocks=None):
    """Write State entries "block.var" into the oracle's COMMONs (canonical blocks via raw layout)."""
    by = {}
    for k, v in s.items():
        if "." not in k or k.startswith("save."):
            continue
        b, var = k.split(".", 1)
        if blocks is not None and b not in blocks:
            continue
        by.setdefault(b, {})[var] = v
    for b, vals in by.items():
        if b in CANON:
            set_canonical(o, b, {f"{b}.{k}": v for k, v in vals.items()})
            continue
        if b not in o.commons_layout:
            continue
        lay = o.commons_layout[b]
        clean = {}
        for k, v in vals.items():
            if k not in lay:
                continue
            v = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            if v.dtype.kind in "SUV":
                continue
            clean[k] = v
        o.set_common(b, **clean)


# ------------------------------------------------------------------------------- shared test drivers

import ctypes as _ct
import platform as _platform

TRACE_STAGE = "sun_atlas9/atlas9"
TRACE_GLIBC = "2.39"            # the golden traces were recorded in the atlas-dev container (Ubuntu 24.04)


def same_libm_as_trace() -> bool:
    """Golden traces are bit-reproducible only with the trace's glibc libm (exp/log/pow differ by ulps
    between glibc releases; an ulp can flip NMOLEC's convergence test)."""
    lib, ver = _platform.libc_ver()
    return lib == "glibc" and ver == TRACE_GLIBC


def golden_tol(smooth: bool = True) -> float:
    if same_libm_as_trace():
        return 0.0
    return 1e-13 if smooth else 1e-6


def merged_golden(golden, routines, hit=0, stage=TRACE_STAGE):
    s = None
    for r in routines:
        b, _ = golden_state(golden, stage, r, hit, "entry")
        s = b if s is None else s.updated(b)
    return s


OPACITY_BASE = ("hotop", "hop", "coolop", "kapp", "hminop", "he1op", "lukeop", "h2collop")


def perturbed_state(base, seed: int, logfreq: float | None = None, tmin=3000.0, tmax=50000.0):
    """Random but physically ranged layers: T in [tmin, tmax] (consistent TK, TKEV, HKT, HCKT, TLOG), XNE and
    RHO scaled by up to e^{+-2}, /FREQ/ for FREQ = 10**logfreq derived as the ATLAS9 main program does;
    ITEMP set to a fresh value so the oracle's ITEMP caches recompute."""
    from .opacity import Freq
    rng = np.random.default_rng(seed)
    s = State(base)
    if rng.random() < 0.4:
        T = torch.from_numpy(np.sort(np.exp(rng.uniform(np.log(tmin), np.log(tmax), KW))))
    else:
        T = torch.clamp(s["temp.t"] * torch.from_numpy(np.exp(rng.uniform(np.log(0.5), np.log(2.5), KW))), tmin, tmax)
    TK = 1.38054e-16 * T
    s.update({"temp.t": T, "temp.tk": TK, "temp.tkev": 8.6171e-5 * T, "temp.hkt": 6.6256e-27 / TK,
              "temp.hckt": 6.6256e-27 / TK * 2.99792458e10, "temp.tlog": torch.log(T),
              "temp.itemp": torch.tensor([1000 + seed % 100000])})
    for k, w in (("state.xne", 2.0), ("state.rho", 1.0), ("state.p", 1.0)):
        if k in s:
            s[k] = s[k] * torch.from_numpy(np.exp(rng.uniform(-w, w, KW)))
    if logfreq is not None:
        n = s.iscalar("rhox.nrhox")
        freq = 10.0 ** logfreq
        fq = Freq.derive(s, KW, freq)
        s.update({"freq.freq": torch.tensor([freq]), "freq.freqlg": torch.tensor([fq.freqlg]),
                  "freq.ehvkt": fq.ehvkt, "freq.stim": fq.stim, "freq.bnu": fq.bnu})
    return s


FUNCTIONS_J = ("c1op", "mg1op", "al1op", "fe1op", "chop", "ohop", "n1op", "o1op", "mg2op", "si2op", "ca2op")


def port_opacity(s, routine: str, variant: str) -> dict:
    """Run one ported opacity routine; returns {name: numpy (n,)}."""
    from . import opacity as O
    n = s.iscalar("rhox.nrhox")
    fq = O.Freq.from_state(s, n)
    if routine == "kapp":
        out = O.kapp(s, fq, variant)
        return {k: out["optot." + k][:n].numpy() for k in ("acont", "scont", "sigmac")}
    r = getattr(O, routine)(s, fq, n, variant)
    if routine in FUNCTIONS_J or routine in ("h2collop", "si1op"):
        r = r.numpy() if isinstance(r, torch.Tensor) else np.full(n, float(r))
        return {routine: r}
    out = {("asi1" if k == "_asi1" else k): v.numpy() for k, v in r.items()}
    if not out:          # early RETURN (H2PLOP, COOLOP above 3.28805D15): /OPS/ untouched
        for k in EARLY_RETURN.get(routine, ()):
            out[k] = (s["bsi.asi1"] if k == "asi1" else s["ops." + k])[:n].numpy()
    return out


EARLY_RETURN = {"h2plop": ("ah2p",), "coolop": ("acool", "asi1")}


_ITEMP_SEQ = [10_000_000]


def oracle_opacity(o, s, routine: str) -> dict:
    """Run the Fortran routine on State `s` through the oracle; same keys as port_opacity."""
    n = s.iscalar("rhox.nrhox")
    load_state(o, s)
    # The oracle library is reused across tests; Fortran routines cache temperature-dependent work keyed on
    # ITEMP (IF(ITEMP.EQ.ITEMP1) reuse). A fresh ITEMP per evaluation forces the recomputation the port always
    # does, so a previous test's state can never leak in through a stale cache.
    _ITEMP_SEQ[0] += 1
    o.common("temp")["itemp"][...] = _ITEMP_SEQ[0]
    if routine in FUNCTIONS_J:
        return {routine: np.array([o.call(routine, j, restype=_ct.c_double)[0] for j in range(1, n + 1)])}
    if routine == "h2collop":
        a = np.zeros(KW, order="F")
        o.call("h2collop", a)
        return {routine: a[:n].copy()}
    if routine == "si1op":
        o.call("si1op")
        return {routine: o.common("bsi")["asi1"][:n].copy()}
    if routine == "kapp":
        o.call("kapp", 0, _ct.c_int(0), _ct.c_double(0.0))
        ot = o.common("optot")
        return {k: ot[k][:n].copy() for k in ("acont", "scont", "sigmac")}
    o.call(routine)
    ops = canonical_from_oracle(o, "ops")
    out = {k.split(".", 1)[1]: v[:n] for k, v in ops.items()}
    if routine == "coolop":
        out["asi1"] = o.common("bsi")["asi1"][:n].copy()
    return out


def max_rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
    d[(a == b) | (np.isnan(a) & np.isnan(b))] = 0.0
    return float(d.max(initial=0.0))


def compare(got: dict, exp: dict, rtol: float, what: str = ""):
    for k, g in got.items():
        e = exp[k]
        r = max_rel(g, e)
        assert r <= rtol, f"{what}{k}: max rel err {r:.3e} > {rtol:.1e}"


def opacity_state(golden, routine: str, hit: int = 0):
    """Golden entry State of `routine` completed with the COMMONs other routines declare (same hit)."""
    base = merged_golden(golden, OPACITY_BASE, hit)
    s, _ = golden_state(golden, TRACE_STAGE, routine, hit, "entry")
    return base.updated(s)


def golden_opacity_check(golden, routine: str, hits=(0, 1, 2)):
    """Golden trace entry -> port -> trace exit (FUNCTIONs, whose value is not traced: vs the oracle on
    the golden entry State)."""
    tol = golden_tol()
    for hit in hits:
        s = opacity_state(golden, routine, hit)
        got = port_opacity(s, routine, "atlas9")
        if routine in FUNCTIONS_J:
            exp = oracle_opacity(make_oracle("atlas9"), s, routine)
            compare(got, exp, 0.0, f"{routine} hit {hit} (oracle on golden entry): ")
            continue
        _, x = golden_state(golden, TRACE_STAGE, routine, hit, "exit")
        n = s.iscalar("rhox.nrhox")
        if routine == "kapp":
            exp = {k: x["optot." + k][:n] for k in got}
        else:
            exp = {k: (x["bsi.asi1"] if k == "asi1" else x["ops." + k])[:n] for k in got}
        compare(got, exp, tol, f"{routine} hit {hit}: ")


def oracle_opacity_check(golden, routine: str, variant: str, seed: int, logfreq: float, rtol: float):
    base = opacity_state(golden, routine, 0)
    s = perturbed_state(base, seed, logfreq)
    got = port_opacity(s, routine, variant)
    exp = oracle_opacity(make_oracle(variant), s, routine)
    compare(got, exp, rtol, f"{routine}[{variant}] seed={seed} freq=1e{logfreq:.4f}: ")
    return max([max_rel(got[k], exp[k]) for k in got] + [0.0])


KAPPA9_RTOL = 4e-15       # libimf (ifx) vs glibc exp/log/pow: a few ulps


# ----------------------------------------------------------------------------- population drivers

NMOLEC_OUT = ("state.xne", "state.xnatom", "state.rho", "xnmol.xnmol", "xnsave.xnsave", "edens.edens",
              "temp.t", "temp.tk", "temp.tkev")


def nmolec_state(golden, hit: int, variant: str = "atlas9"):
    """NMOLEC golden entry + /DEPART/ (from HOP) + /POTION/ (IONPOTS DATA), which PFSAHA reads."""
    from ._data import potion
    s, _ = golden_state(golden, TRACE_STAGE, "nmolec", hit, "entry")
    d, _ = golden_state(golden, TRACE_STAGE, "hop", 0, "entry")
    s = s.updated({k: v for k, v in d.items() if k.startswith("depart.")})
    s["potion.potion"] = torch.from_numpy(potion(variant).copy())
    return s


def oracle_nmolec(o, s, mode=1) -> dict:
    load_state(o, s)
    o.call("nmolec", mode)
    out = {}
    for k in NMOLEC_OUT:
        b, v = k.split(".")
        out[k] = o.common(b)[v].copy()
    return out


def compare_nmolec(got: dict, exp: dict, n: int, rtol: float, what=""):
    for k, g in got.items():
        if k.startswith("_") or k not in exp:
            continue
        g = g.numpy() if isinstance(g, torch.Tensor) else np.asarray(g)
        e = exp[k]
        a = max_rel(g[:n], e[:n])
        assert a <= rtol, f"{what}{k}: max rel {a:.3e} > {rtol:.1e}"
