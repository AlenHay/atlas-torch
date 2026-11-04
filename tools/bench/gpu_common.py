"""Shared helpers of the M7 GPU / float32 / torch.compile study (tools/bench/gpu_*.py).

Nothing here changes the port's defaults: every switch is a context manager that patches module-level
switches of the installed package for the duration of a block and restores them afterwards. Run each mode in
its own process when results must not leak (the scripts do).

Modes (MODES):
  exact   the default code path: glibc/libmvec emulation, CPU, float64 (the reference for speed)
  native  CPU float64, native torch math (no glibc/libmvec emulation; `native_math()`)
  gpu64   GPU float64, native torch math (glibc emulation is CPU-only by design)
  gpu32   GPU, every REAL*8 (float64) torch op computed in float32 (`F32Emu`), REAL*4 parts unchanged
  cpu32   CPU, same float32 emulation
  cpu32p, gpu32p  diagnostic: float32 *precision* (24-bit significand) with float64 *range* (F32Emu(range64=True)),
          separates underflow/overflow failures from rounding failures

Interpreters: CPU modes run with the main env (.venv, torch 2.15 CPU); GPU modes with $ATLAS_EXT/venv-rocm
(torch 2.9.1+rocm6.4, HSA_OVERRIDE_GFX_VERSION=11.0.0). Both inside the atlas-dev container (scripts/dx).
`tools/bench/gpu_run.sh <mode> script.py args` picks the interpreter.
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

REPO = Path(__file__).resolve().parents[2]
EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
SCRATCH = EXT / "scratch" / "m7_gpu"
RESULTS = REPO / "results" / "m7"
PLOTS = REPO / "results" / "plots"
MODES = ("exact", "native", "gpu64", "gpu32", "cpu32", "cpu32p", "gpu32p")

for _p in (str(REPO / "src"), str(REPO / "tools" / "score")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def mode_device(mode: str) -> torch.device:
    return torch.device("cuda") if mode.startswith("gpu") else torch.device("cpu")


def sync(dev=None):
    if torch.cuda.is_available() and (dev is None or torch.device(dev).type == "cuda"):
        torch.cuda.synchronize()


def env_info() -> dict:
    la = os.getloadavg()
    d = {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(),
         "loadavg": [round(x, 2) for x in la], "host": platform.node(),
         "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        d["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
        d["hip"] = getattr(torch.version, "hip", None)
    try:
        d["affinity"] = sorted(os.sched_getaffinity(0))
    except AttributeError:
        pass
    return d


class Timer:
    """with Timer(dev) as t: ... ; t.s  (synchronises the GPU around the block)."""

    def __init__(self, dev=None):
        self.dev = dev

    def __enter__(self):
        sync(self.dev)
        self.load0 = os.getloadavg()[0]
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        sync(self.dev)
        self.s = time.perf_counter() - self.t0
        self.load = round((self.load0 + os.getloadavg()[0]) / 2, 2)


def bench(fn, dev=None, repeat=5, warmup=1, min_time=0.0):
    """Median wall time of fn() over `repeat` runs after `warmup` runs (GPU synchronised)."""
    for _ in range(warmup):
        fn()
    ts = []
    t_all = time.perf_counter()
    while len(ts) < repeat or (time.perf_counter() - t_all) < min_time:
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
        if len(ts) > 200:
            break
    ts.sort()
    return {"median_s": ts[len(ts) // 2], "min_s": ts[0], "n": len(ts), "loadavg": round(os.getloadavg()[0], 2)}


# ----------------------------------------------------------------------------------------- native math
@contextlib.contextmanager
def native_math():
    """Native torch elementary functions on CPU too (no glibc/libmvec emulation).

    The switches the package already has (all default to the exact path; restored on exit):
      atlas9.physics.libm.EXACT, synthe.opacity._core.EXACT   (module flags)
      atlas9.fmath._cpu -> False                              (that module dispatches on the device only)
      synthe.josh.exp_glibc / sqrt_rn, synthe.spectrv_prog's import of exp_glibc
      synthe.lineopac.libm32.native() -> None                 (libmvec/glibc trampolines -> torch fallbacks,
                                                                 the module's documented non-x86 path)
      dfsynthe.xnf.fmath.fsqrt, dfsynthe xnfdf pow10           (device-dispatched numpy paths)
    SPECTRV's libm is an argument (libm="torch"); KAPPA9 already runs its physics with native torch math.
    """
    from atlas_torch.atlas9 import fmath
    from atlas_torch.atlas9.physics import libm
    from atlas_torch.synthe import josh
    from atlas_torch.synthe.lineopac import libm32
    from atlas_torch.synthe.opacity import _core
    saved = []

    def patch(mod, name, val):
        saved.append((mod, name, getattr(mod, name)))
        setattr(mod, name, val)

    patch(libm, "EXACT", False)
    patch(_core, "EXACT", False)
    patch(fmath, "_cpu", lambda x: False)
    patch(josh, "exp_glibc", lambda t: torch.exp(t))
    patch(josh, "sqrt_rn", lambda t: torch.sqrt(t))
    patch(libm32, "native", lambda: None)
    try:
        from atlas_torch.synthe import spectrv_prog
        patch(spectrv_prog, "exp_glibc", lambda t: torch.exp(t))
    except ImportError:
        pass
    try:
        from atlas_torch.dfsynthe.xnf import fmath as xf
        for m in _modules_importing("atlas_torch.dfsynthe", "fsqrt", xf.fsqrt):
            patch(m, "fsqrt", lambda x: torch.sqrt(x) if isinstance(x, torch.Tensor) else xf.math.sqrt(x))
    except ImportError:
        pass
    try:
        yield
    finally:
        for mod, name, val in reversed(saved):
            setattr(mod, name, val)


def _modules_importing(prefix, name, obj):
    import importlib
    import pkgutil
    pkg = importlib.import_module(prefix)
    out = []
    for info in pkgutil.walk_packages(pkg.__path__, prefix + "."):
        if "testing" in info.name or "oracle" in info.name:
            continue
        try:
            m = importlib.import_module(info.name)
        except Exception:  # noqa: BLE001
            continue
        if getattr(m, name, None) is obj:
            out.append(m)
    return out


# ------------------------------------------------------------------------------------- float32 emulation
_F64, _F32 = torch.float64, torch.float32

# ops computed natively in float32 (float64 tensor args cast down, results cast back): accumulations (their
# float32 rounding error grows with the length) and transcendentals (so the device's float32 libm is used)
_NATIVE32 = {
    "sum", "cumsum", "cumprod", "prod", "mean", "matmul", "mm", "bmm", "mv", "dot", "einsum", "tensordot",
    "logsumexp", "trapezoid", "trapz", "cumulative_trapezoid", "std", "var", "norm", "linalg_vector_norm",
    "nansum", "addmm", "baddbmm", "linalg_solve", "solve", "inner", "outer", "cross",
    "exp", "exp2", "expm1", "log", "log10", "log2", "log1p", "pow", "float_power", "sqrt", "rsqrt", "sin", "cos",
    "tan", "atan", "atan2", "sinh", "cosh", "tanh", "erf", "erfc", "reciprocal", "sigmoid", "__pow__", "__rpow__",
}
_ACCUM_INPLACE = {"index_add_", "scatter_add_", "index_put_", "scatter_reduce_", "index_reduce_", "put_",
                  "scatter_"}


_CREATE = {"from_numpy", "as_tensor", "tensor", "asarray", "frombuffer"}


_RANGE64 = [False]
_VELTKAMP = float(2 ** 29 + 1)


def _round(t):
    if _RANGE64[0]:
        # round the float64 significand to 24 bits (Veltkamp split), keeping float64's exponent range
        with torch._C.DisableTorchFunction():
            c = t * _VELTKAMP
            r = c - (c - t)
            return torch.where(torch.isfinite(r), r, t)
    return t.to(_F32).to(_F64)


def _is64(x):
    return isinstance(x, torch.Tensor) and x.dtype == _F64


def _map(obj, fn):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, (list, tuple)):
        r = [_map(o, fn) for o in obj]
        try:
            return type(obj)(r)
        except TypeError:          # namedtuple-like returns (torch.return_types.*)
            return type(obj)(*r) if hasattr(obj, "_fields") else r
    if isinstance(obj, dict):
        return {k: _map(v, fn) for k, v in obj.items()}
    return obj


def _any64(obj):
    if isinstance(obj, torch.Tensor):
        return obj.dtype == _F64
    if isinstance(obj, (list, tuple)):
        return any(_any64(o) for o in obj)
    if isinstance(obj, dict):
        return any(_any64(o) for o in obj.values())
    return False


class F32Emu(TorchFunctionMode):
    """Every float64 torch result is a float32 value (precision and exponent range); storage stays float64.

    * accumulations and transcendentals (_NATIVE32) run in float32 on the tensor's device (float32 summation
      order/rounding and the device's float32 libm), results widened back to float64;
    * other float64 ops run in float64 and their fresh outputs are rounded to float32 (exact float32 results for
      +,-,*,/,sqrt, comparisons, where, ...: double rounding through float64 is innocuous for these);
    * views pass through (the base is already rounded); in-place ops round the mutated tensor; accumulating
      in-place ops (index_add_, scatter_add_, index_put_(accumulate)) run on a float32 copy;
    * `torch.from_numpy`/`as_tensor` results are rounded out of place (the numpy source is not modified).
    REAL*4 code (float32 tensors) is untouched. This is an accuracy model of "the same program with REAL*8 ->
    float32"; its speed is not representative (an extra rounding pass per op, Python dispatch). Use
    `F32Emu.paused()` for regions that must keep float64 (e.g. integer bookkeeping stored in float64).
    """

    _pause = 0

    def __init__(self, exclude=(), range64=False):
        super().__init__()
        self.exclude = set(exclude)
        self.range64 = range64
        self.counts = {"native32": 0, "rounded": 0, "inplace": 0}

    @classmethod
    @contextlib.contextmanager
    def paused(cls):
        cls._pause += 1
        try:
            yield
        finally:
            cls._pause -= 1

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if F32Emu._pause:
            return func(*args, **kwargs)
        name = getattr(func, "__name__", "")
        if name in self.exclude or name in ("__get__", "__set__", "numpy", "item", "tolist", "__repr__",
                                            "__format__", "__bool__", "__index__", "__int__", "__float__",
                                            "__len__", "__iter__", "data_ptr", "size", "dim", "numel"):
            return func(*args, **kwargs)
        if name in _CREATE:
            return _map(func(*args, **kwargs), lambda x: _round(x) if x.dtype == _F64 else x)
        if not (_any64(args) or _any64(kwargs) or kwargs.get("dtype") == _F64):
            return func(*args, **kwargs)
        if name == "__setitem__":
            a = list(args)
            if len(a) >= 3 and _is64(a[2]) and a[0].dtype == _F64:
                a[2] = _round(a[2])
            elif len(a) >= 3 and isinstance(a[2], float) and a[0].dtype == _F64:
                a[2] = float(torch.tensor(a[2], dtype=_F32))
            return func(*a, **kwargs)
        inplace = (name.endswith("_") and not name.endswith("__")) or "out" in kwargs
        if inplace:
            self.counts["inplace"] += 1
            tgt = kwargs.get("out", args[0] if args else None)
            if (name in _ACCUM_INPLACE or (name == "index_put_" and kwargs.get("accumulate"))) and _is64(tgt):
                t32 = tgt.to(_F32)
                a = [t32] + [(_round(x) if _is64(x) else x) for x in args[1:]]
                a = [x.to(_F32) if _is64(x) else x for x in a]
                func(*a, **{k: (v.to(_F32) if _is64(v) else v) for k, v in kwargs.items()})
                tgt.copy_(t32)
                return tgt
            r = func(*args, **kwargs)
            if _is64(tgt):
                tgt.copy_(_round(tgt))
            return r
        if name in _NATIVE32 and kwargs.get("dtype") is None and not self.range64:
            self.counts["native32"] += 1
            a = _map(list(args), lambda x: x.to(_F32) if x.dtype == _F64 else x)
            k = _map(dict(kwargs), lambda x: x.to(_F32) if x.dtype == _F64 else x)
            r = func(*a, **k)
            return _map(r, lambda x: x.to(_F64) if x.dtype == _F32 else x)
        r = func(*args, **kwargs)
        self.counts["rounded"] += 1

        def fix(x):
            if x.dtype != _F64:
                return x
            if x._is_view() and name not in ("from_numpy", "as_tensor", "asarray"):
                return x
            return _round(x)
        return _map(r, fix)


def _rounding_factory(f):
    def g(*a, **k):
        r = f(*a, **k)
        if isinstance(r, torch.Tensor) and r.dtype == _F64 and not F32Emu._pause:
            return _round(r)
        return r
    g.__wrapped__ = f
    return g


@contextlib.contextmanager
def f32_emulation(exclude=(), range64=False):
    """F32Emu plus rounding of the factory functions that bypass __torch_function__ (from_numpy, as_tensor,
    tensor, asarray, full): a float32 program reads its inputs as float32 (copies: numpy aliasing is lost).
    range64=True: float32 precision with float64 exponent range (diagnostic; all ops float64-then-rounded)."""
    m = F32Emu(exclude, range64)
    _RANGE64[0] = range64
    saved = {n: getattr(torch, n) for n in ("from_numpy", "as_tensor", "tensor", "asarray", "full")}
    for n, f in saved.items():
        setattr(torch, n, _rounding_factory(f))
    try:
        with m:
            yield m
    finally:
        for n, f in saved.items():
            setattr(torch, n, f)
        _RANGE64[0] = False


def mode_context(mode: str, coerce: "DeviceCoerce | None" = None):
    """Context of a mode: native math for every non-exact mode, plus float32 emulation for *32 modes.
    coerce: a DeviceCoerce mode entered first (outermost), for GPU runs of code with stray CPU tensors."""
    stack = contextlib.ExitStack()
    if coerce is not None:
        stack.enter_context(coerce)
    if mode != "exact":
        stack.enter_context(native_math())
    if mode.endswith("32"):
        stack.enter_context(f32_emulation())
    elif mode.endswith("32p"):
        stack.enter_context(f32_emulation(range64=True))
    return stack


# ------------------------------------------------------------------------------------------- scoring
def tolerances() -> dict:
    return json.loads((REPO / "results" / "tolerances.json").read_text())["stages"]


def gate(stage: str, metrics: dict) -> dict:
    """{'pass': bool, 'fails': {k: (err, tol)}, 'worst': (k, err/tol)} against the frozen gates of `stage`."""
    tol = tolerances().get(stage, {})
    fails = {k: (metrics.get(k), t["tol"]) for k, t in tol.items() if not (metrics.get(k, float("inf")) <= t["tol"])}
    worst, wr = None, -1.0
    for k, t in tol.items():
        e = metrics.get(k, float("inf"))
        r = (e / t["tol"]) if t["tol"] > 0 else (0.0 if e == 0 else float("inf"))
        if r > wr:
            worst, wr = k, r
    return {"pass": bool(tol) and not fails and "error" not in metrics, "fails": fails,
            "worst": [worst, wr], "n_gated": len(tol)}


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def merge_result(key: str, value, path: Path | None = None) -> None:
    """results/m7/gpu_float32_compile.json[key] = value (read-modify-write; one writer at a time)."""
    path = path or RESULTS / "gpu_float32_compile.json"
    d = json.loads(path.read_text()) if path.exists() else {}
    d[key] = value
    save_json(path, d)


# ------------------------------------------------------------------------------------- device coercion
class DeviceCoerce(TorchFunctionMode):
    """Run code with stray CPU tensors on a GPU: when an op mixes `dev` and CPU tensors (or indexes a CPU tensor
    with device indices), the CPU operands are moved to `dev` and the call site (first frame in src/atlas_torch)
    is counted in `self.sites` -- the device-agnosticism bug list of the study. Ops on CPU tensors only stay on the
    CPU (e.g. code that deliberately works on numpy/CPU). Transfers back to the CPU (.cpu(), .numpy()) are the
    code's own. Diagnostic tool: the transfers cost time, so GPU timings under it are upper bounds."""

    _SKIP = {"to", "cpu", "cuda", "numpy", "tolist", "item", "copy_", "__get__", "__set__", "__repr__",
             "__format__", "detach", "clone", "data_ptr", "size", "dim", "numel", "__bool__", "__len__",
             "__index__", "__int__", "__float__", "__iter__", "_is_view", "is_floating_point"}

    def __init__(self, dev):
        super().__init__()
        self.dev = torch.device(dev)
        self.sites: dict = {}

    def _site(self):
        f = sys._getframe(2)
        src = str(REPO / "src")
        while f is not None:
            if f.f_code.co_filename.startswith(src):
                p = os.path.relpath(f.f_code.co_filename, REPO)
                return f"{p}:{f.f_lineno} ({f.f_code.co_name})"
            f = f.f_back
        return "?"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        if name in ("numpy", "tolist") and args and isinstance(args[0], torch.Tensor) and args[0].device.type != "cpu":
            key = self._site() + f" [{name} of a device tensor]"
            self.sites[key] = self.sites.get(key, 0) + 1
            return func(args[0].cpu(), *args[1:], **kwargs)      # a copy: writes through it do not propagate
        if name in self._SKIP:
            return func(*args, **kwargs)
        flat = []

        def walk(o):
            if isinstance(o, torch.Tensor):
                flat.append(o)
            elif isinstance(o, (list, tuple)):
                for x in o:
                    walk(x)
            elif isinstance(o, dict):
                for x in o.values():
                    walk(x)
        walk(args)
        walk(kwargs)
        on_dev = any(t.device.type == self.dev.type for t in flat)
        on_cpu = any(t.device.type == "cpu" and (t.dim() > 0 or name in ("__getitem__", "__setitem__", "index_put_",
                                                                         "index_select", "gather", "take"))
                     for t in flat)
        if on_dev and on_cpu:
            if name == "__setitem__" and args and args[0].device.type == "cpu":
                pass        # writing device values into a CPU tensor: move the values to the CPU instead
            key = self._site() + f" [{name}]"
            self.sites[key] = self.sites.get(key, 0) + 1
            if name == "__setitem__" and args and args[0].device.type == "cpu":
                a = [args[0]] + [_map(x, lambda t: t.cpu()) for x in args[1:]]
                return func(*a, **kwargs)
            mv = lambda t: t.to(self.dev) if t.device.type == "cpu" else t  # noqa: E731
            return func(*_map(list(args), mv), **_map(dict(kwargs), mv))
        return func(*args, **kwargs)
