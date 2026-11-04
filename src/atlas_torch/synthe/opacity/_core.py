"""Shared machinery for the SYNTHE-variant (atlas7v.for) continuous opacities.

Batching model
--------------
Every opacity kernel works on a `Ctx`: COMMON /FREQ/ for a batch of NF frequencies plus a layer view of
the State. Frequency scalars (FREQ, FREQLG, WAVENO) have shape [NF, 1]; per-layer frequency arrays
(EHVKT, STIM, BNU) [NF, n]; layer-only State arrays are read as [1, n] (n = NRHOX). Results are [NF, n].
A single Fortran call is the NF = 1 case (`Ctx.from_state`); xnfpelsyn's frequency loop is
`Ctx.from_freqs` (its loop-120 setup of /FREQ/ reproduced operation for operation).

Branches on frequency (`IF(WAVENO.LT.x)GO TO 30` chains) become cumulative per-frequency masks applied
with torch.where, so terms the Fortran skips never touch the accumulators (H + 0 would not be a no-op
for -0/NaN terms). Loops over a few dozen levels/lines stay Python loops; layers and frequencies are
tensor dimensions.

Numerics
--------
* REAL*4 literals in REAL*8 expressions: `f32()` (as in atlas_torch.fnum).
* Integer powers `x**n` with n >= 3 (and variable n) are compiled by gfortran -O0 to libgcc
  `__powidf2` (binary exponentiation; checked in the oracle disassembly); `**2` is inlined as x*x.
  `powi()` reproduces __powidf2 exactly.
* libm: with EXACT (default) CPU float64 exp/log/log10/pow go element-wise through Python's math
  module (= glibc, the library the Fortran calls); sqrt through numpy (correctly rounded like sqrtsd;
  torch's CPU sqrt is not). Off-CPU, or with EXACT = False, native torch kernels are used (ulp-level
  differences only). The REAL*4 log10f of XKARSAS always goes through glibc's log10f.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np
import torch

from ...fnum import f32  # noqa: F401  (re-exported for the routine modules)
from ...state import State
from ...tables import table as _table

SRC = "atlas7v.for"
KW = 99
EXACT = True

# ----------------------------------------------------------------------------------------------- libm
_libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
_libm.log10f.restype = ctypes.c_float
_libm.log10f.argtypes = [ctypes.c_float]
def _ieee(fn, name):
    """math.<fn> with C/IEEE results instead of Python exceptions (overflow -> inf, log(0) -> -inf, ...)."""
    if name == "exp":
        def f(v):
            try:
                return fn(v)
            except OverflowError:
                return math.inf
    else:
        def f(v):
            try:
                return fn(v)
            except ValueError:
                return -math.inf if v == 0 else math.nan
    return f


def _pow(x, y):
    try:
        return math.pow(x, y)
    except OverflowError:
        return math.inf if (x > 0 or float(y).is_integer() and int(y) % 2 == 0) else -math.inf
    except ValueError:
        return math.inf if x == 0 else math.nan


_U = {k: np.frompyfunc(_ieee(getattr(math, k), k), 1, 1) for k in ("exp", "log", "log10")}
_POW = np.frompyfunc(_pow, 2, 1)


def _exact(x: torch.Tensor) -> bool:
    return EXACT and x.device.type == "cpu" and x.dtype == torch.float64


def _ufunc(name: str, x: torch.Tensor) -> torch.Tensor:
    if _exact(x):
        a = x.detach().numpy()
        with np.errstate(all="ignore"):
            return torch.from_numpy(np.asarray(_U[name](a), dtype=np.float64).reshape(a.shape))
    return getattr(torch, name)(x)


def exp(x: torch.Tensor) -> torch.Tensor:
    return _ufunc("exp", x)


def log(x: torch.Tensor) -> torch.Tensor:
    return _ufunc("log", x)


def log10(x: torch.Tensor) -> torch.Tensor:
    return _ufunc("log10", x)


def sqrt(x: torch.Tensor) -> torch.Tensor:
    if _exact(x):
        return torch.from_numpy(np.sqrt(x.detach().numpy()))
    return torch.sqrt(x)


def pow_(x, y) -> torch.Tensor:
    """Fortran real**real (REAL*8 pow)."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64, device=y.device)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y, dtype=torch.float64, device=x.device)
    if _exact(x) and _exact(y):
        a, b = np.broadcast_arrays(x.detach().numpy(), y.detach().numpy())
        with np.errstate(all="ignore"):
            return torch.from_numpy(np.asarray(_POW(a, b), dtype=np.float64).reshape(a.shape))
    return torch.pow(x, y)


def log10f(x: np.ndarray) -> np.ndarray:
    """glibc log10f element-wise on a float32 array (REAL*4 LOG10)."""
    x = np.asarray(x, dtype=np.float32)
    return np.array([_libm.log10f(float(v)) for v in x.ravel()], dtype=np.float32).reshape(x.shape)


def rdiv(a, t):
    """a / t for a Python number a: torch's `number / tensor` is reciprocal(t)*a (not correctly rounded)."""
    return torch.div(a, t) if isinstance(t, torch.Tensor) else a / t


def powi(x, m: int):
    """libgcc __powidf2(x, m): what gfortran -O0 emits for x**m (m integer, |m| >= 3 or variable)."""
    n = abs(int(m))
    y = x if n % 2 else (torch.ones_like(x) if isinstance(x, torch.Tensor) else 1.0)
    n >>= 1
    while n:
        x = x * x
        if n % 2:
            y = y * x
        n >>= 1
    return rdiv(1.0, y) if m < 0 else y


def table(unit: str, var: str) -> np.ndarray:
    return _table(SRC, unit, var)


# ------------------------------------------------------------------------------------------- context
class Ctx:
    """COMMON /FREQ/ for NF frequencies + the State's layer arrays (see module docstring)."""

    def __init__(self, s: State, freq, freqlg, waveno, ehvkt, stim, bnu):
        self.s = s
        self.n = s.iscalar("rhox.nrhox")
        self.freq, self.freqlg, self.waveno = freq, freqlg, waveno     # [NF,1]
        self.ehvkt, self.stim, self.bnu = ehvkt, stim, bnu             # [NF,n]
        self.device = freq.device
        self.dtype = freq.dtype

    @property
    def nf(self) -> int:
        return self.freq.shape[0]

    @classmethod
    def from_state(cls, s: State) -> "Ctx":
        """One Fortran call: /FREQ/ exactly as it is in the State."""
        n = s.iscalar("rhox.nrhox")
        z = torch.zeros(1, 1, dtype=torch.float64, device=s["rhox.nrhox"].device)

        def sc(k):
            return s[k].reshape(-1)[:1].reshape(1, 1).to(torch.float64) if k in s else z

        def lay(k):   # routines without /FREQ/ (e.g. ELECOP) never read these
            return s[k][:n].reshape(1, n) if k in s else z.expand(1, n)
        return cls(s, sc("freq.freq"), sc("freq.freqlg"), sc("freq.waveno"),
                   lay("freq.ehvkt"), lay("freq.stim"), lay("freq.bnu"))

    @classmethod
    def from_freqs(cls, s: State, freqs) -> "Ctx":
        """xnfpelsyn loop 125/120: FREQ=FREQSET(NU), FREQ15=FREQ/1.E15, FREQLG=LOG(FREQ),
        EHVKT=EXP(-FREQ*HKT), STIM=1-EHVKT, BNU=1.47439D-2*FREQ15**3*EHVKT/STIM,
        WAVE=2.99792458D17/FREQ, WAVENO=1.E7/WAVE."""
        n = s.iscalar("rhox.nrhox")
        hkt = s["temp.hkt"]
        if isinstance(freqs, np.ndarray):
            freqs = np.ascontiguousarray(freqs, dtype=np.float64)
        freq = torch.as_tensor(freqs, dtype=torch.float64, device=hkt.device).reshape(-1, 1)
        freq15 = freq / f32(1.0e15)
        freqlg = log(freq)
        ehvkt = exp(-freq * hkt[:n].reshape(1, n))
        stim = 1.0 - ehvkt
        bnu = 1.47439e-02 * powi(freq15, 3) * ehvkt / stim
        wave = rdiv(2.99792458e17, freq)
        waveno = rdiv(1.0e7, wave)
        return cls(s, freq, freqlg, waveno, ehvkt, stim, bnu)

    def with_waveno(self, waveno) -> "Ctx":
        c = Ctx.__new__(Ctx)
        c.__dict__.update(self.__dict__)
        c.waveno = waveno
        return c

    def lay(self, key: str, col: int | None = None) -> torch.Tensor:
        """Layer array (Fortran column `col`, 1-based) as [1, n]."""
        v = self.s[key]
        v = v[: self.n] if col is None else v[: self.n, col - 1]
        return v.reshape(1, self.n).to(torch.float64)

    def zeros(self) -> torch.Tensor:
        return torch.zeros(self.nf, self.n, dtype=torch.float64, device=self.device)

    def full(self, v: float) -> torch.Tensor:
        return torch.full((self.nf, self.n), v, dtype=torch.float64, device=self.device)

    def fscal(self, v) -> torch.Tensor:
        """frequency-batch scalar [NF,1] filled with v."""
        return torch.full((self.nf, 1), float(v), dtype=torch.float64, device=self.device)

    def itemp(self) -> int:
        return self.s.iscalar("temp.itemp")


def chain(waveno: torch.Tensor, thresholds) -> list[torch.Tensor]:
    """Masks of a `IF(WAVENO.LT.t_k)GO TO 30` chain: term k runs iff WAVENO >= t_i for all i <= k."""
    out, m = [], torch.ones_like(waveno, dtype=torch.bool)
    for t in thresholds:
        m = m & ~(waveno < t)
        out.append(m)
    return out


def acc(mask, total, term):
    """total = total + term where the Fortran executed the statement."""
    return torch.where(mask, total + term, total)


# ------------------------------------------------------------------------------- SAVEd-local helpers
def save_get(s: State, unit: str, var: str, shape=(), dtype=torch.float64, device=None):
    """`save.<unit>.<var>` or the -finit-local-zero initial value (zeros)."""
    k = f"save.{unit}.{var}"
    if k in s:
        return s[k]
    return torch.zeros(shape if shape else (1,), dtype=dtype, device=device)


def save_itemp1(s: State, unit: str) -> int:
    k = f"save.{unit}.itemp1"
    return int(s[k].reshape(-1)[0]) if k in s else 0


def itemp_tensor(v: int, like: torch.Tensor | None = None) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.int64, device=None if like is None else like.device)


def put_rows(full: torch.Tensor, rows: torch.Tensor, n: int, col: int | None = None) -> torch.Tensor:
    """Write a [n] (or [1,n]) result into a clone of a Fortran-shaped COMMON/SAVE array."""
    out = full.clone().to(torch.float64) if full.dtype != torch.float64 else full.clone()
    r = rows.reshape(-1)[:n].to(out.dtype)
    if col is None:
        out[:n] = r
    else:
        out[:n, col - 1] = r
    return out


def scalar1(v, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(v, dtype=like.dtype, device=like.device).reshape(1)
