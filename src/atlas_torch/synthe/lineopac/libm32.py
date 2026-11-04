"""Single-precision libm exactly as the stock SYNTHE binary sees it.

gfortran -O3 vectorises several REAL*4 loops of synthe.for: each group of 4 iterations calls glibc's
libmvec SSE4 variant (`_ZGVbN4v_expf`, `_ZGVbN4vv_powf`, `_ZGVbN4v_log10f`), the 0-3 leftover iterations
call the scalar glibc function, and every non-vectorised site calls the scalar function (`expf`, `logf`,
`powf`, `log10f`).  These are not correctly rounded (libmvec powf differs from the correctly rounded
value in ~5% of cases, and uses the CPU-specific `rcpps`), so results depend on the host libm exactly as
the Fortran's do.  We therefore evaluate them with the host's own libm/libmvec:

* `vexpf`   libmvec SSE4 expf, pure float32 torch emulation (range reduction with a 1.5*2^23 shifter,
            degree-5 polynomial, exponent from integer ops; |x| > 87.33654 lanes go to scalar expf as in
            libmvec).  Verified bit-exact against the library (tests/oracle/test_synthe_libm32.py).
* `vpowf`, `vlog10f`  libmvec through a native 4-lane trampoline (`_Native`).
* `sexpf`, `slogf`, `spowf`, `slog10f`  scalar glibc through a native element loop.
* `loopvec`  "vectorised loop" semantics: vector function on the first 4*floor(n/4) iterations of the
            loop axis, scalar on the rest.

When native calls are unavailable (non-x86-64 / no exec mmap) everything falls back to double-precision
torch evaluation rounded once to float32 (agreement then only to ~1 ulp in a few % of values).
All functions take/return float32 tensors on any device (native calls round-trip through the CPU).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
import mmap
import platform
from functools import lru_cache

import numpy as np
import torch

# ------------------------------------------------------------------ libmvec SSE4 expf (emulated)
_E_LOG2E = 0x3FB8AA3B
_E_SHIFT = 0x4B400000
_E_LN2HI = 0x3F317200
_E_LN2LO = 0x35BFBE8E
_E_BIAS = 0x7F
_E_C = (0x3F800000, 0x3F7FFFFE, 0x3EFFFF34, 0x3E2AACAC, 0x3D2B8392, 0x3C07D9FE)  # 1, c1..c5
_E_ABSMASK = 0x7FFFFFFF
_E_THRESH = 0x42AEAC4F  # 87.33654


def _f(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint32).view(np.float32)[0])


def vexpf(x: torch.Tensor) -> torch.Tensor:
    """libmvec SSE4 expf, elementwise, bit-exact (float32 in, float32 out)."""
    x = x.to(torch.float32)
    dev = x.device
    c = lambda b: torch.tensor(_f(b), dtype=torch.float32, device=dev)  # noqa: E731
    shift = c(_E_SHIFT)
    t = x * c(_E_LOG2E)
    t = t + shift
    n = t - shift
    r = x - n * c(_E_LN2HI)
    r = r - n * c(_E_LN2LO)
    scale = ((t.view(torch.int32) + _E_BIAS) << 23).view(torch.float32)
    p = r * c(_E_C[5])
    p = p + c(_E_C[4])
    p = p * r
    p = p + c(_E_C[3])
    p = p * r
    p = p + c(_E_C[2])
    p = p * r
    p = p + c(_E_C[1])
    q = r * p
    q = q + c(_E_C[0])
    res = scale * q
    special = (x.view(torch.int32) & _E_ABSMASK) > _E_THRESH
    if bool(special.any()):
        res = res.clone()
        res[special] = sexpf(x[special])
    return res


# ------------------------------------------------------------------ native calls into the host libm
_VEC4 = bytes.fromhex(  # void f(float *a, float *b, float *out, long n4, void *fn): out[4i:4i+4] = fn(a4, b4)
    "53" "4154" "4155" "4156" "4157"
    "4889fb" "4989f4" "4989d5" "4989ce" "4d89c7"
    "4d85f6" "7421"
    "0f1003" "410f100c24"
    "41ffd7"
    "410f114500"
    "4883c310" "4983c410" "4983c510"
    "49ffce" "ebda"
    "415f" "415e" "415d" "415c" "5b" "c3")
_SCAL = bytes.fromhex(  # void f(float *a, float *b, float *out, long n, void *fn): out[i] = fn(a[i], b[i])
    "53" "4154" "4155" "4156" "4157"
    "4889fb" "4989f4" "4989d5" "4989ce" "4d89c7"
    "4d85f6" "7424"
    "f30f1003" "f3410f100c24"
    "41ffd7"
    "f3410f114500"
    "4883c304" "4983c404" "4983c504"
    "49ffce" "ebd7"
    "415f" "415e" "415d" "415c" "5b" "c3")


class _Native:
    def __init__(self):
        if platform.machine() not in ("x86_64", "AMD64"):
            raise OSError("native libm calls need x86-64")
        self._mem = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        self._mem.write(_VEC4)
        self._mem.seek(512)
        self._mem.write(_SCAL)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self._mem))
        proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long,
                                 ctypes.c_void_p)
        self.vec4 = proto(base)
        self.scal = proto(base + 512)
        self.libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        self.libmvec = ctypes.CDLL(ctypes.util.find_library("mvec") or "libmvec.so.1")
        self._addr = {}

    def addr(self, lib, name):
        key = (lib, name)
        if key not in self._addr:
            self._addr[key] = ctypes.cast(getattr(getattr(self, lib), name), ctypes.c_void_p).value
        return self._addr[key]

    def run(self, kind, name, a, b):
        a = np.ascontiguousarray(a, dtype=np.float32).ravel()
        b = np.ascontiguousarray(b, dtype=np.float32).ravel()
        n = a.size
        if kind == "vec":
            m = (n + 3) // 4 * 4
            if m != n:
                a = np.concatenate([a, np.ones(m - n, np.float32)])
                b = np.concatenate([b, np.ones(m - n, np.float32)])
            out = np.empty(m, np.float32)
            self.vec4(a.ctypes.data, b.ctypes.data, out.ctypes.data, m // 4, self.addr("libmvec", name))
            return out[:n]
        out = np.empty(n, np.float32)
        self.scal(a.ctypes.data, b.ctypes.data, out.ctypes.data, n, self.addr("libm", name))
        return out

    def run2d(self, name, a, b):
        """libmvec REAL*8 2-lane function (`_ZGVbN2v_*`, `_ZGVbN2vv_*`) over float64 arrays: the 16-byte
        trampoline (_VEC4 moves raw xmm registers) on consecutive pairs."""
        a = np.ascontiguousarray(a, dtype=np.float64).ravel()
        b = np.ascontiguousarray(b, dtype=np.float64).ravel()
        n = a.size
        m = (n + 1) // 2 * 2
        if m != n:
            a = np.concatenate([a, np.ones(m - n)])
            b = np.concatenate([b, np.ones(m - n)])
        out = np.empty(m, np.float64)
        if m:
            self.vec4(a.ctypes.data, b.ctypes.data, out.ctypes.data, m // 2, self.addr("libmvec", name))
        return out[:n]


@lru_cache(maxsize=None)
def native():
    """The native caller, or None when unavailable."""
    try:
        return _Native()
    except Exception:  # pragma: no cover - platform dependent
        return None


def _np(t):
    return t.detach().to("cpu", torch.float32).numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)


def _call(kind, name, fallback, x, y=None):
    xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x, np.float32))
    if y is not None:
        yt = y if isinstance(y, torch.Tensor) else torch.as_tensor(np.asarray(y, np.float32), device=xt.device)
        xt, yt = torch.broadcast_tensors(xt.to(torch.float32), yt.to(device=xt.device, dtype=torch.float32))
    else:
        xt = xt.to(torch.float32)
    nat = native()
    if nat is None or xt.numel() == 0:
        return fallback(xt, yt) if y is not None else fallback(xt)
    a = _np(xt)
    b = _np(yt) if y is not None else a
    out = nat.run(kind, name, a, b).reshape(xt.shape)
    return torch.from_numpy(out).to(xt.device)


def _fb_exp(x):
    return torch.exp(x.double()).float()


def _fb_log(x):
    return torch.log(x.double()).float()


def _fb_log10(x):
    return torch.log10(x.double()).float()


def _fb_pow(x, y):
    return torch.pow(x.double(), y.double()).float()


def sexpf(x):
    return _call("scal", "expf", _fb_exp, x)


def slogf(x):
    return _call("scal", "logf", _fb_log, x)


LOG10F = "glibc239"   # or "host": the host libm's log10f (to compare with oracle libraries run on this host)


def slog10f(x):
    """glibc 2.39 scalar log10f (sysdeps/ieee754/flt-32/e_log10f.c as compiled in Ubuntu 24.04, which
    produced the reference runs): k + mantissa split, z = (logf(m)*ivln10 + y*log10_2lo) + y*log10_2hi.
    (glibc >= 2.41 replaced it by a correctly rounded log10f, which differs in ~2% of cases.)"""
    if LOG10F == "host":
        return _call("scal", "log10f", _fb_log10, x)
    x = (x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x, np.float32))).to(torch.float32)
    hx = x.view(torch.int32)
    sub = (hx < 0x00800000) & (hx > 0)
    k = torch.where(sub, torch.full_like(hx, -25), torch.zeros_like(hx))
    xs = torch.where(sub, x * _f(0x4C000000), x)
    hx = xs.view(torch.int32)
    k = k + (hx >> 23) - 127
    i = (k < 0).to(torch.int32)
    y = (k + i).to(torch.float32)
    m = ((hx & 0x007FFFFF) | ((0x7F - i) << 23)).view(torch.float32)
    tmp = y * _f(0x355427DB)
    z = slogf(m) * _f(0x3EDE5BD9)
    z = z + tmp
    z = z + y * _f(0x3E9A2080)
    bad = (x <= 0) | ~torch.isfinite(x)
    if bool(bad.any()):
        z = torch.where(bad, _fb_log10(x), z)
    return z


def spowf(x, y):
    return _call("scal", "powf", _fb_pow, x, y)


def vpowf(x, y):
    return _call("vec", "_ZGVbN4vv_powf", _fb_pow, x, y)


def vlog10f(x):
    return _call("vec", "_ZGVbN4v_log10f", _fb_log10, x)


def vexpf_native(x):
    """libmvec expf through the library itself (tests: reference for `vexpf`)."""
    return _call("vec", "_ZGVbN4v_expf", _fb_exp, x)


def dvexp(x) -> np.ndarray:
    """glibc libmvec _ZGVbN2v_exp (REAL*8 EXP in a gfortran -O3 vectorised loop), elementwise (lanes are
    independent). Falls back to scalar glibc exp when native calls are unavailable."""
    x = np.asarray(x, np.float64)
    nat = native()
    if nat is None:
        return np.vectorize(math.exp, otypes=[np.float64])(x) if x.size else x.copy()
    return nat.run2d("_ZGVbN2v_exp", x, x).reshape(x.shape)


def dvpow(x, y) -> np.ndarray:
    """glibc libmvec _ZGVbN2vv_pow (REAL*8 ** REAL*8 in a vectorised loop), elementwise."""
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    nat = native()
    if nat is None:
        return np.vectorize(math.pow, otypes=[np.float64])(x, y) if x.size else x.copy()
    return nat.run2d("_ZGVbN2vv_pow", x, y).reshape(x.shape)


def dloopvec(vfun, sfun, *args) -> np.ndarray:
    """Value of a gfortran -O3 loop over the last axis vectorised with 2-lane REAL*8 libmvec calls: vfun on
    the first 2*floor(n/2) iterations, sfun (scalar glibc) on the odd last one."""
    a = np.broadcast_arrays(*[np.asarray(v, np.float64) for v in args])
    n = a[0].shape[-1]
    nv = n // 2 * 2
    out = np.empty(a[0].shape, np.float64)
    if nv:
        out[..., :nv] = vfun(*[v[..., :nv] for v in a])
    if nv != n:
        out[..., nv:] = sfun(*[v[..., nv:] for v in a])
    return out


def loopvec(vfun, sfun, *args, dim: int = -1):
    """Value of a gfortran-vectorised loop over `dim`: vfun on the first 4*floor(n/4) iterations, sfun on
    the remainder (args broadcast to a common shape first)."""
    ts = torch.broadcast_tensors(*[a.to(torch.float32) for a in args])
    n = ts[0].shape[dim]
    nv = n // 4 * 4
    if nv == n:
        return vfun(*ts)
    if nv == 0:
        return sfun(*ts)
    head = vfun(*[t.narrow(dim, 0, nv) for t in ts])
    tail = sfun(*[t.narrow(dim, nv, n - nv) for t in ts])
    return torch.cat([head, tail], dim=dim)


# scalar conveniences (python floats, exact float32 semantics)
def sexpf1(x: float) -> float:
    return float(sexpf(torch.tensor([x], dtype=torch.float32))[0])


def slogf1(x: float) -> float:
    return float(slogf(torch.tensor([x], dtype=torch.float32))[0])


def spowf1(x: float, y: float) -> float:
    return float(spowf(torch.tensor([x], dtype=torch.float32), torch.tensor([y], dtype=torch.float32))[0])


@lru_cache(maxsize=None)
def _dpow():
    nat = native()
    if nat is None:
        return None
    f = nat.libm.pow
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double, ctypes.c_double]
    return f


def dpow(x, y):
    """Scalar glibc pow (REAL*8 ** REAL*8), elementwise over a small array."""
    f = _dpow()
    xa, ya = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    if f is None:
        return np.power(xa, ya)
    out = np.array([f(float(a), float(b)) for a, b in zip(xa.ravel(), ya.ravel())], dtype=np.float64)
    return out.reshape(xa.shape)
