"""Fortran semantics shared by the line-list programs (synbeg, rgfalllinesnew, rmolecasc, rh2ofast).

* formatted input: `fread` (Fw.d) / `iread` (Iw) as gfortran reads a formatted sequential file opened without
  BLANK= (BLANK='NULL': blanks are ignored, an all-blank field is 0; an F field without a decimal point has
  d implied decimals);
* REAL*4 arithmetic: numpy float32 (IEEE, correctly rounded +-*/ and sqrt = the SSE scalar instructions of
  the gfortran build); scalar `r4` rounds a Python float to REAL*4 (one double op of two REAL*4 operands
  rounded once is the REAL*4 op);
* libm exactly as the stock gfortran -O3 executables call it (verified with objdump: rgfalllinesnew and
  rmolecasc call scalar powf/log10f/exp/log, rh2ofast's TABLOG loop calls libmvec _ZGVbN4vv_powf):
  `powf`, `vpowf` (glibc through atlas_torch.synthe.lineopac.libm32), `log10f` (glibc 2.39's scalar log10f,
  the libm of the reference runs), and REAL*8 exp/log through CPython's math module (glibc exp/log).
"""
from __future__ import annotations

import ctypes
import functools
import math
import mmap

import numpy as np
import torch

from ..lineopac import libm32

F4 = np.float32


def r4(x: float) -> float:
    """Round a Python float to REAL*4 (returned as a Python float)."""
    return float(F4(x))


def fread(s: str, d: int) -> float:
    """Fortran Fw.d input of the field `s` (BLANK='NULL')."""
    if "." in s:
        try:
            return float(s)
        except ValueError:            # embedded blanks
            pass
    t = s.replace(" ", "")
    if not t:
        return 0.0
    if "." in t:
        return float(t)
    u = t.upper().replace("D", "E")
    m, e = u, ""
    for i in range(1, len(u)):                    # exponent: E/D, or a sign after the mantissa (1.0-05)
        if u[i] == "E" or (u[i] in "+-" and u[i - 1] != "E"):
            m, e = u[:i], u[i:].lstrip("E")
            break
    sign = ""
    if m[:1] in "+-":
        sign, m = m[0], m[1:]
    m = m.rjust(d + 1, "0")
    v = float(f"{sign}{m[:len(m) - d]}.{m[len(m) - d:]}")
    return v * 10.0 ** int(e) if e else v


def iread(s: str) -> int:
    """Fortran Iw input of the field `s` (BLANK='NULL')."""
    t = s.replace(" ", "")
    return int(t) if t else 0


def powf(x, y) -> np.ndarray:
    """Scalar glibc powf, elementwise (REAL*4 ** REAL*4)."""
    x, y = np.broadcast_arrays(np.asarray(x, F4), np.asarray(y, F4))
    if x.size == 0:
        return np.zeros(x.shape, F4)
    return libm32.spowf(torch.from_numpy(np.ascontiguousarray(x)),
                        torch.from_numpy(np.ascontiguousarray(y))).numpy().reshape(x.shape)


def vpowf(x, y) -> np.ndarray:
    """libmvec _ZGVbN4vv_powf over a whole vectorised loop of n = 4k iterations."""
    x, y = np.broadcast_arrays(np.asarray(x, F4), np.asarray(y, F4))
    assert x.size % 4 == 0
    return libm32.vpowf(torch.from_numpy(np.ascontiguousarray(x)),
                        torch.from_numpy(np.ascontiguousarray(y))).numpy().reshape(x.shape)


def log10f(x) -> np.ndarray:
    """glibc 2.39 scalar log10f (ALOG10 of a REAL*4), elementwise."""
    x = np.asarray(x, F4)
    if x.size == 0:
        return np.zeros(x.shape, F4)
    return libm32.slog10f(torch.from_numpy(np.ascontiguousarray(x))).numpy().reshape(x.shape)


# void f(double *a, double *b, double *out, long n, void *fn): out[i] = fn(a[i], b[i])  (scalar REAL*8 libm
# calls in a native loop; the movsd/stride-8 twin of libm32's float trampoline)
_DSCAL = bytes.fromhex(
    "53" "4154" "4155" "4156" "4157"
    "4889fb" "4989f4" "4989d5" "4989ce" "4d89c7"
    "4d85f6" "7424"
    "f20f1003" "f2410f100c24"
    "41ffd7"
    "f2410f114500"
    "4883c308" "4983c408" "4983c508"
    "49ffce" "ebd7"
    "415f" "415e" "415d" "415c" "5b" "c3")


@functools.lru_cache(maxsize=None)
def _dnative():
    """(trampoline, libm) for scalar glibc REAL*8 functions, or None (then Python's math = glibc too)."""
    nat = libm32.native()
    if nat is None:
        return None
    try:
        mem = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        mem.write(_DSCAL)
        base = ctypes.addressof(ctypes.c_char.from_buffer(mem))
        proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long,
                                 ctypes.c_void_p)
        fn = proto(base)
        addr = {k: ctypes.cast(getattr(nat.libm, k), ctypes.c_void_p).value for k in ("exp", "log")}
        x = np.array([0.5, 2.0, 700.0, 1e-300])
        out = np.empty(4)
        fn(x.ctypes.data, x.ctypes.data, out.ctypes.data, 4, addr["exp"])
        if out.tolist() != [math.exp(v) for v in x.tolist()]:
            return None
        return mem, fn, addr
    except Exception:  # pragma: no cover - platform dependent
        return None


def _dcall(name, pyfun, x) -> np.ndarray:
    x = np.ascontiguousarray(x, np.float64)
    nat = _dnative()
    if nat is None or x.size < 64:
        return np.array([pyfun(v) for v in x.ravel().tolist()], np.float64).reshape(x.shape)
    out = np.empty(x.shape, np.float64)
    nat[1](x.ctypes.data, x.ctypes.data, out.ctypes.data, x.size, nat[2][name])
    return out


def dexp(x) -> np.ndarray:
    """glibc exp (REAL*8 EXP/DEXP), elementwise (the same libm function as Python's math.exp)."""
    return _dcall("exp", math.exp, x)


def dlog(x) -> np.ndarray:
    """glibc log (REAL*8 DLOG), elementwise. Arguments <= 0 go through math.log (raises like before)."""
    x = np.ascontiguousarray(x, np.float64)
    if x.size and not (x > 0).all():
        return np.array([math.log(v) for v in x.ravel().tolist()], np.float64).reshape(x.shape)
    return _dcall("log", math.log, x)


def nbuff_of(wlvac: np.ndarray, ratiolg: float, ixwlbeg: int) -> np.ndarray:
    """IXWL=DLOG(WLVAC)/RATIOLG+.5D0 (INTEGER: truncation); NBUFF=IXWL-IXWLBEG+1."""
    ixwl = np.trunc(dlog(wlvac) / ratiolg + 0.5).astype(np.int64)
    return (ixwl - ixwlbeg + 1).astype(np.int32)


def ixwlbeg_of(wlbeg: float, ratiolg: float) -> int:
    """IXWLBEG=DLOG(WLBEG)/RATIOLG; IF(DEXP(IXWLBEG*RATIOLG).LT.WLBEG)IXWLBEG=IXWLBEG+1."""
    ix = int(math.log(wlbeg) / ratiolg)
    if math.exp(ix * ratiolg) < wlbeg:
        ix += 1
    return ix


def pack12(nbuff, congf, nelion, elo, gamrf, gamsf, gamwf, alpha) -> bytes:
    """Payloads of WRITE(12)NBUFF,CONGF,NELION,ELO,GAMRF,GAMSF,GAMWF,ALPHA for arrays of records."""
    dt = np.dtype([("nbuff", "<i4"), ("congf", "<f4"), ("nelion", "<i4"), ("elo", "<f4"), ("gamrf", "<f4"),
                   ("gamsf", "<f4"), ("gamwf", "<f4"), ("alpha", "<f4")])
    n = len(nbuff)
    a = np.empty(n, dt)
    for k, v in zip(dt.names, (nbuff, congf, nelion, elo, gamrf, gamsf, gamwf, alpha)):
        a[k] = v
    return a


def write_records12(path, recs: np.ndarray) -> None:
    """Append fixed 32-byte records (gfortran unformatted sequential, 4-byte markers) to `path`."""
    n = len(recs)
    if n == 0:
        open(path, "ab").close()
        return
    out = np.empty(n, np.dtype([("m0", "<i4"), ("r", recs.dtype), ("m1", "<i4")]))
    out["m0"] = recs.dtype.itemsize
    out["m1"] = recs.dtype.itemsize
    out["r"] = recs
    with open(path, "ab") as fh:
        fh.write(out.tobytes())
