"""Flush-to-zero / denormals-are-zero (FTZ/DAZ) floating-point environment of the stock DFSYNTHE build.

The stock BasicATLAS executables (ifx, default `-fp-model fast`) start with MXCSR.FTZ and MXCSR.DAZ set:
every SSE/AVX REAL*4 and REAL*8 result that would be subnormal is replaced by (signed) zero and every
subnormal operand is read as zero, in user code and in the math library alike.  The -O0 / precise builds
(and the oracle) run with IEEE gradual underflow.  This is an *environment* (like the rounding mode), so the
port emulates it the same way: by running the computation under the x86 MXCSR FTZ+DAZ bits.

`torch.set_flush_denormal(True)` sets those bits **on the calling thread only**: torch's OpenMP worker threads
(created with the environment of the moment) keep gradual underflow, so a multi-threaded elementwise op is
only partly flushed (measured: 3/4 of the elements unflushed with 4 threads).  numpy, Python's `math` and
ctypes libm calls on the calling thread are covered.  Hence `scope(True)`:
  * sets FTZ+DAZ on the calling thread and switches torch to one intra-op thread for the duration,
  * verifies by probe that torch float32/float64 ops, float64->float32 conversion, numpy and libm flush,
  * restores the previous MXCSR state and thread count on exit (no leak into other code).
Only the x86-64 CPU device is supported; for other devices (GPU float32 path) the environment is not
emulated (documented deviation).  Nested scopes are allowed.
"""
from __future__ import annotations

import math
import threading
from contextlib import contextmanager

import numpy as np
import torch

_TINY64 = 2.2250738585072014e-308       # smallest normal double
_local = threading.local()


def probe() -> dict:
    """Which parts of the calling thread's arithmetic currently flush subnormals (True = flushes)."""
    t64 = torch.full((64,), _TINY64, dtype=torch.float64)
    t32 = torch.full((64,), 1.0e-37, dtype=torch.float32)
    return dict(
        torch64=bool(((t64 / 2.0) == 0).all()),
        torch32=bool(((t32 * 0.01) == 0).all()),
        cvt=bool((torch.full((64,), 1.0e-40, dtype=torch.float64).to(torch.float32) == 0).all()),
        daz=bool((torch.tensor([5e-324], dtype=torch.float64) * 1e300 == 0).all()),
        numpy=bool(np.all(np.full(8, _TINY64) / 2 == 0) and np.all(np.full(8, 1e-40).astype(np.float32) == 0)),
        libm=math.pow(10.0, -310.0) == 0.0,
    )


def active() -> bool:
    """True if the calling thread currently runs with FTZ+DAZ (probe of float64 arithmetic)."""
    return bool(np.full(1, _TINY64)[0] / 2 == 0) and float(np.float64(5e-324) * 1e300) == 0.0


@contextmanager
def scope(on: bool = True):
    """Run the body in the stock FTZ/DAZ environment (on=True) or in IEEE gradual underflow (on=False)."""
    prev = active()
    nthr = torch.get_num_threads()
    if on == prev:
        if on and nthr != 1:
            torch.set_num_threads(1)
        try:
            yield
        finally:
            if on and nthr != 1:
                torch.set_num_threads(nthr)
        return
    if on:
        if not torch.set_flush_denormal(True):
            raise RuntimeError("FTZ/DAZ not supported on this CPU (torch.set_flush_denormal returned False)")
        torch.set_num_threads(1)
        p = probe()
        if not all(p.values()):
            torch.set_flush_denormal(False)
            torch.set_num_threads(nthr)
            raise RuntimeError(f"FTZ/DAZ environment incomplete: {p}")
    else:
        torch.set_flush_denormal(False)
    try:
        yield
    finally:
        torch.set_flush_denormal(prev)
        torch.set_num_threads(nthr)


def flush_f32(x: torch.Tensor) -> torch.Tensor:
    """Explicit FTZ of a float32 tensor (for devices where the MXCSR environment does not apply)."""
    tiny = torch.finfo(torch.float32).tiny
    return torch.where(x.abs() < tiny, torch.zeros_like(x), x)
