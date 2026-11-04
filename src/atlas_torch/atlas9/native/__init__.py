"""Optional native (C) helpers of the ATLAS9 port: ``vlibm.c`` (glibc libm applied to whole arrays), ``nmolec.c``
(NMOLEC's per-layer Newton iteration incl. SOLVIT), ``josh.c`` (JOSH's REAL*4 Gauss-Seidel sweeps), ``ttaup.c`` (TTAUP with ROSSTAB, MAP1 on rows).

Built with gcc in the atlas-dev container (``scripts/dx '.venv/bin/python -m atlas_torch.atlas9.native'``, also
done by ``make native``). The helpers are used when the shared library exists and matches the sources (sha256 compiled into the library);
otherwise -- or with ATLAS_TORCH_NO_NATIVE set -- the port falls back to its pure Python/torch paths, which give
bit-identical results (the C code calls the same scalar glibc entry points, in the same order).
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import subprocess
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRCS = [HERE / "vlibm.c", HERE / "nmolec.c", HERE / "josh.c", HERE / "ttaup.c"]
LIB = HERE / "libatlas9native.so"
# scalar libm calls only: no builtin folding, no fast-math, no (libmvec) vectorisation, no FMA contraction
CFLAGS = ["-O2", "-fno-builtin", "-fno-fast-math", "-fno-tree-vectorize", "-ffp-contract=off", "-fPIC", "-shared"]


def _src_sha() -> str:
    h = hashlib.sha256()
    for p in SRCS:
        h.update(p.read_bytes())
    h.update(" ".join(CFLAGS).encode())
    return h.hexdigest()


def build(cc: str = "gcc") -> Path:
    tmp = LIB.with_suffix(".so.tmp{}".format(os.getpid()))
    subprocess.run([cc, *CFLAGS, '-DATLAS9_NATIVE_SHA="{}"'.format(_src_sha()), "-o", str(tmp), *map(str, SRCS),
                    "-lm"], check=True)
    os.replace(tmp, LIB)
    _load.cache_clear()
    return LIB


ENABLED = True          # tests switch the helpers off to compare with the pure Python/torch paths


def load():
    """The ctypes library, or None (missing, stale, ATLAS_TORCH_NO_NATIVE set, or ENABLED False)."""
    return _load() if ENABLED else None


@contextlib.contextmanager
def disabled():
    """Run the pure Python/torch fallbacks inside the block."""
    global ENABLED
    old, ENABLED = ENABLED, False
    try:
        yield
    finally:
        ENABLED = old


@lru_cache(maxsize=1)
def _load():
    if os.environ.get("ATLAS_TORCH_NO_NATIVE"):
        return None
    if not LIB.exists():
        return None
    lib = ctypes.CDLL(str(LIB))
    lib.atlas9_native_sha.restype = ctypes.c_char_p
    if lib.atlas9_native_sha().decode() != _src_sha():
        return None                                   # built from other sources / flags
    for name in ("v_exp", "v_log", "v_log10", "v_expf", "v_logf", "v_log10f", "v_pow", "v_powf"):
        f = getattr(lib, name)
        f.restype = None
        nptr = 3 if name in ("v_pow", "v_powf") else 2
        f.argtypes = [ctypes.c_void_p] * nptr + [ctypes.c_long]
    P, I, L, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_double
    lib.nmolec_newton.restype = L
    lib.nmolec_newton.argtypes = [I, I, I, P, P, L, P, P, P, P, P, D, I, P, P, L, D, D, P, P, P]
    lib.ttaup_c.restype = I
    lib.ttaup_c.argtypes = [P, P, P, P, P, P, P, P, D, I, I, D, D, P, I, P, I, P, P, D, P]
    lib.map1b_rows.restype = None
    lib.map1b_rows.argtypes = [L, I, I, P, P, P, P, P]
    lib.rosstab_point.restype = D
    lib.rosstab_point.argtypes = [P, I, P, I, P, P, D, D, D, D, P]
    lib.josh_gs.restype = None
    lib.josh_gs.argtypes = [I, L, P, P, P, P, P, P, I, ctypes.c_float]
    lib.nmolec_terms.restype = None
    lib.nmolec_terms.argtypes = [I, I, P, P, P, P, P]
    return lib

_F64 = {"exp": "v_exp", "log": "v_log", "log10": "v_log10", "pow": "v_pow"}
_F32 = {"exp": "v_expf", "log": "v_logf", "log10": "v_log10f", "pow": "v_powf"}


def vmath(name: str, *args):
    """glibc `name` (exp, log, log10 or pow) of numpy arrays (broadcast), elementwise; float32 arrays use the
    *f entry points. Returns a new array (shape of the broadcast), or None when the library is not available."""
    lib = load()
    if lib is None:
        return None
    import numpy as np
    dt = np.float32 if all(getattr(a, "dtype", None) == np.float32 for a in args) else np.float64
    arrs = [np.ascontiguousarray(a, dtype=dt) for a in (np.broadcast_arrays(*args) if len(args) > 1 else args)]
    out = np.empty(arrs[0].shape, dtype=dt)
    if out.size:
        getattr(lib, (_F32 if dt == np.float32 else _F64)[name])(*[a.ctypes.data for a in arrs], out.ctypes.data,
                                                                    out.size)
    return out
