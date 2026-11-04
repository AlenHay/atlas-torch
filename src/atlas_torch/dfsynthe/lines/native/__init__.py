"""Native (C) kernel for the TAPE12 line loop, see linesum.c.  Built with gcc (in the atlas-dev container:
``scripts/dx 'python -m atlas_torch.dfsynthe.lines.native'``); the accumulator uses it when the shared library
exists (Config.native), otherwise the torch path."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "linesum.c"
LIB = HERE / "liblinesum.so"
CFLAGS = ["-O3", "-fno-fast-math", "-ffp-contract=off", "-mavx2", "-mno-fma", "-fPIC", "-shared", "-lm"]
# IEEE single operations only (vectorised SSE/AVX arithmetic, division and sqrt are exactly rounded; no FMA, no
# reassociation); MXCSR (FTZ/DAZ) is inherited from the calling thread.


SHA = HERE / "liblinesum.so.sha256"


def _src_sha() -> str:
    import hashlib
    return hashlib.sha256(SRC.read_bytes()).hexdigest()


def build(cc: str = "gcc") -> Path:
    subprocess.run([cc, *CFLAGS, "-o", str(LIB), str(SRC)], check=True)
    SHA.write_text(_src_sha())
    return LIB


@lru_cache(maxsize=1)
def load():
    """The ctypes library (linesum, aa_scan), or None (library missing or stale)."""
    if os.environ.get("ATLAS_TORCH_NO_NATIVE"):
        return None
    if not LIB.exists() or not SHA.exists() or SHA.read_text().strip() != _src_sha():
        return None                                   # missing or built from another linesum.c
    lib = ctypes.CDLL(str(LIB))
    lib.linesum.restype = ctypes.c_int
    lib.aa_scan.restype = ctypes.c_int
    lib.xlinop_normal.restype = ctypes.c_int
    lib.xlinop_normal.argtypes = None
    return lib


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "gcc"))
