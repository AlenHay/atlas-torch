"""Native (C) kernel for the TAPE12 line loop, see linesum.c.  Built with gcc (in the atlas-dev container:
``scripts/dx 'python -m atlas_torch.dfsynthe.lines.native'``); the accumulator uses it when the shared library
exists (Config.native), otherwise the torch path."""
from __future__ import annotations

import ctypes
import hashlib
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


# further kernels (same flags, own library + sha256 stamp): name -> source file
EXTRA = {"odfcalc": HERE / "odfcalc.c",      # DFCALC rank sums + turbulent broadening (dfsynthe.py)
         "hline": HERE / "hline.c"}          # XLINOP hydrogen lines (lines/xlinop.py XLinop._h_native)


def _build_one(cc: str, src: Path, lib: Path, sha: Path) -> Path:
    tmp = lib.with_name(f".{lib.name}.{os.getpid()}.tmp")
    subprocess.run([cc, *CFLAGS, "-o", str(tmp), str(src), "-lm"], check=True)
    os.replace(tmp, lib)                      # new inode: running workers keep their mapping
    sha.write_text(hashlib.sha256(src.read_bytes()).hexdigest())
    return lib


def build(cc: str = "gcc") -> Path:
    _build_one(cc, SRC, LIB, SHA)
    for name, src in EXTRA.items():
        _build_one(cc, src, HERE / f"lib{name}.so", HERE / f"lib{name}.so.sha256")
    return LIB


@lru_cache(maxsize=None)
def load_extra(name: str):
    """ctypes library of an EXTRA kernel, or None (disabled, missing or stale)."""
    if os.environ.get("ATLAS_TORCH_NO_NATIVE"):
        return None
    src, lib, sha = EXTRA[name], HERE / f"lib{name}.so", HERE / f"lib{name}.so.sha256"
    if not lib.exists() or not sha.exists() or sha.read_text().strip() != hashlib.sha256(src.read_bytes()).hexdigest():
        return None
    try:
        return ctypes.CDLL(str(lib))
    except OSError:
        return None


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
