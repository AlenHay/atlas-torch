"""Native (C) kernels of the SYNTHE chain, bit-identical to the torch implementations they accelerate:

  spectrv.c   SPECTRV point loop (continuum, Planck, both JOSH calls; IFSURF=1) -> spectrv_prog.spectrv
  synline.c   SYNTHE line loops: XLINOP normal-line Voigt wings (-> lineopac.xlinop), main-line accumulation
  fixedcol.c  Fortran Fw.d / Iw column decoding of the line-list readers (-> lines._fixed fcol/icol)

Built with gcc inside the atlas-dev container (``make native`` or ``python -m atlas_torch.synthe.native``); the
programs use a kernel when its shared library exists and was built from the current source (sha256 stamp), else
the torch path.  ATLAS_TORCH_NO_NATIVE=1 disables all native kernels.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
# IEEE operations only: no fast-math (which would also let gcc call libmvec on its own), no FMA contraction
CFLAGS = ["-O2", "-fno-fast-math", "-ffp-contract=off", "-mno-fma", "-fPIC", "-shared", "-pthread"]
KERNELS = {"spectrv": ["-lmvec", "-lm"], "synline": ["-lm"], "fixedcol": []}


def _paths(name: str):
    return HERE / f"{name}.c", HERE / f"lib{name}.so", HERE / f"lib{name}.so.sha256"


def _sha(src: Path) -> str:
    return hashlib.sha256(src.read_bytes()).hexdigest()


def build(cc: str = "gcc") -> list[Path]:
    out = []
    for name, libs in KERNELS.items():
        src, lib, sha = _paths(name)
        tmp = lib.with_name(f".{lib.name}.{os.getpid()}.tmp")
        subprocess.run([cc, *CFLAGS, "-o", str(tmp), str(src), *libs], check=True)
        os.replace(tmp, lib)                 # new inode: running processes keep their mapping
        sha.write_text(_sha(src))
        out.append(lib)
    return out


@lru_cache(maxsize=None)
def load(name: str):
    """The ctypes library of kernel `name`, or None (disabled, missing, or built from another source)."""
    if os.environ.get("ATLAS_TORCH_NO_NATIVE"):
        return None
    src, lib, sha = _paths(name)
    if not lib.exists() or not sha.exists() or sha.read_text().strip() != _sha(src):
        return None
    try:
        return ctypes.CDLL(str(lib))
    except OSError:
        return None


if __name__ == "__main__":
    for p in build(sys.argv[1] if len(sys.argv) > 1 else "gcc"):
        print(p)
