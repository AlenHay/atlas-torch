"""Initial (DATA) values of SAVEd locals, read from a compiled oracle library (test-only stand-in for
atlas_torch.kurucz_data, which parses the fetched Fortran source and is validated against this)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from elftools.elf.elffile import ELFFile

from .fortran_oracle import read_layout

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
SRC_TO_LIB = {"atlas7v.for": "atlas7v", "atlas9mem.patched.for": "atlas9", "atlas9mem.for": "atlas9",
              "kappa9.for": "kappa9", "xnfdf.for": "xnfdf", "synthe.patched.for": "synthe", "synthe.for": "synthe",
              "dfsynthe.for": "dfsynthe", "spectrv.patched.for": "spectrv", "spectrv.for": "spectrv",
              "xnfpelsyn.for": "xnfpelsyn"}


@lru_cache(maxsize=None)
def _image(lib):
    p = str(EXT / "oracle" / f"lib{lib}.so")
    segs = []
    with open(p, "rb") as fh:
        elf = ELFFile(fh)
        for s in elf.iter_sections():
            a, n = s.header.sh_addr, s.header.sh_size
            if a and n:
                data = b"\0" * n if s.header.sh_type == "SHT_NOBITS" else s.data()
                segs.append((a, n, data))
    return p, segs


def so_table(source: str, unit: str, var: str) -> np.ndarray:
    lib = SRC_TO_LIB[source]
    p, segs = _image(lib)
    _, units, _ = read_layout(p)
    v = units[unit.lower()].statics[var.lower()]
    for a, n, data in segs:
        if a <= v.addr < a + n:
            b = data[v.addr - a: v.addr - a + v.nbytes]
            arr = np.frombuffer(b, dtype=v.dtype).copy()
            return arr.reshape(v.shape, order="F") if v.shape else arr
    raise KeyError(var)
