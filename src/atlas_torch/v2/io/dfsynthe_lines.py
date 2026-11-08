"""DFSYNTHE's repacked line lists (BasicATLAS data/dfsynthe_files), format level (no physical decoding).

*df.bin (lowlines, highlines, diatomics, tiolines, h2olines; repack*.for): fixed records of 16-byte entries
INTEGER*4 IWL (1-based index on the DFSYNTHE wavelength grid), INTEGER*2 NELION, IELO, IGFLOG, IGR, IGS, IGW;
sorted by IWL; the tail of the last record is padding (SELECTLINES' DO-loop counts give the real entries).
nltelinesdf.bin (repacknlte.for): 64-byte records WL (R*8, nm), ELO (R*4, cm^-1), GF (R*4), NBLO, NBUP, NELION,
TYPE, NCON, NELIONX (I*4), GAMMAR, GAMMAS, GAMMAW (R*4), NBUFF (I*4), 8 bytes padding."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

LINE_DTYPE = np.dtype([("iwl", "<i4"), ("nelion", "<i2"), ("ielo", "<i2"), ("igflog", "<i2"),
                       ("igr", "<i2"), ("igs", "<i2"), ("igw", "<i2")])
NLTE_DTYPE = np.dtype([("wl", "<f8"), ("elo", "<f4"), ("gf", "<f4"), ("nblo", "<i4"), ("nbup", "<i4"),
                       ("nelion", "<i4"), ("type", "<i4"), ("ncon", "<i4"), ("nelionx", "<i4"),
                       ("gammar", "<f4"), ("gammas", "<f4"), ("gammaw", "<f4"), ("nbuff", "<i4"),
                       ("pad", "<i4", 2)])
# file -> number of real entries (dfsynthe.for SELECTLINES DO-loop counts)
DF_FILES = {"lowlinesdf.bin": 31108567, "highlinesdf.bin": 10057574, "diatomicsdf.bin": 7771848,
            "tiolinesdf.bin": 36979284, "h2olinesdf.bin": 48999843}


def default_dir() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "BasicATLAS" / "data" / "dfsynthe_files"


def open_df_list(path, count: int | None = None) -> np.memmap:
    """Memory map of a *df.bin list (structured LINE_DTYPE); count: real entries (default from DF_FILES)."""
    p = Path(path)
    m = np.memmap(p, dtype=LINE_DTYPE, mode="r")
    n = DF_FILES.get(p.name) if count is None else count
    return m if n is None else m[:n]


def index_range(lines: np.memmap, lo: int, hi: int) -> slice:
    """Entries with lo <= IWL <= hi: binary search on the sorted IWL column, reading single entries of the memory
    map (about 2 log2(n) page touches; np.searchsorted would copy the strided column)."""
    iwl = lines["iwl"]

    def first(value: int, right: bool) -> int:
        a, b = 0, len(iwl)
        while a < b:
            m = (a + b) // 2
            v = int(iwl[m])
            if v < value or (right and v == value):
                a = m + 1
            else:
                b = m
        return a
    return slice(first(lo, False), first(hi, True))


def read_nlte(path=None) -> np.ndarray:
    return np.fromfile(Path(path) if path else default_dir() / "nltelinesdf.bin", dtype=NLTE_DTYPE)


__all__ = ["LINE_DTYPE", "NLTE_DTYPE", "DF_FILES", "default_dir", "open_df_list", "index_range", "read_nlte"]
