"""SELECTLINES (dfsynthe.for): per-temperature selection of the lines that can matter, REAL*4 exact.

Streams the repacked line lists through numpy memmaps in chunks (bounded memory) and returns the TAPE12
content (NBUFF, CONGF, NELION, GAMRF, GAMSF, GAMWF per line, file order: low, [hi], [diatomic], [TiO], [H2O])
and the TAPE19 content (selected NLTE lines with GF*BOLTSTIM), i.e. everything the Fortran writes to its
scratch units 12 and 19.  The Fortran's NU pointer (monotone search of the edge table) is a running maximum.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..odf_io import LINE_FILES, NLTE_FILE, NLTE_DTYPE, open_linelist, read_nlte
from .f32 import Flavor, lit
from .tables import IXWLBEG, RATIO, RATIOLG, WLBEG

f4 = np.float32
CONGF0 = f4(lit(".026538") / lit("1.77245"))
C17 = lit("2.99792458E17")
FOURPI = lit("12.5664")


def _f(fl: Flavor, fn: str, *a) -> np.ndarray:
    t = [torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) else x for x in a]
    return getattr(fl, fn)(*t).numpy()


def _libimf(fl: Flavor, fn: str, x: np.ndarray) -> np.ndarray:
    """EXP of the scalar SELECTLINES statements: the stock executable calls libimf expf/exp there (imf.py);
    otherwise (or without the Intel runtime) Flavor.expf/exp."""
    if getattr(fl, "fastmath", False):
        from . import imf
        y = getattr(imf, fn)(x)
        if y is not None:
            return y
    return _f(fl, fn, x)


@dataclass
class Tape12:
    nbuff: np.ndarray    # int32
    congf: np.ndarray    # f4  (TAPE12(2))
    nelion: np.ndarray   # int32
    gamr: np.ndarray     # f4  (TAPE12(4))
    gams: np.ndarray     # f4
    gamw: np.ndarray     # f4

    def __len__(self):
        return len(self.nbuff)


@dataclass
class SelectConsts:
    t: np.float32
    hkt: np.float32
    hckt: np.float32
    tablog: np.ndarray   # [32768]
    tabbolt: np.ndarray  # [32768]
    iwltab: np.ndarray   # [nedge] int64
    ratiolg4: np.float32
    wlbeg04: np.float32


def select_consts(teff: float, wledge: np.ndarray, fl: Flavor) -> SelectConsts:
    i = np.arange(1, 32769)
    expo = ((i - 16384).astype(f4) * lit(".001")).astype(f4)
    t = f4(teff)
    hkt = f4(f4(lit("6.6256E-27") / lit("1.38054E-16")) / t)
    hckt = f4(hkt * lit("2.99792458E10"))
    xbolt = (-((i - 1).astype(f4) * hckt)).astype(f4)
    tablog = tabbolt = None
    if getattr(fl, "fastmath", False):
        # stock: the TABLOG and TABBOLT loops are vectorised with __svml_powf4 / __svml_expf4 (not correctly
        # rounded: 6853 / 6874 of the 32768 entries differ from the correctly rounded values; checked against
        # the tables dumped from dfsynthe.exe).  Without the Intel runtime: the correctly rounded tables.
        from . import svml
        tablog = svml.powf_xy(f4(10.0), expo)
        tabbolt = svml.expf(xbolt)
    if tablog is None:
        tablog = _f(fl, "powf", np.full(32768, f4(10.0), f4), expo)
    if tabbolt is None:
        tabbolt = _f(fl, "expf", xbolt)
    wavetab = np.abs(wledge)
    ixwl = np.trunc(_f(fl, "log", wavetab.astype(np.float64)) / RATIOLG).astype(np.int64) + 1
    iwltab = ixwl - IXWLBEG + 1
    return SelectConsts(t, hkt, hckt, tablog, tabbolt, iwltab, f4(RATIOLG), f4(WLBEG / RATIO))


def _nu_index(iwl: np.ndarray, iwltab: np.ndarray, carry: int) -> np.ndarray:
    """0-based NU after `DO WHILE (IWL.GE.IWLTAB(NU)) NU=NU+1` for each line in sequence."""
    if np.any(np.diff(iwltab) < 0):
        raise ValueError("edge table not monotone")
    idx = np.searchsorted(iwltab, iwl, side="right")
    idx = np.maximum.accumulate(np.maximum(idx, carry))
    if idx.size and idx[-1] >= len(iwltab):
        raise ValueError("line beyond last edge (Fortran would run off IWLTAB)")
    return idx


def iter_select_file(mm, count: int, c: SelectConsts, xnfdopmax: np.ndarray, fl: Flavor, *,
                     skip_zero: bool = False, use_tabbolt: bool = False, chunk: int = 2_000_000):
    """Selected lines of one line list, chunk by chunk (file order): yields Tape12."""
    carry = 0
    for s in range(0, count, chunk):
        a = np.asarray(mm[s:min(count, s + chunk)])
        if skip_zero:
            a = a[a["iwl"] != 0]
        iwl = a["iwl"].astype(np.int64)
        nu = _nu_index(iwl, c.iwltab, carry)
        if nu.size:
            carry = int(nu[-1])
        nel = a["nelion"].astype(np.int64)
        xm = xnfdopmax[nu, nel - 1]
        k = np.nonzero(~(xm < f4(1)))[0]
        congf = (CONGF0 * c.tablog[a["igflog"][k].astype(np.int64) - 1]).astype(f4)
        k2 = ~((congf * xm[k]).astype(f4) < f4(1))
        k, congf = k[k2], congf[k2]
        if use_tabbolt:
            congf = (congf * c.tabbolt[a["ielo"][k].astype(np.int64)]).astype(f4)
        else:
            elo = c.tablog[a["ielo"][k].astype(np.int64) - 1]
            congf = (congf * _libimf(fl, "expf", (-(elo * c.hckt)).astype(f4))).astype(f4)
        k3 = ~((congf * xm[k]).astype(f4) < f4(1))
        k, congf = k[k3], congf[k3]
        iw = iwl[k]
        wlvac4 = (_libimf(fl, "expf", (iw.astype(f4) * c.ratiolg4).astype(f4)) * c.wlbeg04).astype(f4)
        freq4 = (C17 / wlvac4).astype(f4)
        stim = (f4(1) - _libimf(fl, "expf", (-(freq4 * c.hkt)).astype(f4))).astype(f4)
        frq4pi = (freq4 * FOURPI).astype(f4)
        if getattr(fl, "fastmath", False):
            # stock (all five list loops): TAPE12(2)=((1.-EXP(..))*CONGF)/FREQ4 (subss, mulss, divss) and
            # TABLOG(..)/FRQ4PI as TABLOG(..)*(1./FRQ4PI) (one divss, mulps/mulss)
            t2 = ((stim * congf).astype(f4) / freq4).astype(f4)
            rf = (f4(1) / frq4pi).astype(f4)
            g = [(c.tablog[a[n][k].astype(np.int64) - 1] * rf).astype(f4) for n in ("igr", "igs", "igw")]
        else:
            t2 = ((congf / freq4).astype(f4) * stim).astype(f4)
            g = [(c.tablog[a[n][k].astype(np.int64) - 1] / frq4pi).astype(f4) for n in ("igr", "igs", "igw")]
        yield Tape12(iw.astype(np.int32), t2, nel[k].astype(np.int32), *g)


def _concat(parts):
    if not parts:
        z = np.zeros(0, np.int32); zf = np.zeros(0, f4)
        return Tape12(z, zf, z, zf, zf, zf)
    return Tape12(*(np.concatenate([getattr(o, f) for o in parts]) for f in
                    ("nbuff", "congf", "nelion", "gamr", "gams", "gamw")))


def select_file(mm, count: int, c: SelectConsts, xnfdopmax: np.ndarray, fl: Flavor, *,
                skip_zero: bool = False, use_tabbolt: bool = False, chunk: int = 2_000_000) -> Tape12:
    return _concat(list(iter_select_file(mm, count, c, xnfdopmax, fl, skip_zero=skip_zero,
                                         use_tabbolt=use_tabbolt, chunk=chunk)))


def select_nlte(nlte: np.ndarray, c: SelectConsts, xnfdopmax: np.ndarray, fl: Flavor) -> np.ndarray:
    nb = nlte["nbuff"].astype(np.int64)
    nu = _nu_index(nb, c.iwltab, 0)
    nel = nlte["nelion"].astype(np.int64)
    xm = xnfdopmax[nu, nel - 1]
    k = np.nonzero(~(xm < f4(1)))[0]
    r = nlte[k]
    e1 = _libimf(fl, "expf", (-(r["elo"] * c.hckt)).astype(f4)).astype(np.float64)
    if getattr(fl, "fastmath", False):      # stock: EXP((HCKT*(-1.E7))/WL) (mulsd, divsd)
        e2 = _libimf(fl, "exp", (np.float64(c.hckt) * -1.0e7) / r["wl"])
    else:
        e2 = _f(fl, "exp", (-((1.0e7 / r["wl"]) * np.float64(c.hckt))))
    boltstim = (e1 * (1.0 - e2)).astype(f4)
    k2 = ~((boltstim * xm[k]).astype(f4) < f4(1))
    r = r[k2].copy()
    r["gf"] = (r["gf"] * boltstim[k2]).astype(f4)
    return r


def _gates(c: SelectConsts) -> dict:
    return {"low": True, "hi": not (c.t < f4(20000.0)), "diat": not (c.t >= f4(10000.0)),
            "tio": not (c.t >= f4(5000.0)), "h2o": not (c.t >= f4(5000.0))}


def iter_tape12(data_dir: str | Path, teff: float, wledge: np.ndarray, xnfdopmax: np.ndarray,
                fl: Flavor | None = None, chunk: int = 2_000_000, counts: dict | None = None):
    """TAPE12 content chunk by chunk in file order (low, [hi], [diatomic], [TiO], [H2O]) without storing it:
    yields (file key, Tape12).  counts (optional dict) receives the number of selected lines per file."""
    fl = fl or Flavor()
    data_dir = Path(data_dir)
    c = select_consts(teff, wledge, fl)
    gates = _gates(c)
    for key in ("low", "hi", "diat", "tio", "h2o"):
        if not gates[key]:
            continue
        _, name, count = LINE_FILES[key]
        p = data_dir / name
        if not p.exists():
            continue
        n = 0
        for t in iter_select_file(open_linelist(p), count, c, xnfdopmax, fl, skip_zero=(key == "hi"),
                                  use_tabbolt=(key == "h2o"), chunk=chunk):
            n += len(t)
            yield key, t
        if counts is not None:
            counts[key] = n


def select_tape19(data_dir: str | Path, teff: float, wledge: np.ndarray, xnfdopmax: np.ndarray,
                  fl: Flavor | None = None) -> np.ndarray:
    fl = fl or Flavor()
    c = select_consts(teff, wledge, fl)
    p = Path(data_dir) / NLTE_FILE
    return select_nlte(read_nlte(p), c, xnfdopmax, fl) if p.exists() else np.zeros(0, NLTE_DTYPE)


def selectlines(data_dir: str | Path, teff: float, wledge: np.ndarray, xnfdopmax: np.ndarray,
                fl: Flavor | None = None, chunk: int = 2_000_000) -> tuple[Tape12, np.ndarray, dict]:
    """-> (tape12, tape19 records, counts {'low','hi','diat','tio','h2o','nlte'})."""
    fl = fl or Flavor()
    counts: dict = {}
    parts = [t for _, t in iter_tape12(data_dir, teff, wledge, xnfdopmax, fl, chunk, counts)]
    tape12 = _concat(parts)
    del parts
    tape19 = select_tape19(data_dir, teff, wledge, xnfdopmax, fl)
    counts["nlte"] = len(tape19)
    return tape12, tape19, counts
