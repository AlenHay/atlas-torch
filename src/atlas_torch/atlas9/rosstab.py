"""ROSSTAB (atlas9mem.for): Rosseland opacity from the /TABTP/ table (kappa.ros read by READ KAPPA, or the
built-in KTAB DATA otherwise) by MAP1 interpolation in log T (per log P, per VTURB), then in VTURB, then in
log P.

Quirks reproduced:
  * TLOG10=MAX(TLOG10,TABT(1)) is executed *before* TLOG10=DLOG10(T): no clamping in T (MAP1 extrapolates);
    log P is clamped below at TABP(1).
  * EXP10 is a statement function with a default-REAL literal: EXP(X*2.30258509299405E0) uses the single
    precision value of ln 10 widened to double.
  * The (TSAVE,VSAVE) cache of PKAP returns what a recomputation gives, so it is not modelled.
  * IFTABK=0 (no READ KAPPA): TABKAP=DFLOAT(KTAB)/1000. and IFTABK=1 (returned as State updates).
"""
from __future__ import annotations

import numpy as np
import torch

from ..fnum import f32
from ..state import State
from . import fmath
from .numerics import map1_shared

LN10_F32 = f32(2.30258509299405)


def ensure_table(s: State) -> dict:
    """ROSSTAB's first-call conversion of the DATA table when no kappa.ros was read."""
    if s.iscalar("tabtp.iftabk") == 1:
        return {}
    return {"tabtp.tabkap": s["tabtp.ktab"].to(torch.float64) / 1000.0,
            "tabtp.iftabk": torch.ones_like(s["tabtp.iftabk"])}


class Rosstab:
    """Callable ROSSTAB bound to a /TABTP/ table (tabkap[57,42,5], tabt, tabp, tabv)."""

    def __init__(self, s: State):
        upd = ensure_table(s)
        tabkap = upd.get("tabtp.tabkap", s["tabtp.tabkap"])
        nt, np_ = s.iscalar("tabtp.nt"), s.iscalar("tabtp.np")
        self.tabt = s["tabtp.tabt"][:nt].to(torch.float64)
        self.tabp = s["tabtp.tabp"][:np_].to(torch.float64)
        self.tabv = s["tabtp.tabv"].to(torch.float64)
        # [(ip,iv), it] so that MAP1 runs over the last dim
        self.np = np_
        self.tk = tabkap[:nt, :np_, :].permute(1, 2, 0).reshape(np_ * 5, nt).contiguous().to(torch.float64)
        self.updates = upd
        self._native = None

    def native_args(self):
        """(tabt, nt, tabp, np, tabv, tk) as contiguous float64 numpy arrays for native/ttaup.c."""
        if self._native is None:
            c = lambda a: np.ascontiguousarray(a.detach().cpu().numpy(), dtype=np.float64)
            self._native = (c(self.tabt), int(self.tabt.shape[0]), c(self.tabp), int(self.np), c(self.tabv), c(self.tk))
        return self._native

    def __call__(self, t, p, v) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float64)
        p = torch.as_tensor(p, dtype=torch.float64)
        v = torch.as_tensor(v, dtype=torch.float64)
        shp = tuple(np.broadcast_shapes(t.shape, p.shape, v.shape))   # (torch's imports torch.fx on first use)
        t, p, v = (a.expand(shp).reshape(-1) for a in (t, p, v))
        tlog10 = fmath.log10(t)
        # PKAPV(IV,IP) = MAP1(TABT, TABKAP(1,IP,IV), NT, TLOG10)
        pkapv = map1_shared(self.tabt, self.tk, tlog10).reshape(-1, self.np, 5)     # [B, ip, iv]
        pkap = map1_shared(self.tabv, pkapv, v)                                      # PKAP(IP) [B, ip]
        plog = torch.maximum(fmath.log10(p), self.tabp[0])
        ablog = map1_shared(self.tabp, pkap.unsqueeze(-2), plog)[..., 0]
        return fmath.exp(ablog * LN10_F32).reshape(shp)


def read_kappa_ros(path) -> tuple[str, torch.Tensor]:
    """READ KAPPA (READIN label 1150): title card, one skipped card, then 57x42 cards FORMAT(5X,5X,5F7.3).

    Returns (first card as read (80 chars), TABKAP[57,42,5])."""
    with open(path, "r", encoding="latin-1") as fh:
        lines = fh.read().split("\n")
    title = lines[0][:80].ljust(80)
    vals = []
    k = 2
    for _ in range(57 * 42):
        ln = lines[k].rstrip("\r")
        k += 1
        row = []
        for i in range(5):
            row.append(_read_f(ln[10 + 7 * i: 17 + 7 * i], 3))
        vals.append(row)
    tab = torch.tensor(vals, dtype=torch.float64).reshape(57, 42, 5)
    return title, tab


def _read_f(field: str, d: int) -> float:
    """Fortran Fw.d input (BN: blanks ignored): with no decimal point the value is scaled by 10**-d."""
    s = field.replace(" ", "")
    if not s:
        return 0.0
    su = s.upper().replace("D", "E")
    if "." in su:
        return float(su)
    if "E" in su[1:] or "+" in su[1:] or "-" in su[1:]:
        raise NotImplementedError(f"F-input with exponent and no point: {field!r}")
    return float(su[:-d or None] + "." + su[-d:]) if len(su.lstrip("+-")) > d else float(int(su)) / 10**d
