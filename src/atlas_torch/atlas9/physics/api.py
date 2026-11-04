"""Public entry points of the ATLAS9/KAPPA9 physics library (see the package docstring)."""
from __future__ import annotations

import torch

from ...state import State
from ._data import VARIANTS, potion as _potion_data
from .opacity import OPS, Freq, kapp
from .opacity_batch import FreqBatch, continuum as continuum_batch
from .pops import electrons as _electrons, pops as _pops, populations as _populations


def _check(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")


def _with_defaults(s, variant):
    """/POTION/ is DATA-initialised by IONPOTS (variant-specific); add it when the State lacks it."""
    s = s if isinstance(s, State) else State(s)
    if "potion.potion" not in s:
        s = s.updated({"potion.potion": torch.from_numpy(_potion_data(variant).copy())})
    return s


def populations(s, variant: str = "atlas9", include_electrons: bool = True) -> dict:
    """The ATLAS9 main program's population block for one iteration (the caller has advanced ITEMP and,
    with IFPRES=1, integrated P). Returns the COMMON entries written (plus save.pops.itemp1)."""
    _check(variant)
    return _populations(_with_defaults(s, variant), variant, include_electrons)


def electrons(s, variant: str = "atlas9") -> dict:
    """POPS(0.D0,1,XNE): NELECT (IFMOL=0) or NMOLEC (IFMOL=1) when ITEMP changed (IFPRES=1)."""
    _check(variant)
    return _electrons(_with_defaults(s, variant), variant)


def pops(s, code: float, mode: int, variant: str = "atlas9"):
    """SUBROUTINE POPS(CODE,MODE,NUMBER) -> (NUMBER (NRHOX, ncols) or None, updates)."""
    _check(variant)
    return _pops(_with_defaults(s, variant), code, mode, variant)


def continuum(s, freq: float | None = None, variant: str = "atlas9", derive_freq: bool | None = None) -> dict:
    """KAPP(N=0): continuous opacity at one frequency.

    freq=None, or `freq` equal to the State's FREQ: use /FREQ/ of the State (the caller's main program
    computed it). Otherwise /FREQ/ (FREQLG, EHVKT, STIM, BNU) is derived for `freq` exactly as the ATLAS9
    main program does (derive_freq=True/False forces either behaviour).
    Returns ops.* (all 25 /OPS/ arrays), optot.* (ACONT, SCONT, ALINE, SLINE, SIGMAC, SIGMAL), bsi.asi1
    and, when derived, freq.*."""
    _check(variant)
    s = _with_defaults(s, variant)
    n = s.iscalar("rhox.nrhox")
    upd = {}
    if derive_freq is None:
        derive_freq = freq is not None and ("freq.freq" not in s or s.scalar("freq.freq") != float(freq))
    if freq is not None and derive_freq:
        fq = Freq.derive(s, n, float(freq))
        for k, v in (("ehvkt", fq.ehvkt), ("stim", fq.stim), ("bnu", fq.bnu)):
            a = s[f"freq.{k}"].clone() if f"freq.{k}" in s else torch.zeros(99, dtype=torch.float64)
            a[:n] = v
            upd[f"freq.{k}"] = a
        upd["freq.freq"] = torch.tensor([fq.freq], dtype=torch.float64)
        upd["freq.freqlg"] = torch.tensor([fq.freqlg], dtype=torch.float64)
    else:
        fq = Freq.from_state(s, n)
    upd.update(kapp(s, fq, variant))
    return upd


class Atlas9Physics:
    """Implementation of atlas9/physics.py's `Physics` protocol for the iteration machinery.

    batched=True (default): the continuum of all frequencies in one pass (:mod:`.opacity_batch`, bit-identical
    to the per-frequency KAPP); batched=False: the per-frequency reference loop."""

    def __init__(self, variant: str = "atlas9", batched: bool = True):
        _check(variant)
        self.variant = variant
        self.batched = batched

    def electrons(self, s) -> dict:
        return electrons(s, self.variant)

    def populations(self, s) -> dict:
        return populations(s, self.variant, include_electrons=False)

    def continuum(self, s, fq: dict) -> dict[str, torch.Tensor]:
        """fq: {'freq': [F], 'freqlg': [F], 'ehvkt','stim','bnu': [F, n]} -> {ops member: [F, n]}."""
        s = _with_defaults(s, self.variant)
        n = s.iscalar("rhox.nrhox")
        freqs = torch.as_tensor(fq["freq"]).reshape(-1)
        fl = torch.as_tensor(fq["freqlg"]).reshape(-1)
        if self.batched:
            fb = FreqBatch(freqs.tolist(), fl.tolist(), fq["ehvkt"][:, :n], fq["stim"][:, :n], fq["bnu"][:, :n])
            u = continuum_batch(s, fb, self.variant)
            zero = torch.zeros((fb.F, n), dtype=torch.float64, device=fb.dev)
            return {k: u[k] if k in u else zero for k in OPS}
        cols = {k: [] for k in OPS}
        for i in range(freqs.shape[0]):
            f = Freq(float(freqs[i]), float(fl[i]), fq["ehvkt"][i][:n], fq["stim"][i][:n], fq["bnu"][i][:n])
            u = kapp(s, f, self.variant)
            for k in OPS:
                cols[k].append(u["ops." + k][:n])
        return {k: torch.stack(v) for k, v in cols.items()}
