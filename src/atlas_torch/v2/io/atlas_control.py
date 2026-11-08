"""ATLAS9 control cards (BasicATLAS atlas_control.com) and restart decks (BasicATLAS restart.dat).

    spec = read_control("atlas_control.com")     # TEFF, log g, VTURB, CONVECTION, SCALE, abundances, iterations
    start = read_restart("restart.dat")          # NaN-tolerant READ PUNCH deck

Card semantics follow atlas9mem.for READIN (labels 100-3000).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..atmosphere import Atmosphere
from ..composition import Composition
from .atlas_deck import _deck6_rows, _f


@dataclass(frozen=True)
class ControlSpec:
    """The model parameters of an ATLAS9 run.

    vturb             microturbulent velocity, cm/s (VTURB card; selects the ODF / kappa.ros vturb)
    mixing_length     l/H_p (CONVECTION ON|OVER l)
    overshoot_weight  CONVECTION OVER l w n: weight of the convective-flux smoothing over 0.5 H_p (CONVEC 774-779)
    nconv             CONVECTION OVER l w n: the convective flux is zeroed in layers 1..n (CONVEC 7779)
    turbulence        (TRBFDG, TRBPOW, TRBSND, TRBCON) of TURBULENCE ON, None for OFF (eq. 4.7)
    tau_grid          SCALE n log10(tau_1) step: the tau_Ross grid
    iterations        total of the ITERATIONS cards
    frequency_set     ('BIG' | 'LITTLE', number of frequencies)
    """
    teff: float
    logg: float
    title: str = ""
    vturb: float = 0.0
    convection: bool = False
    mixing_length: float = 1.0
    overshoot_weight: float = 0.0
    nconv: int = 0
    turbulence: tuple | None = None
    tau_grid: tuple = (72, -6.875, 0.125)
    composition: Composition | None = None
    molecules: bool = False
    iterations: int = 0
    frequency_set: tuple = ("BIG", 337)
    cards: tuple = field(default_factory=tuple)

    @property
    def gravity(self) -> float:
        return 10.0 ** self.logg

    def log_tau(self) -> np.ndarray:
        n, t1, step = self.tau_grid
        return t1 + step * np.arange(int(n))


def _cards(text: str) -> list[str]:
    """Card images: the lines between <<"EOF" and EOF of a shell-wrapped control file, else all lines."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "<<" in ln and "EOF" in ln), None)
    if start is not None:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == "EOF"), len(lines))
        lines = lines[start + 1:end]
    return [ln.rstrip() for ln in lines if ln.strip()]


def _abundance_cards(cards: list[str]) -> Composition | None:
    """ABUNDANCE SCALE / CHANGE cards (READIN labels 1510-1514): H, He number fractions, Z >= 3 log10 before scale."""
    text = " ".join(c for c in cards if c.lstrip().startswith("ABUNDANCE"))
    m = re.search(r"ABUNDANCE SCALE\s+(\S+)", text)
    if m is None:
        return None
    scale = _f(m.group(1))
    change = {int(z): _f(v) for z, v in re.findall(r"(?<![\d.])(\d{1,2})\s+(-?\d+\.\d*|-?nan)", text.split(
        "ABUNDANCE CHANGE", 1)[1], flags=re.I)} if "ABUNDANCE CHANGE" in text else {}
    if any(z not in change for z in range(1, 100)) or not np.isfinite(scale) \
            or any(not math.isfinite(change[z]) for z in range(1, 100)):
        return None
    return Composition.from_atlas(change[1], change[2], [change[z] for z in range(3, 100)], scale)


def read_control(path_or_text: str | Path) -> ControlSpec:
    """ControlSpec of atlas_control.com (or of the card text itself)."""
    s = str(path_or_text)
    text = Path(s).read_text() if "\n" not in s else s
    cards = _cards(text)
    kw: dict = {"cards": tuple(cards)}
    iterations = 0
    for card in cards:
        w = card.split()
        key = w[0].upper()
        if key == "TEFF":
            kw["teff"] = _f(w[1])
            if len(w) > 3 and w[2].upper() == "GRAVITY":
                g = _f(w[3])
                kw["logg"] = g if g < 10.0 else math.log10(g)
        elif key == "GRAVITY":
            g = _f(w[1])
            kw["logg"] = g if g < 10.0 else math.log10(g)
        elif key == "TITLE":
            kw["title"] = card[6:80]
        elif key == "VTURB":
            kw["vturb"] = _f(w[1])
        elif key == "MOLECULES":
            kw["molecules"] = w[1].upper() == "ON"
        elif key == "FREQUENCIES":
            kw["frequency_set"] = (w[4].upper(), int(w[1]))
        elif key == "CONVECTION":
            mode = w[1].upper()
            kw["convection"] = mode in ("ON", "OVER")
            if mode in ("ON", "OVER"):
                kw["mixing_length"] = _f(w[2])
            if mode == "OVER":
                kw["overshoot_weight"], kw["nconv"] = _f(w[3]), int(_f(w[4]))
        elif key == "TURBULENCE":
            kw["turbulence"] = tuple(_f(v) for v in w[2:6]) if w[1].upper() == "ON" else None
        elif key == "SCALE" and len(w) >= 4 and w[1].upper() != "MODEL":
            kw["tau_grid"] = (int(_f(w[1])), _f(w[2]), _f(w[3]))
            if len(w) >= 6:
                kw["teff"] = _f(w[4])
                g = _f(w[5])
                kw["logg"] = g if g < 10.0 else math.log10(g)
        elif key == "ITERATIONS":
            iterations += int(_f(w[1]))
    kw["iterations"] = iterations
    kw["composition"] = _abundance_cards(cards)
    if "teff" not in kw or "logg" not in kw:
        raise ValueError("control cards give no TEFF / GRAVITY (or SCALE n tau1 step teff logg)")
    return ControlSpec(**kw)


@dataclass(frozen=True)
class RestartModel:
    """A punched model used as the starting structure (READ PUNCH, READIN label 1170).

    atmosphere   rhox, T, P, n_e, vturb (NaN where the deck has NAN) and extra ABROSS, ACCRAD, FLXRAD, VCONV,
                 VELSND; teff, logg from the TEFF card (logg NaN in BasicATLAS restarts)
    composition  None when the abundance cards hold NaN
    pradk0       PRADK card: radiation pressure at the surface, dyn cm^-2
    BasicATLAS restart.dat holds tau_Ross in the RHOX column with ABROSS = 1, so tau_ross() returns it.
    """
    atmosphere: Atmosphere
    composition: Composition | None
    pradk0: float
    head: tuple
    iteration: int | None

    def tau_ross(self) -> torch.Tensor:
        """tau_Ross = int ABROSS d(rhox), ABROSS constant above the first layer (READIN 1170: INTEG(RHOX,ABROSS,
        TAUROS,NRHOX,ABROSS(1)*RHOX(1)))."""
        from ..rt.optical_depth import cumulative_integral
        a = self.atmosphere
        k = a.extra["ABROSS"]
        return cumulative_integral(a.rhox, k, initial=k[..., 0] * a.rhox[..., 0], scheme="atlas")


def read_restart(path_or_text: str | Path) -> RestartModel:
    """RestartModel of a punched deck (output_summary.out, restart.dat); NaN-tolerant."""
    s = str(path_or_text)
    text = Path(s).read_text() if "\n" not in s else s
    lines = text.splitlines()
    i_d6 = next(i for i, ln in enumerate(lines) if ln.startswith("READ DECK6"))
    n = int(lines[i_d6][10:13])
    d6 = _deck6_rows(lines[i_d6 + 1:i_d6 + 1 + n])
    tt = lambda k: torch.as_tensor(d6[k], dtype=torch.float64)          # noqa: E731
    head = lines[:i_d6]
    first = " ".join(head)
    teff = re.search(r"TEFF\s+(\S+)", first)
    grav = re.search(r"GRAVITY\s+(\S+)", first)
    tail = lines[i_d6 + 1 + n:]
    pradk = next((_f(ln[5:16]) for ln in tail if ln.startswith("PRADK")), 0.0)
    it = next((int(m.group(1)) for ln in tail for m in [re.search(r"ITERATION\s+(\d+)", ln)] if m), None)
    num = lambda m: torch.tensor(_f(m.group(1)), dtype=torch.float64) if m else None   # noqa: E731
    atm = Atmosphere(rhox=tt("RHOX"), T=tt("T"), P=tt("P"), n_e=tt("XNE"), vturb=tt("VTURB"), teff=num(teff),
                     logg=num(grav), extra={k: tt(k) for k in ("ABROSS", "ACCRAD", "FLXRAD", "VCONV", "VELSND")})
    return RestartModel(atm, _abundance_cards(head), pradk, tuple(head), it)
