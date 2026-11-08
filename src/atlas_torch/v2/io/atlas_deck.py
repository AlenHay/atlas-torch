"""ATLAS model decks (BasicATLAS output_summary.out / output_synthe.out, ATLAS restart decks):
control cards, abundance cards, READ DECK6 structure, trailing cards."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..atmosphere import Atmosphere
from ..composition import Composition

DECK6_COLUMNS = ("RHOX", "T", "P", "XNE", "ABROSS", "ACCRAD", "VTURB", "FLXRAD", "VCONV", "VELSND")


@dataclass
class AtlasDeck:
    """A deck split into the physics (atmosphere, composition) and the cards around it, kept verbatim."""
    atmosphere: Atmosphere
    composition: Composition
    head: list[str] = field(default_factory=list)       # cards before ABUNDANCE SCALE
    middle: list[str] = field(default_factory=list)     # cards between the abundances and READ DECK6
    tail: list[str] = field(default_factory=list)       # cards after the structure (PRADK, BEGIN ...)
    deck6_label: str = "RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB,         FLXRAD,VCONV,VELSND"


def _f(s: str) -> float:
    return float(s.replace("D", "E"))


def _deck6_rows(rows: list[str]) -> dict[str, np.ndarray]:
    """READ DECK6 rows (FORMAT 1PE15.8,0PF9.1,1P8E10.3) -> {column: (L,)}; 'NAN' fields become nan."""
    cells = [[r[0:15], r[15:24]] + [r[24 + 10 * k:34 + 10 * k] for k in range(8)] for r in rows]
    return dict(zip(DECK6_COLUMNS, np.array([[_f(v) for v in r] for r in cells], dtype=np.float64).T))


def read_atlas_deck(path_or_text: str | Path) -> AtlasDeck:
    text = Path(path_or_text).read_text() if "\n" not in str(path_or_text) else str(path_or_text)
    lines = text.splitlines()
    i_ab = next(i for i, ln in enumerate(lines) if ln.startswith("ABUNDANCE SCALE"))
    i_d6 = next(i for i, ln in enumerate(lines) if ln.startswith("READ DECK6"))
    n = int(lines[i_d6][10:13])
    ab_end = i_ab + 1
    while ab_end < len(lines) and lines[ab_end].lstrip().startswith("ABUNDANCE CHANGE"):
        ab_end += 1
    cards = " ".join(lines[i_ab:ab_end])
    scale = _f(re.search(r"ABUNDANCE SCALE\s+(\S+)", cards).group(1))
    change = {int(z): _f(v) for z, v in re.findall(r"(\d+)\s+(-?\d+\.\d*)", cards.split("ABUNDANCE CHANGE", 1)[1])}
    log_metals = [change[z] for z in range(3, 100)]
    comp = Composition.from_atlas(change[1], change[2], log_metals, scale)
    d6 = _deck6_rows(lines[i_d6 + 1:i_d6 + 1 + n])
    tt = lambda k: torch.as_tensor(d6[k], dtype=torch.float64)
    head = " ".join(lines[:i_ab])
    teff = re.search(r"TEFF\s+(\S+)", head)
    grav = re.search(r"GRAVITY\s+(\S+)", head)
    extra = {k: tt(k) for k in ("ABROSS", "ACCRAD", "FLXRAD", "VCONV", "VELSND")}
    atm = Atmosphere(rhox=tt("RHOX"), T=tt("T"), P=tt("P"), n_e=tt("XNE"), vturb=tt("VTURB"),
                     teff=torch.tensor(_f(teff.group(1)), dtype=torch.float64) if teff else None,
                     logg=torch.tensor(_f(grav.group(1)), dtype=torch.float64) if grav else None, extra=extra)
    return AtlasDeck(atm, comp, lines[:i_ab], lines[ab_end:i_d6], lines[i_d6 + 1 + n:], lines[i_d6][14:])


def format_atlas_deck(deck: AtlasDeck) -> str:
    """Deck text (FORMAT of ATLAS PUNCH: 1PE15.8,0PF9.1,1P8E10.3 per layer; BasicATLAS abundance cards)."""
    atm, comp = deck.atmosphere, deck.composition
    h, he, logm = comp.atlas_cards()
    ab = [f"ABUNDANCE SCALE {comp.scale:9.5f} ABUNDANCE CHANGE 1 {h:.5f} 2 {he:.5f}"]
    for k in range(0, 97, 6):
        ab.append(" ABUNDANCE CHANGE" + "".join(f" {z:2d} {logm[z - 3]:6.2f}" for z in range(3 + k, min(9 + k, 100))))
    col = {"RHOX": atm.rhox, "T": atm.T, "P": atm.P, "XNE": atm.n_e, "VTURB": atm.vturb, **atm.extra}
    a = np.stack([col[k].detach().cpu().numpy().reshape(-1) for k in DECK6_COLUMNS], 1)
    rows = [f"{r[0]:15.8E}{r[1]:9.1f}" + "".join(f"{v:10.3E}" for v in r[2:]) for r in a]
    out = deck.head + ab + deck.middle + [f"READ DECK6{len(rows):3d} {deck.deck6_label}"] + rows + deck.tail
    return "\n".join(out) + "\n"


SYNTHE_HEAD = ["SURFACE FLUX", "ITERATIONS 1 PRINT 2 PUNCH 2", "CORRECTION OFF", "PRESSURE OFF", "READ MOLECULES",
               "MOLECULES ON"]


def to_synthe_deck(deck: AtlasDeck) -> AtlasDeck:
    """BasicATLAS synthe(): output_summary.out -> output_synthe.out (control cards for XNFPELSYN prepended,
    the model's VTURB set to 0 so that only SYNTHE's microturbulence applies)."""
    atm = deck.atmosphere.with_(vturb=torch.zeros_like(deck.atmosphere.T))
    head = deck.head if deck.head[:len(SYNTHE_HEAD)] == SYNTHE_HEAD else SYNTHE_HEAD + deck.head
    return AtlasDeck(atm, deck.composition, head, deck.middle, deck.tail, deck.deck6_label)
