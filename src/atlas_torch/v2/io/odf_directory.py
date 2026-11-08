"""BasicATLAS ODF directory (BasicATLAS dfsynthe(): p00big{v}.bdf, p00lit{v}.bdf, kappa.ros, xnfdf.com, and the
abundance blocks of xnfdf.out and kapm40k2.out that BasicATLAS meta()/validate() parse), and the abundance cards of
its control files."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..composition import Composition

VTURBS = (0, 1, 2, 4, 8)


def read_abundance_cards(path_or_text) -> Composition:
    """Composition from the ABUNDANCE SCALE / ABUNDANCE CHANGE cards of an ATLAS-family control file (xnfdf.com,
    kappa9v{v}.com, an ATLAS deck): READIN labels 1510-1514 (H, He number fractions; metals log10 before the
    scale)."""
    text = Path(path_or_text).read_text() if "\n" not in str(path_or_text) else str(path_or_text)
    lines = [ln for ln in text.splitlines() if "ABUNDANCE" in ln]
    cards = " ".join(lines)
    scale = float(re.search(r"ABUNDANCE SCALE\s+(\S+)", cards).group(1).replace("D", "E"))
    change = {}
    for part in cards.split("ABUNDANCE CHANGE")[1:]:
        for z, v in re.findall(r"(\d+)\s+(-?\d*\.\d*(?:[EeDd][-+]?\d+)?)", part):
            change[int(z)] = float(v.replace("D", "E"))
    return Composition.from_atlas(change[1], change[2], [change[z] for z in range(3, 100)], scale)


def abundance_cards(comp: Composition) -> list[str]:
    """The ABUNDANCE SCALE / CHANGE cards as BasicATLAS writes them in xnfdf.com (F6.2 metals, 6 per card)."""
    h, he, logm = comp.atlas_cards()
    out = [f"ABUNDANCE SCALE {comp.scale:9.6f} ABUNDANCE CHANGE 1 {h:.5f} 2 {he:.5f}"]
    for k in range(0, 97, 6):
        out.append(" ABUNDANCE CHANGE" + "".join(f" {z:2d} {max(logm[z - 3], -20.0):6.2f}"
                                                for z in range(3 + k, min(9 + k, 100))))
    return out


def xnfdf_deck6(temperatures, log_p) -> list[str]:
    """READ DECK6 blocks of xnfdf.com (one per temperature: RHOX = P, T, P), as BasicATLAS atlas() reads the
    ODF grid back."""
    out = []
    for t in temperatures:
        out.append(f"READ DECK6 {len(log_p)} RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB,CONVFRAC,VCONV")
        for lp in log_p:
            p = f"{10.0 ** lp:.3E}".replace("E-0", "E-").replace("E+0", "E+")
            out.append(f"   {p:>8s}   {t:.0f}.  {p:>8s}  0 0 0 0 0")
        out.append(" 1")
        out.append("BEGIN")
    return out


def write_xnfdf_com(path, comp: Composition, temperatures, log_p, title: str = "atlas_torch v2 ODF") -> None:
    """xnfdf.com with the cards of a BasicATLAS ODF run (no shell commands are needed by readers)."""
    head = ["READ MOLECULES", "MOLECULES ON", "ITERATIONS 1 PRINT 0 PUNCH 0", "READ FREQUENCIES 1 1 1 TEST", "1 1 1",
            "TEFF   5777.  GRAVITY 4.43770 LTE", f"TITLE {title}",
            " OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0",
            " CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00"]
    Path(path).write_text("\n".join(head + abundance_cards(comp) + xnfdf_deck6(temperatures, log_p) + ["END"])
                          + "\n")


def write_abundance_block(path, comp: Composition, program: str) -> None:
    """Only READIN's abundance echo (0XSCALE line and the element listing, as in xnfdf.out / kapm40k2.out), which
    BasicATLAS meta()/validate() parse; no computed numbers are written."""
    from ..constants import element_symbols
    h, he, logm = comp.atlas_cards()
    sym = [x.upper() for x in element_symbols()][2:]
    rows = [f"0XSCALE {comp.scale:9.6f}   H {h:8.5f}   HE{he:8.5f}"]
    ent = [f" {s:<2s}{max(v, -20.0):6.2f}" for s, v in zip(sym, logm)]
    rows += ["".join(ent[i:i + 10]) for i in range(0, len(ent), 10)]
    Path(path).write_text(f" {program} (atlas_torch v2): READIN abundance echo only\n" + "\n".join(rows) + "\n")


def write_odf_directory(directory, comp: Composition, big=None, little=None, rosseland=None,
                        temperatures=None, log_p=None) -> Path:
    """p00big{v}.bdf / p00lit{v}.bdf (OpacityDistribution.write), kappa.ros (RosselandTable.write), xnfdf.com
    and the abundance blocks of xnfdf.out and kapm40k2.out."""
    from ..opacity.odf import ODF_LOG_P, ODF_LOG_T
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for odf in (big, little):
        if odf is not None:
            odf.write(d)
    if rosseland is not None:
        rosseland.write(d / "kappa.ros")
    t = np.round(10.0 ** np.asarray(ODF_LOG_T if temperatures is None else temperatures))
    write_xnfdf_com(d / "xnfdf.com", comp, t, ODF_LOG_P if log_p is None else log_p)
    write_abundance_block(d / "xnfdf.out", comp, "XNFDF")
    write_abundance_block(d / "kapm40k2.out", comp, "KAPPA9")
    return d
