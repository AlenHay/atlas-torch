"""Standard output of PROGRAM KAPPA9 (kapm40k2.out), as the stock (ifx) executable writes it.

BasicATLAS's validate_run() parses the READIN BEGIN listing of this printout (abundance table and 0XSCALE,
FORMAT 1512) -- reproduced byte for byte.  The whole printout is reproduced:

  READMOL's MOLECULES INPUT listing (FORMATs 10/14/29, MAX 160/450/25), the READIN BEGIN listing (1512, 1517,
  1518 with TAUSCAT, 1521, 1561/1563/1564 + the RCOSET sum 9501, 1576), the main program's ``type*,j,vturb(j)``
  (ifx list-directed), WRITE(6,75) TABT/TABP, and per pressure K = 1..42 NMOLEC's tables (112: RHOX, T, P, XNE,
  XNATOM, RHO; 113: molecular number densities *before* NMOLEC divides them by the partition functions) and
  the KTAB block (the same text as kapk*.dat, FORMAT 77).

The NMOLEC tables need the number densities NMOLEC prints before its partition-function division; the torch
physics (atlas_torch.atlas9.physics.molec) overwrites them, so :class:`NmolecCapture` records them while the
populations are computed (kappa9_prog.continuum_tables).  With the TEST-ONLY oracle physics there is no capture
and the NMOLEC tables are omitted (a note line says so).
"""
from __future__ import annotations

import re

import numpy as np
import torch

from ..dfsynthe import printouts as P

MAXMOL, MAXLOC, MAXEQ = 160, 450, 25


class NmolecCapture:
    """Context manager: records XNMOL(1..NRHOX, :) as NMOLEC holds it before the partition-function division
    (the values its FORMAT 113 prints), for every NMOLEC call of the torch physics inside the block."""

    def __init__(self):
        self.calls: list[np.ndarray] = []
        self._mod = None

    def __enter__(self):
        try:
            from ..atlas9.physics import molec
        except ImportError:
            return self
        self._mod, self._orig = molec, molec._partition_divide
        orig, calls = self._orig, self.calls

        def wrapped(t, s, n, xnz, xnmol, *a, **kw):
            calls.append(np.array(xnmol[:n], dtype=np.float64, copy=True))
            return orig(t, s, n, xnz, xnmol, *a, **kw)
        molec._partition_divide = wrapped
        return self

    def __exit__(self, *exc):
        if self._mod is not None:
            self._mod._partition_divide = self._orig
        return False


def column_record(phys, cap: NmolecCapture, nt: int) -> dict | None:
    """What NMOLEC prints for one pressure column (None when not available: oracle physics)."""
    cur = getattr(phys, "cur", None)
    if cur is None or not cap.calls:
        return None
    g = lambda k: cur[k][:nt].detach().cpu().numpy().astype(np.float64).copy()
    return {"xne": g("state.xne"), "xnatom": g("state.xnatom"), "rho": g("state.rho"), "xnmol": cap.calls[0]}


def _f(v) -> float:
    return float(torch.as_tensor(v).reshape(-1)[0])


def _i(v) -> int:
    return int(torch.as_tensor(v).reshape(-1)[0])


def _np(v) -> np.ndarray:
    return torch.as_tensor(v).detach().cpu().numpy()


def begin_values(rd) -> dict:
    """begin_listing values from the KAPPA9 READIN State."""
    from ..dfsynthe.xnf.fdata import T as tab
    from ..dfsynthe.xnf.readin import integ
    n = _i(rd["rhox.nrhox"])
    rhox, accrad = _np(rd["rhox.rhox"]).astype(np.float64), _np(rd["rad.accrad"]).astype(np.float64)
    prad = np.zeros_like(rhox)
    prad[:n] = integ(rhox[:n], accrad[:n], n, accrad[0] * rhox[0])     # READ DECK: INTEG(RHOX,ACCRAD,PRAD,...)
    bh = _np(rd["depart.bhyd"]).astype(np.float64)
    return dict(teff=_f(rd["teff.teff"]), glog=_f(rd["teff.glog"]), wlte="NLTE" if _i(rd["depart.nlteon"]) else "LTE ",
                title=list(rd["junk.title"]), xscale=_f(rd["junk.xscale"]), elem=list(tab("blocke", "elem")),
                abund=[float(a) for a in _np(rd["elem.abund"])], ifop=[int(x) for x in _np(rd["ifop.ifop"])],
                ifcorr=_i(rd["if.ifcorr"]), ifpres=_i(rd["if.ifpres"]), ifsurf=_i(rd["if.ifsurf"]),
                ifscat=_i(rd["if.ifscat"]), tauscat=_f(rd["if.tauscat"]), ifconv=_i(rd["conv.ifconv"]),
                mixlth=_f(rd["conv.mixlth"]), ifmol=_i(rd["if.ifmol"]), ifturb=_i(rd["turbpr.ifturb"]),
                trbfdg=_f(rd["turbpr.trbfdg"]), trbpow=_f(rd["turbpr.trbpow"]), trbsnd=_f(rd["turbpr.trbsnd"]),
                trbcon=_f(rd["turbpr.trbcon"]), numits=_i(rd["iter.numits"]), ifprnt=list(_np(rd["iter.ifprnt"])),
                ifpnch=list(_np(rd["iter.ifpnch"])), ifwave=_i(rd["wavey.ifwave"]),
                wbegin=_f(rd["wavey.wbegin"]) if "wavey.wbegin" in rd else 0.0,
                deltaw=_f(rd["wavey.deltaw"]) if "wavey.deltaw" in rd else 0.0,
                numnu=_i(rd["freset.numnu"]), freqid=list(rd["junk.freqid"]), nulo=_i(rd["freset.nulo"]),
                nuhi=_i(rd["freset.nuhi"]), freset=_np(rd["freset.freset"]), rcoset=_np(rd["freset.rcoset"]),
                nrhox=n, rhox=rhox, t=_np(rd["temp.t"]), p=_np(rd["state.p"]), xne=_np(rd["state.xne"]),
                abross=_np(rd["abross.abross"]), prad=prad, vturb=_np(rd["turbpr.vturb"]), bhyd=bh,
                bmin=_np(rd["depart.bmin"]))


def kappa9_stdout(stdin_text: str, rd, cont: dict, kabros) -> str:
    """kapm40k2.out for one KAPPA9 run: ``rd`` = readin(stdin), ``cont`` = continuum_tables(rd) (with the
    'printout_columns' its population loop recorded), ``kabros`` = rosseland(...) [42, 57]."""
    from .kappa9_prog import NP, NT, kapk_block, kapk_header, tabp
    out = []
    if re.search(r"^\s*READ\s+MOLE", stdin_text.upper(), re.M):
        nummol = _i(rd["ifequa.nummol"])
        out += P.readmol_listing(_np(rd["ifequa.code"])[:nummol], _np(rd["ifequa.equil"])[:, :nummol],
                                 _i(rd["ifequa.nloc"]), _i(rd["ifequa.nequa"]), MAXMOL, MAXLOC, MAXEQ)
    v = begin_values(rd)
    out += P.begin_listing(v, "kappa9", 1)
    out += [P.ld_int(j + 1) + P.ld_real8(v["vturb"][j]) for j in range(v["nrhox"])]
    text = "\n".join(out) + "\n" + kapk_header()
    kab = _np(kabros)
    tp = tabp()
    t = _np(cont["grid"]["t"]).astype(np.float64)
    p = _np(cont["p"]).astype(np.float64)
    cols = cont.get("printout_columns")
    codes = P.padded_codes(_np(rd["ifequa.code"])[:_i(rd["ifequa.nummol"])], MAXMOL)
    for k in range(NP):
        c = cols[k] if cols else None
        if c is not None:
            lines = P.nmolec_tables(v["rhox"][:NT], t, np.full(NT, p[k]), c["xne"], c["xnatom"], c["rho"], codes,
                                    c["xnmol"], "kappa9")
            text += "\n".join(lines) + "\n"
        else:
            text += " NMOLEC printout not available (KAPPA9 oracle physics)\n"
        text += kapk_block(k + 1, kab[k], float(tp[k]))
    return text
