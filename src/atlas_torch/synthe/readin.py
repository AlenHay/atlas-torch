"""READIN and READMOL of atlas7v.for (SYNTHE variant): the ATLAS card-deck reader used by xnfpelsyn and
spectrv (`CALL READIN(20)` on the BasicATLAS deck output_synthe.out) and the molecule-list reader.

API:
  initial_state() -> State                 the COMMON image before READIN: zeros + BLOCKR/BLOCKE DATA
                                           (+ /POTION/ from IONPOTS' DATA), keys as in the golden traces
  readin(deck_text, molecules_path=None, mode=20, state=None) -> State
                                           CALL READIN(MODE) reading `deck_text` as unit 5
  readmol(path) -> dict                    CALL READMOL (unit 2 = molecules.dat): /IFEQUA/ and /XNMOL/ entries

READIN is a card interpreter (scalar bookkeeping), ported as a Python state machine that follows the
Fortran's labels (98/99/97 card loop, dispatch on IWORDF's base-37 codes); the layer arithmetic of the BEGIN
block is vectorised. Fortran quirks reproduced:
  * CARD(81) and CARD(82) (read by FREEFF/IWORDF since LAST=81) are the two low bytes of READIN's SAVEd
    INTEGER MIAC, which follows the CARD(80) array in static storage (see freeform.py). They change
    whenever MIAC is reassigned; "save.readin.miac" carries MIAC between calls (initially 0).
  * READ DECK sets the misspelled local PZER0 (not /PZERO/ PZERO): PZERO is left unchanged.
  * CHANGE RHOX writes through the DUM1..DUM8/RHOXA EQUIVALENCEs into /CONV/ (they are aliases of
    DLTDLP..VCONV); the State carries the alias keys (conv.dum1 ...) as the traces do.
  * the integer reads (NUMITS=FREEFF(CARD), IZ=..., IFOP(I)=...) truncate toward zero.
Unported paths (need TTAUP / the Kitt Peak tables, never used on BasicATLAS decks): CALCULATE,
READ STARTING T-TAU, SCALE MODEL -> NotImplementedError. MODE=1 consistency checks are ported.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ..atlas9.physics import libm
from ..fnum import f32
from ..interp import map1
from ..state import State
from ..tables import table
from .freeform import CardStream, Free, FortranExit, freefr, freeff, iwordf, miac_tail
from .pfsaha import potion

SRC = "atlas7v.for"
KW = 99
MAXMOL, MAX1, MAXEQ, MAXLOC = 200, 201, 30, 600

# ------------------------------------------------------------------------------------------- COMMON image
# (block, var, shape, kind) of every COMMON member READIN / READMOL / POPS touch (atlas7v declarations).
_R, _I, _C = "f", "i", "c"
_LAYOUT = [
    ("abross", "abross", (KW,), _R), ("abross", "tauros", (KW,), _R),
    ("bal", "bal1", (KW, 9), _R), ("bal", "aal1", (KW,), _R), ("bal", "sal1", (KW,), _R),
    ("bal", "xnfpal", (KW, 2), _R), ("bal", "bal2", (KW, 1), _R),
    ("bb", "bb1", (KW, 7), _R), ("bb", "xnfpb", (KW, 1), _R),
    ("bc", "bc1", (KW, 14), _R), ("bc", "ac1", (KW,), _R), ("bc", "sc1", (KW,), _R), ("bc", "xnfpc", (KW, 2), _R),
    ("bc", "bc2", (KW, 6), _R),
    ("bca", "bca1", (KW, 8), _R), ("bca", "bca2", (KW, 5), _R), ("bca", "xnfpca", (KW, 2), _R),
    ("bfe", "bfe1", (KW, 15), _R), ("bfe", "afe1", (KW,), _R), ("bfe", "sfe1", (KW,), _R),
    ("bfe", "xnfpfe", (KW, 1), _R),
    ("bhe", "bhe1", (KW, 29), _R), ("bhe", "ahe1", (KW,), _R), ("bhe", "she1", (KW,), _R),
    ("bhe", "bhe2", (KW, 6), _R), ("bhe", "ahe2", (KW,), _R), ("bhe", "she2", (KW,), _R),
    ("bhe", "ahemin", (KW,), _R), ("bhe", "sighe", (KW,), _R), ("bhe", "xnfphe", (KW, 3), _R),
    ("bhe", "xnfhe", (KW, 2), _R),
    ("bhyd", "bhyd", (KW, 8), _R), ("bhyd", "ahyd", (KW,), _R), ("bhyd", "shyd", (KW,), _R),
    ("bhyd", "ah2p", (KW,), _R), ("bhyd", "bmin", (KW,), _R), ("bhyd", "ahmin", (KW,), _R),
    ("bhyd", "shmin", (KW,), _R), ("bhyd", "sigh", (KW,), _R), ("bhyd", "sigh2", (KW,), _R),
    ("bhyd", "ahline", (KW,), _R), ("bhyd", "shline", (KW,), _R), ("bhyd", "xnfph", (KW, 2), _R),
    ("bhyd", "xnfh", (KW,), _R),
    ("bk", "bk1", (KW, 8), _R), ("bk", "xnfpk", (KW, 1), _R),
    ("bmg", "bmg1", (KW, 11), _R), ("bmg", "amg1", (KW,), _R), ("bmg", "smg1", (KW,), _R),
    ("bmg", "xnfpmg", (KW, 2), _R), ("bmg", "bmg2", (KW, 6), _R),
    ("bna", "bna1", (KW, 8), _R), ("bna", "xnfpna", (KW, 1), _R),
    ("bo", "bo1", (KW, 13), _R), ("bo", "xnfpo", (KW, 1), _R), ("bo", "bo2", (KW, 4), _R),
    ("bsi", "bsi1", (KW, 11), _R), ("bsi", "asi1", (KW,), _R), ("bsi", "ssi1", (KW,), _R),
    ("bsi", "xnfpsi", (KW, 2), _R), ("bsi", "bsi2", (KW, 10), _R),
    ("conv", "dltdlp", (KW,), _R), ("conv", "heatcp", (KW,), _R), ("conv", "dlrdlt", (KW,), _R),
    ("conv", "velsnd", (KW,), _R), ("conv", "grdadb", (KW,), _R), ("conv", "hscale", (KW,), _R),
    ("conv", "flxcnv", (KW,), _R), ("conv", "vconv", (KW,), _R), ("conv", "mixlth", (1,), _R),
    ("conv", "ifconv", (1,), _I),
    ("elem", "abund", (99,), _R), ("elem", "atmass", (99,), _R), ("elem", "elem", (99,), "S2"),
    ("xabund", "xabund", (99,), _R), ("xabund", "wtmole", (1,), _R),
    ("flux", "flux", (1,), _R), ("flux", "flxerr", (KW,), _R), ("flux", "flxdrv", (KW,), _R),
    ("flux", "flxrad", (KW,), _R),
    ("free", "numcol", (1,), _I), ("free", "letcol", (1,), _I), ("free", "last", (1,), _I),
    ("free", "more", (1,), _I), ("free", "iffail", (1,), _I), ("free", "maxpow", (1,), _I),
    ("free", "word", (6,), "S1"),
    ("freset", "freset", (500,), _R), ("freset", "rcoset", (500,), _R), ("freset", "nulo", (1,), _I),
    ("freset", "nuhi", (1,), _I), ("freset", "numnu", (1,), _I), ("freset", "ifwave", (1,), _I),
    ("freset", "wbegin", (1,), _R), ("freset", "deltaw", (1,), _R),
    ("height", "height", (KW,), _R),
    ("if", "ifcorr", (1,), _I), ("if", "ifpres", (1,), _I), ("if", "ifsurf", (1,), _I), ("if", "ifscat", (1,), _I),
    ("if", "ifmol", (1,), _I), ("if", "nlteon", (1,), _I), ("if", "ifop", (20,), _I),
    ("iter", "iter", (1,), _I), ("iter", "ifprnt", (15,), _I), ("iter", "ifpnch", (15,), _I),
    ("iter", "numits", (1,), _I),
    ("junk", "xscale", (1,), _R), ("junk", "title", (74,), "S1"), ("junk", "freqid", (6,), "S1"),
    ("junk", "wlte", (1,), "S4"),
    ("mus", "angle", (20,), _R), ("mus", "surfi", (20,), _R), ("mus", "nmu", (1,), _I),
    ("pzero", "pzero", (1,), _R), ("pzero", "pcon", (1,), _R), ("pzero", "pradk0", (1,), _R),
    ("pzero", "pturb0", (1,), _R), ("pzero", "knu", (KW,), _R), ("pzero", "pradk", (KW,), _R),
    ("pzero", "edens", (KW,), _R),
    ("rad", "accrad", (KW,), _R), ("rad", "prad", (KW,), _R),
    ("rhox", "rhox", (KW,), _R), ("rhox", "nrhox", (1,), _I),
    ("state", "p", (KW,), _R), ("state", "xne", (KW,), _R), ("state", "xnatom", (KW,), _R),
    ("state", "rho", (KW,), _R), ("state", "ptotal", (KW,), _R),
    ("teff", "teff", (1,), _R), ("teff", "grav", (1,), _R), ("teff", "glog", (1,), _R),
    ("temp", "t", (KW,), _R), ("temp", "tkev", (KW,), _R), ("temp", "tk", (KW,), _R), ("temp", "hkt", (KW,), _R),
    ("temp", "tlog", (KW,), _R), ("temp", "hckt", (KW,), _R), ("temp", "itemp", (1,), _I),
    ("turbpr", "vturb", (KW,), _R), ("turbpr", "pturb", (KW,), _R), ("turbpr", "trbfdg", (1,), _R),
    ("turbpr", "trbcon", (1,), _R), ("turbpr", "trbpow", (1,), _R), ("turbpr", "trbsnd", (1,), _R),
    ("turbpr", "ifturb", (1,), _I),
    ("ifequa", "ifequa", (101,), _I), ("ifequa", "kcomps", (MAXLOC,), _I), ("ifequa", "locj", (MAX1,), _I),
    ("ifequa", "equil", (7, MAXMOL), _R), ("ifequa", "idequa", (MAXEQ,), _I), ("ifequa", "nequa", (1,), _I),
    ("ifequa", "nloc", (1,), _I),
    ("xnmol", "code", (MAXMOL,), _R), ("xnmol", "xnmol", (KW, MAXMOL), _R), ("xnmol", "xnfpmol", (KW, MAXMOL), _R),
    ("xnmol", "nummol", (1,), _I),
]
# EQUIVALENCE aliases visible in the traces: alias key -> storage key
ALIASES = {"conv.dum1": "conv.dltdlp", "conv.dum2": "conv.heatcp", "conv.dum3": "conv.dlrdlt",
           "conv.dum4": "conv.velsnd", "conv.dum5": "conv.grdadb", "conv.dum6": "conv.hscale",
           "conv.dum7": "conv.flxcnv", "conv.dum8": "conv.vconv", "conv.rhoxa": "conv.vconv",
           "state.taustd": "state.xnatom", "xnmol.codemol": "xnmol.code"}

M = dict(TEFF=1020133, GRAVITY=519223721, OPACITY=1070201044, KAPPA=20688921, ITERATIONS=661856797,
         MOLECULES=930198669, CALCULATE=210518764, ABUNDANCE=74175307, PRINT=30911189, PUNCH=31069574,
         READ=918640, LTE=17173, NLTE=726315, BEGIN=4011517, SCATTERING=1323236444, END=7367,
         TITLE=37966926, CONVECTION=236883734, TURBULENCE=1427151802, CHANGE=223095242,
         SURFACE=1357812572, PRESSURE=1143518210, CORRECTION=237080870, WAVELENGTH=1597906832,
         SCALE=35762836, CALL=153784, ON=569, OFF=20763, IFOP=464662, TABLE=37537060,
         FREQUENCIES=450075960, DEPARTURE=287559136, STARTING=1355094447, DECK=209579, DECK6=7754456,
         INTENSITY=651354309, FLUX=321147)


def _ext() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def default_molecules_path() -> Path:
    """BasicATLAS' SYNTHE molecule list (linked as fort.2 for xnfpelsyn / spectrv)."""
    return _ext() / "BasicATLAS" / "data" / "synthe_files" / "molecules.dat"


@lru_cache(maxsize=None)
def _block_data() -> dict:
    """DATA of BLOCKR / BLOCKE (parsed once from the source)."""
    from ..kurucz_data import load_tables
    t = load_tables(_ext() / "builds" / "src" / SRC)
    return {k: np.asarray(v) for k, v in t.items() if k[0] in ("blockr", "blocke")}


def initial_state() -> State:
    """COMMON image at program start: zero-initialised COMMONs, BLOCKR/BLOCKE DATA, /POTION/ DATA."""
    arrs = {}
    for blk, var, shape, kind in _LAYOUT:
        dt = {"f": np.float64, "i": np.int64}.get(kind, kind)
        arrs[f"{blk}.{var}"] = np.zeros(shape, dtype=dt)
    data = _block_data()
    for unit in ("blockr", "blocke"):
        for blk, var, shape, kind in _LAYOUT:
            if (unit, var) not in data:
                continue
            d = data[(unit, var)]
            a = arrs[f"{blk}.{var}"]
            a.reshape(-1)[:d.size] = d.reshape(-1, order="F").astype(a.dtype)
    s = State({k: (torch.from_numpy(v) if v.dtype.kind in "fi" else v) for k, v in arrs.items()})
    s["potion.potion"] = torch.from_numpy(potion().copy())
    s["save.readin.miac"] = torch.tensor([0], dtype=torch.int64)
    for a, k in ALIASES.items():
        s[a] = s[k]
    return s


# ------------------------------------------------------------------------------ formatted F/E input fields

def fread(field: str, d: int) -> float:
    """Fortran formatted input of an Fw.d / Ew.d field (BLANK='NULL': blanks ignored; implied decimal point
    d digits from the right of the mantissa when the field has none; exponent letter optional)."""
    s = field.replace(" ", "").upper().replace("D", "E")
    if not s or s in "+-":
        return 0.0
    mant, exp = s, ""
    if "E" in s:
        mant, exp = s.split("E", 1)
    else:
        for i in range(1, len(s)):
            if s[i] in "+-":
                mant, exp = s[:i], s[i:]
                break
    if "." not in mant and d > 0:
        sign = ""
        if mant[:1] in "+-":
            sign, mant = mant[0], mant[1:]
        mant = sign + (mant.rjust(d + 1, "0")[:-d] + "." + mant.rjust(d + 1, "0")[-d:])
    return float(mant + ("e" + exp if exp else ""))


def _fields(line: str, widths) -> list[str]:
    out, pos = [], 0
    line = line.rstrip("\n").rstrip("\r")
    for w in widths:
        out.append(line[pos:pos + w].ljust(w))
        pos += w
    return out


# ------------------------------------------------------------------------------------------------ READMOL

def readmol(path=None) -> dict:
    """SUBROUTINE READMOL reading unit 2 (`path`, default BasicATLAS synthe_files/molecules.dat).
    Returns the /IFEQUA/ and /XNMOL/ entries it writes (State keys) plus "xnmol.codemol" (alias)."""
    path = default_molecules_path() if path is None else Path(path)
    lines = Path(path).read_text(encoding="latin-1").split("\n")
    xcode = np.asarray(table(SRC, "readmol", "xcode"), dtype=np.float64)
    ifequa = np.zeros(101, np.int64)
    kcomps = np.zeros(MAXLOC, np.int64)
    locj = np.zeros(MAX1, np.int64)
    code = np.zeros(MAXMOL)
    equil = np.zeros((7, MAXMOL))
    idequa = np.zeros(MAXEQ, np.int64)
    kloc = 1
    locj[0] = 1
    pos = 0
    jmol = MAX1 + 1
    for jm in range(1, MAX1 + 1):
        if pos >= len(lines):
            raise FortranExit("READMOL: end of file before the terminating zero code")
        f = _fields(lines[pos], (18, 7) + (11,) * 6)
        pos += 1
        c = fread(f[0], 2)
        e = [fread(f[1], 3)] + [fread(x, 4) for x in f[2:]]
        if c == 0.0:
            jmol = jm
            break
        for ii in range(1, 9):
            if c >= xcode[ii - 1]:
                break
        else:
            raise FortranExit(f"READMOL: bad code {c}")
        x = c
        for i in range(ii, 9):
            idd = int(x / xcode[i - 1])
            x = x - float(idd) * xcode[i - 1]
            if idd == 0:
                idd = 100
            ifequa[idd - 1] = 1
            kcomps[kloc - 1] = idd
            kloc += 1
        ion = int(x * 100.0 + 0.5)
        if ion >= 1:
            ifequa[99] = 1
            ifequa[100] = 1
            for _ in range(ion):
                kcomps[kloc - 1] = 101
                kloc += 1
        locj[jm] = kloc
        code[jm - 1] = c
        equil[:, jm - 1] = e
    nummol = jmol - 1
    nloc = kloc - 1
    iequa = 1
    for i in range(1, 101):
        if ifequa[i - 1] == 0:
            continue
        iequa += 1
        ifequa[i - 1] = iequa
        idequa[iequa - 1] = i
    nequa = iequa
    ifequa[100] = nequa + 1
    for k in range(nloc):
        kcomps[k] = ifequa[kcomps[k] - 1]
    i1 = lambda v: torch.tensor([v], dtype=torch.int64)
    code_t = torch.from_numpy(code)
    return {"ifequa.ifequa": torch.from_numpy(ifequa), "ifequa.kcomps": torch.from_numpy(kcomps),
            "ifequa.locj": torch.from_numpy(locj), "ifequa.equil": torch.from_numpy(equil),
            "ifequa.idequa": torch.from_numpy(idequa), "ifequa.nequa": i1(nequa), "ifequa.nloc": i1(nloc),
            "xnmol.code": code_t, "xnmol.codemol": code_t, "xnmol.nummol": i1(nummol)}


# ------------------------------------------------------------------------------------------------- READIN

def exp10(x):
    """READIN statement function EXP10(X)=EXP(X*2.30258509299405D0)."""
    return libm.exp(x * 2.30258509299405)


class _Readin:
    def __init__(self, s: State, deck: str, molecules_path, mode: int):
        # mutable numpy working copy (aliases resolved to their storage)
        self.a = {}
        for k, v in s.items():
            if k in ALIASES:
                continue
            self.a[k] = v.detach().cpu().numpy().copy() if isinstance(v, torch.Tensor) else np.array(v, copy=True)
        self.fr = Free.from_state(s)
        self.unit = CardStream(deck)
        self.mol_path = molecules_path
        self.mode = mode
        self.miac = int(self.a.get("save.readin.miac", np.zeros(1)).reshape(-1)[0])
        self.card = [" "] * 83

    # -- helpers ---------------------------------------------------------------------------------------
    def g(self, key):
        return self.a[key]

    def sc(self, key):
        return self.a[key].reshape(-1)[0]

    def set(self, key, v):
        self.a[key].reshape(-1)[0] = v

    def _tail(self):
        self.card[81], self.card[82] = miac_tail(self.miac)
        return self.card

    def read_card(self):
        line = self.unit.read_line().rstrip("\n").rstrip("\r")[:80].ljust(80)
        self.card[1:81] = list(line)

    def ff(self) -> float:
        return freeff(self._tail(), self.fr)

    def fr_(self) -> float:
        return freefr(self._tail(), self.fr, self.unit)

    def iw(self) -> int:
        return iwordf(self._tail(), self.fr)

    def word(self):
        self.miac = self.iw()
        return self.miac

    # -- the card loop -----------------------------------------------------------------------------------
    def run(self):
        fr = self.fr
        fr.last = 81
        fr.maxpow = 38
        label = 98
        while True:
            if label == 98:
                fr.more = 0
                fr.letcol = 1
                label = 99
            if label == 99:
                self.read_card()
                self.word()
                fr.numcol = fr.letcol
                label = 3
            if label == 97:
                fr.letcol = max(fr.letcol, fr.numcol)
                fr.more = 1
                self.word()
                if fr.iffail == 1:
                    label = 98
                    continue
                fr.more = 0
                label = 3
            if label == 3:
                r = self.dispatch()
                if r == "return":
                    return
                label = r

    def dispatch(self):
        m = self.miac
        fr = self.fr
        a = self.a
        if m == M["TEFF"]:
            teff = self.ff()
            self.set("teff.teff", teff)
            self.set("flux.flux", 5.6697e-5 / f32(12.5664) * libm.ipow_const(teff, 4))
            return 97
        if m == M["GRAVITY"]:
            grav = self.ff()
            if grav < 10.0:
                grav = exp10(grav)
            self.set("teff.grav", grav)
            self.set("teff.glog", libm.log10(grav))
            return 97
        if m == M["OPACITY"]:
            w = self.word()
            if w == M["IFOP"]:
                fr.numcol = fr.letcol
                for i in range(20):
                    a["if.ifop"][i] = int(self.ff())
                return 98
            if w not in (M["ON"], M["OFF"]):
                return self.fail()
            iswch = 1 if w == M["ON"] else 0
            fr.more = 1
            ifop1 = np.asarray(table(SRC, "readin", "ifop1")).reshape(-1)
            while True:
                w = self.word()
                if fr.iffail == 1:
                    return 97
                hit = np.nonzero(ifop1 == w)[0]
                if hit.size == 0:
                    return self.fail()
                a["if.ifop"][int(hit[0])] = iswch
        if m == M["KAPPA"]:
            return self.fail()
        if m == M["ITERATIONS"]:
            numits = int(self.ff())
            self.set("iter.numits", numits)
            a["iter.ifpnch"][:] = 0
            a["iter.ifpnch"][numits - 1] = 1
            return 97
        if m == M["MOLECULES"]:
            w = self.word()
            if w == M["ON"]:
                self.set("if.ifmol", 1)
            elif w == M["OFF"]:
                self.set("if.ifmol", 0)
            else:
                return self.fail()
            return 97
        if m == M["CALCULATE"]:
            raise NotImplementedError("READIN CALCULATE (needs TTAUP) is not ported")
        if m == M["ABUNDANCE"]:
            w = self.word()
            if w == M["SCALE"]:
                fr.numcol = fr.letcol
                self.set("junk.xscale", self.ff())
                return 97
            if w == M["CHANGE"]:
                fr.more = 1
                while True:
                    iz = int(self.ff())
                    if fr.iffail == 1:
                        return 98
                    a["elem.abund"][iz - 1] = self.ff()
            if w == M["TABLE"]:
                return self.abundance_table()
            return self.fail()
        if m == M["PRINT"]:
            for i in range(int(self.sc("iter.numits"))):
                a["iter.ifprnt"][i] = int(self.ff())
            return 97
        if m == M["PUNCH"]:
            for i in range(int(self.sc("iter.numits"))):
                a["iter.ifpnch"][i] = int(self.ff())
            return 97
        if m == M["READ"]:
            return self.read_cmd()
        if m == M["LTE"]:
            self.set("if.nlteon", 0)
            a["junk.wlte"][0] = b"LTE "
            a["bhyd.bhyd"][:40, :6] = 1.0
            a["bhyd.bmin"][:40] = 1.0
            return 97
        if m == M["NLTE"]:
            self.set("if.nlteon", 1)
            a["junk.wlte"][0] = b"NLTE"
            return 97
        if m == M["BEGIN"]:
            self.begin()
            return "return"
        if m == M["SCATTERING"]:
            return self.onoff("if.ifscat")
        if m == M["END"]:
            if self.mode != 20:
                raise FortranExit("READIN: END")
            self.set("rhox.nrhox", 0)
            return "return"
        if m == M["TITLE"]:
            a["junk.title"][:] = [c.encode("latin-1") for c in self.card[7:81]]
            return 98
        if m == M["CONVECTION"]:
            w = self.word()
            if w == M["ON"]:
                self.set("conv.ifconv", 1)
                fr.numcol = fr.letcol
                self.set("conv.mixlth", self.ff())
            elif w == M["OFF"]:
                self.set("conv.ifconv", 0)
                self.set("conv.mixlth", 1.0)
                n = int(self.sc("rhox.nrhox"))
                for k in ("dltdlp", "heatcp", "dlrdlt", "velsnd", "grdadb", "hscale", "flxcnv", "vconv"):
                    a["conv." + k][:n] = 0.0
            else:
                return self.fail()
            return 97
        if m == M["TURBULENCE"]:
            w = self.word()
            if w == M["ON"]:
                self.set("turbpr.ifturb", 1)
                fr.numcol = fr.letcol
                for k in ("trbfdg", "trbpow", "trbsnd", "trbcon"):
                    self.set("turbpr." + k, self.ff())
            elif w == M["OFF"]:
                self.set("turbpr.ifturb", 0)
                for k in ("trbfdg", "trbpow", "trbsnd", "trbcon"):
                    self.set("turbpr." + k, 0.0)
            else:
                return self.fail()
            return 97
        if m == M["CHANGE"]:
            return self.change_rhox()
        if m == M["SURFACE"]:
            w = self.word()
            if w == M["INTENSITY"]:
                nmu = int(self.ff())
                self.set("mus.nmu", nmu)
                for mu in range(nmu):
                    a["mus.angle"][mu] = self.fr_()
                self.set("if.ifsurf", 2)
            elif w == M["FLUX"]:
                self.set("if.ifsurf", 1)
                self.set("mus.nmu", 1)
            elif w == M["OFF"]:
                self.set("if.ifsurf", 0)
                self.set("mus.nmu", 1)
            else:
                return self.fail()
            return 97
        if m == M["PRESSURE"]:
            return self.onoff("if.ifpres")
        if m == M["CORRECTION"]:
            w = self.word()
            if w == M["ON"]:
                self.set("if.ifcorr", 1)
            elif w == M["OFF"]:
                self.set("if.ifcorr", 0)
                a["flux.flxerr"][:40] = 0.0
                a["flux.flxdrv"][:40] = 0.0
            else:
                return self.fail()
            return 97
        if m == M["WAVELENGTH"]:
            wbegin, deltaw, wend = self.ff(), self.ff(), self.ff()
            self.set("freset.ifwave", 1)
            self.set("freset.nulo", 1)
            nuhi = int((wend - wbegin) / abs(deltaw) + 0.5) + 1
            self.set("freset.nuhi", nuhi)
            self.set("freset.numnu", nuhi)
            if not wbegin < 1e7:
                if wbegin > 1e20:
                    wbegin, deltaw = 1e7 / (wbegin / 1e25), 1e7 / (deltaw / 1e25)
                else:
                    wbegin, deltaw = 2.99792458e17 / wbegin, 2.99792458e17 / deltaw
            self.set("freset.wbegin", wbegin)
            self.set("freset.deltaw", deltaw)
            return 97
        if m == M["SCALE"]:
            raise NotImplementedError("READIN SCALE MODEL (needs TTAUP) is not ported")
        if m == M["CALL"]:
            return 97                                              # CALL DUMMYR
        return self.fail()

    def fail(self):
        raise FortranExit("I DO NOT UNDERSTAND " + "".join(self.card[1:81]))

    def onoff(self, key):
        w = self.word()
        if w == M["ON"]:
            self.set(key, 1)
        elif w == M["OFF"]:
            self.set(key, 0)
        else:
            return self.fail()
        return 97

    # -- READ ... --------------------------------------------------------------------------------------------
    def read_cmd(self):
        fr, a = self.fr, self.a
        w = self.word()
        fr.numcol = fr.letcol
        if w == M["FREQUENCIES"]:
            num = int(self.ff())
            self.set("freset.nulo", int(self.ff()))
            self.set("freset.nuhi", int(self.ff()))
            self.set("freset.numnu", num)
            fr.letcol = fr.numcol
            self.iw()                                           # NDUMMY=IWORDF(CARD) (MIAC unchanged)
            a["junk.freqid"][:] = [c.encode("latin-1") for c in fr.word]
            fr.numcol = fr.letcol
            for _ in range(num):
                nu = int(self.fr_())
                v = self.ff()
                if v < 1e7:
                    v = 2.99792458e17 / v
                if v > 1e20:
                    v = v * (2.99792458e10 / 1e25)
                a["freset.freset"][nu - 1] = v
                a["freset.rcoset"][nu - 1] = self.ff()
            return 98
        if w == M["DEPARTURE"]:
            n = int(self.ff())
            self.set("rhox.nrhox", n)
            for j in range(n):
                fr.numcol = 1
                self.read_card()
                self.ff()                                       # DUMMY
                for i in range(6):
                    a["bhyd.bhyd"][j, i] = self.ff()
                a["bhyd.bmin"][j] = self.ff()
            a["junk.wlte"][0] = b"NLTE"
            self.set("if.nlteon", 1)
            return 98
        if w == M["STARTING"]:
            raise NotImplementedError("READIN READ STARTING T-TAU (needs TTAUP) is not ported")
        if w in (M["DECK"], M["DECK6"]):
            return self.read_deck(w)
        if w == M["MOLECULES"]:
            for k, v in readmol(self.mol_path).items():
                if k in ALIASES:
                    continue
                self.a[k] = v.numpy().copy()
            return 98
        return self.fail()

    def read_deck(self, w):
        fr, a = self.fr, self.a
        n = int(self.ff())
        self.set("rhox.nrhox", n)
        for j in range(n):
            fr.numcol = 1
            self.read_card()
            a["rhox.rhox"][j] = self.ff()
            a["temp.t"][j] = self.ff()
            fr.more = 1
            a["state.p"][j] = self.ff()
            a["state.xne"][j] = self.ff()
            a["abross.abross"][j] = self.ff()
            a["rad.prad"][j] = self.ff()
            a["turbpr.vturb"][j] = self.ff()
            fr.more = 0
        if not a["rhox.rhox"][0] >= 0.0:
            a["rhox.rhox"][:n] = libm.exp(torch.from_numpy(a["rhox.rhox"][:n] * 2.30258509299405)).numpy()
        self.set("pzero.pradk0", 0.0)
        self.set("pzero.pturb0", float(a["turbpr.pturb"][0]))
        self.set("pzero.pcon", 0.0)
        # PZER0=PCON+PRADK0+PTURB0 (misspelled local; /PZERO/ PZERO unchanged)
        if w != M["DECK6"]:
            return 98
        self.read_card()
        fr.numcol = 1
        pradk0 = self.ff()
        self.set("pzero.pradk0", pradk0)
        a["rad.accrad"][:n] = a["rad.prad"][:n]
        from .josh import integ
        x = torch.from_numpy(a["rhox.rhox"][:n].copy())
        f = torch.from_numpy(a["rad.accrad"][:n].copy())
        prad = integ(x, f, float(a["rad.accrad"][0] * a["rhox.rhox"][0]))
        a["rad.prad"][:n] = prad.numpy()
        a["pzero.pradk"][:n] = a["rad.prad"][:n] + pradk0
        return 98

    def abundance_table(self):
        a = self.a
        rec = self.unit.read_line()
        f = _fields(rec, (7, 10, 10, 10))
        a["elem.abund"][0] = fread(f[1], 6)
        a["elem.abund"][1] = fread(f[3], 6)
        xrel = np.zeros(99)
        vals = []
        while len(vals) < 2 * 97:
            rec = self.unit.read_line()
            ff = _fields(rec, (7, 7, 6) * 5)
            for k in range(5):
                if len(vals) < 2 * 97:
                    vals += [fread(ff[3 * k + 1], 3), fread(ff[3 * k + 2], 3)]
        for iz in range(3, 100):
            a["elem.abund"][iz - 1] = vals[2 * (iz - 3)]
            xrel[iz - 1] = vals[2 * (iz - 3) + 1]
        for iz in range(2, 100):
            a["elem.abund"][iz - 1] = a["elem.abund"][iz - 1] + xrel[iz - 1]
        return 98

    def change_rhox(self):
        a = self.a
        nnew = int(self.ff())
        n = int(self.sc("rhox.nrhox"))
        rhoxa = a["conv.vconv"]                                # RHOXA == VCONV == DUM8
        for j in range(nnew):
            rhoxa[j] = self.fr_()
        x = torch.from_numpy(a["rhox.rhox"][:n].copy())
        xn = torch.from_numpy(rhoxa[:nnew].copy())

        def mp(src):
            return map1(x, torch.from_numpy(np.ascontiguousarray(src[:n])), xn)[0].numpy()
        dums = ("conv.dltdlp", "conv.heatcp", "conv.dlrdlt", "conv.velsnd", "conv.grdadb", "conv.hscale",
                "conv.flxcnv")
        srcs = ("temp.t", "state.p", "state.xne", "abross.abross", "turbpr.vturb", "rad.prad", "bhyd.bmin")
        for d, sk in zip(dums, srcs):
            a[d][:nnew] = mp(a[sk])
        pradk0 = float(self.sc("pzero.pradk0"))
        for d, sk in zip(dums, srcs):
            a[sk][:nnew] = a[d][:nnew]
        a["pzero.pradk"][:nnew] = a["rad.prad"][:nnew] + pradk0
        for i in range(6):
            a["conv.dltdlp"][:nnew] = mp(a["bhyd.bhyd"][:, i])
            a["bhyd.bhyd"][:nnew, i] = a["conv.dltdlp"][:nnew]
        self.set("rhox.nrhox", nnew)
        a["rhox.rhox"][:nnew] = rhoxa[:nnew]
        return 97

    # -- BEGIN -------------------------------------------------------------------------------------------------
    def begin(self):
        a = self.a
        if self.mode == 1:
            for key, msg in (("iter.numits", "HOW MANY ITERATIONS"), ("rhox.nrhox", "HOW MANY RHOX"),
                             ("freset.numnu", "HOW MANY FREQUENCIES"), ("teff.teff", "WHAT TEFF"),
                             ("teff.grav", "WHAT GRAVITY")):
                if self.sc(key) == 0:
                    raise FortranExit(msg)
        ab = a["elem.abund"]
        if ab[0] < 0.0:
            ab[0] = libm.exp(ab[0] * 2.30258509299405)
        if ab[1] < 0.0:
            ab[1] = libm.exp(ab[1] * 2.30258509299405)
        for iz in range(3, 100):
            if ab[iz - 1] > 0.0:
                ab[iz - 1] = libm.log10(float(ab[iz - 1]))
        xscale = float(self.sc("junk.xscale"))
        xab = a["xabund.xabund"]
        for iz in range(3, 100):
            xab[iz - 1] = libm.exp(float(ab[iz - 1]) * 2.30258509299405) * xscale
        xab[0] = ab[0]
        xab[1] = ab[1]
        wtmole = 0.0
        atm = a["elem.atmass"]
        for iz in range(99):
            wtmole = wtmole + float(xab[iz]) * float(atm[iz])
        self.set("xabund.wtmole", wtmole)
        n = int(self.sc("rhox.nrhox"))
        t = torch.from_numpy(a["temp.t"][:n].copy())
        tk = 1.38054e-16 * t
        hkt = libm.rdiv(6.6256e-27, tk)
        p = torch.from_numpy(a["state.p"][:n].copy())
        xne = torch.from_numpy(a["state.xne"][:n].copy())
        xnatom = p / tk - xne
        rho = xnatom * wtmole * 1.660e-24
        vt = torch.from_numpy(a["turbpr.vturb"][:n].copy())
        a["temp.tk"][:n] = tk.numpy()
        a["temp.hkt"][:n] = hkt.numpy()
        a["temp.hckt"][:n] = (hkt * 2.99792458e10).numpy()
        a["temp.tkev"][:n] = (8.6171e-5 * t).numpy()
        a["temp.tlog"][:n] = libm.log(t).numpy()
        a["state.xnatom"][:n] = xnatom.numpy()
        a["state.rho"][:n] = rho.numpy()
        a["turbpr.pturb"][:n] = (0.5 * rho * libm.ipow_const(vt, 2)).numpy()

    def state(self) -> State:
        out = State()
        for k, v in self.a.items():
            out[k] = torch.from_numpy(v) if v.dtype.kind in "fiu" else v
        for k, v in self.fr.to_state().items():
            out[k] = v
        out["save.readin.miac"] = torch.tensor([self.miac], dtype=torch.int64)
        for al, k in ALIASES.items():
            if k in out:
                out[al] = out[k]
        return out


def readin(deck_text: str, molecules_path=None, mode: int = 20, state: State | None = None) -> State:
    """CALL READIN(MODE) with `deck_text` as unit 5 (and `molecules_path` as unit 2 for READ MOLECULES,
    default BasicATLAS' synthe_files/molecules.dat). `state` is the COMMON image before the call
    (default: initial_state()). Returns the State after READIN (every COMMON member READIN/READMOL set, with
    the keys of the golden traces, plus potion.potion and save.readin.miac)."""
    s = initial_state() if state is None else state
    r = _Readin(s, deck_text, molecules_path, mode)
    r.run()
    return r.state()
