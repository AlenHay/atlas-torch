"""KAPPA9's READIN (control cards), its free-format parser (IWORDF / FREEFF / FREEFR) and READMOL.

Only the information KAPPA9 uses is produced, as a State ("block.var" keys, Fortran shapes, scalars (1,)):
abundances (/ELEM/, /XABUND/), opacity switches (/IFOP/, /IF/), the frequency set (/FRESET/ incl. the BLOCKR
DATA image), the molecule tables (/IFEQUA/), the LTE departure coefficients (/DEPART/), /TEFF/, /TURBPR/, and
the model deck (/RHOX/, /TEMP/, /STATE/) -- the deck matters because KAPPA9 overwrites T, TK, HKT, TKEV, TLOG
for its 57 grid temperatures but *not* HCKT, which stays as READIN computed it from the deck temperatures
(PFSAHA reads HCKT).

Fortran semantics reproduced:
* FREEFF accumulates digits in REAL*8 (ANSWER = 10*ANSWER + d) and scales by the REAL*4 power 10.**k
  (single precision: the correctly rounded float32 10^k, as the ifx build computes it).
* IWORDF's base-37 word code wraps in INTEGER*4; an 'E' between a digit/'.' and a digit/blank is part of a
  number, not a word.
* REAL -> INTEGER assignments truncate (IFOP(I) = FREEFF(CARD), IZ = FREEFF(CARD), ...).
* READIN's statement function EXP10 uses the REAL*8 constant 2.30258509299405D0.
* READMOL: formatted READ (F18.2, F7.3, 5E11.4) -- implied decimal point when the field has none, blanks
  ignored (BN); component decoding with the REAL*4 constants XCODE = 1.E14 ... 1.E0 widened to REAL*8.

Cards not used by BasicATLAS's KAPPA9 decks that need machinery outside this port (CALCULATE / STARTING T-TAU
(TTAUP), CHANGE RHOX / SCALE MODEL (MAP1), READ KAPPA / PUNCH / DEPARTURE COEFFICIENTS, SMOOTH, WAVELENGTH
with IFWAVE) raise NotImplementedError. INTEG(RHOX,ABROSS,TAUROS) after READ DECK is not evaluated: TAUROS
only matters to KAPPA9 when TAUSCAT > 0 (SCATTERING LINES card), which is rejected.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..state import State
from ..tables import table

KW = 99
SRC = "kappa9.for"
_DIGITS = "0123456789"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"   # IWORDF's A(1..36)


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= 1 << 31 else x


def pow10_f32(n: int) -> float:
    """The REAL*4 power 10.**N (INTEGER N) as the ifx-compiled FREEFF evaluates it: the correctly rounded
    single-precision 10^N (= libm powf(10, N)), not a binary-powering chain (they differ e.g. for N = 25, -13)."""
    return float(np.float32(10.0 ** n)) if n > -46 else 0.0


class ReadOffEnd(SystemExit):
    """FREEFF / IWORDF ran off the card with MORE = 0 (the Fortran prints a message and calls EXIT)."""


class Cards:
    """/FREE/ state + the card stream (unit 5)."""

    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.pos = 0
        self.card = [" "] * 82          # CARD(1..80) + CARD81 = ' '  (index 0 unused)
        self.numcol = 0
        self.letcol = 1
        self.last = 81
        self.more = 0
        self.iffail = 0
        self.maxpow = 38
        self.word = [" "] * 6

    # READ(5,'(80A1)') CARD
    def read(self) -> None:
        if self.pos >= len(self.lines):
            raise EOFError("end of control cards")
        s = self.lines[self.pos].ljust(80)[:80]
        self.pos += 1
        self.card = [" "] + list(s) + [" "]   # 1-based, CARD(81) blank

    def readline(self) -> str:
        if self.pos >= len(self.lines):
            raise EOFError("end of control cards")
        s = self.lines[self.pos]
        self.pos += 1
        return s

    def c(self, i: int) -> str:
        return self.card[i] if 1 <= i <= 81 else " "

    # ------------------------------------------------------------------ FREEFF
    def freeff(self) -> float:
        self.iffail = 0
        if self.numcol > self.last:
            return self._fail_ff()
        answer, asign, isign, npt, if0, n = 0.0, 1.0, 1, 0, 0, 0
        npower = 0
        sw = 100
        for ncol in range(self.numcol, self.last + 1):
            c = self.c(ncol)
            reset = False
            if sw == 100:
                if c == " " or c == ",":
                    if n == 0:
                        reset = True
                    else:
                        self.numcol = ncol + 1
                        return answer * asign
                elif c in _DIGITS:
                    n += 1
                    answer = 10.0 * answer + float(_DIGITS.index(c))
                elif c == ".":
                    sw = 200
                elif c == "-":
                    if n == 0:
                        asign = -1.0
                    else:
                        reset = True
                else:
                    reset = True
            elif sw == 200:
                if c in _DIGITS:
                    n += 1
                    npt += 1
                    answer = 10.0 * answer + float(_DIGITS.index(c))
                elif c == "E":
                    sw = 300
                elif c == "-":
                    isign, npower, sw = -1, 0, 400
                elif c == "+":
                    npower, sw = 0, 400
                elif c == " " or c == ",":
                    if n == 0:
                        reset = True
                    else:
                        self.numcol = ncol + 1
                        return answer * asign / pow10_f32(npt)
                else:
                    reset = True
            elif sw == 300:
                if c in _DIGITS:
                    npower, if0, sw = _DIGITS.index(c), 1, 400
                elif c == " " or c == "+":
                    npower, sw = 0, 400
                elif c == "-":
                    isign, npower, sw = -1, 0, 400
                else:
                    reset = True
            else:  # 400
                if c in _DIGITS:
                    npower = 10 * npower + _DIGITS.index(c)
                    if0 = 1
                    if npower >= self.maxpow:
                        reset = True
                elif c == "," or c == " ":
                    if if0 == 0:
                        reset = True
                    else:
                        self.numcol = ncol + 1
                        return answer * asign * pow10_f32(isign * npower - npt)
                else:
                    reset = True
            if reset:   # label 999 (ISIGN and NPOWER are not reset)
                asign, answer, npt, if0, n, sw = 1.0, 0.0, 0, 0, 0, 100
        self.numcol = self.last + 1
        return self._fail_ff()

    def _fail_ff(self) -> float:
        self.iffail = 1
        if self.more > 0:
            return 0.0
        raise ReadOffEnd("FREEFF HAS READ OFF THE END: " + "".join(self.card[1:82]))

    def freefr(self) -> float:
        self.more = 1
        v = self.freeff()
        if self.iffail == 0:
            return v
        line = self.readline().ljust(80)
        L = self.last - 1
        self.card = [" "] + list(line[:L]) + self.card[L + 1:]
        self.numcol = 1
        return self.freeff()

    # ------------------------------------------------------------------ IWORDF
    def iwordf(self) -> int:
        self.word = [" "] * 6
        self.iffail = 0
        if self.letcol > self.last:
            return self._fail_wf()
        n, miac, sw = 0, 0, 500
        for ncol in range(self.letcol, self.last + 1):
            c = self.c(ncol)
            reset = False
            if sw == 500:
                ii = _LETTERS.find(c) + 1 if c != " " else 0
                if c == " " or not (1 <= ii <= 26):
                    reset = True
                else:
                    start = True
                    if c == "E" and ncol != 1:
                        p = self.c(ncol - 1)
                        if p in _DIGITS or p == ".":
                            nx = self.c(ncol + 1)
                            if nx in _DIGITS or nx == " ":
                                reset, start = True, False
                    if start:
                        n += 1
                        miac = ii
                        self.word[0] = c
                        sw = 600
            else:  # 600
                if c in (" ", "=", ","):
                    self.letcol = ncol + 1
                    return miac
                ii = _LETTERS.find(c) + 1
                if ii >= 1:
                    n += 1
                    if n <= 6:
                        miac = _i32(37 * miac + ii)
                        self.word[n - 1] = c
                else:
                    reset = True
            if reset:   # label 1999
                miac, n, sw = 0, 0, 500
        self.letcol = self.last + 1
        return self._fail_wf()

    def _fail_wf(self) -> int:
        self.iffail = 1
        if self.more > 0:
            return 0
        raise ReadOffEnd("IWORDF HAS READ OFF THE END: " + "".join(self.card[1:82]))


# --------------------------------------------------------------------------------------------- READMOL

def _fread(field: str, d: int) -> float:
    """Formatted input of one Fw.d / Ew.d field (blanks ignored, implied decimal point)."""
    s = field.replace(" ", "")
    if not s:
        return 0.0
    s = s.upper().replace("D", "E")
    mant, exp = s, 0
    if "E" in s:
        mant, e = s.split("E", 1)
        exp = int(e) if e not in ("", "+", "-") else 0
    else:  # exponent without letter: 1.0-05
        for k in range(len(s) - 1, 0, -1):
            if s[k] in "+-" and s[k - 1] not in "+-":
                mant, exp = s[:k], int(s[k:])
                break
    if "." not in mant:
        sign = ""
        if mant[:1] in "+-":
            sign, mant = mant[0], mant[1:]
        mant = mant.rjust(d + 1, "0")
        mant = sign + mant[:len(mant) - d] + "." + mant[len(mant) - d:]
    return float(f"{mant}e{exp}")


def readmol(path: str | Path) -> dict[str, torch.Tensor]:
    """READMOL: molecules.dat (unit 2) -> /IFEQUA/ (IFEQUA, KCOMPS, LOCJ, CODE, EQUIL, IDEQUA, NEQUA, NUMMOL, NLOC)."""
    maxmol, max1, maxloc = 160, 161, 450
    xcode = table(SRC, "READMOL", "XCODE")          # REAL*4 DATA 1.E14 ... 1.E0 (widened)
    ifequa = np.zeros(101, np.int64)
    kcomps = np.zeros(maxloc, np.int64)
    locj = np.zeros(max1, np.int64)
    code = np.zeros(maxmol, np.float64)
    equil = np.zeros((6, maxmol), np.float64)
    lines = Path(path).read_text().splitlines()
    kloc = 1
    locj[0] = 1
    jmol = 0
    li = 0
    for jmol in range(1, max1 + 1):
        if kloc > maxloc:
            raise ValueError("TOO MANY MOLECULES")
        ln = (lines[li] if li < len(lines) else "").ljust(80)
        li += 1
        c = _fread(ln[0:18], 2)
        e = [_fread(ln[18:25], 3)] + [_fread(ln[25 + 11 * k:36 + 11 * k], 4) for k in range(5)]
        if c == 0.0:
            break
        for ii in range(1, 9):
            if c >= xcode[ii - 1]:
                break
        else:
            raise ValueError(f"bad molecule code {c}")
        x = c
        for i in range(ii, 9):
            idd = int(x / xcode[i - 1] + 0.5)
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
        locj[jmol] = kloc
        code[jmol - 1] = c
        equil[:, jmol - 1] = e
    else:
        raise ValueError("TOO MANY MOLECULES")
    nummol = jmol - 1
    nloc = kloc - 1
    idequa = np.zeros(25, np.int64)
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
    t = torch.from_numpy
    return {"ifequa.ifequa": t(ifequa), "ifequa.kcomps": t(kcomps), "ifequa.locj": t(locj),
            "ifequa.code": t(code), "ifequa.equil": t(np.ascontiguousarray(equil)), "ifequa.idequa": t(idequa),
            "ifequa.nequa": torch.tensor([nequa]), "ifequa.nummol": torch.tensor([nummol]),
            "ifequa.nloc": torch.tensor([nloc])}


# --------------------------------------------------------------------------------------------- READIN

# word codes (MIAC) of the cards READIN understands
_W = {"TEFF": 1020133, "GRAVITY": 519223721, "OPACITY": 1070201044, "KAPPA": 20688921, "ITERATIONS": 661856797,
      "MOLECULES": 930198669, "CALCULATE": 210518764, "ABUNDANCE": 74175307, "PRINT": 30911189,
      "PUNCH": 31069574, "READ": 918640, "LTE": 17173, "NLTE": 726315, "BEGIN": 4011517,
      "SCATTERING": 1323236444, "END": 7367, "TITLE": 37966926, "CONVECTION": 236883734,
      "TURBULENCE": 1427151802, "CHANGE": 223095242, "FREQUENCIES": 450075960, "SURFACE": 1357812572,
      "PRESSURE": 1143518210, "CORRECTION": 237080870, "WAVELENGTH": 1597906832, "SCALE": 35762836,
      "CALL": 153784, "VTURB": 42274019, "SMOOTH": 1342680354}
_ON, _OFF = 569, 20763


def _blockr_freset() -> np.ndarray:
    """/FRESET/FRESET(1563) DATA image: WBIG at FRESET(6), WLIT at FRESET(343) (EQUIVALENCE in BLOCKR)."""
    f = np.zeros(1563)
    wbig = np.concatenate([table(SRC, "BLOCKR", f"WBIG{i}") for i in (1, 2, 3)])
    wlit = np.concatenate([table(SRC, "BLOCKR", f"WLIT{i}") for i in range(1, 9)])
    f[5:5 + wbig.size] = wbig
    f[342:342 + wlit.size] = wlit
    return f


def wavedf() -> np.ndarray:
    """/BIGLIT/WAVEDF(1213,2) as BLOCKBIG initialises it (column 1 = LIT, column 2 = BIG boundaries, nm)."""
    w = np.zeros((1213, 2))
    big = np.concatenate([table(SRC, "BLOCKBIG", n) for n in ("BIGA", "BIGB", "BIGC")])
    lit = np.concatenate([table(SRC, "BLOCKBIG", n) for n in ("LITA", "LITB", "LITC", "LITD", "LITE", "LITF", "LITG")])
    w[:lit.size, 0] = lit
    w[:big.size, 1] = big
    return w


def _exp10(x: float) -> float:
    return math.exp(x * 2.30258509299405)


def readin(stdin_text: str, molecules_path: str | Path | None = None, mode: int = 1) -> State:
    """READIN(MODE) over the control cards; READ MOLECULES reads `molecules_path` (Fortran unit 2)."""
    s = State()
    t = lambda a, dt=torch.float64: torch.tensor(np.asarray(a), dtype=dt)
    # BLOCK DATA images used here
    abund = table(SRC, "BLOCKE", "ABUND").astype(np.float64).copy()
    atmass = table(SRC, "BLOCKE", "ATMASS").astype(np.float64).copy()
    ifop = table(SRC, "BLOCKR", "IFOP").astype(np.int64).copy()
    flags = {k: float(table(SRC, "BLOCKR", k.upper())) for k in ("ifcorr", "ifpres", "ifsurf", "ifscat", "tauscat", "ifmol")}
    freset = _blockr_freset()
    rcoset = np.zeros(1563)
    nulo = nuhi = numnu = 0
    numnu = int(table(SRC, "BLOCKR", "NUMNU"))
    bhyd = table(SRC, "BLOCKR", "BHYD").astype(np.float64).copy()
    bmin = table(SRC, "BLOCKR", "BMIN").astype(np.float64).copy()
    nlteon = int(table(SRC, "BLOCKR", "NLTEON"))
    xscale = float(table(SRC, "BLOCKR", "XSCALE"))
    teff = float(table(SRC, "BLOCKR", "TEFF"))
    grav = float(table(SRC, "BLOCKR", "GRAV"))
    glog = 0.0
    ifconv = int(table(SRC, "BLOCKR", "IFCONV"))
    mixlth = float(table(SRC, "BLOCKR", "MIXLTH"))
    ifturb = int(table(SRC, "BLOCKR", "IFTURB"))
    trb = {k: float(table(SRC, "BLOCKR", k.upper())) for k in ("trbfdg", "trbpow", "trbsnd", "trbcon")}
    ifwave = int(table(SRC, "BLOCKR", "IFWAVE"))
    numits = int(table(SRC, "BLOCKR", "NUMITS"))
    ifprnt = table(SRC, "BLOCKR", "IFPRNT").astype(np.int64).copy()
    ifpnch = table(SRC, "BLOCKR", "IFPNCH").astype(np.int64).copy()
    nrhox = int(table(SRC, "BLOCKR", "NRHOX"))
    vturb = table(SRC, "BLOCKR", "VTURB").astype(np.float64).copy()
    pturb = table(SRC, "BLOCKR", "PTURB").astype(np.float64).copy()
    rhox = np.zeros(KW); T = np.zeros(KW); P = np.zeros(KW); xne = np.zeros(KW)
    abross = np.zeros(KW); prad = np.zeros(KW); accrad = np.zeros(KW)
    pradk0 = 0.0
    freqid = [" "] * 6
    title = [" "] * 74
    mol = None

    cd = Cards(stdin_text)
    cd.card[81] = " "
    cd.last, cd.maxpow = 81, 38

    def next_card():
        cd.more = 0
        cd.letcol = 1
        cd.read()
        miac = cd.iwordf()
        cd.numcol = cd.letcol
        return miac

    def more_words():
        """label 97: next keyword on the same card, or None -> label 98 (next card)."""
        cd.letcol = max(cd.letcol, cd.numcol)
        cd.more = 1
        miac = cd.iwordf()
        if cd.iffail == 1:
            return None
        cd.more = 0
        return miac

    miac = next_card()
    while True:
        nxt = "97"
        if miac == _W["TEFF"]:
            teff = cd.freeff()
        elif miac == _W["GRAVITY"]:
            grav = cd.freeff()
            if grav < 10.0:
                grav = _exp10(grav)
            glog = math.log10(grav)
        elif miac == _W["OPACITY"]:
            m = cd.iwordf()
            if m == 464662:      # IFOP
                cd.numcol = cd.letcol
                for i in range(20):
                    ifop[i] = int(cd.freeff())
                nxt = "98"
            elif m in (_ON, _OFF):
                isw = 1 if m == _ON else 0
                cd.more = 1
                ifop1 = list(table(SRC, "READIN", "IFOP1"))
                while True:
                    m = cd.iwordf()
                    if cd.iffail == 1:
                        break
                    if m not in ifop1:
                        raise ValueError("I DO NOT UNDERSTAND " + "".join(cd.card[1:81]))
                    ifop[ifop1.index(m)] = isw
            else:
                raise ValueError("I DO NOT UNDERSTAND " + "".join(cd.card[1:81]))
        elif miac == _W["ITERATIONS"]:
            numits = int(cd.freeff())
            ifpnch[:] = 0
            ifpnch[numits - 1] = 1
        elif miac == _W["MOLECULES"]:
            m = cd.iwordf()
            if m == _ON:
                flags["ifmol"] = 1
            elif m == _OFF:
                flags["ifmol"] = 0
            else:
                raise ValueError("I DO NOT UNDERSTAND " + "".join(cd.card[1:81]))
        elif miac == _W["ABUNDANCE"]:
            m = cd.iwordf()
            if m == _W["SCALE"]:
                cd.numcol = cd.letcol
                xscale = cd.freeff()
            elif m in (_W["CHANGE"], 74075686):   # CHANGE / ABSOLUTE
                cd.more = 1
                while True:
                    iz = int(cd.freeff())
                    if cd.iffail == 1:
                        break
                    abund[iz - 1] = cd.freeff()
                    if m == 74075686 and iz > 2:
                        abund[iz - 1] = abund[iz - 1] / xscale
                nxt = "98"
            else:
                raise ValueError("I DO NOT UNDERSTAND " + "".join(cd.card[1:81]))
        elif miac == _W["PRINT"]:
            for i in range(numits):
                ifprnt[i] = int(cd.freeff())
        elif miac == _W["PUNCH"]:
            for i in range(numits):
                ifpnch[i] = int(cd.freeff())
        elif miac == _W["READ"]:
            m = cd.iwordf()
            cd.numcol = cd.letcol
            if m == _W["MOLECULES"]:
                if molecules_path is None:
                    raise ValueError("READ MOLECULES needs the molecules file (unit 2)")
                mol = readmol(molecules_path)
            elif m in (209579, 7754456):           # DECK / DECK6
                nrhox = int(cd.freeff())
                for j in range(nrhox):
                    cd.numcol = 1
                    cd.read()
                    rhox[j] = cd.freeff()
                    T[j] = cd.freeff()
                    cd.more = 1
                    P[j] = cd.freeff()
                    xne[j] = cd.freeff()
                    abross[j] = cd.freeff()
                    prad[j] = cd.freeff()
                    vturb[j] = cd.freeff()
                    cd.more = 0
                if rhox[0] < 0.0:
                    rhox[:nrhox] = [_exp10(x) for x in rhox[:nrhox]]
                pradk0 = 0.0
                # CALL INTEG(RHOX,ABROSS,TAUROS,...) not evaluated (see module docstring)
                if m == 7754456:
                    cd.read()
                    cd.numcol = 1
                    pradk0 = cd.freeff()
                    accrad[:nrhox] = prad[:nrhox]
                    # CALL INTEG(RHOX,ACCRAD,PRAD,...) not evaluated (PRAD/PRADK unused by KAPPA9)
            else:
                raise NotImplementedError("READ card variant not ported for KAPPA9: " + "".join(cd.card[1:81]))
            nxt = "98"
        elif miac == _W["LTE"]:
            nlteon = 0
            bhyd[:] = 1.0
            bmin[:] = 1.0
        elif miac == _W["NLTE"]:
            nlteon = 1
        elif miac == _W["BEGIN"]:
            break
        elif miac == _W["SCATTERING"]:
            m = cd.iwordf()
            if m == _ON:
                flags["ifscat"] = 1
            elif m == _OFF:
                flags["ifscat"] = 0
            else:
                raise NotImplementedError("SCATTERING LINES (TAUSCAT > 0) is not supported by the KAPPA9 port")
        elif miac == _W["END"]:
            raise SystemExit("END card")
        elif miac == _W["TITLE"]:
            title = [cd.c(i + 6) for i in range(1, 75)]
            nxt = "98"
        elif miac == _W["CONVECTION"]:
            m = cd.iwordf()
            if m == _ON:
                ifconv = 1
                cd.numcol = cd.letcol
                mixlth = cd.freeff()
            elif m == _OFF:
                ifconv, mixlth = 0, 1.0
            elif m == 790116:   # OVER
                ifconv = 1
                cd.numcol = cd.letcol
                mixlth = cd.freeff(); cd.freeff(); cd.freeff()
            else:
                raise ValueError("I DO NOT UNDERSTAND " + "".join(cd.card[1:81]))
        elif miac == _W["TURBULENCE"]:
            m = cd.iwordf()
            if m == _ON:
                ifturb = 1
                cd.numcol = cd.letcol
                for k in ("trbfdg", "trbpow", "trbsnd", "trbcon"):
                    trb[k] = cd.freeff()
            elif m == _OFF:
                ifturb = 0
                trb = dict.fromkeys(trb, 0.0)
            else:
                raise ValueError("I DO NOT UNDERSTAND " + "".join(cd.card[1:81]))
        elif miac == _W["FREQUENCIES"]:
            num = int(cd.freeff())
            nulo = int(cd.freeff())
            nuhi = int(cd.freeff())
            cd.letcol = cd.numcol
            m = cd.iwordf()
            numnu = num
            freqid = list(cd.word)
            if m == 3078:          # BIG
                off = 5
            elif m == 850035822:   # LITTLE
                off = 342
            else:
                raise ValueError("I DO NOT UNDERSTAND " + "".join(cd.card[1:81]))
            for nu in range(1, numnu + 1):
                freset[nu - 1] = 2.997925e17 / freset[nu + off - 1]
            for nu in range(2, numnu):
                rcoset[nu - 1] = (freset[nu - 2] - freset[nu]) / 2.0
            rcoset[0] = (2.997925e17 / float(np.float32(8.97666)) - freset[1]) / 2.0
            rcoset[numnu - 1] = freset[numnu - 2] / 2.0
            nxt = "98"
        elif miac == _W["SURFACE"]:
            m = cd.iwordf()
            if m not in (321147, _OFF):
                raise NotImplementedError("SURFACE INTENSITY not ported")
            flags["ifsurf"] = 1 if m == 321147 else 0
        elif miac == _W["PRESSURE"]:
            m = cd.iwordf()
            flags["ifpres"] = 1 if m == _ON else 0 if m == _OFF else flags["ifpres"]
        elif miac == _W["CORRECTION"]:
            m = cd.iwordf()
            flags["ifcorr"] = 1 if m == _ON else 0 if m == _OFF else flags["ifcorr"]
        elif miac == _W["CALL"]:
            pass
        elif miac == _W["VTURB"]:
            vnew = cd.freeff()
            vturb[:nrhox] = vnew
        else:
            raise NotImplementedError("READIN card not ported for KAPPA9: " + "".join(cd.card[1:81]).rstrip())
        if nxt == "98":
            miac = next_card()
        else:
            miac = more_words()
            if miac is None:
                miac = next_card()

    # ---- BEGIN (label 1500/1510)
    if mode == 1:
        if numits == 0 or nrhox == 0 or numnu == 0 or teff == 0.0 or grav == 0.0:
            raise SystemExit("READIN: missing ITERATIONS / RHOX / FREQUENCIES / TEFF / GRAVITY")
    if abund[0] < 0.0:
        abund[0] = _exp10(abund[0])
    if abund[1] < 0.0:
        abund[1] = _exp10(abund[1])
    for iz in range(2, 99):
        if abund[iz] > 0.0:
            abund[iz] = math.log10(abund[iz])
    xabund = np.zeros(99)
    for iz in range(2, 99):
        xabund[iz] = _exp10(abund[iz]) * xscale
    xabund[0], xabund[1] = abund[0], abund[1]
    wtmole = 0.0
    for iz in range(99):
        wtmole = wtmole + xabund[iz] * atmass[iz]
    tk = np.zeros(KW); hkt = np.zeros(KW); hckt = np.zeros(KW); tkev = np.zeros(KW); tlog = np.zeros(KW)
    xnatom = np.zeros(KW); rho = np.zeros(KW)
    n = nrhox
    tk[:n] = 1.38054e-16 * T[:n]
    hkt[:n] = 6.6256e-27 / tk[:n]
    hckt[:n] = hkt[:n] * 2.99792458e10
    tkev[:n] = 8.6171e-5 * T[:n]
    tlog[:n] = [math.log(x) for x in T[:n]]
    xnatom[:n] = P[:n] / tk[:n] - xne[:n]
    rho[:n] = xnatom[:n] * wtmole * 1.660e-24
    if ifturb > 0:
        pturb[:n] = 0.5 * rho[:n] * vturb[:n] ** 2

    i64 = lambda v: torch.tensor([int(v)], dtype=torch.int64)
    f64 = lambda v: torch.tensor([float(v)], dtype=torch.float64)
    s.update({
        "elem.abund": t(abund), "elem.atmass": t(atmass), "xabund.xabund": t(xabund), "xabund.wtmole": f64(wtmole),
        "ifop.ifop": t(ifop, torch.int64), "junk.xscale": f64(xscale),
        "if.ifcorr": i64(flags["ifcorr"]), "if.ifpres": i64(flags["ifpres"]), "if.ifsurf": i64(flags["ifsurf"]),
        "if.ifscat": i64(flags["ifscat"]), "if.tauscat": f64(flags["tauscat"]), "if.ifmol": i64(flags["ifmol"]),
        "freset.freset": t(freset), "freset.rcoset": t(rcoset), "freset.nulo": i64(nulo), "freset.nuhi": i64(nuhi),
        "freset.numnu": i64(numnu), "wavey.ifwave": i64(ifwave),
        "depart.bhyd": t(bhyd), "depart.bmin": t(bmin), "depart.nlteon": i64(nlteon),
        "teff.teff": f64(teff), "teff.grav": f64(grav), "teff.glog": f64(glog),
        "conv.ifconv": i64(ifconv), "conv.mixlth": f64(mixlth),
        "turbpr.vturb": t(vturb), "turbpr.pturb": t(pturb), "turbpr.ifturb": i64(ifturb),
        **{f"turbpr.{k}": f64(v) for k, v in trb.items()},
        "iter.numits": i64(numits), "iter.ifprnt": t(ifprnt, torch.int64), "iter.ifpnch": t(ifpnch, torch.int64),
        "rhox.rhox": t(rhox), "rhox.nrhox": i64(nrhox),
        "temp.t": t(T), "temp.tk": t(tk), "temp.hkt": t(hkt), "temp.hckt": t(hckt), "temp.tkev": t(tkev),
        "temp.tlog": t(tlog),
        "state.p": t(P), "state.xne": t(xne), "state.xnatom": t(xnatom), "state.rho": t(rho),
        "abross.abross": t(abross), "rad.accrad": t(accrad), "pzero.pradk0": f64(pradk0),
        "biglit.wavedf": t(wavedf()),
    })
    s["junk.freqid"] = np.array([w.encode() for w in freqid], dtype="S1")
    s["junk.title"] = np.array([w.encode() for w in title], dtype="S1")
    if mol is not None:
        s.update(mol)
    if ifwave:
        raise NotImplementedError("WAVELENGTH frequency sets (IFWAVE=1) are not ported for KAPPA9")
    return s
