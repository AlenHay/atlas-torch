"""Kurucz free-format card readers of atlas7v.for (SYNTHE variant): FREEFF, IWORDF, FREEFR, and XFREEF
(xnfpelsyn.for main-program copy of FREEFR reading unit 17).

COMMON /FREE/NUMCOL,LETCOL,LAST,MORE,IFFAIL,MAXPOW,WORD(6) is the explicit `Free` state (State keys
"free.*"). A card is a Python list `card[0..82]` of 1-character strings with card[i] = CARD(i) (index 0
unused). The routines scan up to column LAST (81 in practice), so they read CARD(81) and, in IWORDF's
'E' look-ahead, CARD(82):

* in READIN, CARD is the 80-element SAVEd local CARD(80): CARD(81), CARD(82) are the first two bytes
  (little endian) of READIN's SAVEd INTEGER MIAC, which follows CARD in static storage in every build
  (checked in the DWARF of libatlas7v.so, libxnfpelsyn.so and the traced xnfpelsyn executable). The
  READIN port fills them via `miac_tail(miac)` before every call;
* in XFREEF, CARD(81) is a real (always blank) element.

Numerics: FREEFF accumulates ANSWER=10.D0*ANSWER+DBLE(digit) and scales by 10.D0**NPT /
10.D0**(ISIGN*NPOWER-NPT): a REAL*8 to a run-time INTEGER power, which gfortran (-O0) evaluates with
libgcc's __powidf2 (square-and-multiply, reciprocal for negative powers), reproduced by `_powidf2`.
`CALL EXIT` paths raise FortranExit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

DIGITS = "0123456789"
A36 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"      # IWORDF's DATA A(1..36)


class FortranExit(Exception):
    """CALL EXIT reached in the Fortran control flow."""


def _powidf2(x: float, m: int) -> float:
    """libgcc __powidf2(x, m): REAL*8 ** INTEGER for a run-time exponent."""
    n = -m if m < 0 else m
    y = x if n % 2 else 1.0
    n >>= 1
    while n:
        x = x * x
        if n % 2:
            y = y * x
        n >>= 1
    return 1.0 / y if m < 0 else y


def miac_tail(miac: int) -> tuple[str, str]:
    """(CARD(81), CARD(82)) as seen through READIN's CARD(80): the two low bytes of INTEGER*4 MIAC."""
    u = int(miac) & 0xFFFFFFFF
    return chr(u & 0xFF), chr((u >> 8) & 0xFF)


def card_from_line(line: str, tail: tuple[str, str] = (" ", " ")) -> list[str]:
    """READ(u,'(80A1)') CARD: columns 1..80 of `line` (blank padded); card[81], card[82] = tail."""
    s = line.rstrip("\n").rstrip("\r")[:80].ljust(80)
    return [" "] + list(s) + list(tail)


@dataclass
class Free:
    """COMMON /FREE/ (WORD as a list of 6 one-character strings)."""
    numcol: int = 0
    letcol: int = 0
    last: int = 0
    more: int = 0
    iffail: int = 0
    maxpow: int = 0
    word: list = field(default_factory=lambda: [" "] * 6)

    KEYS = ("numcol", "letcol", "last", "more", "iffail", "maxpow")

    @classmethod
    def from_state(cls, s) -> "Free":
        def i(k):
            v = s.get("free." + k)
            return 0 if v is None else int(np.asarray(v).reshape(-1)[0])
        w = s.get("free.word")
        word = [" "] * 6 if w is None else [c.decode("latin-1") if isinstance(c, bytes) else str(c)
                                            for c in np.asarray(w).reshape(-1)]
        return cls(*(i(k) for k in cls.KEYS), word=word)

    def to_state(self) -> dict:
        d = {"free." + k: torch.tensor([getattr(self, k)], dtype=torch.int64) for k in self.KEYS}
        d["free.word"] = np.array([c.encode("latin-1") for c in self.word], dtype="S1")
        return d


# ------------------------------------------------------------------------------------------------ FREEFF

def freeff(card: list[str], fr: Free) -> float:
    """FUNCTION FREEFF(CARD): next number on the card from column NUMCOL (updates fr in place)."""
    fr.iffail = 0
    if fr.numcol > fr.last:
        return _fail("FREEFF", card, fr)
    answer = 0.0
    asign = 1.0
    isign = 1
    npt = 0
    if0 = 0
    n = 0
    npower = 0          # SAVEd local (only read after being set in this call on every success path)
    nswch = 100

    for ncol in range(fr.numcol, fr.last + 1):
        c = card[ncol]
        reset = False
        if nswch == 100:
            if c == " ":
                if n != 0:
                    fr.numcol = ncol + 1
                    return answer * asign
                reset = True
            elif c in DIGITS:
                n += 1
                answer = 10.0 * answer + float(DIGITS.index(c))
            elif c == ".":
                nswch = 200
            elif c == ",":
                if n != 0:
                    fr.numcol = ncol + 1
                    return answer * asign
                reset = True
            elif c == "-":
                if n == 0:
                    asign = -1.0
                else:
                    reset = True
            else:
                reset = True
        elif nswch == 200:
            if c in DIGITS:
                n += 1
                npt += 1
                answer = 10.0 * answer + float(DIGITS.index(c))
            elif c == "E" or c == "D":
                nswch = 300
            elif c == "-":
                isign, npower, nswch = -1, 0, 400
            elif c == "+":
                npower, nswch = 0, 400
            elif c == " " or c == ",":
                if n != 0:
                    fr.numcol = ncol + 1
                    return answer * asign / _powidf2(10.0, npt)
                reset = True
            else:
                reset = True
        elif nswch == 300:
            if c in DIGITS:
                npower, if0, nswch = DIGITS.index(c), 1, 400
            elif c == " " or c == "+":
                npower, nswch = 0, 400
            elif c == "-":
                isign, npower, nswch = -1, 0, 400
            else:
                reset = True
        else:  # 400
            if c in DIGITS:
                npower = 10 * npower + DIGITS.index(c)
                if0 = 1
                if npower >= fr.maxpow:
                    reset = True
            elif c == "," or c == " ":
                if if0 == 0:
                    reset = True
                else:
                    fr.numcol = ncol + 1
                    return answer * asign * _powidf2(10.0, isign * npower - npt)
            else:
                reset = True
        if reset:       # label 999 (ISIGN and NPOWER are not reset)
            asign, answer, npt, if0, n, nswch = 1.0, 0.0, 0, 0, 0, 100
    fr.numcol = fr.last + 1
    return _fail("FREEFF", card, fr)


def _fail(name, card, fr: Free):
    fr.iffail = 1
    if fr.more > 0:
        return 0.0 if name == "FREEFF" else 0
    raise FortranExit(f"{name} HAS READ OFF THE END: " + "".join(card[1:fr.last + 1]))


# ------------------------------------------------------------------------------------------------ IWORDF

def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def iwordf(card: list[str], fr: Free) -> int:
    """FUNCTION IWORDF(CARD): base-37 code of the next word from column LETCOL; WORD(1..6) its letters."""
    fr.word = [" "] * 6
    fr.iffail = 0
    if fr.letcol > fr.last:
        return _fail("IWORDF", card, fr)
    n = 0
    miac = 0
    nswch = 500
    for ncol in range(fr.letcol, fr.last + 1):
        c = card[ncol]
        if nswch == 500:
            start = False
            if c != " " and c in A36[:26]:
                ii = A36.index(c) + 1
                start = True
                if c == "E" and ncol != 1:
                    cp = card[ncol - 1]
                    if cp in A36[26:] or cp == ".":
                        cn = card[ncol + 1]
                        if cn in A36[26:] or cn == " ":
                            start = False
            if start:
                n += 1
                miac = ii
                fr.word[0] = A36[ii - 1]
                nswch = 600
            else:
                miac, n, nswch = 0, 0, 500
        else:  # 600
            if c == " " or c == "=" or c == ",":
                fr.letcol = ncol + 1
                return miac
            if c in A36:
                ii = A36.index(c) + 1
                n += 1
                if n <= 6:
                    miac = _i32(37 * miac + ii)
                    fr.word[n - 1] = A36[ii - 1]
            else:
                miac, n, nswch = 0, 0, 500
    fr.letcol = fr.last + 1
    return _fail("IWORDF", card, fr)


# ------------------------------------------------------------------------------------- FREEFR / XFREEF

class CardStream:
    """A formatted sequential input unit read with FORMAT(80A1)."""

    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.pos = 0

    def read_line(self) -> str:
        if self.pos >= len(self.lines):
            raise FortranExit("end of file on input unit")
        s = self.lines[self.pos]
        self.pos += 1
        return s

    def remaining(self) -> str:
        return "\n".join(self.lines[self.pos:])


def _read_into(card: list[str], unit: CardStream, n: int):
    """READ(u,'(80A1)')(CARD(I),I=1,n)."""
    s = unit.read_line().rstrip("\n").rstrip("\r")[:80].ljust(80)
    for i in range(1, n + 1):
        card[i] = s[i - 1] if i <= 80 else " "


def freefr(card: list[str], fr: Free, unit: CardStream) -> float:
    """FUNCTION FREEFR(CARD) (unit 5): FREEFF, continued on the next card if the current one is used up.
    Leaves MORE=1 (the Fortran never resets it)."""
    fr.more = 1
    v = freeff(card, fr)
    if fr.iffail == 0:
        return v
    _read_into(card, unit, fr.last - 1)
    fr.numcol = 1
    return freeff(card, fr)


xfreef = freefr     # FUNCTION XFREEF(CARD) of xnfpelsyn.for: identical algorithm on unit 17, CARD(81) blank
