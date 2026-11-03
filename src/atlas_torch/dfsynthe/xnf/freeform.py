"""Kurucz free-format card readers FREEFF / IWORDF / FREEFR / XFREEF (xnfdf.for variant).

COMMON /FREE/WORD(6),NUMCOL,LETCOL,LAST,MORE,IFFAIL,MAXPOW is the explicit `Free` state. A card is the
81-column array CARD(1..81) of single characters (80A1 read, column 81 always blank). FREEFF builds the
number exactly as the Fortran does: ANSWER=10*ANSWER+digit in REAL*8 and scales by 10.D0**k (powi).
`CALL EXIT` paths raise FortranExit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .fmath import powi

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"   # IWORDF A(1..36)


class FortranExit(Exception):
    """CALL EXIT (or STOP) reached in the Fortran control flow."""


def card_from_line(line: str) -> list[str]:
    """READ(u,'(80A1)') CARD -> CARD(1..80) plus blank CARD(81); index 0 unused."""
    s = line.rstrip("\n").rstrip("\r")[:80].ljust(80)
    return [" "] + list(s) + [" "]


@dataclass
class Free:
    word: list = field(default_factory=lambda: [" "] * 6)
    numcol: int = 1
    letcol: int = 1
    last: int = 81
    more: int = 0
    iffail: int = 0
    maxpow: int = 38


def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def freeff(card: list[str], fr: Free) -> float:
    fr.iffail = 0
    if fr.numcol > fr.last:
        return _fail_f(card, fr)
    answer = 0.0
    asign = 1.0
    isign = 1
    npt = 0
    if0 = 0
    n = 0
    npower = 0
    sw = 100
    ncol = fr.numcol
    while ncol <= fr.last:
        c = card[ncol]
        if sw == 100:
            if c == " ":
                if n != 0:
                    fr.numcol = ncol + 1
                    return answer * asign
                sw, asign, answer, npt, if0, n = 100, 1.0, 0.0, 0, 0, 0
            elif c in DIGITS:
                n += 1
                answer = 10.0 * answer + float(DIGITS.index(c))
            elif c == ".":
                sw = 200
            elif c == ",":
                if n != 0:
                    fr.numcol = ncol + 1
                    return answer * asign
                sw, asign, answer, npt, if0, n = 100, 1.0, 0.0, 0, 0, 0
            elif c == "-":
                if n == 0:
                    asign = -1.0
                else:
                    sw, asign, answer, npt, if0, n = 100, 1.0, 0.0, 0, 0, 0
            else:
                sw, asign, answer, npt, if0, n = 100, 1.0, 0.0, 0, 0, 0
        elif sw == 200:
            if c in DIGITS:
                n += 1
                npt += 1
                answer = 10.0 * answer + float(DIGITS.index(c))
            elif c == "E":
                sw = 300
            elif c == "-":
                isign, npower, sw = -1, 0, 400
            elif c == "+":
                npower, sw = 0, 400
            elif c in " ,":
                if n != 0:
                    fr.numcol = ncol + 1
                    return answer * asign / powi(10.0, npt)
                sw, asign, answer, npt, if0, n = 100, 1.0, 0.0, 0, 0, 0
            else:
                sw, asign, answer, npt, if0, n = 100, 1.0, 0.0, 0, 0, 0
        elif sw == 300:
            if c in DIGITS:
                npower, if0, sw = DIGITS.index(c), 1, 400
            elif c == " ":
                npower, sw = 0, 400
            elif c == "-":
                isign, npower, sw = -1, 0, 400
            elif c == "+":
                npower, sw = 0, 400
            else:
                sw, asign, answer, npt, if0, n = 100, 1.0, 0.0, 0, 0, 0
        else:  # 400
            if c in DIGITS:
                npower = 10 * npower + DIGITS.index(c)
                if0 = 1
                if npower >= fr.maxpow:
                    sw, asign, answer, npt, if0, n = 100, 1.0, 0.0, 0, 0, 0
            elif c in ", ":
                if if0 == 0:
                    sw, asign, answer, npt, if0, n = 100, 1.0, 0.0, 0, 0, 0
                else:
                    fr.numcol = ncol + 1
                    return answer * asign * powi(10.0, isign * npower - npt)
            else:
                sw, asign, answer, npt, if0, n = 100, 1.0, 0.0, 0, 0, 0
        ncol += 1
    fr.numcol = fr.last + 1
    return _fail_f(card, fr)


def _fail_f(card, fr: Free) -> float:
    fr.iffail = 1
    if fr.more > 0:
        return 0.0
    raise FortranExit("FREEFF HAS READ OFF THE END: " + "".join(card[1:fr.last + 1]))


def iwordf(card: list[str], fr: Free) -> int:
    fr.word = [" "] * 6
    fr.iffail = 0
    if fr.letcol > fr.last:
        return _fail_w(card, fr)
    n = 0
    miac = 0
    sw = 500
    ncol = fr.letcol
    while ncol <= fr.last:
        c = card[ncol]
        if sw == 500:
            restart = True
            if c != " " and c in LETTERS[:26]:
                ii = LETTERS.index(c) + 1
                start_word = True
                if c == "E" and ncol != 1:
                    cp = card[ncol - 1]
                    if cp in DIGITS or cp == ".":
                        cn = card[ncol + 1] if ncol + 1 < len(card) else " "
                        if cn in DIGITS or cn == " ":
                            start_word = False
                if start_word:
                    n += 1
                    miac = ii
                    fr.word[0] = LETTERS[ii - 1]
                    sw = 600
                    restart = False
            if restart:
                miac, n, sw = 0, 0, 500
        else:  # 600
            if c in " =,":
                fr.letcol = ncol + 1
                return miac
            if c in LETTERS:
                ii = LETTERS.index(c) + 1
                n += 1
                if n <= 6:
                    miac = _i32(37 * miac + ii)
                    fr.word[n - 1] = LETTERS[ii - 1]
            else:
                miac, n, sw = 0, 0, 500
        ncol += 1
    fr.letcol = fr.last + 1
    return _fail_w(card, fr)


def _fail_w(card, fr: Free) -> int:
    fr.iffail = 1
    if fr.more > 0:
        return 0
    raise FortranExit("IWORDF HAS READ OFF THE END: " + "".join(card[1:fr.last + 1]))


class CardStream:
    """Formatted sequential input unit: successive 80A1 cards."""

    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.pos = 0

    def read(self) -> list[str]:
        if self.pos >= len(self.lines):
            raise FortranExit("end of file")
        c = card_from_line(self.lines[self.pos])
        self.pos += 1
        return c


def freefr(card: list[str], fr: Free, unit: CardStream) -> float:
    """FREEFR: FREEFF with continuation onto the next card (reads CARD(1..LAST-1) in place)."""
    fr.more = 1
    v = freeff(card, fr)
    if fr.iffail == 0:
        return v
    new = unit.read()
    card[1:fr.last] = new[1:fr.last]
    fr.numcol = 1
    return freeff(card, fr)


xfreef = freefr   # XFREEF (main program, unit 17) is the same algorithm
