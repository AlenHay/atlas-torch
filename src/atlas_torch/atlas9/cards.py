"""READIN's free-format card scanner (IWORDF, FREEFF, FREEFR) and Fortran formatted-input helpers.

State of COMMON /FREE/ (NUMCOL, LETCOL, LAST, MORE, IFFAIL, MAXPOW, WORD) lives in a `Free` object; the card is
the 80 characters read with FORMAT(80A1) plus CARD(81)=' ' (EQUIVALENCE with CARD81 in READIN).

Quirks reproduced:
  * FREEFF scales with REAL*4 powers of ten: 10.**NPT and 10.**(ISIGN*NPOWER-NPT) are evaluated in single
    precision (libgcc __powisf2: repeated squaring, reciprocal for negative exponents) and widened, e.g.
    '1.0E-5' -> 10 * float32(1e-6) = 9.99999997e-06.
  * the reset path (label 999) does not reset ISIGN.
  * IWORDF's base-37 code is INTEGER*4 arithmetic (wraps); only the first 6 characters count; an 'E' between
    a digit/'.' and a digit/blank is part of a number, not a word.
"""
from __future__ import annotations

import numpy as np

from .errors import FortranExit

_DIG = "0123456789"
_LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def powisf2(x: float, m: int) -> float:
    """libgcc __powisf2 (REAL*4 ** INTEGER) evaluated in float32, returned widened to double."""
    xf = np.float32(x)
    n = -m if m < 0 else m
    y = xf if n % 2 else np.float32(1.0)
    n >>= 1
    while n:
        xf = np.float32(xf * xf)
        if n % 2:
            y = np.float32(y * xf)
        n >>= 1
    if m < 0:
        y = np.float32(np.float32(1.0) / y)
    return float(y)


def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


class Free:
    """COMMON /FREE/ and the routines that use it."""

    def __init__(self, stdout: list[str]):
        self.numcol = 0
        self.letcol = 0
        self.last = 81
        self.more = 0
        self.iffail = 0
        self.maxpow = 38
        self.word = [" "] * 6
        self.stdout = stdout

    # -- card is a 1-based list of characters with index 0 unused: card[1..81]
    @staticmethod
    def make_card(line: str) -> list[str]:
        s = line.rstrip("\n").rstrip("\r")
        s = (s + " " * 80)[:80]
        return [""] + list(s) + [" "]

    def _readoff(self, card, who: str):
        self.stdout.append(f"1{who} HAS READ OFF THE END")
        self.stdout.append(" " + "".join(card[1:self.last + 1])[:80])
        raise FortranExit("", 0)

    def iwordf(self, card) -> int:
        self.word = [" "] * 6
        self.iffail = 0
        if self.letcol > self.last:
            return self._fail_word(card)
        n = 0
        miac = 0
        sw = 500
        for ncol in range(self.letcol, self.last + 1):
            c = card[ncol] if ncol < len(card) else " "
            if sw == 500:
                if c == " " or c not in _LET[:26]:
                    miac, n, sw = 0, 0, 500
                    continue
                ii = _LET.index(c) + 1
                if c == "E" and ncol != 1:
                    cp = card[ncol - 1]
                    if cp in _DIG or cp == ".":
                        cn = card[ncol + 1] if ncol + 1 < len(card) else " "
                        if cn in _DIG or cn == " ":
                            miac, n, sw = 0, 0, 500
                            continue
                n += 1
                miac = ii
                self.word[0] = c
                sw = 600
            else:
                if c in (" ", "=", ","):
                    self.letcol = ncol + 1
                    return miac
                if c not in _LET:
                    miac, n, sw = 0, 0, 500
                    continue
                ii = _LET.index(c) + 1
                n += 1
                if n <= 6:
                    miac = _i32(37 * miac + ii)
                    self.word[n - 1] = c
        self.letcol = self.last + 1
        return self._fail_word(card)

    def _fail_word(self, card) -> int:
        self.iffail = 1
        if self.more > 0:
            return 0
        self._readoff(card, "IWORDF")

    def freeff(self, card) -> float:
        self.iffail = 0
        if self.numcol > self.last:
            return self._fail_num(card)
        answer, asign, isign, npt, if0, n = 0.0, 1.0, 1, 0, 0, 0
        npower = 0
        sw = 100
        for ncol in range(self.numcol, self.last + 1):
            c = card[ncol] if ncol < len(card) else " "
            reset = False
            if sw == 100:
                if c == " " or c == ",":
                    if n == 0:
                        reset = True
                    else:
                        self.numcol = ncol + 1
                        return answer * asign
                elif c in _DIG:
                    n += 1
                    answer = 10.0 * answer + float(_DIG.index(c))
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
                if c in _DIG:
                    n += 1
                    npt += 1
                    answer = 10.0 * answer + float(_DIG.index(c))
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
                        return answer * asign / powisf2(10.0, npt)
                else:
                    reset = True
            elif sw == 300:
                if c in _DIG:
                    npower, if0, sw = _DIG.index(c), 1, 400
                elif c == " " or c == "+":
                    npower, sw = 0, 400
                elif c == "-":
                    isign, npower, sw = -1, 0, 400
                else:
                    reset = True
            else:  # 400
                if c in _DIG:
                    npower = 10 * npower + _DIG.index(c)
                    if0 = 1
                    if npower >= self.maxpow:
                        reset = True
                elif c == "," or c == " ":
                    if if0 == 0:
                        reset = True
                    else:
                        self.numcol = ncol + 1
                        return answer * asign * powisf2(10.0, isign * npower - npt)
                else:
                    reset = True
            if reset:
                asign, answer, npt, if0, n, sw = 1.0, 0.0, 0, 0, 0, 100
        self.numcol = self.last + 1
        return self._fail_num(card)

    def _fail_num(self, card) -> float:
        self.iffail = 1
        if self.more > 0:
            return 0.0
        self._readoff(card, "FREEFF")

    def freefr(self, card, reader) -> float:
        """FREEFR: FREEFF continuing on the next card (READ(5) of CARD(1..LAST-1)) if this one is exhausted."""
        self.more = 1
        v = self.freeff(card)
        if self.iffail == 0:
            return v
        line = reader.read()
        new = self.make_card(line)
        for i in range(1, self.last):
            card[i] = new[i]
        self.numcol = 1
        return self.freeff(card)


def read_int(field: str) -> int:
    s = field.replace(" ", "")
    return int(s) if s else 0


def read_real(field: str, d: int = 0) -> float:
    """Fortran Fw.d / Ew.d input (blanks ignored): explicit decimal point wins, otherwise d implied decimals;
    exponent may be E/D-prefixed or a bare signed integer; 'NaN'/'Inf' accepted (gfortran)."""
    s = field.replace(" ", "")
    if not s:
        return 0.0
    u = s.upper()
    if u.lstrip("+-") in ("NAN",) or u.lstrip("+-").startswith("NAN("):
        return float("nan")
    if u.lstrip("+-") in ("INF", "INFINITY"):
        return float("-inf") if u.startswith("-") else float("inf")
    u = u.replace("D", "E").replace("Q", "E")
    mant, exp = u, ""
    if "E" in u:
        mant, exp = u.split("E", 1)
    else:
        for k in range(1, len(u)):
            if u[k] in "+-":
                mant, exp = u[:k], u[k:]
                break
    sign = ""
    if mant and mant[0] in "+-":
        sign, mant = mant[0], mant[1:]
    if "." not in mant:
        mant = mant.rjust(d + 1, "0")
        mant = mant[:len(mant) - d] + "." + mant[len(mant) - d:]
    e = int(exp) if exp not in ("", "+", "-") else 0
    return float(f"{sign}{mant}e{e}")


class StdinReader:
    """Unit 5 of the Fortran program: sequential card reader; end of file -> gfortran runtime error (status 2)."""

    def __init__(self, text: str):
        self.lines = text.split("\n")
        if self.lines and self.lines[-1] == "":
            self.lines.pop()
        self.pos = 0

    def read(self) -> str:
        if self.pos >= len(self.lines):
            raise FortranExit("Fortran runtime error: End of file", 2)
        ln = self.lines[self.pos]
        self.pos += 1
        return ln
