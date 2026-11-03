"""Fortran sequential-unformatted record I/O (gfortran / ifx conventions) and formatted-number helpers."""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np


def iter_records(f: BinaryIO) -> Iterator[bytes]:
    """Yield the payload of each record of a sequential unformatted file.

    Handles 4-byte record markers including gfortran/ifx subrecords (negative leading marker = continued).
    """
    while True:
        head = f.read(4)
        if not head:
            return
        if len(head) < 4:
            raise EOFError("truncated record marker")
        parts = []
        while True:
            (n,) = struct.unpack("<i", head)
            cont = n < 0
            n = abs(n)
            parts.append(f.read(n))
            tail = f.read(4)
            if not cont:
                break
            head = f.read(4)
        yield b"".join(parts)


def read_records(path: str | Path) -> list[bytes]:
    with open(path, "rb") as f:
        return list(iter_records(f))


def write_record(f: BinaryIO, payload: bytes) -> None:
    n = len(payload)
    if n > 2**31 - 9:
        raise ValueError("subrecords not implemented")
    m = struct.pack("<i", n)
    f.write(m)
    f.write(payload)
    f.write(m)


_FNUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$")


def ffloat(s: str) -> float:
    """Parse a Fortran-formatted real: handles D exponents and exponent-letter-less forms like 1.234-105."""
    s = s.strip().replace("D", "E").replace("d", "e")
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        m = _FNUM.match(s)
        if m:
            return float(m.group(1) + "e" + m.group(2))
        raise


def fixed_fields(line: str, widths: list[int]) -> list[str]:
    out, i = [], 0
    for w in widths:
        out.append(line[i:i + w])
        i += w
    return out
