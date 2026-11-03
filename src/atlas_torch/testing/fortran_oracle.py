"""ctypes oracles over the original Fortran routines (PyTLAS pattern, generalised).

A physics library (e.g. atlas7v.for) is compiled to a shared object with -g -fPIC and the stock
compiler/flags (tools/oracle/build_libs.sh). This module reads the exact memory layout of every COMMON
block and every SAVEd/DATA local from the DWARF debug info, maps them as writable numpy views, and calls
any routine with Fortran by-reference conventions. Each Oracle loads a private copy of the .so, so
SAVEd state starts from the load-time image; `reset()` reloads.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from elftools.elf.elffile import ELFFile

_BASE = {
    ("real", 8): "<f8", ("real", 4): "<f4", ("real", 16): "<f16",
    ("integer", 4): "<i4", ("integer", 2): "<i2", ("integer", 1): "i1", ("integer", 8): "<i8",
    ("logical", 4): "<i4", ("logical", 1): "i1", ("logical", 2): "<i2", ("logical", 8): "<i8",
    ("complex", 16): "<c16", ("complex", 8): "<c8",
}


@dataclass
class Var:
    name: str
    addr: int          # st_value-relative (link-time) address
    dtype: str
    shape: tuple       # Fortran order dims ( () for scalar )
    nbytes: int


@dataclass
class Unit:
    name: str
    params: list = field(default_factory=list)   # [(name, dtype, shape-or-None)]
    statics: dict = field(default_factory=dict)  # saved locals
    commons: set = field(default_factory=set)


def _type_info(die, cache):
    """Return (dtype, shape) for a DWARF type DIE."""
    key = die.offset
    if key in cache:
        return cache[key]
    tag = die.tag
    if tag in ("DW_TAG_typedef", "DW_TAG_const_type", "DW_TAG_volatile_type"):
        r = _type_info(die.get_DIE_from_attribute("DW_AT_type"), cache)
    elif tag == "DW_TAG_base_type":
        nm = die.attributes["DW_AT_name"].value.decode().lower()
        size = die.attributes["DW_AT_byte_size"].value
        if nm.startswith("character") or nm.startswith("char"):
            r = (f"S{size}", ())
        else:
            kind = nm.split("(")[0].split("*")[0].strip()
            r = (_BASE.get((kind, size), f"V{size}"), ())
    elif tag == "DW_TAG_string_type":
        size = die.attributes.get("DW_AT_byte_size")
        r = (f"S{size.value if size else 1}", ())
    elif tag == "DW_TAG_array_type":
        base, _ = _type_info(die.get_DIE_from_attribute("DW_AT_type"), cache)
        dims = []
        for sr in die.iter_children():
            if sr.tag != "DW_TAG_subrange_type":
                continue
            a = sr.attributes
            if "DW_AT_count" in a and a["DW_AT_count"].form.startswith("DW_FORM_data"):
                dims.append(int(a["DW_AT_count"].value))
            elif "DW_AT_upper_bound" in a and a["DW_AT_upper_bound"].form.startswith(("DW_FORM_data", "DW_FORM_sdata", "DW_FORM_udata")):
                lo = a["DW_AT_lower_bound"].value if "DW_AT_lower_bound" in a else 1
                dims.append(int(a["DW_AT_upper_bound"].value) - int(lo) + 1)
            else:
                dims.append(None)  # adjustable / assumed size
        r = (base, tuple(dims))
    else:
        r = ("V1", ())
    cache[key] = r
    return r


def _static_addr(loc):
    """DW_OP_addr a [DW_OP_plus_uconst k] -> a+k; else None."""
    if not isinstance(loc, list) or not loc or loc[0] != 3:
        return None
    a = int.from_bytes(bytes(loc[1:9]), "little")
    rest = loc[9:]
    if rest:
        if rest[0] != 35:
            return None
        # ULEB128
        k, shift = 0, 0
        for b in rest[1:]:
            k |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        a += k
    return a


def _nbytes(dtype, shape):
    n = int(np.dtype(dtype).itemsize)
    for d in shape:
        n *= d if d else 1
    return n


@lru_cache(maxsize=16)
def read_layout(so_path: str):
    """Parse DWARF: returns (commons {block: {var: Var}}, units {name: Unit}, symtab {sym: st_value})."""
    commons: dict[str, dict[str, Var]] = {}
    units: dict[str, Unit] = {}
    with open(so_path, "rb") as fh:
        elf = ELFFile(fh)
        syms = {}
        for sec in elf.iter_sections():
            if sec.header.sh_type in ("SHT_SYMTAB", "SHT_DYNSYM"):
                for s in sec.iter_symbols():
                    if s.name:
                        syms.setdefault(s.name, s.entry.st_value)
        dw = elf.get_dwarf_info()
        cache: dict = {}
        for cu in dw.iter_CUs():
            for die in cu.get_top_DIE().iter_children():
                if die.tag != "DW_TAG_subprogram" or "DW_AT_name" not in die.attributes:
                    continue
                uname = die.attributes["DW_AT_name"].value.decode().lower()
                u = units.setdefault(uname, Unit(uname))
                fill_params = not u.params
                for ch in die.iter_children():
                    if ch.tag == "DW_TAG_formal_parameter" and fill_params and "DW_AT_name" in ch.attributes:
                        t = _type_info(ch.get_DIE_from_attribute("DW_AT_type"), cache) if "DW_AT_type" in ch.attributes else ("V1", ())
                        u.params.append((ch.attributes["DW_AT_name"].value.decode().lower(), t[0], t[1]))
                    elif ch.tag == "DW_TAG_common_block":
                        bname = ch.attributes["DW_AT_name"].value.decode().lower()
                        u.commons.add(bname)
                        blk = commons.setdefault(bname, {})
                        for v in ch.iter_children():
                            if v.tag != "DW_TAG_variable" or "DW_AT_location" not in v.attributes:
                                continue
                            vn = v.attributes["DW_AT_name"].value.decode().lower()
                            if vn in blk:
                                continue
                            a = _static_addr(v.attributes["DW_AT_location"].value)
                            dt, sh = _type_info(v.get_DIE_from_attribute("DW_AT_type"), cache)
                            blk[vn] = Var(vn, a, dt, sh, _nbytes(dt, sh))
                    elif ch.tag == "DW_TAG_variable" and "DW_AT_location" in ch.attributes and "DW_AT_name" in ch.attributes:
                        a = _static_addr(ch.attributes["DW_AT_location"].value)
                        if a is None:
                            continue
                        vn = ch.attributes["DW_AT_name"].value.decode().lower()
                        dt, sh = _type_info(ch.get_DIE_from_attribute("DW_AT_type"), cache)
                        if None in sh:
                            continue
                        u.statics[vn] = Var(vn, a, dt, sh, _nbytes(dt, sh))
    return commons, units, syms


class Oracle:
    """Load a private copy of a Fortran shared library and expose COMMONs, SAVEd locals and routines."""

    def __init__(self, so_path: str | Path):
        self.so_path = str(Path(so_path).resolve())
        self.commons_layout, self.units, self.syms = read_layout(self.so_path)
        self._tmp = None
        self.reset()

    def reset(self):
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = tempfile.mkdtemp(prefix="oracle_")
        p = os.path.join(self._tmp, os.path.basename(self.so_path))
        shutil.copy(self.so_path, p)
        self.lib = ctypes.CDLL(p, mode=os.RTLD_LOCAL)
        # load base = runtime address of an exported symbol - its link-time value
        for s in self.syms:
            if s.endswith("_") and not s.startswith("_"):
                try:
                    rt = ctypes.addressof(ctypes.c_char.in_dll(self.lib, s))
                except ValueError:
                    continue
                self.base = rt - self.syms[s]
                break
        self._views = {}

    def _view(self, var: Var) -> np.ndarray:
        buf = (ctypes.c_char * var.nbytes).from_address(self.base + var.addr)
        a = np.frombuffer(buf, dtype=var.dtype, count=var.nbytes // np.dtype(var.dtype).itemsize)
        return a.reshape(var.shape, order="F") if var.shape else a

    def common(self, block: str) -> dict[str, np.ndarray]:
        block = block.lower()
        return {k: self._view(v) for k, v in self.commons_layout[block].items()}

    def saved(self, unit: str) -> dict[str, np.ndarray]:
        return {k: self._view(v) for k, v in self.units[unit.lower()].statics.items()}

    def set_common(self, block: str, **values):
        views = self.common(block)
        for k, v in values.items():
            dst = views[k.lower()]
            if dst.ndim == 1 and dst.shape == (1,) and np.ndim(v) == 0:
                dst[0] = v
            else:
                src = np.asarray(v, dtype=dst.dtype)
                dst[tuple(slice(0, n) for n in src.shape)] = src

    def call(self, routine: str, *args, restype=None):
        """Call SUBROUTINE/FUNCTION `routine` passing every arg by reference.

        numpy arrays are passed in place (must be F-contiguous, right dtype); python float -> REAL*8,
        int -> INTEGER*4 temporaries. Returns (result, [scalar temporaries' final values]).
        """
        fn = getattr(self.lib, routine.lower() + "_")
        fn.restype = restype
        cargs, temps = [], []
        for a in args:
            if isinstance(a, np.ndarray):
                if not a.flags.f_contiguous:
                    raise ValueError("array args must be Fortran-contiguous")
                cargs.append(ctypes.c_void_p(a.ctypes.data)); temps.append(a)
            elif isinstance(a, (bool, np.bool_)):
                t = ctypes.c_int(int(a)); cargs.append(ctypes.byref(t)); temps.append(t)
            elif isinstance(a, (int, np.integer)):
                t = ctypes.c_int(int(a)); cargs.append(ctypes.byref(t)); temps.append(t)
            elif isinstance(a, (float, np.floating)):
                t = ctypes.c_double(float(a)); cargs.append(ctypes.byref(t)); temps.append(t)
            elif isinstance(a, ctypes._SimpleCData):
                cargs.append(ctypes.byref(a)); temps.append(a)
            else:
                raise TypeError(type(a))
        r = fn(*cargs)
        outs = [t.value if isinstance(t, ctypes._SimpleCData) else t for t in temps]
        return r, outs

    def __del__(self):
        if getattr(self, "_tmp", None):
            shutil.rmtree(self._tmp, ignore_errors=True)
