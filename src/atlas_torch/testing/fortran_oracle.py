"""ctypes oracles over the original Fortran routines (PyTLAS pattern, generalised).

A physics library (e.g. atlas7v.for) is compiled to a shared object with -g -fPIC and the stock
compiler/flags (tools/oracle/build_libs.sh). This module reads the exact memory layout of every COMMON
block and every SAVEd/DATA local from the DWARF debug info, maps them as writable numpy views, and calls
any routine with Fortran by-reference conventions. Each Oracle loads a private copy of the .so (dlopen of
the same path would share one image), so SAVEd state starts from the load-time image; `reset()` loads a new
copy, `restore()` (Oracle(..., snapshot=True)) rewinds the current copy's writable data (.data/.bss: COMMONs, SAVEd locals) to its
load-time image without reloading.

Private copies live under $ATLAS_EXT/scratch/oracle_tmp/oracle_<pid>_* (on disk, not the /tmp tmpfs); they
are removed when the Oracle is collected or at interpreter exit (weakref.finalize), and directories left
behind by processes that no longer exist are swept when the first Oracle of a process is created.

Math-library binding (ifx builds: xnfdf, dfsynthe, kappa9). The stock ifx executables list libimf.so before
libm.so.6, so their LOG10F/LOG10/LOG/EXP/POW bind to libimf (LD_DEBUG=bindings on builds/ifx-O0/bin/xnfdf.exe).
A plain dlopen from Python binds the oracle's copies to glibc's libm instead (already in the global scope).
Oracle(..., deepbind=True) opens the library with RTLD_DEEPBIND, so its own DT_NEEDED list (libimf first) wins,
as in the stock program. The default stays False: most ifx-oracle tests were calibrated against the
glibc-bound oracle (DEEPBIND also rebinds Intel-runtime internals, e.g. formatted-READ conversions), and
switching them over is their owners' call. Example: glibc 2.39 log10f is not correctly rounded and differs
from libimf's in ~3% of arguments, which moves PFIRON by 2 float ulps (tests/oracle/test_xnfdf_pfsaha.py).
"""
from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
import time
import weakref
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


def oracle_tmp_root() -> Path:
    """$ATLAS_EXT/scratch/oracle_tmp: where Oracle keeps its private .so copies."""
    root = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "scratch" / "oracle_tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale(min_age_s: float = 60.0) -> int:
    """Remove oracle_<pid>_* copies whose process no longer exists. Returns the number removed."""
    n = 0
    now = time.time()
    for d in oracle_tmp_root().glob("oracle_*_*"):
        try:
            pid = int(d.name.split("_")[1])
            if _pid_alive(pid) or now - d.stat().st_mtime < min_age_s:
                continue
        except (ValueError, OSError):
            continue
        shutil.rmtree(d, ignore_errors=True)
        n += 1
    return n


_swept = False


@lru_cache(maxsize=16)
def _elf_info(so_path: str):
    """(needed sonames, [(start, end)] writable non-RELRO link-time ranges) of a shared object."""
    with open(so_path, "rb") as fh:
        elf = ELFFile(fh)
        needed = []
        dyn = elf.get_section_by_name(".dynamic")
        if dyn is not None:
            needed = [t.needed for t in dyn.iter_tags() if t.entry.d_tag == "DT_NEEDED"]
        relro = None
        loads = []
        for seg in elf.iter_segments():
            if seg["p_type"] == "PT_GNU_RELRO":
                relro = (seg["p_vaddr"], seg["p_vaddr"] + seg["p_memsz"])
            elif seg["p_type"] == "PT_LOAD" and seg["p_flags"] & 2:
                loads.append((seg["p_vaddr"], seg["p_vaddr"] + seg["p_memsz"]))
    ranges = []
    for a, b in loads:
        if relro and relro[0] <= a < relro[1]:
            a = min(b, relro[1])          # RELRO part is mprotect()ed read-only after relocation
        if b > a:
            ranges.append((a, b))
    return tuple(needed), tuple(ranges)


class Oracle:
    """Load a private copy of a Fortran shared library and expose COMMONs, SAVEd locals and routines."""

    def __init__(self, so_path: str | Path, deepbind: bool = False, snapshot: bool = False):
        global _swept
        self.so_path = str(Path(so_path).resolve())
        self.commons_layout, self.units, self.syms = read_layout(self.so_path)
        self.needed, self._rw = _elf_info(self.so_path)
        self.deepbind = deepbind            # True: bind like the stock executable (see module docstring)
        self.snapshot = snapshot            # keep the load-time image of .data/.bss for restore() (<= 31 MB)
        self._tmp = None
        self._fin = None
        if not _swept:
            _swept = True
            sweep_stale()
        self.reset()

    def reset(self):
        """Load a new private copy of the library (fresh load-time SAVE state, new addresses)."""
        if self._fin is not None:
            self._fin()                    # remove the previous copy (the old mapping stays valid)
        self._tmp = tempfile.mkdtemp(prefix=f"oracle_{os.getpid()}_", dir=oracle_tmp_root())
        # finalize binds shutil.rmtree now and runs at collection or via atexit (before module teardown),
        # unlike the former __del__, which ran after shutil had been torn down at interpreter exit
        self._fin = weakref.finalize(self, shutil.rmtree, self._tmp, True)
        p = os.path.join(self._tmp, os.path.basename(self.so_path))
        shutil.copy(self.so_path, p)
        mode = os.RTLD_LOCAL | (os.RTLD_DEEPBIND if self.deepbind else 0)
        self.lib = ctypes.CDLL(p, mode=mode)
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
        self._image = None
        if self.snapshot:
            self._image = [(a, (ctypes.c_char * (b - a)).from_address(self.base + a).raw) for a, b in self._rw]

    def restore(self):
        """Rewind all writable data of this copy (COMMONs, SAVEd/DATA locals) to the load-time image."""
        if self._image is None:
            raise RuntimeError("Oracle(..., snapshot=True) is required for restore()")
        for a, img in self._image:
            ctypes.memmove(self.base + a, img, len(img))
        return self

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

    def close(self):
        """Remove the private copy now (the library stays mapped until the process exits)."""
        if self._fin is not None:
            self._fin()
