"""Validation tests for Kurucz Fortran 77 DATA statement parser against compiled oracle shared libraries."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from elftools.elf.elffile import ELFFile

from atlas_torch.kurucz_data import load_tables, parse_data, table
from atlas_torch.testing.fortran_oracle import read_layout

ATLAS_EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))

LIBRARY_PAIRS = [
    ("libatlas7v.so", ATLAS_EXT / "builds/src/atlas7v.for"),
    ("libatlas9.so", ATLAS_EXT / "builds/src/atlas9mem.patched.for"),
    ("libkappa9.so", ATLAS_EXT / "builds/src/kappa9.for"),
    ("libxnfdf.so", ATLAS_EXT / "builds/src/xnfdf.for"),
    ("libsynthe.so", ATLAS_EXT / "builds/src/synthe.patched.for"),
    ("libdfsynthe.so", ATLAS_EXT / "builds/src/dfsynthe.for"),
]


def _get_static_bytes(elf: ELFFile, addr: int, nbytes: int) -> bytes | None:
    """Read link-time initialized bytes from the ELF file for a given virtual address."""
    for sec in elf.iter_sections():
        h = sec.header
        if h.sh_addr <= addr < h.sh_addr + h.sh_size:
            if sec.name == ".bss" or h.sh_type == "SHT_NOBITS":
                return bytes(nbytes)
            offset = h.sh_offset + (addr - h.sh_addr)
            sec.stream.seek(offset)
            return sec.stream.read(nbytes)
    return None


@pytest.mark.parametrize("so_name, for_path", LIBRARY_PAIRS)
def test_kurucz_data_library(so_name: str, for_path: Path):
    if not ATLAS_EXT.exists() or not for_path.exists():
        pytest.skip(f"{ATLAS_EXT} is missing or source file {for_path} does not exist")

    so_path = ATLAS_EXT / "oracle" / so_name
    if not so_path.exists():
        pytest.skip(f"Oracle library {so_path} does not exist")

    # 1. Parse DATA statements
    parsed_tables = parse_data(for_path)
    assert len(parsed_tables) > 0, f"No DATA variables parsed from {for_path}"

    # 2. Read layout from DWARF
    commons, units, syms = read_layout(str(so_path))

    with open(so_path, "rb") as fh:
        elf = ELFFile(fh)

        # 3. Bitwise verification of all parsed DATA variables
        verified_count = 0
        mismatches: list[str] = []

        for (uname, vname), arr in parsed_tables.items():
            var_info = None
            if uname in units and vname in units[uname].statics:
                var_info = units[uname].statics[vname]
            elif uname in units:
                for bname in units[uname].commons:
                    if bname in commons and vname in commons[bname]:
                        var_info = commons[bname][vname]
                        break
            if var_info is None:
                for bname, blk in commons.items():
                    if vname in blk:
                        var_info = blk[vname]
                        break

            if var_info is not None and var_info.addr is not None:
                so_raw = _get_static_bytes(elf, var_info.addr, var_info.nbytes)
                assert so_raw is not None, f"Could not read bytes for {uname}.{vname} at 0x{var_info.addr:x}"

                parsed_raw = arr.tobytes(order="F")
                if parsed_raw == so_raw:
                    verified_count += 1
                else:
                    mismatches.append(
                        f"Mismatch for {uname}.{vname}: parsed shape {arr.shape} dtype {arr.dtype} "
                        f"!= so shape {var_info.shape} dtype {var_info.dtype}"
                    )

        assert len(mismatches) == 0, "\n".join(mismatches[:10])

        # 4. Coverage reporting
        nonzero_statics = 0
        missed_statics: list[str] = []

        for uname, u in units.items():
            for vname, v in u.statics.items():
                if "$" in vname or vname.startswith((".", "_")):
                    continue
                raw = _get_static_bytes(elf, v.addr, v.nbytes)
                if raw and any(raw):
                    nonzero_statics += 1
                    if (uname, vname) not in parsed_tables:
                        missed_statics.append(f"{uname}.{vname} (shape={v.shape}, dtype={v.dtype})")

        print(
            f"\n[{so_name}] Parsed: {len(parsed_tables)}, Verified: {verified_count}, "
            f"Mismatches: {len(mismatches)}, Nonzero statics: {nonzero_statics}, "
            f"Misses (EQUIVALENCE aliases/uninitialized): {len(missed_statics)}"
        )
        if missed_statics:
            print("  Sample misses:", missed_statics[:5])


def test_load_tables_and_cache(tmp_path):
    if not ATLAS_EXT.exists():
        pytest.skip(f"{ATLAS_EXT} is missing")

    src = ATLAS_EXT / "builds/src/atlas7v.for"
    if not src.exists():
        pytest.skip(f"{src} is missing")

    # First load (populates cache)
    data1 = load_tables(src)
    assert len(data1) > 0

    # Second load (loads from cache)
    data2 = load_tables(src)
    assert len(data1) == len(data2)
    for k in data1:
        assert data1[k].tobytes() == data2[k].tobytes() and data1[k].shape == data2[k].shape


def test_table_convenience():
    if not ATLAS_EXT.exists():
        pytest.skip(f"{ATLAS_EXT} is missing")

    # Test convenience lookup for atlas7v.for putout.on
    on_arr = table("atlas7v.for", "putout", "on")
    assert isinstance(on_arr, np.ndarray)
    assert on_arr.shape == ()
    assert on_arr.dtype == np.float64
