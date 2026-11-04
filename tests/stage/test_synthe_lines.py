"""Acceptance and unit tests for SYNTHE line list preparation programs.

Tests Python ports:
- synbeg.py
- rgfalllines.py
- rmolecasc.py
- rh2ofast.py
- prepare_lines orchestrator

Verifies outputs against Fortran reference directories:
- /home/deck/atlas-ext/work/sun_500_510_stock/pre_synthe/
- /home/deck/atlas-ext/work/sun_380_420_stock/pre_synthe/
"""
from __future__ import annotations

import os
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from atlas_torch.io.fortran import read_records
from atlas_torch.synthe.lines import (
    prepare_lines,
    run_synbeg,
    run_rgfalllines,
    run_rmolecasc,
    run_rh2ofast,
    read_fort93,
    parse_synbeg_card,
)

pytestmark = pytest.mark.stage   # whole-program / big-data tier (tests/stage/)

EXT_DIR = Path("/home/deck/atlas-ext")
SYNTHE_DATA = EXT_DIR / "BasicATLAS/data/synthe_files"
DF_DATA = EXT_DIR / "BasicATLAS/data/dfsynthe_files"
REF_500_510 = EXT_DIR / "work/sun_500_510_stock"
REF_380_420 = EXT_DIR / "work/sun_380_420_stock"

BASICATLAS_DAT = SYNTHE_DATA / "BasicATLAS.dat"
H2OFASTFIX_BIN = DF_DATA / "h2ofastfix.bin"

MOL_NAMES = [
    "chmasseron_corrected.asc", "mgh_exomol.asc", "nh.asc", "ohupdate.asc", "sihax.asc",
    "h2.asc", "h2xx.asc", "hdxx.asc", "c2ax.asc", "c2ba.asc",
    "c2dabrookek.asc", "c2ea.asc", "cnaxbrookek.asc", "cnbxbrookek.asc", "cnxx12brooke.asc",
    "coax.asc", "coxx.asc", "sioax.asc", "sioex.asc", "sioxx.asc", "tiototo.asc"
]
MOL_LISTS = [SYNTHE_DATA / m for m in MOL_NAMES]


def _first_diff(actual_recs: list[bytes], ref_recs: list[bytes], fmt: str) -> str:
    for i, (act, ref) in enumerate(zip(actual_recs, ref_recs)):
        if act != ref:
            return f"record {i}: act={struct.unpack(fmt, act)} ref={struct.unpack(fmt, ref)}"
    return ""


def compare_fort12_records(actual_recs: list[bytes], ref_recs: list[bytes]) -> None:
    """fort.12 records (NBUFF,CONGF,NELION,ELO,GAMRF,GAMSF,GAMWF,ALPHA) must be byte-identical to the stock
    (gfortran -O3) programs: the port reproduces their REAL*4 arithmetic and libm calls (scalar powf/log10f/exp,
    libmvec powf for rh2ofast's TABLOG)."""
    assert len(actual_recs) == len(ref_recs), f"Record count mismatch: {len(actual_recs)} vs {len(ref_recs)}"
    nd = sum(a != r for a, r in zip(actual_recs, ref_recs))
    assert nd == 0, f"{nd} fort.12 records differ; first {_first_diff(actual_recs, ref_recs, '<ifif4f')}"


def compare_fort19_records(actual_recs: list[bytes], ref_recs: list[bytes]) -> None:
    """fort.19 records (WLVAC,ELO,GF,NBLO,NBUP,NELION,TYPE,NCON,NELIONX,GAMMAR,GAMMAS,GAMMAW,ALPHA,NBUFF,LIM) must
    be byte-identical to the stock programs."""
    assert len(actual_recs) == len(ref_recs), f"fort.19 count mismatch: {len(actual_recs)} vs {len(ref_recs)}"
    nd = sum(a != r for a, r in zip(actual_recs, ref_recs))
    assert nd == 0, f"{nd} fort.19 records differ; first {_first_diff(actual_recs, ref_recs, '<d2f6i4f2i')}"


@pytest.mark.stage
def test_synbeg_standalone():
    """Unit test for synbeg standalone."""
    with tempfile.TemporaryDirectory() as td:
        card = (REF_500_510 / "synbeg.in").read_text().splitlines()[0]
        p = run_synbeg(td, card=card)
        assert p["length"] == 11882
        assert p["ifvac"] == 1
        assert p["nlines"] == 0
        assert p["n19"] == 0
        assert (Path(td) / "fort.12").stat().st_size == 0
        assert (Path(td) / "fort.14").stat().st_size == 0
        assert (Path(td) / "fort.19").stat().st_size == 0
        assert (Path(td) / "fort.20").stat().st_size == 0
        assert (Path(td) / "fort.93").stat().st_size == 2856  # 2848 + 8 byte markers


@pytest.mark.stage
def test_rh2ofast_standalone():
    """Unit test for rh2ofast standalone: byte-identical records against Fortran."""
    with tempfile.TemporaryDirectory() as td:
        card = (REF_500_510 / "synbeg.in").read_text().splitlines()[0]
        run_synbeg(td, card=card)
        res = run_rh2ofast(td, H2OFASTFIX_BIN)
        assert res["n_added"] == 224
        
        # Last 224 lines of fort.12 in sun_500_510_stock are rh2ofast: byte-identical, including the
        # TABLOG-derived REAL*4 fields (TABLOG built with libmvec _ZGVbN4vv_powf like the stock -O3 executable)
        ref_h2o = read_records(REF_500_510 / "pre_synthe/fort.12")[-224:]
        actual_h2o = read_records(Path(td) / "fort.12")
        compare_fort12_records(actual_h2o, ref_h2o)


@pytest.mark.stage
def test_rmolecasc_standalone():
    """Unit test for rmolecasc standalone on chmasseron_corrected.asc."""
    with tempfile.TemporaryDirectory() as td:
        card = (REF_500_510 / "synbeg.in").read_text().splitlines()[0]
        run_synbeg(td, card=card)
        res = run_rmolecasc(td, SYNTHE_DATA / "chmasseron_corrected.asc")
        assert res["n_added"] == 1421
        ref_12 = read_records(REF_500_510 / "pre_synthe/fort.12")[4547 : 4547 + 1421]
        actual_12 = read_records(Path(td) / "fort.12")
        compare_fort12_records(actual_12, ref_12)


@pytest.mark.stage
def test_rgfalllines_standalone():
    """Unit test for rgfalllines standalone on BasicATLAS.dat."""
    with tempfile.TemporaryDirectory() as td:
        card = (REF_500_510 / "synbeg.in").read_text().splitlines()[0]
        run_synbeg(td, card=card)
        res = run_rgfalllines(td, BASICATLAS_DAT)
        assert res["n14"] == 4547
        assert res["n19"] == 3
        ref_12 = read_records(REF_500_510 / "pre_synthe/fort.12")[:4547]
        actual_12 = read_records(Path(td) / "fort.12")
        compare_fort12_records(actual_12, ref_12)
        
        ref_19 = read_records(REF_500_510 / "pre_synthe/fort.19")
        actual_19 = read_records(Path(td) / "fort.19")
        compare_fort19_records(actual_19, ref_19)


@pytest.mark.stage
def test_prepare_lines_sun_500_510():
    """Full end-to-end acceptance test for 500-510 nm window."""
    ref_dir = REF_500_510 / "pre_synthe"
    card = (REF_500_510 / "synbeg.in").read_text().splitlines()[0]
    parsed = parse_synbeg_card(card)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        res = prepare_lines(
            workdir=td_path,
            wlbeg=parsed["wlbeg"],
            wlend=parsed["wlend"],
            resolu=parsed["resolu"],
            vturb=parsed["turbv"],
            linelist=BASICATLAS_DAT,
            mol_lists=MOL_LISTS,
            h2o_path=H2OFASTFIX_BIN,
            airorvac=parsed["airvac"],
            ifnlte=parsed["ifnlte"],
            linout=parsed["linout"],
            cutoff=parsed["cutoff"],
            ifpred=parsed["ifpred"],
            nread=parsed["nread"],
        )
        elapsed = time.time() - t0
        print(f"\n[500-510 nm] completed in {elapsed:.2f} s (Fortran reference ~40 s)")

        # fort.93, fort.14, fort.20 must be 100% BYTE-IDENTICAL
        assert (td_path / "fort.93").read_bytes() == (ref_dir / "fort.93").read_bytes(), "fort.93 not byte-identical"
        assert (td_path / "fort.14").read_bytes() == (ref_dir / "fort.14").read_bytes(), "fort.14 not byte-identical"
        assert (td_path / "fort.20").read_bytes() == (ref_dir / "fort.20").read_bytes(), "fort.20 not byte-identical"

        # Compare decoded records for fort.12 and fort.19
        actual_12 = read_records(td_path / "fort.12")
        ref_12 = read_records(ref_dir / "fort.12")
        compare_fort12_records(actual_12, ref_12)

        actual_19 = read_records(td_path / "fort.19")
        ref_19 = read_records(ref_dir / "fort.19")
        compare_fort19_records(actual_19, ref_19)


@pytest.mark.stage
def test_prepare_lines_sun_380_420():
    """Full end-to-end acceptance test for 380-420 nm window."""
    ref_dir = REF_380_420 / "pre_synthe"
    card = (REF_380_420 / "synbeg.in").read_text().splitlines()[0]
    parsed = parse_synbeg_card(card)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        res = prepare_lines(
            workdir=td_path,
            wlbeg=parsed["wlbeg"],
            wlend=parsed["wlend"],
            resolu=parsed["resolu"],
            vturb=parsed["turbv"],
            linelist=BASICATLAS_DAT,
            mol_lists=MOL_LISTS,
            h2o_path=H2OFASTFIX_BIN,
            airorvac=parsed["airvac"],
            ifnlte=parsed["ifnlte"],
            linout=parsed["linout"],
            cutoff=parsed["cutoff"],
            ifpred=parsed["ifpred"],
            nread=parsed["nread"],
        )
        elapsed = time.time() - t0
        print(f"\n[380-420 nm] completed in {elapsed:.2f} s")

        # fort.93, fort.14, fort.20 must be 100% BYTE-IDENTICAL
        assert (td_path / "fort.93").read_bytes() == (ref_dir / "fort.93").read_bytes(), "fort.93 not byte-identical"
        assert (td_path / "fort.14").read_bytes() == (ref_dir / "fort.14").read_bytes(), "fort.14 not byte-identical"
        assert (td_path / "fort.20").read_bytes() == (ref_dir / "fort.20").read_bytes(), "fort.20 not byte-identical"

        # Compare decoded records for fort.12 and fort.19
        actual_12 = read_records(td_path / "fort.12")
        ref_12 = read_records(ref_dir / "fort.12")
        compare_fort12_records(actual_12, ref_12)

        actual_19 = read_records(td_path / "fort.19")
        ref_19 = read_records(ref_dir / "fort.19")
        compare_fort19_records(actual_19, ref_19)
