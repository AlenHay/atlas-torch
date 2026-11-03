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


def compare_fort12_records(actual_recs: list[bytes], ref_recs: list[bytes], rtol: float = 5e-4) -> None:
    """Compare fort.12 records with exact match on integer/packed fields and rtol on floats."""
    assert len(actual_recs) == len(ref_recs), f"Record count mismatch: {len(actual_recs)} vs {len(ref_recs)}"
    
    # '<ifif4f': (nbuff, congf, nelion, elo, gamrf, gamsf, gamwf, alpha)
    fmt = "<ifif4f"
    for i, (act, ref) in enumerate(zip(actual_recs, ref_recs)):
        a = struct.unpack(fmt, act)
        r = struct.unpack(fmt, ref)
        # nbuff and nelion must be exact integers
        assert a[0] == r[0], f"Record {i} nbuff mismatch: act={a[0]} ref={r[0]}"
        assert a[2] == r[2], f"Record {i} nelion mismatch: act={a[2]} ref={r[2]}"
        # alpha must be exact
        assert a[7] == r[7], f"Record {i} alpha mismatch: act={a[7]} ref={r[7]}"
        # elo must match closely or exact
        assert np.isclose(a[3], r[3], rtol=1e-5, atol=1e-5), f"Record {i} elo mismatch: act={a[3]} ref={r[3]}"
        # Float fields: congf, gamrf, gamsf, gamwf
        for f_idx, name in ((1, "congf"), (4, "gamrf"), (5, "gamsf"), (6, "gamwf")):
            diff = abs(a[f_idx] - r[f_idx])
            denom = max(abs(a[f_idx]), abs(r[f_idx]), 1e-35)
            rel = diff / denom
            assert rel <= rtol or np.isclose(a[f_idx], r[f_idx], rtol=rtol, atol=1e-35), (
                f"Record {i} field {name} rel diff {rel:.2e} > {rtol:.2e}: act={a[f_idx]} ref={r[f_idx]}"
            )


def compare_fort19_records(actual_recs: list[bytes], ref_recs: list[bytes], rtol: float = 5e-4) -> None:
    """Compare fort.19 records with exact match on integer/double fields and rtol on floats."""
    assert len(actual_recs) == len(ref_recs), f"fort.19 count mismatch: {len(actual_recs)} vs {len(ref_recs)}"
    
    # '<d2f6i4f2i': (wlvac, elo, cgf, nblo, nbup, nelion, type, ncon, nelionx, gammar, gammas, gammaw, alpha, nbuff, lim)
    fmt = "<d2f6i4f2i"
    for i, (act, ref) in enumerate(zip(actual_recs, ref_recs)):
        a = struct.unpack(fmt, act)
        r = struct.unpack(fmt, ref)
        # wlvac (d) exact
        assert a[0] == r[0], f"Record {i} wlvac mismatch: {a[0]} vs {r[0]}"
        # elo (f) exact
        assert a[1] == r[1], f"Record {i} elo mismatch: {a[1]} vs {r[1]}"
        # cgf (f) float comparison
        assert np.isclose(a[2], r[2], rtol=rtol, atol=1e-35), f"Record {i} cgf mismatch: {a[2]} vs {r[2]}"
        # integers exact: nblo, nbup, nelion, type, ncon, nelionx
        assert a[3:9] == r[3:9], f"Record {i} integer fields mismatch: {a[3:9]} vs {r[3:9]}"
        # alpha exact
        assert a[12] == r[12], f"Record {i} alpha mismatch: {a[12]} vs {r[12]}"
        # nbuff, lim exact
        assert a[13] == r[13], f"Record {i} nbuff mismatch: {a[13]} vs {r[13]}"
        assert a[14] == r[14], f"Record {i} lim mismatch: {a[14]} vs {r[14]}"
        # gammar, gammas, gammaw
        for f_idx, name in ((9, "gammar"), (10, "gammas"), (11, "gammaw")):
            diff = abs(a[f_idx] - r[f_idx])
            denom = max(abs(a[f_idx]), abs(r[f_idx]), 1e-35)
            rel = diff / denom
            assert rel <= rtol or np.isclose(a[f_idx], r[f_idx], rtol=rtol, atol=1e-35), (
                f"Record {i} field {name} rel diff {rel:.2e} > {rtol:.2e}: act={a[f_idx]} ref={r[f_idx]}"
            )


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
        
        # Last 224 lines of fort.12 in sun_500_510_stock are rh2ofast
        ref_h2o = read_records(REF_500_510 / "pre_synthe/fort.12")[-224:]
        actual_h2o = read_records(Path(td) / "fort.12")
        # Stock rh2ofast (gfortran -O3) builds TABLOG with glibc's vectorised libmvec powf; the port uses scalar
        # powf (= the gfortran -O0 noise variant), which differs by 1 ulp in some entries. So: every field
        # byte-identical except the three TABLOG-derived REAL*4 fields (congf, gamsf, gamwf), which may differ
        # by at most 1 ulp in TABLOG, i.e. <= 2 ulp after congf = .01502*TABLOG/FREQ*XISO (three REAL*4 roundings).
        # (With the -O3 TABLOG values these records were byte-identical: TABLOG is the only difference.)
        import numpy as np
        assert len(actual_h2o) == len(ref_h2o)
        dt = np.dtype([("nbuff", "<i4"), ("congf", "<f4"), ("code", "<i4"), ("elo", "<f4"), ("gamrf", "<f4"),
                       ("gamsf", "<f4"), ("gamwf", "<f4"), ("alpha", "<f4")])
        a = np.frombuffer(b"".join(actual_h2o), dt)
        r = np.frombuffer(b"".join(ref_h2o), dt)
        for f in ("nbuff", "code", "elo", "gamrf", "alpha"):
            assert (a[f] == r[f]).all(), f
        for f in ("congf", "gamsf", "gamwf"):
            ulp = np.abs(a[f].view(np.int32).astype(np.int64) - r[f].view(np.int32).astype(np.int64))
            assert ulp.max() <= 2, (f, int(ulp.max()))


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
