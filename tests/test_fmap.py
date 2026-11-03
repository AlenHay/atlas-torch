"""Tests for tools/fmap.py: Fortran 77 source mapping and variant analysis."""

import json
from pathlib import Path
import pytest
from tools.fmap import (
    parse_file_to_units,
    analyze_units,
    compare_variants,
    read_fixed_statements,
)


def test_continuation_lines(tmp_path: Path):
    """Test fixed-form continuation handling, tab continuation, and comment interleaving."""
    src = tmp_path / "test_cont.for"
    src.write_text(
        "      SUBROUTINE CONTTEST(X, Y)\n"
        "C     A comment before continuation\n"
        "      X = 1.0 +\n"
        "     1    2.0 +\n"
        "C     Interleaved comment\n"
        "     2    3.0\n"
        "C     Test tab continuation\n"
        "      Y = 10.0 +\n"
        "\t1 20.0\n"
        "      RETURN\n"
        "      END\n"
    )

    units = parse_file_to_units(src)
    assert len(units) == 1
    u = units[0]
    assert u.name == "CONTTEST"
    assert u.kind == "SUBROUTINE"
    assert u.dummy_args == ["X", "Y"]

    # Check statements
    # Stmt 1: SUBROUTINE CONTTEST(X, Y)
    # Stmt 2: X = 1.0 + 2.0 + 3.0 (spans 3 code lines: lines 3, 4, 6)
    # Stmt 3: Y = 10.0 + 20.0 (spans 2 code lines: lines 8, 9)
    assert len(u.statements) == 5
    s_x = u.statements[1]
    assert "1.0" in s_x.text and "2.0" in s_x.text and "3.0" in s_x.text
    assert s_x.lines == [3, 4, 6]

    s_y = u.statements[2]
    assert "10.0" in s_y.text and "20.0" in s_y.text
    assert s_y.lines == [8, 9]


def test_common_read_write_detection(tmp_path: Path):
    """Test COMMON variable classification for READ, WRITTEN, and maybe-written."""
    src = tmp_path / "test_common.for"
    src.write_text(
        "      SUBROUTINE COMMTEST\n"
        "      COMMON /BLOCK1/ A, B(10), C, D, E, F, G\n"
        "      A = 1.0\n"
        "      B(C) = 2.0\n"
        "      READ(5, *) D\n"
        "      DO 10 E = 1, 5\n"
        "         CALL WORK(F)\n"
        "   10 CONTINUE\n"
        "      IF (A .GT. 0.0) WRITE(6, *) 'VAL'\n"
        "      RETURN\n"
        "      END\n"
    )

    units = parse_file_to_units(src)
    res = analyze_units(units)
    u_data = res["units"]["COMMTEST"]

    c_vars = u_data["common_vars"]
    # A is assigned (written) and checked in IF (read)
    assert c_vars["A"]["written"] is True
    assert c_vars["A"]["read"] is True

    # B is array assignment target (written)
    assert c_vars["B"]["written"] is True
    assert c_vars["B"]["read"] is False

    # C is array subscript in B(C) = 2.0 (read)
    assert c_vars["C"]["read"] is True
    assert c_vars["C"]["written"] is False

    # D is target of READ (written)
    assert c_vars["D"]["written"] is True
    assert c_vars["D"]["read"] is False

    # E is DO loop variable (written)
    assert c_vars["E"]["written"] is True

    # F is passed to CALL WORK(F) (maybe_written)
    assert c_vars["F"]["maybe_written"] is True
    assert c_vars["F"]["written"] is False

    # G is unused
    assert c_vars["G"]["read"] is False
    assert c_vars["G"]["written"] is False
    assert c_vars["G"]["maybe_written"] is False


def test_data_detection(tmp_path: Path):
    """Test extraction of DATA-initialised local variables (excluding common vars)."""
    src = tmp_path / "test_data.for"
    src.write_text(
        "      SUBROUTINE DATATEST\n"
        "      COMMON /MYBLK/ CVAL\n"
        "      DIMENSION Z(5)\n"
        "      DATA X, Y / 1.0, 2.0 /\n"
        "      DATA (Z(I), I=1,5) / 5*0.0 /\n"
        "      DATA CVAL / 99.0 /\n"
        "      RETURN\n"
        "      END\n"
    )

    units = parse_file_to_units(src)
    res = analyze_units(units)
    u_data = res["units"]["DATATEST"]

    # Local variables X, Y, Z should be in data_vars; CVAL is in COMMON so excluded
    assert "X" in u_data["data_vars"]
    assert "Y" in u_data["data_vars"]
    assert "Z" in u_data["data_vars"]
    assert "CVAL" not in u_data["data_vars"]


def test_typed_function_headers(tmp_path: Path):
    """Test parsing typed function headers and program unit kinds."""
    src = tmp_path / "test_units.for"
    src.write_text(
        "C     Implicit main program before first header\n"
        "      X = 1.0\n"
        "      CALL S1(X)\n"
        "      END\n"
        "      REAL*8 FUNCTION F1(A, B)\n"
        "      F1 = A + B\n"
        "      END\n"
        "      DOUBLE PRECISION FUNCTION F2(C)\n"
        "      F2 = C * 2.0\n"
        "      END\n"
        "      INTEGER FUNCTION F3()\n"
        "      F3 = 42\n"
        "      END\n"
        "      CHARACTER*(*) FUNCTION F4(S)\n"
        "      F4 = S\n"
        "      END\n"
        "      SUBROUTINE S1(Q)\n"
        "      RETURN\n"
        "      END\n"
        "      BLOCK DATA BD\n"
        "      COMMON /B/ V\n"
        "      DATA V / 1.0 /\n"
        "      END\n"
    )

    units = parse_file_to_units(src)
    res = analyze_units(units)
    u_dict = res["units"]

    # Implicit main
    assert "MAIN" in u_dict
    assert u_dict["MAIN"]["kind"] == "PROGRAM"
    assert "S1" in u_dict["MAIN"]["calls"]

    # Functions
    assert u_dict["F1"]["kind"] == "FUNCTION"
    assert u_dict["F1"]["dummy_args"] == ["A", "B"]

    assert u_dict["F2"]["kind"] == "FUNCTION"
    assert u_dict["F2"]["dummy_args"] == ["C"]

    assert u_dict["F3"]["kind"] == "FUNCTION"

    assert u_dict["F4"]["kind"] == "FUNCTION"
    assert u_dict["F4"]["dummy_args"] == ["S"]

    # Subroutine & Block Data
    assert u_dict["S1"]["kind"] == "SUBROUTINE"
    assert u_dict["BD"]["kind"] == "BLOCK DATA"


def test_cosmetic_vs_semantic_classification(tmp_path: Path):
    """Test cosmetic vs semantic classification across variants."""
    file_orig = tmp_path / "orig.for"
    file_cosm = tmp_path / "cosm.for"
    file_seman = tmp_path / "seman.for"

    file_orig.write_text(
        "      SUBROUTINE FOO(A, B)\n"
        "      PARAMETER (kw=99)\n"
        "      DIMENSION ARR(kw)\n"
        "      WRITE(6, 100) A\n"
        "  100 FORMAT(1X, F10.2)\n"
        "      ARR(1) = A + B\n"
        "      RETURN\n"
        "      END\n"
    )

    # Cosm differs only in PARAMETER (kw=72), DIMENSION, WRITE/FORMAT
    file_cosm.write_text(
        "      SUBROUTINE FOO(A, B)\n"
        "      PARAMETER (kw=72)\n"
        "      DIMENSION ARR(kw)\n"
        "      WRITE(6, 200) A, B\n"
        "  200 FORMAT(1X, 2F10.2)\n"
        "      ARR(1) = A + B\n"
        "      RETURN\n"
        "      END\n"
    )

    # Seman differs in calculation (ARR(1) = A + B * 2.0)
    file_seman.write_text(
        "      SUBROUTINE FOO(A, B)\n"
        "      PARAMETER (kw=99)\n"
        "      DIMENSION ARR(kw)\n"
        "      WRITE(6, 100) A\n"
        "  100 FORMAT(1X, F10.2)\n"
        "      ARR(1) = A + B * 2.0\n"
        "      RETURN\n"
        "      END\n"
    )

    res = compare_variants([file_orig, file_cosm, file_seman])
    pairs = res["pairs"]

    orig_cosm = next(p for p in pairs if {"orig.for", "cosm.for"} == {p["file_a"], p["file_b"]})
    assert orig_cosm["classification"] == "cosmetic"
    assert orig_cosm["changed_lines"] > 0

    orig_seman = next(p for p in pairs if {"orig.for", "seman.for"} == {p["file_a"], p["file_b"]})
    assert orig_seman["classification"] == "semantic"
    assert orig_seman["changed_lines"] > 0
    assert len(orig_seman["diff_sample"]) > 0


def test_topological_port_order(tmp_path: Path):
    """Test leaves-first topological sorting and deterministic cycle breaking."""
    src = tmp_path / "test_topo.for"
    src.write_text(
        "      SUBROUTINE ROOT\n"
        "      CALL MID\n"
        "      END\n"
        "      SUBROUTINE MID\n"
        "      CALL LEAF1\n"
        "      CALL LEAF2\n"
        "      END\n"
        "      SUBROUTINE LEAF1\n"
        "      RETURN\n"
        "      END\n"
        "      SUBROUTINE LEAF2\n"
        "      RETURN\n"
        "      END\n"
    )

    units = parse_file_to_units(src)
    res = analyze_units(units)
    order = res["order"]

    # LEAF1 and LEAF2 have no dependencies, should precede MID; MID should precede ROOT
    assert order.index("LEAF1") < order.index("MID")
    assert order.index("LEAF2") < order.index("MID")
    assert order.index("MID") < order.index("ROOT")
