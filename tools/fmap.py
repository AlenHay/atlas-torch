#!/usr/bin/env python3
"""fmap: Fortran 77 source mapping and variant analysis tool for Kurucz codes.

Analyzes fixed-form Fortran 77 sources, extracting per-unit metadata, call graphs,
COMMON block usage, DATA variables, and leaf-first topological port order.
Also compares routine variants across source files to classify differences
as identical, cosmetic, or semantic.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Preprocessing and Statement Joining
# ---------------------------------------------------------------------------

def strip_inline_comments(s: str) -> str:
    """Strip inline '!' comments outside single- or double-quoted strings."""
    in_quote: Optional[str] = None
    res: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if in_quote:
            res.append(c)
            if c == in_quote:
                # Handle escaped quotes '' or ""
                if i + 1 < n and s[i + 1] == in_quote:
                    res.append(s[i + 1])
                    i += 1
                else:
                    in_quote = None
        else:
            if c in ("'", '"'):
                in_quote = c
                res.append(c)
            elif c == "!":
                break
            else:
                res.append(c)
        i += 1
    return "".join(res)


def mask_strings(s: str) -> str:
    """Replace content inside single and double quotes with spaces, preserving length."""
    in_quote: Optional[str] = None
    res: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if in_quote:
            if c == in_quote:
                if i + 1 < n and s[i + 1] == in_quote:
                    res.append("  ")
                    i += 2
                    continue
                else:
                    in_quote = None
                    res.append(" ")
            else:
                res.append(" ")
        else:
            if c in ("'", '"'):
                in_quote = c
                res.append(" ")
            else:
                res.append(c)
        i += 1
    return "".join(res)


def mask_holleriths(s: str) -> str:
    """Mask Hollerith constants like 4HLTE or 10HITERATION with spaces."""
    def repl(m: re.Match[str]) -> str:
        count = int(m.group(1))
        return " " * (len(m.group(1)) + 1 + count)

    return re.sub(r"\b(\d+)H", repl, s)


class FortranStatement:
    def __init__(self, label: Optional[str], text: str, lines: List[int]):
        self.label = label
        self.text = text
        self.lines = lines
        self.is_executable = False

    def __repr__(self) -> str:
        lbl = f"{self.label}: " if self.label else ""
        return f"<Stmt lines={self.lines[0]}-{self.lines[-1]} {lbl}{self.text}>"


def read_fixed_statements(lines: List[str]) -> List[FortranStatement]:
    """Parse fixed-form Fortran lines, joining continuations into logical statements.
    
    Fixed-form rules:
    - Col 1 in ('C', 'c', '*', '!') or empty/whitespace -> comment line
    - Col 6 non-blank and non-zero -> continuation line
    - Cols 7-72 -> statement text (truncate beyond 72)
    - Cols 1-5 -> label
    - Leading tab -> jumps to col 7, except if next char is 1-9 (continuation)
    - Strip inline '!' outside quoted strings
    """
    statements: List[FortranStatement] = []
    curr_label: Optional[str] = None
    curr_parts: List[str] = []
    curr_lines: List[int] = []

    def flush():
        nonlocal curr_label, curr_parts, curr_lines
        if curr_lines:
            joined = curr_parts[0].rstrip()
            for p in curr_parts[1:]:
                joined = joined.rstrip() + p
            stmt_text = joined.strip()
            if stmt_text:
                statements.append(FortranStatement(curr_label, stmt_text, list(curr_lines)))
            curr_label = None
            curr_parts = []
            curr_lines = []

    for line_idx, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        c1 = line[0]
        if c1 in ("C", "c", "*", "!"):
            continue

        # Tab handling
        if line.startswith("\t"):
            if len(line) > 1 and line[1] in "123456789":
                is_cont = True
                label = ""
                text = line[2:]
            else:
                is_cont = False
                label = ""
                text = line[1:]
        elif 0 < line.find("\t") <= 5:
            tab_idx = line.find("\t")
            label = line[:tab_idx].strip()
            if len(line) > tab_idx + 1 and line[tab_idx + 1] in "123456789":
                is_cont = True
                text = line[tab_idx + 2:]
            else:
                is_cont = False
                text = line[tab_idx + 1:]
        else:
            label = line[:5].strip()
            col6 = line[5:6] if len(line) > 5 else " "
            is_cont = (col6 != " " and col6 != "0")
            text = line[6:]

        # Truncate beyond column 72 (66 characters from column 7)
        text = text[:66]
        # Strip inline '!' outside strings
        text = strip_inline_comments(text)

        if is_cont:
            if curr_lines:
                curr_parts.append(text)
                curr_lines.append(line_idx)
            else:
                curr_label = label or None
                curr_parts = [text]
                curr_lines = [line_idx]
        else:
            flush()
            curr_label = label or None
            curr_parts = [text]
            curr_lines = [line_idx]

    flush()
    return statements


# ---------------------------------------------------------------------------
# Program Unit Header & Statement Classification
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(
    r"^(?:"
    r"(?:(?:REAL(?:\*\d+)?|DOUBLE\s+PRECISION|INTEGER(?:\*\d+)?|LOGICAL(?:\*\d+)?|"
    r"CHARACTER(?:\*(?:\*|\d+|\([^\)]*\)))?|COMPLEX(?:\*\d+)?|DOUBLE\s+COMPLEX|BYTE)\s+)?"
    r"FUNCTION\s+([A-Za-z0-9_]+)\s*(?:\((.*?)\))?"
    r"|SUBROUTINE\s+([A-Za-z0-9_]+)\s*(?:\((.*?)\))?"
    r"|PROGRAM\s+([A-Za-z0-9_]+)"
    r"|BLOCK\s*DATA(?:\s+([A-Za-z0-9_]+))?"
    r")$",
    re.IGNORECASE,
)

END_RE = re.compile(
    r"^END(?:\s+(?:SUBROUTINE|FUNCTION|PROGRAM|BLOCK\s*DATA)(?:\s+[A-Za-z0-9_]+)?)?$",
    re.IGNORECASE,
)

NON_EXEC_KW = {
    "PROGRAM", "SUBROUTINE", "FUNCTION", "BLOCK DATA", "BLOCKDATA", "ENTRY",
    "DIMENSION", "COMMON", "EQUIVALENCE", "PARAMETER", "IMPLICIT", "SAVE",
    "EXTERNAL", "INTRINSIC", "DATA", "FORMAT",
}

TYPE_KEYWORDS = {
    "REAL", "INTEGER", "LOGICAL", "CHARACTER", "DOUBLE PRECISION",
    "COMPLEX", "DOUBLE COMPLEX", "BYTE",
}


def find_depth0_char(s: str, char: str) -> int:
    """Find the index of char at parenthesis depth 0, ignoring strings."""
    depth = 0
    in_quote: Optional[str] = None
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if in_quote:
            if c == in_quote:
                if i + 1 < n and s[i + 1] == in_quote:
                    i += 2
                    continue
                else:
                    in_quote = None
        else:
            if c in ("'", '"'):
                in_quote = c
            elif c == "(":
                depth += 1
            elif c == ")":
                if depth > 0:
                    depth -= 1
            elif c == char and depth == 0:
                return i
        i += 1
    return -1


def is_non_executable_stmt(text: str) -> bool:
    """Determine whether a statement is non-executable (specification, data, format, etc.)."""
    s = text.strip()
    u = s.upper()

    # Check headers
    if HEADER_RE.match(s):
        return True

    # Check format and data
    if re.match(r"^\d*\s*FORMAT\s*\(", u):
        return True
    if re.match(r"^DATA\b", u):
        return True

    # Check common, dimension, parameter, implicit, equivalence, save
    for kw in ("DIMENSION", "COMMON", "EQUIVALENCE", "PARAMETER", "IMPLICIT", "EXTERNAL", "INTRINSIC"):
        if re.match(rf"^{kw}\b", u):
            return True

    # SAVE statement vs SAVE = ... assignment
    if re.match(r"^SAVE\b", u):
        eq_pos = find_depth0_char(s, "=")
        if eq_pos == -1:
            return True

    # Type declarations: REAL, INTEGER, etc.
    # Note: 'REAL = 1.0' is assignment, whereas 'REAL A, B' or 'REAL*8 X' is type declaration.
    for kw in (r"DOUBLE\s+PRECISION", r"DOUBLE\s+COMPLEX", "REAL", "INTEGER", "LOGICAL", "CHARACTER", "COMPLEX", "BYTE"):
        m = re.match(rf"^{kw}(?:\*\d+|\*\([^\)]*\))?\s+([A-Za-z0-9_]+)", u)
        if m:
            var_name = m.group(1)
            # If immediately followed by '=' without comma or dims, e.g. REAL X = 1 (F90 style) or REAL = 1
            # In F77, type statements declare variables.
            eq_pos = find_depth0_char(s, "=")
            # If there's an '=' before any comma or paren, check if it's 'REAL = ...'
            if re.match(rf"^{kw}\s*=", u):
                return False  # variable named REAL being assigned
            return True

    return False


# ---------------------------------------------------------------------------
# Program Unit Analysis
# ---------------------------------------------------------------------------

class ProgramUnit:
    def __init__(self, name: str, kind: str, dummy_args: List[str], start_line: int, end_line: int):
        self.name = name.upper()
        self.kind = kind.upper()
        self.dummy_args = [a.upper() for a in dummy_args]
        self.line_range = [start_line, end_line]
        self.executable_lines = 0
        self.code_lines = 0
        self.statements: List[FortranStatement] = []

        self.calls: List[str] = []
        self.functions: List[str] = []
        self.callees: List[str] = []

        self.common_blocks: Dict[str, List[str]] = {}
        self.common_vars: Dict[str, Dict[str, bool]] = {}
        self.common_usage: Dict[str, Dict[str, Dict[str, bool]]] = {}

        self.data_vars: List[str] = []
        self.save: List[str] = []
        self.save_statements: List[str] = []
        self.equivalence: List[List[str]] = []
        self.io_units: List[Union[int, str]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "dummy_args": self.dummy_args,
            "line_range": self.line_range,
            "executable_lines": self.executable_lines,
            "code_lines": self.code_lines,
            "calls": self.calls,
            "functions": self.functions,
            "callees": self.callees,
            "common_blocks": self.common_blocks,
            "common_vars": self.common_vars,
            "common_usage": self.common_usage,
            "data_vars": self.data_vars,
            "save": self.save,
            "equivalence": self.equivalence,
            "io_units": self.io_units,
        }


def parse_common_statement(stmt_text: str) -> Dict[str, List[str]]:
    """Parse a COMMON statement into block_name -> ordered list of variables with dimensions stripped."""
    u = stmt_text.strip()
    # Strip leading COMMON
    u = re.sub(r"^COMMON\b", "", u, flags=re.IGNORECASE).strip()
    blocks: Dict[str, List[str]] = {}
    curr_block = ""  # blank common

    # Pattern matches /block_name/ or // or variables
    i = 0
    n = len(u)
    while i < n:
        if u[i] == "/":
            end_slash = u.find("/", i + 1)
            if end_slash == -1:
                break
            bname = u[i + 1:end_slash].strip().upper()
            curr_block = bname
            i = end_slash + 1
            if curr_block not in blocks:
                blocks[curr_block] = []
        else:
            # Parse variable declarators separated by comma until next '/' or end
            next_slash = u.find("/", i)
            chunk = u[i:next_slash] if next_slash != -1 else u[i:]
            i = next_slash if next_slash != -1 else n
            # Split chunk by comma outside parentheses
            items: List[str] = []
            depth = 0
            start = 0
            for ci, c in enumerate(chunk):
                if c == "(":
                    depth += 1
                elif c == ")":
                    if depth > 0:
                        depth -= 1
                elif c == "," and depth == 0:
                    item = chunk[start:ci].strip()
                    if item:
                        items.append(item)
                    start = ci + 1
            last_item = chunk[start:].strip()
            if last_item:
                items.append(last_item)

            for it in items:
                # Strip dimensions and char lengths e.g. BHE1(kw,29) -> BHE1, STR*10(5) -> STR
                m = re.match(r"^([A-Za-z0-9_]+)", it)
                if m:
                    var = m.group(1).upper()
                    if curr_block not in blocks:
                        blocks[curr_block] = []
                    blocks[curr_block].append(var)
    return blocks


def parse_data_statement(stmt_text: str) -> List[str]:
    """Extract variable names initialized in DATA statements."""
    u = stmt_text.strip()
    u = re.sub(r"^DATA\b", "", u, flags=re.IGNORECASE).strip()
    # DATA has pairs of: var_list / val_list / [,] var_list / val_list / ...
    var_names: List[str] = []
    # Split by /.../
    parts = u.split("/")
    # Indices 0, 2, 4, ... are variable lists
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip().rstrip(",")
        if not chunk:
            continue
        # Split chunk by comma outside parens
        depth = 0
        start = 0
        items: List[str] = []
        for ci, c in enumerate(chunk):
            if c == "(":
                depth += 1
            elif c == ")":
                if depth > 0:
                    depth -= 1
            elif c == "," and depth == 0:
                item = chunk[start:ci].strip()
                if item:
                    items.append(item)
                start = ci + 1
        last = chunk[start:].strip()
        if last:
            items.append(last)

        for item in items:
            # item can be X, X(1), or (X(I), I=1,10)
            # Find identifier
            m = re.search(r"\b([A-Za-z][A-Za-z0-9_]*)\b", item)
            if m:
                var_names.append(m.group(1).upper())
    return var_names


def parse_equivalence_statement(stmt_text: str) -> List[List[str]]:
    """Parse EQUIVALENCE (a, b), (c, d) into list of lists."""
    u = stmt_text.strip()
    u = re.sub(r"^EQUIVALENCE\b", "", u, flags=re.IGNORECASE).strip()
    groups: List[List[str]] = []
    # Find all top-level (...)
    depth = 0
    start = -1
    for i, c in enumerate(u):
        if c == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and start != -1:
                group_text = u[start:i].strip()
                # Split by comma at depth 0
                g_items = []
                g_depth = 0
                g_start = 0
                for gj, gc in enumerate(group_text):
                    if gc == "(":
                        g_depth += 1
                    elif gc == ")":
                        if g_depth > 0:
                            g_depth -= 1
                    elif gc == "," and g_depth == 0:
                        it = group_text[g_start:gj].strip().upper()
                        if it:
                            g_items.append(it)
                        g_start = gj + 1
                it_last = group_text[g_start:].strip().upper()
                if it_last:
                    g_items.append(it_last)
                if g_items:
                    groups.append(g_items)
                start = -1
    return groups


def parse_save_statement(stmt_text: str) -> List[str]:
    """Parse SAVE statement items."""
    u = stmt_text.strip()
    u = re.sub(r"^SAVE\b", "", u, flags=re.IGNORECASE).strip()
    if not u:
        return ["ALL"]
    items = [x.strip().upper() for x in u.split(",") if x.strip()]
    return items or ["ALL"]


def extract_io_unit(stmt_text: str) -> Optional[Union[int, str]]:
    """Extract I/O unit from OPEN, CLOSE, READ, WRITE, REWIND statement."""
    u = stmt_text.strip()
    m = re.match(r"^(OPEN|CLOSE|READ|WRITE|REWIND)\b(?:\s*\((.*?)\)|\s+([A-Za-z0-9_]+))?", u, re.IGNORECASE)
    if not m:
        return None
    verb = m.group(1).upper()
    paren_content = m.group(2)
    bare_arg = m.group(3)

    if bare_arg:
        # e.g. REWIND 9 or REWIND ITAPE
        unit_str = bare_arg.strip()
    elif paren_content is not None:
        # Split items inside paren at depth 0
        items: List[str] = []
        depth = 0
        start = 0
        for i, c in enumerate(paren_content):
            if c == "(":
                depth += 1
            elif c == ")":
                if depth > 0:
                    depth -= 1
            elif c == "," and depth == 0:
                items.append(paren_content[start:i].strip())
                start = i + 1
        items.append(paren_content[start:].strip())

        unit_str = ""
        # Check for UNIT=
        for it in items:
            um = re.match(r"^UNIT\s*=\s*(.*)", it, re.IGNORECASE)
            if um:
                unit_str = um.group(1).strip()
                break
        if not unit_str and items:
            unit_str = items[0].strip()
    else:
        # e.g. READ *
        m_star = re.match(r"^READ\s*\*", u, re.IGNORECASE)
        if m_star:
            return "*"
        return None

    if not unit_str:
        return None

    # Clean up unit_str
    unit_str = unit_str.strip()
    if re.match(r"^\d+$", unit_str):
        return int(unit_str)
    return unit_str.upper()


# ---------------------------------------------------------------------------
# COMMON Variable Usage Analysis
# ---------------------------------------------------------------------------

def analyze_statement_for_common_usage(
    stmt_text: str,
    common_vars: Set[str],
    usage_map: Dict[str, Dict[str, bool]],
):
    """Analyze an executable statement for READ, WRITTEN, maybe-written COMMON variables."""
    s = stmt_text.strip()
    # Mask strings and Holleriths
    masked = mask_strings(s)
    masked = mask_holleriths(masked)
    u = masked.strip()
    if not u:
        return

    # Check for IF statement: block IF vs logical IF
    m_if = re.match(r"^IF\s*\(", u, re.IGNORECASE)
    if m_if:
        # Find closing paren of IF condition
        depth = 0
        cond_end = -1
        for i in range(len(u)):
            if u[i] == "(":
                depth += 1
            elif u[i] == ")":
                depth -= 1
                if depth == 0:
                    cond_end = i
                    break
        if cond_end != -1:
            cond = u[m_if.end() - 1:cond_end]
            # Condition is an expression: all common variables here are READ
            for w in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", cond):
                uw = w.upper()
                if uw in common_vars:
                    usage_map[uw]["read"] = True
            rest = u[cond_end + 1:].strip()
            if rest and not re.match(r"^THEN\b", rest, re.IGNORECASE):
                # Logical IF: process sub-statement
                analyze_statement_for_common_usage(rest, common_vars, usage_map)
            return

    # Check DO loop: DO [label] var = start, stop [, step]
    m_do = re.match(r"^DO\b(?:\s+\d+)?\s+([A-Za-z][A-Za-z0-9_]*)\s*=(.*)", u, re.IGNORECASE)
    if m_do:
        var = m_do.group(1).upper()
        if var in common_vars:
            usage_map[var]["written"] = True
        bounds = m_do.group(2)
        for w in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", bounds):
            uw = w.upper()
            if uw in common_vars:
                usage_map[uw]["read"] = True
        return

    # Check READ statement: READ(...) iolist
    m_read = re.match(r"^READ\b(?:\s*\((.*?)\)|\s*\*)\s*(.*)", u, re.IGNORECASE)
    if m_read:
        ctrl = m_read.group(1) or ""
        iolist = m_read.group(2) or ""
        # Variables in control spec are READ
        for w in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", ctrl):
            uw = w.upper()
            if uw in common_vars:
                usage_map[uw]["read"] = True

        # In iolist: target variables are WRITTEN, subscripts and implied-DO bounds are READ/WRITTEN
        _analyze_read_iolist(iolist, common_vars, usage_map)
        return

    # Check CALL statement: CALL Subname [(args)]
    m_call = re.match(r"^CALL\s+([A-Za-z0-9_]+)(?:\s*\((.*?)\))?", u, re.IGNORECASE)
    if m_call:
        args_text = m_call.group(2)
        if args_text:
            # Split args by comma at depth 0
            args: List[str] = []
            depth = 0
            start = 0
            for i, c in enumerate(args_text):
                if c == "(":
                    depth += 1
                elif c == ")":
                    if depth > 0:
                        depth -= 1
                elif c == "," and depth == 0:
                    args.append(args_text[start:i].strip())
                    start = i + 1
            args.append(args_text[start:].strip())

            for arg in args:
                if not arg:
                    continue
                # If arg is VAR or VAR(subscripts)
                m_var = re.match(r"^([A-Za-z][A-Za-z0-9_]*)(?:\s*\((.*)\))?$", arg)
                if m_var:
                    base_var = m_var.group(1).upper()
                    if base_var in common_vars:
                        usage_map[base_var]["maybe_written"] = True
                    subs = m_var.group(2)
                    if subs:
                        for w in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", subs):
                            uw = w.upper()
                            if uw in common_vars:
                                usage_map[uw]["read"] = True
                else:
                    # General expression passed as arg: common vars are read
                    for w in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", arg):
                        uw = w.upper()
                        if uw in common_vars:
                            usage_map[uw]["read"] = True
        return

    # Check Assignment: target = expression
    # Ensure '=' is at depth 0
    eq_pos = find_depth0_char(u, "=")
    if eq_pos != -1:
        target = u[:eq_pos].strip()
        expr = u[eq_pos + 1:].strip()

        # Parse target: VAR or VAR(...)
        m_t = re.match(r"^([A-Za-z][A-Za-z0-9_]*)(?:\s*\((.*)\))?", target)
        if m_t:
            target_var = m_t.group(1).upper()
            if target_var in common_vars:
                usage_map[target_var]["written"] = True
            subs = m_t.group(2)
            if subs:
                for w in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", subs):
                    uw = w.upper()
                    if uw in common_vars:
                        usage_map[uw]["read"] = True

        for w in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", expr):
            uw = w.upper()
            if uw in common_vars:
                usage_map[uw]["read"] = True
        return

    # Any other statement: all common variables are READ
    for w in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", u):
        uw = w.upper()
        if uw in common_vars:
            usage_map[uw]["read"] = True


def _analyze_read_iolist(iolist: str, common_vars: Set[str], usage_map: Dict[str, Dict[str, bool]]):
    """Parse items in READ iolist to mark target variables as written and subscripts as read."""
    if not iolist.strip():
        return
    # Split items by comma at depth 0
    items: List[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(iolist):
        if c == "(":
            depth += 1
        elif c == ")":
            if depth > 0:
                depth -= 1
        elif c == "," and depth == 0:
            items.append(iolist[start:i].strip())
            start = i + 1
    items.append(iolist[start:].strip())

    for item in items:
        it = item.strip()
        if not it:
            continue
        # Check implied DO: (items, do_var = start, end)
        if it.startswith("(") and it.endswith(")"):
            inner = it[1:-1].strip()
            # Find the last '=' at depth 0
            idepth = 0
            last_eq = -1
            for j, c in enumerate(inner):
                if c == "(":
                    idepth += 1
                elif c == ")":
                    idepth -= 1
                elif c == "=" and idepth == 0:
                    last_eq = j
            if last_eq != -1:
                # Find the comma before last_eq at depth 0
                pre_eq = inner[:last_eq]
                last_comma = -1
                jdepth = 0
                for k, c in enumerate(pre_eq):
                    if c == "(":
                        jdepth += 1
                    elif c == ")":
                        jdepth -= 1
                    elif c == "," and jdepth == 0:
                        last_comma = k
                if last_comma != -1:
                    inner_targets = pre_eq[:last_comma]
                    do_var = pre_eq[last_comma + 1:].strip().upper()
                    bounds = inner[last_eq + 1:]
                    if do_var in common_vars:
                        usage_map[do_var]["written"] = True
                    for w in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", bounds):
                        uw = w.upper()
                        if uw in common_vars:
                            usage_map[uw]["read"] = True
                    _analyze_read_iolist(inner_targets, common_vars, usage_map)
                    continue

        # Simple variable or array element: VAR or VAR(...)
        m = re.match(r"^([A-Za-z][A-Za-z0-9_]*)(?:\s*\((.*)\))?", it)
        if m:
            var = m.group(1).upper()
            if var in common_vars:
                usage_map[var]["written"] = True
            subs = m.group(2)
            if subs:
                for w in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", subs):
                    uw = w.upper()
                    if uw in common_vars:
                        usage_map[uw]["read"] = True


# ---------------------------------------------------------------------------
# File Mapping & Analysis
# ---------------------------------------------------------------------------

def parse_file_to_units(filepath: Union[str, Path]) -> List[ProgramUnit]:
    """Parse a fixed-form Fortran file into ProgramUnit objects."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    stmts = read_fixed_statements(lines)
    units: List[ProgramUnit] = []
    curr_unit: Optional[ProgramUnit] = None

    for s in stmts:
        m = HEADER_RE.match(s.text)
        if m:
            if curr_unit:
                curr_unit.line_range[1] = s.lines[0] - 1
                units.append(curr_unit)

            if m.group(1):
                name = m.group(1).upper()
                kind = "FUNCTION"
                args = m.group(2) or ""
            elif m.group(3):
                name = m.group(3).upper()
                kind = "SUBROUTINE"
                args = m.group(4) or ""
            elif m.group(5):
                name = m.group(5).upper()
                kind = "PROGRAM"
                args = ""
            else:
                name = (m.group(6) or "BLOCKDATA").upper()
                kind = "BLOCK DATA"
                args = ""

            dummy_args = [a.strip().upper() for a in args.split(",") if a.strip()] if args else []
            curr_unit = ProgramUnit(name, kind, dummy_args, s.lines[0], s.lines[-1])
            curr_unit.statements.append(s)
        elif curr_unit:
            curr_unit.statements.append(s)
            curr_unit.line_range[1] = s.lines[-1]
            if END_RE.match(s.text):
                units.append(curr_unit)
                curr_unit = None
        else:
            # Code appearing before the first unit header -> implicit main
            curr_unit = ProgramUnit("MAIN", "PROGRAM", [], s.lines[0], s.lines[-1])
            curr_unit.statements.append(s)
            if END_RE.match(s.text):
                units.append(curr_unit)
                curr_unit = None

    if curr_unit:
        units.append(curr_unit)

    return units


def analyze_units(units: List[ProgramUnit]) -> Dict[str, Any]:
    """Perform intra-file analysis across all program units."""
    known_function_names = {u.name for u in units if u.kind == "FUNCTION"}
    all_unit_names = {u.name for u in units}

    for unit in units:
        # 1. Identify declared arrays and executable statements
        declared_arrays: Set[str] = set()
        raw_io_units: Set[Union[int, str]] = set()

        for s in unit.statements:
            s.is_executable = not is_non_executable_stmt(s.text)

            # Check declarations for arrays
            u_text = s.text.upper()
            if re.match(r"^DIMENSION\b", u_text):
                for am in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(", u_text[9:]):
                    declared_arrays.add(am.group(1))
            elif re.match(r"^COMMON\b", u_text):
                cb_dict = parse_common_statement(s.text)
                for bname, var_list in cb_dict.items():
                    if bname not in unit.common_blocks:
                        unit.common_blocks[bname] = []
                    unit.common_blocks[bname].extend(var_list)
                # Check for array dimensions in common
                for am in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(", u_text):
                    declared_arrays.add(am.group(1))
            elif re.match(r"^DATA\b", u_text):
                dvars = parse_data_statement(s.text)
                for dv in dvars:
                    if dv not in unit.data_vars:
                        unit.data_vars.append(dv)
            elif re.match(r"^SAVE\b", u_text) and not s.is_executable:
                unit.save_statements.append(s.text)
                for sv in parse_save_statement(s.text):
                    if sv not in unit.save:
                        unit.save.append(sv)
            elif re.match(r"^EQUIVALENCE\b", u_text):
                eq_groups = parse_equivalence_statement(s.text)
                unit.equivalence.extend(eq_groups)
            else:
                for kw in TYPE_KEYWORDS:
                    if re.match(rf"^{kw}\b", u_text):
                        for am in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(", u_text):
                            declared_arrays.add(am.group(1))

            # I/O units
            io_u = extract_io_unit(s.text)
            if io_u is not None:
                raw_io_units.add(io_u)

        # Sort I/O units deterministically: ints first, then strings
        int_units = sorted([x for x in raw_io_units if isinstance(x, int)])
        str_units = sorted([x for x in raw_io_units if isinstance(x, str)])
        unit.io_units = int_units + str_units

        # Filter data_vars to local variables: not in common, not in dummy_args
        all_common_vars_set = set()
        for bvars in unit.common_blocks.values():
            all_common_vars_set.update(bvars)
        unit.data_vars = [v for v in unit.data_vars if v not in all_common_vars_set and v not in unit.dummy_args]

        # 2. Extract calls and referenced functions
        called_routines: Set[str] = set()
        ref_functions: Set[str] = set()

        for s in unit.statements:
            masked = mask_strings(s.text)
            masked = mask_holleriths(masked)
            # Find CALL statements
            for cm in re.finditer(r"\bCALL\s+([A-Za-z0-9_]+)", masked, re.IGNORECASE):
                cname = cm.group(1).upper()
                called_routines.add(cname)

            # Find referenced functions: name followed by '(' that is a known function unit in same file
            # and not a declared array in this unit
            for fm in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(", masked):
                fname = fm.group(1).upper()
                if fname in known_function_names and fname not in declared_arrays and fname != unit.name:
                    ref_functions.add(fname)

        unit.calls = sorted(called_routines)
        unit.functions = sorted(ref_functions)
        unit.callees = sorted(called_routines | ref_functions)

        # 3. Executable and non-comment lines
        exec_lines = sum(len(s.lines) for s in unit.statements if s.is_executable)
        total_code_lines = sum(len(s.lines) for s in unit.statements)
        unit.executable_lines = exec_lines
        unit.code_lines = total_code_lines

        # 4. COMMON variable READ / WRITTEN usage analysis
        all_common_vars = set()
        for bname, var_list in unit.common_blocks.items():
            all_common_vars.update(var_list)

        usage_map: Dict[str, Dict[str, bool]] = {
            v: {"read": False, "written": False, "maybe_written": False}
            for v in all_common_vars
        }

        for s in unit.statements:
            if s.is_executable:
                analyze_statement_for_common_usage(s.text, all_common_vars, usage_map)

        unit.common_vars = usage_map
        unit.common_usage = {}
        for bname, var_list in unit.common_blocks.items():
            unit.common_usage[bname] = {v: usage_map[v] for v in var_list if v in usage_map}

    # 5. Build call graph (intra-file dependencies)
    call_graph: Dict[str, List[str]] = {}
    deps: Dict[str, Set[str]] = {}
    for unit in units:
        intra_deps = {c for c in unit.callees if c in all_unit_names and c != unit.name}
        call_graph[unit.name] = sorted(intra_deps)
        deps[unit.name] = intra_deps

    # 6. Topological port order: leaves first, break cycles deterministically
    rem = set(all_unit_names)
    order: List[str] = []
    while rem:
        leaves = [u for u in rem if not (deps[u] & rem)]
        if leaves:
            chosen = min(leaves)
        else:
            # Cycle detected: choose node with minimal remaining dependencies, tie-break alphabetically
            chosen = min(rem, key=lambda u: (len(deps[u] & rem), u))
        order.append(chosen)
        rem.remove(chosen)

    return {
        "units": {u.name: u.to_dict() for u in units},
        "program_units": [u.to_dict() for u in units],
        "call_graph": call_graph,
        "order": order,
    }


# ---------------------------------------------------------------------------
# Variant Comparison
# ---------------------------------------------------------------------------

def normalize_body(unit: ProgramUnit) -> List[str]:
    """Normalize statement list of a routine body for variant comparison.
    
    Rules:
    - Uppercase
    - Collapse whitespace
    - Drop comments (already omitted from FortranStatement)
    - Drop labels on CONTINUE and FORMAT lines; keep labels otherwise
    """
    normalized: List[str] = []
    for s in unit.statements:
        u = re.sub(r"\s+", " ", s.text).strip().upper()
        if not u:
            continue
        # Drop labels on CONTINUE and FORMAT statements
        is_cont_or_fmt = bool(re.match(r"^(?:CONTINUE\b|FORMAT\b)", u))
        lbl = ""
        if s.label and not is_cont_or_fmt:
            lbl = f"{s.label} "
        normalized.append(f"{lbl}{u}")
    return normalized


def strip_dimension_specs(stmt: str) -> str:
    """Remove array dimension specifications (...) for dimension-equivalence check."""
    res = []
    depth = 0
    in_quote = None
    i = 0
    n = len(stmt)
    while i < n:
        c = stmt[i]
        if in_quote:
            res.append(c)
            if c == in_quote:
                in_quote = None
        else:
            if c in ("'", '"'):
                in_quote = c
                res.append(c)
            elif c == "(":
                depth += 1
            elif c == ")":
                if depth > 0:
                    depth -= 1
            elif depth == 0:
                res.append(c)
        i += 1
    return re.sub(r"\s+", " ", "".join(res)).strip()


def is_cosmetic_statement(s: str) -> bool:
    """Check if statement is FORMAT, WRITE, PRINT, PARAMETER, or DIMENSION."""
    u = s.strip().upper()
    # Strip optional label
    u = re.sub(r"^\d+\s+", "", u)
    if u.startswith(("FORMAT(", "FORMAT ")):
        return True
    if u.startswith(("WRITE(", "WRITE ", "PRINT ", "PRINT(")):
        return True
    if u.startswith(("PARAMETER(", "PARAMETER ")):
        return True
    if u.startswith("DIMENSION "):
        return True
    return False


def is_hunk_cosmetic(lines1: List[str], lines2: List[str]) -> bool:
    """Check if a diff hunk between lines1 and lines2 is purely cosmetic."""
    # If all statements in both sides are cosmetic types
    all1 = all(is_cosmetic_statement(s) for s in lines1)
    all2 = all(is_cosmetic_statement(s) for s in lines2)
    if all1 and all2:
        return True

    # Check if lines differ only in dimension sizes (e.g. COMMON or type declarations)
    if len(lines1) == len(lines2):
        dims_match = True
        for s1, s2 in zip(lines1, lines2):
            if strip_dimension_specs(s1) != strip_dimension_specs(s2):
                dims_match = False
                break
        if dims_match:
            return True

    return False


def classify_pair(
    stmts1: List[str],
    stmts2: List[str],
) -> Tuple[str, int, List[Tuple[str, str]]]:
    """Classify pair of routine bodies as 'identical', 'cosmetic', or 'semantic'.
    
    Returns (classification, changed_lines, diff_samples).
    """
    if stmts1 == stmts2:
        return "identical", 0, []

    sm = difflib.SequenceMatcher(None, stmts1, stmts2)
    opcodes = sm.get_opcodes()
    changed_lines = sum((i2 - i1) + (j2 - j1) for tag, i1, i2, j1, j2 in opcodes if tag != "equal")

    semantic = False
    diff_sample: List[Tuple[str, str]] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        hunk1 = stmts1[i1:i2]
        hunk2 = stmts2[j1:j2]
        if not is_hunk_cosmetic(hunk1, hunk2):
            semantic = True
            # Collect first few differing statement pairs
            max_len = max(len(hunk1), len(hunk2))
            for k in range(max_len):
                if len(diff_sample) < 3:
                    s1 = hunk1[k] if k < len(hunk1) else "<none>"
                    s2 = hunk2[k] if k < len(hunk2) else "<none>"
                    diff_sample.append((s1, s2))

    classification = "semantic" if semantic else "cosmetic"
    return classification, changed_lines, diff_sample


def compare_variants(filepaths: List[Union[str, Path]]) -> Dict[str, Any]:
    """Compare all routine names present in more than one file."""
    file_map: Dict[str, Dict[str, ProgramUnit]] = {}
    normalized_map: Dict[str, Dict[str, List[str]]] = {}

    for fp in filepaths:
        path = Path(fp)
        pname = path.name
        units = parse_file_to_units(path)
        file_map[pname] = {u.name: u for u in units}
        normalized_map[pname] = {u.name: normalize_body(u) for u in units}

    # Find routines present in >= 2 files
    from collections import defaultdict
    routine_files: Dict[str, List[str]] = defaultdict(list)
    for pname, rdict in file_map.items():
        for rname in rdict:
            routine_files[rname].append(pname)

    shared_routines = sorted([r for r, flist in routine_files.items() if len(flist) > 1])

    pairs_data: List[Dict[str, Any]] = []
    identical_count = 0
    cosmetic_count = 0
    semantic_count = 0

    for rname in shared_routines:
        flist = sorted(routine_files[rname])
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                f1, f2 = flist[i], flist[j]
                s1 = normalized_map[f1][rname]
                s2 = normalized_map[f2][rname]
                cls, chg, sample = classify_pair(s1, s2)
                if cls == "identical":
                    identical_count += 1
                elif cls == "cosmetic":
                    cosmetic_count += 1
                else:
                    semantic_count += 1

                pairs_data.append({
                    "routine": rname,
                    "file_a": f1,
                    "file_b": f2,
                    "classification": cls,
                    "changed_lines": chg,
                    "diff_sample": [[a, b] for a, b in sample],
                })

    summary = {
        "files": [Path(p).name for p in filepaths],
        "total_shared_routines": len(shared_routines),
        "total_pairs": len(pairs_data),
        "identical": identical_count,
        "cosmetic": cosmetic_count,
        "semantic": semantic_count,
    }

    return {
        "summary": summary,
        "shared_routines": shared_routines,
        "pairs": pairs_data,
    }


def format_markdown_table(variants_result: Dict[str, Any]) -> str:
    """Format variants comparison into a GitHub Flavored Markdown report."""
    summary = variants_result["summary"]
    lines = [
        "# Fortran Routine Variants Analysis",
        "",
        "## Summary",
        "",
        f"- **Files compared**: {', '.join(summary['files'])}",
        f"- **Shared routine names**: {summary['total_shared_routines']}",
        f"- **Pairwise comparisons**: {summary['total_pairs']}",
        f"- **Identical pairs**: {summary['identical']}",
        f"- **Cosmetic pairs**: {summary['cosmetic']}",
        f"- **Semantic pairs**: {summary['semantic']}",
        "",
        "## Pairwise Comparison Table",
        "",
        "| Routine | File A | File B | Classification | Changed Lines | First Semantic Diff |",
        "| :--- | :--- | :--- | :--- | :---: | :--- |",
    ]

    for p in variants_result["pairs"]:
        sample_str = ""
        if p["diff_sample"]:
            first_pair = p["diff_sample"][0]
            s_a = first_pair[0].replace("|", "\\|")
            s_b = first_pair[1].replace("|", "\\|")
            sample_str = f"`{s_a}` vs `{s_b}`"
        lines.append(
            f"| `{p['routine']}` | `{p['file_a']}` | `{p['file_b']}` | **{p['classification']}** | {p['changed_lines']} | {sample_str} |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Fortran 77 source mapping and variant comparison tool.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: map
    map_parser = subparsers.add_parser("map", help="Map program units, call graph, COMMON usage, and port order.")
    map_parser.add_argument("file", help="Path to Fortran source file (.for)")
    map_parser.add_argument("--json", dest="json_out", help="Output path for JSON map")

    # Subcommand: variants
    var_parser = subparsers.add_parser("variants", help="Compare routine variants across multiple files.")
    var_parser.add_argument("files", nargs="+", help="Paths to Fortran source files")
    var_parser.add_argument("--json", dest="json_out", help="Output path for JSON report")
    var_parser.add_argument("--md", dest="md_out", help="Output path for Markdown table")

    args = parser.parse_args()

    if args.subcommand == "map":
        path = Path(args.file)
        if not path.is_file():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1
        units = parse_file_to_units(path)
        result = analyze_units(units)
        result["file"] = path.name

        if args.json_out:
            out_p = Path(args.json_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=1)
            print(f"Wrote map to {out_p} ({len(units)} units)")
        else:
            print(json.dumps(result, indent=1))

    elif args.subcommand == "variants":
        paths = [Path(f) for f in args.files]
        for p in paths:
            if not p.is_file():
                print(f"Error: file not found: {p}", file=sys.stderr)
                return 1
        result = compare_variants(paths)

        if args.json_out:
            out_p = Path(args.json_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=1)
            print(f"Wrote variants JSON to {out_p}")

        md_content = format_markdown_table(result)
        if args.md_out:
            out_m = Path(args.md_out)
            out_m.parent.mkdir(parents=True, exist_ok=True)
            with open(out_m, "w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"Wrote variants Markdown to {out_m}")

        sum_info = result["summary"]
        print(f"Variants summary: {sum_info['total_shared_routines']} shared routines, "
              f"{sum_info['total_pairs']} pairs: "
              f"{sum_info['identical']} identical, {sum_info['cosmetic']} cosmetic, {sum_info['semantic']} semantic")

    return 0


if __name__ == "__main__":
    sys.exit(main())
