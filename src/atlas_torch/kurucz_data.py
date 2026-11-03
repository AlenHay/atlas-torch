"""Pure-Python parser for Fortran 77 DATA statements in Kurucz stellar atmosphere codes."""

from __future__ import annotations

import ast
import hashlib
import operator as op
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

_OP_MAP = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_expr(expr_str: str, env: dict[str, Any]) -> Any:
    """Evaluate simple arithmetic on integer/float constants and PARAMETERs."""
    s = expr_str.strip()
    if not s:
        return 0
    # Normalize D exponents in floating literals (e.g. 1.5D-3 -> 1.5e-3)
    s_norm = re.sub(r"(?<=\d)[Dd](?=[+-]?\d)", "e", s)
    s_norm = re.sub(r"(?<=\.)[Dd](?=[+-]?\d)", "e", s_norm)
    tree = ast.parse(s_norm, mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            key = node.id.lower()
            if key in env:
                return env[key]
            raise KeyError(f"Undefined parameter or variable: {node.id}")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type is ast.Div and isinstance(left, int) and isinstance(right, int):
                # Fortran integer division
                return left // right
            return _OP_MAP[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            return _OP_MAP[type(node.op)](operand)
        raise ValueError(f"Unsupported AST node: {ast.dump(node)}")

    return _eval(tree.body)


def _read_statements(source_path: str | Path) -> list[tuple[str, list[str]]]:
    """Read a Fortran 77 source file into a list of (unit_name, statements) per program unit.

    Follows fixed-form rules:
    - Col 1: C/c/*/! = comment line
    - Col 6: non-blank and not '0' = continuation line
    - Statement text in cols 7-72 (truncated beyond 72)
    - Tab format: line starting with tab is cols 7..; if followed by 1-9, it is continuation.
      Tab lines are NOT truncated at 72.
    """
    with open(source_path, "r", encoding="latin-1") as fh:
        raw_lines = fh.readlines()

    # Step 1: Reconstruct logical statements
    logical_stmts: list[str] = []
    curr: str = ""

    for line in raw_lines:
        line = line.rstrip("\r\n")
        if not line or not line.strip():
            continue

        c0 = line[0]
        if c0 in ("C", "c", "*", "!"):
            continue

        # Check tab format
        if line.startswith("\t"):
            if len(line) > 1 and line[1] in "123456789":
                curr += line[2:]
            else:
                if curr:
                    logical_stmts.append(curr)
                curr = line[1:]
        elif "\t" in line[:6]:
            parts = line.split("\t", 1)
            if len(parts[0]) < 6:
                rest = parts[1]
                if len(rest) > 0 and rest[0] in "123456789":
                    curr += rest[1:]
                else:
                    if curr:
                        logical_stmts.append(curr)
                    curr = rest
            else:
                line72 = line[:72]
                col6 = line72[5] if len(line72) >= 6 else " "
                text = line72[6:] if len(line72) >= 7 else ""
                if col6 not in (" ", "0"):
                    curr += text
                else:
                    if curr:
                        logical_stmts.append(curr)
                    curr = text
        else:
            line72 = line[:72]
            col6 = line72[5] if len(line72) >= 6 else " "
            text = line72[6:] if len(line72) >= 7 else ""
            if col6 not in (" ", "0"):
                curr += text
            else:
                if curr:
                    logical_stmts.append(curr)
                curr = text

    if curr:
        logical_stmts.append(curr)

    # Step 2: Group statements into program units
    units: list[tuple[str, list[str]]] = []
    curr_unit = "main"
    curr_unit_stmts: list[str] = []

    unit_header_re = re.compile(
        r"^\s*(?:(?:REAL\s*\*\s*8|REAL\s*\*\s*4|REAL|DOUBLE\s+PRECISION|INTEGER\s*\*\s*4|INTEGER\s*\*\s*2|INTEGER|LOGICAL\s*\*\s*4|LOGICAL|CHARACTER(?:\s*\*\s*\d+)?)\s+)?"
        r"(PROGRAM|SUBROUTINE|FUNCTION|BLOCK\s*DATA)\s*([a-zA-Z0-9_]*)",
        re.I,
    )
    end_re = re.compile(
        r"^\s*END(?:\s+(?:PROGRAM|SUBROUTINE|FUNCTION|BLOCK\s*DATA)(?:\s+[a-zA-Z0-9_]+)?)?\s*$",
        re.I,
    )

    for stmt in logical_stmts:
        m_head = unit_header_re.match(stmt)
        if m_head:
            kwd = m_head.group(1).upper()
            name = m_head.group(2).lower()
            if "BLOCK" in kwd and not name:
                name = "blockdata"
            if not name:
                name = "main"
            if curr_unit_stmts:
                units.append((curr_unit, curr_unit_stmts))
            curr_unit = name
            curr_unit_stmts = [stmt]
        elif end_re.match(stmt):
            curr_unit_stmts.append(stmt)
            units.append((curr_unit, curr_unit_stmts))
            curr_unit = "main"
            curr_unit_stmts = []
        else:
            curr_unit_stmts.append(stmt)

    if curr_unit_stmts:
        units.append((curr_unit, curr_unit_stmts))

    return units


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    """Split text by delimiter, ignoring delimiters inside quotes and parentheses."""
    tokens = []
    i = 0
    n = len(text)
    cur = []
    depth = 0
    in_quote = False

    while i < n:
        ch = text[i]
        if ch == "\x27":
            cur.append(ch)
            i += 1
            while i < n:
                cur.append(text[i])
                if text[i] == "\x27":
                    if i + 1 < n and text[i + 1] == "\x27":
                        cur.append(text[i + 1])
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    i += 1
            continue

        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == delimiter and depth == 0:
            tokens.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1

    if cur:
        tokens.append("".join(cur).strip())
    return [t for t in tokens if t]


def _split_data_pairs(stmt: str) -> list[tuple[str, str]]:
    """Split a DATA statement into pairs of (var_list_str, val_list_str)."""
    m = re.match(r"^\s*DATA\s*", stmt, re.I)
    if not m:
        return []
    rest = stmt[m.end():]

    chunks = []
    cur = []
    i = 0
    n = len(rest)
    in_slash = False

    while i < n:
        ch = rest[i]
        if ch == "\x27":
            j = i + 1
            while j < n:
                if rest[j] == "\x27":
                    if j + 1 < n and rest[j + 1] == "\x27":
                        j += 2
                    else:
                        j += 1
                        break
                else:
                    j += 1
            cur.append(rest[i:j])
            i = j
        elif in_slash and ch.isdigit():
            m_h = re.match(r"^(\d+)[Hh]", rest[i:])
            if m_h:
                h_len = int(m_h.group(1))
                h_start = i + len(m_h.group(0))
                h_end = h_start + h_len
                cur.append(rest[i:h_end])
                i = h_end
            else:
                cur.append(ch)
                i += 1
        elif ch == "/":
            chunks.append("".join(cur).strip())
            cur = []
            in_slash = not in_slash
            i += 1
        else:
            cur.append(ch)
            i += 1

    pairs = []
    # chunks: [vars0, vals0, vars1, vals1, ...]
    for idx in range(0, len(chunks) - 1, 2):
        v_str = chunks[idx]
        if v_str.startswith(","):
            v_str = v_str[1:].strip()
        pairs.append((v_str, chunks[idx + 1]))

    return pairs


def _split_val_list(val_str: str) -> list[str]:
    """Split val_list on commas, respecting quotes and Hollerith literals."""
    tokens = []
    i = 0
    n = len(val_str)
    cur = []
    depth = 0

    while i < n:
        ch = val_str[i]
        if ch == "\x27":
            j = i + 1
            while j < n:
                if val_str[j] == "\x27":
                    if j + 1 < n and val_str[j + 1] == "\x27":
                        j += 2
                    else:
                        j += 1
                        break
                else:
                    j += 1
            cur.append(val_str[i:j])
            i = j
        elif ch.isdigit():
            m_h = re.match(r"^(\d+)[Hh]", val_str[i:])
            if m_h:
                h_len = int(m_h.group(1))
                h_start = i + len(m_h.group(0))
                h_end = h_start + h_len
                cur.append(val_str[i:h_end])
                i = h_end
            else:
                cur.append(ch)
                i += 1
        elif ch == "(":
            depth += 1
            cur.append(ch)
            i += 1
        elif ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
            i += 1
        elif ch == "," and depth == 0:
            tokens.append("".join(cur).strip())
            cur = []
            i += 1
        else:
            cur.append(ch)
            i += 1

    if cur:
        tokens.append("".join(cur).strip())
    return [t for t in tokens if t]


def _parse_implied_do(text: str) -> tuple[list[str], str, str, str, str] | None:
    """Parse an implied DO construct: (items, var = start, end [, step])."""
    s = text.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    inner = s[1:-1].strip()

    parts = []
    depth = 0
    cur = []
    eq_in_part = []
    cur_has_eq_depth0 = False

    for ch in inner:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "=" and depth == 0:
            cur_has_eq_depth0 = True
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            eq_in_part.append(cur_has_eq_depth0)
            cur = []
            cur_has_eq_depth0 = False
        else:
            cur.append(ch)

    if cur:
        parts.append("".join(cur).strip())
        eq_in_part.append(cur_has_eq_depth0)

    eq_idx = -1
    for idx, has_eq in enumerate(eq_in_part):
        if has_eq:
            eq_idx = idx
            break

    if eq_idx == -1:
        return None

    items = parts[:eq_idx]
    var_start = parts[eq_idx]
    var_name, start_expr = var_start.split("=", 1)
    var_name = var_name.strip().lower()
    end_expr = parts[eq_idx + 1]
    step_expr = parts[eq_idx + 2] if len(parts) > eq_idx + 2 else "1"

    return items, var_name, start_expr, end_expr, step_expr


def _expand_target(
    target_str: str,
    loop_env: dict[str, Any],
    declared_shapes: dict[str, tuple[int, ...]],
    parameters: dict[str, Any],
) -> list[tuple[str, int]]:
    """Expand a target string into a list of (var_name, linear_0based_index)."""
    target_str = target_str.strip()

    # Check for implied DO loop
    parsed_do = _parse_implied_do(target_str)
    if parsed_do is not None:
        items, loop_var, start_expr, end_expr, step_expr = parsed_do
        start_val = int(_eval_expr(start_expr, {**parameters, **loop_env}))
        end_val = int(_eval_expr(end_expr, {**parameters, **loop_env}))
        step_val = int(_eval_expr(step_expr, {**parameters, **loop_env}))

        res: list[tuple[str, int]] = []
        if step_val > 0:
            rng = range(start_val, end_val + 1, step_val)
        else:
            rng = range(start_val, end_val - 1, step_val)

        for val in rng:
            new_env = dict(loop_env)
            new_env[loop_var] = val
            for item in items:
                res.extend(_expand_target(item, new_env, declared_shapes, parameters))
        return res

    # Check for array element or section: var(subscripts)
    m = re.match(r"^([a-zA-Z0-9_ ]+)\s*\((.*)\)$", target_str)
    if m:
        var_name = re.sub(r"\s+", "", m.group(1)).lower()
        subscripts_str = m.group(2)
        shape = declared_shapes.get(var_name, ())

        sub_parts = _split_top_level(subscripts_str, ",")
        # Check if any subscript is a range like lo:hi
        has_slice = any(":" in sp for sp in sub_parts)
        if has_slice:
            # Handle slice ranges (e.g. A(1:5))
            dim_ranges = []
            for dim_idx, sp in enumerate(sub_parts):
                if ":" in sp:
                    lo_s, hi_s = sp.split(":", 1)
                    lo = int(_eval_expr(lo_s, {**parameters, **loop_env})) if lo_s.strip() else 1
                    hi = int(_eval_expr(hi_s, {**parameters, **loop_env})) if hi_s.strip() else (shape[dim_idx] if dim_idx < len(shape) else 1)
                    dim_ranges.append(list(range(lo, hi + 1)))
                else:
                    val = int(_eval_expr(sp, {**parameters, **loop_env}))
                    dim_ranges.append([val])

            # Generate cartesian product in Fortran order (first index varies fastest)
            import itertools
            res = []
            for coords in itertools.product(*reversed(dim_ranges)):
                # coords is in reversed order, reverse back
                actual_coords = list(reversed(coords))
                # compute linear index
                if len(shape) <= 1:
                    lin = actual_coords[0] - 1
                else:
                    lin = 0
                    mult = 1
                    for c, d in zip(actual_coords, shape):
                        lin += (c - 1) * mult
                        mult *= d
                res.append((var_name, lin))
            return res

        # Normal element subscript
        sub_vals = [int(_eval_expr(sp, {**parameters, **loop_env})) for sp in sub_parts]
        if len(sub_vals) == 1 and len(shape) > 1:
            # 1D linear subscript into multidimensional array
            lin_idx = sub_vals[0] - 1
        elif len(sub_vals) == 1:
            lin_idx = sub_vals[0] - 1
        else:
            # Multidimensional 1-based indexing in Fortran order
            lin_idx = 0
            mult = 1
            for coord, dim in zip(sub_vals, shape):
                lin_idx += (coord - 1) * mult
                mult *= dim
        return [(var_name, lin_idx)]

    # Whole variable target
    var_name = re.sub(r"\s+", "", target_str).lower()
    shape = declared_shapes.get(var_name, ())
    if not shape:
        return [(var_name, 0)]

    total_elements = 1
    for d in shape:
        total_elements *= d
    return [(var_name, i) for i in range(total_elements)]


def _detect_compiler(source_path: Path) -> str:
    """Detect compiler (gfortran vs ifx) from corresponding oracle shared library if present."""
    atlas_ext = os.environ.get("ATLAS_EXT", str(Path.home() / "atlas-ext"))
    oracle_dir = Path(atlas_ext) / "oracle"
    base = source_path.stem.replace(".patched", "").replace(".gf", "")
    for candidate in [f"lib{base}.so", f"lib{source_path.stem}.so"]:
        so_file = oracle_dir / candidate
        if so_file.exists():
            try:
                from elftools.elf.elffile import ELFFile

                with open(so_file, "rb") as fh:
                    elf = ELFFile(fh)
                    dw = elf.get_dwarf_info()
                    for cu in dw.iter_CUs():
                        top = cu.get_top_DIE()
                        prod = top.attributes.get("DW_AT_producer")
                        if prod:
                            val = prod.value.decode("latin-1")
                            if "Intel" in val:
                                return "ifx"
                            if "GNU" in val:
                                return "gfortran"
                        break
            except Exception:
                pass
    if any(k in source_path.name.lower() for k in ("kappa", "xnfdf", "dfsynthe")):
        return "ifx"
    return "gfortran"


def _parse_float32(text: str, compiler: str = "gfortran") -> np.float32:
    """Parse text into float32 matching compiler semantics (ifx ties away from zero, gfortran ties to even)."""
    val_norm = text.lower().replace("d", "e")
    f = float(val_norm)
    f32 = np.float32(f)
    if compiler == "ifx":
        if f > 0:
            f32_up = np.nextafter(f32, np.float32(np.inf))
            if abs(float(f32_up) - f) == abs(float(f32) - f):
                return f32_up
        elif f < 0:
            f32_down = np.nextafter(f32, np.float32(-np.inf))
            if abs(float(f32_down) - f) == abs(float(f32) - f):
                return f32_down
    return f32


def _convert_val(
    val_str: str,
    dtype: np.dtype,
    parameters: dict[str, Any],
    compiler: str = "gfortran",
) -> Any:
    """Convert a value token string to the requested numpy dtype matching Fortran compiler semantics."""
    s = val_str.strip()

    if dtype.kind == "S":
        length = dtype.itemsize
        if s.startswith("\x27"):
            txt = s[1:-1].replace("\x27\x27", "\x27")
            raw = txt.encode("latin-1")
        elif re.match(r"^\d+[Hh]", s):
            m = re.match(r"^(\d+)[Hh](.*)$", s, re.DOTALL)
            assert m is not None
            n_chars = int(m.group(1))
            raw = m.group(2)[:n_chars].encode("latin-1")
        else:
            raw = s.encode("latin-1")
        return np.bytes_(raw.ljust(length, b" ")[:length])

    if dtype.kind in ("i", "u"):
        if s.startswith("\x27") or re.match(r"^\d+[Hh]", s):
            if s.startswith("\x27"):
                txt = s[1:-1].replace("\x27\x27", "\x27")
                raw = txt.encode("latin-1")
            else:
                m = re.match(r"^(\d+)[Hh](.*)$", s, re.DOTALL)
                assert m is not None
                raw = m.group(2)[:int(m.group(1))].encode("latin-1")
            raw = raw.ljust(dtype.itemsize, b" ")[:dtype.itemsize]
            return np.frombuffer(raw, dtype=dtype)[0]

        if s.lower() in parameters:
            num = parameters[s.lower()]
            return dtype.type(int(round(float(num))))
        return dtype.type(int(round(float(s))))

    if dtype.kind == "f":
        if s.startswith("\x27") or re.match(r"^\d+[Hh]", s):
            if s.startswith("\x27"):
                txt = s[1:-1].replace("\x27\x27", "\x27")
                raw = txt.encode("latin-1")
            else:
                m = re.match(r"^(\d+)[Hh](.*)$", s, re.DOTALL)
                assert m is not None
                raw = m.group(2)[:int(m.group(1))].encode("latin-1")
            raw = raw.ljust(dtype.itemsize, b" ")[:dtype.itemsize]
            return np.frombuffer(raw, dtype=dtype)[0]

        if s.lower() in parameters:
            num = parameters[s.lower()]
            if dtype == np.float64 and isinstance(num, float):
                return np.float64(num)
            return dtype.type(num)

        if dtype == np.float64:
            if "d" in s.lower():
                return np.float64(s.lower().replace("d", "e"))
            elif re.match(r"^[+-]?\d+$", s):
                # Integer literal converted to double exactly
                return np.float64(int(s))
            else:
                # Single-precision constant widened to double
                return np.float64(_parse_float32(s, compiler=compiler))
        else:
            return _parse_float32(s, compiler=compiler)

    if dtype.kind == "b":
        vup = s.upper()
        if vup in (".TRUE.", "TRUE", "T"):
            return True
        if vup in (".FALSE.", "FALSE", "F"):
            return False
        return bool(int(s))

    return dtype.type(s)


def parse_data(
    source_path: str | Path,
    compiler: str | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Parse every DATA statement in all program units of a Fortran 77 source file.

    Returns:
        dict[(unit_name_lower, var_name_lower), numpy.ndarray]
    """
    source_p = Path(source_path)
    if compiler is None:
        compiler = _detect_compiler(source_p)

    program_units = _read_statements(source_p)
    result: dict[tuple[str, str], np.ndarray] = {}

    for unit_name, stmts in program_units:
        parameters: dict[str, Any] = {}
        declared_types: dict[str, np.dtype] = {}
        declared_shapes: dict[str, tuple[int, ...]] = {}

        # Default implicit typing (Fortran 77: I-N integer, else REAL*4)
        implicit_types: dict[str, np.dtype] = {}
        for ch in "abcdefghijklmnopqrstuvwxyz":
            implicit_types[ch] = np.dtype(np.int32) if ch in "ijklmn" else np.dtype(np.float32)

        # PASS 1: Collect declarations (PARAMETER, IMPLICIT, DIMENSION, COMMON, types)
        for stmt in stmts:
            s = stmt.strip()
            if not s:
                continue

            # PARAMETER
            m_param = re.match(r"^PARAMETER\s*\((.*)\)\s*$", s, re.I)
            if m_param:
                for assign in _split_top_level(m_param.group(1), ","):
                    if "=" in assign:
                        pname, pexpr = assign.split("=", 1)
                        parameters[pname.strip().lower()] = _eval_expr(pexpr, parameters)
                continue

            # IMPLICIT
            m_impl = re.match(r"^IMPLICIT\s+(.*)$", s, re.I)
            if m_impl:
                rest = m_impl.group(1).strip()
                if re.match(r"^NONE\s*$", rest, re.I):
                    implicit_types.clear()
                else:
                    # Match specs like: REAL*8 (A-H,O-Z), INTEGER (I-N)
                    # Find each (...) group
                    spec_matches = re.finditer(r"([a-zA-Z0-9_* ]+)\s*\(([^)]+)\)", rest)
                    for sm in spec_matches:
                        tspec = sm.group(1).strip().upper().replace(" ", "")
                        ltrs = sm.group(2).strip()

                        if tspec in ("REAL*8", "DOUBLEPRECISION"):
                            t_dt = np.dtype(np.float64)
                        elif tspec in ("REAL*4", "REAL"):
                            t_dt = np.dtype(np.float32)
                        elif tspec in ("INTEGER*4", "INTEGER"):
                            t_dt = np.dtype(np.int32)
                        elif tspec == "INTEGER*2":
                            t_dt = np.dtype(np.int16)
                        elif tspec == "INTEGER*8":
                            t_dt = np.dtype(np.int64)
                        elif tspec in ("LOGICAL*4", "LOGICAL"):
                            t_dt = np.dtype(bool)
                        elif tspec.startswith("CHARACTER"):
                            m_c = re.match(r"^CHARACTER\*?(\d+)?", tspec)
                            c_len = int(m_c.group(1)) if m_c and m_c.group(1) else 1
                            t_dt = np.dtype(f"S{c_len}")
                        else:
                            t_dt = np.dtype(np.float32)

                        for l_range in _split_top_level(ltrs, ","):
                            if "-" in l_range:
                                c_lo, c_hi = l_range.split("-", 1)
                                for code in range(ord(c_lo.strip().lower()), ord(c_hi.strip().lower()) + 1):
                                    implicit_types[chr(code)] = t_dt
                            else:
                                implicit_types[l_range.strip().lower()] = t_dt
                continue

            # DIMENSION
            m_dim = re.match(r"^DIMENSION\s+(.*)$", s, re.I)
            if m_dim:
                for decl in _split_top_level(m_dim.group(1), ","):
                    m_arr = re.match(r"^([a-zA-Z0-9_ ]+)\s*\((.*)\)$", decl.strip())
                    if m_arr:
                        vname = re.sub(r"\s+", "", m_arr.group(1)).lower()
                        try:
                            dims = tuple(int(_eval_expr(d, parameters)) for d in _split_top_level(m_arr.group(2), ","))
                            declared_shapes[vname] = dims
                        except (KeyError, ValueError):
                            pass
                continue

            # COMMON
            m_com = re.match(r"^COMMON\s*(/.*|[a-zA-Z0-9_].*)$", s, re.I)
            if m_com:
                com_text = m_com.group(1).strip()
                # Tokens: /block/ or decls
                # Split along /.../
                parts = re.split(r"/([^/]*)/", com_text)
                # parts: [before, block1, inside1, block2, inside2, ...]
                inside_idx = 2 if len(parts) > 1 else 0
                while inside_idx < len(parts):
                    inside = parts[inside_idx].strip()
                    for decl in _split_top_level(inside, ","):
                        m_arr = re.match(r"^([a-zA-Z0-9_ ]+)\s*\((.*)\)$", decl.strip())
                        if m_arr:
                            vname = re.sub(r"\s+", "", m_arr.group(1)).lower()
                            try:
                                dims = tuple(int(_eval_expr(d, parameters)) for d in _split_top_level(m_arr.group(2), ","))
                                declared_shapes[vname] = dims
                            except (KeyError, ValueError):
                                pass
                    inside_idx += 2
                continue

            # Type statements
            type_re = re.compile(
                r"^(DOUBLE\s+PRECISION|REAL\s*\*\s*8|REAL\s*\*\s*4|REAL|"
                r"INTEGER\s*\*\s*8|INTEGER\s*\*\s*4|INTEGER\s*\*\s*2|INTEGER\s*\*\s*1|INTEGER|"
                r"LOGICAL\s*\*\s*4|LOGICAL\s*\*\s*1|LOGICAL|"
                r"CHARACTER(?:\s*\*\s*(?:\([^)]+\)|\d+))?)\b",
                re.I,
            )
            m_type = type_re.match(s)
            if m_type:
                # Make sure it is not a FUNCTION statement
                rest = s[m_type.end():].strip()
                if re.match(r"^FUNCTION\b", rest, re.I):
                    continue

                tspec = m_type.group(1).strip().upper().replace(" ", "")
                if tspec in ("REAL*8", "DOUBLEPRECISION"):
                    base_dt = np.dtype(np.float64)
                elif tspec in ("REAL*4", "REAL"):
                    base_dt = np.dtype(np.float32)
                elif tspec in ("INTEGER*4", "INTEGER"):
                    base_dt = np.dtype(np.int32)
                elif tspec == "INTEGER*2":
                    base_dt = np.dtype(np.int16)
                elif tspec == "INTEGER*8":
                    base_dt = np.dtype(np.int64)
                elif tspec in ("LOGICAL*4", "LOGICAL"):
                    base_dt = np.dtype(bool)
                elif tspec.startswith("CHARACTER"):
                    m_c = re.match(r"^CHARACTER\*?(?:\(([^)]+)\)|(\d+))?", tspec)
                    if m_c and m_c.group(1):
                        c_len = int(_eval_expr(m_c.group(1), parameters))
                    elif m_c and m_c.group(2):
                        c_len = int(m_c.group(2))
                    else:
                        c_len = 1
                    base_dt = np.dtype(f"S{c_len}")
                else:
                    base_dt = np.dtype(np.float32)

                for decl in _split_top_level(rest, ","):
                    decl = decl.strip()
                    # Could be: var, var*len, var(dims), var(dims)*len, var*len(dims)
                    # Check dimensions
                    m_arr = re.match(r"^([a-zA-Z0-9_ ]+)\s*\((.*)\)(?:\s*\*\s*(\d+))?$", decl)
                    if m_arr:
                        vname = re.sub(r"\s+", "", m_arr.group(1)).lower()
                        try:
                            dims = tuple(int(_eval_expr(d, parameters)) for d in _split_top_level(m_arr.group(2), ","))
                            declared_shapes[vname] = dims
                        except (KeyError, ValueError):
                            pass
                        if m_arr.group(3):
                            declared_types[vname] = np.dtype(f"S{m_arr.group(3)}")
                        else:
                            declared_types[vname] = base_dt
                        continue

                    m_arr2 = re.match(r"^([a-zA-Z0-9_ ]+)\s*\*\s*(\d+)\s*\((.*)\)$", decl)
                    if m_arr2:
                        vname = re.sub(r"\s+", "", m_arr2.group(1)).lower()
                        try:
                            dims = tuple(int(_eval_expr(d, parameters)) for d in _split_top_level(m_arr2.group(3), ","))
                            declared_shapes[vname] = dims
                        except (KeyError, ValueError):
                            pass
                        declared_types[vname] = np.dtype(f"S{m_arr2.group(2)}")
                        continue

                    m_scalar = re.match(r"^([a-zA-Z0-9_ ]+)(?:\s*\*\s*(\d+))?$", decl)
                    if m_scalar:
                        vname = re.sub(r"\s+", "", m_scalar.group(1)).lower()
                        if m_scalar.group(2):
                            declared_types[vname] = np.dtype(f"S{m_scalar.group(2)}")
                        else:
                            declared_types[vname] = base_dt

        # PASS 2: Process DATA statements
        unit_arrays: dict[str, np.ndarray] = {}
        unit_masks: dict[str, np.ndarray] = {}

        for stmt in stmts:
            s = stmt.strip()
            # Fortran ignores blanks: "DATAC10/.../" is a DATA statement; "DATAX = ..." is an assignment
            if not (re.match(r"^DATA", s, re.I) and "/" in s
                    and not re.match(r"^DATA\w*\s*(\([^)]*\))?\s*=", s, re.I)):
                continue

            pairs = _split_data_pairs(s)
            for var_list_str, val_list_str in pairs:
                # 1. Expand values
                raw_val_tokens = _split_val_list(val_list_str)
                expanded_vals: list[str] = []
                for tok in raw_val_tokens:
                    tok = tok.strip()
                    # Check repeat count r*c
                    # Don't split if * is inside quotes
                    if tok.startswith("\x27"):
                        expanded_vals.append(tok)
                    elif "*" in tok:
                        # Find repeat multiplier
                        parts = tok.split("*", 1)
                        r_str = parts[0].strip()
                        c_str = parts[1].strip()
                        # Verify r_str is an integer or parameter
                        try:
                            rep = int(_eval_expr(r_str, parameters))
                        except Exception:
                            # Not a repeat, keep as is
                            rep = 1
                            c_str = tok
                        for _ in range(rep):
                            expanded_vals.append(c_str)
                    else:
                        expanded_vals.append(tok)

                # 2. Expand targets
                target_tokens = _split_top_level(var_list_str, ",")
                expanded_targets: list[tuple[str, int]] = []
                for t_tok in target_tokens:
                    expanded_targets.extend(
                        _expand_target(t_tok, {}, declared_shapes, parameters)
                    )

                # 3. Assign values to targets
                val_idx = 0
                for var_name, lin_idx in expanded_targets:
                    if val_idx >= len(expanded_vals):
                        break
                    val_str = expanded_vals[val_idx]
                    val_idx += 1

                    # Look up variable dtype and shape
                    var_dtype = declared_types.get(
                        var_name,
                        implicit_types.get(var_name[0], np.dtype(np.float32)),
                    )
                    var_shape = declared_shapes.get(var_name, ())

                    if var_name not in unit_arrays:
                        unit_arrays[var_name] = np.zeros(
                            var_shape, dtype=var_dtype, order="F"
                        )

                    if var_name not in unit_masks:
                        unit_masks[var_name] = np.zeros(max(1, int(np.prod(var_shape))), dtype=bool)
                    unit_masks[var_name][lin_idx if var_shape else 0] = True
                    arr = unit_arrays[var_name]
                    typed_val = _convert_val(
                        val_str, var_dtype, parameters, compiler=compiler
                    )

                    if not var_shape:
                        arr[()] = typed_val
                    else:
                        arr.reshape(-1, order="F")[lin_idx] = typed_val

                    result[(unit_name, var_name)] = arr

        # PASS 3: EQUIVALENCE — storage association between DATA-initialised members and their aliases
        groups = []
        for stmt in stmts:
            st = stmt.strip()
            if not re.match(r"^EQUIVALENCE\b", st, re.I):
                continue
            for grp in re.findall(r"\(((?:[^()]|\([^()]*\))*)\)", st[len("EQUIVALENCE"):]):
                members = []
                for item in _split_top_level(grp, ","):
                    m = re.match(r"^\s*([a-zA-Z0-9_]+)\s*(?:\((.*)\))?\s*$", item)
                    if m:
                        members.append((m.group(1).lower(), m.group(2)))
                if len(members) > 1:
                    groups.append(members)
        if groups:
            def dtype_of(v):
                return declared_types.get(v, implicit_types.get(v[0], np.dtype(np.float32)))

            def elem_offset(v, subs):  # byte offset of the referenced element from the start of v
                if not subs:
                    return 0
                shape = declared_shapes.get(v, ())
                idx = [(int(x) if x.strip().isdigit() else int(_eval_expr(x, parameters))) - 1
                       for x in _split_top_level(subs, ",")]
                if len(idx) == 1 or not shape:
                    lin = idx[0]
                else:
                    lin, mul = 0, 1
                    for k, i in enumerate(idx):
                        lin += i * mul
                        mul *= shape[k]
                return lin * dtype_of(v).itemsize

            # union-find with offsets: start[v] relative to root
            parent, rel = {}, {}

            def find(v):
                if parent.setdefault(v, v) == v:
                    rel.setdefault(v, 0)
                    return v, 0
                r, o = find(parent[v])
                parent[v], rel[v] = r, rel[v] + o
                return r, rel[v]

            for members in groups:
                a, asub = members[0]
                for b, bsub in members[1:]:
                    # element a(asub) and b(bsub) share an address: start_b = start_a + off_a - off_b
                    ra, oa = find(a)
                    rb, ob = find(b)
                    want = oa + elem_offset(a, asub) - elem_offset(b, bsub)  # start_b relative to ra
                    if ra != rb:
                        parent[rb], rel[rb] = ra, want - ob
            storages = {}
            for v in list(parent):
                r, o = find(v)
                storages.setdefault(r, []).append((v, o))
            for r, mem in storages.items():
                if not any(v in unit_arrays for v, _ in mem):
                    continue
                lo = min(o for _, o in mem)
                size = max(o - lo + max(1, int(np.prod(declared_shapes.get(v, ())))) * dtype_of(v).itemsize for v, o in mem)
                buf = np.zeros(size, dtype=np.uint8)
                for v, o in mem:
                    if v in unit_arrays:
                        a = unit_arrays[v]
                        raw = np.ascontiguousarray(a.reshape(-1, order="F")).view(np.uint8)
                        isz = a.dtype.itemsize
                        msk = np.repeat(unit_masks[v], isz)
                        seg = buf[o - lo:o - lo + raw.size]
                        seg[msk] = raw[msk]
                for v, o in mem:
                    dt = dtype_of(v)
                    shape = declared_shapes.get(v, ())
                    n = max(1, int(np.prod(shape)))
                    arr = buf[o - lo:o - lo + n * dt.itemsize].view(dt).copy()
                    result[(unit_name, v)] = arr.reshape(shape, order="F") if shape else arr.reshape(())

    return result


def load_tables(source_path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    """Load DATA tables from source_path with an on-disk npz cache under ~/.cache/atlas_torch/."""
    source_p = Path(source_path).resolve()
    if not source_p.exists():
        raise FileNotFoundError(f"Source file not found: {source_p}")

    content = source_p.read_bytes()
    sha = hashlib.sha256(content).hexdigest()

    cache_dir = Path.home() / ".cache" / "atlas_torch"
    cache_path = cache_dir / f"{sha}.npz"

    if cache_path.exists():
        with np.load(cache_path) as npz:
            return {
                (k.split("/", 1)[0], k.split("/", 1)[1]): npz[k]
                for k in npz.files
            }

    data = parse_data(source_p)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        **{f"{unit}/{var}": arr for (unit, var), arr in data.items()},
    )
    return data


def table(source_name: str, unit: str, var: str) -> np.ndarray:
    """Convenience function that finds the source under $ATLAS_EXT/builds/src and returns the array."""
    atlas_ext = os.environ.get("ATLAS_EXT", str(Path.home() / "atlas-ext"))
    source_dir = Path(atlas_ext) / "builds" / "src"

    p = source_dir / source_name
    if not p.exists() and not source_name.endswith(".for"):
        p = source_dir / f"{source_name}.for"
    if not p.exists():
        raise FileNotFoundError(f"Kurucz source file not found: {p}")

    tables = load_tables(p)
    key = (unit.lower(), var.lower())
    if key not in tables:
        raise KeyError(f"Variable {var} not found in unit {unit} of {source_name}")
    return tables[key]
