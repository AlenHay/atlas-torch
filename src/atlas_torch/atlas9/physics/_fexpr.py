"""Tiny evaluator for Fortran arithmetic expressions with exact literal kinds.

Used for PFGROUND's case table (read from the source at run time). Kinds follow Fortran rules:
REAL*4 literals ('1.5', '2.E3') are binary32 and REAL*4 op REAL*4 is evaluated in binary32 (as gfortran's
constant folding does); 'D' literals are REAL*8; integer literals are INTEGER. Mixed operations promote to
the wider kind. Constant integer powers use gfortran's powi chains, real powers libm pow, EXP libm exp.
Variables are bound by the caller (e.g. T -> tensor, HCK -> float).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import torch

from . import libm
from ._data import r4

_TOK = re.compile(r"\s*(?:(\d+\.\d*(?:[EeDd][+-]?\d+)?|\.\d+(?:[EeDd][+-]?\d+)?|\d+[EeDd][+-]?\d+)|(\d+)|(\w+)|(\*\*|[-+*/(),]))")


def _tokens(s):
    pos, out = 0, []
    s = s.strip()
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m:
            raise SyntaxError(f"bad token at {s[pos:]!r}")
        pos = m.end()
        if m.group(1):
            t = m.group(1)
            out.append(("num", t, "r8" if re.search(r"[dD]", t) else "r4"))
        elif m.group(2):
            out.append(("num", m.group(2), "i4"))
        elif m.group(3):
            out.append(("name", m.group(3).upper(), None))
        else:
            out.append(("op", m.group(4), None))
    return out


@lru_cache(maxsize=None)
def parse(src: str):
    toks = _tokens(src)
    i = 0

    def peek():
        return toks[i] if i < len(toks) else ("end", None, None)

    def take():
        nonlocal i
        t = toks[i]
        i += 1
        return t

    def expr():
        sign = None
        if peek()[0] == "op" and peek()[1] in "+-":
            sign = take()[1]
        node = term()
        if sign == "-":
            node = ("neg", node)
        while peek()[0] == "op" and peek()[1] in ("+", "-"):
            op = take()[1]
            node = (op, node, term())
        return node

    def term():
        node = factor()
        while peek()[0] == "op" and peek()[1] in ("*", "/"):
            op = take()[1]
            node = (op, node, factor())
        return node

    def factor():
        base = primary()
        if peek()[0] == "op" and peek()[1] == "**":
            take()
            return ("**", base, factor())
        return base

    def primary():
        t = take()
        if t[0] == "num":
            return ("num", t[1], t[2])
        if t[0] == "name":
            if peek() == ("op", "(", None):
                take()
                args = [expr()]
                while peek() == ("op", ",", None):
                    take()
                    args.append(expr())
                assert take()[1] == ")"
                return ("call", t[1], args)
            return ("var", t[1])
        if t == ("op", "(", None):
            e = expr()
            assert take()[1] == ")", src
            return e
        raise SyntaxError(f"unexpected {t} in {src!r}")

    tree = expr()
    if i != len(toks):
        raise SyntaxError(f"trailing tokens in {src!r}")
    return tree


def _kind(v):
    if isinstance(v, torch.Tensor):
        return "r4" if v.dtype == torch.float32 else "r8"
    if isinstance(v, np.float32):
        return "r4"
    if isinstance(v, (int, np.integer)):
        return "i4"
    return "r8"


def _to8(v):
    if isinstance(v, torch.Tensor):
        return v.double()
    return float(v)


def _binop(op, a, b):
    ka, kb = _kind(a), _kind(b)
    if ka == kb == "i4":
        if op == "/":
            q = abs(a) // abs(b)
            return q if (a >= 0) == (b >= 0) else -q
        return {"+": a + b, "-": a - b, "*": a * b}[op]
    if ka == kb == "r4" or {ka, kb} == {"r4", "i4"}:
        a = np.float32(a) if not isinstance(a, torch.Tensor) else a
        b = np.float32(b) if not isinstance(b, torch.Tensor) else b
    else:
        a, b = _to8(a), _to8(b)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return libm.rdiv(a, b) if isinstance(b, torch.Tensor) and not isinstance(a, torch.Tensor) else a / b


def evaluate_slow(src: str, env: dict):
    def ev(n):
        tag = n[0]
        if tag == "num":
            txt, k = n[1], n[2]
            if k == "i4":
                return int(txt)
            if k == "r4":
                return np.float32(r4(txt))
            return float(txt.lower().replace("d", "e"))
        if tag == "var":
            return env[n[1]]
        if tag == "neg":
            return -ev(n[1])
        if tag == "call":
            fn = n[1]
            args = [ev(a) for a in n[2]]
            if fn in ("EXP", "DEXP"):
                return libm.exp(args[0])
            if fn in ("SQRT", "DSQRT"):
                return libm.sqrt(args[0])
            raise NotImplementedError(fn)
        if tag == "**":
            base, ex = ev(n[1]), n[2]
            if ex[0] == "num" and ex[2] == "i4":
                return libm.ipow_const(base, int(ex[1]))
            e = ev(ex)
            if _kind(e) == "i4":
                return libm.ipow_var(base, e)
            if _kind(base) == "r4" and _kind(e) == "r4":
                return libm.pow(np.float32(base), np.float32(e))
            return libm.pow(_to8(base), _to8(e))
        return _binop(tag, ev(n[1]), ev(n[2]))

    return ev(parse(src))


@lru_cache(maxsize=None)
def compile_expr(src: str):
    """Compile `src` to a function env -> value (literal conversion done once)."""
    def comp(n):
        tag = n[0]
        if tag == "num":
            txt, k = n[1], n[2]
            v = int(txt) if k == "i4" else np.float32(r4(txt)) if k == "r4" else float(txt.lower().replace("d", "e"))
            return lambda env: v
        if tag == "var":
            name = n[1]
            return lambda env: env[name]
        if tag == "neg":
            f = comp(n[1])
            return lambda env: -f(env)
        if tag == "call":
            fs = [comp(a) for a in n[2]]
            fn = {"EXP": libm.exp, "DEXP": libm.exp, "SQRT": libm.sqrt, "DSQRT": libm.sqrt}[n[1]]
            return lambda env: fn(*[f(env) for f in fs])
        if tag == "**":
            fb, ex = comp(n[1]), n[2]
            if ex[0] == "num" and ex[2] == "i4":
                k = int(ex[1])
                return lambda env: libm.ipow_const(fb(env), k)
            fe = comp(ex)

            def pw(env):
                base, e = fb(env), fe(env)
                if _kind(e) == "i4":
                    return libm.ipow_var(base, e)
                if _kind(base) == "r4" and _kind(e) == "r4":
                    return libm.pow(np.float32(base), np.float32(e))
                return libm.pow(_to8(base), _to8(e))
            return pw
        fa, fb2 = comp(n[1]), comp(n[2])
        return lambda env: _binop(tag, fa(env), fb2(env))
    return comp(parse(src))


def evaluate(src: str, env: dict):
    return compile_expr(src)(env)
