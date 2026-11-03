"""Portability shim so the Intel-dialect DFSYNTHE/KAPPA9 sources compile with gfortran.

Used ONLY for the gfortran noise-floor builds (gf-O0/gf-O3/fpe); the stock reference is compiled with
ifx from the unmodified sources. Rewrites are I/O-only:
  OPEN(..., TYPE='X' ...)  -> STATUS='X';  READONLY / SHARED removed;  NAME= -> FILE=
  STATUS='NEW' without FILE= -> adds FILE='fort.N' (what ifx does implicitly; also STATUS='UNKNOWN' so reruns work)
  tab-led statement lines longer than 72 columns are split into a continuation line
    (ifx accepts long tab-format lines; gfortran truncates at column 72).
usage: python tools/gf_shim.py in.for out.for
"""
import re
import sys


def shim(text: str) -> str:
    lines = text.splitlines()
    out = []
    for ln in lines:
        if ln[:1] in "cC*!":
            out.append(ln)
            continue
        if ln.startswith("\t"):
            body = ln[1:]
            if len(body) > 66:
                out.append("      " + body[:66])
                rest = body[66:]
                while rest:
                    out.append("     &" + rest[:66])
                    rest = rest[66:]
                continue
            ln = "      " + body
        out.append(ln)
    # join OPEN statements (with continuations) to rewrite them
    res, i = [], 0
    while i < len(out):
        ln = out[i]
        if ln[:1] not in "cC*!" and re.search(r"\bOPEN\s*\(", ln[6:72], re.I):
            stmt = [ln[:72]]
            j = i + 1
            while j < len(out) and len(out[j]) > 5 and out[j][:1] not in "cC*!" and out[j][5] not in " 0":
                stmt.append(out[j][:72])
                j += 1
            label = stmt[0][:6]
            body = stmt[0][6:] + "".join(s[6:] for s in stmt[1:])
            body = re.sub(r"\bTYPE\s*=", "STATUS=", body, flags=re.I)
            body = re.sub(r",\s*(READONLY|SHARED)\b", "", body, flags=re.I)
            body = re.sub(r"\b(READONLY|SHARED)\s*,", "", body, flags=re.I)
            body = re.sub(r"\bNAME\s*=", "FILE=", body, flags=re.I)
            body = re.sub(r",\s*RECORDTYPE\s*=\s*'[^']*'", "", body, flags=re.I)
            if re.search(r"STATUS\s*=\s*'NEW'", body, re.I) and not re.search(r"\bFILE\s*=", body, re.I):
                m = re.search(r"UNIT\s*=\s*(\d+)", body, re.I)
                body = re.sub(r"STATUS\s*=\s*'NEW'", f"STATUS='UNKNOWN',FILE='fort.{m.group(1)}'", body, flags=re.I)
            body = body.strip()
            chunks = [body[k:k + 60] for k in range(0, len(body), 60)]
            res.append(label + chunks[0])
            res.extend("     &" + c for c in chunks[1:])
            i = j
            continue
        res.append(ln)
        i += 1
    return "\n".join(res) + "\n"


if __name__ == "__main__":
    open(sys.argv[2], "w").write(shim(open(sys.argv[1]).read()))
