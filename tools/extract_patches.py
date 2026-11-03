"""Extract the heredoc patches that BasicATLAS's compile.com applies to the Kurucz sources.

BasicATLAS has no license, so its patches are never stored in this repo: they are read at build
time from the pinned external checkout and applied with `patch`.

usage: python tools/extract_patches.py <BasicATLAS dir> <out dir>
writes <out dir>/<name>.diff for every `patch -o src/<name>.patched.for src/<name>.for <<EOF` block,
plus PyTLAS diffs are left in place (PyTLAS/src/*.diff in the checkout).
"""
import re
import sys
from pathlib import Path


def extract(compile_com: str) -> dict[str, str]:
    out = {}
    lines = compile_com.splitlines(keepends=True)
    i = 0
    pat = re.compile(r"^patch -o src/(\w+)\.patched\.for src/(\w+)\.for\s*<<\s*EOF")
    while i < len(lines):
        m = pat.match(lines[i])
        if m:
            name = m.group(2)
            body = []
            i += 1
            while lines[i].rstrip("\n") != "EOF":
                body.append(lines[i])
                i += 1
            out[name] = "".join(body)
        i += 1
    return out


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    patches = extract((src / "compile.com").read_text())
    for name, body in patches.items():
        (dst / f"{name}.diff").write_text(body)
        print(name, len(body.splitlines()), "lines")
