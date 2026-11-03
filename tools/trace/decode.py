"""Companion to gdbtrace.py.

  python tools/trace/decode.py --layout <exe-or-lib> <layout.json>   # write layout for gdbtrace (TRACE_LAYOUT)
  python tools/trace/decode.py <exe-or-lib> <trace_dir>              # raw.bin + index.jsonl -> <routine>_<hit>_<phase>.npz
Decoded npz keys: arg:<name>, static:<name>, <block>.<var>  (numpy arrays in Fortran shape).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from atlas_torch.testing.fortran_oracle import read_layout  # noqa: E402


def write_layout(lib, out):
    commons, units, syms = read_layout(str(Path(lib).resolve()))
    size = {}
    for b, vars_ in commons.items():
        base = min(v.addr for v in vars_.values())
        size[b] = max(v.addr + v.nbytes for v in vars_.values()) - base
    json.dump({"unit_commons": {u: sorted(x.commons) for u, x in units.items()}, "common_size": size},
              open(out, "w"))


def decode(lib, d):
    commons, units, _ = read_layout(str(Path(lib).resolve()))
    d = Path(d)
    raw = (d / "raw.bin").read_bytes()
    groups = defaultdict(dict)
    for ln in (d / "index.jsonl").read_text().splitlines():
        r = json.loads(ln)
        key = (r["routine"], r["hit"], r["phase"])
        data = raw[r["off"]:r["off"] + r["size"]]
        if r["tag"] == "common":
            vars_ = commons[r["name"]]
            base = min(v.addr for v in vars_.values())
            for v in vars_.values():
                o = v.addr - base
                a = np.frombuffer(data[o:o + v.nbytes], dtype=v.dtype)
                groups[key][f"{r['name']}.{v.name}"] = a.reshape(v.shape, order="F") if v.shape else a
        else:
            u = units.get(r["routine"])
            dt = None
            if r["tag"] == "arg" and u:
                for pn, pdt, psh in u.params:
                    if pn == r["name"]:
                        dt = pdt
            elif r["tag"] == "static" and u and r["name"].lower() in u.statics:
                dt = u.statics[r["name"].lower()].dtype
            elif r["tag"] == "static" and u:
                # gdb lists COMMON members as statics of the routine: map back to <block>.<var>
                for b in sorted(u.commons):
                    v = commons[b].get(r["name"].lower())
                    if v is not None and v.nbytes == len(data):
                        a = np.frombuffer(data, dtype=v.dtype)
                        groups[key][f"{b}.{v.name}"] = a.reshape(v.shape, order="F") if v.shape else a
                        break
                else:
                    groups[key][f"static:{r['name']}"] = np.frombuffer(data, dtype="u1")
                continue
            groups[key][f"{r['tag']}:{r['name']}"] = np.frombuffer(data, dtype=dt or "u1")
    for (rt, hit, ph), arrs in groups.items():
        np.savez(d / f"{rt}_{hit}_{ph}.npz", **arrs)
    return len(groups)


if __name__ == "__main__":
    if sys.argv[1] == "--layout":
        write_layout(sys.argv[2], sys.argv[3])
    else:
        print(decode(sys.argv[1], sys.argv[2]), "records")
