"""gdb-Python golden-trace recorder for the Fortran programs.

Run (inside the container, from the program's working directory):
  gdb -q -batch -x tools/trace/gdbtrace.py --args <exe>   with env:
    TRACE_ROUTINES = "hop,pops,nelect"   routines to trace (Fortran names, case-insensitive)
    TRACE_MAX      = 3                   max recorded calls per routine (entry + exit)
    TRACE_OUT      = /path/trace_dir     output dir (raw.bin + index.jsonl)
    TRACE_STDIN    = file                optional stdin redirection for the program
    TRACE_LIB      = /path/lib.so        optional: library whose DWARF lists the COMMONs of each routine
                                         (default: the executable itself; build it with -g)
For every recorded hit, at entry and at exit, it dumps: each formal argument (gdb-evaluated runtime size,
so adjustable arrays come out whole), every COMMON block the routine declares (whole block by symbol size)
and every static (SAVEd/DATA) local of the routine. tools/trace/decode.py turns the dump into .npz files
using the DWARF layout (atlas_torch.testing.fortran_oracle.read_layout).
"""
import json
import os

import gdb

routines = [r.strip().lower() for r in os.environ.get("TRACE_ROUTINES", "").split(",") if r.strip()]
maxhits = int(os.environ.get("TRACE_MAX", "3"))
out = os.environ["TRACE_OUT"]
os.makedirs(out, exist_ok=True)
raw = open(os.path.join(out, "raw.bin"), "wb")
idx = open(os.path.join(out, "index.jsonl"), "w")
hits = {r: 0 for r in routines}

gdb.execute("set pagination off")
gdb.execute("set confirm off")
gdb.execute("set breakpoint pending on")


_last = []


def dump(tag, name, addr, size, meta):
    _last.append((tag, name, addr, size))
    try:
        data = bytes(gdb.selected_inferior().read_memory(addr, size))
    except gdb.MemoryError:
        return
    off = raw.tell()
    raw.write(data)
    rec = dict(meta, tag=tag, name=name, addr=int(addr), size=int(size), off=off)
    idx.write(json.dumps(rec) + "\n")


def symaddr(sym):
    try:
        v = gdb.parse_and_eval("(long)&'%s'" % sym)
        return int(v)
    except gdb.error:
        return None


def symsize(sym):
    try:
        return int(gdb.parse_and_eval("sizeof('%s')" % sym))
    except gdb.error:
        return None


def record(routine, phase, frame, hit):
    meta = {"routine": routine, "phase": phase, "hit": hit}
    blk = gdb.block_for_pc(frame.pc())
    # formal arguments + statics visible in the function block
    fblock = blk
    while fblock and fblock.function is None:
        fblock = fblock.superblock
    seen = set()
    if fblock is not None:
        for s in fblock:
            if s.name in seen:
                continue
            seen.add(s.name)
            try:
                if s.is_argument:
                    v = s.value(frame)
                    while v.type.code in (gdb.TYPE_CODE_PTR, gdb.TYPE_CODE_REF):
                        v = v.referenced_value()
                    if v.address is not None:
                        dump("arg", s.name, int(v.address), v.type.sizeof, meta)
                elif s.addr_class == gdb.SYMBOL_LOC_STATIC:
                    v = s.value(frame)
                    if v.address is not None:
                        dump("static", s.name, int(v.address), v.type.sizeof, meta)
            except (gdb.error, RuntimeError) as e:
                idx.write(json.dumps(dict(meta, tag="error", name=s.name, msg=str(e))) + "\n")
    # COMMON blocks: gdb exposes them as globals named <block>_
    for cb in COMMONS.get(routine, []):
        a = symaddr(cb + "_")
        n = COMMON_SIZE.get(cb)
        if a is not None and n:
            dump("common", cb, a, n, meta)


class ExitBP(gdb.FinishBreakpoint):
    def __init__(self, frame, routine, hit, items):
        super().__init__(frame, internal=True)
        self.routine, self.hit, self.items = routine, hit, items

    def stop(self):
        # the callee frame is gone: re-read the same memory captured at entry (args are by reference)
        meta = {"routine": self.routine, "phase": "exit", "hit": self.hit}
        for tag, name, addr, size in self.items:
            dump(tag, name, addr, size, meta)
        return False

    def out_of_scope(self):
        pass


class EntryBP(gdb.Breakpoint):
    def __init__(self, routine):
        super().__init__(routine + "_", internal=False)
        self.routine = routine

    def stop(self):
        h = hits[self.routine]
        if h >= maxhits:
            self.enabled = False
            return False
        hits[self.routine] = h + 1
        f = gdb.newest_frame()
        _last.clear()
        record(self.routine, "entry", f, h)
        ExitBP(f, self.routine, h, list(_last))
        return False


# COMMON membership per routine + block sizes from the DWARF layout file written by decode.py --layout
lay = json.load(open(os.environ["TRACE_LAYOUT"]))
COMMONS = lay["unit_commons"]
COMMON_SIZE = lay["common_size"]

for r in routines:
    EntryBP(r)
stdin = os.environ.get("TRACE_STDIN")
gdb.execute("run" + (" < " + stdin if stdin else ""))
raw.close()
idx.close()
