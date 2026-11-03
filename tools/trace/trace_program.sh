#!/usr/bin/env bash
# Golden trace of one Fortran program run in a prepared work dir (see scripts/synthe_workdir.sh).
#   tools/trace/trace_program.sh <exe> <workdir> <outdir> <stdin-file|-> [routines|ALL] [maxhits]
# ALL = every subprogram the executable's DWARF knows (except the main program).
set -euo pipefail
EXE="$(realpath "$1")"; W="$(realpath "$2")"; OUT="$3"; IN="$4"; R="${5:-ALL}"; MAX="${6:-3}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"; PY="$repo/.venv/bin/python"
mkdir -p "$OUT"; OUT="$(realpath "$OUT")"
$PY "$repo/tools/trace/decode.py" --layout "$EXE" "$OUT/layout.json"
if [ "$R" = ALL ]; then
  R=$($PY -c "import json,sys; d=json.load(open('$OUT/layout.json')); print(','.join(u for u in d['unit_commons'] if u not in ('main','main__') and not u.startswith('_')))")
fi
cd "$W"
TRACE_ROUTINES="$R" TRACE_MAX="$MAX" TRACE_OUT="$OUT" TRACE_LAYOUT="$OUT/layout.json" \
  TRACE_STDIN="$([ "$IN" = - ] || realpath "$IN")" gdb -q -batch -x "$repo/tools/trace/gdbtrace.py" --args "$EXE" > "$OUT/gdb.log" 2>&1 || true
$PY "$repo/tools/trace/decode.py" "$EXE" "$OUT"
rm -f "$OUT/raw.bin"
du -sh "$OUT"
