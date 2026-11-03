#!/usr/bin/env bash
# Produce all Fortran reference + noise-floor runs (inside the container). Idempotent per step via markers.
set -uo pipefail
cd "$(dirname "$0")/.."
EXT="${ATLAS_EXT:-$HOME/atlas-ext}"; M="$EXT/runs/.done"; mkdir -p "$M"
PY=.venv/bin/python
step() { local key="$1"; shift; [ -f "$M/$key" ] && return 0
  echo "[$(date +%T)] start $key"; if "$@" > "$EXT/runs/log_$key.txt" 2>&1; then touch "$M/$key"; echo "[$(date +%T)] ok $key"; else echo "[$(date +%T)] FAIL $key"; fi; }
for o in solar alpha m10 m20; do step odf_stock_$o $PY scripts/run_fortran.py stock odf $o --parallel; done
for m in sun t4500g25 t6500g40m10 t9000g40 t5000g45m20 t5500g40alpha; do step atlas_stock_$m $PY scripts/run_fortran.py stock atlas $m; done
step synthe_stock_sun $PY scripts/run_fortran.py stock synthe sun
for v in gf-O0 gf-O3 ifx-O0 ifx-O2 ifx-O3 ifx-precise; do
  for m in sun t4500g25 t6500g40m10 t9000g40 t5000g45m20 t5500g40alpha; do step atlas_${v}_$m $PY scripts/run_fortran.py $v atlas $m --odf-from stock; done
  step synthe_${v}_sun $PY scripts/run_fortran.py $v synthe sun --model-from stock
done
for v in ifx-O3 ifx-precise ifx-O0; do for o in solar alpha; do step odf_${v}_$o $PY scripts/run_fortran.py $v odf $o --parallel; done; done

step examples_stock $PY scripts/run_examples.py stock
echo "[$(date +%T)] examples finished"
