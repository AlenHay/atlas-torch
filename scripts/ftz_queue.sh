#!/usr/bin/env bash
# DFSYNTHE/KAPPA9 noise variants in the stock floating-point environment (flush-to-zero), see RESULTS.md
cd "$(dirname "$0")/.."; EXT="${ATLAS_EXT:-$HOME/atlas-ext}"; M="$EXT/runs/.done"; mkdir -p "$M"
for v in ifx-precise-ftz ifx-O0-ftz; do for o in solar alpha; do k=odf_${v}_$o; [ -f "$M/$k" ] && continue
  echo "[$(date +%T)] start $k"; if nice -n 19 .venv/bin/python scripts/run_fortran.py $v odf $o --parallel > "$EXT/runs/log_$k.txt" 2>&1; then touch "$M/$k"; echo "[$(date +%T)] ok $k"; else echo "[$(date +%T)] FAIL $k"; fi; done; done
for v in ifx-precise-ftz ifx-O0-ftz; do for m in solar alpha; do nice -n 19 scripts/kappa9_workdir.sh $EXT/builds/$v/bin $EXT/runs/stock/odf/$m $EXT/work/kappa9_${m}_$v > /dev/null; done; done
echo "[$(date +%T)] ftz queue finished"
