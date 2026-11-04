#!/usr/bin/env bash
# ATLAS9 models x modes of the M7 study, sequentially.  tools/bench/gpu_atlas9_all.sh "<models>" "<cpu modes>" "<gpu modes>"
set -uo pipefail
cd "$(dirname "$0")/../.."
MODELS="${1:-sun t4500g25 t6500g40m10 t9000g40 t5000g45m20 t5500g40alpha}"
CPU="${2:-fortran exact native}"; GPU="${3:-}"
for m in $MODELS; do
  for mode in $CPU; do tools/bench/gpu_run.sh cpu tools/bench/gpu_atlas9.py --model $m --mode $mode 2>&1 | tail -1; done
  for mode in $GPU; do tools/bench/gpu_run.sh gpu tools/bench/gpu_atlas9.py --model $m --mode $mode 2>&1 | grep -v amdgpu.ids | tail -1; done
done
