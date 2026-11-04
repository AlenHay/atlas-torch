#!/usr/bin/env bash
# All SYNTHE windows x modes of the M7 study, sequentially (one heavy job at a time). Log: stdout.
#   tools/bench/gpu_synthe_all.sh [windows=0,1,2,3]
set -uo pipefail
cd "$(dirname "$0")/../.."
W="${1:-0,1,2,3}"
tools/bench/gpu_run.sh cpu tools/bench/gpu_synthe.py prepare --windows "$W"
for m in exact native cpu32 cpu32p; do tools/bench/gpu_run.sh cpu tools/bench/gpu_synthe.py run --mode $m --windows "$W"; done
for m in gpu64 gpu32 gpu32p; do tools/bench/gpu_run.sh gpu tools/bench/gpu_synthe.py run --mode $m --windows "$W" 2>&1 | grep -v amdgpu.ids; done
