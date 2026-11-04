#!/usr/bin/env bash
# Run a tools/bench/gpu_*.py script with the interpreter of a mode, inside the atlas-dev container.
#   tools/bench/gpu_run.sh cpu|gpu <script.py> [args...]
# cpu: .venv (torch CPU, the reference env); gpu: $ATLAS_EXT/venv-rocm with HSA_OVERRIDE_GFX_VERSION=11.0.0.
# Long jobs run at nice 10. Set CORE=<n> to pin to one core (taskset).
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
kind="$1"; shift
ext="${ATLAS_EXT:-$HOME/atlas-ext}"
if [[ "$kind" == gpu ]]; then
  py="HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src $ext/venv-rocm/bin/python"
else
  py="PYTHONPATH=src .venv/bin/python"
fi
pin=""
if [[ -n "${CORE:-}" ]]; then pin="taskset -c $CORE"; fi
if [[ -n "${CONTAINER_ID:-}" ]]; then
  cd "$repo"; eval "nice -n 10 $pin env $py $*"
else
  exec "$repo/scripts/dx" "nice -n 10 $pin env $py $*"
fi
