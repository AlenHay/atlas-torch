#!/usr/bin/env bash
# Usage (host): tools/bench/atlas9_run.sh <label> [iters] [core] [extra args]  -> 15-iteration solar model, step timers,
# byte-identical check vs the cached stock run; JSON in $ATLAS_EXT/scratch/atlas9_bench/<label>.json, summary on stdout.
label=$1; iters=${2:-15}; core=${3:-5}; shift 3 2>/dev/null
cd "$(dirname "$0")/../.." && scripts/dx "taskset -c $core .venv/bin/python tools/bench/atlas9_profile.py --iters $iters --steps --timers --check --json \$HOME/atlas-ext/scratch/atlas9_bench/$label.json --label $label $*" 2>&1 \
 | grep -v '^\s*\[\|Setting\|Container\|Integrat' | python3 -c "
import sys,json; d=json.loads(sys.stdin.read())
print({k:d.get(k) for k in ('torch_seconds','torch_cpu_seconds','fortran_seconds','fortran_cpu_seconds','identical_stdout','identical_fort7','loadavg_before','loadavg_after')})
print('steps', d.get('steps')); print('timers', {k:v[0] for k,v in d.get('timers',{}).items()})"
