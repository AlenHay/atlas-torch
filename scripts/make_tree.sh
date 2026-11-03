#!/usr/bin/env bash
# Create a shadow BasicATLAS tree for a build variant: BasicATLAS python files (copied, outside this repo),
# data/ and restarts/ symlinked to the pinned checkout, bin/ -> $ATLAS_EXT/builds/<variant>/bin.
# "stock" uses the checkout itself (bin built by its own compile.com).
set -euo pipefail
v="$1"; EXT="${ATLAS_EXT:-$HOME/atlas-ext}"; BA="$EXT/BasicATLAS"
[ "$v" = stock ] && { echo "$BA"; exit 0; }
t="$EXT/trees/$v"; mkdir -p "$t"
cp -f "$BA"/*.py "$t/"
for d in data restarts src; do ln -sfn "$BA/$d" "$t/$d"; done
ln -sfn "$EXT/builds/$v/bin" "$t/bin"
echo "$t"
