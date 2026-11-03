#!/usr/bin/env bash
# Build -g -fPIC shared objects of the (patched) Fortran sources for ctypes oracles.
# Stock compiler per program family (ifx: xnfdf/dfsynthe/kappa9; gfortran: atlas9mem + SYNTHE family),
# stock SAVE/zero-init semantics, -O0 (oracles are compared within the Fortran noise floor anyway).
# Output: $ATLAS_EXT/oracle/lib<name>.so
set -euo pipefail
EXT="${ATLAS_EXT:-$HOME/atlas-ext}"; S="$EXT/builds/src"; O="$EXT/oracle"; mkdir -p "$O"; cd "$O"
GF="gfortran -g -O0 -fPIC -shared -fno-automatic -finit-local-zero -w -std=legacy -ffpe-summary=none"
IF="ifx -g -O0 -fPIC -shared -save"
$GF -o libatlas7v.so   "$S/atlas7v.for"
$GF -o libxnfpelsyn.so "$S/xnfpelsyn.for" "$S/atlas7v.for"
$GF -o libspectrv.so   "$S/spectrv.patched.for" "$S/atlas7v.for"
$GF -o libsynthe.so    "$S/synthe.patched.for"
$GF -o libatlas9.so    "$S/atlas9mem.patched.for"
$IF -o libkappa9.so    "$S/kappa9.for"   2>/dev/null
$IF -o libxnfdf.so     "$S/xnfdf.for"    2>/dev/null
$IF -o libdfsynthe.so  "$S/dfsynthe.for" 2>/dev/null
ls -la "$O"
