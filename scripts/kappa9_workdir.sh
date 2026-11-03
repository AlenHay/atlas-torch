#!/usr/bin/env bash
# KAPPA9 + KAPREADTS step by step on an existing ODF run, keeping kapk{v}.dat (BasicATLAS deletes them).
#   scripts/kappa9_workdir.sh <bindir> <odf_run_dir> <workdir>
set -euo pipefail
BIN="$1"; ODF="$(realpath "$2")"; W="$3"; EXT="${ATLAS_EXT:-$HOME/atlas-ext}"; D="$EXT/BasicATLAS/data/dfsynthe_files"
mkdir -p "$W"; W="$(realpath "$W")"; cd "$W"; rm -f fort.* kapk*.dat kappa.ros
for v in 0 1 2 4 8; do
  cp "$D/molecules.dat" fort.2; cp "$ODF/p00big$v.bdf" fort.9
  sed -n '/<<EOF/,/^EOF/p' "$ODF/kappa9v$v.com" | sed '1d;$d' > stdin_v$v.txt
  "$BIN/kappa9.exe" < stdin_v$v.txt > kapm_v$v.out
  mv fort.7 kapk$v.dat; rm -f fort.9 fort.2
done
for v in 0 1 2 4 8; do case $v in 0) u=11;; 1) u=12;; 2) u=13;; 4) u=14;; 8) u=15;; esac; cp kapk$v.dat fort.$u; done
"$BIN/kapreadts.exe" > kapreadts.out; mv fort.2 kappa.ros; rm -f fort.*
echo "kappa9 workdir ready: $W"
