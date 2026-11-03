#!/usr/bin/env bash
# Run the SYNTHE chain step by step in a work dir, KEEPING all intermediate fort.* files, so that any
# single program can be re-run (under gdb, or replaced by its torch port) on identical inputs.
# Mirrors BasicATLAS templates.synthe_control (line lists, molecule list, fort.25 card) without cleanup.
#   scripts/synthe_workdir.sh <bindir> <output_synthe.out> <wlbeg nm> <wlend nm> <workdir> [res] [vturb] [linelist]
set -euo pipefail
BIN="$1"; MODEL="$(realpath "$2")"; WB="$3"; WE="$4"; W="$5"; RES="${6:-600000.}"; VT="${7:-1.5}"
EXT="${ATLAS_EXT:-$HOME/atlas-ext}"; S="$EXT/BasicATLAS/data/synthe_files"; D="$EXT/BasicATLAS/data/dfsynthe_files"
LL="${8:-$S/BasicATLAS.dat}"
mkdir -p "$W"; cd "$W"; chmod -R u+w . 2>/dev/null; rm -rf fort.* c12c13.dat pre_synthe pre_spectrv
ln -s "$S/molecules.dat" fort.2; ln -s "$S/continua.dat" fort.17
"$BIN/xnfpelsyn.exe" < "$MODEL" > xnfpelsyn.out; cp fort.10 xnfpelsyn.dat
printf "%-3s       %-10.4f%-10.4f%-9.2f %-10.4f%-3d%-7d%-10.5f%-5d%-5d\nAIRorVAC  WLBEG     WLEND     RESOLU    TURBV  IFNLTE LINOUT CUTOFF        NREAD\n" \
  VAC "$WB" "$WE" "$RES" "$VT" 0 -1 0.0001 1 0 > synbeg.in
"$BIN/synbeg.exe" < synbeg.in > synbeg.out
ln -sf "$LL" fort.11; "$BIN/rgfalllinesnew.exe" > rgfalllinesnew.out; rm fort.11
for m in chmasseron_corrected.asc mgh_exomol.asc nh.asc ohupdate.asc sihax.asc h2.asc h2xx.asc hdxx.asc c2ax.asc c2ba.asc \
         c2dabrookek.asc c2ea.asc cnaxbrookek.asc cnbxbrookek.asc cnxx12brooke.asc coax.asc coxx.asc sioax.asc sioex.asc sioxx.asc tiototo.asc; do
  ln -sf "$S/$m" fort.11; "$BIN/rmolecasc.exe" > "rmol_${m%.asc}.out"; rm fort.11; done
ln -sf "$D/h2ofastfix.bin" fort.11; "$BIN/rh2ofast.exe" > h2ofastfix.out; rm fort.11
mkdir -p pre_synthe; for f in fort.*; do cp -L "$f" pre_synthe/; done; chmod -R a-w pre_synthe   # snapshot of synthe.exe inputs
ln -sf xnfpelsyn.dat fort.10; ln -sf "$S/he1tables.dat" fort.18
"$BIN/synthe.exe" > synthe.out
mkdir -p pre_spectrv; for f in fort.*; do cp -L "$f" pre_spectrv/; done; chmod -R a-w pre_spectrv
printf "0.0       0.        1.        0.        0.        0.        0.        0.\n0.\nRHOXJ     R1        R101      PH1       PC1       PSI1      PRDDOP    PRDPOW\n" > fort.25
"$BIN/spectrv.exe" < "$MODEL" > spectrv.out
cp fort.7 spectrum.bin
echo "workdir ready: $W"
