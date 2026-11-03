#!/usr/bin/env bash
# Build the BasicATLAS Fortran programs in one of several compiler/flag variants.
# Run inside the atlas-dev container (scripts/dx). Sources come from the pinned BasicATLAS checkout
# ($ATLAS_EXT/BasicATLAS/src, fetched by its download.com); patches are extracted from its compile.com.
#
#   scripts/build_fortran.sh <variant>      variant in: prep stock gf-O0 gf-O3 ifx-O0 ifx-O2 ifx-O3 ifx-precise ifx-O0-ftz ifx-precise-ftz debug fpe
#
# Output: $ATLAS_EXT/builds/<variant>/bin/*.exe  (+ build.json with compilers/flags)
# "stock" reproduces compile.com exactly (ifx -save for DFSYNTHE/KAPPA9, gfortran for ATLAS9/SYNTHE).
set -euo pipefail
variant="$1"
EXT="${ATLAS_EXT:-$HOME/atlas-ext}"
BA="$EXT/BasicATLAS"
repo="$(cd "$(dirname "$0")/.." && pwd)"
out="$EXT/builds/$variant"; mkdir -p "$out/bin" "$EXT/builds/src"
psrc="$EXT/builds/src"
# patched sources: prepared by `build_fortran.sh prep` (run once, before parallel builds)
if [ "$variant" = prep ]; then
rm -rf "$psrc"; mkdir -p "$psrc"
python3 "$repo/tools/extract_patches.py" "$BA" "$psrc/patches" >/dev/null
for f in "$BA"/src/*.for; do cp -f "$f" "$psrc/"; done
for p in "$psrc"/patches/*.diff; do n=$(basename "$p" .diff); patch -s -o "$psrc/$n.patched.for" "$psrc/$n.for" < "$p"; done
for n in xnfdf dfsynthe separatedf dfsortp kappa9 kapreadts; do python3 "$repo/tools/gf_shim.py" "$psrc/$n.for" "$psrc/$n.gf.for"; done
exit 0
fi

GF_STD="-fno-automatic -w -std=legacy"
case "$variant" in
  stock)  DF_FC=ifx;      DF_FL="-save";  AT_FC=gfortran; AT_FL="-ffpe-summary=none -finit-local-zero $GF_STD"; SY_FC=gfortran; SY_FL="$GF_STD -O3";;
  gf-O0)  DF_FC=gfortran; DF_FL="$GF_STD -finit-local-zero -O0"; AT_FC=gfortran; AT_FL="$GF_STD -finit-local-zero -ffpe-summary=none -O0"; SY_FC=gfortran; SY_FL="$GF_STD -finit-local-zero -O0";;
  gf-O3)  DF_FC=gfortran; DF_FL="$GF_STD -finit-local-zero -O3"; AT_FC=gfortran; AT_FL="$GF_STD -finit-local-zero -ffpe-summary=none -O3"; SY_FC=gfortran; SY_FL="$GF_STD -finit-local-zero -O3";;
  ifx-O0) DF_FC=ifx; DF_FL="-save -zero -O0"; AT_FC=ifx; AT_FL="-save -zero -O0"; SY_FC=ifx; SY_FL="-save -zero -O0 -assume byterecl";;
  ifx-O2) DF_FC=ifx; DF_FL="-save -zero -O2"; AT_FC=ifx; AT_FL="-save -zero -O2"; SY_FC=ifx; SY_FL="-save -zero -O2 -assume byterecl";;
  ifx-O3) DF_FC=ifx; DF_FL="-save -zero -O3"; AT_FC=ifx; AT_FL="-save -zero -O3"; SY_FC=ifx; SY_FL="-save -zero -O3 -assume byterecl";;
  ifx-precise) DF_FC=ifx; DF_FL="-save -zero -O2 -fp-model precise"; AT_FC=ifx; AT_FL="-save -zero -O2 -fp-model precise"; SY_FC=ifx; SY_FL="-save -zero -O2 -fp-model precise -assume byterecl";;
  ifx-O0-ftz) DF_FC=ifx; DF_FL="-save -zero -O0 -ftz"; AT_FC=ifx; AT_FL="-save -zero -O0 -ftz"; SY_FC=ifx; SY_FL="-save -zero -O0 -ftz -assume byterecl";;
  ifx-precise-ftz) DF_FC=ifx; DF_FL="-save -zero -O2 -fp-model precise -ftz"; AT_FC=ifx; AT_FL="-save -zero -O2 -fp-model precise -ftz"; SY_FC=ifx; SY_FL="-save -zero -O2 -fp-model precise -ftz -assume byterecl";;
  debug)  DF_FC=ifx; DF_FL="-save -g -O0"; AT_FC=gfortran; AT_FL="-ffpe-summary=none -finit-local-zero $GF_STD -g -O0"; SY_FC=gfortran; SY_FL="$GF_STD -g -O0";;
  fpe)    F="$GF_STD -g -O0 -ffpe-trap=invalid,zero,overflow -finit-real=snan"
          DF_FC=gfortran; DF_FL="$F"; AT_FC=gfortran; AT_FL="$F"; SY_FC=gfortran; SY_FL="$F";;
  *) echo "unknown variant $variant"; exit 2;;
esac
cd "$out"
log="$out/build.log"; : > "$log"
b() { echo "+ $*" >> "$log"; "$@" >> "$log" 2>&1 || { echo "FAILED: $*" | tee -a "$log"; return 0; }; }
# ONLY=synthe rebuilds just the SYNTHE family (safe while DFSYNTHE runs use this bin dir)
if [ "${ONLY:-}" != synthe ]; then
# DFSYNTHE / KAPPA9 family
sfx=""; DFX=""; [ "$DF_FC" = gfortran ] && { sfx=".gf"; DFX="-fdec"; }
for p in xnfdf dfsynthe separatedf dfsortp kappa9 kapreadts; do b $DF_FC $DF_FL $DFX -o bin/$p.exe "$psrc/$p$sfx.for"; done
# ATLAS9 (patched)
b $AT_FC $AT_FL -o bin/atlas9mem.exe "$psrc/atlas9mem.patched.for"
fi
# SYNTHE family (SYNTHE sources assume byte RECL units: ifx needs -assume byterecl)
b $SY_FC $SY_FL -c "$psrc/atlas7v.for" -o atlas7v.o
b $SY_FC $SY_FL -c "$psrc/xnfpelsyn.for" -o xnfpelsyn.o
b $SY_FC $SY_FL -o bin/xnfpelsyn.exe xnfpelsyn.o atlas7v.o
b $SY_FC $SY_FL -c "$psrc/spectrv.patched.for" -o spectrv.o
b $SY_FC $SY_FL -o bin/spectrv.exe spectrv.o atlas7v.o
for p in synbeg rgfalllinesnew rpredict rh2ofast; do b $SY_FC $SY_FL -o bin/$p.exe "$psrc/$p.for"; done
# ifx rejects a Hollerith in rmolecasc (line-list reader only): ifx variants use gfortran -O3 for it
if [ "$SY_FC" = ifx ]; then b gfortran $GF_STD -O3 -o bin/rmolecasc.exe "$psrc/rmolecasc.patched.for"; else b $SY_FC $SY_FL -o bin/rmolecasc.exe "$psrc/rmolecasc.patched.for"; fi
b $SY_FC $SY_FL -o bin/synthe.exe "$psrc/synthe.patched.for"
rm -f *.o
cat > build.json <<J
{"variant": "$variant", "dfsynthe_kappa9": "$DF_FC $DF_FL", "atlas9": "$AT_FC $AT_FL", "synthe": "$SY_FC $SY_FL",
 "ifx": "$(ifx --version 2>/dev/null | head -1)", "gfortran": "$(gfortran --version | head -1)",
 "basicatlas_commit": "$(git -C "$BA" rev-parse HEAD)"}
J
ls bin | wc -l
