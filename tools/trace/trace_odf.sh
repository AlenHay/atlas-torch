#!/usr/bin/env bash
# Golden traces of xnfdf, dfsynthe (one temperature) and kappa9 (vturb 2) from the stock solar ODF run inputs.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"; EXT="${ATLAS_EXT:-$HOME/atlas-ext}"
R="$EXT/runs/stock/odf/solar"; D="$EXT/BasicATLAS/data/dfsynthe_files"; B="$EXT/builds/debug/bin"; T="$EXT/traces/odf_solar"
mkdir -p "$T"/{xw,dw,kw}
# xnfdf: stdin = the heredoc of xnfdf.com
cd "$T/xw"; rm -f fort.*; cp "$D/molecules.dat" fort.2; cp "$D/pfiron.dat" fort.4; cp "$D/continua.dat" fort.17
sed -n '/<<EOF/,/^EOF/p' "$R/xnfdf.com" | sed '1d;$d' > stdin.txt
"$repo/tools/trace/trace_program.sh" "$B/xnfdf.exe" "$T/xw" "$T/xnfdf" "$T/xw/stdin.txt" ALL 3
# dfsynthe at the 5012 K temperature (index 20 of the 57)
cd "$T/dw"; rm -f fort.*
ln -s "$R/xnfpdf.dat" fort.10; ln -s "$R/xnfpdfmax.dat" fort.22
for u in "11 lowlinesdf" "21 highlinesdf" "31 diatomicsdf" "41 tiolinesdf" "43 h2olinesdf" "51 nltelinesdf"; do set -- $u; ln -s "$D/$2.bin" fort.$1; done
python3 -c "print('0'*20+'1'+'0'*36)" > stdin.txt
"$repo/tools/trace/trace_program.sh" "$B/dfsynthe.exe" "$T/dw" "$T/dfsynthe" "$T/dw/stdin.txt" ALL 3
# kappa9 with vturb 2 on the stock solar ODF
cd "$T/kw"; rm -f fort.*; cp "$D/molecules.dat" fort.2; cp "$R/p00big2.bdf" fort.9
sed -n '/<<EOF/,/^EOF/p' "$R/kappa9v2.com" | sed '1d;$d' > stdin.txt
"$repo/tools/trace/trace_program.sh" "$B/kappa9.exe" "$T/kw" "$T/kappa9" "$T/kw/stdin.txt" ALL 3
echo done
