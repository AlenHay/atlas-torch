# atlas-torch

PyTorch port of the Kurucz ATLAS9 / DFSYNTHE / KAPPA9 / SYNTHE suite, as a drop-in backend for
[BasicATLAS](https://github.com/Roman-UCSD/BasicATLAS). Correctness (agreement with the Fortran within its
own noise floor) comes first; see `RESULTS.md` for the current scoreboard and `BRIEF.md` for the plan.

## Layout

| path | what |
|---|---|
| `container/` | Containerfile + `setup.sh` (Ubuntu 24.04, gfortran, Intel ifx, gdb, uv) — all work runs in the `atlas-dev` distrobox (`scripts/dx <cmd>`) |
| `scripts/build_fortran.sh` | Fortran builds: `stock` (= BasicATLAS `compile.com`) and noise-floor variants (gf-O0/O3, ifx-O0/O2/O3/precise, debug, fpe) |
| `scripts/run_fortran.py`, `scripts/reference_queue.sh` | reference + noise-variant runs of the frozen reference set (`reference/reference_set.json`) |
| `scripts/synthe_workdir.sh` | SYNTHE chain step by step, keeping every intermediate `fort.*` (stage-isolated inputs) |
| `scripts/run_examples.py` | executes BasicATLAS `examples/*.ipynb` against the Fortran or the torch backend |
| `tools/fmap.py` | Fortran map: call graphs, COMMON use, SAVE/DATA, variant diffs (`results/fmap/`) |
| `tools/trace/` | gdb-Python golden tracer (routine entry/exit: args, COMMONs, SAVEd locals → `.npz`) |
| `tools/oracle/` + `atlas_torch.testing.fortran_oracle` | ctypes oracles: any Fortran routine callable on arbitrary inputs; COMMON/SAVE layouts from DWARF |
| `tools/score/` | noise floor → frozen `results/tolerances.json`; `make score` → `results/scoreboard.jsonl`, `RESULTS.md` |
| `src/atlas_torch/` | the port |

Large data (line lists, Fortran sources, reference runs) live outside git in `$ATLAS_EXT` (default `~/atlas-ext`):
the pinned BasicATLAS checkout (`download.com` with its MD5s, `test.py` clean), builds, runs, traces.

**All runs and tests happen inside the `atlas-dev` container** (`scripts/dx`, `make ...`): the Fortran references
were produced there (glibc 2.39); the SteamOS host's glibc 2.43 has a different `log10f`, which changes results.

## Porting conventions

* **Fidelity first.** Port the *patched* Fortran semantics exactly: REAL*4 vs REAL*8 arithmetic, single-precision
  literals inside double expressions (`.77*3.288051D15` → `f32(.77)*3.288051e15`, `atlas_torch.fnum`),
  INT truncation, loop orders that change rounding, branch thresholds, iteration caps.
* **Explicit state.** COMMON blocks and SAVEd locals become explicit: `atlas_torch.state.State` maps
  `"block.var"` / `"save.<unit>.<var>"` to tensors with the Fortran shapes (same keys as golden traces).
  Routines are pure: they read the State and return the entries they write.
* **Variants are separate.** The physics library exists in three variants (SYNTHE `atlas7v`, ATLAS9/KAPPA9
  `atlas9mem`/`kappa9`, DFSYNTHE `xnfdf`). Share code only where `results/fmap/variants.md` says the copies are
  identical or cosmetic.
* **No transcribed tables, no copied sources.** DATA tables are read at run time from the fetched Fortran
  sources (`atlas_torch.kurucz_data`, validated bit-for-bit against the compiled libraries). BasicATLAS (no
  license) is never copied; its API is reimplemented.
* **Batched, float64, device-agnostic.** Layers (and later models) are tensor batch dimensions.
* **Every routine has an oracle test** `tests/oracle/test_<variant>_<routine>.py`: golden-trace entry → port →
  trace exit, plus hypothesis inputs → port vs the Fortran routine through the ctypes oracle.
