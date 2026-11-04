## GPU, float32 and torch.compile (M7 study, draft for RESULTS.md)

Study only: no default changed, no file under `src/` edited. Scripts: `tools/bench/gpu_*.py` (docstrings say how
to re-run each; `tools/bench/gpu_run.sh cpu|gpu <script>` picks the interpreter inside the container), raw results
`results/m7/gpu_float32_compile.json`, summary `results/m7/gpu_summary.json`, plots `results/plots/m7_accuracy.png`,
`m7_speed.png`, `m7_compile.png` (`tools/bench/gpu_report.py` regenerates them and the two tables below).
Measured 2026-11-04 on the shared machine (load average 3-9, other agents changing `src/` at the same time; the code
was commit f6a611f plus their uncommitted work), so **the timings are indicative only**. Re-run them on a quiet
machine; the accuracy results do not depend on load.

**Modes.** *exact* = the default path (glibc/libmvec emulation, C kernels, FTZ; CPU float64).
*native* = CPU float64 with native torch math, using the package's own switches (`atlas9.physics.libm.EXACT`,
`synthe.opacity._core.EXACT`, `libm32.native() -> None` (the documented non-x86 fallback), SPECTRV `libm="torch"`,
`atlas9.fmath`/`josh` glibc helpers patched to torch, DFSYNTHE without its C kernels). *gpu64* = the same on the GPU
(AMD RDNA3 8 GB, torch 2.9.1+rocm6.4, `HSA_OVERRIDE_GFX_VERSION=11.0.0`). *cpu32/gpu32* = every float64 (REAL*8)
torch op computed in float32 (`gpu_common.F32Emu`: a TorchFunctionMode that runs sums, matmuls and
transcendentals natively in float32 and rounds every other float64 result to float32; inputs are read as float32;
the REAL*4 code paths are unchanged). *cpu32p/gpu32p* (diagnostic) = float32 precision (24-bit significand) with
float64 exponent range, which separates rounding failures from underflow/overflow. Per program, each torch program
runs on the stock inputs of that program and the rest of the chain is the stock Fortran, so the stage metric
measures that program alone. All stages are scored with `atlas_torch.scoring` against the frozen
`results/tolerances.json`. Single CPU thread throughout.

### 1. Device-agnosticism

| family | runs on the GPU as is? | what was needed / found |
|---|---|---|
| SYNTHE `synthe_prog` | **yes** (`device=`), output identical | nothing. The CPU exact path uses a C accumulation kernel and the GPU uses the torch path |
| SPECTRV `spectrv_prog` | **yes** (`device=`, `libm="torch"`) | nothing |
| xnfpelsyn (KAPP batch) | **no** (`device=` crashes) | (a) the READIN/POPS State of `TorchPhysics` stays on the CPU, so KAPP mixes devices; (b) the DOPPLE block mixes GPU `tk` with CPU `atmass`; (c) `synthe/opacity/_core.py:save_get` creates absent SAVE arrays on the CPU (`device=None`) even for a GPU State. Bench-side helper `gpu_synthe.device_physics` runs the program on the CPU, moves the State and the frequency batch to the GPU for the KAPP call, and patches `save_get` to default to the State's device. Proposed one-line src fix for (c): `device = device or <device of any State tensor>` (identical on CPU) |
| KAPPA9 | **no** (`device=` crashes in HOP) | `atlas9/physics/opacity.py` builds its tables with `torch.from_numpy(...)` without a device, and `kappa9_prog.populations` writes CPU tensors into the State: 47 call sites (`gpu_common.DeviceCoerce` moves CPU operands when an op mixes devices and logs each call site) |
| ATLAS9 | **no** (no device argument; the State comes from READIN on the CPU) | runs under `torch.set_default_device('cuda')` + `DeviceCoerce`: 52-63 sites in `atlas9/physics/{molec,pfsaha,opacity,opacity_batch,api}.py`, `interp.py`, `atlas9/{putout,main,linop,readin,numerics}.py` (CPU tables / `from_numpy` / `.numpy()` round trips). Too many sites to fix as small helpers; left to the ATLAS9 owner. List: `results/m7/gpu_float32_compile.json["atlas/sun/gpu64"]["device_coercions"]` |
| DFSYNTHE | **yes**: `Config.lines_device='cuda'` (TAPE12 lines only) and `Config(device='cuda')` (everything) | only 2 sites, `.tolist()` of device tensors (`lines/accumulate.py:498`, `lines/xlinop.py:136`), which are legitimate host syncs. FTZ is not emulated off the CPU (documented) |

### 2. Accuracy against the frozen gates (worst error / gate; ✓ within all gates of the stage)

Plot: `results/plots/m7_accuracy.png`. "0" = byte-identical to stock. DFSYNTHE rows: one temperature compared with
that temperature's entries of the stock ODF, rated against the 57-temperature table gates (orientation only).

<!-- TABLE_ACCURACY -->
| stage / program | CPU exact | CPU f64 native | GPU f64 | CPU f32 | GPU f32 | CPU f32-prec | GPU f32-prec |
|---|---|---|---|---|---|---|---|
| SYNTHE w0 xnfpelsyn | 0 ✓ | 0 ✓ | 0 ✓ | 83 ✗ | 83 ✗ | 31 ✗ | 31 ✗ |
| SYNTHE w0 synthe | 0 ✓ | 0 ✓ | 0 ✓ | 0.095 ✓ | 0.095 ✓ | 0.095 ✓ | 0.095 ✓ |
| SYNTHE w0 spectrv | 0 ✓ | 0 ✓ | 0 ✓ | 3.2 ✗ | 2.9 ✗ | 3.2 ✗ | 3.2 ✗ |
| SYNTHE w0 chain | 0 ✓ | 0 ✓ | 0 ✓ | 85 ✗ | 85 ✗ | 34 ✗ | 34 ✗ |
| SYNTHE w1 xnfpelsyn | 0 ✓ | 0 ✓ | 0 ✓ | 60 ✗ | 60 ✗ | 29 ✗ | 29 ✗ |
| SYNTHE w1 synthe | 0 ✓ | 0 ✓ | 0.0015 ✓ | 4.8 ✗ | 4.8 ✗ | 4.8 ✗ | 4.8 ✗ |
| SYNTHE w1 spectrv | 0 ✓ | 1.4e-11 ✓ | 1.4e-11 ✓ | 4.4 ✗ | 4.2 ✗ | 4.2 ✗ | 4.2 ✗ |
| SYNTHE w1 chain | 0 ✓ | 1.4e-11 ✓ | 0.0015 ✓ | 63 ✗ | 63 ✗ | 33 ✗ | 33 ✗ |
| SYNTHE w2 xnfpelsyn | 0 ✓ | 0 ✓ | 0 ✓ | 42 ✗ | 42 ✗ | 6 ✗ | 6 ✗ |
| SYNTHE w2 synthe | 0 ✓ | 0 ✓ | 0.00018 ✓ | 0.82 ✓ | 0.82 ✓ | 0.82 ✓ | 0.82 ✓ |
| SYNTHE w2 spectrv | 0 ✓ | 0 ✓ | 0 ✓ | 44 ✗ | 44 ✗ | 44 ✗ | 44 ✗ |
| SYNTHE w2 chain | 0 ✓ | 0 ✓ | 0.00018 ✓ | 86 ✗ | 86 ✗ | 49 ✗ | 49 ✗ |
| SYNTHE w3 xnfpelsyn | 0 ✓ | 0 ✓ | 0 ✓ | 5.5e+02 ✗ | 5.5e+02 ✗ | 6.4 ✗ | 6.4 ✗ |
| SYNTHE w3 synthe | 0 ✓ | 0 ✓ | 0.068 ✓ | 8.3 ✗ | 8.3 ✗ | 8.3 ✗ | 8.3 ✗ |
| SYNTHE w3 spectrv | 0 ✓ | 0 ✓ | 0 ✓ | 1.5 ✗ | 1.6 ✗ | 1.5 ✗ | 1.5 ✗ |
| SYNTHE w3 chain | 0 ✓ | 0 ✓ | 0.068 ✓ | 5.5e+02 ✗ | 5.5e+02 ✗ | 13 ✗ | 12 ✗ |
| KAPPA9 solar | 0 ✓ | 0 ✓ | 0 ✓ | overflow ✗ | overflow ✗ | 1 ✗ | 1 ✗ |
| KAPPA9 alpha | 0 ✓ | 0 ✓ | 0 ✓ | — | — | — | — |
| ATLAS9 sun | 0 ✓ | ∞ ✗ | ∞ ✗ | ∞ ✗ | ∞ ✗ | ∞ ✗ | — |
| ATLAS9 t4500g25 | 0 ✓ | 0.33 ✓ | 0.18 ✓ | — | — | — | — |
| ATLAS9 t6500g40m10 | 0 ✓ | 0.017 ✓ | — | — | — | — | — |
| ATLAS9 t9000g40 | 0 ✓ | 0.33 ✓ | — | — | — | — | — |
| ATLAS9 t5000g45m20 | 0 ✓ | 0.0033 ✓ | — | — | — | — | — |
| ATLAS9 t5500g40alpha | 0 ✓ | 0.78 ✓ | — | — | — | — | — |
| DFSYNTHE 5012 K | 0.0029 ✓ | — | — | — | — | — | — |
| DFSYNTHE 12589 K | 0.063 ✓ | 0.063 ✓ | 0.0029 ✓ | 1.1e+02 ✗ | crash | — | — |
<!-- /TABLE_ACCURACY -->

Plain answers:
* **CPU float64 native math:** all four SYNTHE windows (each program and the whole chain), KAPPA9, and DFSYNTHE stay
  within the gates, mostly byte-identical: SYNTHE writes REAL*4 spectra, so ulp-level differences vanish.
  ATLAS9 stays within the gates on 5 of 6 models (worst 0.78 of a gate). It **fails sun**: FLXDRV differs by 0.01,
  FLXERR by 1e-5 and P by 1.1e-6 rel, one printed digit each, but sun's FLXDRV/FLXERR/HEIGHT gates are zero
  because every Fortran variant printed the same digits there.
* **GPU float64:** same verdict as CPU native. SYNTHE: byte-identical or ≤0.07 of a gate (w3 synthe); the GPU
  line-opacity sums are float32 atomics, so the summation order differs. KAPPA9: byte-identical. DFSYNTHE
  12589 K: ≤1 step. ATLAS9: t4500g25 within the gates (0.18), sun fails by one printed digit (FLXDRV 0.01,
  FLXERR 1e-5, HEIGHT 0.001 km), the same as CPU native.
* **float32 for the REAL*8 parts (CPU or GPU, same result): not viable.** Only the SYNTHE line-opacity program,
  which is REAL*4 in the Fortran anyway, survives in w0 and w2 (0.095, 0.82 of a gate). It fails w1 (4.8×) and w3
  (8.3×): its REAL*8 wavelength grid matters at 380-420 nm and 1.5-1.6 µm. Failures by stage:
  * xnfpelsyn: 42-550× the gate. Trace populations fall below FLT_MIN (3929 of 30215 XNFPEL entries flushed to 0),
    and contscat is off by up to 4.5 dex.
  * spectrv: 1.5-44×.
  * KAPPA9: the Rosseland table overflows its print field (`*****`).
  * ATLAS9 sun: T 4.5× the gate, P/ρ/XNE ~1e-3 vs 1e-5, flux error 0.85 %.
  * DFSYNTHE 12589 K: up to 40 dex, with 14 % of entries off by more than 1 step.
  * XNFDF populations in float32 make DFSYNTHE's C kernel segfault, and the GPU float32 run stops with
    `bincount minlength < 0`.

  The precision-only diagnostic still fails almost everywhere (xnfpelsyn 6-31×, spectrv 1.5-44×, ATLAS9 sun,
  KAPPA9 at its 1-quantum gate). So it is the 24-bit significand, not just the exponent range: rescaling or log
  storage alone would not rescue float32.

### 3. Speed (seconds; one CPU thread for CPU work)

Plot: `results/plots/m7_speed.png`. Fortran = the stock executable on the same inputs.

<!-- TABLE_SPEED -->
| stage / program | Fortran | CPU exact | CPU f64 native | GPU f64 | GPU f32 | GPU lines | CPU exact, no FTZ |
|---|---|---|---|---|---|---|---|
| SYNTHE w0 xnfpelsyn | 0.11 | 0.65 | 0.49 | 0.56 | 1.78 | — | — |
| SYNTHE w0 synthe | 5.65 | 1.12 | 1.14 | 0.21 | 0.29 | — | — |
| SYNTHE w0 spectrv | 0.14 | 0.11 | 0.51 | 0.12 | 0.21 | — | — |
| SYNTHE w0 chain | — | 1.85 | 2.27 | 0.88 | 2.22 | — | — |
| SYNTHE w1 xnfpelsyn | 0.11 | 0.61 | 0.74 | 0.70 | 3.08 | — | — |
| SYNTHE w1 synthe | 9.87 | 4.58 | 12.80 | 1.84 | 3.34 | — | — |
| SYNTHE w1 spectrv | 0.79 | 0.57 | 3.22 | 0.58 | 1.26 | — | — |
| SYNTHE w1 chain | — | 5.78 | 17.51 | 2.97 | 7.08 | — | — |
| SYNTHE w2 xnfpelsyn | 0.12 | 0.61 | 0.54 | 0.55 | 3.39 | — | — |
| SYNTHE w2 synthe | 10.31 | 1.40 | 1.82 | 0.20 | 0.40 | — | — |
| SYNTHE w2 spectrv | 0.20 | 0.15 | 0.88 | 0.16 | 0.44 | — | — |
| SYNTHE w2 chain | — | 2.17 | 3.09 | 0.92 | 3.40 | — | — |
| SYNTHE w3 xnfpelsyn | 0.12 | 0.61 | 0.48 | 0.55 | 1.80 | — | — |
| SYNTHE w3 synthe | 17.89 | 2.58 | 2.58 | 3.54 | 6.51 | — | — |
| SYNTHE w3 spectrv | 0.41 | 0.30 | 1.67 | 0.30 | 0.48 | — | — |
| SYNTHE w3 chain | — | 3.51 | 4.77 | 4.55 | 8.56 | — | — |
| KAPPA9 solar | 2.06 | 5.06 | 13.64 | 9.94 | — | — | — |
| KAPPA9 alpha | 7.31 | 22.63 | 19.37 | 10.22 | — | — | — |
| ATLAS9 sun | 5.55 | 9.57 | 9.18 | 31.83 | 218.65 | — | — |
| ATLAS9 t4500g25 | 6.72 | 29.62 | 20.37 | 140.95 | — | — | — |
| ATLAS9 t6500g40m10 | 6.85 | 29.68 | 42.08 | — | — | — | — |
| ATLAS9 t9000g40 | 18.00 | 92.03 | 63.85 | — | — | — | — |
| ATLAS9 t5000g45m20 | 4.22 | 29.48 | 28.56 | — | — | — | — |
| ATLAS9 t5500g40alpha | 3.84 | 38.08 | 35.75 | — | — | — | — |
| DFSYNTHE 5012 K | 41.57 | 13.10 | — | — | — | 27.89 | 32.24 |
| DFSYNTHE 12589 K | 26.75 | 14.13 | 143.32 | 341.23 | — | 19.97 | 26.98 |
<!-- /TABLE_SPEED -->

Rows were last timed at different loads (each record carries its load average). Re-timed on a quiet machine
(load 2-3, 12:06-12:20): SYNTHE w0-w3 exact/gpu64, KAPPA9 solar, ATLAS9 sun, DFSYNTHE both temperatures. The
other rows ran at load 4-9 and are 1.5-8× slower than quiet (e.g. DFSYNTHE 12589 K exact: 117 s loaded, 14 s
quiet). Quiet-machine conclusions:
* **SYNTHE line opacity (`synthe_prog`) is the one GPU win:**

  | window | GPU | CPU exact | Fortran | GPU speed-up vs CPU exact |
  |---|---|---|---|---|
  | w0 | 0.21 s | 1.12 s | 5.6 s | 5.3× |
  | w1 | 1.84 s | 4.58 s | 9.9 s | 2.5× |
  | w2 | 0.20 s | 1.40 s | 10.3 s | 7× |
  | w3 (1.5-1.6 µm) | 3.54 s | 2.58 s | 17.9 s | 0.73× (GPU slower) |

  The whole chain on the GPU is 1.9-2.4× faster on w0-w2 and slower on w3. The CPU exact path already beats the
  Fortran 2-7×: its C kernels are the SYNTHE agent's current work.
* **Not worth it on the GPU** (launch-bound: thousands of small ops on 72-99-layer tensors):
  * xnfpelsyn KAPP: 0.55 s vs 0.6 s;
  * SPECTRV: about even with exact;
  * KAPPA9: 9.9 s vs 5.1 s;
  * ATLAS9: 2.0 s vs 0.6 s per iteration, sun 32 s vs 9.6 s (Fortran 5.6 s);
  * DFSYNTHE `lines_device='cuda'`: 20 s vs 14 s at 12589 K and 28 s vs 13 s at 5012 K (Fortran 27 s / 42 s),
    because the C line kernel on one CPU core is faster than the torch profile path on the GPU;
  * DFSYNTHE all-GPU: 341 s (per-line Python loop in XLINOP).
* Native math on the CPU is not faster than exact anywhere it matters. The exact helpers are vectorised C/glibc
  now, and SPECTRV's native `libm="torch"` path is 3-5× slower than its libmvec-emulating exact path.
* float32 does not change the GPU picture: these programs are launch-bound, not FLOP-bound. On this GPU float64
  runs at 1/27 of float32 throughput, so the float64 cost would only show in a FLOP-bound kernel.

### 4. torch.compile on the hot kernels (captured real inputs, native math)

Plot: `results/plots/m7_compile.png`. `tools/bench/gpu_compile.py capture|bench`.

| kernel | CPU inductor (torch 2.15) | GPU inductor/triton (torch 2.9 ROCm) |
|---|---|---|
| SPECTRV `josh_batch` (8192 points) | **2.19×**, bit-identical (5 graph breaks; first call 23 s) | 1.03×, max rel 3e-7 (FMA contraction), first call 59 s |
| SYNTHE `voigt` (largest batch) | 1.39×, bit-identical | **2.6×**, max rel 3e-7 |
| ATLAS9 `josh` (one iteration) | **0.21× (slower)**, not identical (max abs 7e-6; 31 graph breaks: `nonzero`, `.item()`, `open()` of the tables inside the call) | dynamo AssertionError in `numerics.map1b` (`np.broadcast_shapes`; torch 2.9) |
| DFSYNTHE line chunk (torch path) | 0.99×, bit-identical (57 graph breaks, recompile limit hit in `_values`) | not measured: earlyoom killed the compilation (host RAM) |

Compiled code stays bit-identical on the CPU where it helps (SPECTRV JOSH, Voigt). On the GPU, triton contracts
multiply-adds into FMAs (~1 ulp in float32). The data-dependent control flow (`nonzero`, `.item()`, per-line loops)
breaks the graphs and leaves the line-accumulation kernels without a gain. These kernels are better served by
the existing C kernels on the CPU and by eager torch on the GPU.

### 5. Side finding for the DFSYNTHE owner (not a study change)

At 12589 K (IT=33) the default CPU path with FTZ differs from stock by up to 14-15 steps (5e-4 of the entries
> 1 step; within the table gates). The same run with `Config.ftz=False`, and the all-GPU run (no FTZ), match
stock to ≤1 step. At 5012 K it is the other way round: FTZ gives ≤1 step and no FTZ gives up to 18 steps, as
RESULTS.md documents. So at hot temperatures the port's FTZ scope flushes something the stock binary does not,
and should be checked (`tools/bench/gpu_dfsynthe.py --temps 33 --mode exact|exact_noftz`). An apparent 93 s
XLINOP under FTZ seen at load ~7 does not reproduce on a quiet machine (14 s total): that was contention.

### Recommendations

1. Keep float64. float32 for REAL*8 quantities fails every stage that has them, and precision is the cause, not
   only range.
2. Expose one GPU option, SYNTHE `synthe_prog(device='cuda')`:
   * 2.5-7× faster than the current CPU exact path on the 10-40 nm windows;
   * within the gates on all four windows (≤0.07 of a gate);
   * not byte-identical (float32 atomics), so not a default;
   * worth more for the wide BasicATLAS batches (up to 2 M points).

   Measure it on a 100-3000 nm example before deciding. DFSYNTHE `lines_device` is slower than the C kernel and
   brings nothing.
3. Making xnfpelsyn/KAPPA9/ATLAS9 device-clean means:
   * the `save_get` default-device fix;
   * a device argument threaded through READIN/POPS and the `atlas9.physics` tables (≈50-60 sites listed).

   It brings no speed on its own (launch-bound). It matters for the batched-model / differentiable work: a
   leading model axis is what would make the GPU useful there.
4. Use torch.compile only for SPECTRV's JOSH on the CPU (2.2×, bit-identical) if SPECTRV ever becomes a
   bottleneck. Elsewhere it gives no gain or breaks identity.
