## Gradients (M8, draft for RESULTS.md)

Package `src/atlas_torch/diff/` (new; no file outside it changed). All numbers: solar reference model
(`runs/stock/atlas/sun`), SYNTHE window 500-510 nm (R=600000, vturb 1.5 km/s, 11882 points), run in the
atlas-dev container against a frozen snapshot of commit `f6a611f` + the diff package (the ATLAS9/SYNTHE perf
agents were editing those modules concurrently). Load averages 5-10 during the runs (shared machine): timings
are indicative, re-run `tools/bench/diff_gradients.py` on a quiet machine.

### API

```python
from atlas_torch.diff import SyntheDiff, spectrum_and_jacobian, ChainDiff
spec, J = spectrum_and_jacobian(run_dir, (500., 510.), wrt=["abund:Fe", "abund:Mg", "vturb", "mh", "T[40]"])
J["abund:Fe"]["norm"]                      # d(F/Fcont)/d[Fe] per dex at every point of spec["wave"]
sd = SyntheDiff(run_dir, 500., 510.)       # line lists prepared once; sd(params) -> flux, cont, norm
sd.jvp({}, {"T": dT, "P": dP})             # directional derivative along any structure perturbation
sd.structure_jacobian(quantities=("T", "P"))   # per layer (one JVP per layer and quantity)
ch = ChainDiff(run_dir, (500., 510.)).prepare()
ch.structure_tangents()["teff"]["T"]       # dT(layer)/dTeff of the converged ATLAS9 model
ch.spectrum_derivatives()                  # {"teff"|"logg"|"mh": {"flux","cont","norm"}}
```
SYNTHE parameters: `abund` {element: dex} (BasicATLAS `synthe(abun_adjust=...)` semantics: N(X)*10**dex, all
number fractions renormalised), `mh` (ABUNDANCE SCALE), `vturb` [km/s], per-layer `T`, `P`, `XNE`, `RHOX`.
Example: `examples/gradients_demo.py`; validation driver: `python -m atlas_torch.diff.validate`.

### Design (what is differentiated, and how)

* **Differentiable mode = the exact path's formulas in float64 with native torch math.** The port's libm
  emulation is switched to torch kernels through its existing `EXACT` flags inside `diff.native.native_math()`
  (context manager, restores everything on exit); PFIRON's REAL*4 interpolation runs in float64 (table cells
  chosen as in REAL*4). XNFPELSYN's physics (PFSAHA, KAPP and all continuum opacity kernels) is the port's own
  torch code run as is; POPS is re-implemented in `diff.xnf` only so that XABUND stays a tensor. SYNTHE's line
  loop (`diff.lines`), XLINOP's hydrogen lines with HPROF4 (`diff.hydrogen`) and SPECTRV/JOSH (`diff.rt`) are
  float64 transcriptions.
* **Tabulated functions** (FASTEX exp table, H0/H1/H2TAB, VOIGT): the value is the exact path's table entry
  (same REAL*4 index arithmetic), the derivative that of the function the table samples ("straight-through",
  `profile="exact"`, default). `profile="smooth"` evaluates the smooth functions themselves.
* **Discrete decisions are taken once at the base point and held fixed**: SYNTHE's two cutoff tests per
  (line, layer) and the profile extents (core exit, far-wing MAXSTEP), the Balmer-wing walks, JOSH's
  thick/interval choices. The derivative is that of the smooth piece containing the base point (it ignores
  contributions switching on/off at the 1e-4*continuum cutoff). `sd(params, redecide=True)` re-takes them.
* **JOSH**: the scattering problem on the 51-point grid is solved exactly (float64 LU) instead of REAL*4
  Gauss-Seidel sweeps stopped at |dS/S| < 1e-5.
* **NMOLEC: implicit differentiation at convergence** (`diff.nmolec`). The Fortran's Newton iteration is
  replayed exactly (bit-identical iterate, checked); then, per layer, X = X* + (D - D.detach()) with
  D = -J*^-1 EQ(X*; theta), J* the analytic Jacobian at X*: the primal is the exact iterate, the tangent is
  -J^-1 dEQ/dtheta (IFT), never through the iterations. Two Fortran details couple layers: the ion
  equilibrium constants use the *start* XNE(J) = XNE*(J-1)*P(J)/P(J-1) (Debye lowering), so the tangent solve is a
  forward substitution over layers; CPF uses the deck XNE. `polish=True` converges the root to machine
  precision first (smooth forward function, used for the FD checks).
* **Forward-mode AD** (`torch.autograd.forward_ad`, one JVP per parameter direction): ~1e4 outputs, few
  parameters; and it runs through the port's torch opacity kernels unchanged. Reverse mode was tried first and
  gives NaN gradients: `torch.where` over masked non-finite branches (e.g. H2COLLOP's 10**x beyond its
  table) makes 0*inf products in backward; forward mode never forms them.

### Deviation of the differentiable mode from the exact path (500-510 nm vs stock Fortran spectrum.bin)

| profile | flux rel max | flux rel median | cont rel max | norm abs max | frozen gate (flux / cont / norm) |
|---|---|---|---|---|---|
| exact (default) | 4.4e-7 | 6.6e-8 | 3.4e-7 | 4.9e-7 | 2.2e-6 / 9.8e-7 / 1.6e-6 |
| smooth | 1.0e-3 | 4.4e-6 | 3.4e-7 | 4.5e-4 | (not a gate candidate: table resolution) |

Components: XNE/RHO/H populations bit-identical; XNFPEL 1.1e-6 rel (float64 PFIRON vs REAL*4; 6e-8 with the
REAL*4 PFIRON); log10 continuum 8e-8 dex; line buffer vs exact SYNTHE 9e-8 median / 1.7e-6 max (float32
accumulation of the exact path); Balmer wings same walk extents, 2e-7 median.

### Validation: SYNTHE (forward-mode AD vs central finite differences)

`results/m8/validate_synthe_w0.json`. Metrics over the 11882 points of d(F/Fc)/dp: max = max|AD-FD|/max|FD|,
p99 = 99th percentile of |AD-FD|/max|FD|, rms = ||AD-FD||/||FD||. "refined": the worst points re-checked with
a 10x smaller step.

(a) same function, decisions frozen, `profile="smooth"`, polished NMOLEC (tests the derivatives themselves):

| parameter | step | max | p99 | rms | refined max | max abs derivative |
|---|---|---|---|---|---|---|
| [Fe] | 1e-5 dex | 1.8e-4 | 2.6e-10 | 8.8e-6 | 1.0e-4 | 0.67 /dex |
| [Mg] | 1e-5 dex | 1.2e-3 | 1.4e-9 | 5.5e-5 | 8.0e-4 | 0.12 /dex |
| [C] | 1e-5 dex | 1.3e-5 | 2.9e-10 | 8.5e-7 | - | 0.62 /dex |
| [M/H] (synthesis) | 1e-5 dex | 5.1e-4 | 2.9e-10 | 3.0e-5 | 4.5e-4 | 0.87 /dex |
| vturb | 1e-5 km/s | 2.5e-5 | 7.3e-7 | 4.1e-6 | - | 0.65 /(km/s) |
| T layer 31 | 4.5e-4 K | 7.0e-6 | 2.6e-8 | 5.6e-7 | - | 2.8e-5 /K |
| T layer 46 | 5.0e-4 K | 4.6e-8 | 1.0e-8 | 2.1e-8 | - | 7.3e-5 /K |
| P layer 41 | 1e-7 rel | 7.3e-7 | 4.3e-7 | 8.8e-7 | - | 5.4e-7 /(dyn cm-2) |
| RHOX layer 41 | 1e-7 rel | 7.1e-7 | 4.6e-7 | 1.2e-6 | - | 1.4e-2 /(g cm-2) |
| XNE layer 41 | 1e-7 rel | 0 | 0 | 0 | - | 0 (exactly: with MOLECULES ON NMOLEC recomputes XNE; the deck XNE only enters CPF, which is 1 in LTE) |

The remaining max-norm outliers sit at points next to a kink of Kurucz's blended-parabola interpolation
(PARCOE/MAP1 weights |c|/(|c|+|c'|) are continuous, not differentiable, where a curvature changes sign); at
each such point the FD converges to the AD value as the step shrinks (e.g. [Mg], point 4689: FD(1e-4) differs
by 2.3e-4, FD(1e-6) agrees to 3e-9). Tangents through every stage up to the line opacity agree to 1e-8..1e-10.

(b) `profile="exact"` (table values, straight-through derivatives), decisions frozen: same table for the
abundances ([Fe] rms 1.4e-5, [Mg] 8.4e-5, [C] 1.1e-5, [M/H] 4.3e-5; p99 <= 2e-7).

(c) AD (`profile="exact"`) vs central FD of the **exact path** (the byte-identical port: REAL*4 tables,
decisions re-taken, abundances through the deck's printed 2-decimal cards, TURBV through fort.93):

| parameter | step | max | p99 | rms |
|---|---|---|---|---|
| [Fe] | 0.05 dex | 4.3e-3 | 1.9e-3 | 3.0e-3 |
| [Mg] | 0.05 dex | 1.5e-2 | 6.1e-3 | 1.0e-2 |
| [M/H] | 0.02 dex | 6.5e-3 | 3.0e-3 | 4.5e-3 |
| vturb | 0.05 km/s | 1.5e-2 | 4.9e-3 | 1.2e-2 |

The exact path is a staircase in these parameters (printed cards, table entries, cutoffs), so its FD needs
steps of this size; the remaining differences are of the size of the O(h^2) truncation of those steps
((0.05 ln10)^2/6 = 2e-3 relative for an exponential dependence).

Stage tests `tests/stage/test_diff_synthe.py` (8 tests, 2 min) pass against both the snapshot and the current
working tree. Toy checks (fast tier, `tests/stage/test_diff_toy.py`, 4 tests, 7 s; fast tier 435 passed): NMOLEC IFT on 6 layers (primal bit-identical to
the exact iterate; tangent vs FD of the polished root < 1e-6; unpolished vs polished tangent < 1e-4), JOSH
tangent vs FD < 1e-6, VOIGT straight-through values = REAL*4 table values (< 2e-6) and derivatives = smooth
derivatives, IFT algebra on a linear map.

### ATLAS9: implicit differentiation of the converged model (`diff.atlas9_ift`)

Fixed-point map: x = the restart deck (RHOX, T, P, XNE, ABROSS, ACCRAD per layer, PRADK0; 433 numbers, in full
float64 precision: the punched deck has 4-9 digits, so x replaces READ PUNCH's columns in memory), G = one
iteration of the port's ATLAS9 from x with theta = (Teff, log g, [M/H] = log10 ABUNDANCE SCALE); ODF, VTURB
and the abundance pattern fixed ([M/H] is therefore the partial derivative at fixed line opacity). One
iteration: 5-14 s depending on load (pre-M7 code). Probe (all layers of one quantity x(1+1e-3)): G depends on
input RHOX (2.8e-3) and T (6.1e-3) only; P, XNE, ABROSS have exactly zero influence (recomputed), ACCRAD 1e-7,
PRADK0 3e-9. dG/dtheta by central differences (Teff +/-2 K, log g and [M/H] +/-0.002).

Solvers for dx*/dtheta = (I - dG/dx)^-1 dG/dtheta, tried in this order:

1. **Dense FD Jacobian** of one iteration (the brief's suggestion; 144 RHOX/T columns, central differences,
   ~300 iterations, 12-20 min with 2 workers), steps 1e-5 and 2e-4 relative: **unusable**. The two Jacobians
   give different, unphysical tangents (dT/dTeff = -1.1 resp. -0.5 at the top layer; physical: +0.6), and a
   chord-Newton iteration with either Jacobian diverges. Diagnosis: (i) G is smooth to 1e-9 along single
   columns at the 1e-5 scale, but TCORR maps the corrected model back onto the standard tau grid with MAP1, and
   at the fixed point the model's own tau points coincide with that grid, so G is only piecewise smooth there
   and per-column derivatives do not combine into the response to a collective perturbation (directional FD
   along the IFT direction differs from J.v by 10-12% at 0.1-1 K amplitude; dG/dTeff from +/-0.1 K steps is
   0.16 at the top layer, from +/-1 K and +/-5 K 0.028); (ii) I - dG/dx is ill-conditioned (largest
   eigenvalues of dG/dx 0.9973-0.998, cond ~1e6) with the slow modes localised at the convective boundary
   (layers 56-64), where even the restoring slope scatters by +/-50% with the step (fixed point undetermined
   along that mode at ~7e-4 relative, ~5 K).
2. **Jacobian-free GMRES** (operator v -> v - J v with directional central differences, ~1 K amplitude; 8-13
   products, 10 min for three parameters): accurate for the dominant P/RHOX responses to log g and [M/H]
   (1-2%), not for the small T responses (the operator is only piecewise linear; true residual 1-9%).
3. **Picard linear response** (default): t <- J t + dG/dtheta, J t by directional central differences with a
   fixed amplitude, 40 steps = the linearised ATLAS9 iteration itself (2 x 40 iterations per parameter,
   15-19 min each at load ~6-10). Robust to the kinks; converges at ATLAS9's rate (40 steps leave the slow
   convective mode partly unconverged, as for ATLAS9 itself).

Validation against central differences of **fully re-converged models**: 40 iterations of G at theta +/- h
from x* (the gold-tolerance stop removed; last max|dx/x| per iteration 4-5e-7, one run 1.7e-5), h = 5 K,
0.005 dex, 0.005 dex. Errors rms over the 72 layers relative to the rms of the FD derivative (max in brackets):

| theta | T | ln P | ln RHOX | ln XNE | dense FD Jacobian (T) | GMRES (T) |
|---|---|---|---|---|---|---|
| Teff | 2.0e-2 (4.6e-2) | 4.8e-3 (5.8e-3) | 7.7e-3 (1.2e-2) | 3.0e-2 (5.7e-2) | 3.2 | 0.27 |
| log g | 1.9e-1 (3.1e-1) | 4.7e-4 (2.3e-3) | 2.3e-3 (9.1e-3) | 1.4e-2 (5.7e-2) | 61 | 4.9 |
| [M/H] | 4.6e-2 (7.4e-2) | 2.1e-3 (3.6e-3) | 1.9e-3 (3.7e-3) | 1.0e-2 (4.9e-2) | - | - |

Values, e.g. dT/dTeff at layers 1/21/37/51/57/61/72: IFT 0.59/0.74/0.79/0.88/1.08/1.20/1.00, FD
0.62/0.75/0.79/0.88/1.11/1.20/0.97; dT/dlog g [K/dex] IFT 38/3.9/-7.2/-0.8/72/-123/-8.2, FD
40/4.6/-7.1/1.0/35/-142/-7.7 (the log g T error is concentrated in the convective-boundary layers, where the
response is weak and ATLAS9's own convergence is slow); dlnP/dlog g = 1.3-2.0 per dex (P = g RHOX at fixed
tau with kappa ~ P^0.6).

### Chain: d(spectrum)/d(Teff, log g, [M/H])

`ChainDiff`: dS/dtheta = (dS/dx) . (dx*/dtheta) (+ dS/d[M/H] at fixed structure, since the SYNTHE deck carries the
same ABUNDANCE SCALE), the structure tangent of all layers as one forward-mode JVP per parameter, evaluated at the
full-precision fixed point. Validation (`tests/stage/test_diff_atlas9.py`, `profile="smooth"`, polished NMOLEC):
FD = central difference of the differentiable spectrum of the re-converged models above. Metrics over the
11882 points of d(F/Fc)/dtheta:

| theta | chained (IFT tangent) rms (max) | composition only: JVP along the FD structure tangent, rms (max) |
|---|---|---|
| Teff | 1.9e-2 (1.4e-2) | 1.3e-3 (3.4e-3) |
| log g | 5.5e-2 (3.1e-2) | 9.3e-3 (1.3e-2) |
| [M/H] | 7.6e-3 (7.0e-3) | 1.3e-3 (3.8e-3) |

(max is normalised by max|FD|, rms by rms|FD|.) The composition is exact to the linearity of the spectrum over
the FD steps; the end-to-end error is that of the ATLAS9 structure tangent (mostly T at the convective
boundary). With `profile="exact"` the chained derivative is the straight-through one; an FD check of the
frozen-decision exact-profile function is not meaningful for T changes (it keeps the Voigt-table positions,
set by the Doppler width, fixed: 5% apparent disagreement), hence the smooth profile in this check.

### Runtimes (`tools/bench/diff_gradients.py`, `results/m8/bench_diff_gradients.json`: load 4.5, 4 torch threads; ATLAS9 rows at load 5-10)

| step | time |
|---|---|
| SyntheDiff setup, 500-510 nm (cutoff decisions for 836k line-layer pairs, Balmer walks, base RT decisions) | 6.6 s |
| line lists of a new window (exact-path line programs, 1 nm) | included in 5.6 s setup |
| forward pass, 500-510 nm (xnfpelsyn 1.0, lines 1.2, H lines 0.3, spectrv 0.6 s) | 3.1 s (exact path: synthe 0.7 s) |
| one JVP: [Fe] / vturb / T(layer 41) | 6.9 / 4.2 / 8.0 s |
| peak RSS (setup + 3 JVPs) | 2.0 GB |
| per-layer structure Jacobian (T and P, 72 layers) | ~144 JVPs, ~30 min |
| ATLAS9 iteration (pre-M7 code) | 5-14 s |
| ATLAS9 IFT (Picard, 3 parameters) | 3 x 80 iterations, 45-60 min |
| re-converged FD reference (6 models x 40 iterations) | ~45 min |

### Limitations

* Decisions are frozen: the derivative is that of the piece of the (piecewise smooth) SYNTHE function at the
  base point; cutoff switchings (1e-4 of the continuum) and MAP1/PARCOE kinks are not represented.
* fort.19 line types other than hydrogen (He I, autoionising, merged continua, TYPE 0/3 Voigt lines) and fort.12
  lines with ALPHA != 0 raise NotImplementedError (none in the reference windows' blue part checked here;
  the Ca H&K/H-alpha windows were not run). SURFACE FLUX only (IFSURF=1); velocity fields with fort.19 lines
  unsupported; air wavelengths with H series merging unsupported.
* XNE of the SYNTHE deck has zero influence (by construction of the Fortran), so its derivative is 0.
* The ATLAS9 part uses finite differences of the exact-path iteration (not autograd); its accuracy is limited
  by ATLAS9's own piecewise structure and slow convective-boundary convergence (table above). [M/H] is at
  fixed ODF. Cost is comparable to re-converging perturbed models (the IFT's advantage over FD is that one
  linear response serves every downstream quantity, e.g. the spectrum, without re-running SYNTHE per model).
* Reverse mode (gradients of a scalar loss in one pass) is not supported through KAPP (NaN from masked
  branches); use forward mode (few parameters) or structure_jacobian.
