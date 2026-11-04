"""Gradients of the port's outputs (M8): differentiable SYNTHE, implicit differentiation of NMOLEC and of the
converged ATLAS9 model, and their composition.

API
---
    from atlas_torch.diff import SyntheDiff, spectrum_and_jacobian, Atlas9FixedPoint, ChainDiff

    # 1. SYNTHE: d(spectrum)/d(abundances, vturb, [M/H], structure)
    spec, J = spectrum_and_jacobian("runs/my_model", (500., 510.), wrt=["abund:Fe", "abund:Mg", "vturb"])
    J["abund:Fe"]["norm"]          # d(F/Fcont)/d[Fe] per dex at every SYNTHE wavelength point (spec["wave"], nm)
    sd = SyntheDiff("runs/my_model", 500., 510.)                    # reusable (line lists prepared once)
    sd({"abund": {"Fe": 0.1}, "vturb": 1.2})                         # spectrum at other parameters (torch)
    sd.jvp({}, {"T": dT, "P": dP})                                   # directional derivative along a structure
    sd.structure_jacobian(quantities=("T", "P"))                     # per-layer (one JVP per layer, ~5 s each)

    # 2. ATLAS9: d(T, P, XNE, RHOX ... per layer)/d(Teff, log g, [M/H]) at convergence (implicit function thm.)
    fp = Atlas9FixedPoint("runs/my_model")
    # 3. both: d(spectrum)/d(Teff, log g, [M/H])
    ch = ChainDiff("runs/my_model", (500., 510.)).prepare()
    d = ch.spectrum_derivatives()  # {"teff"|"logg"|"mh": {"flux","cont","norm"}}

Parameters of the SYNTHE part ("params"): "abund" {element: dex} (BasicATLAS synthe(abun_adjust=...) semantics:
N(X)*10**dex, number fractions renormalised), "mh" (ABUNDANCE SCALE, dex), "vturb" (km/s), "T", "P", "XNE",
"RHOX" (n,) per-layer structure. Spectrum keys: "flux", "cont" (SPECTRV's HNU, as in spectrum.bin), "norm".

Design
------
* Differentiable mode = the exact path's formulas in float64 with native torch math (diff.native: the port's
  libm emulation switched to torch kernels via its existing EXACT flags, PFIRON in float64). The tabulated
  functions of SYNTHE (FASTEX, H0/H1/H2TAB, VOIGT) keep the exact path's table values with the derivative of
  the function they tabulate ("straight-through"). Discrete choices (SYNTHE's 1e-4*continuum cutoffs, Balmer
  wing walks, JOSH thick/interval choices) are taken once at the base point and held fixed. Deviation from the
  stock Fortran spectrum on 500-510 nm: <= 4.4e-7 relative flux (frozen gate 2.2e-6), see RESULTS.
* NMOLEC (sequential over layers, Newton) is differentiated implicitly at its converged iterate: one Newton step
  with the detached Jacobian at the converged point (primal unchanged, tangent = -J^-1 dF/dtheta), with the
  layer-to-layer start-value coupling as a forward substitution (diff.nmolec).
* Forward-mode AD (torch.autograd.forward_ad): one JVP per parameter direction. The spectrum has ~10^4 outputs
  and few parameters (abundances, vturb, Teff/logg/[M/H]), and forward mode runs through the port's existing
  torch opacity kernels as they are (reverse mode would meet NaN*0 products from torch.where over masked
  non-finite branches, which forward mode never forms).
* ATLAS9 (diff.atlas9_ift): x* = G(x*, theta) for one iteration G of the port (restart-deck variables),
  dx*/dtheta = (I - dG/dx)^-1 dG/dtheta solved Jacobian-free (Picard linear response with directional central
  differences of one iteration; the ATLAS9 port is the exact path and not autograd-capable, and a dense FD
  Jacobian fails because G is only piecewise smooth at the fixed point).
"""
from .atlas9_ift import (Atlas9FixedPoint, chord_newton, reconverge_iterating, tangents_jfree, tangents_picard,
                         theta_derivative)
from .chain import ChainDiff
from .native import native_math, straight_through
from .spectrum import SyntheDiff, spectrum_and_jacobian

__all__ = ["SyntheDiff", "spectrum_and_jacobian", "Atlas9FixedPoint", "chord_newton", "reconverge_iterating",
           "tangents_jfree", "tangents_picard", "theta_derivative", "ChainDiff", "native_math", "straight_through"]
