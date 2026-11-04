"""Continuous opacity in LTE (Kurucz 1970 ch. 5; the SYNTHE-variant routines of atlas7v.for as reference).

    from atlas_torch.v2.opacity.continuum import ContinuousOpacity
    op = ContinuousOpacity()(pops, nu)       # pops: v2 Populations (..., L); nu (F,) Hz
    op.absorption, op.scattering             # cm^2 g^-1, (..., L, F); op.absorbers["H- bf"], ...

Processes are listed in PROCESSES by name; KuruczCompat switches individual choices back to atlas7v.for's.
Departure coefficients are 1 (LTE), so every absorber has the Planck source function (Opacity.source = None).
"""
from __future__ import annotations

import torch

from ...constants import CODATA2018, PhysicalConstants
from ...eos.equilibrium import Populations
from .. import Opacity
from . import helium, hydrogen, metals, molecules, scattering
from .state import Conditions, KuruczCompat

# name -> (function of Conditions returning cm^-1, atlas7v.for routine)
ABSORBERS = {
    "H I bf": (hydrogen.h1_bound_free, "HOP"),
    "H I ff": (hydrogen.h1_free_free, "HOP"),
    "H- bf": (hydrogen.h_minus_bound_free, "HMINOP"),
    "H- ff": (hydrogen.h_minus_free_free, "HMINOP"),
    "H2+": (molecules.h2_plus, "H2PLOP"),
    "He I bf": (helium.he1_bound_free, "HE1OP"),
    "He I ff": (helium.he1_free_free, "HE1OP"),
    "He II bf": (helium.he2_bound_free, "HE2OP"),
    "He II ff": (helium.he2_free_free, "HE2OP"),
    "He- ff": (helium.he_minus_free_free, "HEMIOP"),
    "C I bf": (metals.c1_bound_free, "C1OP"),
    "Mg I bf": (metals.mg1_bound_free, "MG1OP"),
    "Al I bf": (metals.al1_bound_free, "AL1OP"),
    "Si I bf": (metals.si1_bound_free, "SI1OP"),
    "Fe I bf": (metals.fe1_bound_free, "FE1OP"),
    "CH": (molecules.ch, "CHOP"),
    "OH": (molecules.oh, "OHOP"),
    "H2-H2 CIA": (molecules.h2_h2_cia, "H2COLLOP"),
    "H2-He CIA": (molecules.h2_he_cia, "H2COLLOP"),
    "N I bf": (metals.n1_bound_free, "N1OP"),
    "O I bf": (metals.o1_bound_free, "O1OP"),
    "Mg II bf": (metals.mg2_bound_free, "MG2OP"),
    "Si II bf": (metals.si2_bound_free, "SI2OP"),
    "Ca II bf": (metals.ca2_bound_free, "CA2OP"),
    "C N O Ne ions bf": (metals.hot_bound_free, "HOTOP"),
    "metal ff": (metals.metal_free_free, "HOTOP"),
}
SCATTERERS = {
    "electrons": (scattering.thomson, "ELECOP"),
    "H I Rayleigh": (scattering.h1_rayleigh, "HRAYOP"),
    "He I Rayleigh": (scattering.he1_rayleigh, "HERAOP"),
    "H2 Rayleigh": (scattering.h2_rayleigh, "H2RAOP"),
}
PROCESSES = tuple(ABSORBERS) + tuple(SCATTERERS)


class ContinuousOpacity:
    """Continuous absorption and scattering, cm^2 g^-1, of a gas in LTE.

    processes  names from PROCESSES (default: all)
    const      constant set (CODATA2018; KURUCZ for the atlas7v.for literals)
    compat     KuruczCompat switches (default: v2 physics)"""

    def __init__(self, processes=None, const: PhysicalConstants = CODATA2018, compat: KuruczCompat = KuruczCompat()):
        names = PROCESSES if processes is None else tuple(processes)
        unknown = set(names) - set(PROCESSES)
        if unknown:
            raise ValueError(f"unknown processes {sorted(unknown)}")
        self.names, self.const, self.compat = names, const, compat

    def __call__(self, pops: Populations, nu) -> Opacity:
        nu = torch.as_tensor(nu, dtype=torch.float64, device=pops.T.device)
        c = Conditions(pops, nu, self.const, self.compat)
        shape = c.T.shape[:-1] + nu.shape
        per_mass = lambda f: (f(c) / c.rho).expand(shape)                # noqa: E731
        a = {k: per_mass(ABSORBERS[k][0]) for k in self.names if k in ABSORBERS}
        s = {k: per_mass(SCATTERERS[k][0]) for k in self.names if k in SCATTERERS}
        zero = torch.zeros(shape, dtype=nu.dtype, device=nu.device)
        return Opacity(sum(a.values(), zero), sum(s.values(), zero), None, a, s)


__all__ = ["ContinuousOpacity", "KuruczCompat", "PROCESSES", "ABSORBERS", "SCATTERERS"]
