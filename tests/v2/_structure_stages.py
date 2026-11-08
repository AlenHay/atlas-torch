"""Stage comparisons of the v2 structure modules with the exact ATLAS9 port, on the port's own iteration States:
the frequency integrals (RadiationMoments; continuum opacities either v2's or the port's own as a stand-in),
the convection (ThermoDerivatives + MixingLength) and the temperature correction (TemperatureCorrection)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

import _atlas9_oracle as O

HOOKS = ("frequencies", "ross3", "convec", "tcorr3", "end")


def states(model: str, iterations=(1,)):
    return O.exact_states(model, iterations=iterations, at=HOOKS)


def _v(s, key):
    return torch.as_tensor(O.vec(s, key))[None]


def solver(model: str):
    from atlas_torch.v2.structure import ModelSpec, StructureConfig, StructureSolver
    mix = {"sun": "solar", "t4500g25": "solar", "t9000g40": "solar", "t6500g40m10": "m10", "t5000g45m20": "m20",
           "t5500g40alpha": "alpha"}[model]
    d = O.stock_dir(model)
    spec = ModelSpec.from_control(d / "atlas_control.com", O.EXT / "runs" / "stock" / "odf" / mix)
    return StructureSolver(StructureConfig.kurucz(), ModelSpec.stack([spec]))


class PortContinuum:
    """The exact port's KAPP(0) continuum of a State as a v2 continuum provider: absorption ACONT, scattering
    SIGMAC, at the State's 337 frequencies."""

    def __init__(self, s):
        from atlas_torch.atlas9.kapp import OPS, kapp_combine
        ops = O.continuum(s)
        t = {k: torch.as_tensor(ops[k]) for k in OPS}
        bnu = torch.zeros_like(t["ahyd"])
        c = kapp_combine(t, bnu)
        self.ab, self.sc = c["acont"].T[None], c["sigmac"].T[None]

    def __call__(self, pops, nu):
        from atlas_torch.v2.opacity import Opacity
        return Opacity(self.ab.clone(), self.sc.clone())


def radiation(sv, s, continuum=None):
    """v2 RadiationMoments on the State `s` (hook 'frequencies': T, P, rhox before the integration)."""
    from atlas_torch.v2.atmosphere import Atmosphere
    from atlas_torch.v2.structure.radiation import RadiationMoments
    b, c = sv.batch, sv.config
    rhox, T, P = _v(s, "rhox.rhox"), _v(s, "temp.t"), _v(s, "state.p")
    pops = sv.eos.solve(Atmosphere(rhox=rhox, T=T, P=P), b.composition)
    with torch.no_grad():
        return RadiationMoments.compute(rhox, pops, sv.quadrature, sv.samples, continuum or sv.continuum, b.odf,
                                        sv.transfer, const=c.const, truncation=c.odf_truncation,
                                        lambda_diagonal=c.lambda_diagonal, normalization=c.rosseland_norm,
                                        derivative_scheme=sv.scheme)


def port_radiation(s_ross3, s_convec) -> dict:
    """The port's integrals: raw sums at hook 'ross3', ACCRAD / ABROSS / PRADK0 after RADIAP(3) / ROSS(3)."""
    T = _v(s_ross3, "temp.t")
    return {"flux_rad": _v(s_ross3, "flux.flxrad"), "j_minus_s": _v(s_ross3, "save.tcorr.rjmins"),
            "dkappa_flux": _v(s_ross3, "save.tcorr.rdabh"), "lambda_denominator": _v(s_ross3, "save.tcorr.rdiagj"),
            "accel_rad_raw": _v(s_ross3, "rad.accrad") * 12.5664 / 2.997925e10, "kappa_ross": _v(s_convec, "abross.abross"),
            "p_rad_top_raw": float(s_ross3.scalar("pzero.pradk0")) * 12.5664 / 2.997925e10, "T": T}


def rel(a, b, mask=None):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    r = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    return float(r[mask].max() if mask is not None else r.max())


def convection(sv, s):
    """v2 ThermoDerivatives (kurucz) + MixingLength on the State at hook 'convec' (after ROSS3/HIGH/RADIAP3)."""
    from atlas_torch.v2.atmosphere import Atmosphere
    from atlas_torch.v2.structure.grid import derivative
    from atlas_torch.v2.structure.thermodynamics import ThermoDerivatives
    b = sv.batch
    rhox, T, P = _v(s, "rhox.rhox"), _v(s, "temp.t"), _v(s, "state.p")
    ptot, kr, tau = _v(s, "ptotal.ptotal"), _v(s, "abross.abross"), _v(s, "abross.tauros")
    pradk, height = _v(s, "pzero.pradk"), _v(s, "height.height") * 1e5
    g = b.gravity
    pops = sv.eos.solve(Atmosphere(rhox=rhox, T=T, P=P), b.composition)
    td = ThermoDerivatives.of(sv.eos, T, P, b.composition, ptot, pradk, -torch.expm1(-tau), "kurucz", pops=pops)
    grad = ptot / (T * g[:, None]) * derivative(rhox, T, "atlas")
    k0 = sv.table_kappa(T, P, torch.full_like(T, float(b.vturb[0])))
    bubble = lambda Tb: sv.table_kappa(Tb, P, torch.full_like(T, float(b.vturb[0]))) / k0      # noqa: E731
    F = 5.6697e-5 * b.teff ** 4 / 12.5664
    cv = sv.mlt(T, td.rho, ptot, g, kr, td, grad, b.mixing_length, nconv=b.nconv, overshoot_weight=b.overshoot_weight,
                flux=F, height=height, bubble_ratio=bubble)
    return td, grad, cv


def correction(sv, s):
    """v2 TemperatureCorrection (kurucz) on the State at hook 'tcorr3', fed the port's moments and convection."""
    from atlas_torch.v2.structure.convection import ConvectiveFlux, mixing_length_scales
    b = sv.batch
    F = torch.tensor([float(s.scalar("flux.flux"))])
    T, rho, ptot, grad = _v(s, "temp.t"), _v(s, "state.rho"), _v(s, "ptotal.ptotal"), _v(s, "conv.dltdlp")
    rad = SimpleNamespace(flux_rad=_v(s, "flux.flxrad"), kappa_ross=_v(s, "abross.abross"),
                          j_minus_s=_v(s, "save.tcorr.rjmins"), dkappa_flux=_v(s, "save.tcorr.rdabh"),
                          lambda_denominator=_v(s, "save.tcorr.rdiagj"))
    td = SimpleNamespace(grad_ad=_v(s, "conv.grdadb"), heat_cp=_v(s, "conv.heatcp"), dlnrho_dlnT=_v(s, "conv.dlrdlt"),
                         rho=rho)
    hp, v0, f0 = mixing_length_scales(T, rho, ptot, b.gravity, td, b.mixing_length)
    cv = ConvectiveFlux(_v(s, "conv.flxcnv"), _v(s, "conv.flxcnv0"), torch.zeros_like(T), hp, grad, grad - td.grad_ad,
                        v0, f0, b.mixing_length[:, None])
    return sv.tcorr(_v(s, "rhox.rhox"), T, _v(s, "abross.tauros"), rad, cv, td, F, b.teff)
