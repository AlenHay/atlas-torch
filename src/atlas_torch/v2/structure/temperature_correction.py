"""Temperature correction (Kurucz 1970 ch. 7): flux correction (Avrett & Krook 1963, modified for convection,
eqs. 7.3-7.15), Lambda correction from the flux derivative (eqs. 7.16-7.25), surface correction (eqs. 7.26-7.28);
atlas9mem.for TCORR(3).

    tc = TemperatureCorrection(mlt=MixingLength())
    out = tc(rhox, T, tau_ross, moments, convection, thermo, flux, teff)   # Correction: dT (M, L), errors, parts
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..rt.optical_depth import cumulative_integral
from .convection import ConvectiveFlux, MixingLength
from .grid import derivative, map1


@dataclass(frozen=True)
class Correction:
    """dT = dt_flux + dt_lambda + dt_surface (K) and the flux error / derivative (percent), all (M, L);
    flux_conv: the convective flux the correction used (smoothed with smooth_flux)."""
    dT: torch.Tensor
    dt_flux: torch.Tensor
    dt_lambda: torch.Tensor
    dt_surface: torch.Tensor
    flux_error: torch.Tensor
    flux_derivative: torch.Tensor
    flux_conv: torch.Tensor


def _clip(x, lim):
    return torch.maximum(-lim, torch.minimum(lim, x))


@dataclass(frozen=True)
class TemperatureCorrection:
    """kurucz=True: TCORR's version: the convective flux zero in the first two layers, the convective terms of
    the Delta M coefficient only where H_conv/H_rad > 1e-3, |Delta tau| <= tau/5 instead of eq. 7.15, the Lambda
    correction cut where convective or tau >= 1 with the five layers above halved per cut layer.
    kurucz=False: eqs. 7.13-7.15, 7.21, 7.26-7.28 with the paper's limits (eq. 7.15; Teff/25 for the Lambda and
    surface corrections), the Lambda correction tapered linearly to 0 at tau = 1 and off in convective layers.
    mlt: the MixingLength whose radiative-loss parameter D enters eqs. 7.11-7.14 (evaluated at kappa_R, as TCORR).
    scheme: depth integrals and derivatives ("parabolic" | "atlas": INTEG/DERIV).
    smooth_flux: the convective flux is smoothed with weights (1/4, 1/2, 1/4) (TCORR loop 3101; without it the
    mixing-length flux keeps an alternating layer-to-layer error of ~0.2 % that the correction cannot remove)."""
    kurucz: bool = False
    mlt: MixingLength = field(default_factory=MixingLength)
    scheme: str = "parabolic"
    smooth_flux: bool = True

    def convective_flux(self, cv: ConvectiveFlux, convection_on=None) -> torch.Tensor:
        """H_conv as the correction sees it (TCORR loops 3100-3102), (M, L)."""
        cnv = cv.flux if convection_on is None else torch.where(convection_on[:, None], cv.flux,
                                                                 torch.zeros_like(cv.flux))
        if self.kurucz:
            cnv = torch.cat([torch.zeros_like(cnv[:, :2]), cnv[:, 2:]], -1)
        if self.smooth_flux:
            cnv = torch.cat([cnv[:, :1], 0.25 * cnv[:, :-2] + 0.5 * cnv[:, 1:-1] + 0.25 * cnv[:, 2:],
                             0.25 * cnv[:, -3:-2] + 0.25 * cnv[:, -2:-1] + 0.5 * cnv[:, -1:]], -1)
        return cnv

    def flux_correction(self, rhox, T, tau, rad, cv: ConvectiveFlux, rho, h_conv, flux) -> torch.Tensor:
        """Delta T_flux = -(dT/dm) Delta M (eq. 7.8), Delta M of eq. 7.14 integrated in tau_Ross (TCORR) and limited
        by eq. 7.15, (M, L)."""
        sch, F = self.scheme, flux[:, None]
        kr, h_rad = rad.kappa_ross, rad.flux_rad
        j = torch.arange(T.shape[-1], device=T.device)
        dtdm = derivative(rhox, T, sch)
        conv = (h_conv > 0) & (cv.flux_local > 0)
        delta = torch.where(conv, cv.delta, torch.ones_like(T))
        D = torch.where(conv, self.mlt.loss_parameter(T, rho, kr, cv), torch.zeros_like(T))
        damp = 1.0 + D / (D + delta)                                         # 1 + D/(D + nabla - nabla_A)
        # coefficient of d Delta M/dM (eq. 7.13) and of Delta M: the convective terms of eq. 7.12 and TCORR's
        # opacity-gradient term int dln(kappa+sigma)/dm H dnu - H_rad dln kappa_R/dm
        if self.kurucz:
            on = (h_conv / h_rad > 1e-3) & (cv.flux_local / h_rad > 1e-3)
            h_c = torch.where(on, h_conv, torch.zeros_like(h_conv))
        else:
            h_c = h_conv
        a = h_rad + h_conv * 1.5 * cv.grad / delta * damp
        b = rad.dkappa_flux - h_rad * derivative(rhox, kr, sch) / kr \
            + h_c * (dtdm / T * (1.0 - 9.0 * D / (D + delta)) + 1.5 * derivative(rhox, cv.grad, sch) / delta * damp)
        # integrating factor g = exp(int b/a dm) (eq. 7.14), zero coefficient in the two top layers (TCORR)
        g = torch.exp(cumulative_integral(rhox, torch.where(j < 2, torch.zeros_like(T), b / a), scheme=sch))
        a_ddel = h_rad + h_conv * 1.5 * cv.grad * torch.where(conv, damp / delta, torch.zeros_like(T))
        dtau = cumulative_integral(tau, g * (h_rad + h_conv - F) / a_ddel, scheme=sch) / g
        dtau = _clip(dtau, tau / 5.0) if self.kurucz else torch.clamp(dtau, min=-0.5 * tau, max=tau)
        return -dtau * dtdm / kr

    def __call__(self, rhox, T, tau, rad, cv: ConvectiveFlux, td, flux, teff, convection_on=None) -> Correction:
        """rhox, T, tau (the model's Rosseland depths) (M, L); rad: RadiationMoments (flux_rad, kappa_ross,
        j_minus_s, dkappa_flux, lambda_denominator); cv: ConvectiveFlux; td: ThermoDerivatives (rho); flux (target
        H), teff (M,); convection_on (M,) bool."""
        sch, F = self.scheme, flux[:, None]
        kr, h_rad = rad.kappa_ross, rad.flux_rad
        h_conv = self.convective_flux(cv, convection_on)
        dt_flux = self.flux_correction(rhox, T, tau, rad, cv, td.rho, h_conv, flux)
        # flux error and derivative (percent); radiative layers: dH/dtau = int (kappa+sigma)(J-S) dnu / kappa_R
        # (eq. 7.16)
        flxerr = (h_rad + h_conv - F) / F * 100.0
        radiative = h_conv / h_rad < 1e-5
        flxdrv = torch.where(radiative, rad.j_minus_s / kr / F * 100.0, derivative(tau, flxerr, sch))
        # Lambda correction (eq. 7.21), off in convective layers and below tau = 1 (sec. 7.3), limit Teff/25
        lim = (teff / 25.0)[:, None]
        lam = -flxdrv * F / 100.0 / rad.lambda_denominator * kr
        cut = ~(radiative & (tau < 1.0))
        if self.kurucz:
            c = cut.to(lam.dtype)
            n_cut = sum(torch.cat([c[:, k:], torch.zeros_like(c[:, :k])], -1) for k in range(1, 6))
            dt_lambda = torch.where(cut, torch.zeros_like(lam), _clip(lam, lim)) * torch.pow(0.5, n_cut)
        else:
            dt_lambda = torch.where(cut, torch.zeros_like(lam), torch.clamp(1.0 - tau, min=0.0) * _clip(lam, lim))
        # surface correction (eqs. 7.26-7.27): 4 T_0 Delta H_0/H less the mean of the other two over 0.1-2
        dtsur = _clip((F[:, 0] - h_rad[:, 0]) / F[:, 0] * 0.25 * T[:, 0], lim[:, 0])
        tint = cumulative_integral(tau, dt_flux + dt_lambda, scheme=sch)
        at = map1(torch.tensor([0.1, 2.0], dtype=T.dtype, device=T.device).expand(T.shape[0], 2), tau, tint)
        tav = (at[:, 1] - at[:, 0]) / 2.0
        tav = torch.where(dtsur * tav <= 0, torch.zeros_like(tav), tav)
        tav = torch.where(tav.abs() > dtsur.abs(), dtsur, tav)
        dt_surface = (dtsur - tav)[:, None].expand(T.shape)
        return Correction(dt_flux + dt_lambda + dt_surface, dt_flux, dt_lambda, dt_surface, flxerr, flxdrv, h_conv)


def accelerate(dT: torch.Tensor, previous: torch.Tensor, convective: torch.Tensor, n_free: torch.Tensor) -> torch.Tensor:
    """TCORR loop 61 (sec. 7.5): where not convective and above layer n_free (M,), dT x1.25 if it kept its sign
    and shrank since the previous iteration, x0.5 if it changed sign."""
    j = torch.arange(dT.shape[-1], device=dT.device)
    free = ~convective & (j[None, :] < n_free[:, None])
    grow = (previous * dT > 0) & (previous.abs() > dT.abs())
    flip = previous * dT < 0
    out = torch.where(free & grow, dT * 1.25, dT)
    return torch.where(free & flip, out * 0.5, out)
