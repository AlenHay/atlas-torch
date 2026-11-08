"""Mixing-length convection (Kurucz 1970 ch. 6, eqs. 6.1-6.16; atlas9mem.for CONVEC).

    mlt = MixingLength()                             # v2: kappa_R of the layer for the bubble
    cv = mlt(T, rho, p_total, gravity, kappa_ross, td, grad, mixing_length, ...)   # ConvectiveFlux (M, L)
    D = mlt.loss_parameter(T, rho, kappa_ross, cv)                                 # radiative-loss parameter
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..rt.optical_depth import cumulative_integral
from .grid import map1


@dataclass(frozen=True)
class ConvectiveFlux:
    """flux (H_conv, Eddington units, after overshooting and the top-layer cut), flux_local (before
    overshooting, CONVEC's FLXCNV0), velocity (cm s^-1), scale_height H_p = P_total/(rho g) (cm, eq. 4.47),
    grad (dln T/dln P_total), delta = grad - grad_ad, velocity_scale v_0 with v = v_0 (nabla - nabla_B)^1/2
    (cm s^-1, eq. 6.10), flux_scale f_0 with H_conv = f_0 v (nabla - nabla_B) (eq. 6.5 / 4 pi), all (M, L);
    mixing_length l/H_p (M, 1)."""
    flux: torch.Tensor
    flux_local: torch.Tensor
    velocity: torch.Tensor
    scale_height: torch.Tensor
    grad: torch.Tensor
    delta: torch.Tensor
    velocity_scale: torch.Tensor
    flux_scale: torch.Tensor
    mixing_length: torch.Tensor


def mixing_length_scales(T, rho, p_total, gravity, td, mixing_length):
    """(H_p, v_0, f_0), (M, L): H_p = P_total/(rho g) (eq. 4.47), v_0 = sqrt(-(1/8) g H_p (l/H_p)^2
    (dln rho/dln T)_P) (eq. 6.10), f_0 = (1/2)(l/H_p) rho c_P T / 4 pi (eq. 6.5)."""
    l = mixing_length[:, None]
    hp = p_total / (rho * gravity[:, None])
    v0 = 0.5 * l * torch.sqrt(torch.clamp(-0.5 * p_total / rho * td.dlnrho_dlnT, min=0.0))
    f0 = 0.5 * rho * td.heat_cp * T * l / (4.0 * math.pi)
    return hp, v0, f0


def radiative_loss(T, rho, kappa, hp, v0, f0, mixing_length, sigma: float, form: str = "atlas9") -> torch.Tensor:
    """Radiative-loss parameter D of eq. 6.15, (M, L), with d0 = 8 sigma T^4/(kappa H_p rho)/(4 pi f_0)/v_0:
    "atlas9" (CONVEC, TCORR): D = (d0 tau_b^2/(2 + tau_b^2))^2/2 with the bubble optical thickness
    tau_b = kappa rho l (the thin-bubble factor CONVEC attributes to Mihalas); this D is 8 times smaller than
    eq. 6.14 for an opaque bubble. "kurucz1970": eq. 6.14, D = (2 sqrt 2 d0)^2/2 (constant 32 sqrt 2, no thin-bubble
    factor)."""
    d0 = 8.0 * sigma * T ** 4 / (kappa * hp * rho) / (f0 * 4.0 * math.pi) / v0
    if form == "atlas9":
        taub = kappa * rho * mixing_length[:, None] * hp
        d = d0 * taub ** 2 / (2.0 + taub ** 2)
    elif form == "kurucz1970":
        d = 2.0 * math.sqrt(2.0) * d0
    else:
        raise ValueError("radiative_loss: 'atlas9' | 'kurucz1970'")
    return d ** 2 / 2.0


def bubble_gradient_excess(delta: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """nabla - nabla_bubble from eq. 6.15 with delta = nabla - nabla_ad >= 0 and the radiative-loss parameter D
    (radiative_loss): D + delta - sqrt(D^2 + 2 D delta), evaluated as delta^2 / ((D + delta)(1 + sqrt(1 - q))),
    q = (delta / (D + delta))^2 (the closed form of the series eq. 6.16, free of cancellation)."""
    s = d + delta
    s = torch.where(s > 0, s, torch.ones_like(s))
    q = (delta / s) ** 2
    return delta ** 2 / (s * (1.0 + torch.sqrt(torch.clamp(1.0 - q, min=0.0))))


@dataclass(frozen=True)
class MixingLength:
    """Mixing-length theory with a radiative-loss term (eqs. 6.5, 6.10, 6.14-6.16).

    kurucz        CONVEC's choices: the first three layers never convect (J < 4), the bubble opacity is the
                  harmonic mean of kappa_R(T +- dT_bubble) iterated with dT_bubble (needs bubble_ratio), the
                  series eq. 6.16 truncated at 1e-6; v2: kappa_R of the layer and the closed form of eq. 6.15
    sigma         Stefan-Boltzmann constant of the radiative-loss term (CONVEC: 5.6697E-5)
    min_layer     convection allowed from this layer (0-based); kurucz: 3
    bubble_average  the harmonic-mean bubble opacity (None: as kurucz)
    loss          radiative_loss form: "atlas9" (CONVEC's D, both configurations) | "kurucz1970" (eq. 6.14)"""
    kurucz: bool = False
    sigma: float = 5.670374419e-5
    min_layer: int = 0
    bubble_average: bool | None = None
    loss: str = "atlas9"

    @property
    def averages_bubble(self) -> bool:
        return self.kurucz if self.bubble_average is None else self.bubble_average

    def loss_parameter(self, T, rho, kappa, cv: ConvectiveFlux) -> torch.Tensor:
        """D (eq. 6.15) at opacity kappa for the scales of cv, (M, L); 0 where v_0 = 0 or l = 0."""
        ok = (cv.velocity_scale > 0) & (cv.mixing_length > 0)
        one = torch.ones_like(T)
        d = radiative_loss(T, rho, kappa, cv.scale_height, torch.where(ok, cv.velocity_scale, one),
                           torch.where(ok, cv.flux_scale, one), cv.mixing_length[:, 0], self.sigma, self.loss)
        return torch.where(ok, d, torch.zeros_like(d))

    def __call__(self, T, rho, p_total, gravity, kappa_ross, td, grad, mixing_length, *, nconv=None,
                 overshoot_weight=None, flux=None, height=None, bubble_ratio=None, enabled=None) -> ConvectiveFlux:
        """T, rho, p_total, kappa_ross, grad (dln T/dln P_total) (M, L); td: ThermoDerivatives; gravity,
        mixing_length (l/H_p), overshoot_weight, flux (target H), nconv (M,); height (M, L) cm;
        bubble_ratio(T_new) -> kappa_R(T_new, P)/kappa_R(T, P) (M, L); enabled (M,) bool."""
        l = mixing_length[:, None]
        hp, vco, fluxco = mixing_length_scales(T, rho, p_total, gravity, td, mixing_length)
        delta = grad - td.grad_ad
        L = T.shape[-1]
        first = 3 if self.kurucz else self.min_layer
        active = (delta >= 0) & (vco > 0) & (l > 0) & (torch.arange(L, device=T.device) >= first)
        if enabled is not None:
            active = active & enabled[:, None]
        vco_s = torch.where(active, vco, torch.ones_like(vco))
        fco_s = torch.where(active, fluxco, torch.ones_like(fluxco))

        def excess(kappa_b):
            """nabla - nabla_B (eq. 6.15)."""
            d = radiative_loss(T, rho, kappa_b, hp, vco_s, fco_s, mixing_length, self.sigma, self.loss)
            if self.kurucz:
                dd = torch.where(active, (delta / (d + delta)) ** 2, torch.zeros_like(d))
                return _kurucz_series(dd) * delta ** 2 / (d + delta)
            return bubble_gradient_excess(torch.where(active, delta, torch.zeros_like(delta)), d)

        if self.averages_bubble and bubble_ratio is not None:
            dt = torch.zeros_like(T)
            kappa_b = kappa_ross
            done = ~active
            for _ in range(30):
                kb = 2.0 / (1.0 / bubble_ratio(T + dt) + 1.0 / bubble_ratio(T - dt)) * kappa_ross
                kappa_b = torch.where(done, kappa_b, kb)
                ex = excess(kappa_b)
                new = torch.minimum(T * l * ex, 0.15 * T) * 0.7 + dt * 0.3
                conv = (new < dt + 0.5) & (new > dt - 0.5)
                dt = torch.where(done, dt, new)
                done = done | conv
                if bool(done.all()):
                    break
        else:
            kappa_b = kappa_ross
        ex = torch.where(active, excess(kappa_b), torch.zeros_like(T))
        v = torch.where(active, vco * torch.sqrt(ex), torch.zeros_like(T))
        fl = torch.clamp(fluxco * v * ex, min=0.0)
        fl = torch.where(active, fl, torch.zeros_like(fl))
        local = fl
        if overshoot_weight is not None and bool((overshoot_weight > 0).any()):
            fl = self._overshoot(fl, hp, height, flux, overshoot_weight)
        if nconv is not None:
            cut = torch.arange(L, device=T.device)[None, :] < nconv[:, None]
            fl = torch.where(cut, torch.zeros_like(fl), fl)
        return ConvectiveFlux(fl, local, v, hp, grad, delta, vco, fluxco, l)

    @staticmethod
    def _overshoot(fl, hp, height, flux, weight):
        """CONVEC 774-779: in the lower half of the atmosphere the convective flux becomes the larger of its local
        value and its mean over +- dz = min(w min(max H_conv/H, 1) H_p/2, distance to the boundaries)."""
        wt = torch.clamp((fl / flux[:, None]).amax(-1), max=1.0) * weight
        dz = torch.minimum(torch.minimum(hp * 0.5 * wt[:, None], height[:, -1:] - height), height - height[:, :1])
        integ = cumulative_integral(height, fl, scheme="atlas")
        L = fl.shape[-1]
        j = torch.arange(L, device=fl.device)
        lower = (j >= L // 2 - 1) & (j < L - 1)
        dzs = torch.where(dz > 0, dz, torch.ones_like(dz))
        avg = (map1(height + dzs, height, integ) - map1(height - dzs, height, integ)) / dzs / 2.0
        mean = torch.where(lower & (dz > 0), torch.clamp(avg, min=0.0), torch.zeros_like(avg))
        return torch.maximum(fl, mean)


def _kurucz_series(q: torch.Tensor) -> torch.Tensor:
    """(1 - sqrt(1 - q)) / q as CONVEC evaluates it: the closed form for q >= .5, else the series of eq. 6.16
    summed until a term drops below 1e-6."""
    closed = (1.0 - torch.sqrt(torch.clamp(1.0 - q, min=0.0))) / torch.where(q > 0, q, torch.ones_like(q))
    s = torch.full_like(q, 0.5)
    term = torch.full_like(q, 0.5)
    up, down = -1.0, 2.0
    live = torch.ones_like(q, dtype=torch.bool)
    for _ in range(200):
        up += 2.0
        down += 2.0
        term = up / down * q * term
        s = torch.where(live, s + term, s)
        live = live & (term > 1e-6)
        if not bool(live.any()):
            break
    return torch.where(q >= 0.5, closed, s)
