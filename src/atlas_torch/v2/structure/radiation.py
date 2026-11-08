"""Frequency-integrated moments of the radiation field of a model with ODF line opacity: the integrals the
structure equations need (Kurucz 1970 secs. 2.10-2.12, 7.2-7.3; atlas9mem.for TCORR(2), RADIAP(2), ROSS(2)).

    rad = RadiationMoments.compute(rhox, pops, quadrature, samples, continuum, odf, transfer)
    rad.flux_rad, rad.kappa_ross, rad.accel_rad ...         # (M, L)

Every sample k (frequency nu_i, ODF step s, weight w_i omega_s) is one transfer problem with extinction
continuum + ODF step opacity, source function S = (1 - alpha) B + alpha J (LTE + coherent scattering).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..constants import CODATA2018, PhysicalConstants
from ..opacity import planck, planck_derivative
from ..opacity.frequencies import SampleSet
from ..opacity.rosseland import rosseland_mean
from .grid import derivative

EULER_GAMMA = 0.5772156649015329
# ATLAS9 main program: a frequency's ODF steps end once H_step/H_cont > .9995 at the surface or >= .998 at every
# depth (labels 24-71); a remaining weight below 1e-4 is dropped (label 79)
TRUNCATE_SURFACE, TRUNCATE_ALL, MIN_REMAINDER = 0.9995, 0.998, 1e-4


def expint3(x: torch.Tensor) -> torch.Tensor:
    """Exponential integral E_3(x) = (exp(-x) (1 - x) + x^2 E_1(x)) / 2 for 0 < x <= 10, E_1 by its power series
    (Abramowitz & Stegun 5.1.11, 5.1.14)."""
    term = -x
    s = torch.zeros_like(x)
    for k in range(1, 61):
        s = s - term / k
        term = -term * x / (k + 1)
    e1 = -EULER_GAMMA - torch.log(x) + s
    return 0.5 * (torch.exp(-x) * (1.0 - x) + x * x * e1)


def kurucz_lambda_diagonal(tau: torch.Tensor) -> torch.Tensor:
    """TCORR(2)'s estimate of diag(Lambda) on the monochromatic depths tau (..., L): S linear between the layers,
    each layer coupled to the interval below it and the one above (Kurucz 1970 eqs. 7.22-7.23):
    half-interval term (d + E3(d) - 1/2) / (2 d) of the interval d = tau_j+1 - tau_j (the last layer reuses the
    interval above it), its small-d expansion for d <= .01, E3 = 0 for d >= 10."""
    d = tau[..., 1:] - tau[..., :-1]
    d = torch.cat([d, d[..., -1:]], -1).clamp(min=1e-10)
    big = d > 0.01
    db = torch.where(big, d, torch.ones_like(d))
    e3 = torch.where(db < 10.0, expint3(db.clamp(max=10.0)), torch.zeros_like(db))
    t_big = 0.5 * (db + e3 - 0.5) / db
    ds = torch.where(big, torch.full_like(d, 0.01), d)
    t_small = (0.922784335098467 - torch.log(ds)) * ds / 4.0 + ds ** 2 / 12.0 - ds ** 3 / 96.0 + ds ** 4 / 720.0
    t2 = torch.where(big, t_big, t_small)
    return t2 + torch.cat([torch.zeros_like(t2[..., :1]), t2[..., :-1]], -1)


@dataclass(frozen=True)
class RadiationMoments:
    """Frequency integrals over the samples, (M, L) unless noted, Eddington-flux units (erg s^-1 cm^-2 sr^-1):

    flux_rad            H_rad = int H_nu dnu (FLXRAD)
    j_minus_s           int (kappa + sigma)(J - S) dnu (RJMINS), the flux-derivative error dH/dm (eq. 2.32)
    dkappa_flux         int dln(kappa + sigma)/dm H dnu (RDABH), the opacity-change term of eq. 7.4
    lambda_denominator  int (kappa + sigma)(Lambda_d - 1)/(1 - alpha Lambda_d)(1 - alpha) dB/dT dnu (RDIAGJ, eq. 7.21)
    accel_rad           radiative acceleration g_rad = (4 pi / c) int (kappa + sigma) H dnu, cm s^-2 (eq. 2.83)
    p_rad_top           P_rad at tau = 0: (4 pi / c) int K(0) dnu, dyn cm^-2 (M,)
    energy_density      (4 pi / c) int J dnu, erg cm^-3 (RADEN)
    kappa_ross          Rosseland mean of the same extinction and weights, cm^2 g^-1 (ROSS)
    surface_flux        H_nu(0) per frequency of the quadrature (M, F), when requested
    """
    flux_rad: torch.Tensor
    j_minus_s: torch.Tensor
    dkappa_flux: torch.Tensor
    lambda_denominator: torch.Tensor
    accel_rad: torch.Tensor
    p_rad_top: torch.Tensor
    energy_density: torch.Tensor
    kappa_ross: torch.Tensor
    surface_flux: torch.Tensor | None = None

    @classmethod
    def compute(cls, rhox: torch.Tensor, pops, quadrature, samples: SampleSet, continuum, odf, transfer, *,
                const: PhysicalConstants = CODATA2018, table_index=None, truncation: bool = False,
                lambda_diagonal: str = "transfer", normalization: str = "quadrature", derivative_scheme="parabolic",
                chunk: int = 1024, surface: bool = False) -> "RadiationMoments":
        """Moments of the models (M, L) with populations `pops` on the column-mass grid rhox (M, L).

        quadrature: FrequencyQuadrature (F nodes); samples: its SampleSet; continuum: ContinuousOpacity;
        odf: OpacityDistribution (one vturb; table_index (M,) for a stacked table); transfer: rt.Transfer.
        truncation: ATLAS9's early end of the ODF steps (the remaining weight of a frequency to its continuum
        solution); lambda_diagonal: "transfer" (diag(Lambda) of the solver) | "kurucz" (kurucz_lambda_diagonal).
        Samples are solved `chunk` at a time, never splitting a frequency."""
        T, P = pops.T, pops.P
        M, L = T.shape
        dev = T.device
        nu_f = quadrature.nu.to(dev)
        F = nu_f.shape[0]
        cont = continuum(pops, nu_f)
        ab_c, sc_c = cont.absorption, cont.scattering                         # (M, L, F)
        b_f, db_f = planck(nu_f, T, const), planck_derivative(nu_f, T, const)
        x = rhox[:, None, :]
        lt = lambda v: v.transpose(-1, -2)                                     # noqa: E731

        def solve(ab, sc, b):
            sol = transfer.solve(x, lt(ab), lt(sc), lt(b), moments=True)
            return sol

        def lam_d(sol):
            return kurucz_lambda_diagonal(sol.tau) if lambda_diagonal == "kurucz" else sol.lam_diag

        acc = {k: torch.zeros(M, L, dtype=T.dtype, device=dev) for k in ("H", "jms", "dkh", "den", "acc", "J")}
        k0 = torch.zeros(M, dtype=T.dtype, device=dev)
        surf = torch.zeros(M, F, dtype=T.dtype, device=dev) if surface else None

        def add(sol, ab, sc, db, w):
            """Accumulate the moments of solutions (M, k, L) with weights w (M, k)."""
            nonlocal k0
            ext = lt(ab + sc)                                                  # (M, k, L)
            alpha = lt(sc) / ext
            ld = lam_d(sol)
            wk = w[..., None]
            acc["H"] += (wk * sol.H).sum(1)
            acc["J"] += (wk * sol.J).sum(1)
            acc["jms"] += (wk * ext * (sol.J - sol.S)).sum(1)
            acc["acc"] += (wk * ext * sol.H).sum(1)
            dlk = derivative(x, ext, derivative_scheme) / ext
            acc["dkh"] += (wk * dlk * sol.H).sum(1)
            acc["den"] += (wk * ext * (ld - 1.0) / (1.0 - alpha * ld) * (1.0 - alpha) * lt(db)).sum(1)
            k0 = k0 + (w * sol.K0).sum(1)

        cont_sol = None
        if truncation:
            cont_sol = solve(ab_c, sc_c, b_f)
            h_c = cont_sol.H                                                   # (M, F, L)
        remainder = torch.zeros(M, F, dtype=T.dtype, device=dev)
        ext_parts, w_parts, nu_parts = [], [], []
        w_s_all = samples.weight.to(dev)
        for sl in samples.chunks(chunk):
            sub = SampleSet(*(getattr(samples, k)[sl].to(dev) for k in ("nu", "weight", "frequency", "interval", "step")))
            fi = sub.frequency
            line = odf.sampled(T, P, sub, table_index=table_index)              # (M, L, k)
            ab = ab_c[..., fi] + line
            sc = sc_c[..., fi]
            sol = solve(ab, sc, b_f[..., fi])
            w = w_s_all[sl].expand(M, -1).clone()
            if truncation:
                w, rem = _truncate(sub, sol.H, h_c, w)
                remainder.index_add_(1, torch.unique(fi), rem)
            add(sol, ab, sc, db_f[..., fi], w)
            if surface:
                surf.index_add_(1, fi, w * sol.H[..., 0] / quadrature.weight.to(dev)[fi])
            ext_parts.append(ab + sc)
            w_parts.append(w)
            nu_parts.append(sub.nu)
        if truncation and bool((remainder > 0).any()):
            add(cont_sol, ab_c, sc_c, db_f, remainder)
            if surface:
                surf += remainder * cont_sol.H[..., 0] / quadrature.weight.to(dev)
            ext_parts.append(ab_c + sc_c)
            w_parts.append(remainder)
            nu_parts.append(nu_f)
        ext_all = torch.cat(ext_parts, -1)
        w_all = torch.cat(w_parts, -1)[:, None, :]
        kappa_r = rosseland_mean(ext_all, (torch.cat(nu_parts), w_all), T, normalization=normalization, const=const)
        c4 = 4.0 * math.pi / const.c
        return cls(flux_rad=acc["H"], j_minus_s=acc["jms"], dkappa_flux=acc["dkh"], lambda_denominator=acc["den"],
                   accel_rad=c4 * acc["acc"], p_rad_top=c4 * k0, energy_density=c4 * acc["J"], kappa_ross=kappa_r,
                   surface_flux=surf)


def _truncate(sub: SampleSet, h_line: torch.Tensor, h_cont: torch.Tensor, w: torch.Tensor):
    """ATLAS9's ODF truncation for the complete frequencies of a chunk: the steps of a frequency are taken
    strongest first; after the first step whose flux H_step/H_cont exceeds TRUNCATE_SURFACE at the surface or
    reaches TRUNCATE_ALL at every depth, the remaining steps are replaced by the continuum solution with the
    remaining weight. Returns the sample weights (M, k) and the continuum weights (M, n_freq) of the chunk's
    frequencies (in increasing order)."""
    fi = sub.frequency
    freqs, inv = torch.unique(fi, return_inverse=True)
    hc = h_cont[:, fi, :]                                                     # (M, k, L)
    r = torch.where(hc > 0, h_line / torch.where(hc > 0, hc, torch.ones_like(hc)), torch.ones_like(hc))
    stop = (r[..., 0] > TRUNCATE_SURFACE) | (r >= TRUNCATE_ALL).all(-1)          # (M, k)
    lines = sub.interval >= 0
    ns = int(sub.step.max()) + 1 if bool(lines.any()) else 1
    M, nf = w.shape[0], len(freqs)
    # samples of each frequency in ATLAS order (strongest stored step first): position ns - 1 - step
    pos = torch.where(lines, ns - 1 - sub.step, torch.zeros_like(sub.step))
    grid_stop = torch.zeros(M, nf, ns, dtype=torch.bool, device=w.device)
    grid_stop[:, inv, pos] = stop
    stopped_before = torch.cumsum(grid_stop.to(torch.int64), -1) - grid_stop.to(torch.int64) > 0
    taken = ~stopped_before[:, inv, pos]
    taken = torch.where(lines, taken, torch.ones_like(taken))
    w_new = torch.where(taken, w, torch.zeros_like(w))
    total = torch.zeros(M, nf, dtype=w.dtype, device=w.device).index_add_(1, inv, w)
    used = torch.zeros(M, nf, dtype=w.dtype, device=w.device).index_add_(1, inv, w_new)
    rem = total - used
    rem = torch.where(rem > MIN_REMAINDER * total, rem, torch.zeros_like(rem))
    return w_new, rem
