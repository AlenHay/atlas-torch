"""The model-atmosphere iteration (Kurucz 1970 sec. 1 and ch. 7; atlas9mem.for main program): on the fixed Rosseland
grid, hydrostatic equilibrium -> populations -> radiation field with ODF line opacity -> Rosseland depth, radiation
pressure -> mixing-length convection -> temperature correction, until the total flux is constant.

    batch = ModelSpec.stack([ModelSpec.from_control(ctl, odf_dir) for ctl in controls])
    solver = StructureSolver(StructureConfig.kurucz(), batch)
    result = solver.run(solver.start([read_restart(p) for p in restarts]))
    result.write(run_dir, model=0)             # output_main.out, output_last_iteration.out, output_summary.out

Iteration path: the temperature correction moves T at the model's own Rosseland depths tau_model =
int kappa_R dm; rhox, T, P_rad, P_turb, n_e and kappa_R are interpolated (MAP1) from tau_model onto the fixed grid,
and the column masses are scaled by the change of P_total(tau) in hydrostatic equilibrium (eq. 4.10) between the
old and the corrected T, with kappa_R = (the model's kappa_R) x (kappa(T, P)/kappa(T_old, P_old)) from the
kappa.ros table or from dln kappa_R/dln T, P of the model's opacity (atlas9mem.for TCORR: rhox correction + MAP1
remap). At convergence the scaling is 1 and int kappa_R dm is the fixed grid, with the model's own kappa_R.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from ..atmosphere import Atmosphere
from ..rt.optical_depth import cumulative_integral
from .config import ModelBatch, StructureConfig, t_floor
from .convection import MixingLength
from .grid import derivative, map1
from .hydrostatic import gas_pressure, pressure_on_tau
from .radiation import RadiationMoments
from .temperature_correction import TemperatureCorrection, accelerate
from .thermodynamics import ThermoDerivatives


@dataclass(frozen=True)
class StructureState:
    """The structure on the Rosseland grid tau (M, L): rhox (g cm^-2), T (K), P (gas, dyn cm^-2), n_e (cm^-3),
    vturb (cm/s), p_rad (P_rad - P_rad(0), dyn cm^-2), p_rad_top (P_rad(0), (M,)), p_turb, kappa_ross, and the
    last iteration's accel_rad, flux_rad, flux_conv, velocity, sound_speed (M, L); dT_previous: the last
    correction (sign-history acceleration)."""
    tau: torch.Tensor
    rhox: torch.Tensor
    T: torch.Tensor
    P: torch.Tensor
    n_e: torch.Tensor
    vturb: torch.Tensor
    p_rad: torch.Tensor
    p_rad_top: torch.Tensor
    p_turb: torch.Tensor
    kappa_ross: torch.Tensor
    accel_rad: torch.Tensor
    flux_rad: torch.Tensor
    flux_conv: torch.Tensor
    velocity: torch.Tensor
    sound_speed: torch.Tensor
    dT_previous: torch.Tensor | None = None

    def atmosphere(self, batch: ModelBatch | None = None) -> Atmosphere:
        """v2 Atmosphere with extra ABROSS, ACCRAD, FLXRAD, VCONV, VELSND (READ DECK6 columns)."""
        teff = logg = None
        if batch is not None:
            teff, logg = batch.teff, torch.log10(batch.gravity)
        return Atmosphere(rhox=self.rhox, T=self.T, P=self.P, n_e=self.n_e, vturb=self.vturb, teff=teff, logg=logg,
                          extra={"ABROSS": self.kappa_ross, "ACCRAD": self.accel_rad, "FLXRAD": self.flux_rad,
                                 "VCONV": self.velocity, "VELSND": self.sound_speed})

    def select(self, idx: torch.Tensor) -> "StructureState":
        return StructureState(**{f.name: (None if getattr(self, f.name) is None else getattr(self, f.name)[idx])
                                 for f in fields(self)})


@dataclass
class ModelResult:
    """Per model: the iteration records (numpy dicts, see StructureSolver.step), the final state (one model),
    the best iteration (BasicATLAS atlas_converged rule) and why the iteration stopped."""
    batch: ModelBatch
    records: list
    final: list
    stopped: list
    seconds: float = 0.0
    quadrature: object = None

    def best(self, m: int) -> int:
        from ..io.atlas_output import select_best_iteration
        recs = self.records[m]
        return select_best_iteration([r["max_flux_error"] for r in recs], [r["max_flux_derivative"] for r in recs])

    def write(self, directory, m: int = 0) -> dict:
        from ..io.atlas_output import write_atlas_run
        q = self.quadrature
        return write_atlas_run(directory, self.batch.specs[m], self.records[m], self.best(m), self.stopped[m],
                               frequencies=None if q is None else (q.nu, q.weight))


class StructureSolver:
    """The iteration of a batch of models (ModelBatch) under a StructureConfig, on the batch's device."""

    def __init__(self, config: StructureConfig, batch: ModelBatch, *, chunk: int = 1024):
        self.config, self.batch, self.chunk = config, batch, chunk
        c = config
        self.device = batch.device
        self.const = c.const
        self.eos = c.equation_of_state(device=self.device)
        self.continuum = c.continuous_opacity()
        self.transfer = c.transfer_solver()
        self.quadrature = c.quadrature(batch.odf.intervals).to(self.device)
        self.samples = self.quadrature.samples(batch.odf.fractions)
        self.mlt = MixingLength(kurucz=bool(c.convection.get("kurucz", False)), sigma=c.const.sigma,
                                bubble_average=c.convection.get("bubble_average"),
                                loss=c.convection.get("radiative_loss", "atlas9"))
        self.scheme = c.depth_scheme
        self.tcorr = TemperatureCorrection(kurucz=(c.correction == "kurucz"), mlt=self.mlt, scheme=self.scheme,
                                           smooth_flux=bool(c.convection.get("smooth_flux", True)))
        self.t_min = t_floor(c, batch.odf)
        self._ross = None
        self._pops = None

    # ------------------------------------------------------------------ opacity helpers

    def rosseland_opacity(self, batch: ModelBatch | None = None):
        """RosselandOpacity of the model physics on the ODF table(s) of `batch` (rosseland='model')."""
        odf = (batch or self.batch).odf
        if self._ross is None or self._ross.odf is not odf:
            from ..opacity.rosseland import RosselandOpacity
            self._ross = RosselandOpacity(odf, self.config.rosseland_config())
            self._ross.eos = self.eos                  # the same physics, on the batch's device
        return self._ross

    def table_kappa(self, T, P, vturb, batch: ModelBatch | None = None):
        """kappa_R from the kappa.ros tables, (M, L)."""
        b = batch or self.batch
        out = torch.empty_like(T)
        for k, tab in enumerate(b.kappa_tables):
            if tab is None:
                raise ValueError("rosseland='table' needs a kappa.ros table for every model")
            sel = b.kappa_index == k
            out[sel] = tab.kappa(T[sel], P[sel], vturb[sel])
        return out

    def kappa_function(self, vturb, batch: ModelBatch | None = None):
        """kappa_R(T, P) of the configured Rosseland source (start model)."""
        b = batch or self.batch
        if self.config.rosseland == "table":
            return lambda T, P: self.table_kappa(T, P, vturb, b)
        ross = self.rosseland_opacity(b)
        return lambda T, P: ross(T, P, b.composition, table_index=b.table_index)

    def kappa_response(self, T0, P0, vturb, batch: ModelBatch | None = None):
        """r(T, P) = kappa_R(T, P) / kappa_R(T0, P0): from the tables, or log-linear with the model's
        dln kappa_R / dln T, P (forward-mode derivatives through the EOS and opacities; central differences with
        the lagged lowering)."""
        b = batch or self.batch
        if self.config.rosseland == "table":
            k0 = self.table_kappa(T0, P0, vturb, b)
            return lambda T, P: self.table_kappa(T, P, vturb, b) / k0
        ross = self.rosseland_opacity(b)
        if self.config.eos.get("lagged_lowering"):          # kappa_R is non-local: central differences
            h = 1e-3
            k = lambda t, p: torch.log(ross(t, p, b.composition, table_index=b.table_index))    # noqa: E731
            a = (k(T0 * (1 + h), P0) - k(T0 * (1 - h), P0)) / (math.log1p(h) - math.log1p(-h))
            bb = (k(T0, P0 * (1 + h)) - k(T0, P0 * (1 - h))) / (math.log1p(h) - math.log1p(-h))
        else:
            with torch.enable_grad():
                a, bb = ross.log_derivatives(T0, P0, b.composition, table_index=b.table_index)
        a, bb = a.detach(), bb.detach()
        return lambda T, P: torch.exp(a * torch.log(T / T0) + bb * torch.log(P / P0))

    # ------------------------------------------------------------------ start

    def start(self, restarts: list) -> StructureState:
        """Start model from punched decks (READ PUNCH + SCALE, READIN 1170/2700): T(tau_Ross) scaled by
        Teff/Teff_deck onto the grid, P_rad(0) and P_rad by (Teff/Teff_deck)^4, then hydrostatic equilibrium on the
        grid (TTAUP with the kappa.ros table, or the model's kappa_R)."""
        b = self.batch
        M, dev = len(b), self.device
        tau = torch.pow(10.0, b.log_tau)
        Ts, prads, tops = [], [], []
        for m, r in enumerate(restarts):
            a = r.atmosphere
            tr = r.tau_ross().to(dev)
            ratio = float(b.teff[m]) / float(a.teff) if a.teff is not None and math.isfinite(float(a.teff)) else 1.0
            Ts.append(map1(tau[m], tr, a.T.to(dev)) * ratio)
            acc = torch.nan_to_num(a.extra["ACCRAD"]).to(dev)
            rhox = a.rhox.to(dev)
            pr = cumulative_integral(rhox, acc, initial=acc[0] * rhox[0], scheme="atlas")
            prads.append(map1(tau[m], tr, pr) * ratio ** 4)
            tops.append(float(r.pradk0) * ratio ** 4)
        T = torch.stack(Ts)
        if self.t_min is not None:
            T = torch.clamp(T, min=self.t_min)
        p_rad = torch.stack(prads)
        vturb = b.vturb[:, None].expand(M, tau.shape[-1]).clone()
        p_turb = torch.zeros_like(T)
        kap = self.kappa_function(vturb)
        p0 = None
        if self.config.rosseland == "model" and all(t is not None for t in b.kappa_tables):
            with torch.no_grad():      # Newton start of the model-opacity solution: the kappa.ros solution
                p0 = pressure_on_tau(b.log_tau, T, b.gravity, lambda t, p: self.table_kappa(t, p, vturb), p_rad=p_rad,
                                     p_turb=p_turb, scheme=self.scheme)[1]
        with torch.no_grad():
            P, p_tot, rhox = pressure_on_tau(b.log_tau, T, b.gravity, kap, p_rad=p_rad, p_turb=p_turb, p_total0=p0,
                                             scheme=self.scheme)
        z = torch.zeros_like(T)
        return StructureState(tau=tau, rhox=rhox, T=T, P=P, n_e=torch.full_like(T, float("nan")), vturb=vturb,
                              p_rad=p_rad, p_rad_top=torch.as_tensor(tops, dtype=T.dtype, device=dev), p_turb=p_turb,
                              kappa_ross=kap(T, P), accel_rad=z, flux_rad=z, flux_conv=z, velocity=z, sound_speed=z)

    # ------------------------------------------------------------------ one iteration

    def step(self, s: StructureState, batch: ModelBatch | None = None, iteration_in_batch: int = 2):
        """One iteration; returns (new state, record). record (numpy, per model axis first): the iteration
        table (RHOX, T, P, XNE, RHO, ABROSS, HEIGHT, TAUROS, FLXCNV, PRADK, FLXERR, FLXDRV), the punched deck
        (deck_*), max_flux_error, max_flux_derivative, max_dT."""
        b = batch or self.batch
        c = self.config
        const = self.const
        g = b.gravity
        F = const.sigma * b.teff ** 4 / (4.0 * math.pi)
        # hydrostatic equilibrium on the column-mass grid (eq. 4.5) and populations
        P = gas_pressure(s.rhox, g, s.p_rad, s.p_turb)
        guess = self._pops if self._pops is not None and self._pops.T.shape == s.T.shape else None
        pops = self.eos.solve(Atmosphere(rhox=s.rhox, T=s.T, P=P), b.composition, guess=guess)
        self._pops = pops
        rad = RadiationMoments.compute(
            s.rhox, pops, self.quadrature, self.samples, self.continuum, b.odf, self.transfer, const=const,
            table_index=b.table_index, truncation=c.odf_truncation, lambda_diagonal=c.lambda_diagonal,
            normalization=c.rosseland_norm, derivative_scheme=self.scheme, chunk=self.chunk)
        kr = rad.kappa_ross
        tau_m = cumulative_integral(s.rhox, kr, initial=kr[:, 0] * s.rhox[:, 0], scheme=self.scheme)
        height = cumulative_integral(s.rhox, 1.0 / pops.rho, scheme=self.scheme)
        acc, top = rad.accel_rad, rad.p_rad_top
        if c.radiation_pressure_fudge:      # RADIAP(3): fudge against large flux errors
            hr = rad.flux_rad / F[:, None]
            acc = torch.where(hr > 1.0, acc / hr, acc)
            top = torch.where(hr[:, 0] > 1.0, top / hr[:, 0], top)
        p_rad = cumulative_integral(s.rhox, acc, initial=acc[:, 0] * s.rhox[:, 0], scheme=self.scheme)
        p_radk = p_rad + top[:, None]
        p_total = g[:, None] * s.rhox + s.p_rad_top[:, None] + s.p_turb[:, :1]
        rho = pops.rho
        # convection
        grad = p_total / (s.T * g[:, None]) * derivative(s.rhox, s.T, self.scheme)
        dil = -torch.expm1(-tau_m)
        td = ThermoDerivatives.of(self.eos, s.T, P, b.composition, p_total, p_radk, dil,
                                  mode="kurucz" if c.thermodynamics == "kurucz" else "exact", pops=pops)
        bubble = None
        if self.mlt.averages_bubble:         # kappa_R(T +- dT_bubble, P) / kappa_R(T, P)
            r_b = self.kappa_response(s.T, P, s.vturb, b)
            bubble = lambda Tb: r_b(Tb, P)                                          # noqa: E731
        cv = self.mlt(s.T, rho, p_total, g, kr, td, grad, b.mixing_length, nconv=b.nconv,
                      overshoot_weight=b.overshoot_weight, flux=F, height=height, bubble_ratio=bubble,
                      enabled=b.convection)
        # temperature correction
        corr = self.tcorr(s.rhox, s.T, tau_m, rad, cv, td, F, b.teff, convection_on=b.convection)
        dT = corr.dT
        if c.accelerate and iteration_in_batch > 1 and s.dT_previous is not None:
            convective = b.convection[:, None] & (corr.flux_conv / (corr.flux_conv + rad.flux_rad) > 0)
            n_free = torch.where(b.convection, dT.shape[-1] // 3 - 1, dT.shape[-1])
            dT = accelerate(dT, s.dT_previous, convective, n_free)
        T_new = s.T + dT
        if c.monotonic_T:       # TCORR loop 6050: T_j <= T_j+1 - 1, from the bottom up
            j = torch.arange(T_new.shape[-1], dtype=T_new.dtype, device=T_new.device)
            T_new = torch.flip(torch.cummin(torch.flip(T_new - j, [-1]), -1).values, [-1]) + j
        if self.t_min is not None:
            T_new = torch.clamp(T_new, min=self.t_min)
        # onto the fixed grid, then hydrostatic equilibrium there
        x_new = s.tau
        rm = lambda f: map1(x_new, tau_m, f)                                          # noqa: E731
        T_std, T_old, P_old = rm(T_new), rm(s.T), rm(P)
        kr_std, pr_std, pt_std = rm(kr), rm(p_rad), rm(s.p_turb)
        if c.turbulent_pressure:
            pt_std = rm(0.5 * rho * s.vturb ** 2)
        resp = self.kappa_response(T_old, P_old, s.vturb, b)
        kap = lambda Tn, Pn: kr_std * resp(Tn, Pn)                                    # noqa: E731
        rhox_std = rm(s.rhox)
        solve = lambda Tn: pressure_on_tau(b.log_tau, Tn, g, kap, p_rad=pr_std, p_turb=pt_std,   # noqa: E731
                                           p_total0=g[:, None] * rhox_std, scheme=self.scheme)[1]
        rhox_new = rhox_std * solve(T_std) / solve(T_old)
        P_std = gas_pressure(rhox_new, g, pr_std, pt_std)
        new = StructureState(tau=s.tau, rhox=rhox_new, T=T_std, P=P_std, n_e=rm(pops.n_e), vturb=s.vturb,
                             p_rad=pr_std, p_rad_top=top, p_turb=pt_std, kappa_ross=kr_std, accel_rad=acc,
                             flux_rad=rad.flux_rad, flux_conv=cv.flux, velocity=cv.velocity,
                             sound_speed=td.sound_speed, dT_previous=dT)
        npy = lambda t: t.detach().cpu().numpy()                                      # noqa: E731
        rec = {"RHOX": npy(rhox_new), "T": npy(T_std), "P": npy(P_std), "XNE": npy(new.n_e), "RHO": npy(rho),
               "ABROSS": npy(kr_std), "HEIGHT": npy(height) * 1e-5, "TAUROS": npy(s.tau), "FLXCNV": npy(corr.flux_conv),
               "PRADK": npy(pr_std + top[:, None]), "FLXERR": npy(corr.flux_error), "FLXDRV": npy(corr.flux_derivative),
               "deck_RHOX": npy(rhox_new), "deck_T": npy(T_std), "deck_P": npy(P_std), "deck_XNE": npy(new.n_e),
               "deck_ABROSS": npy(kr_std), "deck_ACCRAD": npy(acc), "deck_VTURB": npy(s.vturb),
               "deck_FLXRAD": npy(rad.flux_rad), "deck_VCONV": npy(cv.velocity), "deck_VELSND": npy(td.sound_speed),
               "PRADK0": npy(top), "max_flux_error": npy(corr.flux_error.abs().amax(-1)),
               "max_flux_derivative": npy(corr.flux_derivative.abs().amax(-1)), "max_dT": npy(dT.abs().amax(-1)),
               "dt_flux": npy(corr.dt_flux), "dt_lambda": npy(corr.dt_lambda), "dt_surface": npy(corr.dt_surface)}
        return new, rec

    # ------------------------------------------------------------------ iteration loop

    def run(self, state: StructureState, max_iterations: int = 450, on_iteration=None) -> ModelResult:
        """Iterate until every model stops (config.stop) or max_iterations; on_iteration(k, state, record,
        active_index) after each iteration. Models leave the batch when they stop."""
        import time
        t0 = time.time()
        M = len(self.batch)
        records = [[] for _ in range(M)]
        final = [None] * M
        stopped = ["max_iterations"] * M
        active = torch.arange(M)
        b, s = self.batch, state
        with torch.no_grad():
            for k in range(1, max_iterations + 1):
                in_batch = (k - 1) % 15 + 1
                s, rec = self.step(s, b, in_batch)
                for i, m in enumerate(active.tolist()):
                    records[m].append({key: v[i] for key, v in rec.items()} | {"iteration": k,
                                                                               "iteration_in_batch": in_batch})
                if on_iteration is not None:
                    on_iteration(k, s, rec, active)
                done = torch.zeros(len(active), dtype=torch.bool)
                err, der, dts = (torch.as_tensor(rec[x]) for x in ("max_flux_error", "max_flux_derivative", "max_dT"))
                bad = ~torch.isfinite(err) | ~torch.isfinite(der) | ~torch.isfinite(s.T).all(-1)
                if self.config.stop == "gold_every_15":
                    if k % 15 == 0:
                        gold = (err < 1.0) & (der < 10.0)
                        div = der > 1e5
                        for i, m in enumerate(active.tolist()):
                            if bool(gold[i]):
                                stopped[m] = "gold"
                            elif bool(div[i]):
                                stopped[m] = "diverged"
                        done = gold | div
                elif self.config.stop == "converged":
                    conv = (dts < 0.5) & (err < 0.1) & (der < 1.0)
                    for i, m in enumerate(active.tolist()):
                        if bool(conv[i]):
                            stopped[m] = "converged"
                    done = conv
                else:
                    raise ValueError("stop: 'converged' | 'gold_every_15'")
                for i, m in enumerate(active.tolist()):
                    if bool(bad[i]):
                        stopped[m] = "invalid"
                done = done | bad
                if bool(done.any()):
                    for i, m in enumerate(active.tolist()):
                        if bool(done[i]):
                            final[m] = s.select(torch.tensor([i], device=self.device))
                    keep = ~done
                    if not bool(keep.any()):
                        break
                    active = active[keep]
                    s = s.select(torch.nonzero(keep).flatten().to(self.device))
                    b = self.batch.subset(torch.isin(torch.arange(M), active))
        for i, m in enumerate(active.tolist()):
            if final[m] is None:
                final[m] = s.select(torch.tensor([i], device=self.device))
        return ModelResult(self.batch, records, final, stopped, time.time() - t0, self.quadrature)


def compute_model(specs: list, restarts: list, config: StructureConfig | None = None, **kw) -> ModelResult:
    """Convenience: ModelSpec list + RestartModel list -> ModelResult (StructureConfig() by default)."""
    from .config import ModelSpec
    config = config or StructureConfig()
    solver = StructureSolver(config, ModelSpec.stack(specs))
    return solver.run(solver.start(restarts), **kw)
