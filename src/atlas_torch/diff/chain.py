"""d(spectrum)/d(Teff, log g, [M/H]) through the converged ATLAS9 model and SYNTHE.

    from atlas_torch.diff.chain import ChainDiff
    ch = ChainDiff("runs/my_atlas_run", (500., 510.))     # ATLAS run dir (atlas_control*.com, output_summary.out,
                                                          #  output_synthe.out from one BasicATLAS synthe() call)
    ch.prepare()                                          # x* + IFT tangents (~240 ATLAS9 iterations, cached)
    d = ch.spectrum_derivatives()                         # {"teff"|"logg"|"mh": {"flux","cont","norm"} (LENGTH,)}

Composition: dS/dtheta = dS/dx . dx*/dtheta (+ dS/d[M/H]|_x for [M/H], since the SYNTHE deck carries the same
ABUNDANCE SCALE), with dx*/dtheta from atlas_torch.diff.atlas9_ift (implicit function theorem on the ATLAS9
fixed point, Picard linear response) and dS/dx . v from atlas_torch.diff.spectrum (one forward-mode JVP per parameter, the tangent v
being the structure tangent of all layers at once). The spectrum is evaluated at the full-precision fixed point
x* (T, P, RHOX, XNE of every layer), not at the 4-9 digit punched deck. Teff and log g enter SYNTHE only through
the structure (the deck's TEFF/GRAVITY cards are labels for SYNTHE with PRESSURE OFF).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .atlas9_ift import NAMES, THETA, Atlas9FixedPoint
from .spectrum import SyntheDiff

D = torch.float64


class ChainDiff:
    def __init__(self, model_dir, window, cache_dir=None, synthe_kw: dict | None = None, polish: bool = False):
        self.dir = Path(model_dir)
        self.window = tuple(window)
        if cache_dir is None:                     # never write into the run directory (it may be a reference run)
            import os
            ext = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
            cache_dir = ext / "scratch" / "m8" / "cache" / self.dir.name
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.fp = Atlas9FixedPoint(self.dir)
        self.synthe_kw = dict(synthe_kw or {})
        self.polish = polish
        self.J = None
        self.xstar = None
        self.dx = None
        self.sd = None

    # ------------------------------------------------------------------------------- ATLAS9 part
    def prepare(self, warmup: int = 12, method: str = "picard", n_picard: int = 40, log=None):
        """x* (warm-up iterations of G from the punched deck, in full precision), dG/dtheta and dx*/dtheta.

        method="picard" (default): the linear response of ATLAS9's iteration, t <- J t + dG/dtheta with
        Jacobian-vector products by directional central differences of one iteration (2*n_picard iterations
        per parameter). method="jfree": GMRES on (I - dG/dx) t = dG/dtheta with the same products (fewer
        iterations; accurate for the dominant P/RHOX responses, not for small T responses: the operator is
        only piecewise linear). method="fd_jacobian": the full
        per-column FD Jacobian (~290 iterations) and a dense solve -- NOT reliable for ATLAS9 (G is only
        piecewise smooth at its fixed point: TCORR's MAP1 onto the standard tau grid is evaluated at its own
        nodes there, so per-column derivatives do not combine linearly; see RESULTS); kept for reference.
        Everything is cached in cache_dir."""
        from .atlas9_ift import tangents_jfree, theta_derivative
        fp = self.fp
        c = self.cache
        xs_path = c / "diff_atlas9_x0.npy"
        if xs_path.exists():
            self.xstar = np.load(xs_path)
        else:
            self.xstar, _ = fp.converge(fp.x_deck, n=warmup, log=log)
            np.save(xs_path, self.xstar)
        x0 = self.xstar
        gt_path = c / "diff_atlas9_Gtheta.npy"
        if gt_path.exists():
            self.Gt = np.load(gt_path)
        else:
            self.Gt = theta_derivative(fp, x0)
            np.save(gt_path, self.Gt)
        dx_path = c / f"diff_atlas9_tangents_{method}.npy"
        if dx_path.exists():
            self.dx = np.load(dx_path)
        elif method == "picard":
            from .atlas9_ift import tangents_picard
            self.dx = np.stack([tangents_picard(fp, x0, self.Gt[:, k], n=n_picard, log=log)[0] for k in range(3)], 1)
            np.save(dx_path, self.dx)
        elif method == "jfree":
            self.dx, self.info = tangents_jfree(fp, x0, self.Gt, log=log)
            np.save(dx_path, self.dx)
        else:
            steps = fp.default_steps(x0, {"RHOX": 2e-4, "T": 2e-4})
            self.J = fp.jacobian(x0, steps=steps, theta_steps=(2.0, 2e-3, 2e-3),
                                 skip=("P", "XNE", "ABROSS", "ACCRAD", "PRADK0"), cache=c / "diff_atlas9_jacobian.npz",
                                 log=log)
            self.dx = fp.tangents(self.J)
            np.save(dx_path, self.dx)
        return self

    def structure_tangents(self) -> dict:
        """{theta: {"RHOX","T","P","XNE","ABROSS","ACCRAD": (n,)}}: dx*/dtheta per quantity."""
        n = self.fp.n
        return {th: {nm: self.dx[i * n:(i + 1) * n, k] for i, nm in enumerate(NAMES)} for k, th in enumerate(THETA)}

    def reconverge(self, theta, n: int = 40, log=None) -> np.ndarray:
        """Re-converged model at theta: n plain iterations of G from x* (ATLAS9's own iteration, no gold stop)."""
        from .atlas9_ift import reconverge_iterating
        p = self.cache / ("diff_atlas9_reconv_" + "_".join(f"{v:.6g}" for v in theta) + f"_{n}.npy")
        if p.exists():
            return np.load(p)
        x, _ = reconverge_iterating(self.fp, self.xstar, np.asarray(theta, dtype=np.float64), n=n, log=log)
        np.save(p, x)
        return x

    # ------------------------------------------------------------------------------- SYNTHE part
    def synthe(self) -> SyntheDiff:
        if self.sd is None:
            self.sd = SyntheDiff(self.dir, *self.window, polish=self.polish, **self.synthe_kw)
        return self.sd

    def structure_params(self, x: np.ndarray) -> dict:
        n = self.fp.n
        d = {nm: torch.as_tensor(x[i * n:(i + 1) * n].copy(), dtype=D) for i, nm in enumerate(NAMES)}
        return {k: d[k] for k in ("T", "P", "XNE", "RHOX")}

    def spectrum_at(self, x: np.ndarray, mh: float = 0.0) -> dict:
        sd = self.synthe()
        p = self.structure_params(x)
        if mh:
            p["mh"] = torch.tensor(mh, dtype=D)
        with torch.no_grad():
            o = sd(p)
        return {k: o[k] for k in ("flux", "cont", "norm")}

    def spectrum_derivatives(self, which=THETA) -> dict:
        sd = self.synthe()
        tg = self.structure_tangents()
        p = self.structure_params(self.xstar)
        out = {}
        for th in which:
            t = {k: torch.as_tensor(tg[th][k], dtype=D) for k in ("T", "P", "XNE", "RHOX")}
            if th == "mh":
                t["mh"] = 1.0
            r = sd.jvp(p, t)
            out[th] = {k: r["d" + k] for k in ("flux", "cont", "norm")}
            out.setdefault("_primal", {k: r[k] for k in ("flux", "cont", "norm")})
        return out
