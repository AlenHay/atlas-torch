"""Implicit differentiation of a converged ATLAS9 model: d(structure)/d(Teff, log g, [M/H]).

Fixed-point formulation. An ATLAS9 iteration maps the model it starts from to the next model. We use the map
BasicATLAS itself iterates across restarts: x = the restart deck contents (READ DECK6 / READ PUNCH)
    x = (RHOX, T, P, XNE, ABROSS, ACCRAD)(J=1..NRHOX) and PRADK0       (6*NRHOX + 1 numbers)
and G(x; theta) = the same quantities after one iteration of the port's ATLAS9 (atlas_torch.atlas9.main.
iteration) started from x with parameters theta = (Teff, log g, [M/H] = log10 ABUNDANCE SCALE). Every internal
quantity not in the deck is rebuilt from x as READ PUNCH + BEGIN do (TAUROS, PRAD, PRADK, TK..., XNATOM, RHO,
PTURB, XABUND, WTMOLE, FLUX); x is carried in full float64 precision (the punched deck has only 4-9 digits, so
the map is evaluated on in-memory States: READ PUNCH reads the base deck, then x replaces its columns).
VTURB, the ODF, the opacity switches and the abundance pattern are fixed.

A converged model is (to ATLAS9's own tolerance) a fixed point x* = G(x*; theta), so
    dx*/dtheta = (I - dG/dx)^-1 dG/dtheta.
G is the exact-path ATLAS9 (glibc libm emulation, numpy round trips, REAL*4 tables, the ODF sampling control
flow), which autograd cannot traverse; dG/dtheta and the products dG/dx . v come from central differences of
one iteration. Three solvers (RESULTS, M8):
  * `jacobian` + `tangents`: dense per-column FD Jacobian (~290 iterations) and a direct solve. NOT usable for
    ATLAS9: G is only piecewise smooth at its fixed point (TCORR re-maps onto the standard tau grid with MAP1,
    whose nodes coincide with the model's tau points there), so column derivatives do not combine linearly, and
    I - dG/dx is ill-conditioned (slow modes at the convective boundary, eigenvalues 0.997-0.998).
  * `tangents_jfree`: GMRES with directional derivatives (few products; good for the dominant P/RHOX
    responses, not for small T responses).
  * `tangents_picard` (used by atlas_torch.diff.chain): the linearised ATLAS9 iteration t <- J t + dG/dtheta
    with directional derivatives of fixed amplitude; robust, converges at ATLAS9's own rate.
`reconverge_iterating` gives the validation reference (re-converged models by iterating G).

    fp = Atlas9FixedPoint("runs/stock/atlas/sun")        # an ATLAS run directory (BasicATLAS layout)
    x0, _ = fp.converge(fp.x_deck, n=12)                  # the fixed point in full precision
    gt = theta_derivative(fp, x0)                         # dG/dtheta (433, 3)
    t, _ = tangents_picard(fp, x0, gt[:, 0])              # dx*/dTeff (433,)
"""
from __future__ import annotations

import contextlib
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

COLS = ("rhox.rhox", "temp.t", "state.p", "state.xne", "abross.abross", "rad.accrad")
NAMES = ("RHOX", "T", "P", "XNE", "ABROSS", "ACCRAD")
THETA = ("teff", "logg", "mh")


def _control_cards(model_dir: Path) -> tuple[str, dict]:
    """The control cards of an ATLAS run (atlas_control.com, up to the first ITERATIONS) and its unit files."""
    ctl = (model_dir / "atlas_control.com").read_text()
    body = ctl.split('<<"EOF"', 1)[1].split("\n", 1)[1]
    head = body.split("ITERATIONS", 1)[0]
    files = {}
    st = (model_dir / "atlas_control_start.com").read_text()
    for m in re.finditer(r"ln -s\s+(\S+)\s+fort\.(\d+)", st):
        files[int(m.group(2))] = Path(m.group(1))
    return head, files


@contextlib.contextmanager
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


@dataclass
class Jac:
    x0: np.ndarray          # (N,) fixed point
    theta0: np.ndarray      # (3,)
    g0: np.ndarray          # G(x0)
    Gx: np.ndarray          # (N, N)
    Gt: np.ndarray          # (N, 3)
    steps: np.ndarray       # (N,) absolute FD steps of x
    tsteps: np.ndarray      # (3,)
    seconds: float


class Atlas9FixedPoint:
    def __init__(self, model_dir, deck=None, physics="torch", odf=None):
        from ..atlas9.program import make_physics
        self.dir = Path(model_dir)
        self.cards, files = _control_cards(self.dir)
        deck = Path(deck) if deck is not None else self.dir / "output_summary.out"
        files[3] = deck
        if odf is not None:
            files[9] = Path(odf) / files[9].name
            files[1] = Path(odf) / "kappa.ros"
        self.files = files
        self.physics = make_physics(physics)
        m = re.search(r"SCALE\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", self.cards)
        sc = re.search(r"ABUNDANCE SCALE\s+(\S+)", self.cards)
        self.theta0 = np.array([float(m.group(4)), float(m.group(5)), math.log10(float(sc.group(1)))])
        with _float64():
            self.s0 = self._readin()
        self.n = self.s0.iscalar("rhox.nrhox")
        self.x_deck = self.x_of(self.s0)
        self.N = len(self.x_deck)
        self.evals = 0
        self.eval_seconds = 0.0

    # ------------------------------------------------------------------------------- READIN once
    def _readin(self):
        from ..atlas9 import linop
        from ..atlas9.cards import StdinReader
        from ..atlas9.commons import initial_state
        from ..atlas9.readin import IO, readin
        cards = self.cards + "ITERATIONS 1 PRINT 1\nPUNCH 0\nBEGIN                    ITERATION  10 COMPLETED\n"
        io = IO(stdin=StdinReader(cards), files=self.files)
        s = initial_state()
        s["temp.itemp"] = torch.tensor([0])
        s = readin(s, io, 1)
        self.odf = linop.read_odf(str(self.files[9]))
        return s

    # ------------------------------------------------------------------------------- x <-> State
    def x_of(self, s) -> np.ndarray:
        n = s.iscalar("rhox.nrhox")
        cols = [s[k][:n].detach().cpu().numpy().astype(np.float64) for k in COLS]
        return np.concatenate(cols + [np.array([s.scalar("pzero.pradk0")])])

    def split(self, x: np.ndarray) -> dict:
        n = self.n
        d = {nm: x[i * n:(i + 1) * n] for i, nm in enumerate(NAMES)}
        d["PRADK0"] = x[6 * n]
        return d

    def state_of(self, x: np.ndarray, theta) -> object:
        """The State an iteration starts from: base READIN State with x and theta substituted and the
        derived quantities of READ PUNCH / SCALE / BEGIN recomputed."""
        from ..atlas9 import fmath
        from ..atlas9.numerics import integ
        from ..atlas9.readin import derived_temperature, flux_of
        from ..fnum import f32
        s = self.s0.updated({})
        n = self.n
        upd = {}
        for i, k in enumerate(COLS):
            a = s[k].clone()
            a[:n] = torch.as_tensor(x[i * n:(i + 1) * n], dtype=torch.float64)
            upd[k] = a
        s = s.updated(upd)
        rx, ab, acc = s["rhox.rhox"], s["abross.abross"], s["rad.accrad"]
        pradk0 = float(x[6 * n])
        prad = s["rad.prad"].clone(); prad[:n] = integ(rx[:n], acc[:n], acc[0] * rx[0])
        pradk = s["pzero.pradk"].clone(); pradk[:n] = prad[:n] + pradk0
        tauros = s["abross.tauros"].clone(); tauros[:n] = integ(rx[:n], ab[:n], ab[0] * rx[0])
        teff, logg, mh = (float(v) for v in theta)
        grav = 10.0 ** logg
        xscale = 10.0 ** mh
        abund = s["elem.abund"]
        xabund = s["xabund.xabund"].clone()
        for iz in range(2, 99):
            xabund[iz] = 10.0 ** float(abund[iz]) * xscale
        atmass = s["elem.atmass"]
        wtmole = float((xabund[:99] * atmass[:99]).sum())
        s = s.updated({"rad.prad": prad, "pzero.pradk": pradk, "abross.tauros": tauros,
                       "pzero.pradk0": torch.tensor([pradk0]),
                       "pzero.pzero": torch.tensor([s.scalar("pzero.pcon") + pradk0 + s.scalar("pzero.pturb0")]),
                       "teff.teff": torch.tensor([teff]), "flux.flux": torch.tensor([flux_of(teff)]),
                       "teff.grav": torch.tensor([grav]), "teff.glog": torch.tensor([logg]),
                       "junk.xscale": torch.tensor([xscale]), "xabund.xabund": xabund,
                       "xabund.wtmole": torch.tensor([wtmole])})
        s = s.updated(derived_temperature(s))
        tk = s["temp.tk"]
        xnatom = s["state.xnatom"].clone(); rho = s["state.rho"].clone()
        xnatom[:n] = s["state.p"][:n] / tk[:n] - s["state.xne"][:n]
        rho[:n] = xnatom[:n] * wtmole * 1.660e-24
        upd = {"state.xnatom": xnatom, "state.rho": rho}
        if s.iscalar("turbpr.ifturb") > 0:
            pt = s["turbpr.pturb"].clone()
            v = s["turbpr.vturb"][:n]
            pt[:n] = f32(.5) * rho[:n] * (v * v)
            upd["turbpr.pturb"] = pt
        return s.updated(upd)

    # ------------------------------------------------------------------------------- the map
    def G(self, x: np.ndarray, theta=None, return_state: bool = False):
        from ..atlas9.main import iteration
        theta = self.theta0 if theta is None else theta
        t0 = time.perf_counter()
        with _float64():
            s = self.state_of(x, theta)
            s = s.updated({"iter.iter": torch.tensor([1]), "temp.itemp": s["temp.itemp"] + 1})
            s = iteration(s, self.physics, self.odf, [], [])
        self.evals += 1
        self.eval_seconds += time.perf_counter() - t0
        out = self.x_of(s)
        return (out, s) if return_state else out

    def converge(self, x, theta=None, n: int = 30, tol: float = 0.0, log=None):
        """Iterate x <- G(x) n times (or until max |dx/x| < tol). Returns (x, history of max |dx/x|)."""
        hist = []
        for k in range(n):
            xn = self.G(x, theta)
            r = float(np.max(np.abs(xn - x) / np.maximum(np.abs(x), 1e-300)))
            hist.append(r)
            if log:
                log(f"iteration {k + 1}: max|dx/x| = {r:.3e}")
            x = xn
            if r < tol:
                break
        return x, hist

    # ------------------------------------------------------------------------------- Jacobians
    def default_steps(self, x: np.ndarray, rel: dict | None = None) -> np.ndarray:
        rel = {"RHOX": 1e-4, "T": 1e-4, "P": 1e-4, "XNE": 1e-3, "ABROSS": 1e-3, "ACCRAD": 1e-3,
               "PRADK0": 1e-3, **(rel or {})}
        n = self.n
        steps = np.empty_like(x)
        d = self.split(x)
        for i, nm in enumerate(NAMES):
            v = np.abs(d[nm])
            if nm == "ACCRAD":
                v = np.maximum(v, 1e-3 * np.max(v))
            steps[i * n:(i + 1) * n] = rel[nm] * v
        steps[6 * n] = rel["PRADK0"] * max(abs(d["PRADK0"]), 1e-30)
        return steps

    def probe(self, x, theta=None, rel=1e-3) -> dict:
        """Influence of each input column (all layers perturbed together) on G: {name: max|dG/G|}."""
        theta = self.theta0 if theta is None else theta
        g0 = self.G(x, theta)
        out = {}
        n = self.n
        for i, nm in enumerate(NAMES + ("PRADK0",)):
            xp = x.copy()
            sl = slice(i * n, (i + 1) * n) if nm != "PRADK0" else slice(6 * n, 6 * n + 1)
            xp[sl] = xp[sl] * (1.0 + rel)
            g = self.G(xp, theta)
            out[nm] = float(np.max(np.abs(g - g0) / np.maximum(np.abs(g0), 1e-300)))
        return out

    def jacobian(self, x, theta=None, steps=None, theta_steps=(5.0, 0.005, 0.005), skip=(), cache=None,
                 log=None, part=None) -> Jac:
        """Central finite-difference dG/dx and dG/dtheta at (x, theta). skip: input names whose columns are
        known to have no (or negligible, see probe()) influence: their columns are left 0. cache: .npz path
        (loaded if it matches x, saved otherwise). part=(k, m): compute only every m-th of the remaining
        columns starting at k (and dG/dtheta only for k=0) -- for parallel workers, merged by merge_parts()."""
        theta = np.asarray(self.theta0 if theta is None else theta, dtype=np.float64)
        if cache is not None and Path(cache).exists():
            z = np.load(cache)
            if z["x0"].shape == x.shape and np.array_equal(z["x0"], x) and np.array_equal(z["theta0"], theta):
                return Jac(**{k: z[k] for k in Jac.__dataclass_fields__ if k != "seconds"}, seconds=float(z["seconds"]))
        t0 = time.perf_counter()
        steps = self.default_steps(x) if steps is None else steps
        n = self.n
        N = len(x)
        g0 = self.G(x, theta)
        Gx = np.zeros((N, N))
        names = [nm for nm in NAMES for _ in range(n)] + ["PRADK0"]
        todo = [i for i in range(N) if names[i] not in skip]
        if part is not None:
            todo = todo[part[0]::part[1]]
        ck = Path(str(cache) + ".partial.npz") if cache is not None else None
        done = set()
        if ck is not None and ck.exists():                  # resume an interrupted run (same point and steps)
            z = np.load(ck)
            if np.array_equal(z["x0"], x) and np.array_equal(z["steps"], steps):
                Gx = z["Gx"].copy()
                done = set(int(i) for i in z["done"])
        for i in todo:
            if i in done:
                continue
            xp, xm = x.copy(), x.copy()
            xp[i] += steps[i]
            xm[i] -= steps[i]
            Gx[:, i] = (self.G(xp, theta) - self.G(xm, theta)) / (2.0 * steps[i])
            done.add(i)
            if ck is not None and len(done) % 8 == 0:
                np.savez(ck, x0=x, steps=steps, Gx=Gx, done=np.array(sorted(done)))
            if log and (i % 24 == 0):
                log(f"dG/dx column {i + 1}/{N} ({names[i]}), {time.perf_counter() - t0:.0f} s")
        ts = np.asarray(theta_steps, dtype=np.float64)
        Gt = np.zeros((N, 3))
        for k in (range(3) if part is None or part[0] == 0 else ()):
            tp, tm = theta.copy(), theta.copy()
            tp[k] += ts[k]
            tm[k] -= ts[k]
            Gt[:, k] = (self.G(x, tp) - self.G(x, tm)) / (2.0 * ts[k])
        J = Jac(x, theta, g0, Gx, Gt, steps, ts, time.perf_counter() - t0)
        if cache is not None:
            np.savez(cache, **{k: getattr(J, k) for k in Jac.__dataclass_fields__})
            if ck is not None and ck.exists():
                ck.unlink()
        return J

    @staticmethod
    def merge_parts(paths, out=None) -> Jac:
        """Sum the column subsets of jacobian(part=...) runs (each .npz holds zeros elsewhere)."""
        zs = [np.load(p) for p in paths]
        for z in zs[1:]:
            if not np.array_equal(z["x0"], zs[0]["x0"]):
                raise ValueError("parts computed at different points")
        J = Jac(zs[0]["x0"], zs[0]["theta0"], zs[0]["g0"], sum(z["Gx"] for z in zs), sum(z["Gt"] for z in zs),
                zs[0]["steps"], zs[0]["tsteps"], float(max(z["seconds"] for z in zs)))
        if out is not None:
            np.savez(out, **{k: getattr(J, k) for k in Jac.__dataclass_fields__})
        return J

    @staticmethod
    def tangents(J: Jac) -> np.ndarray:
        """dx*/dtheta = (I - dG/dx)^-1 dG/dtheta, solved in the scaled variables x/|x| (N, 3)."""
        s = np.maximum(np.abs(J.x0), 1e-300)
        A = np.eye(len(s)) - (J.Gx * s[None, :]) / s[:, None]
        b = J.Gt / s[:, None]
        return np.linalg.solve(A, b) * s[:, None]

    def structure(self, x: np.ndarray) -> dict:
        return self.split(x)


def chord_newton(fp: Atlas9FixedPoint, J: Jac, x, theta, n: int = 30, tol: float = 1e-10, log=None,
                 vars_=("RHOX", "T")):
    """Fixed point of G(.; theta) by the chord method with the Jacobian J (x <- x + (I - dG/dx)^-1 (G(x) - x)).
    The converged point satisfies G(x) = x to `tol` (max relative change of the RHOX and T columns); the
    Jacobian only sets the convergence rate. Returns (x, history)."""
    s = np.maximum(np.abs(J.x0), 1e-300)
    A = np.eye(len(s)) - (J.Gx * s[None, :]) / s[:, None]
    lu = np.linalg.inv(A)
    nn = fp.n
    idx = np.concatenate([np.arange(NAMES.index(v) * nn, (NAMES.index(v) + 1) * nn) for v in vars_])
    hist = []
    for k in range(n):
        g = fp.G(x, theta)
        r = (g - x) / s
        res = float(np.max(np.abs(r[idx])))
        hist.append(res)
        if log:
            log(f"chord {k}: max|G(x)-x|/|x| (RHOX,T) = {res:.3e}")
        if res < tol:
            break
        x = x + (lu @ r) * s
    return x, hist


# ------------------------------------------------------------------------------------- Jacobian-free IFT

def theta_derivative(fp: Atlas9FixedPoint, x, theta=None, steps=(2.0, 2e-3, 2e-3)) -> np.ndarray:
    """dG/dtheta (N, 3) by central differences of one iteration (Teff [K], log g, [M/H] [dex])."""
    theta = np.asarray(fp.theta0 if theta is None else theta, dtype=np.float64)
    out = np.zeros((len(x), 3))
    for k in range(3):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += steps[k]
        tm[k] -= steps[k]
        out[:, k] = (fp.G(x, tp) - fp.G(x, tm)) / (2.0 * steps[k])
    return out


def tangents_jfree(fp: Atlas9FixedPoint, x, Gt, theta=None, amplitude: float = 1.0, inputs=("RHOX", "T"),
                   maxiter: int = 40, rtol: float = 1e-3, log=None) -> tuple[np.ndarray, dict]:
    """Solve (I - dG/dx) t = dG/dtheta for each column of Gt without forming dG/dx: GMRES on the operator
    v -> v - J v, J v = (G(x + a v) - G(x - a v)) / (2 a), the step a scaled so that a*v moves T by
    ~`amplitude` K. Only the deck quantities that influence G (`inputs`) are perturbed; the others follow
    as t = J t + Gt. Returns (t (N, 3), info)."""
    from scipy.sparse.linalg import LinearOperator, gmres
    theta = fp.theta0 if theta is None else theta
    n = fp.n
    idx = np.concatenate([np.arange(NAMES.index(v) * n, (NAMES.index(v) + 1) * n) for v in inputs])
    g0 = None
    info = {"matvecs": [], "residual": []}
    out = np.zeros((len(x), 3))
    for k in range(Gt.shape[1]):
        count = [0]

        def Jfull(v_red):
            v = np.zeros(len(x))
            v[idx] = v_red
            tmax = np.max(np.abs(v[n:2 * n])) if "T" in inputs else np.max(np.abs(v_red / x[idx]))
            a = amplitude / max(tmax, 1e-30) if "T" in inputs else 1e-4 / max(tmax, 1e-30)
            count[0] += 1
            return (fp.G(x + a * v, theta) - fp.G(x - a * v, theta)) / (2.0 * a)

        b = Gt[idx, k]
        sc = np.abs(x[idx])
        A = LinearOperator((len(idx), len(idx)), dtype=np.float64,
                           matvec=lambda w: (w * sc - Jfull(w * sc)[idx]) / sc)
        sol, st = gmres(A, b / sc, rtol=rtol, restart=maxiter, maxiter=1,
                        callback=(lambda r: log(f"  gmres theta[{k}] residual {r:.2e}") if log else None),
                        callback_type="pr_norm")
        t_red = sol * sc
        t = np.zeros(len(x))
        t[idx] = t_red
        full = Jfull(t_red) + Gt[:, k] if np.any(t_red) else Gt[:, k].copy()
        full[idx] = t_red
        out[:, k] = full
        info["matvecs"].append(count[0])
        info["residual"].append(float(np.linalg.norm(A.matvec(sol) - b / sc) / np.linalg.norm(b / sc)))
    return out, info


def reconverge_iterating(fp: Atlas9FixedPoint, x, theta, n: int = 40, log=None) -> tuple[np.ndarray, list]:
    """Fully re-converged model at theta by plain iteration of G from x (what ATLAS9 itself does, without the
    gold-tolerance stop). Returns (x, max |dx/x| of the RHOX and T columns per iteration)."""
    hist = []
    nn = fp.n
    for k in range(n):
        xn = fp.G(x, theta)
        hist.append(float(np.max(np.abs(xn[:2 * nn] - x[:2 * nn]) / np.abs(x[:2 * nn]))))
        x = xn
        if log and k % 10 == 9:
            log(f"  iteration {k + 1}: max|dx/x| = {hist[-1]:.2e}")
    return x, hist


def tangents_picard(fp: Atlas9FixedPoint, x, gt: np.ndarray, theta=None, n: int = 40, amplitude: float = 1.0,
                    inputs=("RHOX", "T"), log=None) -> tuple[np.ndarray, list]:
    """dx*/dtheta for one parameter by the Picard iteration of the linearised map, t <- J t + dG/dtheta, with
    J t = (G(x + a t) - G(x - a t)) / 2a and a fixed a (T moved by ~`amplitude` K at the first step). This is
    the linear response of ATLAS9's own iteration: robust to the kinks of G (unlike a dense FD Jacobian or a
    Krylov solver) and converging at ATLAS9's own rate. Returns (t (N,), max relative update per step)."""
    theta = fp.theta0 if theta is None else theta
    nn = fp.n
    idx = np.concatenate([np.arange(NAMES.index(v) * nn, (NAMES.index(v) + 1) * nn) for v in inputs])
    t = gt.copy()
    a = amplitude / max(np.max(np.abs(gt[nn:2 * nn])), 1e-30)
    hist = []
    for k in range(n):
        v = np.zeros_like(t)
        v[idx] = t[idx]
        tn = (fp.G(x + a * v, theta) - fp.G(x - a * v, theta)) / (2.0 * a) + gt
        d = float(np.max(np.abs(tn[idx] - t[idx]) / np.maximum(np.abs(tn[idx]), 1e-30 + np.abs(tn[idx]).max() * 1e-6)))
        hist.append(d)
        t = tn
        if log and k % 10 == 9:
            log(f"  picard {k + 1}: max rel update {d:.2e}")
    return t, hist
