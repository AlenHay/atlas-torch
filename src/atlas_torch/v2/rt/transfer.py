"""Plane-parallel transfer with coherent isotropic scattering by the integral (Lambda-operator) method.

S = (1 - alpha) S_thermal + alpha J (Kurucz 1970 eq. 2.64) is solved as the linear system (I - alpha Lambda) S =
(1 - alpha) S_thermal (eq. 2.67) by a direct batched solve. Lambda is built on the model's own monochromatic
optical-depth grid (not the fixed 51-point tau set onto which JOSH interpolates, sec. 2.8) by integrating the
formal solution (eqs. 2.27a,b) of every Gauss-Legendre ray mu in (0, 1] analytically over each interval, with S
a cubic Hermite polynomial whose node slopes are those of the three-point parabolas (third-order accurate,
linear in S). Boundaries: no incident radiation at tau = 0; above the top layer a slab 0 < tau < tau_1 of
constant S = S_1 (as JOSH, which sets S(tau < tau_1) = S_1); below the deepest layer S continues linearly
(I+ = S + mu dS/dtau, the diffusion limit of eq. 2.76).

Nodes t = (0, tau_1, ..., tau_L); interval k = (t_k, t_k+1) carries S_k-1 -> S_k (layer indices; the slab k = 0
carries S_0 with zero slope). Seen from node i, interval k contributes exp(-|t - t_i|/mu) times the integral of
the Hermite cubic against exp(-x/mu) dx/mu over the interval (weights w00, w01 for the end values, w10, w11 for
the end slopes times the interval length).

method="ali": accelerated Lambda iteration (Olson, Auer & Buchler 1986, JQSRT 35, 431) with the local
operator Lambda* = the self-coupling of each layer through its two adjacent intervals and Ng acceleration, each
iteration one recurrence sweep per ray (O(L n_mu)); converged to max |dS/S| < tol, the fixed point is the solution
of the same discrete system (wavelengths still unconverged after max_iter are solved directly). method="direct": the dense Lambda (O(L^2 n_mu) per wavelength) and a batched linear solve.

grid="xtau" (Kurucz-compatible, JOSH): S-bar and alpha are interpolated (MAP1) onto the fixed optical depths XTAU,
the transfer is solved there with the Lambda, H and K(0) operators of the same Hermite integration built once on
that grid, and S, J - S and H are interpolated back; below XTAU = 20 the diffusion forms of eq. 2.76 with
S = (1 - alpha) S-bar + alpha J iterated (Kurucz 1970 sec. 2.8; atlas9mem.for JOSH).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .optical_depth import derivative, optical_depth

_SERIES_TERMS = 18
# M_3(d) = int_0^1 s^3 exp(-d s) d ds = d sum_m (-d)^m / (m! (m + 4)) for d < 1
_M3_SERIES = tuple((-1.0) ** m / (math.factorial(m) * (m + 4)) for m in range(_SERIES_TERMS))


# atlas9mem.for JOSH DATA XTAU8: the fixed optical depths of the pretabulated operators (Kurucz 1970 sec. 2.8 lists
# the 43-point set of ATLAS5)
XTAU = (0.0, 3.2e-6, 5.6e-6, 1e-5, 1.8e-5, 3.2e-5, 5.6e-5, 1e-4, 1.8e-4, 3.2e-4, 5.6e-4, 1e-3, 1.8e-3, 3.2e-3, 5.6e-3,
        0.01, 0.016, 0.025, 0.042, 0.065, 0.096, 0.139, 0.196, 0.273, 0.375, 0.5, 0.63, 0.78, 0.95, 1.15, 1.35, 1.6,
        1.85, 2.15, 2.45, 2.75, 3.15, 3.65, 4.25, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.5, 13.0, 14.5, 16.0, 18.0, 20.0)


def gauss_mu(n: int, dtype=torch.float64, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre nodes and weights on (0, 1) (weights sum to 1)."""
    x, w = np.polynomial.legendre.leggauss(n)
    return (torch.as_tensor(0.5 * (x + 1.0), dtype=dtype, device=device),
            torch.as_tensor(0.5 * w, dtype=dtype, device=device))


def _hermite_weights(d: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """int_0^1 h(s) exp(-d s) d ds (d >= 0) for the cubic Hermite basis h00, h01 (end values) and h10, h11 (end
    slopes times the interval length), from M_n = int_0^1 s^n exp(-d s) d ds: M_3 by its power series and
    M_n-1 = d (M_n + exp(-d)) / n for d < 1, M_n = (n/d) M_n-1 - exp(-d) for d >= 1."""
    small = d < 1.0
    ds = torch.where(small, d, torch.zeros_like(d))
    dr = torch.where(small, torch.ones_like(d), d)
    m3s = torch.full_like(ds, _M3_SERIES[-1])
    for c in _M3_SERIES[-2::-1]:                                       # Horner: M_3 = d sum_m c_m d^m
        m3s = torch.addcmul(torch.full_like(ds, c), m3s, ds)
    m3s = m3s * ds
    es = torch.exp(-ds)
    m2s = ds * (m3s + es) / 3.0
    m1s = ds * (m2s + es) / 2.0
    er = torch.exp(-dr)
    m0r = -torch.expm1(-dr)
    m1r = m0r / dr - er
    m2r = 2.0 * m1r / dr - er
    m3r = 3.0 * m2r / dr - er
    m0 = torch.where(small, -torch.expm1(-ds), m0r)
    m1, m2, m3 = (torch.where(small, a, b) for a, b in ((m1s, m1r), (m2s, m2r), (m3s, m3r)))
    return m0 - 3.0 * m2 + 2.0 * m3, 3.0 * m2 - 2.0 * m3, m1 - 2.0 * m2 + m3, m3 - m2


@dataclass(frozen=True)
class Slopes:
    """dS/dtau at node j = sum_m coef[..., j, m] S[..., col[j, m]]: slope of the parabola through nodes j-1, j, j+1
    (one-sided three-point parabolas at the ends)."""
    coef: torch.Tensor          # (..., L, 3)
    col: torch.Tensor           # (L, 3)

    @classmethod
    def of(cls, t: torch.Tensor) -> "Slopes":
        L = t.shape[-1]
        h = t[..., 1:] - t[..., :-1]
        h0, h1 = h[..., :-1], h[..., 1:]
        mid = torch.stack([-h1 / (h0 * (h0 + h1)), (h1 - h0) / (h0 * h1), h0 / (h1 * (h0 + h1))], -1)
        a, b = h[..., 0], h[..., 1]
        first = torch.stack([-(2 * a + b) / (a * (a + b)), (a + b) / (a * b), -a / (b * (a + b))], -1)
        a, b = h[..., -2], h[..., -1]
        last = torch.stack([b / (a * (a + b)), -(a + b) / (a * b), (a + 2 * b) / (b * (a + b))], -1)
        j = torch.arange(L, device=t.device)
        col = torch.stack([j - 1, j, j + 1], -1)
        col[0] = torch.tensor([0, 1, 2], device=t.device)
        col[-1] = torch.tensor([L - 3, L - 2, L - 1], device=t.device)
        return cls(torch.cat([first[..., None, :], mid, last[..., None, :]], -2), col)

    def __call__(self, S: torch.Tensor) -> torch.Tensor:
        """D @ S, (..., L)."""
        return (self.coef * S[..., self.col]).sum(-1)

    def rmul(self, m: torch.Tensor) -> torch.Tensor:
        """m @ D for m (..., R, L)."""
        out = torch.zeros_like(m)
        for k in range(3):
            out = out.index_add(-1, self.col[:, k], m * self.coef[..., None, :, k])
        return out

    def __getitem__(self, rows) -> "Slopes":
        return Slopes(self.coef[rows], self.col)


def _left(m: torch.Tensor) -> torch.Tensor:
    """Columns per interval k -> columns per layer of its left node (k-1; the slab k = 0 -> layer 0)."""
    return torch.cat([m[..., :1] + m[..., 1:2], m[..., 2:], torch.zeros_like(m[..., :1])], -1)


@dataclass(frozen=True)
class TransferSolution:
    """flux: Eddington flux H(0) (units of S); intensity: I(0, mu) at the requested mu (..., n_mu_out);
    S, J: source function and mean intensity on the depth grid (..., L); tau: optical depth (..., L).
    With moments=True: H the Eddington flux at every layer (..., L), K0 the second moment K(0) at the surface (...),
    lam_diag the diagonal of the Lambda operator J = Lambda S (..., L)."""
    flux: torch.Tensor
    intensity: torch.Tensor | None
    S: torch.Tensor
    J: torch.Tensor | None
    tau: torch.Tensor
    H: torch.Tensor | None = None
    K0: torch.Tensor | None = None
    lam_diag: torch.Tensor | None = None


class Transfer:
    """Transfer(n_mu=6).solve(rhox, absorption, scattering, thermal_source) -> TransferSolution.

    n_mu: Gauss-Legendre rays on (0, 1) for J and H.
    scattering="coherent": coherent isotropic scattering (S of eq. 2.64; JOSH with IFSCAT=1);
    "absorption": scattering only extinguishes, S = S_thermal (JOSH with IFSCAT=0).
    opaque_top="formal": the surface flux of the formal solution; "diffusion": where tau_1 > 20, H(0) is the
    diffusion flux (1/3) dS/dtau of the top layer (JOSH's MAXJ=1 branch, eq. 2.76), Kurucz-compatible.
    tau_scheme: see optical_depth. method: "ali" | "direct" (see the module docstring); tol, max_iter: ALI
    convergence. chunk: wavelengths per batch (default 2048 for ALI, 512 for the dense Lambda, ~100 chunk L^2 bytes).
    deep="formal": moments from the formal solution at every depth; "diffusion": where tau > tau_switch,
    H = (1/3) dS/dtau and J = S + dH/dtau (Kurucz 1970 eq. 2.76; atlas9mem.for JOSH below XTAU = 20), the
    derivatives by optical_depth.derivative with deep_scheme ("parabolic" | "atlas": DERIV).
    grid="model": the Lambda operator on the model's own depths; "xtau": on the fixed XTAU depths (see the module
    docstring; method, deep, tau_switch and opaque_top do not apply, the diffusion forms hold below XTAU = 20).
    """

    def __init__(self, n_mu: int = 6, *, scattering: str = "coherent", opaque_top: str = "formal",
                 tau_scheme: str = "parabolic", method: str = "ali", tol: float = 1e-11, max_iter: int = 60,
                 chunk: int | None = None, deep: str = "formal", tau_switch: float = 20.0,
                 deep_scheme: str = "parabolic", grid: str = "model"):
        if scattering not in ("coherent", "absorption") or opaque_top not in ("formal", "diffusion") \
                or method not in ("ali", "direct") or deep not in ("formal", "diffusion") \
                or grid not in ("model", "xtau"):
            raise ValueError("scattering: 'coherent'|'absorption'; opaque_top, deep: 'formal'|'diffusion'; "
                             "method: 'ali'|'direct'; grid: 'model'|'xtau'")
        self.grid = grid
        self._xtau_ops = {}
        self.n_mu, self.scattering, self.opaque_top = n_mu, scattering, opaque_top
        self.deep, self.tau_switch, self.deep_scheme = deep, tau_switch, deep_scheme
        self.tau_scheme, self.method, self.tol, self.max_iter = tau_scheme, method, tol, max_iter
        self.chunk = chunk or (2048 if method == "ali" else 512)

    @staticmethod
    def _ray_weights(t, mus):
        """Per ray (R, ..., L): interval transmissions exp(-dtau/mu) and the Hermite weights (slab slopes zeroed)."""
        L = t.shape[-1] - 1
        dt = t[..., 1:] - t[..., :-1]
        mu = torch.as_tensor(mus, dtype=t.dtype, device=t.device).reshape((-1,) + (1,) * dt.dim())
        w00, w01, w10, w11 = _hermite_weights(dt / mu)
        slab = torch.arange(L, device=t.device) == 0
        w10 = torch.where(slab, 0.0, w10 * dt)
        w11 = torch.where(slab, 0.0, w11 * dt)
        return torch.exp(-dt / mu), w00, w01, w10, w11, mu

    @staticmethod
    def _sweep(S, slopes, rays):
        """I+ at the nodes 0..L and I- at the nodes 1..L per ray, for source function S (..., L), by the
        recurrences I+(t_k) = T_k I+(t_k+1) + u_k (from I+(t_L) = S_L + mu dS/dtau) and
        I-(t_k+1) = T_k I-(t_k) + d_k (from I-(0) = 0)."""
        T, w00, w01, w10, w11, mu = rays
        sp = slopes(S)
        sl = torch.cat([S[..., :1], S[..., :-1]], -1)
        spl = torch.cat([sp[..., :1], sp[..., :-1]], -1)
        u = w00 * sl + w01 * S + w10 * spl + w11 * sp
        d = w00 * S + w01 * sl - w10 * sp - w11 * spl
        L = S.shape[-1]
        up = [S[..., -1] + mu[..., 0] * sp[..., -1]]
        for k in range(L - 1, -1, -1):
            up.append(T[..., k] * up[-1] + u[..., k])
        up = torch.stack(up[::-1], -1)                                     # nodes 0..L
        dn = [torch.zeros_like(up[..., 0])]
        for k in range(L):
            dn.append(T[..., k] * dn[-1] + d[..., k])
        return up, torch.stack(dn[1:], -1)                                 # nodes 1..L

    @classmethod
    def _formal(cls, S, slopes, rays, wj):
        """J = sum_rays w (I+ + I-)/2 at the layers."""
        up, dn = cls._sweep(S, slopes, rays)
        return (wj * (up[..., 1:] + dn)).sum(0)

    @classmethod
    def _moments(cls, S, slopes, rays, wj):
        """J, H at the layers and K at tau = 0 from one sweep: J = sum w I/2, H = sum w mu (I+ - I-)/2,
        K(0) = sum w mu^2 I+(0)/2 (Kurucz 1970 eqs. 2.28-2.30)."""
        T, w00, w01, w10, w11, mu = rays
        up, dn = cls._sweep(S, slopes, rays)
        J = (wj * (up[..., 1:] + dn)).sum(0)
        H = (wj * mu * (up[..., 1:] - dn)).sum(0)
        K0 = (wj[..., 0] * mu[..., 0] ** 2 * up[..., 0]).sum(0)
        return J, H, K0

    def _deep_moments(self, tau, S, J, H):
        """Diffusion moments below tau_switch (eq. 2.76): H = (1/3) dS/dtau, J = S + dH/dtau."""
        deep = tau > self.tau_switch
        if not bool(deep.any()):
            return J, H
        hd = derivative(tau, S, self.deep_scheme) / 3.0
        H = torch.where(deep, hd, H)
        J = torch.where(deep, S + derivative(tau, H, self.deep_scheme), J)
        return J, H

    @staticmethod
    def _ng(hist):
        """Ng (1974) acceleration of order 2 from four successive iterates, weights 1/S^2 (Olson, Auer & Buchler
        1986, sec. 3); rows with a singular 2x2 system keep the last iterate."""
        s0, s1, s2, s3 = hist[-1], hist[-2], hist[-3], hist[-4]
        d0, d1, d2 = s0 - s1, s1 - s2, s2 - s3
        w = 1.0 / s0 ** 2
        dot = lambda a, b: (w * a * b).sum(-1)
        a11, a12, a22 = dot(d0 - d1, d0 - d1), dot(d0 - d1, d0 - d2), dot(d0 - d2, d0 - d2)
        b1, b2 = dot(d0, d0 - d1), dot(d0, d0 - d2)
        det = a11 * a22 - a12 * a12
        ok = det.abs() > 1e-14 * (a11 * a22).abs()
        det = torch.where(ok, det, torch.ones_like(det))
        a = torch.where(ok, (b1 * a22 - b2 * a12) / det, 0.0)[..., None]
        b = torch.where(ok, (b2 * a11 - b1 * a12) / det, 0.0)[..., None]
        return (1.0 - a - b) * s0 + a * s1 + b * s2

    def _ali(self, t, slopes, ab, sc, sth, rays, weights):
        """S by accelerated Lambda iteration (Ng acceleration every 4th iteration); rows not converged
        after max_iter are solved directly."""
        wj = 0.5 * torch.as_tensor(weights, dtype=t.dtype, device=t.device).reshape((-1,) + (1,) * t.dim())
        lam_star = self._lambda_star(rays, wj)
        tot = ab + sc
        alpha = sc / tot
        source = ab / tot * sth
        S, hist = sth, []
        for it in range(self.max_iter):
            J = self._formal(S, slopes, rays, wj)
            dS = (source + alpha * J - S) / (1.0 - alpha * lam_star)
            S = S + dS
            err = (dS.abs() / S.abs()).amax(-1)
            if float(err.max()) < self.tol:
                break
            hist = (hist + [S])[-4:]
            if len(hist) == 4 and it % 4 == 3:
                S = self._ng(hist)
                hist = []
        else:
            bad = err >= self.tol
            lam = self._lambda(t[bad], slopes[bad], tuple(r[:, bad] for r in rays[:5]) + (rays[5],), weights)
            a = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device) - alpha[bad][..., :, None] * lam
            S = S.clone()
            S[bad] = torch.linalg.solve(a, source[bad][..., None])[..., 0]
        return S

    @staticmethod
    def _surface_row(t, slopes, rays, coeffs):
        """r (..., L) with sum_rays coeffs * I+(0, mu) = r @ S (rays: _ray_weights(t, mus))."""
        _, w00, w01, w10, w11, mu = rays
        c = torch.as_tensor(coeffs, dtype=t.dtype, device=t.device).reshape(mu.shape)
        x = c * torch.exp(-t[..., :-1] / mu)
        a_id, a_left = (x * w01).sum(0), (x * w00).sum(0)
        a_sl, a_sll = (x * w11).sum(0), (x * w10).sum(0)
        xb = c[..., 0] * torch.exp(-t[..., -1] / mu[..., 0])
        bottom, bottom_s = xb.sum(0), (mu[..., 0] * xb).sum(0)
        last = torch.zeros_like(a_id).index_add(-1, slopes.col[-1], slopes.coef[..., -1, :])   # row L of D
        r = a_id + _left(a_left) + slopes.rmul((a_sl + _left(a_sll))[..., None, :])[..., 0, :] + bottom_s[..., None] * last
        return torch.cat([r[..., :-1], r[..., -1:] + bottom[..., None]], -1)

    def _lambda(self, t, slopes, rays, weights):
        """Lambda (..., L, L): J at the layers = Lambda @ S (rays: _ray_weights(t, mus))."""
        dist = (t[..., 1:, None] - t[..., None, :]).abs()                   # layer i (node i+1) to node k
        a_id = a_left = a_sl = a_sll = bottom = bottom_s = 0.0
        _, W00, W01, W10, W11, MU = rays
        for r, w in enumerate(weights):
            mu = float(MU.reshape(-1)[r])
            w00, w01, w10, w11 = (a[r][..., None, :] for a in (W00, W01, W10, W11))
            x = torch.exp(-dist / mu)
            up = torch.triu(x[..., :-1], 1)                                 # interval k below node i+1 (k >= i+1)
            dn = torch.tril(x[..., 1:])                                     # interval k above it (k + 1 <= i + 1)
            c = 0.5 * w
            a_id = a_id + c * (up * w01 + dn * w00)
            a_left = a_left + c * (up * w00 + dn * w01)
            a_sl = a_sl + c * (up * w11 - dn * w10)
            a_sll = a_sll + c * (up * w10 - dn * w11)
            bottom = bottom + c * x[..., -1]
            bottom_s = bottom_s + c * mu * x[..., -1]
        last = torch.zeros_like(bottom).index_add(-1, slopes.col[-1], slopes.coef[..., -1, :])
        lam = a_id + _left(a_left) + slopes.rmul(a_sl + _left(a_sll)) + bottom_s[..., :, None] * last[..., None, :]
        return torch.cat([lam[..., :-1], lam[..., -1:] + bottom[..., :, None]], -1)

    @staticmethod
    def _lambda_star(rays, wj):
        """The local operator of _ali: the self-coupling of each layer through its two adjacent intervals."""
        _, w00, w01, _, _, _ = rays
        up = torch.cat([w00[..., 1:], torch.ones_like(w00[..., :1])], -1)
        dn = torch.cat([w00[..., :1] + w01[..., :1], w00[..., 1:]], -1)
        return (wj * (up + dn)).sum(0)

    def _solve_chunk(self, tau, ab, sc, sth, mu_out, moments=False):
        mus, weights = gauss_mu(self.n_mu, tau.dtype, tau.device)
        mus, weights = mus.tolist(), weights.tolist()
        t = torch.cat([torch.zeros_like(tau[..., :1]), tau], -1)
        slopes = Slopes.of(tau)
        rays = self._ray_weights(t, mus)
        wj = 0.5 * torch.as_tensor(weights, dtype=t.dtype, device=t.device).reshape((-1,) + (1,) * t.dim())
        J = H = K0 = lam_diag = None
        if self.scattering == "coherent" and self.method == "ali":
            S = self._ali(t, slopes, ab, sc, sth, rays, weights)
        elif self.scattering == "coherent":
            lam = self._lambda(t, slopes, rays, weights)
            tot = ab + sc
            a = torch.eye(tau.shape[-1], dtype=tau.dtype, device=tau.device) - (sc / tot)[..., :, None] * lam
            S = torch.linalg.solve(a, (ab / tot * sth)[..., None])[..., 0]
            J = (lam @ S[..., None])[..., 0]
            if moments:
                lam_diag = torch.diagonal(lam, dim1=-2, dim2=-1)
        else:
            S = sth
        if moments:                                    # one formal solution gives J, H and K(0)
            J, H, K0 = self._moments(S, slopes, rays, wj)
        elif J is None and self.scattering == "coherent":
            J = self._formal(S, slopes, rays, wj)
        h_row = self._surface_row(t, slopes, rays, [0.5 * w * mu for mu, w in zip(mus, weights)])
        flux = (h_row * S).sum(-1)
        top_opaque = tau[..., 0] > 20.0
        if self.opaque_top == "diffusion":
            hd = (S[..., 1] - S[..., 0]) / (3.0 * (tau[..., 1] - tau[..., 0]))
            flux = torch.where(top_opaque, hd, flux)
        inten = None
        if mu_out is not None:
            inten = torch.stack([(self._surface_row(t, slopes, self._ray_weights(t, [float(m)]), [1.0]) * S).sum(-1)
                                 for m in mu_out], -1)
        if moments:
            if self.deep == "diffusion":
                J, H = self._deep_moments(tau, S, J, H)
            if self.opaque_top == "diffusion":
                K0 = torch.where(top_opaque, J[..., 0] / 3.0, K0)
            if lam_diag is None:
                lam_diag = self._lambda_star(rays, wj)
        return flux, inten, S, J, H, K0, lam_diag

    def _xtau_operators(self, dtype, device):
        """(XTAU (X,), Lambda (X, X), the H operator (X, X), the K(0) row (X,)) on the fixed depths: J = Lambda S,
        H = Lambda_H S, K(0) = k0 . S from one formal solution of each unit source vector."""
        key = (dtype, str(device))
        if key not in self._xtau_ops:
            xt = torch.as_tensor(XTAU, dtype=dtype, device=device)
            mus, weights = gauss_mu(self.n_mu, dtype, device)
            t = torch.cat([xt[:1], xt])[None]                                # zero-thickness slab above XTAU_1 = 0
            rays = self._ray_weights(t, mus.tolist())
            wj = 0.5 * weights.reshape(-1, 1, 1)
            J, H, K0 = self._moments(torch.eye(len(XTAU), dtype=dtype, device=device), Slopes.of(xt), rays, wj)
            self._xtau_ops[key] = (xt, J.T, H.T, K0)
        return self._xtau_ops[key]

    def _solve_xtau(self, tau, ab, sc, sth, mu_out, moments):
        """grid="xtau" for rows (N, L): S-bar, alpha -> XTAU (MAP1; the layer-1 values above tau_1), (I - alpha
        Lambda) S = (1 - alpha) S-bar there, S, J - S, H back (MAP1) at tau <= XTAU_last; below, eq. 2.76 with S
        iterated (to tol, at most L sweeps); all layers deeper than XTAU_last: diffusion with K(0) = J(0)/3."""
        from ..opacity.continuum.interpolate import map1
        xt, lam, lam_h, k0 = self._xtau_operators(tau.dtype, tau.device)
        tot = ab + sc
        alpha = sc / tot if self.scattering == "coherent" else torch.zeros_like(sc)
        x = xt.expand(tau.shape[0], -1)
        above = x < tau[:, :1]
        a_x = torch.where(above, alpha[:, :1], map1(x, tau, alpha)).clamp(min=0.0)
        s_x = torch.where(above, sth[:, :1], map1(x, tau, sth)).clamp(min=0.0)
        eye = torch.eye(len(XTAU), dtype=tau.dtype, device=tau.device)
        S_x = torch.linalg.solve(eye - a_x[:, :, None] * lam, ((1.0 - a_x) * s_x)[..., None])[..., 0]
        back = lambda f: map1(tau, x, f)                                       # noqa: E731
        deep = tau > xt[-1]
        S = torch.where(deep, sth, back(S_x))
        jms_x = S_x @ lam.T - S_x
        h_up = back(S_x @ lam_h.T)
        for _ in range(tau.shape[-1]):
            h_d = derivative(tau, S, self.deep_scheme) / 3.0
            jms_d = derivative(tau, h_d, self.deep_scheme)
            new = torch.where(deep, (1.0 - alpha) * sth + alpha * (S + jms_d), S)
            done = float(((new - S).abs() / new.abs().clamp(min=1e-300)).max()) < self.tol
            S = new
            if done:
                break
        h_d = derivative(tau, S, self.deep_scheme) / 3.0
        jms_d = derivative(tau, h_d, self.deep_scheme)
        J = torch.where(deep, S + jms_d, S + back(jms_x))
        H = torch.where(deep, h_d, h_up)
        thick = deep[:, 0]
        flux = torch.where(thick, H[:, 0], S_x @ lam_h[0])
        K0 = torch.where(thick, J[:, 0] / 3.0, S_x @ k0)
        inten = None
        if mu_out is not None:
            t = torch.cat([xt[:1], xt])
            rows = [self._surface_row(t, Slopes.of(xt), self._ray_weights(t, [float(m)]), [1.0]) for m in mu_out]
            inten = torch.stack([S_x @ r for r in rows], -1)
        lam_diag = torch.where(deep, torch.ones_like(S), back(torch.diagonal(lam).expand_as(x))) if moments else None
        if not moments:
            H = K0 = None
        return flux, inten, S, J, H, K0, lam_diag

    def solve(self, rhox: torch.Tensor, absorption: torch.Tensor, scattering: torch.Tensor,
              thermal_source: torch.Tensor, *, mu=None, moments: bool = False) -> TransferSolution:
        """Emergent flux (and intensity at mu) for opacities per unit mass (cm^2 g^-1) on the rhox grid.

        rhox (..., L) broadcasts against absorption, scattering, thermal_source (..., L) (e.g. models x wavelengths
        x layers with rhox (models, 1, L)). absorption includes stimulated emission; thermal_source is the
        absorption-weighted source function (eq. 2.65; B_nu in LTE). moments: also H at every layer, K(0) and
        the Lambda diagonal (diag(Lambda) for method="direct", the ALI local operator for "ali").
        """
        shape = torch.broadcast_shapes(rhox.shape, absorption.shape, scattering.shape, thermal_source.shape)
        if shape[-1] < 3:
            raise ValueError("need at least 3 layers")
        ab, sc, sth = (v.expand(shape) for v in (absorption, scattering, thermal_source))
        tau = optical_depth(rhox.expand(shape), ab + sc, scheme=self.tau_scheme)
        flat = lambda v: v.reshape(-1, shape[-1])
        tf, af, sf, bf = flat(tau), flat(ab), flat(sc), flat(sth)
        n = self.chunk
        one = self._solve_xtau if self.grid == "xtau" else self._solve_chunk
        parts = [one(tf[i:i + n], af[i:i + n], sf[i:i + n], bf[i:i + n], mu, moments) for i in range(0, tf.shape[0], n)]
        cat = lambda k: None if parts[0][k] is None else torch.cat([p[k] for p in parts])
        flux, inten, S, J, H, K0, lam_diag = (cat(k) for k in range(7))
        opt = lambda v, sh: None if v is None else v.reshape(sh)         # noqa: E731
        return TransferSolution(
            flux=flux.reshape(shape[:-1]), intensity=None if inten is None else inten.reshape(shape[:-1] + (-1,)),
            S=S.reshape(shape), J=opt(J, shape), tau=tau, H=opt(H, shape), K0=opt(K0, shape[:-1]),
            lam_diag=opt(lam_diag, shape))
