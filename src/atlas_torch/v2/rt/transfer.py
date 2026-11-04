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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .optical_depth import optical_depth

_SERIES_TERMS = 18


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
    term = ds.clone()
    m3s = term / 4.0
    for m in range(1, _SERIES_TERMS):
        term = term * (-ds) / m
        m3s = m3s + term / (m + 4)
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
        col[0] = torch.tensor([0, 1, 2]); col[-1] = torch.tensor([L - 3, L - 2, L - 1])
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
    S, J: source function and mean intensity on the depth grid (..., L); tau: optical depth (..., L)."""
    flux: torch.Tensor
    intensity: torch.Tensor | None
    S: torch.Tensor
    J: torch.Tensor | None
    tau: torch.Tensor


class Transfer:
    """Transfer(n_mu=6).solve(rhox, absorption, scattering, thermal_source) -> TransferSolution.

    n_mu: Gauss-Legendre rays on (0, 1) for J and H.
    scattering="coherent": coherent isotropic scattering (S of eq. 2.64; JOSH with IFSCAT=1);
    "absorption": scattering only extinguishes, S = S_thermal (JOSH with IFSCAT=0).
    opaque_top="formal": the surface flux of the formal solution; "diffusion": where tau_1 > 20, H(0) is the
    diffusion flux (1/3) dS/dtau of the top layer (JOSH's MAXJ=1 branch, eq. 2.76), Kurucz-compatible.
    tau_scheme: see optical_depth. method: "ali" | "direct" (see the module docstring); tol, max_iter: ALI
    convergence. chunk: wavelengths per batch (default 8192 for ALI, 512 for the dense Lambda, ~100 chunk L^2 bytes).
    """

    def __init__(self, n_mu: int = 6, *, scattering: str = "coherent", opaque_top: str = "formal",
                 tau_scheme: str = "parabolic", method: str = "ali", tol: float = 1e-11, max_iter: int = 60,
                 chunk: int | None = None):
        if scattering not in ("coherent", "absorption") or opaque_top not in ("formal", "diffusion") \
                or method not in ("ali", "direct"):
            raise ValueError("scattering: 'coherent'|'absorption'; opaque_top: 'formal'|'diffusion'; "
                             "method: 'ali'|'direct'")
        self.n_mu, self.scattering, self.opaque_top = n_mu, scattering, opaque_top
        self.tau_scheme, self.method, self.tol, self.max_iter = tau_scheme, method, tol, max_iter
        self.chunk = chunk or (8192 if method == "ali" else 512)

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
    def _formal(S, slopes, rays, wj):
        """J at the layers for source function S (..., L) by the recurrences
        I+(t_k) = T_k I+(t_k+1) + u_k (from I+(t_L) = S_L + mu dS/dtau) and I-(t_k+1) = T_k I-(t_k) + d_k
        (from I-(0) = 0)."""
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
        dn = torch.stack(dn[1:], -1)                                       # nodes 1..L
        return (wj * (up[..., 1:] + dn)).sum(0)

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

    def _ali(self, t, slopes, ab, sc, sth, mus, weights):
        """S and J by accelerated Lambda iteration (Ng acceleration every 4th iteration); rows not converged
        after max_iter are solved directly."""
        rays = self._ray_weights(t, mus)
        wj = 0.5 * torch.as_tensor(weights, dtype=t.dtype, device=t.device).reshape((-1,) + (1,) * t.dim())
        _, w00, w01, _, _, _ = rays
        up = torch.cat([w00[..., 1:], torch.ones_like(w00[..., :1])], -1)
        dn = torch.cat([w00[..., :1] + w01[..., :1], w00[..., 1:]], -1)
        lam_star = (wj * (up + dn)).sum(0)
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
            lam = self._lambda(t[bad], slopes[bad], mus, weights)
            a = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device) - alpha[bad][..., :, None] * lam
            S = S.clone()
            S[bad] = torch.linalg.solve(a, source[bad][..., None])[..., 0]
        return S, self._formal(S, slopes, rays, wj)

    @staticmethod
    def _surface_row(t, slopes, mus, weights):
        """r (..., L) with sum_rays weight * I+(0, mu) = r @ S."""
        L = t.shape[-1] - 1
        dt = t[..., 1:] - t[..., :-1]
        slab = torch.arange(L, device=t.device) == 0
        a_id = a_left = a_sl = a_sll = bottom = bottom_s = 0.0
        for mu, c in zip(mus, weights):
            w00, w01, w10, w11 = _hermite_weights(dt / mu)
            x = c * torch.exp(-t[..., :-1] / mu)
            a_id = a_id + x * w01
            a_left = a_left + x * w00
            a_sl = a_sl + torch.where(slab, 0.0, x * w11 * dt)
            a_sll = a_sll + torch.where(slab, 0.0, x * w10 * dt)
            xb = c * torch.exp(-t[..., -1] / mu)
            bottom = bottom + xb
            bottom_s = bottom_s + mu * xb
        last = torch.zeros_like(a_id).index_add(-1, slopes.col[-1], slopes.coef[..., -1, :])   # row L of D
        r = a_id + _left(a_left) + slopes.rmul((a_sl + _left(a_sll))[..., None, :])[..., 0, :] + bottom_s[..., None] * last
        return torch.cat([r[..., :-1], r[..., -1:] + bottom[..., None]], -1)

    def _lambda(self, t, slopes, mus, weights):
        """Lambda (..., L, L): J at the layers = Lambda @ S."""
        L = t.shape[-1] - 1
        dt = t[..., 1:] - t[..., :-1]
        slab = torch.arange(L, device=t.device) == 0
        dist = (t[..., 1:, None] - t[..., None, :]).abs()                   # layer i (node i+1) to node k
        a_id = a_left = a_sl = a_sll = bottom = bottom_s = 0.0
        for mu, w in zip(mus, weights):
            w00, w01, w10, w11 = _hermite_weights(dt / mu)
            w10 = torch.where(slab, 0.0, w10 * dt)[..., None, :]
            w11 = torch.where(slab, 0.0, w11 * dt)[..., None, :]
            w00, w01 = w00[..., None, :], w01[..., None, :]
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

    def _solve_chunk(self, tau, ab, sc, sth, mu_out):
        mus, weights = gauss_mu(self.n_mu, tau.dtype, tau.device)
        mus, weights = mus.tolist(), weights.tolist()
        t = torch.cat([torch.zeros_like(tau[..., :1]), tau], -1)
        slopes = Slopes.of(tau)
        J = None
        if self.scattering == "coherent" and self.method == "ali":
            S, J = self._ali(t, slopes, ab, sc, sth, mus, weights)
        elif self.scattering == "coherent":
            lam = self._lambda(t, slopes, mus, weights)
            tot = ab + sc
            a = torch.eye(tau.shape[-1], dtype=tau.dtype, device=tau.device) - (sc / tot)[..., :, None] * lam
            S = torch.linalg.solve(a, (ab / tot * sth)[..., None])[..., 0]
            J = (lam @ S[..., None])[..., 0]
        else:
            S = sth
        h_row = self._surface_row(t, slopes, mus, [0.5 * w * mu for mu, w in zip(mus, weights)])
        flux = (h_row * S).sum(-1)
        if self.opaque_top == "diffusion":
            hd = (S[..., 1] - S[..., 0]) / (3.0 * (tau[..., 1] - tau[..., 0]))
            flux = torch.where(tau[..., 0] > 20.0, hd, flux)
        inten = None
        if mu_out is not None:
            inten = torch.stack([(self._surface_row(t, slopes, [float(m)], [1.0]) * S).sum(-1) for m in mu_out], -1)
        return flux, inten, S, J

    def solve(self, rhox: torch.Tensor, absorption: torch.Tensor, scattering: torch.Tensor,
              thermal_source: torch.Tensor, *, mu=None) -> TransferSolution:
        """Emergent flux (and intensity at mu) for opacities per unit mass (cm^2 g^-1) on the rhox grid.

        rhox (..., L) broadcasts against absorption, scattering, thermal_source (..., L) (e.g. models x wavelengths
        x layers with rhox (models, 1, L)). absorption includes stimulated emission; thermal_source is the
        absorption-weighted source function (eq. 2.65; B_nu in LTE).
        """
        shape = torch.broadcast_shapes(rhox.shape, absorption.shape, scattering.shape, thermal_source.shape)
        if shape[-1] < 3:
            raise ValueError("need at least 3 layers")
        ab, sc, sth = (v.expand(shape) for v in (absorption, scattering, thermal_source))
        tau = optical_depth(rhox.expand(shape), ab + sc, scheme=self.tau_scheme)
        flat = lambda v: v.reshape(-1, shape[-1])
        tf, af, sf, bf = flat(tau), flat(ab), flat(sc), flat(sth)
        n = self.chunk
        parts = [self._solve_chunk(tf[i:i + n], af[i:i + n], sf[i:i + n], bf[i:i + n], mu)
                 for i in range(0, tf.shape[0], n)]
        cat = lambda k: None if parts[0][k] is None else torch.cat([p[k] for p in parts])
        flux, inten, S, J = (cat(k) for k in range(4))
        return TransferSolution(
            flux=flux.reshape(shape[:-1]), intensity=None if inten is None else inten.reshape(shape[:-1] + (-1,)),
            S=S.reshape(shape), J=None if J is None else J.reshape(shape), tau=tau)
