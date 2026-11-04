"""Voigt function H(a, v) = Re w(v + i a), normalised H(0, 0) = 1 (Kurucz & Avrett 1981 p. 4-5 convention:
the line profile is H(a, v) / (sqrt(pi) Delta nu_D))."""
from __future__ import annotations

import math

import torch

SQRT_PI = math.sqrt(math.pi)


def _w1(t):
    return t * 0.5641896 / (0.5 + t * t)


def _w2(t):
    u = t * t
    return t * (1.410474 + u * 0.5641896) / (0.75 + u * (3.0 + u))


def _w3(t):
    return ((16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + t * 0.5642236))))
            / (16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))))


def _w4(t):
    u = t * t
    num = t * (36183.31 - u * (3321.9905 - u * (1540.787 - u * (219.0313 - u * (35.76683 - u * (1.320522
                                                                                          - u * 0.56419))))))
    den = 32066.6 - u * (24322.84 - u * (9022.228 - u * (2186.181 - u * (364.2191 - u * (61.57037
                                                                                         - u * (1.841439 - u))))))
    return torch.exp(u) - num / den


def faddeeva(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """w(x + i y) for y >= 0, complex; Humlicek 1982, JQSRT 27, 437 (algorithm W4, |error| < 1e-4 relative).
    Each region is evaluated only on its own points, so gradients stay finite."""
    x, y = torch.broadcast_tensors(x, y)
    ax = x.abs()
    s = ax + y
    r1 = s >= 15.0
    r2 = (s >= 5.5) & ~r1
    r3 = (s < 5.5) & (y >= 0.195 * ax - 0.176)
    r4 = (s < 5.5) & ~r3
    out = torch.zeros(x.shape, dtype=torch.complex128 if x.dtype == torch.float64 else torch.complex64,
                      device=x.device)
    for m, f in ((r1, _w1), (r2, _w2), (r3, _w3), (r4, _w4)):
        if bool(m.any()):
            out = out.index_put((m.nonzero(as_tuple=True)), f(torch.complex(y[m], -x[m])))
    return out


def _lorentz_region(x, y):
    """Re w of Humlicek's region 1 (|x| + y >= 15) in real arithmetic."""
    return 0.5641896 * y * (0.5 + x * x + y * y) / ((0.5 + y * y - x * x) ** 2 + 4.0 * x * x * y * y)


def _masked(shape_like, parts):
    """Tensor assembled from (mask, function) pairs, each evaluated on its own points only."""
    out = torch.zeros_like(shape_like)
    for m, f in parts:
        if bool(m.any()):
            out = out.index_put(m.nonzero(as_tuple=True), f(m))
    return out


def voigt(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """H(a, v) = (a/pi) int exp(-t^2)/((v - t)^2 + a^2) dt (exact to the accuracy of W4)."""
    a, v = torch.broadcast_tensors(a, v)
    x = v.abs()
    far = x + a >= 15.0
    return _masked(x, ((far, lambda m: _lorentz_region(x[m], a[m])),
                       (~far, lambda m: faddeeva(x[m], a[m]).real)))


def _synthe_asymptotic(a, v):
    """synthe.for FUNCTION VOIGT, a > 1.4 or a + v > 3.2: asymptotic series of the Lorentz wing."""
    aa, vv = a * a, v * v
    u = (aa + vv) * 1.4142
    lor = a * 0.79788 / u
    asym = ((((aa / u - 10.0 * vv / u) * aa / u * 3.0 + 15.0 * (vv / u) ** 2) + 3.0 * vv - aa) / (u * u) + 1.0) * lor
    return torch.where(a > 100.0, lor, asym)


def _synthe_series(a, v, h0, h1):
    """synthe.for FUNCTION VOIGT, 0.2 <= a <= 1.4, a + v <= 3.2: expansion in a about the H0, H1, H2 tables
    with an empirical correction factor."""
    vv = v * v
    h2 = h0 - 2.0 * vv * h0
    hh1 = h1 + h0 * 1.12838
    hh2 = h2 + hh1 * 1.12838 - h0
    hh3 = (1.0 - h2) * 0.37613 - hh1 * 0.66667 * vv + hh2 * 1.12838
    hh4 = (3.0 * hh3 - hh1) * 0.37613 + h0 * 0.66667 * vv * vv
    return ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0) * (
        ((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032)


def voigt_kurucz(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """SYNTHE's profile (synthe.for main line loop and FUNCTION VOIGT; Kurucz & Avrett 1981 p. 5), to attribute
    differences: within 10 Doppler widths H0(v) + a H1(v) for a < 0.2 (H1 = (2/sqrt(pi)) (2 v F(v) - 1), F the
    Dawson function = sqrt(pi)/2 Im w(v)) and the fast approximation of VOIGT for a >= 0.2 (errors up to 13 %
    near a = 1.4, v = 1.2); beyond, the value at v = 10 continued as v^-2 (low by 100/(100 + a^2) for large a)."""
    a, v = torch.broadcast_tensors(a, v)
    x = v.abs()
    near = x <= 10.0

    def core(m):
        xm, am = x[m], a[m]
        h0 = torch.exp(-xm * xm)
        h1 = (2.0 / SQRT_PI) * (xm * SQRT_PI * faddeeva(xm, torch.zeros_like(xm)).imag - 1.0)
        series = (am >= 0.2) & (am <= 1.4) & (am + xm <= 3.2)
        asym = (am >= 0.2) & ~series
        return _masked(xm, ((am < 0.2, lambda k: h0[k] + am[k] * h1[k]),
                            (series, lambda k: _synthe_series(am[k], xm[k], h0[k], h1[k])),
                            (asym, lambda k: _synthe_asymptotic(am[k], xm[k]))))

    ten = torch.tensor([10.0], dtype=torch.float64)
    h1_10 = (2.0 / SQRT_PI) * (10.0 * SQRT_PI * float(faddeeva(ten, torch.zeros_like(ten)).imag[0]) - 1.0)

    def wing(m):
        am, xm = a[m], x[m]
        h10 = torch.where(am < 0.2, am * h1_10, _synthe_asymptotic(am, torch.full_like(xm, 10.0)))
        return h10 * (10.0 / xm) ** 2

    return _masked(x, ((near, core), (~near, wing)))
