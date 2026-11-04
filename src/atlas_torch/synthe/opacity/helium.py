"""Helium continuous opacities of atlas7v.for: HE1OP, HE2OP, HEMIOP, HERAOP (no SAVEd state)."""
from __future__ import annotations

import torch

from ...state import State
from ._core import Ctx, acc, chain, exp, f32, put_rows, rdiv, sqrt
from .hydrogen import coulff_k

# HE1OP levels: (threshold, X kind, weight g, level energy (None: no Boltzmann factor), BHE1 column).
# X kinds: ("f3", d) X=FREQ3/d;  ("e", a, b) X=EXP(a-b*FREQLG);  ("e2s",) the 2S 3S fit;  ("e1s",) 1S 1S.
_HE1 = [
    (4368.190, ("f3", 3125.0), 3.0, 193942.57, 29), (4388.260, ("f3", 3125.0), 9.0, 193922.5, 28),
    (4388.260, ("f3", 3125.0), 27.0, 193922.5, 27), (4389.390, ("f3", 3125.0), 7.0, 193921.37, 26),
    (4389.450, ("f3", 3125.0), 15.0, 193921.31, 25), (4392.369, ("f3", 3125.0), 5.0, 193918.391, 24),
    (4393.515, ("f3", 3125.0), 15.0, 193917.245, 23), (4509.980, ("f3", 3125.0), 9.0, 193800.78, 22),
    (4647.133, ("f3", 3125.0), 1.0, 193663.627, 21), (4963.671, ("f3", 3125.0), 3.0, 193347.089, 20),
    (6817.943, ("f3", 1024.0), 3.0, 191492.817, 19), (6858.680, ("f3", 1024.0), 7.0, 191452.08, 18),
    (6858.960, ("f3", 1024.0), 21.0, 191451.80, 17), (6864.201, ("f3", 1024.0), 5.0, 191446.559, 16),
    (6866.172, ("f3", 1024.0), 15.0, 191444.588, 15), (7093.620, ("f3", 1024.0), 9.0, 191217.14, 14),
    (7370.429, ("f3", 1024.0), 1.0, 190940.331, 13), (8012.550, ("f3", 1024.0), 3.0, 190298.210, 12),
    (12101.289, ("e", 58.81, 2.89), 3.0, 186209.471, 11), (12205.695, ("e", 85.20, 3.69), 5.0, 186105.065, 10),
    (12209.106, ("e", 85.20, 3.69), 15.0, 186101.654, 9), (12746.066, ("e", 49.30, 2.60), 9.0, 185564.694, 8),
    (13445.824, ("e", 23.85, 1.86), 1.0, 184864.936, 7), (15073.868, ("e", 12.69, 1.54), 3.0, 183236.892, 6),
    (27175.760, ("e", 81.35, 3.5), 3.0, 171135.000, 5), (29223.753, ("e", 61.21, 2.9), 9.0, 169087.007, 4),
    (32033.214, ("e", 26.83, 1.91), 1.0, 166277.546, 3), (38454.691, ("e2s",), 3.0, 159856.069, 2),
    (198310.760, ("e1s",), 1.0, None, 1),
]


def he1op_k(c: Ctx):
    """SUBROUTINE HE1OP: He I bound-free (29 levels + n>=6 continuum) and free-free -> (AHE1, SHE1)."""
    freq, lg, stim, bnu, ehvkt = c.freq, c.freqlg, c.stim, c.bnu, c.ehvkt
    hckt = c.lay("temp.hckt")
    bhe2 = c.lay("bhe.bhe2", 1)
    freq3 = rdiv(f32(2.815e29), freq) / freq / freq
    h = freq3 * 4.0 / 2.0 / (109722.267 * hckt) * (
        exp(-torch.clamp(198310.76 - c.waveno, min=195262.919) * hckt) - exp(-198310.76 * hckt)) * stim * bhe2
    s = h * bnu
    for m, (_, xk, g, e, lvl) in zip(chain(c.waveno, [t[0] for t in _HE1]), _HE1):
        if not bool(m.any()):
            break
        if xk[0] == "f3":
            x = freq3 / xk[1]
        elif xk[0] == "e":
            x = exp(xk[1] - xk[2] * lg)
        elif xk[0] == "e2s":
            x = exp(-390.026 + (21.035 - 0.318 * lg) * lg)
        else:
            x = exp(33.32 - 2.0 * lg)
        b1 = c.lay("bhe.bhe1", lvl)
        if e is None:
            a = x * g * 1.0 * (b1 - bhe2 * ehvkt)
        else:
            a = x * g * exp(-e * hckt) * (b1 - bhe2 * ehvkt)
        h = acc(m, h, a)
        s = acc(m, s, a * bnu * stim / (b1 / bhe2 - ehvkt))
    rho = c.lay("state.rho")
    xnfphe1 = c.lay("bhe.xnfphe", 1)
    h = h * xnfphe1 / rho
    s = s * xnfphe1 / rho
    a = (rdiv(3.619e8, sqrt(c.lay("temp.t"))) * coulff_k(c, 1) / freq * c.lay("state.xne") / freq
         * c.lay("bhe.xnfhe", 2) / freq * stim / rho)
    h = h + a
    s = s + a * bnu
    she1 = torch.where(h > 0.0, s / h, bnu.expand_as(h))
    return h, she1


def he1op(s: State) -> dict:
    c = Ctx.from_state(s)
    a, sh = he1op_k(c)
    return {"bhe.ahe1": put_rows(s["bhe.ahe1"], a, c.n), "bhe.she1": put_rows(s["bhe.she1"], sh, c.n)}


# HE2OP levels n = 9..1: (threshold, kind, g, energy, BHE2 column)
_HE2 = [
    (5418.390, "n9", 162.0, 433490.46, None), (6857.660, ("hi", 32768.0), 128.0, 432051.19, None),
    (8956.950, ("hi", 16807.0), 98.0, 429951.90, None),
    (12191.437, ("lo", 7776.0, 1.0986, -2.704e13, 1.229e27), 72.0, 426717.413, 6),
    (17555.715, ("lo", 3125.0, 1.102, -3.909e13, 2.371e27), 50.0, 421353.135, 5),
    (27430.925, ("lo", 1024.0, 1.101, -5.765e13, 4.593e27), 32.0, 411477.925, 4),
    (48766.491, ("lo", 243.0, 1.101, -9.863e13, 1.035e28), 18.0, 390142.359, 3),
    (109726.529, ("lo", 32.0, 1.105, -2.375e14, 4.077e28), 8.0, 329182.321, 2),
    (438908.850, ("lo", 1.0, f32(0.9916), 2.719e13, -2.268e30), 2.0, None, 1),
]


def he2op_k(c: Ctx):
    """SUBROUTINE HE2OP: He II bound-free (n=1..9 + n>=10) and free-free -> (AHE2, SHE2)."""
    freq, stim, bnu, ehvkt = c.freq, c.stim, c.bnu, c.ehvkt
    hckt = c.lay("temp.hckt")
    freq3 = rdiv(2.815e29, freq) / freq / freq
    xnfprho = c.lay("bhe.xnfphe", 2) / c.lay("state.rho")
    h = freq3 * 16.0 * 2.0 / 2.0 / (438889.068 * hckt) * (
        exp(-torch.clamp(438908.85 - c.waveno, min=434519.959) * hckt) - exp(-438908.85 * hckt)) * stim * xnfprho
    s = h * bnu
    for m, (_, kind, g, e, lvl) in zip(chain(c.waveno, [t[0] for t in _HE2]), _HE2):
        if not bool(m.any()):
            break
        if kind == "n9":
            x = freq3 / 59049.0 * 16.0
        elif kind[0] == "hi":
            x = freq3 * 16.0 / kind[1]
        else:
            _, d, a0, a1, a2 = kind
            x = freq3 * 16.0 / d * (a0 + (a1 + rdiv(a2, freq)) / freq)
        if lvl is None:
            a = x * g * exp(-e * hckt) * stim * xnfprho
            ds = a * bnu
        else:
            b = c.lay("bhe.bhe2", lvl)
            a = (x * g * 1.0 if e is None else x * g * exp(-e * hckt)) * (b - ehvkt) * xnfprho
            ds = a * bnu * stim / (b - ehvkt)
        h = acc(m, h, a)
        s = acc(m, s, ds)
    a = (rdiv(3.6919e8 * 4.0, sqrt(c.lay("temp.t"))) * coulff_k(c, 2) / freq * c.lay("state.xne") / freq
         * c.lay("bhe.xnfphe", 3) / freq * stim / c.lay("state.rho"))
    h = h + a
    s = s + a * bnu
    return h, torch.where(h > 0.0, s / h, bnu.expand_as(h))


def he2op(s: State) -> dict:
    c = Ctx.from_state(s)
    a, sh = he2op_k(c)
    return {"bhe.ahe2": put_rows(s["bhe.ahe2"], a, c.n), "bhe.she2": put_rows(s["bhe.she2"], sh, c.n)}


def hemiop_k(c: Ctx) -> torch.Tensor:
    """SUBROUTINE HEMIOP: He- free-free -> AHEMIN."""
    f = c.freq
    a = 3.397e-01 + (-5.216e14 + rdiv(7.039e30, f)) / f
    b = -4.116e03 + (1.067e19 + rdiv(8.135e34, f)) / f
    cc = 5.081e08 + (-8.724e22 - rdiv(5.659e37, f)) / f
    t = c.lay("temp.t")
    return ((a * t + b + cc / t) / 1.0e15 * c.lay("state.xne") / 1.0e15 * c.lay("bhe.xnfphe", 1) / 1.0e15
            / c.lay("state.rho"))


def hemiop(s: State) -> dict:
    c = Ctx.from_state(s)
    return {"bhe.ahemin": put_rows(s["bhe.ahemin"], hemiop_k(c), c.n)}


def heraop_k(c: Ctx) -> torch.Tensor:
    """SUBROUTINE HERAOP: He I Rayleigh scattering -> SIGHE."""
    wave = rdiv(2.99792458e18, torch.clamp(c.freq, max=5.15e15))
    ww = wave * wave
    t = 1.0 + (2.44e5 + rdiv(5.94e10, ww - 2.90e5)) / ww
    sig = rdiv(5.484e-14, ww) / ww * (t * t)
    return sig * c.lay("bhe.xnfphe", 1) / c.lay("state.rho") * c.lay("bhe.bhe1", 1)


def heraop(s: State) -> dict:
    c = Ctx.from_state(s)
    return {"bhe.sighe": put_rows(s["bhe.sighe"], heraop_k(c), c.n)}
