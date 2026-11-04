"""Bound-free and free-free absorption by metals (Kurucz 1970 secs. 5.9-5.11; atlas7v.for COOLOP, LUKEOP, HOTOP and
their callees). Per unit volume, cm^-1, (..., L, F).

Level data (edge, g, excitation energy, cross-section) as in the SYNTHE-variant routines C1OP (Luo & Pradhan 1989,
Burke & Taylor 1979 fits), MG1OP, AL1OP, SI1OP (Nahar & Pradhan resonance fits), FE1OP, N1OP, O1OP, MG2OP, SI2OP,
CA2OP (Peach, Henry) and HOTOP (Henry, Hidalgo, Peach, Flower)."""
from __future__ import annotations

import torch

from .data import fortran_data
from .hydrogenic import gaunt_free_free
from .interpolate import linear_rows
from .state import C_CM, Conditions, Levels, stack


def _levels(rows, sigma, unit="cm-1", edge_unit="cm-1") -> Levels:
    """rows: (edge, g, energy); edge in cm^-1 (or Hz)."""
    f = C_CM if edge_unit == "cm-1" else 1.0
    return Levels(tuple(r[1] for r in rows), tuple(r[2] for r in rows), tuple(r[0] * f for r in rows), sigma, unit)


def _pl(a, w0, p):
    """a (w0/w)^p as a function of nu."""
    return lambda nu: a * (w0 * C_CM / nu) ** p


def _res(w, w0, width, a, b):
    """Resonance fit (a eps + b) / (eps^2 + 1), eps = 2 (w - w0)/width."""
    eps = (w - w0) * 2.0 / width
    return (a * eps + b) / (eps * eps + 1.0)


# ------------------------------------------------------------------------------------------- low-temperature

def c1_bound_free(c: Conditions) -> torch.Tensor:
    """C I (C1OP): 2p2 3P, 1D, 1S (Luo & Pradhan 1989 with resonances), each split over the C II 2P1/2 (1/3) and
    2P3/2 (2/3) limits, and the excited 2p3s / 2p3p terms."""
    ryd = 109732.298

    def ground(w, w0):
        return 10.0 ** (-16.80 - (w - w0) / 3.0 / ryd)

    def sigma(nu):
        w = nu / C_CM
        s1s = (ground(w, 69172.400) + _res(w, 97700.0, 2743.0, 68.0e-18, 118.0e-18)) / 3.0
        s1d = (ground(w, 80627.760) + _res(w, 93917.0, 9230.0, 22.0e-18, 26.0e-18)
               + _res(w, 111130.0, 2743.0, -10.5e-18, 46.0e-18)) / 3.0
        s3p = ground(w, 90777.000) / 3.0
        return stack(_pl(2.1e-18, 22006.370, 1.5)(nu), _pl(1.54e-18, 28880.880, 1.2)(nu),
                     _pl(0.2e-18, 30489.700, 1.2)(nu), s1s, 2.0 * s1s, _pl(16.0e-18, 69767.350, 3)(nu), s1d, 2.0 * s1d,
                     s3p, s3p, s3p, 2.0 * s3p, 2.0 * s3p, 2.0 * s3p, _pl(1.0e-18, 100121.000, 3)(nu))

    rows = ((22006.370, 3.0, 68856.33), (28880.880, 3.0, 61981.82), (30489.700, 9.0, 60373.00),
            (69172.400, 1.0, 21648.02), (69235.820, 1.0, 21648.02), (69767.350, 15.0, 64088.85),
            (80627.760, 5.0, 10192.66), (80691.180, 5.0, 10192.66),
            (90777.000, 5.0, 43.42), (90804.000, 3.0, 16.42), (90820.420, 1.0, 0.0),
            (90840.420, 5.0, 43.42), (90867.420, 3.0, 16.42), (90883.840, 1.0, 0.0), (100121.000, 5.0, 33735.20))
    return c.n_over_u(6, 1) * _levels(rows, sigma).weighted_sum(c) * c.stim


def mg1_bound_free(c: Conditions) -> torch.Tensor:
    """Mg I (MG1OP): power-law fits to Peach's cross-sections; the 3p 3P term with its three J levels."""
    def sigma(nu):
        r3p = 39759.842 * C_CM / nu
        s3p = torch.maximum(20.0e-18 * r3p ** 2.7, 40.0e-18 * r3p ** 14)
        r = 26619.756 * C_CM / nu
        return stack(_pl(25.0e-18, 13713.986, 2.7)(nu), _pl(33.8e-18, 13823.223, 2.8)(nu),
                     _pl(45.0e-18, 15267.955, 2.7)(nu), _pl(0.43e-18, 18167.687, 2.6)(nu),
                     _pl(2.1e-18, 20473.617, 2.6)(nu), 16.0e-18 * r ** 2.1 - 7.8e-18 * r ** 9.5, s3p, s3p, s3p,
                     _pl(1.1e-18, 61671.020, 10)(nu))

    rows = ((13713.986, 15.0, 47957.034), (13823.223, 9.0, 47847.797), (15267.955, 5.0, 46403.065),
            (18167.687, 1.0, 43503.333), (20473.617, 3.0, 41197.043), (26619.756, 3.0, 35051.264),
            (39759.842, 5.0, 21911.178), (39800.556, 3.0, 21870.464), (39820.615, 1.0, 21850.405),
            (61671.020, 1.0, 0.0))
    return c.n_over_u(12, 1) * _levels(rows, sigma).weighted_sum(c) * c.stim


def al1_bound_free(c: Conditions) -> torch.Tensor:
    """Al I (AL1OP): ground 3p 2P (measured threshold cross-section, Kurucz 1970 p. 84) and excited terms."""
    def sigma(nu):
        s2p = _pl(65.0e-18, 48166.309, 5)(nu)
        return stack(_pl(50.0e-18, 8002.467, 3)(nu), _pl(50.0e-18, 9346.231, 3)(nu), _pl(56.7e-18, 10588.957, 1.9)(nu),
                     _pl(14.5e-18, 15318.007, 1)(nu), _pl(47.0e-18, 15842.129, 1.83)(nu),
                     _pl(10.0e-18, 22930.614, 2)(nu), s2p, s2p, _pl(10.0e-18, 55903.260, 2)(nu))

    rows = ((8002.467, 6.0, 40275.903), (9346.231, 10.0, 38932.139), (10588.957, 2.0, 37689.413),
            (15318.007, 6.0, 32960.363), (15842.129, 10.0, 32436.241), (22930.614, 2.0, 25347.756),
            (48166.309, 4.0, 112.061), (48278.370, 2.0, 0.0), (55903.260, 12.0, 29097.11))
    return c.n_over_u(13, 1) * _levels(rows, sigma).weighted_sum(c) * c.stim


def si1_bound_free(c: Conditions) -> torch.Tensor:
    """Si I (SI1OP): 3p2 3P, 1D, 1S with Nahar & Pradhan resonances, each split over the Si II 2P1/2, 2P3/2 limits,
    and excited terms."""
    def sigma(nu):
        w = nu / C_CM
        res1s = _res(w, 70000.0, 6500.0, 97.0e-18, 94.0e-18)
        res1d = _res(w, 78600.0, 13000.0, -10.0e-18, 77.0e-18)
        r3p = 65524.393 / w
        s3p = torch.where(w > 74000.0, 93.0e-18 * r3p ** 4, 72.0e-18 * r3p ** 1.90) / 3.0
        return stack(_pl(18.0e-18, 17777.641, 3)(nu), _pl(4.09e-18, 24947.216, 2)(nu),
                     _pl(1.25e-18, 26079.180, 2)(nu),
                     (_pl(37.0e-18, 50353.180, 2.40)(nu) + res1s) / 3.0,
                     (_pl(37.0e-18, 50640.630, 2.40)(nu) + res1s) * 2.0 / 3.0,
                     (_pl(24.5e-18, 59448.700, 1.85)(nu) + res1d) / 3.0,
                     (_pl(24.5e-18, 59736.150, 1.85)(nu) + res1d) * 2.0 / 3.0,
                     _pl(18.0e-18, 63446.510, 3)(nu), s3p, s3p, s3p, 2.0 * s3p, 2.0 * s3p, 2.0 * s3p,
                     _pl(15.0e-18, 75423.767, 3)(nu))

    rows = ((17777.641, 15.0, 48161.459), (24947.216, 3.0, 40991.884), (26079.180, 9.0, 39859.920),
            (50353.180, 1.0, 15394.370), (50640.630, 1.0, 15394.370), (59448.700, 5.0, 6298.850),
            (59736.150, 5.0, 6298.850), (63446.510, 15.0, 45303.310),
            (65524.393, 5.0, 223.157), (65670.435, 3.0, 77.115), (65747.550, 1.0, 0.0),
            (65811.843, 5.0, 223.157), (65957.885, 3.0, 77.115), (66035.000, 1.0, 0.0), (75423.767, 5.0, 33326.053))
    return c.n_over_u(14, 1) * _levels(rows, sigma).weighted_sum(c) * c.stim


def fe1_bound_free(c: Conditions) -> torch.Tensor:
    """Fe I (FE1OP): 48 pseudo-levels, sigma = 3E-18 / (1 + ((w_i + 3000 - w) / (0.1 w_i))^4) above w_i (cm^-1)."""
    dev = c.nu.device
    g, e, wno = (fortran_data("fe1op", k, dev) for k in ("g", "e", "wno"))

    def sigma(nu):
        w = (nu / C_CM)[:, None]
        return 3.0e-18 / (1.0 + ((wno + 3000.0 - w) / wno / 0.1) ** 4)

    lv = Levels(tuple(g.tolist()), tuple(e.tolist()), tuple((wno * C_CM).tolist()), sigma)
    return c.n_over_u(26, 1) * lv.weighted_sum(c) * c.stim


# ------------------------------------------------------------------------------------- intermediate-temperature

def _seaton(nu0, s0, p, a):
    """Seaton's form s0 (a + (1 - a) r) r^p, r = nu0/nu (FUNCTION SEATON; HOTOP)."""
    return lambda nu: s0 * (a + (1.0 - a) * (nu0 / nu)) * (nu0 / nu) ** p


def n1_bound_free(c: Conditions) -> torch.Tensor:
    """N I 2p3 4S, 2D, 2P (Henry; N1OP), energies in eV."""
    s = (_seaton(3.517915e15, 1.142e-17, 2.0, 4.29), _seaton(2.941534e15, 4.41e-18, 1.5, 3.85),
         _seaton(2.653317e15, 4.2e-18, 1.5, 4.34))
    rows = ((3.517915e15, 4.0, 0.0), (2.941534e15, 10.0, 2.384), (2.653317e15, 6.0, 3.575))
    lv = _levels(rows, lambda nu: stack(*(f(nu) for f in s)), unit="eV", edge_unit="Hz")
    return c.n_over_u(7, 1) * lv.weighted_sum(c) * c.stim


def o1_bound_free(c: Conditions) -> torch.Tensor:
    """O I 2p4 3P (Peterson after Peach; O1OP)."""
    lv = _levels(((3.28805e15, 9.0, 0.0),), lambda nu: _seaton(3.28805e15, 2.94e-18, 1.0, 2.66)(nu)[:, None],
                 unit="eV", edge_unit="Hz")
    return c.n_over_u(8, 1) * lv.weighted_sum(c) * c.stim


def mg2_bound_free(c: Conditions) -> torch.Tensor:
    """Mg II 3s 2S, 3p 2P (fits to Peach; MG2OP)."""
    lv = _levels(((3.635492e15, 2.0, 0.0), (2.564306e15, 6.0, 4.43)),
                 lambda nu: stack(_seaton(3.635492e15, 1.40e-19, 4.0, 6.7)(nu), 5.11e-19 * (2.564306e15 / nu) ** 3),
                 unit="eV", edge_unit="Hz")
    return c.n_over_u(12, 2) * lv.weighted_sum(c) * c.stim


def ca2_bound_free(c: Conditions) -> torch.Tensor:
    """Ca II 4s 2S, 3d 2D, 4p 2P (fits to Peach; CA2OP)."""
    lv = _levels(((2.870454e15, 2.0, 0.0), (2.460127e15, 10.0, 1.697), (2.110779e15, 6.0, 3.142)),
                 lambda nu: stack(5.4e-20 * (2.870454e15 / nu) ** 3, 1.64e-17 * torch.sqrt(2.460127e15 / nu),
                                  _seaton(2.110779e15, 4.13e-18, 3.0, 0.69)(nu)),
                 unit="eV", edge_unit="Hz")
    return c.n_over_u(20, 2) * lv.weighted_sum(c) * c.stim


def si2_bound_free(c: Conditions) -> torch.Tensor:
    """Si II: Peach's cross-section per atom in the 3p 2P ground term (g = 6), tabulated for 10000-20000 K (SI2OP DATA
    PEACH, ln sigma): linear in ln nu between the nodes FLOG on the side of each edge FREQSI, then linear in ln T
    (extrapolated outside the table)."""
    dev = c.nu.device
    peach = fortran_data("si2op", "peach", dev)                         # (6 temperatures, 14 columns)
    freqsi, flog, tlg = (fortran_data("si2op", k, dev) for k in ("freqsi", "flog", "tlg"))
    k = torch.searchsorted(-freqsi, -c.nu, right=True)                # interval 0..7: first edge below nu
    d = ((torch.log(c.nu) - flog[k]) / (flog[k + 1] - flog[k]))[:, None]
    col = torch.where(k >= 2, 2 * k - 1, k).clamp(max=12)                   # left column (0-based) of the pair
    ln_sigma = peach[:, col].T * (1.0 - d) + peach[:, col + 1].T * d     # (F, 6)
    ln_s = linear_rows(torch.log(c.T), tlg, ln_sigma.T)
    return c.n_over_u(14, 2) * 6.0 * torch.exp(ln_s) * c.stim


# ------------------------------------------------------------------------------------------ high-temperature

# HOTOP column of XNFP -> (Z, stage): C I-IV, N I-V, O I-VI, Ne I-VI
_HOT_SPECIES = [(6, s) for s in range(1, 5)] + [(7, s) for s in range(1, 6)] + [(8, s) for s in range(1, 7)] \
    + [(10, s) for s in range(1, 7)]


def hot_bound_free(c: Conditions) -> torch.Tensor:
    """Low-lying levels of C I-IV, N I-V, O I-VI, Ne I-VI with edges in the H Lyman continuum (sec. 5.11; HOTOP DATA A:
    nu_0, sigma_0, alpha, 2p, g, E (eV), species): sigma = sigma_0 (alpha + (1 - alpha) r) r^p, r = nu_0/nu."""
    a = fortran_data("hotop", "a", c.nu.device).reshape(-1, 7)
    nu0, s0, alpha, p2, g, e = (a[:, i] for i in range(6))
    species = [_HOT_SPECIES[int(i) - 1] for i in a[:, 6].tolist()]
    r = nu0 / c.nu[:, None]
    sigma = torch.where(c.nu[:, None] >= nu0, s0 * (alpha + (1.0 - alpha) * r) * r ** (torch.trunc(p2) / 2.0),
                        torch.zeros((), dtype=r.dtype))                  # (F, 60)
    nou = torch.cat([c.n_over_u(z, s) for z, s in species], -1)          # (..., L, 60)
    xx = nou * g                                                         # n g per edge
    if not c.compat.hot_edge_cut:
        return (xx * torch.exp(-e / c.kt_ev)) @ sigma.T * c.stim
    start = metal_free_free(c) / c.stim                                  # HOTOP's running total starts at its ff
    total = start
    for i in range(sigma.shape[1]):
        term = xx[..., i:i + 1] * sigma[:, i]
        total = torch.where(term > total / 100.0, total + term * torch.exp(-e[i:i + 1] / c.kt_ev), total)
    return (total - start) * c.stim


HOTOP_FF_ELEMENTS = (6, 7, 8, 10, 12, 14, 16, 26)


def metal_free_free(c: Conditions) -> torch.Tensor:
    """Free-free of electrons on metal ions (Z >= 3), hydrogenic with the ion charge q:
    sum_q q^2 g(q) n_q free_free n_e / (nu^3 sqrt T) (1 - e^-hnu/kT) (HOTOP, there for C, N, O, Ne, Mg, Si, S, Fe
    ions of charge 1-5)."""
    n = c.pops.n_atom
    if c.compat.metal_ff_hotop:
        z = torch.as_tensor([e - 1 for e in HOTOP_FF_ELEMENTS], device=n.device)
        ions = n[..., z, 1:6].sum(-2)
    else:
        ions = n[..., 2:, 1:].sum(-2)                                    # (..., L, charges 1..9)
    total = 0.0
    for q in range(1, ions.shape[-1] + 1):
        total = total + q * q * gaunt_free_free(c, float(q)) * ions[..., q - 1, None]
    return c.rad.free_free * total * c.n_e / (c.nu ** 3 * torch.sqrt(c.T)) * c.stim
