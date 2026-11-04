"""Hydrogen lines: Stark + Doppler (fine structure) + Lorentz profiles of SYNTHE's HPROF4 (Peterson's
approximation to Griem's theory: Kurucz 1970, SAO SR 309, sec. 5.14, eqs. 5.11-5.21), and XLINOP's treatment of
overlapping series members (each member is cut where a neighbour dominates, and the series merges into the
continuum at the Inglis-Teller limit).

Profiles are returned per unit frequency (s): phi = HPROF4 / (sqrt(pi) nu_nm Delta v_D/c).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from ....tables import table
from ...constants import decimal32
from .linelist import LineList
from .voigt import SQRT_PI

C_NM = 2.99792458e17
C_A = 2.99792458e18
RYD_H_HZ = 3.2880515e15                # R_H c (HPROF4 DATA RYDH)
R_H = 109677.576                       # cm^-1
R_INF_LIM = 109678.764                 # H I ionisation wavenumber


def level_energy(n: int) -> float:
    """H I level energy, cm^-1 (XLINOP EHYD: measured for n <= 8, Rydberg formula above)."""
    tab = (0.0, 82259.105, 97492.302, 102823.893, 105291.651, 106632.160, 107440.444, 107965.051)
    return tab[n - 1] if n <= 8 else R_INF_LIM - R_H / n ** 2


def line_wavelength(n: int, m: int) -> float:
    return 1e7 / (level_energy(m) - level_energy(n))


def oscillator_strength(n: int, m: int) -> float:
    """f(n -> m), Kurucz's fit to the hydrogenic values (synthe.for FUNCTION HFNM, after Menzel & Pekeris)."""
    if m <= n:
        return 0.0
    xn, xm = float(n), float(m)
    ginf = 0.2027 / xn ** 0.71
    gca = 0.124 / xn
    fkn = xn * 1.9603
    wtc = 0.45 - 2.4 / xn ** 3 * (xn - 1.0)
    xmn = xm - xn
    fk = fkn * (xm / (xmn * (xm + xn))) ** 3
    xmn12 = xmn ** 1.2
    wt = (xmn12 - 1.0) / (xmn12 + wtc)
    return fk * (1.0 - wt * ginf - (0.222 + gca / xm) * (1.0 - wt))


@lru_cache(maxsize=None)
def _tables():
    t = lambda unit, v: decimal32(table("synthe.for", unit, v))
    prob = np.stack([t("sofbet", f"prob{i}").reshape(15, 5) for i in range(1, 8)])     # [indx, beta, P]
    c = np.stack([t("sofbet", f"c{i}") for i in range(1, 8)])                          # [indx, P]
    d = np.stack([t("sofbet", f"d{i}") for i in range(1, 8)])
    return dict(prob=prob, c=c, d=d, pp=t("sofbet", "pp"), beta=t("sofbet", "beta"),
                xknmtb=t("hprof4", "xknmtb"), y1wtm=t("hprof4", "y1wtm"), stalph=t("hprof4", "stalph"),
                stwtal=t("hprof4", "stwtal"), istal=table("synthe.for", "hprof4", "istal"),
                lnghal=table("synthe.for", "hprof4", "lnghal"), stcomp=t("hprof4", "stcomp"),
                stcpwt=t("hprof4", "stcpwt"), lncomp=table("synthe.for", "hprof4", "lncomp"),
                asum=t("hprof4", "asum"), asumlyman=t("hprof4", "asumlyman"))


def e1(x: torch.Tensor) -> torch.Tensor:
    """Exponential integral E1(x), x > 0 (Abramowitz & Stegun 5.1.53, 5.1.56; HPROF4 uses the same, VCSE1F)."""
    xs = x.clamp(min=1e-300)
    small = -torch.log(xs) - 0.57721566 + xs * (0.99999193 + xs * (-0.24991055 + xs * (
        0.05519968 + xs * (-0.00976004 + xs * 0.00107857))))
    xl = x.clamp(min=1.0)
    large = (xl * (xl + 2.334733) + 0.25062) / (xl * (xl + 3.330657) + 1.681534) / xl * torch.exp(-xl)
    return torch.where(x <= 0, torch.zeros_like(x), torch.where(x <= 1.0, small, large))


def stark_shape(beta: torch.Tensor, p: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """S(beta, P): quasistatic Holtsmark profile with Debye-shielding (P) corrections, renormalised to the full
    oscillator strength; alpha and beta lines of the first three series tabulated, H18 for the rest
    (synthe.for FUNCTION SOFBET; Kurucz 1970 eqs. 5.14-5.18)."""
    tb = _tables()
    dev, dt = beta.device, beta.dtype
    mmn = m - n
    indx = 2 * (n - 1) + mmn if (n <= 3 and mmn <= 2) else 7
    prob = torch.as_tensor(tb["prob"][indx - 1], dtype=dt, device=dev)       # [beta, P]
    cc_t = torch.as_tensor(tb["c"][indx - 1], dtype=dt, device=dev)
    dd_t = torch.as_tensor(tb["d"][indx - 1], dtype=dt, device=dev)
    bt = torch.as_tensor(tb["beta"], dtype=dt, device=dev)
    pp = torch.as_tensor(tb["pp"], dtype=dt, device=dev)
    b, p = torch.broadcast_tensors(beta, p)
    im = torch.clamp((5.0 * p).floor().long(), max=3)                         # IM - 1
    wtpp = 5.0 * (p - pp[im])
    wtpm = 1.0 - wtpp
    jp = torch.searchsorted(bt, b.contiguous()).clamp(1, 14)                  # first BETA(J) >= b, J >= 2
    jm = jp - 1
    wtbp = (b - bt[jm]) / (bt[jp] - bt[jm])
    cbp = prob[jp, im + 1] * wtpp + prob[jp, im] * wtpm
    cbm = prob[jm, im + 1] * wtpp + prob[jm, im] * wtpm
    corr_in = 1.0 + cbp * wtbp + cbm * (1.0 - wtbp)
    bs = b.clamp(min=1e-30)
    b2 = bs * bs
    sb = torch.sqrt(bs)
    wt = torch.clamp(0.5 * (10.0 - b), 0.0, 1.0)
    pr1 = torch.where(b <= 10.0, 8.0 / (83.0 + (2.0 + 0.95 * b2) * b), torch.zeros_like(b))
    asym = (1.5 / sb + 27.0 / b2) / b2
    pr2 = torch.where(b >= 8.0, asym, torch.zeros_like(b))
    inner = (pr1 * wt + pr2 * (1.0 - wt)) * corr_in
    ccv = cc_t[im + 1] * wtpp + cc_t[im] * wtpm
    ddv = dd_t[im + 1] * wtpp + dd_t[im] * wtpm
    outer = asym * (1.0 + ddv / (ccv + bs * sb))
    return torch.where(b > 500.0, asym, torch.where(b > 25.12, outer, inner))


class HydrogenLine:
    """Line constants of H I n -> m (HPROF4 'set up for this line')."""

    def __init__(self, n: int, m: int):
        if n == 1 and m == 2:
            raise NotImplementedError("Lyman alpha (quasi-molecular H2+/H2 wings) is not implemented")
        tb = _tables()
        self.n, self.m = n, m
        mmn = m - n
        xn2, xm2 = float(n * n), float(m * m)
        gnm = (xm2 - xn2) / (xm2 * xn2)
        if mmn <= 3 and n <= 4:
            xknm = float(tb["xknmtb"][n - 1, mmn - 1])
        else:
            xknm = 5.5e-5 / gnm * xm2 * xn2 / (1.0 + 0.13 / mmn)
        self.y1num = 550.0 if m == 2 else 380.0 if m == 3 else 320.0
        self.y1wht = float(tb["y1wtm"][n - 1, mmn - 1]) if (mmn <= 2 and n <= 2) else 1e14 if mmn <= 3 else 1e13
        self.freq = RYD_H_HZ * gnm
        self.dbeta = C_A / self.freq ** 2 / xknm
        self.wave_a = C_A / self.freq
        self.c1con = xknm / self.wave_a * gnm * (xm2 - xn2)
        self.c2con = (xknm / self.wave_a) ** 2
        radamp = float(tb["asumlyman"][m - 1]) if n == 1 else float(tb["asum"][n - 1] + tb["asum"][m - 1])
        self.radamp = radamp / 12.5664 / self.freq
        resont = oscillator_strength(1, m) / m / (1.0 - 1.0 / xm2)
        if n != 1:
            resont += oscillator_strength(1, n) / n / (1.0 - 1.0 / xn2)
        self.resont = resont * 3.579e-24 / gnm
        self.vdw = 4.45e-26 / gnm * (xm2 * (7.0 * xm2 + 5.0)) ** 0.4
        self.stark = 1.6678e-18 * self.freq * xknm
        if n > 4 or m > 10:
            fs, fw = [0.0], [1.0]
        elif mmn == 1:
            i0, k = int(tb["istal"][n - 1]), int(tb["lnghal"][n - 1])
            fs = [float(tb["stalph"][i0 - 1 + i]) * 1e7 for i in range(k)]
            fw = [float(tb["stwtal"][i0 - 1 + i]) / xn2 / 3.0 for i in range(k)]
        else:
            k = int(tb["lncomp"][n - 1])
            fs = [float(tb["stcomp"][i, n - 1]) * 1e7 for i in range(k)]
            fw = [float(tb["stcpwt"][i, n - 1]) / xn2 for i in range(k)]
        self.fine_shift, self.fine_weight = fs, fw

    def profile(self, dlam_nm: torch.Tensor, lay: dict, dopph: torch.Tensor) -> torch.Tensor:
        """phi(nu) in s at wavelength offset dlam_nm (M, K) from the line centre; lay: per-layer (M, 1) quantities
        of depth_quantities; dopph (M, 1) Doppler width / c of H."""
        delstark = -10.0 * dlam_nm / self.wave_a * self.freq
        wl_a = self.wave_a + 10.0 * dlam_nm
        freq = C_A / wl_a
        delta = (freq - self.freq).abs()
        hwstk = self.stark * lay["fo"]
        hwvdw = self.vdw * lay["t3nhe"] + 2.0 * self.vdw * lay["t3nh2"]
        hwres = self.resont * lay["n_h1_ground"]
        hwlor = hwres + hwvdw + self.radamp
        hwdop = dopph
        dop = self.freq * hwdop
        doppler_wide = (hwdop >= hwstk) & (hwdop >= hwlor)
        lorentz_wide = ~doppler_wide & (hwlor >= hwstk)
        stark_wide = ~doppler_wide & ~lorentz_wide
        core = delta <= self.freq * torch.maximum(torch.maximum(hwdop, hwlor), hwstk)
        # Doppler core with the fine-structure pattern (normalised like H(a, v))
        pd = torch.zeros_like(delta)
        for s, w in zip(self.fine_shift, self.fine_weight):
            d = (freq - self.freq - s).abs() / dop
            pd = pd + torch.where(d <= 7.0, torch.exp(-d * d), torch.zeros_like(d)) * w
        hhw = self.freq * hwlor
        pl = hhw / math.pi / (delta ** 2 + hhw ** 2) * SQRT_PI * dop
        # Stark: quasistatic ions (S(beta)) with the electron impact width (Griem 1960 asymptotic theory)
        wty1 = 1.0 / (1.0 + lay["n_e"] / self.y1wht)
        y1scal = self.y1num * lay["y1s"] * wty1 + lay["y1b"] * (1.0 - wty1)
        c1 = lay["c1d"] * self.c1con * y1scal
        c2 = lay["c2d"] * self.c2con
        g1 = 6.77 * torch.sqrt(c1)
        gnot = g1 * torch.clamp(0.2114 + torch.log(torch.sqrt(c2) / c1), min=0.0) * (1.0 - lay["gcon1"] - lay["gcon2"])
        beta = delstark.abs() / lay["fo"] * self.dbeta
        y1 = c1 * beta
        y2 = c2 * beta ** 2
        gam = g1 * (0.5 * torch.exp(-torch.clamp(y1, max=80.0)) + e1(y1) - 0.5 * e1(y2)) * (
            1.0 - lay["gcon1"] / (1.0 + (90.0 * y1) ** 3) - lay["gcon2"] / (1.0 + 2000.0 * y1))
        gam = torch.where((y2 <= 1e-4) & (y1 <= 1e-5), gnot.expand_as(gam), gam)
        gam = torch.where(gam <= 1e-20, torch.zeros_like(gam), gam)
        prqs = stark_shape(beta, lay["pp"], self.n, self.m)
        f = torch.where(gam > 0, gam / math.pi / (gam ** 2 + beta ** 2), torch.zeros_like(gam))
        p1 = (0.9 * y1) ** 2
        fns = (p1 + 0.03 * torch.sqrt(y1)) / (p1 + 1.0)
        ps = (prqs * (1.0 + fns) + f) / lay["fo"] * self.dbeta * SQRT_PI * dop
        wings = pd + pl + ps
        h = torch.where(core, torch.where(doppler_wide, pd, torch.where(lorentz_wide, pl, ps)), wings)
        return h / (SQRT_PI * dop)


def depth_quantities(gas) -> dict:
    """Per-layer (M, 1) quantities of HPROF4 ('set up depth vectors')."""
    T, ne = gas.T[:, None], gas.n_e[:, None]
    ne16 = ne ** (1.0 / 6.0)
    fo = ne16 ** 4 * 1.25e-9                      # Holtsmark normal field, Kurucz 1970 eq. 5.16
    t4 = T / 1e4
    t43 = t4 ** 0.3
    return dict(n_e=ne, fo=fo, pp=ne16 * 0.08989 / torch.sqrt(T), y1b=2.0 / (1.0 + 0.012 / T * torch.sqrt(ne / T)),
                y1s=t43 / ne16, t3nhe=t43 * gas.n_he1[:, None], t3nh2=t43 * gas.n_h2[:, None],
                c1d=fo * 78940.0 / T, c2d=fo ** 2 / 5.96e-23 / ne,
                gcon1=0.2 + 0.09 * torch.sqrt(t4) / (1.0 + ne / 1e13), gcon2=0.2 / (1.0 + ne / 1e15),
                n_h1_ground=2.0 * gas.n_over_u_atom[:, 0, 0][:, None])


def hydrogen_lines(op, gas, cont: torch.Tensor, ll: LineList) -> torch.Tensor:
    """Opacity per gram (M, N) of the H I lines of ll (XLINOP labels 600-634; deuterium: Doppler width / sqrt 2)."""
    M, N = cont.shape
    out = torch.zeros_like(cont)
    if not len(ll):
        return out
    opt, grid = op.opt, op.grid
    lay = depth_quantities(gas)
    wl_grid = grid.wavelengths(cont.device)
    strength = op._strength(gas, ll)                                    # (M, n)
    mass_h = torch.full((len(ll),), 1.008, dtype=cont.dtype, device=cont.device)
    dop_all = op._doppler(gas, torch.where(ll.mass > 1.5, ll.mass, mass_h) if opt.doppler_mass == "line"
                          else torch.where(ll.mass > 1.5, 2.0 * mass_h, mass_h))
    pos_all = grid.position(ll.wavelength)
    emerge = (R_H / (1600.0 / gas.n_e ** (2.0 / 15.0) - 1.5) ** 2)[:, None]
    chunk = max(256, opt.budget // (16 * M))
    for i in range(len(ll)):
        n, m = int(ll.n_lower[i]), int(ll.n_upper[i])
        hl = HydrogenLine(n, m)
        wl0 = float(ll.wavelength[i])
        dop = dop_all[:, i:i + 1]
        k0 = strength[:, i:i + 1] / (SQRT_PI * (C_NM / wl0) * dop)
        c = int(round(float(pos_all[i])))
        kc = cont[:, min(max(c, 0), N - 1)][:, None]
        active = k0 >= opt.cutoff * kc
        if not bool(active.any()):
            continue
        s = strength[:, i:i + 1]
        ncon = int(ll.series_limit[i]) if int(ll.series_limit[i]) else 0
        alpha_wl = line_wavelength(n, n + 1)
        kind = "isolated" if (ncon == 0 or m == n + 1) else "beta" if m == n + 2 else "member"
        if kind == "member":
            conth = float(table("synthe.for", "xlinop", "conth")[ncon - 1])
            wshift = 1e7 / (conth - R_H / 81.0 ** 2)
            wmerge = 1e7 / (conth - emerge)
            wmerge = torch.where(wmerge < 0, torch.full_like(wmerge, 2 * wshift), wmerge)
            wcon = torch.clamp(wmerge, min=wshift)
            wtail = 1e7 / (1e7 / wcon - 500.0)
            wcon = torch.clamp(wcon, max=2 * wshift)
            wtail = torch.where(wtail < 0, 2 * wcon, wtail)
            wtail = torch.minimum(2 * wcon, wtail)
            f_m = oscillator_strength(n, m)
            s_red = s * oscillator_strength(n, m - 2) / f_m if m - 2 > n else None
            s_blue = s * oscillator_strength(n, m + 2) / f_m
            hl_red = HydrogenLine(n, m - 2) if m - 2 > n else None
            hl_blue = HydrogenLine(n, m + 2)
            wl_m1, wl_m2 = line_wavelength(n, m - 1), line_wavelength(n, m - 2)
            wl_p1, wl_p2 = line_wavelength(n, m + 1), line_wavelength(n, m + 2)
            redcut = 1e7 / (R_INF_LIM - R_H / (m - 0.8) ** 2 - level_energy(n))
            bluecut = 1e7 / (R_INF_LIM - R_H / (m + 0.8) ** 2 - level_energy(n))

        def ramp(w):
            return torch.where(w < wtail, (w - wcon) / (wtail - wcon), torch.ones_like(w))

        def values(ks, side):
            """kappa and the continue-walking condition at grid points ks (XLINOP wing loops)."""
            w = wl_grid[ks][None, :]
            kap = s * hl.profile(w - wl0, lay, dop)
            if kind != "member":
                return kap, kap >= opt.cutoff * cont[:, ks]
            r = ramp(w)
            kap = kap * r
            ok = kap >= opt.cutoff * cont[:, ks]
            if side == "red":
                skip = w < wcon
                kr = s_red * hl_red.profile(w - wl_m2, lay, dop) * r
                ok = (ok & ~((w > redcut) & (kr >= kap))) | skip
                return torch.where(skip, torch.zeros_like(kap), kap), ok
            kb = s_blue * hl_blue.profile(w - wl_p2, lay, dop) * r
            return kap, ok & (w >= wcon) & ~((w < bluecut) & (kb > kap))

        for side in ("red", "blue"):
            if side == "red":
                if c > N - 1 or (kind == "member" and grid.wl_begin > alpha_wl):
                    continue
                ks = torch.arange(max(c, 0), N, device=cont.device)
                if kind == "beta":
                    ks = ks[wl_grid[ks] <= alpha_wl]
                elif kind == "member":
                    ks = ks[wl_grid[ks] <= wl_m1]
            else:
                if c <= 0:
                    continue
                ks = torch.arange(min(c, N) - 1, -1, -1, device=cont.device)
                if kind == "member":
                    ks = ks[wl_grid[ks] >= wl_p1]
            alive = active
            for k0_ in range(0, len(ks), chunk):
                kc_ = ks[k0_:k0_ + chunk]
                kap, ok = values(kc_, side)
                alive = torch.cumprod(ok.to(torch.int8), -1).bool() & alive[:, -1:]
                out[:, kc_] += torch.where(alive, kap, torch.zeros_like(kap))
                if not bool(alive[:, -1].any()):
                    break
    return out
