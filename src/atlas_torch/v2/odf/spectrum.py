"""The line opacity spectrum of one ODF temperature on a block of the DFSYNTHE grid (25 pressures x N points):
lines of the packed lists with Voigt profiles, the NLTE list (hydrogen Stark profiles, autoionizing and series
lines, merged continua), the continuum and its 10^-3 floor (Castelli 2005, MSAIS 8, 34, p. 35; Kurucz 2005,
MSAIS 8, 76; dfsynthe.for MAIN and XLINOP). Profiles are deposited as wings of (layer, line) pairs (wings.py)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from ..constants import CODATA2018, PhysicalConstants
from ..opacity.lines.grid import LogWavelengthGrid
from ..opacity.lines.linelist import LineKind, LineList
from ..opacity.lines.opacity import LineOpacity, LineOpacityOptions, pi_e2_over_mc, series_limits
from ..opacity.lines.voigt import voigt, voigt_kurucz
from .gas import OdfGas
from .grid import DF_GRID
from .lines import C_NM, SQRT_PI, DfLines, DfLineSource, SeriesContinua, center_opacity, nelion_species
from .wings import WingPairs, deposit, first_below, first_below_from, grid_wavelength

HALO = 10046        # widest wing (MAXPROF = 10000 points) plus the widest turbulence kernel (VTTAB NVT <= 46)


@dataclass(frozen=True)
class SpectrumOptions:
    """lines: LineOpacityOptions (voigt, cutoff_rule, max_wing_points, cutoff; microturbulence must be 0: the ODF
    spectrum is the 0 km/s one); voigt "kurucz" also selects DFSYNTHE's wavelength offsets (n/R Doppler widths,
    XLINOP |lambda - lambda_0|), else |nu - nu_0|.
    weak_line_doppler_cap / molecular_doppler_cap: wings of lines with (pi e^2/mc)/sqrt(pi) gf e^(-E/kT)
    (1 - e^(-h nu/kT)) < 1e-4, and of molecular lines, cut at this many Doppler widths (MAIN: N10DOP*10); cap_slots
    'molecules' | 'nelion6' (MAIN's MOD(NELION,6) = 0 test, which also catches the sixth ion stages).
    nlte_wings: the NLTE-list Voigt lines 'halo': within HALO points of the block, each wing point kept where it
    exceeds the cutoff of the local continuum; 'walk' (XLINOP labels 200-214): each wing added out to and including
    its first point below the cutoff, across the whole grid for strong lines (centre > half the continuum; also
    every autoionizing line), within xlinop_window points for the others.
    lyman_alpha: 'impact' (opacity.lines.hydrogen.HydrogenLine: HPROF4 without its quasi-molecular H2/H2+ wings)
    | 'omit'.
    hydrogen_sampling: hydrogen lines on every k-th point, log-interpolated between (XLINOP every 20th point);
    budget: profile points per block; line_chunk: packed-list lines decoded at a time (memory)."""
    lines: LineOpacityOptions = field(default_factory=lambda: LineOpacityOptions(cutoff=1e-3, microturbulence=0.0))
    weak_line_doppler_cap: float | None = None
    molecular_doppler_cap: float | None = None
    cap_slots: str = "molecules"
    nlte_wings: str = "halo"
    xlinop_window: int | None = None
    lyman_alpha: str = "impact"
    hydrogen_sampling: int = 1
    budget: int = 1 << 20
    line_chunk: int = 20000

    @classmethod
    def kurucz(cls, **changes) -> "SpectrumOptions":
        lo = LineOpacityOptions(cutoff=1e-3, microturbulence=0.0, voigt="kurucz", cutoff_rule="kurucz",
                                max_wing_points=10000, line_center="grid", doppler_mass="species", helium_ii="voigt")
        return replace(cls(lines=lo, weak_line_doppler_cap=100.0, molecular_doppler_cap=100.0, cap_slots="nelion6",
                           nlte_wings="walk", xlinop_window=1000, hydrogen_sampling=20), **changes)


class LineSpectrum:
    """kappa_lines(lo, hi) -> (L, hi - lo + 1) cm^2 g^-1 for one OdfGas row (on the gas's device).
    counts: lines, (layer, line) pairs and profile points of the packed lists."""

    def __init__(self, gas: OdfGas, source: DfLineSource | None, nlte: tuple | None = None,
                 options: SpectrumOptions = SpectrumOptions(), const: PhysicalConstants = CODATA2018):
        if options.nlte_wings not in ("halo", "walk") or options.lyman_alpha not in ("impact", "omit"):
            raise ValueError("nlte_wings: 'halo' | 'walk'; lyman_alpha: 'impact' | 'omit'")
        self.gas, self.source, self.opt, self.const = gas, source, options, const
        self.device = gas.T.device
        self.kurucz = options.lines.voigt == "kurucz"
        self.nlte_list, cont_rec = nlte if nlte is not None else (None, None)
        self.series = SeriesContinua(cont_rec) if cont_rec is not None and len(cont_rec) else None
        _, _, mol = nelion_species()
        capped = mol if options.cap_slots == "molecules" else (np.arange(594) % 6 == 5)
        self.capped = torch.as_tensor(capped, device=self.device)
        self._walk_pairs = None
        self.counts = {"lines": 0, "pairs": 0, "points": 0}

    def continuum(self, lo: int, hi: int) -> torch.Tensor:
        """Continuous absorption + scattering (L, hi - lo + 1)."""
        return self.gas.continuum(_window(lo, hi))

    def _floor_at(self, layer: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return self.opt.lines.cutoff * self.gas.continuum_at(layer, grid_wavelength(index))

    def _profile(self, a: torch.Tensor, v: torch.Tensor, form: str = "main") -> torch.Tensor:
        """H(a, v): voigt_kurucz `form` (MAIN for the packed lists, FUNCTION VOIGT for XLINOP) or voigt."""
        return voigt_kurucz(a, v, form) if self.kurucz else voigt(a, v)

    # ------------------------------------------------------------------ packed lists (MAIN)

    def _bounds(self, k0, a, pd, contmin):
        """Estimated wing extent (points) of (layer, line) pairs: where the Gaussian core or the Lorentz wing
        k0 a/(sqrt(pi) v^2) falls below contmin, limited by max_wing_points."""
        vg = torch.sqrt(torch.log((k0 / contmin).clamp(min=1.0)))
        vl = torch.sqrt(k0 * a / (SQRT_PI * contmin))
        cap = float(self.opt.lines.max_wing_points or HALO)
        return (torch.ceil(torch.maximum(vg, vl) * 1.02 * pd) + 2.0).clamp(1.0, cap)

    def _doppler_caps(self, n, a, pd, weak, capped):
        """MAIN: wings of a <= 0.2 lines that are weak, or molecular, cut at N10DOP * 10 points."""
        o, n10, small = self.opt, torch.floor(10.0 * pd), a <= 0.2
        if o.weak_line_doppler_cap is not None:
            n = torch.where(weak & small, torch.minimum(n, n10 * (o.weak_line_doppler_cap / 10.0)), n)
        if o.molecular_doppler_cap is not None:
            n = torch.where(capped & small, torch.minimum(n, n10 * (o.molecular_doppler_cap / 10.0)), n)
        return n

    def _main_extent(self, k0, a, pd, contmin, weak, capped):
        """MAIN's wing extent (dfsynthe.for loops 1321-1325): the profile is deposited out to its first point below
        contmin (the cutoff at the line centre), within N10DOP = 10 Doppler widths for a <= 0.2 and MAXPROF
        points otherwise; an a <= 0.2 profile still above contmin at N10DOP continues as X/n^2 out to MAXSTEP =
        sqrt(X/contmin) + 1 (X = profile(N10DOP) N10DOP^2), all points kept, Doppler caps applied."""
        cap = float(self.opt.lines.max_wing_points or HALO)
        n10 = torch.floor(10.0 * pd)
        small = a <= 0.2
        p10 = k0 * self._profile(a, n10 / pd)
        far = small & (p10 >= contmin)
        maxstep = torch.floor(torch.sqrt(p10 * n10 ** 2 / contmin) + 1.0).clamp(max=cap)
        maxstep = self._doppler_caps(maxstep, a, pd, weak, capped)
        out = maxstep.long()
        w = (~far).nonzero().squeeze(1)
        ratio = lambda it, n: k0[w[it]] * self._profile(a[w[it]], n / pd[w[it]]) / contmin[w[it]]   # noqa: E731
        n_max = torch.where(small[w], n10[w], cap).long()
        guess = torch.minimum(self._bounds(k0[w], a[w], pd[w], contmin[w]).long(), n_max)
        out[w] = first_below_from(ratio, guess, torch.ones_like(guess), n_max)
        return out

    def _packed_pairs(self, lines: DfLines, cont_ext: torch.Tensor, ext_lo: int) -> WingPairs | None:
        """The (layer, line) pairs of packed-list lines whose centre opacity reaches the cutoff (MAIN label 301)."""
        o, g = self.opt, self.gas
        k0, a, dop = center_opacity(lines, g, self.const)                              # (L, n)
        c = lines.center.to(self.device)
        contmin = o.lines.cutoff * cont_ext[:, (c - ext_lo).clamp(0, cont_ext.shape[1] - 1)]
        active = k0 >= contmin
        if not bool(active.any()):
            return None
        jj, ll = active.nonzero(as_tuple=True)
        self.counts["lines"] += int(active.any(0).sum())
        self.counts["pairs"] += int(jj.shape[0])
        k0, a, dop, contmin = k0[jj, ll], a[jj, ll], dop[jj, ll], contmin[jj, ll]
        pd = dop * DF_GRID.resolution if self.kurucz else dop / DF_GRID.step              # points per Doppler width
        nu = C_NM / lines.wavelength[ll]
        stim = -torch.expm1(-self.const.h * nu / (self.const.k * g.T[0]))
        weak = (pi_e2_over_mc(self.const) / SQRT_PI) * 10.0 ** lines.log_gf[ll] \
            * torch.exp(-lines.e_lower[ll] * self.const.hc_k / g.T[0]) * stim < 1e-4
        capped = self.capped[lines.slot[ll]]
        if o.lines.cutoff_rule == "kurucz":
            n = self._main_extent(k0, a, pd, contmin, weak, capped)
        else:
            n = self._doppler_caps(self._bounds(k0, a, pd, contmin), a, pd, weak, capped).long()
        zero = torch.zeros_like(n)
        return WingPairs(jj, c[ll], grid_wavelength(c[ll]), k0, a, dop, zero, n, n, zero.double(), zero.double())

    def _packed_value(self, cont_ext: torch.Tensor, ext_lo: int):
        """kappa at grid points of packed-list pairs: k0 H(a, v); v = n/(R Delta nu_D/nu) (MAIN) or |nu - nu_0| /
        Delta nu_D; with cutoff_rule 'local' only where it reaches the cutoff of the local continuum."""
        cut = self.opt.lines.cutoff

        def value(p: WingPairs, it: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            if self.kurucz:
                v = (idx - p.center[it]).abs() / (p.dop[it] * DF_GRID.resolution)
            else:
                v = (p.wl0[it] / grid_wavelength(idx) - 1.0).abs() / p.dop[it]
            val = p.k0[it] * self._profile(p.a[it], v)
            if self.opt.lines.cutoff_rule == "kurucz":
                return val
            return torch.where(val >= cut * cont_ext[p.layer[it], idx - ext_lo], val, torch.zeros_like(val))
        return value

    # ------------------------------------------------------------------ NLTE list (XLINOP)

    def _nlte(self, lo: int, hi: int, cont: torch.Tensor) -> torch.Tensor:
        o, g = self.opt, self.gas
        grid = _window(lo, hi)
        out = torch.zeros_like(cont)
        if self.series is not None:
            out += self.series(g, grid.wavelengths(self.device), self.const)
        ll = self.nlte_list
        if ll is None or not len(ll):
            return out
        ll = ll.to(self.device)
        lopt = replace(o.lines, max_wing_points=HALO, cutoff_rule="local")     # XLINOP: CONTINUUM(IBUFF) per point
        hyd = ll.kind == int(LineKind.HYDROGEN)
        auto = ll.kind == int(LineKind.AUTOIONIZING)
        near = (ll.wavelength >= _wl(lo - HALO)) & (ll.wavelength <= _wl(hi + HALO))
        if o.nlte_wings == "walk":
            pairs = self._walk_pairs if self._walk_pairs is not None else self._make_walk_pairs(
                ll.select(~hyd & ~auto))
            reach = ((pairs.center + pairs.red >= lo) & (pairs.center - pairs.blue <= hi)).nonzero().squeeze(1)
            if len(reach):
                deposit(pairs.select(reach), lo, hi, out, self._xlinop_value, o.budget)
            near = near | auto
        else:
            out += self._halo_lines(ll.select(near & ~hyd & ~auto), grid, cont, lopt)
        if bool((near & auto).any()):
            out += LineOpacity(ll.select(near & auto), grid, lopt)(g.state, cont, kinds=(LineKind.AUTOIONIZING,))
        if bool(hyd.any()):
            out += self._hydrogen(ll.select(hyd), lo, hi, cont, lopt)
        return out

    def _halo_lines(self, ll: LineList, grid: LogWavelengthGrid, cont: torch.Tensor, lopt) -> torch.Tensor:
        """Voigt lines of the NLTE list near the block; with xlinop_window, lines whose centre stays below half the
        continuum in every layer keep only that many wing points (XLINOP MAXRED/MAXBLUE)."""
        if not len(ll):
            return torch.zeros_like(cont)
        groups = [(ll, lopt)]
        if self.opt.xlinop_window is not None:
            k0, _, _ = LineOpacity(ll, grid, lopt).center_opacity(self.gas.state)
            kc = cont[:, grid.position(ll.wavelength).round().long().clamp(0, cont.shape[1] - 1)]
            strong = (k0 > 0.5 * kc).any(0)
            weak = replace(lopt, max_wing_points=self.opt.xlinop_window)
            groups = [(ll.select(strong), lopt), (ll.select(~strong), weak)]
        return sum(LineOpacity(s, grid, op)(self.gas.state, cont) for s, op in groups)

    def _make_walk_pairs(self, ll: LineList) -> WingPairs:
        """XLINOP's (layer, line) pairs of the Voigt lines with their wing extents (label 200: KAPPA0 >=
        CONTINUUM(NBUFF); strong if KAPPA0 > 500 CONTINUUM(NBUFF), i.e. half the continuum)."""
        o, g, dev = self.opt, self.gas, self.device
        L = g.T.shape[0]
        k0, a, dop = LineOpacity(ll, _window(0, 1), o.lines).center_opacity(g.state)       # (L, n)
        k0 = k0 * -torch.expm1(-self.const.h * (C_NM / ll.wavelength)[None, :] / (self.const.k * g.T[:, None]))
        c = torch.round(DF_GRID.position(ll.wavelength)).long()
        layer = torch.arange(L, device=dev)[:, None].expand(L, len(ll))
        floor = self._floor_at(layer.reshape(-1), c.expand(L, -1).reshape(-1)).reshape(L, -1)
        active = k0 >= floor
        strong = k0 > 0.5 * floor / o.lines.cutoff
        wcon, wtail = series_limits(ll.species, ll.series_limit, g.n_e)
        series = (ll.series_limit > 0)[None, :].expand(L, -1)
        wcon, wtail = torch.where(series, wcon, 0.0), torch.where(series, wtail, 0.0)
        jj, ii = active.nonzero(as_tuple=True)
        cc = c[ii]
        last = DF_GRID.length - 1
        window = o.xlinop_window if o.xlinop_window is not None else last
        st = strong[jj, ii]
        red = torch.where(st, last - cc, torch.clamp(last - cc, max=window))
        blue = torch.where(st, cc, torch.clamp(cc, max=window))
        wc = wcon[jj, ii]
        i_con = torch.where(wc > 0, torch.ceil(DF_GRID.position(wc.clamp(min=1e-300)) - 1e-9).long(),
                            cc.new_tensor(0))
        red0 = (i_con - cc).clamp(min=0)
        blue = torch.minimum(blue, torch.where(wc > 0, cc - i_con, blue))
        zero = torch.zeros_like(cc)
        p = WingPairs(jj, cc, ll.wavelength[ii], k0[jj, ii], a[jj, ii], dop[jj, ii], red0, zero, zero, wc,
                      wtail[jj, ii])

        def ratio(side):
            return lambda it, n: (self._xlinop_value(p, it, p.center[it] + side * n)
                                  / self._floor_at(p.layer[it], p.center[it] + side * n))
        p = p.replace(red=first_below(ratio(1), red0, red), blue=first_below(ratio(-1), torch.ones_like(cc), blue))
        self._walk_pairs = p
        return p

    def _xlinop_value(self, p: WingPairs, it: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """kappa = k0 H(a, v) at grid points of NLTE-list pairs, v = |lambda - lambda_0|/(lambda_0 Delta nu_D/nu)
        (XLINOP) or |nu - nu_0|/Delta nu_D, tapered by (lambda - lambda_con)/(lambda_tail - lambda_con) below
        lambda_tail (series lines, XLINOP label 211)."""
        wl, w0 = grid_wavelength(idx), p.wl0[it]
        v = ((wl / w0 - 1.0) if self.kurucz else (w0 / wl - 1.0)).abs() / p.dop[it]
        val = p.k0[it] * self._profile(p.a[it], v, "function")
        wc, wt = p.wcon[it], p.wtail[it]
        taper = torch.where(wl < wt, ((wl - wc) / (wt - wc)).clamp(min=0.0), torch.ones_like(wl))
        return val * taper

    def _hydrogen(self, ll: LineList, lo: int, hi: int, cont: torch.Tensor, lopt) -> torch.Tensor:
        """Hydrogen lines (opacity.lines.hydrogen; XLINOP labels 600-634), Lyman alpha unless lyman_alpha='omit'."""
        if self.opt.lyman_alpha == "omit":
            ll = ll.select(~((ll.n_lower == 1) & (ll.n_upper == 2)))
        return self._sampled(ll, lo, hi, cont, lopt)

    def _sampled(self, ll: LineList, lo: int, hi: int, cont: torch.Tensor, lopt) -> torch.Tensor:
        """Hydrogen lines on every k-th grid point (k = hydrogen_sampling), log10 interpolated linearly in the grid
        index between them (XLINOP: every 20th point beyond the first 21)."""
        k = max(1, int(self.opt.hydrogen_sampling))
        if not len(ll):
            return torch.zeros_like(cont)
        if k == 1:
            return LineOpacity(ll, _window(lo, hi), lopt)(self.gas.state, cont, kinds=(LineKind.HYDROGEN,))
        a, b = (DF_GRID.first + lo + k - 1) // k, (DF_GRID.first + hi) // k + 1
        cg = LogWavelengthGrid(1.0 / math.expm1(k * DF_GRID.step), a, b - a + 1)
        idx = torch.arange(b - a + 1, device=self.device) * k + (a * k - DF_GRID.first - lo)
        kh = LineOpacity(ll, cg, lopt)(self.gas.state, cont[:, idx.clamp(0, cont.shape[1] - 1)],
                                       kinds=(LineKind.HYDROGEN,))
        pos = (torch.arange(cont.shape[1], dtype=torch.float64, device=self.device) - idx[0]) / k
        i0 = pos.floor().long().clamp(0, kh.shape[1] - 2)
        t = (pos - i0).clamp(0.0, 1.0)
        lk = torch.log(kh.clamp(min=1e-300))
        val = torch.exp(lk[:, i0] * (1 - t) + lk[:, i0 + 1] * t)
        return torch.where(val > 1e-299, val, torch.zeros_like(val))

    # ------------------------------------------------------------------ public

    def __call__(self, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(line opacity, continuum) on grid points lo..hi, each (L, hi - lo + 1)."""
        cont = self.continuum(lo, hi)
        out = torch.zeros_like(cont)
        if self.source is not None:
            elo, ehi = max(0, lo - HALO), min(DF_GRID.length - 1, hi + HALO)
            cont_ext = self.continuum(elo, ehi)
            value = self._packed_value(cont_ext, elo)
            for lines in self.source.iter(elo, ehi, float(self.gas.T[0]), self.opt.line_chunk, self.device):
                pairs = self._packed_pairs(lines, cont_ext, elo)
                if pairs is not None:
                    self.counts["points"] += deposit(pairs, lo, hi, out, value, self.opt.budget)
            del cont_ext
        out += self._nlte(lo, hi, cont)
        return out, cont


def _window(lo: int, hi: int) -> LogWavelengthGrid:
    return LogWavelengthGrid(DF_GRID.resolution, DF_GRID.first + lo, hi - lo + 1)


def _wl(i: int) -> float:
    return math.exp((DF_GRID.first + i) * DF_GRID.step)


__all__ = ["SpectrumOptions", "LineSpectrum", "HALO"]
