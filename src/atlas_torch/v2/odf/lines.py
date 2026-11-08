"""Line data for the ODF: DFSYNTHE's packed lists decoded to physical units, their species (the NELION slots of
XNFDF), the NLTE list (hydrogen, autoionizing, series lines and merged continua; XLINOP), and the line-centre
opacity used for selection (Castelli 2005, MSAIS 8, 34, p. 35: lines weaker than 10^-3 of the continuum at their
centre are skipped; dfsynthe.for SELECTLINES and MAIN)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from ...tables import table
from ..constants import CODATA2018, PhysicalConstants, atomic_masses, decimal32
from ..io.dfsynthe_lines import DF_FILES, LINE_DTYPE, default_dir, index_range, open_df_list, read_nlte
from ..opacity.lines.linelist import LineKind, LineList
from ..opacity.lines.opacity import doppler_width, line_strength
from .grid import DF_GRID, DF_WL_BEGIN

SQRT_PI = math.sqrt(math.pi)
C_NM = 2.99792458e17
TABLOG_ZERO = 16384               # packed log10 value: 10**((i - 16384) * 0.001) (SELECTLINES TABLOG)
CGF_KURUCZ = 0.026538 / 1.77245   # (pi e^2/mc)/sqrt(pi) of the packed lists (repacknlte.for, SELECTLINES)

# temperature range (K) in which DFSYNTHE reads each list (SELECTLINES labels 669, 769, 869, 969)
FILE_T_LIMITS = {"lowlinesdf.bin": (0.0, math.inf), "highlinesdf.bin": (20000.0, math.inf),
                 "diatomicsdf.bin": (0.0, 10000.0), "tiolinesdf.bin": (0.0, 5000.0),
                 "h2olinesdf.bin": (0.0, 5000.0)}


@lru_cache(maxsize=None)
def nelion_species() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the 594 NELION slots (XNFDF loops 260-265): species code (0 = empty), mass (amu) of the Doppler width,
    molecule flag. Slot (Z-1)*6 + s holds element Z in stage s (s <= 6); slots 5 of 'elements' 50-58, 60-68, 70-78,
    80-88 hold stages 7-10 of Ca-Ni (loop 250); slot 6 of 40-99 the molecules IDMOL(1..60) (MOMASS)."""
    code = np.zeros(594)
    mass = np.zeros(594)
    mol = np.zeros(594, dtype=bool)
    am = atomic_masses().numpy()
    idmol = decimal32(table("xnfdf.for", "xnfdf", "idmol"))
    momass = decimal32(table("xnfdf.for", "xnfdf", "momass"))
    for z in range(1, 100):
        for s in range(1, 7):
            i = (z - 1) * 6 + s - 1
            code[i], mass[i] = z + (s - 1) / 100.0, am[z - 1]
    for z in range(20, 29):
        for k, stage in ((30, 7), (40, 8), (50, 9), (60, 10)):
            i = (z + k - 1) * 6 + 4
            code[i], mass[i] = z + (stage - 1) / 100.0, am[z - 1]
    for z in range(40, 100):
        i = (z - 1) * 6 + 5
        code[i], mass[i], mol[i] = idmol[z - 40], momass[z - 40], True
    for z in range(29, 100):                      # stages above 3 (POPS(Z.02)) and slot 4 stay empty
        for s in (4, 5):
            i = (z - 1) * 6 + s - 1
            if code[i] == z + (s - 1) / 100.0:
                code[i] = 0.0
    return code, mass, mol


@dataclass(frozen=True)
class DfLines:
    """Decoded lines: center (n,) 0-based DF_GRID index; slot (n,) NELION - 1; wavelength (n,) nm; log_gf;
    e_lower (cm^-1); gamma_rad (s^-1), gamma_stark (s^-1 cm^3), gamma_vdw (s^-1 cm^3)."""
    center: torch.Tensor
    slot: torch.Tensor
    wavelength: torch.Tensor
    log_gf: torch.Tensor
    e_lower: torch.Tensor
    gamma_rad: torch.Tensor
    gamma_stark: torch.Tensor
    gamma_vdw: torch.Tensor

    def __len__(self) -> int:
        return self.center.shape[0]

    def select(self, idx) -> "DfLines":
        return DfLines(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))

    @staticmethod
    def cat(parts: list["DfLines"]) -> "DfLines":
        return DfLines(*(torch.cat([getattr(p, f) for p in parts]) for f in DfLines.__dataclass_fields__))


def decode(rec: np.ndarray, h2o: bool = False, device=None) -> DfLines:
    """Packed entries -> physical units: lambda = 8.97666 (1 + 1/R)^(IWL-1) nm (the list wavelength rounded to the
    grid); gf, Gamma = 10**((I - 16384)/1000); E_lo likewise, or I itself in cm^-1 for H2O (SELECTLINES TABBOLT)."""
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=device)     # noqa: E731
    iwl = np.asarray(rec["iwl"], dtype=np.int64)
    lg = lambda f: (np.asarray(rec[f], dtype=np.float64) - TABLOG_ZERO) * 1e-3          # noqa: E731
    e = np.asarray(rec["ielo"], dtype=np.float64) if h2o else 10.0 ** lg("ielo")
    return DfLines(torch.as_tensor(iwl - 1, device=device), torch.as_tensor(np.asarray(rec["nelion"], np.int64) - 1,
                                                                             device=device),
                   t(DF_WL_BEGIN * np.exp((iwl - 1) * math.log1p(1.0 / DF_GRID.resolution))), t(lg("igflog")),
                   t(e), t(10.0 ** lg("igr")), t(10.0 ** lg("igs")), t(10.0 ** lg("igw")))


class DfLineSource:
    """The *df.bin lists as memory maps, read by grid-index range (the lists are sorted by grid index).
    temperature_limits: read each list only in DFSYNTHE's temperature range (FILE_T_LIMITS)."""

    def __init__(self, directory=None, files=tuple(DF_FILES), temperature_limits: bool = True):
        d = default_dir() if directory is None else directory
        self.maps = {f: open_df_list(d / f) for f in files if (d / f).exists()}
        self.temperature_limits = temperature_limits

    def files_for(self, T: float) -> list[str]:
        if not self.temperature_limits:
            return list(self.maps)
        return [f for f in self.maps if FILE_T_LIMITS[f][0] <= T < FILE_T_LIMITS[f][1]]

    def iter(self, lo: int, hi: int, T: float, chunk: int = 20000, device=None):
        """Lines with lo <= center <= hi (0-based grid indices), decoded `chunk` entries at a time (memory bound:
        a low-temperature segment holds ~10^7 lines)."""
        for f in self.files_for(T):
            m = self.maps[f]
            r = index_range(m, lo + 1, hi + 1)
            for a in range(r.start, r.stop, chunk):
                rec = np.asarray(m[a:min(a + chunk, r.stop)])
                rec = rec[rec["iwl"] > 0]
                if len(rec):
                    yield decode(rec, h2o=f.startswith("h2o"), device=device)

    def read(self, lo: int, hi: int, T: float, device=None) -> DfLines:
        """All lines with lo <= center <= hi (0-based grid indices)."""
        parts = list(self.iter(lo, hi, T, 1 << 22, device))
        return DfLines.cat(parts) if parts else decode(np.zeros(0, dtype=LINE_DTYPE), device=device)

    def count(self, T: float) -> int:
        return sum(len(self.maps[f]) for f in self.files_for(T))


def center_opacity(lines: DfLines, gas, const: PhysicalConstants = CODATA2018):
    """Line-centre opacity kappa_0 = (pi e^2/mc) gf (n/U)/rho e^(-E hc/kT) (1 - e^(-h nu/kT)) / (sqrt(pi) Delta nu_D)
    (cm^2 g^-1, H(a,0) = 1 normalisation; K&A 1981 p. 4), damping a = Gamma/(4 pi Delta nu_D) and Delta nu_D/nu,
    each (L, n) for the layers of gas (OdfGas)."""
    slot = lines.slot.to(gas.T.device)
    nou = gas.n_over_u[:, slot]
    dop = gas.doppler[:, slot]
    T = gas.T[:, None]
    nu = C_NM / lines.wavelength[None, :]
    stim = -torch.expm1(-const.h * nu / (const.k * T))
    k0 = line_strength(T, gas.rho[:, None], nou, lines.log_gf[None, :], lines.e_lower[None, :], const) * stim \
        / (SQRT_PI * nu * dop)
    gam = lines.gamma_rad[None, :] + lines.gamma_stark[None, :] * gas.n_e[:, None] \
        + lines.gamma_vdw[None, :] * gas.n_pert[:, None]
    return k0, gam / (4.0 * math.pi * nu * dop), dop


def doppler_slots(T: torch.Tensor, xi2: torch.Tensor, const: PhysicalConstants = CODATA2018,
                  molecular_turbulence: bool = False) -> torch.Tensor:
    """Delta nu_D / nu per NELION slot, (L, 594): thermal + xi^2 (cm^2 s^-2) with the slot's mass; molecules
    without xi unless molecular_turbulence (XNFDF loop 265)."""
    code, mass, mol = nelion_species()
    m = torch.as_tensor(np.where(mass > 0, mass, 1.0), dtype=T.dtype, device=T.device)
    x = torch.where(torch.as_tensor(mol, device=T.device) & (not molecular_turbulence), torch.zeros_like(m), 1.0)
    return doppler_width(T[:, None], m[None, :], (xi2[:, None] * x[None, :]), const)


# ---------------------------------------------------------------------------------------------- NLTE list

def nlte_lines(records: np.ndarray | None = None) -> tuple[LineList, np.ndarray]:
    """The hydrogen (TYPE -1, -2), autoionizing (1) and other (0, 3) lines of nltelinesdf.bin as a v2 LineList
    (gf and Gamma restored from repacknlte.for's CGF = 0.026538/1.77245 gf/nu and Gamma/(4 pi nu)), and the
    merged-continuum records (TYPE > 3; SeriesContinua). XLINOP treats He I lines as other lines (label 97)."""
    r = read_nlte() if records is None else records
    typ = r["type"]
    cont = r[typ > 3]
    r = r[(typ <= 3) & (typ != 2)]
    typ = r["type"]
    code, _, _ = nelion_species()
    wl = np.asarray(r["wl"], dtype=np.float64)
    nu = C_NM / wl
    auto = typ == 1
    gf = np.where(auto, r["gf"], r["gf"] * nu / CGF_KURUCZ).astype(np.float64)
    g4 = lambda f: np.where(auto, r[f], r[f] * 4.0 * np.pi * nu).astype(np.float64)    # noqa: E731
    gr, gs, gw = g4("gammar"), g4("gammas"), g4("gammaw")
    sp = code[r["nelion"] - 1]
    kind = np.where(typ < 0, int(LineKind.HYDROGEN), np.where(auto, int(LineKind.AUTOIONIZING), int(LineKind.ATOMIC)))
    lgs = np.where(auto & (gs < 0), -np.log10(np.abs(gs)), np.log10(np.abs(gs)))       # v2 Fano sign convention
    am = atomic_masses().numpy()
    mass = np.where(typ == -2, 2.014, am[np.floor(sp).astype(int) - 1])
    t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt)          # noqa: E731
    ll = LineList(wavelength=t(wl), species=t(sp), log_gf=t(np.log10(gf)), log_isotope=t(np.zeros(len(r))),
                  e_lower=t(r["elo"]), log_gamma_rad=t(np.log10(gr)), log_gamma_stark=t(lgs),
                  log_gamma_vdw=t(np.log10(gw)), vdw_alpha=t(np.zeros(len(r))), mass=t(mass), kind=t(kind, torch.long),
                  n_lower=t(r["nblo"], torch.long), n_upper=t(r["nbup"], torch.long),
                  series_limit=t(r["ncon"], torch.long), wing_class=t(np.zeros(len(r)), torch.long))
    return ll, cont


class SeriesContinua:
    """Merged continua of series dissolving at the Inglis-Teller limit (XLINOP 'MERGED CONTINUUM'): from the
    edge lambda_0 to lambda_merge = 1/(1/lambda_0 - sigma_merge(n_e)) kappa = sigma g (n/U)/rho e^(-E hc/kT)
    (1 - e^(-hc/lambda_0 kT)) (SELECTLINES BOLTSTIM), tapering linearly to 0 at 1/(1/lambda_merge - 500 cm^-1);
    lambda_merge limited to [lambda_shift, 2 lambda_shift], lambda_shift = 1/(1/lambda_0 - R/n_last^2).
    records: TYPE > 3 entries of nltelinesdf.bin (TYPE = n_last)."""

    def __init__(self, records: np.ndarray):
        code, _, _ = nelion_species()
        f = lambda k: torch.as_tensor(np.asarray(records[k], dtype=np.float64))      # noqa: E731
        self.wl, self.sigma, self.e_lower, self.nlast = f("wl"), f("gf"), f("elo"), f("type")
        self.slot = torch.as_tensor(np.asarray(records["nelion"], dtype=np.int64) - 1)
        self.hydrogen = torch.as_tensor(code[records["nelion"] - 1] == 1.0)

    def __len__(self) -> int:
        return self.wl.shape[0]

    def __call__(self, gas, wl_grid: torch.Tensor, const: PhysicalConstants = CODATA2018) -> torch.Tensor:
        """kappa (L, N) cm^2 g^-1 on the wavelengths wl_grid (N,) nm."""
        from ..opacity.lines.opacity import merging_wavenumber
        out = torch.zeros(gas.T.shape[0], wl_grid.shape[0], dtype=torch.float64, device=wl_grid.device)
        w = wl_grid[None, :]
        hc_kt = const.hc_k / gas.T[:, None]
        for i in range(len(self)):
            w0, h = float(self.wl[i]), bool(self.hydrogen[i])
            sigma, e_lower, slot = float(self.sigma[i]), float(self.e_lower[i]), int(self.slot[i])
            if w0 > float(wl_grid[-1]):
                continue
            ryd = 109677.576 if h else 109737.312
            wshift = 1e7 / (1e7 / w0 - ryd / float(self.nlast[i]) ** 2)
            wm = 1e7 / (1e7 / w0 - merging_wavenumber(gas.n_e, h)[:, None])
            wm = torch.where(wm < 0, torch.full_like(wm, 2 * wshift), wm).clamp(wshift, 2 * wshift)
            wt = 1e7 / (1e7 / wm - 500.0)
            wt = torch.minimum(torch.where(wt < 0, 2 * wm, wt), 2 * wm)
            if float(wt.max()) < float(wl_grid[0]):
                continue
            tail = torch.where(w > wm, (wt - w) / (wt - wm), torch.ones_like(w)).clamp(min=0.0)
            k = (sigma * gas.n_over_u[:, slot][:, None] / gas.rho[:, None]
                 * torch.exp(-e_lower * hc_kt) * -torch.expm1(-1e7 / w0 * hc_kt))
            out += torch.where((w >= w0) & (w <= wt), k * tail, torch.zeros_like(tail))
        return out


__all__ = ["nelion_species", "DfLines", "decode", "DfLineSource", "FILE_T_LIMITS", "center_opacity",
           "doppler_slots", "nlte_lines", "SeriesContinua", "CGF_KURUCZ"]
