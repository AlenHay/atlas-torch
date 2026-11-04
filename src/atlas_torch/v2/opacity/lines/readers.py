"""Readers for the BasicATLAS / Kurucz line lists, restricted to a wavelength range.

The lists are sorted by wavelength and up to GBs long: each reader bisects the file for the first line of
the range and parses only the block of lines inside it (bounded memory). Formats:
  gfall (BasicATLAS.dat, gfall*.dat)  rgfalllinesnew.for FORMAT 140 (one line per transition)
  molecular .asc                      rmolecasc.for FORMAT 1111
  h2ofastfix.bin                      rh2ofast.for: 8-byte records (int32 IWL, int16 E_lower, int16 log gf)
Vacuum wavelengths are taken from the energy levels, as SYNTHE does for vacuum (IFVAC=1) runs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ....tables import table
from ...constants import MAX_STAGES, N_ELEMENTS, atomic_masses
from .linelist import LineKind, LineList

RYDBERG_INF = 109737.31          # cm^-1, rgfalllinesnew.for (n_eff of the damping estimates)
DELLIM = (100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1)  # nm, rgfalllinesnew.for DATA DELLIM


def synthe_files() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "BasicATLAS" / "data" / "synthe_files"


def dfsynthe_files() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "BasicATLAS" / "data" / "dfsynthe_files"


# ------------------------------------------------------------------------------------------ fixed-width text

def _grid(lines: list[bytes], width: int) -> np.ndarray:
    a = np.array(lines, dtype=f"S{width}").view(np.uint8).reshape(len(lines), width).copy()
    a[a == 0] = 32
    return a


def _f(a: np.ndarray, c0: int, c1: int, decimals: int = 0) -> np.ndarray:
    """Fortran Fw.d field of every row (blank = 0; no decimal point = implied decimals)."""
    fld = np.ascontiguousarray(a[:, c0:c1])
    s = fld.view(f"S{c1 - c0}").ravel().copy()
    blank = (fld == 32).all(1)
    s[blank] = b"0"
    v = s.astype(np.float64)
    if decimals:
        nodot = ~(fld == ord(".")).any(1) & ~blank
        v[nodot] /= 10.0 ** decimals
    return v


def _i(a: np.ndarray, c0: int, c1: int) -> np.ndarray:
    fld = np.ascontiguousarray(a[:, c0:c1])
    s = fld.view(f"S{c1 - c0}").ravel().copy()
    s[(fld == 32).all(1)] = b"0"
    return s.astype(np.int64)


def _s(a: np.ndarray, c0: int, c1: int) -> np.ndarray:
    return np.ascontiguousarray(a[:, c0:c1]).view(f"S{c1 - c0}").ravel()


def _key(line: bytes, width: int) -> float | None:
    try:
        return abs(float(line[:width]))
    except ValueError:
        return None


def _start_offset(fh, size: int, target: float, width: int) -> int:
    """Offset of a line start at or before the first line with |wavelength| >= target (bisection on bytes)."""
    lo, hi = 0, size
    while hi - lo > 1 << 16:
        mid = (lo + hi) // 2
        fh.seek(mid)
        fh.readline()
        pos = fh.tell()
        key = None
        while key is None and fh.tell() < hi:
            key = _key(fh.readline(), width)
        if key is None or key >= target:
            hi = mid
        else:
            lo = pos
    return lo


def _text_block(path: Path, wl_min: float, wl_max: float, width: int, chunk: int = 1 << 23):
    """Lines (bytes) of a wavelength-sorted text file whose listed |wavelength| lies in [wl_min, wl_max],
    yielded in blocks."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        fh.seek(_start_offset(fh, size, wl_min, width))
        rest = b""
        while True:
            buf = fh.read(chunk)
            if not buf and not rest:
                return
            buf = rest + buf
            if buf and fh.tell() < size:
                cut = buf.rfind(b"\n") + 1
                buf, rest = buf[:cut], buf[cut:]
            else:
                rest = b""
            lines = [ln for ln in buf.split(b"\n") if ln.strip()]
            if not lines:
                if not rest:
                    return
                continue
            wl = np.abs(np.array([_key(ln, width) or 0.0 for ln in lines]))
            keep = (wl >= wl_min) & (wl <= wl_max)
            yield [ln for ln, k in zip(lines, keep) if k]
            if wl[-1] > wl_max or fh.tell() >= size and not rest:
                return


# ------------------------------------------------------------------------------------------ damping estimates

@lru_cache(maxsize=None)
def ionisation_energies_cm() -> np.ndarray:
    """chi[Z-1, charge] (cm^-1), NIST ASD 2014 as tabulated in rgfalllinesnew.for SUBROUTINE IONPOTS
    (POTION index Z(Z+1)/2 + charge for Z <= 30, 5Z + 341 + charge above); 0 where not tabulated."""
    pot = table("rgfalllinesnew.for", "ionpots", "potion")
    chi = np.zeros((N_ELEMENTS, MAX_STAGES))
    for z in range(1, N_ELEMENTS + 1):
        for c in range(min(z, 5) if z > 30 else z):
            if c < MAX_STAGES:
                chi[z - 1, c] = pot[(z * (z + 1) // 2 if z <= 30 else 5 * z + 341) + c - 1]
    return chi


def estimate_log_gamma_stark(z, charge, e_upper, chi_cm=None) -> np.ndarray:
    """log10 Gamma_S/n_e = log10(1e-8 n_eff^5), n_eff^2 = R Z_eff^2/(E_ion - E_up) (25 if E_up >= E_ion)
    (Kurucz & Avrett 1981 p. 4). rgfalllinesnew.for's 14nov13 cap at -3 is applied to GS after GAMMAS is set,
    so it has no effect there; it is not applied here either."""
    chi = (ionisation_energies_cm() if chi_cm is None else chi_cm)[z - 1, charge]
    zeff = charge + 1.0
    de = chi - e_upper
    nsq = np.where(de > 0, RYDBERG_INF * zeff ** 2 / np.where(de > 0, de, 1.0), 25.0)
    return np.log10(1e-8 * nsq ** 2.5)


def estimate_log_gamma_vdw(z, charge, e_lower, e_upper, chi_cm=None) -> np.ndarray:
    """log10 Gamma_W/n_H at 10^4 K = log10(4.5e-9 (<r^2>_up - <r^2>_lo)^0.4), <r^2> = 2.5 (n_eff^2/Z_eff)^2
    (n_eff^2 <= 1000), iron-group sequences (21 < S < 29) <r^2>_up = (45 - S)/Z_eff, <r^2>_lo = 0
    (Kurucz & Avrett 1981 p. 4, Unsold's approximation; rgfalllinesnew.for label 138)."""
    chi = (ionisation_energies_cm() if chi_cm is None else chi_cm)[z - 1, charge]
    zeff = charge + 1.0
    de_up = chi - e_upper
    nsq_up = np.where(de_up > 0, RYDBERG_INF * zeff ** 2 / np.where(de_up > 0, de_up, 1.0), 25.0)
    nsq_lo = RYDBERG_INF * zeff ** 2 / (chi - e_lower)
    rsq_up = 2.5 * (np.minimum(nsq_up, 1000.0) / zeff) ** 2
    rsq_lo = 2.5 * (np.minimum(nsq_lo, 1000.0) / zeff) ** 2
    seq = z - charge
    iron = (seq > 20) & (seq < 29)
    rsq_up = np.where(iron, (45.0 - seq) / zeff, rsq_up)
    rsq_lo = np.where(iron, 0.0, rsq_lo)
    rsq_up = np.where(rsq_up < rsq_lo, 2.0 * rsq_lo, rsq_up)
    return np.log10(4.5e-9 * (rsq_up - rsq_lo) ** 0.4)


def classical_log_gamma_rad(wavelength_nm) -> np.ndarray:
    """log10 Gamma_R = log10(2.223e13 / lambda^2), lambda in nm (Kurucz & Avrett 1981 p. 3)."""
    return np.log10(2.223e13 / np.asarray(wavelength_nm) ** 2)


# ------------------------------------------------------------------------------------------ gfall

def _standard_mass(z: np.ndarray) -> np.ndarray:
    return atomic_masses().numpy()[np.clip(z, 1, 99) - 1]


def parse_gfall(lines: list[bytes]) -> tuple[LineList, np.ndarray]:
    """gfall records -> (LineList, LINESIZE digits). Coronal-approximation lines (AUTO='COR') are dropped;
    lines whose upper level is a continuum (LABELP='CONTINUUM') are not supported."""
    if not lines:
        return LineList.empty(), np.zeros(0, np.int64)
    a = _grid(lines, 166)
    wl_listed = np.abs(_f(a, 0, 11, 4))
    log_gf = _f(a, 11, 18, 3)
    code = _f(a, 18, 24, 2)
    e1, e2 = _f(a, 24, 36, 3), _f(a, 52, 64, 3)
    labelp = _s(a, 70, 80)
    gr, gs, gw = _f(a, 80, 86, 2), _f(a, 86, 92, 2), _f(a, 92, 98, 2)
    nblo, nbup = np.abs(_i(a, 102, 104)), np.abs(_i(a, 104, 106))
    iso1, x1, iso2, x2 = _i(a, 106, 109), _f(a, 109, 115, 3), _i(a, 115, 118), _f(a, 118, 124, 3)
    hfs_lo, hfs_up = _i(a, 124, 129), _i(a, 129, 134)
    linesize = _i(a, 140, 141)
    auto = _s(a, 141, 144)
    isoshift = _i(a, 154, 160)
    alpha = _f(a, 160, 166, 3)
    if (labelp == b"CONTINUUM ").any():
        raise NotImplementedError("gfall lines to a continuum (merged-continuum pseudo-lines)")
    keep = auto != b"COR"
    e_lo_shift = np.abs(e1) + 0.001 * hfs_lo
    e_up_shift = np.abs(e2) + 0.001 * hfs_up
    wl = 1e7 / np.abs(e_up_shift - e_lo_shift) + 1e-4 * isoshift
    e_lower = np.minimum(np.abs(e1), np.abs(e2))
    e_upper = np.maximum(np.abs(e1), np.abs(e2))
    z = np.floor(code + 1e-6).astype(np.int64)
    charge = np.round((code - z) * 100.0).astype(np.int64)
    atom = code < 100.0
    if not atom.all():
        raise NotImplementedError("molecular species in a gfall-format list")
    kind = np.full(len(a), int(LineKind.ATOMIC))
    levels = (nblo > 0) & (nbup > 0)            # H and He lines without level numbers are ordinary lines
    kind[(z == 1) & levels] = int(LineKind.HYDROGEN)
    kind[(z == 2) & (charge == 0) & levels] = int(LineKind.HELIUM_I)
    kind[(z == 2) & (charge == 1) & levels] = int(LineKind.HELIUM_II)
    kind[auto == b"AUT"] = int(LineKind.AUTOIONIZING)
    special = (kind == int(LineKind.AUTOIONIZING))
    lgr = np.where(gr == 0.0, classical_log_gamma_rad(wl), gr)
    lgs = np.where(gs == 0.0, estimate_log_gamma_stark(np.clip(z, 1, 99), charge, e_upper), gs)
    lgw = np.where(gw == 0.0, estimate_log_gamma_vdw(np.clip(z, 1, 99), charge, e_lower, e_upper), gw)
    # autoionizing lines carry the Fano parameters q-like A (GS, negative if GS > 0) and B (GW) linearly
    lgs = np.where(special, gs, lgs)
    lgw = np.where(special, gw, lgw)
    mass = np.where(iso1 > 0, iso1.astype(np.float64), _standard_mass(z))
    series = np.where((iso1 == 0) & (iso2 > 0) & (nblo + nbup > 0), iso2, 0)
    out = LineList.from_numpy(
        wavelength=wl, species=code, log_gf=log_gf, log_isotope=x1 + x2, e_lower=e_lower, log_gamma_rad=lgr,
        log_gamma_stark=lgs, log_gamma_vdw=lgw, vdw_alpha=alpha, mass=mass, kind=kind, n_lower=nblo,
        n_upper=nbup, series_limit=series, wing_class=linesize).select(np.flatnonzero(keep))
    return out, linesize[keep]


def read_gfall(path: str | Path, wl_min: float, wl_max: float) -> LineList:
    """Lines of a gfall-format list with vacuum wavelength in [wl_min, wl_max] nm."""
    slack = 1.0 + 4e-4 * wl_max                 # listed air wavelengths are shorter than vacuum
    parts = [parse_gfall(b)[0] for b in _text_block(Path(path), max(wl_min - slack, 0.0), wl_max + slack, 11)]
    return LineList.concat(parts).window(wl_min, wl_max)


def read_gfall_kurucz(path: str | Path, wl_begin: float, wl_end: float) -> LineList:
    """Lines rgfalllinesnew.for passes to SYNTHE for the window [wl_begin, wl_end]: within DELLIM(LIM) x
    max(1, wl_begin/500) nm of the window, LIM = min(8 - LINESIZE, 7) (1 for hydrogen), reading stopped at the
    first line (file order) beyond its own red limit or beyond wl_end + 100 nm."""
    fac = max(1.0, wl_begin / 500.0)
    blocks = _text_block(Path(path), max(wl_begin - DELLIM[0] * fac - 2.0, 0.0), wl_end + DELLIM[0] + 2.0, 11)
    parts = []
    for b in blocks:
        ll, size = parse_gfall(b)
        lim = np.minimum(8 - size, 7)
        lim = np.where(ll.species.numpy() == 1.0, 1, lim)
        dl = np.asarray(DELLIM)[lim - 1] * fac
        wl = ll.wavelength.numpy()
        stop = (wl > wl_end + DELLIM[0]) | (wl > wl_end + dl)
        n = int(np.argmax(stop)) if stop.any() else len(wl)
        keep = np.arange(len(wl)) < n
        parts.append(ll.select(np.flatnonzero(keep & (wl >= wl_begin - dl))))
        if stop.any():
            break
    return LineList.concat(parts)


# ------------------------------------------------------------------------------------------ molecules (.asc)

# log10 isotopic fractions per (Z, A) (the values rmolecasc.for adds to log gf); 1H2 kept at 0
ISOTOPE_LOG_FRACTION = {
    (1, 1): 0.0, (1, 2): -4.469, (6, 12): -0.005, (6, 13): -1.955, (7, 14): -0.002, (7, 15): -2.444,
    (8, 16): -0.001, (8, 17): -3.398, (8, 18): -2.690, (12, 24): -0.105, (12, 25): -0.996, (12, 26): -0.947,
    (14, 28): -0.035, (14, 29): -1.328, (14, 30): -1.510, (22, 46): -1.101, (22, 47): -1.138, (22, 48): -0.131,
    (22, 49): -1.259, (22, 50): -1.272}

# (code, ISO field) -> isotopes {Z: A} of the BasicATLAS molecular lists (rmolecasc.for computed GO TO)
ISOTOPOLOGUES = {
    (101, 1): ((1, 1), (1, 1)), (101, 2): ((1, 1), (1, 2)),
    (106, 12): ((1, 1), (6, 12)), (106, 13): ((1, 1), (6, 13)),
    (107, 14): ((1, 1), (7, 14)), (107, 15): ((1, 1), (7, 15)),
    (108, 16): ((1, 1), (8, 16)), (108, 18): ((1, 1), (8, 18)),
    (112, 24): ((1, 1), (12, 24)), (112, 25): ((1, 1), (12, 25)), (112, 26): ((1, 1), (12, 26)),
    (114, 28): ((1, 1), (14, 28)), (114, 29): ((1, 1), (14, 29)), (114, 30): ((1, 1), (14, 30)),
    (606, 12): ((6, 12), (6, 12)), (606, 13): ((6, 12), (6, 13)), (606, 33): ((6, 13), (6, 13)),
    (607, 12): ((6, 12), (7, 14)), (607, 13): ((6, 13), (7, 14)), (607, 15): ((6, 12), (7, 15)),
    (608, 12): ((6, 12), (8, 16)), (608, 13): ((6, 13), (8, 16)), (608, 17): ((6, 12), (8, 17)),
    (608, 18): ((6, 12), (8, 18)),
    (814, 28): ((14, 28), (8, 16)), (814, 29): ((14, 29), (8, 16)), (814, 30): ((14, 30), (8, 16)),
    (814, 18): ((14, 28), (8, 18)),
    (822, 46): ((22, 46), (8, 16)), (822, 47): ((22, 47), (8, 16)), (822, 48): ((22, 48), (8, 16)),
    (822, 49): ((22, 49), (8, 16)), (822, 50): ((22, 50), (8, 16))}

# rmolecasc.for X1 + X2 where they differ from ISOTOPE_LOG_FRACTION sums
KURUCZ_ISOTOPOLOGUE_LOG = {(101, 1): -5.0, (114, 29): -1.331, (114, 30): -1.516, (822, 46): -1.101,
                           (822, 47): -1.138, (822, 48): -0.131, (822, 49): -1.259, (822, 50): -1.272}


def isotopologue(code: int, iso: int, kurucz: bool = False) -> tuple[float, float]:
    """(log10 isotopologue fraction, mass in amu) of molecule `code` with list ISO number `iso`."""
    atoms = ISOTOPOLOGUES[(code, iso)]
    if kurucz and (code, iso) in KURUCZ_ISOTOPOLOGUE_LOG:
        logf = KURUCZ_ISOTOPOLOGUE_LOG[(code, iso)]
    else:
        logf = sum(ISOTOPE_LOG_FRACTION[a] for a in atoms)
    return logf, float(sum(a for _, a in atoms))


def parse_molecular_asc(lines: list[bytes], kurucz_isotopes: bool = False) -> LineList:
    """rmolecasc.for FORMAT(F10.4,F7.3,F5.1,F10.3,F5.1,F11.3,I4,A8,A8,I2,I4); Gamma_R = 10^(LOGGR/100);
    Gamma_S/n_e = 3e-5 and Gamma_W/n_H = 1e-7 (3e-8 and 1e-8 for transitions to an X state)."""
    if not lines:
        return LineList.empty()
    a = _grid(lines, 74)
    log_gf = _f(a, 10, 17, 3)
    e1, e2 = _f(a, 22, 32, 3), _f(a, 37, 48, 3)
    code = _i(a, 48, 52)
    labelp = _s(a, 60, 68)
    iso = _i(a, 68, 70)
    loggr = _i(a, 70, 74)
    wl = 1e7 / np.abs(np.abs(e2) - np.abs(e1))
    keys = sorted(set(zip(code.tolist(), iso.tolist())))
    lf = np.zeros(len(a))
    mass = np.zeros(len(a))
    for c, i in keys:
        m = (code == c) & (iso == i)
        lf[m], mass[m] = isotopologue(c, i, kurucz_isotopes)
    x = np.char.startswith(labelp, b"X")
    return LineList.from_numpy(
        wavelength=wl, species=code.astype(np.float64), log_gf=log_gf, log_isotope=lf,
        e_lower=np.minimum(np.abs(e1), np.abs(e2)), log_gamma_rad=0.01 * loggr,
        log_gamma_stark=np.where(x, np.log10(3e-8), np.log10(3e-5)),
        log_gamma_vdw=np.where(x, -8.0, -7.0), mass=mass, kind=np.full(len(a), int(LineKind.MOLECULAR)))


def read_molecular_asc(path: str | Path, wl_min: float, wl_max: float, kurucz_isotopes: bool = False,
                       stop_at_first_above: bool = False) -> LineList:
    """Lines of a molecular .asc list with vacuum wavelength (from the levels) in [wl_min, wl_max] nm.
    stop_at_first_above: end at the first line (file order) above wl_max, as rmolecasc.for does (the lists
    are sorted by listed, not by level-derived, wavelength)."""
    slack = 1.0 + 4e-4 * wl_max
    parts = []
    for b in _text_block(Path(path), max(wl_min - slack, 0.0), wl_max + slack, 10):
        ll = parse_molecular_asc(b, kurucz_isotopes)
        if stop_at_first_above:
            above = (ll.wavelength > wl_max).nonzero()
            if len(above):
                parts.append(ll.select(slice(0, int(above[0, 0]))))
                break
        parts.append(ll)
    return LineList.concat(parts).window(wl_min, wl_max)


# ------------------------------------------------------------------------------------------ H2O (binary)

H2O_RECORD = np.dtype([("iwl", "<i4"), ("elo", "<i2"), ("gflog", "<i2")])
H2O_STEP = np.log1p(1.0 / 2_000_000.0)
# rh2ofast.for DATA XISO (linear fractions) for H2 16O, H2 17O, H2 18O, HDO and their masses
H2O_FRACTION = np.array([0.9976, 0.0004, 0.0020, 0.00001])
H2O_MASS = np.array([18.0, 19.0, 20.0, 19.0])


def read_h2o(path: str | Path | None, wl_min: float, wl_max: float) -> LineList:
    """Partridge & Schwenke H2O lines from h2ofastfix.bin in [wl_min, wl_max] nm (vacuum). Record fields:
    lambda = exp(IWL ln(1 + 1/2e6)) nm; E_lower = |IELO| cm^-1; log gf = (|IGFLOG| - 16384)/1000; isotopologue
    from the signs (rh2ofast.for label 19). Gamma_R = 2.223e10/lambda^2, Gamma_W/n_H = 1e-7, no Stark term."""
    path = Path(path) if path else dfsynthe_files() / "h2ofastfix.bin"
    rec = np.memmap(path, dtype=H2O_RECORD, mode="r")

    def first_at_least(target):
        lo, hi = 0, len(rec)
        while lo < hi:
            mid = (lo + hi) // 2
            if rec[mid]["iwl"] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    lo, hi = first_at_least(np.log(wl_min) / H2O_STEP), first_at_least(np.log(wl_max) / H2O_STEP + 1.0)
    r = np.array(rec[lo:hi])
    del rec
    wl = np.exp(r["iwl"] * H2O_STEP)
    elo, gfl = r["elo"].astype(np.int64), r["gflog"].astype(np.int64)
    iso = np.where((elo > 0) & (gfl > 0), 0, np.where(elo > 0, 1, np.where(gfl > 0, 2, 3)))
    n = len(r)
    return LineList.from_numpy(
        wavelength=wl, species=np.full(n, 10108.0), log_gf=(np.abs(gfl) - 16384) * 1e-3,
        log_isotope=np.log10(H2O_FRACTION[iso]), e_lower=np.abs(elo).astype(np.float64),
        log_gamma_rad=np.log10(2.223e13 * 1e-3 / wl ** 2), log_gamma_stark=np.full(n, -16.383),
        log_gamma_vdw=np.full(n, -7.0), mass=H2O_MASS[iso], kind=np.full(n, int(LineKind.MOLECULAR))
    ).window(wl_min, wl_max)


# ------------------------------------------------------------------------------------------ BasicATLAS set

MOLECULAR_LISTS = ("chmasseron_corrected.asc", "mgh_exomol.asc", "nh.asc", "ohupdate.asc", "sihax.asc", "h2.asc",
                   "h2xx.asc", "hdxx.asc", "c2ax.asc", "c2ba.asc", "c2dabrookek.asc", "c2ea.asc",
                   "cnaxbrookek.asc", "cnbxbrookek.asc", "cnxx12brooke.asc", "coax.asc", "coxx.asc", "sioax.asc",
                   "sioex.asc", "sioxx.asc", "tiototo.asc")


@dataclass(frozen=True)
class LineSelection:
    """Wavelength margins around a synthesis window [wl_begin, wl_end] within which lines are read.

    "kurucz": the SYNTHE readers' rules (rgfalllinesnew.for DELLIM by LINESIZE, rmolecasc.for
    [wl_begin - 0.1, wl_end + 1], rh2ofast.for [wl_begin - 1, wl_end + 1]).
    "wide": every atomic line within atomic_margin x max(1, wl_begin/500) nm, molecules within molecular_margin;
    which of them contribute is decided by the opacity cutoff alone.
    """
    rule: str = "wide"
    atomic_margin: float = 100.0
    molecular_margin: float = 1.0


def read_basicatlas_lines(wl_begin: float, wl_end: float, selection: LineSelection = LineSelection(),
                          atomic: str | Path | None = None, molecular=MOLECULAR_LISTS, h2o: bool = True,
                          kurucz_isotopes: bool = False, directory: str | Path | None = None) -> LineList:
    """BasicATLAS synthe() line set for a window: the atomic list (default BasicATLAS.dat), the 21 molecular
    .asc lists and h2ofastfix.bin (templates.synthe_control order), sorted by wavelength."""
    d = Path(directory) if directory else synthe_files()
    atomic = Path(atomic) if atomic else d / "BasicATLAS.dat"
    parts = []
    if selection.rule == "kurucz":
        parts.append(read_gfall_kurucz(atomic, wl_begin, wl_end))
        mol_lo, mol_hi, h2o_lo = wl_begin - 0.1, wl_end + 1.0, wl_begin - 1.0
    elif selection.rule == "wide":
        m = selection.atomic_margin * max(1.0, wl_begin / 500.0)
        parts.append(read_gfall(atomic, max(wl_begin - m, 1.0), wl_end + m))
        mol_lo, mol_hi = wl_begin - selection.molecular_margin, wl_end + selection.molecular_margin
        h2o_lo = mol_lo
    else:
        raise ValueError("selection.rule: 'kurucz' | 'wide'")
    for name in molecular:
        parts.append(read_molecular_asc(d / name, mol_lo, mol_hi, kurucz_isotopes, selection.rule == "kurucz"))
    if h2o:
        parts.append(read_h2o(None, h2o_lo, mol_hi))
    out = LineList.concat(parts)
    del parts
    return out.sorted()
