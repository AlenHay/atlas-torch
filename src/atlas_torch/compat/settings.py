"""Model settings and chemical-abundance conversions, compatible with BasicATLAS's ``settings.Settings``.

Two abundance formats are in play:

* *standard*: helium mass fraction ``Y`` (outside [0, 1] means "solar"), metallicity ``zscale`` = [M/H] in dex,
  and per-element enhancements ``abun`` = {symbol: [A/M] in dex}, relative to the solar mixture in
  ``data/solar.csv`` (columns: Z, atomic weight, log eps, error, source, symbol, name).
* *ATLAS*: a 100-entry list indexed by atomic number (entry 0 unused, ``0``). Entries 1 and 2 are the H and He
  number fractions, entries 3..99 are log10 number fractions *without* the metallicity (ATLAS applies
  ``ABUNDANCE SCALE`` = 10**zscale itself), floored at -20.

The arithmetic (operation order, numpy pairwise sums over the solar.csv element order, sequential Python sums
where the reference accumulates sequentially) is chosen so that results agree bit-for-bit with BasicATLAS:
the numbers end up in fixed-format control cards, where a last-ulp difference could flip a rounded digit.

The solar table lives in the BasicATLAS data directory (see :func:`basicatlas_dir`).
"""
import os
import pwd
from collections import OrderedDict
from functools import lru_cache

import numpy as np

__all__ = ["Settings", "basicatlas_dir", "python_path"]


def basicatlas_dir():
    """Root of the BasicATLAS checkout whose ``data/`` (and ``restarts/``) we read.

    ``$ATLAS_TORCH_BASICATLAS``, else ``$ATLAS_EXT/BasicATLAS``, else ``<login home>/atlas-ext/BasicATLAS``.
    The login home comes from the passwd database, not ``$HOME`` (the example runner repoints ``$HOME``).
    """
    p = os.environ.get("ATLAS_TORCH_BASICATLAS")
    if not p:
        ext = os.environ.get("ATLAS_EXT") or os.path.join(pwd.getpwuid(os.getuid()).pw_dir, "atlas-ext")
        p = os.path.join(ext, "BasicATLAS")
    return os.path.realpath(p)


# Same role as BasicATLAS's module-level ``python_path``: every data path is ``python_path + '/data/...'``.
python_path = basicatlas_dir()


@lru_cache(maxsize=None)
def _solar_table(path):
    """(Z, A, log_eps, symbols) from solar.csv, in file order (= atomic-number order)."""
    cols = np.loadtxt(path, usecols=[0, 1, 2], unpack=True, delimiter=",")
    sym = np.loadtxt(path, usecols=[5], unpack=True, delimiter=",", dtype=str)
    for a in (*cols, sym):
        a.setflags(write=False)
    return cols[0], cols[1], cols[2], sym


def _table():
    return _solar_table(os.path.join(python_path, "data", "solar.csv"))


class Settings:
    """Stellar parameters + chemistry of a model (``teff`` [K], ``logg`` [cgs dex], ``zscale`` = [M/H],
    ``Y`` = helium mass fraction (-0.1 = solar), ``vturb`` [km/s] (selects the ODF), ``abun`` = {el: [A/M]})."""

    def __init__(self):
        self.teff = 5770.0
        self.logg = 4.44
        self.zscale = 0.0
        self.Y = -0.1
        self.vturb = 2
        self.abun = {}

    # ---------------------------------------------------------------- metallicity with format validation
    @property
    def zscale(self):
        return self._zscale

    @zscale.setter
    def zscale(self, value):
        # ATLAS reads ABUNDANCE SCALE (= 10**zscale) through a 9-column field; the value must survive printing
        # with 6 decimals in 9 columns and still represent the requested [M/H] to 0.01 dex.
        value = float(value)
        text = "{:>9.6f}".format(10 ** value)
        if len(text) > 9:
            raise ValueError("Overflow in zscale value")
        linear = float(text)
        if linear == 0.0 or np.abs(np.log10(linear) - value) > 0.01:
            raise ValueError("Underflow in zscale value")
        self._zscale = value

    # ---------------------------------------------------------------- tables
    def abun_solar(self):
        """Solar log-eps abundances (log10 N(X)/N(H) + 12) keyed by symbol, in atomic-number order (fresh dict)."""
        _, _, logeps, sym = _table()
        return OrderedDict(zip(sym, logeps))

    # ---------------------------------------------------------------- conversions
    def abun_std_to_atlas(self, Y=-0.1, zscale=0.0, abun={}):
        """Standard (Y, [M/H], {el: [A/M]}) -> 100-entry ATLAS abundance list (see module docstring)."""
        Z, A, _, sym = _table()
        mix = self.abun_solar()
        for el in abun:
            if el not in mix:
                raise ValueError("Unknown element: {}".format(el))
            if el == "H":
                raise ValueError("Cannot set H abundance directly: it is calculated automatically from other abundances")
            if el == "He":
                raise ValueError("Cannot set He abundance directly: use helium mass fraction (Y) instead")
            mix[el] = mix[el] + abun[el]

        # number densities relative to H (metals scaled by [M/H]), in solar.csv order
        logn = np.array([mix[s] + zscale * (z > 2.0) for s, z in zip(sym, Z)])
        n = 10 ** logn
        he = Z == 2
        if 0.0 <= Y <= 1.0:
            # choose n(He) such that m(He) / (m(He) + m(rest)) = Y
            n[he] = -np.sum(n[~he] * A[~he]) * Y / (A[he][0] * Y - A[he][0])

        out = list(range(100))
        out[1] = n[Z == 1][0] / np.sum(n)
        out[2] = n[he][0] / np.sum(n)
        for z in range(3, 100):
            v = np.log10(n[Z == z][0] / np.sum(n)) - zscale
            out[z] = -20.0 if v < -20.0 else v
        if float("{:>7.5f}".format(out[1])) == 0.0:
            raise ValueError("Underflow in Y value. Hydrogen abundance cannot be zero!")
        return out

    def abun_atlas_to_std(self, elements, zscale, rounding=2):
        """ATLAS abundance list (+ [M/H]) -> {'abun': {el: [A/M]} (non-zero only), 'Y': Y, 'zscale': zscale}.

        Enhancements are rounded to ``rounding`` decimals; elements without a solar abundance must be at the
        ATLAS floor (<= -20)."""
        Z, A, _, sym = _table()
        mix = self.abun_solar()
        enh = OrderedDict()
        mass = 0.0
        for z in range(3, 100):
            el = sym[Z == z][0]
            ref = mix[el]
            if ref < -90.0 and elements[z] <= -20.0:
                d = 0.0
            elif ref < -90.0 and elements[z] > -20.0:
                raise ValueError("No solar abundance for element {}".format(el))
            else:
                d = np.round(elements[z] - np.log10(elements[1]) + 12.0 - ref, rounding)
            if d != 0.0:
                enh[el] = d
            mass += 10 ** (d + ref + zscale) * A[Z == z][0]
        m_he = elements[2] / (elements[1] / 1e12) * A[Z == 2][0]
        mass = mass + 1e12 * A[Z == 1][0] + m_he
        return {"abun": enh, "Y": m_he / mass, "zscale": zscale}

    def atlas_abun(self):
        """This model's abundances in ATLAS format."""
        return self.abun_std_to_atlas(self.Y, self.zscale, self.abun)

    def mass_fractions(self):
        """(X, Y, Z) mass fractions of hydrogen, helium and metals."""
        Z, A, _, _ = _table()
        el = self.atlas_abun()
        m_metal = 0.0
        for z in range(3, len(el)):
            m_metal += 10 ** (el[z] + self.zscale) * A[Z == z][0]
        m_h = el[1] * A[Z == 1][0]
        m_he = el[2] * A[Z == 2][0]
        total = m_metal + m_h + m_he
        return m_h / total, m_he / total, m_metal / total

    def effective_zscale(self):
        """[M/H] of the scaled-solar mixture with the same Z/X mass ratio as this model."""
        _, A, _, sym = _table()
        X, _, Zm = self.mass_fractions()
        mix = self.abun_solar()
        m_metal = 0.0
        for el in mix:
            if el != "H" and el != "He":
                m_metal += A[sym == el][0] * 10 ** mix[el]
        m_h = 10 ** mix["H"] * A[sym == "H"][0]
        return np.log10((m_h * Zm) / (m_metal * X))

    def check_ODF(self, ODF):
        """Raise ValueError unless the ODF (``meta()`` of a DFSYNTHE run) was computed for this chemistry."""
        Z, _, _, sym = _table()
        Y = self.mass_fractions()[1]
        if np.abs(ODF["zscale"] - self.zscale) > 0.01:
            raise ValueError("Incomplatible ODF: ODF calculated for zscale={} as opposed to {}".format(ODF["zscale"], self.zscale))
        if np.abs(ODF["Y"] - Y) > 0.001:
            raise ValueError("Incomplatible ODF: ODF calculated for Y={} as opposed to {}".format(ODF["Y"], Y))
        odf_atlas = self.abun_std_to_atlas(ODF["Y"], ODF["zscale"], ODF["abun"])
        for el in self.abun_solar():
            if el in ("H", "He"):
                continue
            mine = self.abun.get(el, 0.0)
            theirs = ODF["abun"].get(el, 0.0)
            # an ODF abundance can legitimately be higher than requested when it sits on the -20 ATLAS floor
            floored = odf_atlas[int(Z[sym == el][0])] == -20.0 and theirs > mine
            if np.abs(theirs - mine) > 0.01001 and not floored:
                raise ValueError("Incomplatible ODF: ODF calculated for enhancements {} as opposed to {}".format(dict(ODF["abun"]), dict(self.abun)))
