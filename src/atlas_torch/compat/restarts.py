"""Initial temperature profiles ("restarts") for ATLAS runs, compatible with BasicATLAS's ``restarts`` module.

The restart grid is an HDF5 file in the BasicATLAS checkout: ``restarts/light.h5`` (full grid) or, failing
that, ``restarts/restarts.h5`` (reduced grid). Datasets ``rhox`` (..., 72, 1) and ``structure`` (..., 72, 2)
over the 5 parameter axes (teff, logg, zscale, alpha, carbon), and ``header``: a pickled dict with the axes,
``solar`` (log-eps, H first), ``helium``/``pradk`` per grid model, ``columns``/``units`` and ``carbon_map``
({(zscale, logg): [C/M] - carbon}).

Restart files are written in the ATLAS ``output_summary.out`` card format. ATLAS only uses the temperature
as a function of Rosseland optical depth from them, so the restart maps T onto the standard 72-point
tau grid and fills in a consistent (rhox, kappa=1) pair; unused fields are NaN.
"""
import os
import pickle
import re

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from atlas_torch.compat.settings import python_path

__all__ = ["load_header", "interpolate_structure", "generate_model", "prepare_restart", "model_card", "TAU_STD"]

# Rosseland optical depths of the 72 layers fixed by the SCALE card of the ATLAS control file
TAU_STD = np.logspace(-6.875, 2.0, 72)
PARAM_KEYS = ("teff", "logg", "zscale", "alpha", "carbon")


def model_card(teff, logg, scale, abundances, rows, pradk):
    """Text of a model in the restart-grid card format (no trailing newline).

    ``scale`` is the linear ABUNDANCE SCALE, ``abundances`` the 99 ATLAS abundances (Z = 1..99), ``rows`` the
    already formatted structure lines."""
    lines = [
        "TEFF {:6.0f}.  GRAVITY{:8.5f}     ".format(teff, logg),
        "{:<80s}".format("TITLE   B a s i c A T L A S"),
        " OPACITY IFOP" + " 1" * 13 + " 0 1" + " 0" * 5,
        " CONVECTION ON   1.25 TURBULENCE OFF" + "  0.00" * 4,
        "ABUNDANCE SCALE {:9.5f} ABUNDANCE CHANGE 1{:8.5f} 2{:8.5f}".format(scale, abundances[0], abundances[1]),
    ]
    for first in range(3, 100, 6):
        lines.append(" ABUNDANCE CHANGE " + " ".join(
            "{:2d}{:7.2f}".format(z, abundances[z - 1]) for z in range(first, min(first + 6, 100))))
    lines.append("READ DECK6 72 RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB,         FLXRAD,VCONV,VELSND")
    lines.extend(rows)
    lines.append("PRADK{:11.4E}".format(pradk))
    lines.append("BEGIN" + " " * 20 + "ITERATION  15 COMPLETED")
    return "\n".join(lines)


def load_header():
    """Header of the restart grid (light.h5, else restarts.h5), with ``grid`` = path of the HDF5 file and
    ``carbon_map`` turned into an interpolator over (zscale, logg)."""
    grid = os.path.join(python_path, "restarts", "light.h5")
    if not os.path.isfile(grid):
        grid = os.path.join(python_path, "restarts", "restarts.h5")
    with h5py.File(grid, "r") as f:
        header = pickle.loads(bytes(f["header"][()]))     # trusted data file of the BasicATLAS checkout
    header["grid"] = os.path.realpath(grid)
    cmap = header["carbon_map"]
    zs = sorted({k[0] for k in cmap})
    gs = sorted({k[1] for k in cmap})
    table = np.array([[cmap[(z, g)] for g in gs] for z in zs])
    header["carbon_map"] = RegularGridInterpolator([zs, gs], table)
    return header


def interpolate_structure(params, header=False):
    """Multilinear interpolation of the restart grid at ``params`` (keys teff, logg, zscale, alpha, carbon).

    Returns (structure (72 x [rhox, T, P]), params, header, helium (absolute log-eps He), pradk)."""
    if isinstance(header, bool):
        header = load_header()
    for k in PARAM_KEYS:
        if params[k] > np.max(header[k]) or params[k] < np.min(header[k]):
            raise ValueError("Requested restart parameters exceed grid bounds along axis {}".format(k))

    # the grid cell containing the point: [lo, lo + 1] along every axis (the last cell at the upper edge)
    lows = []
    for k in PARAM_KEYS:
        axis, v = header[k], params[k]
        lows.append(len(axis) - 2 if v == axis[-1] else int(np.nonzero(axis > v)[0][0]) - 1)
    cell = tuple(slice(lo, lo + 2) for lo in lows)
    with h5py.File(header["grid"], "r") as f:
        corners = np.concatenate((f["rhox"][cell], f["structure"][cell]), axis=-1)
    he, pk = header["helium"][cell], header["pradk"][cell]

    live = [i for i, n in enumerate(corners.shape[:-2]) if n != 1]
    axes = [[header[PARAM_KEYS[i]][lows[i]], header[PARAM_KEYS[i]][lows[i] + 1]] for i in live]
    point = np.array([params[PARAM_KEYS[i]] for i in live])

    def interp(values):
        return RegularGridInterpolator(axes, np.squeeze(values))(point)[0]

    return interp(corners), params, header, interp(he), interp(pk)


def generate_model(structure, params, header, helium, pradk):
    """Model text in the ATLAS card format from the output of :func:`interpolate_structure` (params also need
    teff, logg; alpha is applied to O, Ne, Mg, Si, S, Ar, Ca, Ti; carbon is [C/M] relative to the carbon map)."""
    ab = np.array(header["solar"])
    ab[1] = helium
    ab[2:] += params["zscale"]
    ab[[7, 9, 11, 13, 15, 17, 19, 21]] += params["alpha"]
    ab[5] += params["carbon"] + header["carbon_map"]([params["zscale"], params["logg"]])[0]
    ab = 10 ** ab / np.sum(10 ** ab)
    ab[2:] = np.log10(ab[2:]) - params["zscale"]
    ab[ab < -20] = -20
    rows = [("{:15.8E}{:9.1f}{:10.3E}" + "       NaN" * 8).format(*row) for row in structure]
    return model_card(params["teff"], params["logg"], 10 ** params["zscale"], ab, rows, pradk)


def _nearest(axis, value):
    return axis[np.argmin(np.abs(value - axis))]


def prepare_restart(restart, save_to, settings, read_structure=None, meta=None):
    """Write the initial model for an ATLAS run to ``save_to`` and return a description of it.

    ``restart``: "auto" (nearest model of the restart grid in teff, logg, [Fe/H], [Mg/Fe] and C/O), "gray"/"grey"
    (Eddington gray atmosphere at settings.teff), a previous ATLAS run directory, or an output_summary.out-style
    file. ``read_structure``/``meta`` are the run readers of the API module (needed for run directories)."""
    if restart == "grey" or restart == "gray":
        teff = settings.teff
        tau = TAU_STD
        temp = teff * ((3 / 4) * (TAU_STD + (2 / 3))) ** (1 / 4)
        message = "Using gray model with teff={} as restart".format(teff)

    elif restart == "auto":
        header = load_header()
        enh = settings.abun
        teff = _nearest(header["teff"], settings.teff)
        logg = _nearest(header["logg"], settings.logg)
        zscale = _nearest(header["zscale"], settings.zscale + enh["Fe"] if "Fe" in enh else settings.zscale)
        want_alpha = 0.0
        if "Mg" in enh:
            want_alpha += enh["Mg"]
        if "Fe" in enh:
            want_alpha -= enh["Fe"]
        alpha = _nearest(header["alpha"], want_alpha)
        want_co = 0.0
        if "C" in enh:
            want_co += enh["C"]
        if "O" in enh:
            want_co -= enh["O"]
        c_offset = header["carbon_map"]([zscale, logg])[0]
        carbon = header["carbon"][np.argmin(np.abs(want_co - (header["carbon"] + c_offset - alpha)))]
        point = {"teff": teff, "logg": logg, "zscale": zscale, "alpha": alpha, "carbon": carbon}
        structure, _, header, _, _ = interpolate_structure(point, header=header)
        temp = structure[:, header["columns"].index("Temperature")]
        tau = TAU_STD
        message = ("Chose closest model from the restart grid with (Teff,log(g),[M/H],[a/M],[C/M])=({},{},{},{},{})"
                   .format(teff, logg, zscale, alpha, carbon + header["carbon_map"]([zscale, logg])[0]))

    elif os.path.isdir(restart):
        structure, _ = read_structure(restart)
        teff = meta(restart)["teff"]
        tau = structure["rosseland_optical_depth"]
        temp = structure["temperature"]
        message = "Using {} as restart".format(restart)

    elif os.path.isfile(restart):
        with open(restart) as f:
            text = f.read()
        teffs = re.findall("TEFF *([0-9.eE-]+)", text)
        decks = re.findall("FLX...,VCONV,VELSND(.+)PRADK", text, re.DOTALL)
        if len(decks) != 1 or len(teffs) != 1:
            raise ValueError("{} is not a valid restart file".format(restart))
        teff = float(teffs[0])
        # these files carry rhox and ABROSS, not tau: integrate d(tau) = kappa d(rhox) (trapezoid)
        rhox, temp, kappa = np.loadtxt(decks[0].split("\n"), unpack=True, usecols=[0, 1, 4])
        dtau = (kappa[1:] + kappa[:-1]) / 2.0 * np.diff(rhox)
        tau = np.cumsum(np.r_[rhox[0] * kappa[0], dtau])
        message = "Using {} as restart".format(restart)

    else:
        raise ValueError("{} is an invalid choice of restart".format(restart))

    temp = np.interp(TAU_STD, tau, temp)
    # unit opacity everywhere; the rhox that reproduces TAU_STD with it (trapezoid rule inverted)
    kappa = np.ones(np.shape(TAU_STD))
    rhox = np.cumsum(np.r_[TAU_STD[0] / kappa[0], np.diff(TAU_STD) / (kappa[1:] + kappa[:-1]) * 2.0])
    nan = np.nan
    rows = [("{:15.8E}{:9.1f}" + "{:10.3E}" * 8).format(rhox[i], temp[i], nan, nan, kappa[i], 0.0, nan, nan, nan, nan)
            for i in range(len(TAU_STD))]
    with open(save_to, "w") as f:
        f.write(model_card(teff, nan, nan, [nan] * 99, rows, 0.0))
    return message
