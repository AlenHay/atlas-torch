"""BasicATLAS-compatible API on top of atlas-torch.

``import atlas`` (with ``src/atlas_torch/compat`` on ``sys.path``) gives the public functions of BasicATLAS's
``atlas.py`` with the same signatures, defaults, printed messages, exceptions and run-directory layout:
:func:`atlas` (ATLAS9 model atmosphere), :func:`synthe` (SYNTHE spectrum), :func:`dfsynthe` (ODFs + Rosseland
opacities), the readers :func:`read_structure`, :func:`read_spectrum`, :func:`load_binary_spectrum`, :func:`meta`,
the checks :func:`validate_run`, :func:`atlas_converged`, and :func:`synphot`, :func:`synbeg`, :func:`bin_spec`,
blackbody helpers. ``Settings`` is re-exported from :mod:`atlas_torch.compat.settings`.

Every run writes the same shell command files as BasicATLAS (``atlas_control*.com``, ``synthe_launch.com``,
``xnfdf.com``, ``dft_*/dfp.com``, ``kappa9v*.com``, ...). They are executed by the interpreter in
:mod:`atlas_torch.compat.runners`, which turns each Fortran program invocation into a ``run_program()`` call
(torch implementation, or the stock executable as a fallback; see that module).

Data (restart grids, solar ODF, line lists, filters, solar.csv) is read from the BasicATLAS checkout given by
``$ATLAS_TORCH_BASICATLAS`` (``python_path`` below plays the role of BasicATLAS's install directory, also in the
command files).
"""
import copy
import os
import re
import threading
import time
import warnings
from datetime import datetime

import numpy as np
import scipy.constants as spc
from scipy.optimize import brentq

from atlas_torch.compat import restarts, runners
from atlas_torch.compat.settings import Settings, python_path

__all__ = ["Settings", "python_path", "notify", "cmd", "bin_spec", "blackbody", "blackbody_peak", "blackbody_dBdT",
           "blackbody_dBdT_peak", "atlas_converged", "atlas", "synbeg", "synthe", "load_binary_spectrum", "dfsynthe",
           "meta", "meta_dfsynthe", "meta_atlas", "parse_atlas_abundances", "validate_run", "read_structure",
           "read_spectrum", "synphot", "restarts", "runners"]

# DFSYNTHE grid: 57 temperatures (K, as written into the control cards) and the 5 microturbulences (km/s)
DFSYNTHE_TEMPERATURES = (
    "1995.", "2089.", "2188.", "2291.", "2399.", "2512.", "2630.", "2754.", "2884.", "3020.", "3162.", "3311.",
    "3467.", "3631.", "3802.", "3981.", "4169.", "4365.", "4571.", "4786.", "5012.", "5370.", "5754.", "6166.",
    "6607.", "7079.", "7586.", "8128.", "8710.", "9333.", "10000.", "11220.", "12589.", "14125.", "15849.",
    "17783.", "19953.", "22387.", "25119.", "28184.", "31623.", "35481.", "39811.", "44668.", "50119.", "56234.",
    "63096.", "70795.", "79433.", "89125.", "100000.", "112202.", "125893.", "141254.", "158489.", "177828.",
    "199526.")
DFSYNTHE_VTURB = ("0", "1", "2", "4", "8")


# ============================================================================================================
# small helpers
# ============================================================================================================
def notify(message, silent):
    """Print ``message`` unless ``silent``."""
    if not silent:
        print(message)


def cmd(command):
    """Run a shell command; print its stdout, raise ValueError on any stderr (BasicATLAS helper, kept for
    user code; the API itself never shells out)."""
    import subprocess
    p = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.stdout.decode().strip(), p.stderr.decode().strip()
    if out:
        print(out)
    if err:
        raise ValueError("Command {} returned an error: {}".format(command, err))


def _bash(script):
    """Execute one of our command files (BasicATLAS ran them with ``bash``); echo un-redirected output."""
    out = runners.run_script(script).strip()
    if out:
        print(out)


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _read(path):
    with open(path) as f:
        return f.read()


def bin_spec(wl, flux, num_bins=1000):
    """Average ``flux`` in ``num_bins`` equal-width wavelength bins; empty bins are dropped.
    Returns (bin centres, mean flux)."""
    sums, edges = np.histogram(wl, bins=num_bins, weights=flux)
    counts, _ = np.histogram(wl, bins=num_bins)
    used = counts > 0
    return np.array(edges[1:] + edges[:-1])[used] / 2.0, sums[used] / counts[used]


def _x(nu, T):
    return spc.h * nu / spc.k / T


def blackbody(nu, T):
    """Planck B_nu(T) in SI (0 where the exponent would overflow)."""
    x = _x(nu, T)
    if x < 675 and np.e ** x - 1 > 0.0:
        return 2 * spc.h * nu ** 3.0 / spc.c ** 2.0 / (np.e ** x - 1)
    return 0.0


def blackbody_peak(T):
    """(frequency, value) of the maximum of B_nu(T) (Wien's law)."""
    nu = 5.8789232e10 * T
    return nu, blackbody(nu, T)


def blackbody_dBdT(nu, T):
    """dB_nu/dT in SI (0 where the exponent would overflow)."""
    x = _x(nu, T)
    if x < 350 and np.e ** x - 1 > 0:
        return 2 * spc.h ** 2.0 * nu ** 4.0 / (spc.c ** 2.0 * spc.k * T ** 2.0) * np.e ** x / (np.e ** x - 1) ** 2.0
    return 0.0


def blackbody_dBdT_peak(T):
    """(frequency, value) of the maximum of dB_nu/dT; x = 3.8300160963 solves x^2 coth(x/2) = 4x."""
    nu = 3.8300160963 / spc.h * spc.k * T
    return nu, blackbody_dBdT(nu, T)


def synbeg(min_wl, max_wl, res):
    """Number of wavelength points SYNTHE (synbeg.for) puts in [min_wl, max_wl] nm at resolution ``res``:
    points are exp(i * ln(1 + 1/res)) for integer i."""
    step = np.log(1.0 + 1.0 / res)
    first = int(np.log(min_wl) / step)
    if np.e ** (first * step) < min_wl:
        first += 1
    last = int(np.log(max_wl) / step)
    if np.e ** (last * step) > max_wl:
        last -= 1
    return int(last - first + 1)


# ============================================================================================================
# command files (byte-compatible with BasicATLAS's)
# ============================================================================================================
def _abundance_cards(scale, el):
    """ABUNDANCE SCALE / ABUNDANCE CHANGE cards as read by ATLAS9, KAPPA9, XNFDF (no trailing newline)."""
    rows = ["ABUNDANCE SCALE {:>9.6f} ABUNDANCE CHANGE 1 {:>7.5f} 2 {:>7.5f}".format(scale, el[1], el[2])]
    for first in range(3, 100, 6):
        rows.append(" ABUNDANCE CHANGE " + " ".join("{:2d} {:>6.2f}".format(z, el[z])
                                                    for z in range(first, min(first + 6, 100))))
    return "\n".join(rows)


_OPACITY_CARD = " OPACITY IFOP" + " 1" * 13 + " 0 1" + " 0" * 5
_CONVECTION_CARD = " CONVECTION ON   1.25 TURBULENCE OFF" + "  0.00" * 4
_BEGIN_15 = "BEGIN" + " " * 20 + "ITERATION  15 COMPLETED"


def _atlas_iterations(n):
    return ("ITERATIONS {} PRINT".format(n) + " 1" * 15 + "\nPUNCH" + " 1" * 15 + "\n"
            + "BEGIN" + " " * 20 + "ITERATION  10 COMPLETED\n")


def _atlas_scripts(c):
    """(start, control, end) command files of an ATLAS run; ``c`` holds paths and parameters."""
    start = ("\ncd {od}\nln -s {kapparos} fort.1\nln -s {odf} fort.9\nln -s {molecules} fort.2\n"
             "ln -s {restart} fort.3 \n").format(**c)
    deck = "\n".join([
        "READ KAPPA", "READ PUNCH", "MOLECULES {}".format(c["molecules_switch"]), "READ MOLECULES",
        "FREQUENCIES 337 1 337 BIG", "VTURB {}.0E+5".format(c["vturb"]), "CONVECTION OVER 1.25 0 36  ",
        "TITLE  BasicATLAS", "SCALE 72 -6.875 0.125 {:<6.0f} {:<7.4f}".format(c["teff"], c["logg"]),
        _abundance_cards(c["scale"], c["el"]),
    ]) + "\n" + c["iterations"] + "\n"
    control = '\ncd {od}\n{exe}<<"EOF">{out_main}\n'.format(**c) + deck + "EOF\n"
    end = "\ncd {od}\nmv fort.7 {out_summary}\nrm fort.*\n".format(**c)
    return start, control, end


_SYNTHE_MOLECULES = (  # (file in synthe_files, output name, extra space after "ln -s")
    ("chmasseron_corrected.asc", "chmasseron", True), ("mgh_exomol.asc", "mgh", False), ("nh.asc", "nh", False),
    ("ohupdate.asc", "oh", True), ("sihax.asc", "sihax", True), ("h2.asc", "h2", False), ("h2xx.asc", "h2xx", False),
    ("hdxx.asc", "hdxx", False), ("c2ax.asc", "c2ax", False), ("c2ba.asc", "c2ba", False),
    ("c2dabrookek.asc", "c2da", False), ("c2ea.asc", "c2ea", True), ("cnaxbrookek.asc", "cnax", False),
    ("cnbxbrookek.asc", "cnbx", False), ("cnxx12brooke.asc", "cnxx12", False), ("coax.asc", "coax", False),
    ("coxx.asc", "coxx", False), ("sioax.asc", "sioax", False), ("sioex.asc", "sioex", False),
    ("sioxx.asc", "sioxx", False))

_SPECTRV_CARDS = ("0.0       0.        1.        0.        0.        0.        0.        0.", "0.",
                  "RHOXJ     R1        R101      PH1       PC1       PSI1      PRDDOP    PRDPOW")


def _synthe_script(c):
    """synthe_launch.com: xnfpelsyn, synbeg, line readers, synthe, spectrv in <od>/synthe_<n>/."""
    s, d, b = c["s_files"], c["d_files"], c["bin"]
    card = ("{:<3s}       {:<10.4f}{:<10.4f}{:<9.2f} {:<10.4f}{:<3d}{:<7d}{:<10.5f}{:<5d}{:<5d}"
            .format(c["airorvac"], c["wlbeg"], c["wlend"], c["resolu"], c["turbv"], 0, -1, 0.0001, 1, 0))
    L = [
        "cd {}".format(c["od"]), "mkdir -p synthe_{}".format(c["n"]), "cd synthe_{}/".format(c["n"]),
        "ln -s {}/molecules.dat fort.2".format(s), "ln -s {}/continua.dat fort.17".format(s), "",
        "# xnfpelsyn.exe computes the chemical equilibrium",
        "{}/xnfpelsyn.exe< {}>xnfpelsyn.out".format(b, c["model"]), "mv fort.10 xnfpelsyn.dat", "rm fort.*", "",
        "# synbeg.exe initializes the computation",
        '{}/synbeg.exe<<"EOF">synbeg.out'.format(b), card,
        "AIRorVAC  WLBEG     WLEND     RESOLU    TURBV  IFNLTE LINOUT CUTOFF        NREAD", "EOF", "",
        "# Import atomic lines", "ln -s {} fort.11".format(c["linelist"]),
        "{}/rgfalllinesnew.exe>rgfalllinesnew.out".format(b), "rm fort.11", "",
        "# Set the C12/C13 ratio", c["c12c13"], "",
        "# Import diatomic molecular lines",
    ]
    for fname, out, wide in _SYNTHE_MOLECULES:
        L += ["ln -s{}{}/{} fort.11".format("  " if wide else " ", s, fname),
              "{}/rmolecasc.exe>{}.out".format(b, out), "rm fort.11"]
    L += ["", "# Import TiO lines", "ln -s {}/tiototo.asc fort.11".format(s), "{}/rmolecasc.exe>tio.out".format(b),
          "rm fort.11", "",
          "# Import H2O lines", "ln -s {}/h2ofastfix.bin fort.11".format(d), "{}/rh2ofast.exe>h2ofastfix.out".format(b),
          "rm fort.11", "",
          "# synthe.exe computes line opacities", "ln xnfpelsyn.dat fort.10",
          "ln -s {}/he1tables.dat fort.18".format(s), "{}/synthe.exe>synthe.out".format(b), "",
          "# spectrv.exe computes the synthetic spectrum", "ln -s {}/molecules.dat fort.2".format(s),
          'cat <<"EOF" >fort.25', *_SPECTRV_CARDS, "EOF",
          '{}/spectrv.exe<"{}">spectrv.out'.format(b, c["model"]), "",
          "mv fort.7 spectrum.bin", "rm fort.*", "rm xnfpelsyn.dat", ""]
    return "\n".join(L)


def _synthe_cleanup_script(od):
    return "cd {}\nrm -f output_synthe.out\nrm -f synthe_launch.com\nrm -f spectrum.dat\nrm -rf synthe_*\n".format(od)


# Dummy (solar) model deck that KAPPA9 is fed: only needed to satisfy its input reader. It is the one embedded
# in BasicATLAS's KAPPA9 command template; we read it from the BasicATLAS checkout at run time.
def _kappa9_deck():
    src = _read(os.path.join(python_path, "templates.py"))
    head = "READ DECK6 72 RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB\n"
    i = src.index(head, src.index("kappa9_control"))
    j = src.index("\n", src.index("\nPRADK", i) + 1)
    return src[i:j + 1]


def _xnfdf_pressure(k):
    """k-th of the 25 rhox/P values 1e-4 .. 1e8 at half-dex steps, written as in the XNFDF deck."""
    e, half = divmod(k, 2)
    e -= 4
    return "{}E{}{}".format("3.162" if half else "1.000", "-" if e <= 0 else "+", abs(e))


def _xnfdf_script(c):
    head = ("cd {od}\ncp {d}/molecules.dat fort.2\ncp {d}/pfiron.dat fort.4\ncp {d}/continua.dat fort.17\n"
            "{b}/xnfdf.exe<<EOF>xnfdf.out\nREAD MOLECULES\nMOLECULES ON\nITERATIONS 1 PRINT 0 PUNCH 0\n"
            "READ FREQUENCIES 1 1 1 TEST\n1 1 1\n").format(od=c["od"], d=c["d_data"], b=c["bin"])
    body = []
    for dft in DFSYNTHE_TEMPERATURES:
        body.append("\n".join([
            "TEFF   5777.  GRAVITY 4.43770 LTE",
            "TITLE TEMPERATUES AND PRESSURES FOR DISTRIBUTION FUNCTION CALCULATION",
            _OPACITY_CARD, _CONVECTION_CARD, _abundance_cards(c["scale"], c["el"]),
            "READ DECK6 25 RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB,CONVFRAC,VCONV",
            *("   {p}   {t}  {p}  0 0 0 0 0".format(p=_xnfdf_pressure(k), t=dft) for k in range(25)),
            " 1", "BEGIN"]) + "\n")
    tail = "END\nEOF\nmv fort.10  xnfpdf.dat\nmv fort.11 xnfpdfmax.dat\nrm fort.*\n"
    return head + "".join(body) + tail


_DF_LINKS = (("../xnfpdf.dat", 10), ("../xnfpdfmax.dat", 22), ("lowlinesdf.bin", 11), ("highlinesdf.bin", 21),
             ("diatomicsdf.bin", 31), ("tiolinesdf.bin", 41), ("h2olinesdf.bin", 43), ("nltelinesdf.bin", 51))
_DF_OUT_UNITS = ((15, "0"), (16, "1"), (17, "2"), (18, "4"), (20, "8"))


def _dfsynthe_script(od, d_data, b, dft, switches):
    """dft_<i>/dfp.com: DFSYNTHE at one temperature, then DFSORTP for each microturbulence."""
    links = "".join("ln -s {} fort.{}\n".format(f if f.startswith("..") else d_data + "/" + f, u)
                    for f, u in _DF_LINKS)
    run = "\ncd {od}\n{b}/dfsynthe.exe<<EOF>dfp00t{t}.out \n{sw}\nEOF\n".format(od=od, b=b, t=dft, sw=switches)
    run += "".join("mv fort.{} dfp00t{}vt{}.bin\n".format(u, dft, v) for u, v in _DF_OUT_UNITS)
    for _, v in _DF_OUT_UNITS:
        run += ("\nmv dfp00t{t}vt{v}.bin fort.1\n{b}/dfsortp.exe>dfsortp.out\nmv fort.2 ../dfp00t{t}vt{v}sortp.asc\n"
                "mv fort.1 ../dfp00t{t}vt{v}.bin\n").format(t=dft, v=v, b=b)
    cleanup = "\ncd {}\n".format(od) + "".join("rm fort.{}\n".format(u) for _, u in _DF_LINKS)
    return "\ncd {}\n".format(od) + links + run + cleanup


def _separatedf_script(od, b, v):
    moves = "".join("mv {od}/dfp00t{t}vt{v}sortp.asc {od}/fort.{n}\n".format(od=od, t=str(int(float(t))), v=v, n=i + 10)
                    for i, t in enumerate(DFSYNTHE_TEMPERATURES))
    return moves + ("cd {od}\n{b}/separatedf.exe\nmv fort.2 p00big{v}.bdf\nmv fort.3 p00lit{v}.bdf\nrm fort.*\n"
                    .format(od=od, b=b, v=v))


def _kappa9_script(c, v):
    deck = "\n".join(["MOLECULES ON", "READ MOLECULES", "FREQUENCIES 337 1 337 BIG", "ITERATIONS 1 PRINT 1 PUNCH 0 ",
                      "TITLE ROSSELAND OPACITY", _OPACITY_CARD, _CONVECTION_CARD, "TEFF   5777.  GRAVITY 4.43770 LTE",
                      _abundance_cards(c["scale"], c["el"])]) + "\n"
    deck += _kappa9_deck() + "VTURB {}.0E5\n{}\nEND\n".format(v, _BEGIN_15)
    return ("\ncd {od}\ncp {d}/molecules.dat fort.2\nmv ./p00big{v}.bdf fort.9\n{b}/kappa9.exe<<EOF>kapm40k2.out\n"
            .format(od=c["od"], d=c["d_data"], v=v, b=c["bin"])
            + deck + "EOF\nmv fort.9 ./p00big{v}.bdf\nmv fort.7 kapk{v}.dat\n".format(v=v))


def _kapreadts_script(od, b):
    moves = "".join("mv kapk{}.dat fort.{}\n".format(v, 11 + i) for i, v in enumerate(DFSYNTHE_VTURB))
    return ("\ncd {}\n".format(od) + moves
            + "{}/kapreadts.exe\nmv fort.2 kappa.ros\nrm fort.*\nrm -r dft_*\nrm *.bin\n".format(b))


# ============================================================================================================
# output parsers
# ============================================================================================================
def _lines_back(text, pos, n):
    for _ in range(n):
        pos = text[:pos].rfind("\n")
    return pos


def parse_atlas_abundances(file, classic_style=True, lookahead=0, lookbehind=0, params=[], to_float=True):
    """Abundances (100-list in ATLAS format, entry 0 = 0) and optionally other card values from an
    ATLAS/SYNTHE/DFSYNTHE input or output file.

    ``classic_style``: control-card listing (ABUNDANCE SCALE / ABUNDANCE CHANGE); otherwise the element-symbol
    table printed by the programs (e.g. output_main.out). ``params``: names whose following value is returned,
    searched in the listing extended by ``lookbehind`` lines before and ``lookahead`` lines after; "NAME/k"
    takes the k-th value after NAME. Returns ``elements`` or ``(elements, {param: value})``."""
    text = _read(file)
    elements = [0]
    if classic_style:
        a = re.search("ABUNDANCE CHANGE +3 +[0-9.-]+ +4 +[0-9.-]+ +5 +", text).span()[0]
        b = re.search("ABUNDANCE CHANGE +99 +[0-9.-]+ ?", text).span()[1]
        a = _lines_back(text, a, 2)
        block = text[a:b]
        for z in range(1, 100):
            elements.append(float(re.findall(r"[^0-9]{} +([0-9-]+\.[0-9]+)".format(z), block)[0].strip()))
    else:
        text = text.replace("0XSCALE", "0XSCALE ").replace("TEFF", "TEFF ")
        a = re.search("LI[0-9 .-]+BE[0-9 .-]+B", text).span()[0]
        b = re.search("BK[0-9 .-]+CF[0-9 .-]+ES[0-9 .-]+", text).span()[1]
        a = _lines_back(text, a, 2)
        block = text[a:b]
        for sym in Settings().abun_solar().keys():
            label = sym.upper().ljust(2)
            elements.append(float(re.findall("[^A-Z]{}([0-9 .-]+)".format(label), block)[0].strip()))

    if len(params) == 0:
        return elements
    a = _lines_back(text, a, lookbehind)
    for _ in range(lookahead):
        b += text[b:].find("\n") + 1
    if a == -1:
        a = 0
    window = text[a:b]
    values = {}
    for param in params:
        name, _, k = param.partition("/")
        k = int(k) if k else 1
        pattern = name + " +[^ \\n]+" * (k - 1) + " +([^ \\n]+)"
        values[param] = re.findall(pattern, window)[0]
        if to_float:
            values[param] = float(values[param])
    return elements, values


_STRUCTURE_SUMMARY = (  # output_summary.out columns
    (1, "temperature", "K"), (2, "gas_pressure", "Ba"), (3, "electron_number_density", "cm^-3"),
    (4, "rosseland_opacity", "cm^2 g^-1"), (5, "radiative_acceleration", "cm s^-2"),
    (6, "turbulent_velocity", "cm s^-1"), (7, "radiative_flux", "erg cm^-2 s^-1"),
    (8, "convective_speed", "cm s^-1"), (9, "speed_of_sound", "cm s^-1"))
_STRUCTURE_TABLE = (  # output_last_iteration.out columns
    (1, "mass_column_density", "g cm^-2"), (5, "density", "g cm^-3"), (7, "physical_depth", "km"),
    (8, "rosseland_optical_depth", ""), (9, "convective_flux", "erg cm^-2 s^-1"), (10, "radiation_pressure", "Ba"),
    (11, "flux_error", "percent"), (12, "flux_error_derivative", "percent"))


def read_structure(run_dir):
    """Depth profiles of an ATLAS run (outermost layer first): (structure {name: array}, units {name: str})."""
    if not os.path.isdir(run_dir):
        raise ValueError("Run directory {} not found!".format(run_dir))
    if not (os.path.isfile(run_dir + "/output_last_iteration.out") and os.path.isfile(run_dir + "/output_summary.out")):
        raise ValueError("Run directory {} does not contain ATLAS-9 output!".format(run_dir))
    structure, units = {}, {}

    deck = _read(run_dir + "/output_summary.out")
    deck = deck[deck.rfind("RHOX"):deck.rfind("PRADK")]
    # fixed-width fields may touch ("1.0E+05-2.0"): separate on minus signs that are not exponent signs
    deck = deck.replace("E-", "E=").replace("-", " -").replace("E=", "E-")
    cols = np.loadtxt(deck.split("\n"), unpack=True, skiprows=1)
    structure["layer"] = np.arange(1, np.shape(cols)[1] + 1)
    units["layer"] = ""
    for i, key, unit in _STRUCTURE_SUMMARY:
        structure[key], units[key] = cols[i], unit

    table = _read(run_dir + "/output_last_iteration.out").replace("*", "9")   # "****" = field overflow
    old_format = table.strip().find("TEFF") == 0
    cols = np.loadtxt(table.split("\n"), unpack=True, **({"skiprows": 3} if old_format else {}))
    for i, key, unit in _STRUCTURE_TABLE:
        structure[key], units[key] = cols[i], unit
    return structure, units


def load_binary_spectrum(filename, mask=False):
    """Read SYNTHE's spectrum.bin: (wl [A], flux, continuum [erg s^-1 cm^-2 A^-1 sr^-1], flux/continuum)."""
    header_t = np.dtype("i4,f8,f8,S74,f8,f8,i4,i4,i4," + ",".join(["f8"] * 20 + ["i4"] + ["f8"] * 377))
    with open(filename, "rb") as f:
        header = np.fromfile(f, dtype=header_t, count=1)
        data = np.fromfile(f, dtype=np.dtype("f8,f8,f8"), count=-1)
    wlbeg, resolu, n = header["f4"][0], header["f5"][0], header["f6"][0]
    wl = 10 ** (np.log10(wlbeg) + np.arange(n) * np.log10(1 + 1 / resolu)) * 10
    if isinstance(mask, bool):
        mask = np.full(len(wl), True)
    wl = wl[mask]
    skip = 0 if mask[0] else 1              # the first point is always computed
    flux = 4.0 * data["f1"][skip:] * spc.c * 1e10 / (wl ** 2.0)
    cont = 4.0 * data["f2"][skip:] * spc.c * 1e10 / (wl ** 2.0)
    return wl, flux, cont, flux / cont


def read_spectrum(run_dir, num_bins=-1):
    """Spectrum of a SYNTHE run (all batches): {'wl' [A], 'flux', 'cont', 'line' = flux/cont}, optionally binned."""
    if not os.path.isdir(run_dir):
        raise ValueError("Run directory {} not found!".format(run_dir))
    if not os.path.isdir(run_dir + "/synthe_1"):
        raise ValueError("Run directory {} does not contain SYNTHE output!".format(run_dir))
    data = np.array([[], [], [], []])
    n = 1
    while os.path.isdir(run_dir + "/synthe_{}".format(n)):
        data = np.append(data, load_binary_spectrum(run_dir + "/synthe_{}/spectrum.bin".format(n)), axis=1)
        n += 1
    wl, flux, cont, line = data
    if num_bins > 0:
        _, flux = bin_spec(wl, flux, num_bins=num_bins)
        _, cont = bin_spec(wl, cont, num_bins=num_bins)
        wl, line = bin_spec(wl, line, num_bins=num_bins)
    return {"wl": wl, "flux": flux, "cont": cont, "line": line}


# ============================================================================================================
# run metadata and validation
# ============================================================================================================
def meta(run_dir):
    """Metadata of a run directory (DFSYNTHE, ATLAS or ATLAS+SYNTHE), see meta_dfsynthe / meta_atlas."""
    if not os.path.isdir(run_dir):
        raise ValueError("Run directory {} not found!".format(run_dir))
    if os.path.isfile(run_dir + "/xnfdf.out"):
        return meta_dfsynthe(run_dir)
    if os.path.isfile(run_dir + "/output_summary.out"):
        return meta_atlas(run_dir)
    raise ValueError("Run {} type unknown!".format(run_dir))


def meta_dfsynthe(run_dir):
    """{'abun', 'Y', 'zscale', 'type': 'DFSYNTHE'} of an ODF directory (from the XNFDF printout)."""
    el, p = parse_atlas_abundances(run_dir + "/xnfdf.out", classic_style=False, lookbehind=1, params=["0XSCALE"])
    out = Settings().abun_atlas_to_std(el, np.log10(p["0XSCALE"]))
    out["type"] = "DFSYNTHE"
    return out


def meta_atlas(run_dir):
    """Metadata of an ATLAS run: abun, Y, zscale, teff, logg, vturb [km/s], type ('ATLAS' / 'SYNTHE'); for SYNTHE
    runs also res, synthe_vturb (False if depth dependent), medium ('vacuum'/'air'/'unknown'), abun_adjust."""
    el, p = parse_atlas_abundances(run_dir + "/output_summary.out", classic_style=True, lookbehind=4,
                                   params=["ABUNDANCE SCALE", "TEFF", "GRAVITY"])
    out = Settings().abun_atlas_to_std(el, np.log10(p["ABUNDANCE SCALE"]))
    out["teff"] = p["TEFF"]
    out["logg"] = p["GRAVITY"]
    out["vturb"] = read_structure(run_dir)[0]["turbulent_velocity"][0] * 1e-5
    out["type"] = "ATLAS"
    if not os.path.isfile(run_dir + "/synthe_launch.com"):
        return out

    out["type"] = "SYNTHE"
    received = validate_run(run_dir, return_received_synthe=True)
    out["res"] = received["resolu"]
    # SYNTHE adds its TURBV in quadrature to the model's VTURB column (which synthe() zeroes)
    text = _read(run_dir + "/output_synthe.out")
    deck = text[text.find("\n", text.find("FLXRAD,VCONV,VELSND")) + 1: text.rfind("\n", 0, text.find("PRADK"))].strip()
    model_vturb = np.array([float(row[-40:-30].strip()) for row in deck.split("\n")]) * 1e-5
    if np.all(model_vturb[0] == model_vturb):
        out["synthe_vturb"] = np.sqrt(model_vturb[0] ** 2.0 + received["turbv"] ** 2.0)
    else:
        out["synthe_vturb"] = False
    launch = _read(run_dir + "/synthe_launch.com")
    out["medium"] = "air" if "\nAIR " in launch else "vacuum" if "\nVAC " in launch else "unknown"
    out["abun_adjust"] = {}
    el, p = parse_atlas_abundances(run_dir + "/output_synthe.out", lookbehind=1, params=["ABUNDANCE SCALE"])
    syn = Settings().abun_atlas_to_std(el, np.log10(p["ABUNDANCE SCALE"]))["abun"]
    model = out["abun"]
    for e in list(syn.keys()) + list(model.keys()):
        if e in syn and e not in model:
            out["abun_adjust"][e] = syn[e]
        elif e not in syn and e in model:
            out["abun_adjust"][e] = -model[e]
        elif (delta := np.round(syn[e] - model[e], 2)) != 0:
            out["abun_adjust"][e] = delta
    return out


def _synthe_requested(run_dir):
    text = _read(run_dir + "/synthe_launch.com")
    pos = _lines_back(text, text.find("AIRorVAC  WLBEG     WLEND     RESOLU    TURBV  IFNLTE"), 2)
    card = text[pos:pos + text[pos + 1:].find("\n") + 1].strip()
    fields = (("wlbeg", 10, 20), ("wlend", 20, 30), ("resolu", 30, 40), ("turbv", 40, 50), ("ifnlte", 50, 53),
              ("linout", 53, 60), ("cutoff", 60, 70), ("ifpred", 70, 75), ("nread", 75, 80))
    req = {k: float(card[a:b].strip()) for k, a, b in fields}
    req["turbv"] = np.round(req["turbv"], 2)
    return req


def validate_run(run_dir, return_received_synthe=False, silent=False):
    """Check that the programs of a run received the parameters that were requested in its command files
    (fixed-format cards can silently truncate). Returns True, or raises ValueError on a mismatch.
    With ``return_received_synthe`` returns the SYNTHE parameters echoed in synbeg.out instead."""
    if not os.path.isdir(run_dir):
        raise ValueError("Run directory {} not found!".format(run_dir))

    if os.path.isfile(run_dir + "/atlas_control.com"):
        el_req, p_req = parse_atlas_abundances(run_dir + "/atlas_control.com", lookbehind=4,
                                               params=["VTURB", "ABUNDANCE SCALE", "SCALE 72/3", "SCALE 72/4"])
        el_got, p_got = parse_atlas_abundances(run_dir + "/output_main.out", classic_style=False, lookbehind=2,
                                               params=["0XSCALE", "TEFF", "LOG G"])
        vturb = read_structure(run_dir)[0]["turbulent_velocity"]
        assert np.all(vturb[0] == vturb)
        vturb = vturb[0]
        for z in range(len(el_req)):
            if el_req[z] != el_got[z]:
                raise ValueError("Element {} requested/received mismatch: {} vs {}".format(z, el_req[z], el_got[z]))
        for what, a, b in (("Metallicity", p_req["ABUNDANCE SCALE"], p_got["0XSCALE"]),
                           ("Teff", p_req["SCALE 72/3"], p_got["TEFF"]),
                           ("Log(g)", p_req["SCALE 72/4"], p_got["LOG G"]),
                           ("Vturb", p_req["VTURB"], vturb)):
            if a != b:
                raise ValueError("{} requested/received mismatch: {} vs {}".format(what, a, b))
        if not return_received_synthe:
            notify("ATLAS requested/received validation for {} successful".format(run_dir), silent)

    if os.path.isfile(run_dir + "/synthe_launch.com"):
        req = _synthe_requested(run_dir)
        n = 1
        while os.path.isdir(run_dir + "/synthe_{}".format(n + 1)):
            n += 1
        echo = _read(run_dir + "/synthe_{}/synbeg.out".format(n))
        got = {}
        for k in req:
            got[k] = float(re.findall("{}={{0,1}} *([^ \\n]+)".format(k.upper()), echo)[0])
            if req[k] != got[k]:
                raise ValueError("{} requested/received mismatch: {} vs {}".format(k, req[k], got[k]))
        if return_received_synthe:
            return got
        notify("SYNTHE requested/received validation for {} successful".format(run_dir), silent)

    if os.path.isfile(run_dir + "/xnfdf.com"):
        for requested, received in (("xnfdf.com", "xnfdf.out"), ("kappa9v0.com", "kapm40k2.out")):
            el_req, p_req = parse_atlas_abundances(run_dir + "/" + requested, lookbehind=1, params=["ABUNDANCE SCALE"])
            el_got, p_got = parse_atlas_abundances(run_dir + "/" + received, classic_style=False, lookbehind=1,
                                                   params=["0XSCALE"])
            for z in range(len(el_req)):
                if el_req[z] != el_got[z]:
                    raise ValueError("Element {} requested/received mismatch: {} vs {}".format(z, el_req[z], el_got[z]))
            if p_req["ABUNDANCE SCALE"] != p_got["0XSCALE"]:
                raise ValueError("Metallicity requested/received mismatch: {} vs {}".format(p_req["ABUNDANCE SCALE"], p_got["0XSCALE"]))
        notify("DFSYNTHE requested/received validation for {} successful".format(run_dir), silent)
    return True


def atlas_converged(run_dir, print_result=False, silent=False, niter=0):
    """Pick the best iteration of an ATLAS run (flux error / flux-derivative error classes GOLD < 1 % & 10 %,
    SILVER < 10 % & 100 %, BRONZE < 1000 %, else UNCONVERGED; failed chemistry counts as 88888.888), write it to
    output_last_iteration.out (and keep only its punch in fort.7 if present). Raises if the run stopped early
    (fewer than ``niter`` iterations) for no known reason. Returns (flux errors, derivative errors) of it."""
    text = _read(run_dir + "/output_main.out")
    tables = re.findall("START TABLE.+?END TABLE", text, re.DOTALL)
    max_err, max_de = [], []
    cursor = 0
    for table in tables:
        end = text.find("END TABLE", cursor)
        cursor = end + 1
        begin = max(text.rfind("END TABLE", 0, cursor), 0)   # the previous table's end
        if text[begin:end].find("CHEMFAIL") != -1:
            max_err.append(88888.888)
            max_de.append(88888.888)
            continue
        rows = table.split("\n")[4:-1]
        assert len(rows) == 72
        err, de = np.loadtxt(rows, unpack=True, usecols=[11, 12])
        max_err.append(np.max(np.abs(err)))
        max_de.append(np.max(np.abs(de)))
    if len(max_err) == 0:
        raise ValueError("The model diverged immediately: no iterations could be completed")

    max_err, max_de = np.array(max_err), np.array(max_de)
    idx = np.arange(len(max_err))
    gold = (max_err < 1.0) & (max_de < 10.0)
    silver = (max_err < 10.0) & (max_de < 100.0)
    bronze = max_err < 1000.0
    for label, sel in (("GOLD", gold), ("SILVER", silver), ("BRONZE", bronze), ("UNCONVERGED", ~gold & ~silver & ~bronze)):
        if np.count_nonzero(sel) > 0 or label == "UNCONVERGED":
            best = idx[sel][np.argmin(max_err[sel])]
            conv = label
            break
    if print_result:
        notify("Total iterations: {} | Best iteration: {}".format(len(max_de), best + 1), silent)
        notify("For the best iteration: max[|err|] = {} | max[|de|] = {} | convergence class: {}".format(max_err[best], max_de[best], conv), silent)

    _write(run_dir + "/output_last_iteration.out", "\n".join(tables[best].split("\n")[4:-1]))
    err, de = np.loadtxt(run_dir + "/output_last_iteration.out", unpack=True, usecols=[11, 12])

    if os.path.isfile(run_dir + "/fort.7"):
        punches = _read(run_dir + "/fort.7").split("\n==========\n")[:-1]
        assert len(punches) == len(max_err)
        _write(run_dir + "/fort.7", punches[best])

    ok = niter == 0 or niter == len(max_err) or text.find("REACHED GOLD TOLERANCES") != -1
    if text.find("MODEL DIVERGED") != -1 or text.find("INVALID STRUCTURE") != -1:
        ok = True
        if print_result:
            notify("**WARNING:** Model terminated due to divergence", silent)
    if text.find("HYDROFAIL") != -1:
        ok = True
        if print_result:
            notify("**WARNING:** Model terminated due to hydrostatic failure", silent)
    if not ok:
        raise ValueError("Model terminated due to unknown reason")
    return err, de


# ============================================================================================================
# ATLAS
# ============================================================================================================
def _frequency_grid(main_out):
    """Frequencies (Hz) of ATLAS's integration grid from the FREQID table of output_main.out."""
    nu, reading = [], False
    with open(main_out) as f:
        for line in f:
            if line.find("0FREQID") != -1:
                reading = True
                continue
            if reading and len(line) < 60:
                break
            if reading:
                # a negative first quadrature weight runs into the previous field
                nu += list(np.loadtxt([line.replace("-", " -")])[1::3])
    return np.array(sorted(nu))


def _odf_temperatures(odf_dir):
    temps, reading = [], False
    with open(odf_dir + "/xnfdf.com") as f:
        for line in f:
            if line.find("RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB,CONVFRAC,VCONV") != -1:
                reading = True
                continue
            if len(line) < 10:
                reading = False
                continue
            if reading:
                temps.append(np.loadtxt([line])[1])
    return temps


def atlas(output_dir, settings=Settings(), restart="auto", niter=450, ODF=python_path + "/data/solar_ODF",
          molecules=True, silent=False):
    """Compute an ATLAS9 model atmosphere in ``output_dir`` (must not exist).

    ``settings``: Settings (teff, logg, chemistry, vturb selecting the ODF); ``restart``: "auto", "gray", a
    previous run directory or an output_summary.out-style file (see restarts.prepare_restart); ``niter``: maximum
    number of iterations, run in batches of 15 that stop early once GOLD convergence is reached; ``ODF``: DFSYNTHE
    output directory for this chemistry; ``molecules``: molecular equilibrium on/off; ``silent``: no messages."""
    t0 = datetime.now()
    if os.path.isdir(output_dir):
        raise ValueError("Directory {} already exists".format(output_dir))
    os.mkdir(output_dir)
    output_dir = os.path.realpath(output_dir)

    odf_meta = meta(ODF)
    if odf_meta["type"] != "DFSYNTHE":
        raise ValueError("Directory {} does not have ODFs".format(ODF))
    settings.check_ODF(odf_meta)
    if not os.path.isfile(ODF + "/p00big{}.bdf".format(settings.vturb)):
        have = [m for name in os.listdir(ODF) for m in re.findall(r"p00big([0-9]+)\.bdf", name)]
        raise ValueError("ODF not calculated for vturb={}. Available vturb: {}".format(settings.vturb, have))

    message = restarts.prepare_restart(restart, output_dir + "/restart.dat", settings, read_structure, meta)
    notify(message, silent)

    niter = int(niter)
    iterations = _atlas_iterations("15") * (niter // 15)
    if niter % 15 != 0:
        iterations += _atlas_iterations(str(niter % 15))
    c = {
        "od": output_dir,
        "kapparos": os.path.realpath(ODF + "/kappa.ros"),
        "odf": os.path.realpath(ODF + "/p00big{}.bdf".format(settings.vturb)),
        "molecules": python_path + "/data/atlas_files/molecules.dat",
        "restart": output_dir + "/restart.dat",
        "out_main": output_dir + "/output_main.out",
        "out_summary": output_dir + "/output_summary.out",
        "exe": python_path + "/bin/atlas9mem.exe",
        "scale": 10 ** float(settings.zscale),
        "el": settings.atlas_abun(),
        "teff": settings.teff,
        "logg": settings.logg,
        "vturb": str(int(settings.vturb)),
        "molecules_switch": ["OFF", "ON"][molecules],
        "iterations": iterations,
    }
    for name, text in zip(("atlas_control_start.com", "atlas_control.com", "atlas_control_end.com"), _atlas_scripts(c)):
        _write(output_dir + "/" + name, text)
    notify("Launcher created", silent)

    _bash(output_dir + "/atlas_control_start.com")
    _bash(output_dir + "/atlas_control.com")
    notify("ATLAS-9 halted", silent)
    atlas_converged(output_dir, True, silent, niter)
    _bash(output_dir + "/atlas_control_end.com")
    if not (os.path.isfile(c["out_main"]) and os.path.isfile(c["out_summary"])):
        raise ValueError("ATLAS-9 did not output expected files")
    validate_run(output_dir, silent=silent)

    # the frequency grid is fixed: make sure B_nu and dB/dT vanish at its ends for all layer temperatures
    nu = _frequency_grid(c["out_main"])
    structure, _ = read_structure(output_dir)
    t_range = [np.min(structure["temperature"]), np.max(structure["temperature"])]
    for T in t_range:
        for f in [np.min(nu), np.max(nu)]:
            if blackbody(f, T) > 1e-3 * blackbody_peak(T)[1]:
                warnings.warn("Planck's law in one of the layers at temperature {} K does not vanish at the frequency grid bound {} Hz. The result may be inaccurate!".format(T, f))
            if blackbody_dBdT(f, T) > 1e-3 * blackbody_dBdT_peak(T)[1]:
                warnings.warn("Derivative of Planck's law (dB/dT) in one of the layers at temperature {} K does not vanish at the frequency grid bound {} Hz. The result may be inaccurate!".format(T, f))
    odf_t = _odf_temperatures(ODF)
    for T in t_range:
        if T < np.min(odf_t) or T > np.max(odf_t):
            warnings.warn("Gas temperature {} K  in one of the layers falls outside the ODF range [{}, {}]. The result may be inaccurate!".format(T, np.min(odf_t), np.max(odf_t)))
    notify("Finished running ATLAS-9 in " + str(datetime.now() - t0) + " s", silent)


# ============================================================================================================
# SYNTHE
# ============================================================================================================
def _run_with_progress(script, progress_file):
    """Run a SYNTHE command file in a thread, showing synthe's progress.dat ("layer / layers") with tqdm."""
    import tqdm
    failure = []

    def target():
        try:
            _bash(script)
        except BaseException as e:      # re-raised in the caller's thread
            failure.append(e)

    worker = threading.Thread(target=target)
    worker.start()
    bar = tqdm.tqdm(total=100)
    while worker.is_alive():
        time.sleep(2)
        frac = 0
        if os.path.isfile(progress_file):
            last = _read(progress_file).strip().split("\n")[-1].split("/")
            try:
                frac = (int(last[0].strip()) - 1) / int(last[1].strip())
            except (ValueError, IndexError):     # line being written
                frac = bar.n / 100
        bar.update(np.round(frac * 100) - bar.n)
    worker.join()
    bar.update(100 - bar.n)
    bar.close()
    if failure:
        raise failure[0]


def synthe(output_dir, min_wl, max_wl, res=600000.0, vturb=1.5, abun_adjust={}, C12C13=False, linelist="BasicATLAS",
           buffsize=2010001, overwrite_prev=False, air_wl=False, silent=False, progress=True):
    """Synthetic spectrum of the ATLAS model in ``output_dir`` between ``min_wl`` and ``max_wl`` nm.

    ``res``: lambda/dlambda sampling; ``vturb``: SYNTHE microturbulence [km/s] (the model's own VTURB is zeroed);
    ``abun_adjust``: {el: dex} changes applied in the synthesis only; ``C12C13``: isotope ratio for molecules
    (False: rmolecasc default); ``linelist``: data/synthe_files/<name>.dat; ``buffsize``: max points per SYNTHE
    batch (synthe_1, synthe_2, ...); ``overwrite_prev``: replace a previous SYNTHE run; ``air_wl``: air
    wavelengths; ``progress``: tqdm progress bar."""
    t0 = datetime.now()
    output_dir = os.path.realpath(output_dir)
    if not (os.path.isfile(output_dir + "/output_last_iteration.out") and os.path.isfile(output_dir + "/output_summary.out")):
        raise ValueError("ATLAS run output not found in {}".format(output_dir))
    if os.path.isdir(output_dir + "/synthe_1"):
        if not overwrite_prev:
            raise ValueError("Previous SYNTHE run output found in {}. To overwrite, set overwrite_prev=True".format(output_dir))
        _write(output_dir + "/synthe_cleanup.com", _synthe_cleanup_script(output_dir))
        _bash(output_dir + "/synthe_cleanup.com")
        notify("Removed the output of a previous SYNTHE run", silent)

    # SYNTHE's input model: the ATLAS model with VTURB zeroed (SYNTHE adds its own) behind SYNTHE's header cards
    rows = _read(output_dir + "/output_summary.out").split("\n")
    inside = False
    for i, row in enumerate(rows):
        if row.find("FLXRAD,VCONV,VELSND") != -1:
            inside = True
        elif row.find("PRADK") != -1:
            inside = False
        elif inside:
            rows[i] = row[:-40] + " {:9.3E}".format(0) + row[-30:]
    prefix = ("SURFACE FLUX\nITERATIONS 1 PRINT 2 PUNCH 2\nCORRECTION OFF\nPRESSURE OFF\nREAD MOLECULES\n"
              "MOLECULES ON\n")
    model = output_dir + "/output_synthe.out"
    _write(model, prefix + "\n".join(rows))
    notify("Adapted the ATLAS-9 model to SYNTHE in output_synthe.out", silent)

    if len(abun_adjust) != 0:
        el, p = parse_atlas_abundances(model, lookbehind=1, params=["ABUNDANCE SCALE"])
        std = Settings().abun_atlas_to_std(el, np.log10(p["ABUNDANCE SCALE"]), rounding=7)
        for e in abun_adjust:
            if e not in std["abun"]:
                std["abun"][e] = 0
            std["abun"][e] += abun_adjust[e]
        el = Settings().abun_std_to_atlas(**std)
        text = _read(model)
        a = text.find("ABUNDANCE SCALE")
        b = text.find("\n", text.find("ABUNDANCE CHANGE 99"))
        _write(model, text[:a] + _abundance_cards(10 ** std["zscale"], el) + text[b:])
        notify("Updated abundances in output_synthe.out for spectral synthesis", silent)

    linelist = os.path.realpath(python_path + "/data/synthe_files/{}.dat".format(linelist))
    if not os.path.isfile(linelist):
        raise ValueError("Linelist {} not found".format(linelist))

    batch, done, lo = 0, False, min_wl
    while not done:
        batch += 1
        if synbeg(lo, max_wl, res) > buffsize:
            hi = int(brentq(lambda x: synbeg(lo, x, res) - buffsize, lo, max_wl))
            if hi == lo:
                raise ValueError("Requested resolution too high for buffer size")
        else:
            hi, done = max_wl, True

        if type(C12C13) is not bool:
            c13 = 1 / (C12C13 + 1)
            c12 = 1 - c13
            c12c13 = 'echo "{} {}" > c12c13.dat'.format(np.log10(c12), np.log10(c13))
        else:
            c12c13 = "rm -f c12c13.dat"
        c = {"od": output_dir, "n": batch, "s_files": python_path + "/data/synthe_files/",
             "d_files": python_path + "/data/dfsynthe_files/", "bin": python_path + "/bin/",
             "airorvac": ["VAC", "AIR"][air_wl], "wlbeg": float(lo), "wlend": float(hi), "resolu": float(res),
             "turbv": float(vturb), "model": model, "c12c13": c12c13, "linelist": linelist}
        script = output_dir + "/synthe_launch.com"
        _write(script, _synthe_script(c))
        notify("Launcher created for wavelength range ({}, {}), batch {}. Expected number of points: {} (buffer {})"
               .format(lo, hi, batch, synbeg(lo, hi, res), buffsize), silent)

        if progress:
            _run_with_progress(script, output_dir + "/synthe_{}/progress.dat".format(batch))
        else:
            _bash(script)
        if not os.path.isfile(output_dir + "/synthe_{}/spectrum.bin".format(batch)):
            raise ValueError("SYNTHE did not output expected files")
        notify("SYNTHE halted", silent)
        validate_run(output_dir, silent=silent)
        lo = hi
    notify("Finished running SYNTHE in " + str(datetime.now() - t0) + " s", silent)


# ============================================================================================================
# DFSYNTHE
# ============================================================================================================
def dfsynthe(output_dir, settings, parallel=False, silent=False):
    """Opacity distribution functions (5 microturbulences 0, 1, 2, 4, 8 km/s) and Rosseland mean opacity tables
    (kappa.ros) for the chemistry of ``settings``, in ``output_dir`` (must not exist). ``parallel`` runs the 57
    DFSYNTHE temperatures concurrently (threads)."""
    t0 = datetime.now()
    temps = DFSYNTHE_TEMPERATURES
    if os.path.isdir(output_dir):
        raise ValueError("Directory {} already exists".format(output_dir))
    os.mkdir(output_dir)
    output_dir = os.path.realpath(output_dir)

    el = settings.atlas_abun()
    c = {"od": output_dir, "d_data": python_path + "/data/dfsynthe_files/", "bin": python_path + "/bin/",
         "scale": 10 ** settings.zscale, "el": [el[0]] + [float(x) for x in el[1:]]}
    c["el"][1], c["el"][2] = el[1], el[2]
    _write(output_dir + "/xnfdf.com", _xnfdf_script(c))
    notify("Will run XNFDF to tabulate atomic and molecular number densities", silent)
    notify("Launcher created for " + str(len(temps)) + " temperatures from " + str(min(map(float, temps))) + " K to "
           + str(max(map(float, temps))) + " K", silent)
    _bash(output_dir + "/xnfdf.com")
    if not (os.path.isfile(output_dir + "/xnfpdf.dat") and os.path.isfile(output_dir + "/xnfpdfmax.dat")):
        raise ValueError("XNFDF did not output expected files")
    notify("XNFDF halted", silent)

    notify("Will run DFSYNTHE to tabulate the ODFs (Opacity Distribution Functions)", silent)

    def one_temperature(i):
        wd = "{}/dft_{}/".format(output_dir, i)
        os.mkdir(wd)
        script = output_dir + "/dft_{}/dfp.com".format(i)
        switches = "0" * i + "1" + "0" * (len(temps) - 1 - i)
        _write(script, _dfsynthe_script(wd, c["d_data"], c["bin"], str(int(float(temps[i]))), switches))
        _bash(script)
        notify(str(float(temps[i])) + " K done! (" + str(i + 1) + "/" + str(len(temps)) + ")", silent)

    if parallel:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            for _ in pool.map(one_temperature, range(len(temps))):
                pass
    else:
        for i in range(len(temps)):
            one_temperature(i)

    notify("Will run SEPARATEDF to merge the output in a single file for every standard turbulent velocity (0, 1, 2, 4 and 8 km/s)", silent)
    for j, v in enumerate(DFSYNTHE_VTURB):
        _write(output_dir + "/separatedf.com", _separatedf_script(output_dir, c["bin"], v))
        _bash(output_dir + "/separatedf.com")
        if not (os.path.isfile(output_dir + "/p00big" + v + ".bdf") and os.path.isfile(output_dir + "/p00lit" + v + ".bdf")):
            raise ValueError("SEPARATEDF did not output expected files")
        notify(v + " km/s done! (" + str(j + 1) + "/" + str(len(DFSYNTHE_VTURB)) + ")", silent)
    notify("SEPARATEDF halted", silent)
    notify("Finished running DFSYNTHE in " + str(datetime.now() - t0) + " s", silent)

    notify("Will run KAPPA9 for every standard turbulent velocity", silent)
    for i, v in enumerate(DFSYNTHE_VTURB):
        script = output_dir + "/kappa9v" + v + ".com"
        _write(script, _kappa9_script(c, v))
        _bash(script)
        notify(v + " km/s done! (" + str(i + 1) + "/" + str(len(DFSYNTHE_VTURB)) + ")", silent)

    _write(output_dir + "/kapreadts.com", _kapreadts_script(output_dir, c["bin"]))
    _bash(output_dir + "/kapreadts.com")
    notify("Merged all velocities in a single table. Final output saved in kappa.ros", silent)
    validate_run(output_dir, silent=silent)


# ============================================================================================================
# synthetic photometry
# ============================================================================================================
def synphot(run_dir, mag_system, reddening=0.0, Rv=3.1, filters_dir=python_path + "/data/filters/", spectrum=False,
            max_spill=1e-5, silent=False):
    """Bolometric corrections {filter: BC} of a SYNTHE run (or of ``spectrum`` = {'teff', 'wl' [A], 'flux'}) in
    'VEGAMAG' or 'ABMAG' (Gerasimov et al. 2022, eqs. 4-7), optionally reddened by E(B-V) = ``reddening``
    (scalar or 1D array; Gordon et al. 2023 law via dust_extinction). Filters whose throughput spills more than
    ``max_spill`` outside the spectrum (or outside the Vega reference) give nan."""
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    C = 71.197425 - 2.5 * np.log10(4 * np.pi * spc.sigma * (10 * spc.parsec) ** 2.0 * 1000 ** 4.0)
    c_aa = spc.c * 1e+10
    if type(spectrum) != bool:
        teff = spectrum["teff"]
    else:
        teff = meta(run_dir)["teff"]
        spectrum = read_spectrum(run_dir)
    if type(filters_dir) is str:
        filters = [os.path.join(filters_dir, fn) for fn in os.listdir(filters_dir)]
    else:
        filters = filters_dir
    try:
        reddening[0]
        is_array, reddenings = True, reddening
    except Exception:
        is_array, reddenings = False, [reddening]

    bc = {}
    for path in filters:
        f_wl, f_t = np.loadtxt(path, unpack=True)
        name = os.path.basename(path) if type(filters_dir) is str else path
        if mag_system == "VEGAMAG":
            ref_wl, ref_flux = np.loadtxt(python_path + "/data/vega_bohlin_2004.dat", unpack=True, delimiter=",")
            if f_wl.max() > ref_wl.max() or f_wl.min() < ref_wl.min():
                notify("Filter {} has exceed the wavelength range of the reference Vega spectrum".format(name), silent)
                bc[name] = np.nan
                continue
            order = np.argsort(f_wl)
            t_vega = np.interp(ref_wl, f_wl[order], f_t[order], left=0, right=0)
        elif mag_system == "ABMAG":
            ref_flux = (3631 * 1e-23 * c_aa) / (spectrum["wl"] ** 2)
        else:
            raise ValueError("Unknown magnitude system {}".format(mag_system))

        if f_wl.max() > spectrum["wl"].max() or f_wl.min() < spectrum["wl"].min():
            inside = (f_wl >= spectrum["wl"].min()) & (f_wl <= spectrum["wl"].max())
            norm = trapz(f_t, f_wl)
            if (norm - trapz(f_t[inside], f_wl[inside])) / norm > max_spill:
                notify("Filter {} has exceed the wavelength range of the model spectrum".format(name), silent)
                bc[name] = np.nan
                continue
        order = np.argsort(f_wl)
        t = np.interp(spectrum["wl"], f_wl[order], f_t[order], left=0, right=0)

        values = []
        for E in reddenings:
            if E != 0.0:
                try:
                    from dust_extinction.parameter_averages import G23
                    from astropy import units as u
                    law = G23(Rv=Rv)
                except Exception:
                    raise ValueError("Could not import the dust_extinction module. For extinguished photometry, make sure the module is installed ( https://dust-extinction.readthedocs.io/en/latest/ ).")
                wl = np.asarray(spectrum["wl"])
                A = np.zeros(len(wl))
                ok = (wl > 912) & (wl < 320000)
                A[ok] = law(wl[ok] * u.Angstrom) * E * Rv
            else:
                A = 0.0
            a = trapz(spectrum["wl"] * spectrum["flux"] * t * 10 ** (-A / 2.5) * np.pi, spectrum["wl"])
            if mag_system == "VEGAMAG":
                b = trapz(ref_wl * ref_flux * t_vega, ref_wl)
            else:
                b = trapz(spectrum["wl"] * ref_flux * t, spectrum["wl"])
            values.append(2.5 * np.log10(a / b) - 10 * np.log10(teff / 1000) + C)
        bc[name] = np.array(values)

    for name in bc:
        if is_array:
            if np.ndim(bc[name]) == 0:
                bc[name] = np.full([len(reddenings)], bc[name])
        elif np.ndim(bc[name]) > 0:
            bc[name] = bc[name][0]
    return bc
