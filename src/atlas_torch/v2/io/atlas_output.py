"""ATLAS9 run products as BasicATLAS reads them: output_main.out (the READIN echo of the input with the frequency
set, then one iteration table per iteration between START TABLE / END TABLE, atlas9mem.for PUTOUT(5) FORMAT 541/542
as patched by BasicATLAS), output_last_iteration.out (the rows of the best iteration, BasicATLAS atlas_converged)
and output_summary.out (its punched deck)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..atmosphere import Atmosphere
from .atlas_deck import AtlasDeck, format_atlas_deck

# READIN's element labels (ELEM) for Z = 1..99
ELEMENTS = ("H ", "HE", "LI", "BE", "B ", "C ", "N ", "O ", "F ", "NE", "NA", "MG", "AL", "SI", "P ", "S ", "CL", "AR",
            "K ", "CA", "SC", "TI", "V ", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR",
            "RB", "SR", "Y ", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I ", "XE",
            "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF",
            "TA", "W ", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
            "PA", "U ", "NP", "PU", "AM", "CM", "BK", "CF", "ES")
IFOP = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0)
IFOP_NAMES = ("H1", "H2PLUS", "HMINUS", "HRAY", "HE1", "HE2", "HEMINUS", "HERAY", "COOL", "LUKE", "HOT", "ELECTRON",
              "H2RAY", "HLINES", "LINES", "LINESCAT", "XLINES", "XLSCAT", "XCONT", "XSCAT")
TABLE_COLUMNS = ("RHOX", "T", "P", "XNE", "RHO", "ABROSS", "HEIGHT", "TAUROS", "FLXCNV", "PRADK", "FLXERR", "FLXDRV")
TABLE_HEAD = (
    "0                                    ELECTRON             ROSSELAND    HEIGHT   ROSSELAND             RADIATION"
    "        PER CENT FLUX",
    "        RHOX      TEMP    PRESSURE    NUMBER    DENSITY      MEAN       (KM)      DEPTH    CONV FLUX  PRESSURE"
    "        ERROR    DERIV")


def fortran_e(x: float, w: int, d: int, scale: int = 1) -> str:
    """Fortran Ew.d with scale factor 1P (d.ddd..E+ee) or 0P (0.ddd..E+ee), right-justified in w."""
    x = float(x)
    if not np.isfinite(x):
        return ("NaN" if np.isnan(x) else ("Infinity" if x > 0 else "-Infinity")).rjust(w)
    if scale == 1:
        s = f"{x:.{d}E}"
    else:
        if x == 0.0:
            s = "0." + "0" * d + "E+00"
        else:
            m, e = f"{abs(x):.{d - 1}E}".split("E")
            s = ("-" if x < 0 else "") + "0." + m.replace(".", "") + f"E{int(e) + 1:+03d}"
    return s.rjust(w)


def fortran_f(x: float, w: int, d: int) -> str:
    """Fortran Fw.d (F7.0 prints '  5777.')."""
    s = f"{float(x):.{d}f}" + ("." if d == 0 else "")
    return s.rjust(w)


def atlas_title(text: str) -> str:
    """The 74-character TITLE as READIN stores it: each character of card columns 7-43 followed by a blank."""
    return "".join(ch + " " for ch in text)[:74].ljust(74)


def format_iteration_table(record: dict, teff: float, logg: float, title: str, iteration: int) -> list[str]:
    """PUTOUT(5) of the patched atlas9mem.for: START TABLE, header (FORMAT 541, 542), one row per layer
    (I4,1X,1PE17.5,1X,0PE17.5,1X,1P8E17.5,1X,0PE15.3,1X,E15.3), END TABLE."""
    out = [" START TABLE", f" TEFF{fortran_f(teff, 8, 0)}   LOG G{logg:9.5f}" + " " * 10 + atlas_title(title) + "  ITERATION"
           + f"{iteration:3d}", *TABLE_HEAD]
    cols = [np.asarray(record[k], dtype=np.float64).reshape(-1) for k in TABLE_COLUMNS]
    for j in range(len(cols[0])):
        v = [c[j] for c in cols]
        row = f"{j + 1:4d} " + fortran_e(v[0], 17, 5) + " " + fortran_e(v[1], 17, 5, 0) + " " \
            + "".join(fortran_e(x, 17, 5) for x in v[2:10]) + " " + fortran_e(v[10], 15, 3, 0) + " " \
            + fortran_e(v[11], 15, 3, 0)
        out.append(row)
    out.append(" END TABLE")
    return out


def readin_echo(spec, frequencies=None, n_iterations: int = 15) -> list[str]:
    """READIN's echo of the model input (atlas9mem.for READIN FORMAT 1512-1564): TEFF / LOG G, TITLE, XSCALE with
    H and He, the log10 abundances of Z = 3..99 before the scale, the IFOP switches, the IFCORR / IFTURB line,
    NUMITS, and the frequency set FREQID with (index, nu, weight) in four columns and the weight sum.
    frequencies: (nu, weight) in Hz (default ATLAS9's 337 BIG frequencies)."""
    if frequencies is None:
        from ..opacity.frequencies import FrequencyQuadrature
        q = FrequencyQuadrature.atlas_big()
        frequencies = (q.nu, q.weight)
    nu, w = (np.asarray(torch.as_tensor(v).detach().cpu(), dtype=np.float64).reshape(-1) for v in frequencies)
    h, he, logm = spec.composition.atlas_cards()
    out = ["1", "", "", "", "", f" TEFF{fortran_f(spec.teff, 7, 0)}   LOG G{spec.logg:8.4f}" + " " * 7,
           "0TITLE " + atlas_title(spec.title),
           f"0XSCALE{spec.composition.scale:10.6f}   {ELEMENTS[0]}{h:8.5f}   {ELEMENTS[1]}{he:8.5f}"]
    ab = [f" {ELEMENTS[z]}{logm[z - 2]:6.2f}" for z in range(2, 99)]
    out += ["".join(ab[k:k + 10]) for k in range(0, len(ab), 10)]
    sw = [f"{n}{i:2d}" for n, i in zip(IFOP_NAMES, IFOP)]
    out += ["0" + " ".join(sw[:10]), " " + " ".join(sw[10:])]
    turb = spec.turbulence or (0.0, 0.0, 0.0, 0.0)
    out += [f"0IFCORR 1  IFPRES 1  IFSURF 0  IFSCAT 1  TAUSCAT{fortran_e(0.0, 10, 2)}  IFCONV{int(spec.convection):2d}"
            f"  MIXLTH{spec.mixing_length:6.2f}  IFMOL 1",
            f" IFTURB{int(spec.turbulence is not None):2d}" + "".join(
                f"  {k}{v:6.2f}" for k, v in zip(("TRBFDG", "TRBPOW", "TRBSND", "TRBCON"), turb)),
            f" NUMITS{min(n_iterations, 999):3d}  IFPRNT" + " 1" * 15 + "  IFPNCH" + " 1" * 15]
    n = len(nu)
    out.append(f"0FREQID {'':6s}   NUMNU{n:4d}   NULO{1:4d}   NUHI{n:4d}")
    rows = (n + 3) // 4
    for i in range(rows):
        out.append("".join(f"{k + 1:6d}{nu[k]:13.6E}{w[k]:13.6E}" for k in range(i, n, rows)))
    out.append(f"{w.sum():13.6E}")
    return out


def select_best_iteration(max_err, max_de) -> int:
    """BasicATLAS atlas_converged: the iteration (0-based) with the smallest max|FLXERR| among the GOLD ones
    (max|err| < 1, max|de| < 10), else SILVER (< 10, < 100), else BRONZE (err < 1000), else all."""
    err, de = np.asarray(max_err, dtype=np.float64), np.asarray(max_de, dtype=np.float64)
    err = np.where(np.isfinite(err), err, 88888.888)
    de = np.where(np.isfinite(de), de, 88888.888)
    for cls in ((err < 1.0) & (de < 10.0), (err < 10.0) & (de < 100.0), err < 1000.0, np.ones_like(err, bool)):
        if cls.any():
            idx = np.flatnonzero(cls)
            return int(idx[np.argmin(err[idx])])
    return 0


def summary_deck(spec, record: dict, iteration: int) -> AtlasDeck:
    """The punched model (PUTOUT(5) FORMAT 552-571) of one iteration record (deck_* columns, PRADK0)."""
    t = lambda k: torch.as_tensor(np.asarray(record["deck_" + k], dtype=np.float64))   # noqa: E731
    atm = Atmosphere(rhox=t("RHOX"), T=t("T"), P=t("P"), n_e=t("XNE"), vturb=t("VTURB"),
                     extra={k: t(k) for k in ("ABROSS", "ACCRAD", "FLXRAD", "VCONV", "VELSND")})
    conv = "ON " if spec.convection else "OFF"
    turb = spec.turbulence
    tcard = (" TURBULENCE ON " + "".join(f"{x:6.2f}" for x in turb)) if turb else \
        " TURBULENCE OFF" + "  0.00" * 4
    head = [f"TEFF {fortran_f(spec.teff, 7, 0)}  GRAVITY{spec.logg:8.5f}     ", "TITLE " + atlas_title(spec.title),
            " OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0",
            f" CONVECTION {conv}{spec.mixing_length:6.2f}" + tcard]
    tail = [f"PRADK{fortran_e(float(np.asarray(record['PRADK0'])), 11, 4)}",
            f"BEGIN                    ITERATION{iteration:4d} COMPLETED"]
    return AtlasDeck(atm, spec.composition, head, [], tail)


def write_atlas_run(directory, spec, records: list[dict], best: int, stopped: str = "", frequencies=None) -> dict:
    """output_main.out (the READIN echo with the frequency set (nu, weight) Hz, then every iteration's table),
    output_last_iteration.out (the best one's rows), output_summary.out (its punched deck) in `directory`;
    returns their paths."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    main = readin_echo(spec, frequencies, len(records))
    flux = 5.670374419e-5 * spec.teff ** 4 / (4.0 * np.pi)
    tables = []
    for r in records:
        tab = format_iteration_table(r, spec.teff, spec.logg, spec.title, int(r.get("iteration_in_batch",
                                                                                   r.get("iteration", 0))))
        tables.append(tab)
        main.append("0" + " " * 108 + "FLUX" + fortran_e(flux, 12, 4))
        main.extend(tab)
    if stopped == "gold":
        main.append(" REACHED GOLD TOLERANCES")
    elif stopped == "diverged":
        main.append(" MODEL DIVERGED")
    paths = {"main": d / "output_main.out", "last": d / "output_last_iteration.out",
             "summary": d / "output_summary.out"}
    paths["main"].write_text("\n".join(main) + "\n")
    paths["last"].write_text("\n".join(tables[best][4:-1]))
    rec = records[best]
    deck = summary_deck(spec, rec, int(rec.get("iteration_in_batch", rec.get("iteration", 0))))
    paths["summary"].write_text(format_atlas_deck(deck))
    return paths
