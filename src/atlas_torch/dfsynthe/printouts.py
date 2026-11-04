"""Standard-output printouts of the DFSYNTHE-chain programs, as the stock (ifx) executables write them.

BasicATLAS parses two of these printouts (``meta()`` / ``meta_dfsynthe()`` and ``validate_run()``): the
READIN BEGIN listing of ``xnfdf.out`` and ``kapm40k2.out`` -- the abundance table (element symbol + log
abundance, FORMAT 1512) and the ``0XSCALE`` value on the line before it.  Those must be byte-identical to
the Fortran; this module reproduces the whole printouts:

``xnfdf_stdout``     PROGRAM XNFDF (xnfdf.out): continuum-edge listing (reading order, sorted, NUMNU), and per
                     model READMOL's MOLECULES INPUT listing (where a READ MOLECULES card is), the READIN
                     BEGIN listing (FORMATs 1512/1517/1518/1521/1561/1563/1564/1576), NMOLEC's tables
                     (112/113: number densities; 153: number densities / partition functions), the frequency
                     loop (FORMAT 15 + log10 continuum opacity, FORMAT 105) and the per-layer XNFPEL/DOPPLE
                     tables (FORMATs 280/290/333).
``dfsynthe_stdout``  PROGRAM DFSYNTHE (dfp00t*.out): the switches, DFINTERVALS (list-directed WBEGIN/WEND and
                     the interval table, incl. the Fortran's NN(I) instead of NN(I+328) for the little intervals),
                     VTTAB, the TEFF line, per skipped temperature LOGT, and for the computed one SELECTLINES'
                     NU/IWLTAB trace and selected-line counts and per layer the MLINES / broadening lines
                     (``selectlines_report`` recomputes the counts from the port's public line-selection pieces;
                     ``dfsynthe_job`` is the compat worker that also runs the temperature).  Known difference:
                     VTTAB prints the port's REAL*4 VTPROF, which differs from stock's in the last digit in 9 of
                     131 lines -- stock -O2 evaluates I*((1/V)*(2.99792458E5/500000.)) (hoisted invariant) and
                     VTPROF*(1/SUM); that reproduces all 131 lines (and differs from the port's VTPROF, used for the
                     turbulent broadening, in 730 of 1197 entries by up to 3e-6 relative).

The KAPPA9 printout (kapm40k2.out) is in :mod:`atlas_torch.kappa9.printout`; it shares the READIN/READMOL/
NMOLEC writers below.  DFSORTP and SEPARATEDF write nothing to standard output.

Numbers are formatted with :mod:`atlas_torch.atlas9.fformat` (correctly rounded digits, as both gfortran and
ifx produce them; 3-digit exponents drop the E: ``7.541-120``).  The printed values are the port's REAL*8
values; they agree with the stock build's to the printed precision except where a value sits on a rounding
boundary of the printed digits (stock is ifx -O2 fast-math; see tests/stage/test_dfsynthe_printouts.py for the
measured line differences).
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..atlas9 import fformat as ff

I, F, E, X = ff.I, ff.F, ff.E, ff.X


def E1(v, w, d):
    """1PEw.d"""
    return E(float(v), w, d, 1)


def _chunks(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]


def _ch(c) -> str:
    """A CHARACTER item: str / bytes, or a REAL*8 word holding Hollerith text (ELEM, TITLE)."""
    if isinstance(c, (bytes, bytearray)):
        return c.decode("latin-1")
    if isinstance(c, (float, np.floating)):
        return np.float64(c).tobytes().decode("latin-1")
    if isinstance(c, np.ndarray) and c.ndim == 0:
        return _ch(c.item())
    return str(c)


# ------------------------------------------------------------------------------------------------ READMOL
def readmol_listing(code, equil, nloc: int, nequa: int, maxmol: int, maxloc: int, maxeq: int) -> list[str]:
    """FORMATs 10, 14, 29 of READMOL. ``equil``: (NE, NUMMOL) with E1 (F7.3) first, the rest 1PE11.4."""
    out = ["1MOLECULES INPUT"]
    equil = np.asarray(equil, dtype=np.float64)
    for j, c in enumerate(code):
        out.append(I(j + 1, 5) + F(float(c), 18, 2) + F(float(equil[0, j]), 7, 3)
                   + "".join(E1(equil[k, j], 11, 4) for k in range(1, equil.shape[0])))
    out += [" MOLECULES  USED" + I(len(code), 4) + "  MAX" + I(maxmol, 4),
            " COMPONENTS USED" + I(nloc, 4) + "  MAX" + I(maxloc, 4),
            " EQUATIONS  USED" + I(nequa, 4) + "  MAX" + I(maxeq, 4)]
    return out


# ------------------------------------------------------------------------------------------------ READIN BEGIN
_IFOP_NAMES1 = ("0H1", " H2PLUS", " HMINUS", " HRAY", " HE1", " HE2", " HEMINUS", " HERAY", " COOL", " LUKE")
_IFOP_NAMES2 = (" HOT", " ELECTRON", " H2RAY", " HLINES", " LINES", " LINESCAT", " XLINES", " XLSCAT", " XCONT",
                " XSCAT")


def begin_listing(v: dict, variant: str, mode: int = 1) -> list[str]:
    """READIN's BEGIN printout (label 1500 on). ``v``: plain values (see the keys used below);
    ``variant``: 'xnfdf' or 'kappa9' (FORMATs 1518, 1563/1564, 1576 and the SUM print differ)."""
    out = ["1", "", "", "", "",
           " TEFF" + F(v["teff"], 7, 0) + "   LOG G" + F(v["glog"], 8, 4) + X(3) + ff.A(v["wlte"], 4),
           "0TITLE " + "".join(_ch(c)[:1] for c in v["title"])[:74].ljust(74)]
    elem, abund = v["elem"], v["abund"]
    out.append("0XSCALE" + F(v["xscale"], 10, 6) + "".join(X(3) + ff.A(_ch(elem[i]), 2) + F(abund[i], 8, 5)
                                                         for i in range(2)))
    for r0 in range(2, 99, 10):
        out.append("".join(X(1) + ff.A(_ch(elem[i]), 2) + F(abund[i], 6, 2) for i in range(r0, min(r0 + 10, 99))))
    ifop = [int(x) for x in v["ifop"]]
    out.append("".join(nm + I(ifop[i], 2) for i, nm in enumerate(_IFOP_NAMES1)))
    out.append("".join(nm + I(ifop[10 + i], 2) for i, nm in enumerate(_IFOP_NAMES2)))
    tauscat = "" if variant == "xnfdf" else "  TAUSCAT" + E1(v["tauscat"], 10, 2)
    out.append("0IFCORR" + I(v["ifcorr"], 2) + "  IFPRES" + I(v["ifpres"], 2) + "  IFSURF" + I(v["ifsurf"], 2)
               + "  IFSCAT" + I(v["ifscat"], 2) + tauscat + "  IFCONV" + I(v["ifconv"], 2)
               + "  MIXLTH" + F(v["mixlth"], 6, 2) + "  IFMOL" + I(v["ifmol"], 2))
    out.append(" IFTURB" + I(v["ifturb"], 2) + "  TRBFDG" + F(v["trbfdg"], 6, 2) + "  TRBPOW" + F(v["trbpow"], 6, 2)
               + "  TRBSND" + F(v["trbsnd"], 6, 2) + "  TRBCON" + F(v["trbcon"], 6, 2))
    if mode == 1:
        out.append(" NUMITS" + I(v["numits"], 3) + "  IFPRNT" + "".join(I(int(x), 2) for x in v["ifprnt"])
                   + "  IFPNCH" + "".join(I(int(x), 2) for x in v["ifpnch"]))
        if v["ifwave"] != 0:
            out.append("0WBEGIN" + F(v["wbegin"], 11, 4) + "   DELTAW" + F(v["deltaw"], 7, 4) + "   NUMNU"
                       + I(v["numnu"], 5))
        else:
            numnu = int(v["numnu"])
            out.append("0FREQID " + "".join(_ch(c)[:1] for c in v["freqid"])[:6].ljust(6) + "   NUMNU" + I(numnu, 4)
                       + "   NULO" + I(v["nulo"], 4) + "   NUHI" + I(v["nuhi"], 4))
            fre, rco = v["freset"], v["rcoset"]
            item = ((lambda nu: X(3) + I(nu, 3)) if variant == "xnfdf" else (lambda nu: I(nu, 6)))
            item3 = lambda nu: item(nu) + E1(fre[nu - 1], 13, 6) + E1(rco[nu - 1], 13, 6)
            nn = (numnu + 3) // 4
            nnn = numnu - nn * 3
            if nnn < 1:
                nn, nnn = 1, 1
            seq = [item3(nu) for i in range(1, nnn + 1) for nu in range(i, numnu + 1, nn)]
            out += ["".join(c) for c in _chunks(seq, 4)]
            if nn != nnn:
                seq = [item3(nu) for i in range(nnn + 1, nn + 1) for nu in range(i, numnu + 1, nn)]
                out += ["".join(c) for c in _chunks(seq, 3)]
            if variant == "kappa9":
                s = 0.0
                for nu in range(int(v["nulo"]), int(v["nuhi"]) + 1):
                    s = s + float(rco[nu - 1])
                out.append(E1(s, 13, 6))
    n = int(v["nrhox"])
    if variant == "xnfdf":
        head = (X(8) + "RHOX" + X(9) + "T" + X(8) + "P" + X(8) + "XNE" + X(6) + "ABROSS" + X(5) + "PRAD" + X(6)
                + "VTURB" + X(24) + "BHYD" + X(25) + "BMIN")
        j0, sep = (lambda j: I(j, 3)), X(1)
    else:
        head = (X(9) + "RHOX" + X(9) + "T" + X(8) + "P" + X(8) + "XNE" + X(6) + "ABROSS" + X(5) + "PRAD" + X(6)
                + "VTURB" + X(23) + "BHYD" + X(25) + "BMIN")
        j0, sep = (lambda j: I(j, 4)), ""
    out += ["1", "", "", "", "", head]
    bh, bm = v["bhyd"], v["bmin"]
    for j in range(n):
        out.append(j0(j + 1) + E1(v["rhox"][j], 13, 6) + F(v["t"][j], 9, 1)
                   + "".join(E1(a[j], 10, 3) for a in (v["p"], v["xne"], v["abross"], v["prad"], v["vturb"])) + sep
                   + "".join(F(bh[j, i], 8, 4) for i in range(6)) + F(bm[j], 8, 4))
    return out


def model_values(m) -> dict:
    """begin_listing values from an xnf.readin.Model (state at BEGIN)."""
    return dict(teff=m.teff, glog=m.glog, wlte=m.wlte, title=m.title, xscale=m.xscale, elem=list(m.elem),
                abund=[float(a) for a in m.abund], ifop=m.ifop, ifcorr=m.ifcorr, ifpres=m.ifpres, ifsurf=m.ifsurf,
                ifscat=m.ifscat, ifconv=m.ifconv, mixlth=m.mixlth, ifmol=m.ifmol, ifturb=m.ifturb,
                trbfdg=m.trbfdg, trbpow=m.trbpow, trbsnd=m.trbsnd, trbcon=m.trbcon, numits=m.numits,
                ifprnt=m.ifprnt, ifpnch=m.ifpnch, ifwave=m.ifwave, wbegin=m.wbegin, deltaw=m.deltaw,
                numnu=m.numnu, freqid=m.freqid, nulo=m.nulo, nuhi=m.nuhi, freset=m.freset, rcoset=m.rcoset,
                nrhox=m.nrhox, rhox=m.rhox, t=m.t, p=m.p, xne=m.xne, abross=m.abross, prad=m.prad,
                vturb=m.vturb, bhyd=m.b["bhyd"], bmin=m.b["bmin"])


# ------------------------------------------------------------------------------------------------ NMOLEC
def nmolec_tables(rhox, t, p, xne, xnatom, rho, code, xnmol, variant: str, xnfpmol=None) -> list[str]:
    """NMOLEC's FORMATs 112 and 113 (and 153 for XNFDF). ``code``: CODE(1..MAXMOL) (zeros past NUMMOL);
    ``xnmol``/``xnfpmol``: (NRHOX, >= NUMMOL) (columns past NUMMOL are printed as 0, as the COMMON holds)."""
    n = len(t)
    nummol = len(code)
    sp = (11, 50, 40) if variant == "xnfdf" else (10, 49, None)
    out = ["1" + X(sp[0]) + "RHOX" + X(9) + "T" + X(11) + "P" + X(10) + "XNE" + X(8) + "XNATOM" + X(8) + "RHO"]
    for j in range(n):
        out.append(I(j + 1, 5) + "".join(E1(a[j], 12, 3) for a in (rhox, t, p, xne, xnatom, rho)))

    def tables(title_line, arr):
        arr = np.asarray(arr, dtype=np.float64)
        nm = arr.shape[1]
        res = []
        for j1 in range(0, nummol, 10):
            res.append(title_line)
            res.append(X(5) + "".join(F(float(code[k]), 12, 2) for k in range(j1, j1 + 10)))
            for j in range(n):
                res.append(I(j + 1, 5) + "".join(E1(arr[j, k] if k < nm else 0.0, 12, 3) for k in range(j1, j1 + 10)))
        return res
    out += tables("1" + X(sp[1]) + "MOLECULAR NUMBER DENSITIES", xnmol)
    if xnfpmol is not None:
        out += tables("1" + X(sp[2]) + "MOLECULAR NUMBER DENSITIES / PARTITION FUNCTIONS", xnfpmol)
    return out


def padded_codes(code, maxmol: int) -> list[float]:
    """CODE(1..NN) with NN = NUMMOL rounded up to a multiple of 10 (READMOL zeroes CODE(1..MAXMOL))."""
    c = [float(x) for x in code]
    nn = -(-len(c) // 10) * 10
    return c + [0.0] * (min(nn, maxmol) - len(c))


# ------------------------------------------------------------------------------------------------ XNFDF
def _rec15(i, *vals) -> str:
    """FORMAT(I5,1PE25.15,0PF20.7,F20.7) for the given leading items."""
    s = I(i, 5)
    if len(vals) > 0:
        s += E1(vals[0], 25, 15)
    if len(vals) > 1:
        s += F(vals[1], 20, 7)
    if len(vals) > 2:
        s += F(vals[2], 20, 7)
    return s


def edges_listing(continua_text: str) -> list[str]:
    """Main program before READIN: PRINT 15,IN,EDGE + WRITE(6,15) per edge as read, the sorted table, NUMNU.

    Arithmetic as xnf.readin.read_edges (the -O0 semantics the port writes to xnfpdf.dat).  The stock -O2
    fast-math build differs in the 16th significant digit of FRQEDG (1PE25.15) for ~1/6 of the wavenumber
    edges (its FREEFF/EDGE arithmetic is reassociated; not emulated)."""
    from .xnf.freeform import CardStream, Free, xfreef
    unit = CardStream(continua_text)
    fr = Free(more=1, maxpow=99, last=81, numcol=1)
    card = [" "] * 82
    out, rows = [], []
    while True:
        fr.more = 1
        edge = xfreef(card, fr, unit)
        if edge == 0.:
            break
        out.append(_rec15(len(rows) + 1, edge))
        if abs(edge) < 1e6:
            w = edge
            c = 1e7 / edge
            f = 2.99792458e17 / w
        elif abs(edge) < 1e25:
            f = edge
            w = 2.99792458e17 / edge
            c = 1e7 / w
        else:
            c = edge / 1e25
            w = 1e7 / c
            f = 2.99792458e17 / w
        rows.append([abs(w), f, w, c])
        out.append(_rec15(len(rows), f, w, c))
    n = len(rows)
    for last in range(2, n + 1):           # DO 17 LAST=2,IN ; DO 16 I=2,IN-LAST+2 (bubble sort on |WLEDGE|)
        for i in range(1, n - last + 2):
            if not rows[i][0] >= rows[i - 1][0]:
                rows[i - 1], rows[i] = rows[i], rows[i - 1]
    out += [_rec15(i + 1, r[1], r[2], r[3]) for i, r in enumerate(rows)]
    out.append(I(3 * (n - 1), 5))
    return out


def _molecule_blocks(stdin: str) -> list[bool]:
    """Per BEGIN block of the XNFDF input: does it contain a READ MOLECULES card (READMOL prints there)."""
    flags, cur = [], False
    for line in stdin.splitlines():
        s = line.upper()
        if re.match(r"^\s*READ\s+MOLE", s):
            cur = True
        if re.match(r"^\s*BEGI", s):
            flags.append(cur)
            cur = False
    return flags


def xnfdf_model_lines(m, r, freqset, idmol, print_readmol: bool) -> list[str]:
    """Everything one READIN/BEGIN model of PROGRAM XNFDF prints. ``m``: xnf.readin.Model at BEGIN;
    ``r``: that model's entry of xnfdf.run()['results'] (needs xnmol/xnfpmol when IFMOL = 1)."""
    out = []
    if print_readmol and m.mol is not None:
        mol = m.mol
        out += readmol_listing(mol.code, mol.equil, len(mol.kcomps), mol.nequa, 200, 600, 30)
    out += begin_listing(model_values(m), "xnfdf", 1)
    nj = m.nrhox
    a = lambda k: r[k].numpy() if hasattr(r[k], "numpy") else np.asarray(r[k])
    t, rhox, p, xne, xnatom, rho = a("t"), m.rhox[:nj], m.p[:nj], a("xne"), a("xnatom"), a("rho")
    if m.ifmol == 1:
        code = padded_codes(m.mol.code, 200)
        out += nmolec_tables(rhox, t, p, xne, xnatom, rho, code, a("xnmol"), "xnfdf", a("xnfpmol"))
    else:   # MOLECULES OFF (never in BasicATLAS): NELECT's electron-contribution table is not reproduced
        out.append(" NELECT printout not reproduced (atlas_torch)")
    cont = a("continall")
    for nu in range(len(freqset)):
        freq = float(freqset[nu])
        wave = 2.99792458e17 / freq
        out.append(_rec15(nu + 1, freq, wave, 1e7 / wave))
        vals = [F(cont[j, nu], 5, 2) for j in range(nj)]
        out += [" " + "".join(c) for c in _chunks(vals, 20)]
    xnfpel, dop = a("xnfpel"), a("dopple")
    tk, hkt, tkev, tlog = a("tk"), a("hkt"), a("tkev"), a("tlog")
    xnfh, xnfhe, xnfh2 = a("xnfh"), a("xnfhe"), a("xnfh2")
    for j in range(nj):
        out.append("0XNFPEL.....DOPPLE" + I(j + 1, 10))
        for ne in range(99):
            s = I(ne + 1, 5) + "".join(E1(xnfpel[j, ion, ne], 12, 4) for ion in range(6)) + X(5) + E1(dop[j, 0, ne], 12, 4)
            if ne >= 39:
                s += E1(dop[j, 5, ne], 12, 4) + F(float(idmol[ne - 39]), 10, 2) + I((ne + 1) * 6, 5)
            out.append(s)
        out.append("0   T     " + E1(t[j], 12, 4) + "    TK    " + E1(tk[j], 12, 4) + "    HKT   " + E1(hkt[j], 12, 4)
                   + "    TKEV  " + E1(tkev[j], 12, 4) + "    TLOG  " + E1(tlog[j], 12, 4))
        out.append("0   RHOX  " + E1(rhox[j], 12, 4) + "    P     " + E1(p[j], 12, 4) + "    XNE   " + E1(xne[j], 12, 4)
                   + "    RHO   " + E1(rho[j], 12, 4))
        out.append("0   XNFH  " + E1(xnfh[j], 12, 4) + "    XNFHE " + E1(xnfhe[j, 0], 12, 4) + "    XNFH2 "
                   + E1(xnfh2[j], 12, 4))
    out.append("1")
    return out


def xnfdf_stdout(res: dict, stdin: str, continua_text: str) -> str:
    """Standard output of PROGRAM XNFDF for ``res = xnfdf.run(stdin, continua_text, ...)``."""
    from .xnf.fdata import T as tab
    idmol = tab("xnfdf", "idmol").astype(np.float64)
    out = edges_listing(continua_text)
    flags = _molecule_blocks(stdin)
    for i, (m, r) in enumerate(zip(res["models"], res["results"])):
        out += xnfdf_model_lines(m, r, res["freqset"], idmol, flags[i] if i < len(flags) else False)
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ list-directed
def ld_int(v) -> str:
    """ifx list-directed INTEGER*4 item (I12)."""
    return I(int(v), 12)


def ld_real8(v: float) -> str:
    """ifx list-directed REAL*8 item: G24.15E3 (F form for 0.1 <= |v| < 1e15, else 1PE24.15E3)."""
    v = float(v)
    a = abs(v)
    d, w, n = 15, 24, 5
    if a == 0.0:
        return F(v, w - n, d - 1) + X(n)
    if 0.1 - 0.5 * 10.0 ** (-d - 1) <= a < 10.0 ** d - 0.5:
        k = int(f"{a:.{d - 1}e}".split("e")[1]) + 1        # 10**(k-1) <= |v| (rounded to d digits) < 10**k
        return F(v, w - n, max(d - k, 0)) + X(n)
    s = f"{a:.{d - 1}e}"
    m, ex = s.split("e")
    ex = int(ex)
    return ((" -" if v < 0 else "  ") + m + "E" + ("-" if ex < 0 else "+") + f"{abs(ex):03d}").rjust(w)


# ------------------------------------------------------------------------------------------------ DFSYNTHE
def _xnfpdf_header(path) -> tuple[float, float, str]:
    """TEFF, GLOG, TITLE (74 A1 of REAL*8 words) of the first record of xnfpdf.dat."""
    import struct
    from ..io.fortran import iter_records
    with open(path, "rb") as f:
        r0 = next(iter_records(f))
    _, teff, glog = struct.unpack("<idd", r0[:20])
    title = "".join(chr(r0[20 + 8 * i]) for i in range(74))
    return teff, glog, title


def dfintervals_lines(fl=None) -> list[str]:
    """SUBROUTINE DFINTERVALS: WRITE(6,*) WBEGIN, WEND and FORMAT(4I10) per interval (328 big, 1212 little)."""
    from ..tables import table
    from .lines.f32 import Flavor
    from .lines.tables import SRC, dfintervals
    fl = fl or Flavor()
    nbeg, nend, _, nn = dfintervals()
    big = np.concatenate([table(SRC, "dfintervals", v) for v in ("biga", "bigb", "bigc")]).astype(np.float64)
    lit_ = np.concatenate([table(SRC, "dfintervals", v) for v in
                           ("lita", "litb", "litc", "litd", "lite", "litf", "litg")]).astype(np.float64)
    lr = math.log(1.0 + 1.0 / 500000.0)
    out = []
    for w, sl in ((big, slice(0, 328)), (lit_, slice(328, 1540))):
        nzero = int(math.log(w[0]) / lr) + 1
        r = 1.0 + 1.0 / 500000.0
        wbegin = float(fl.powi8_np(r, np.array([nzero], np.int64))[0])
        wend = float(fl.powi8_np(r, np.array([int(nend[sl][-1]) + nzero - 1], np.int64))[0])
        out += [ld_real8(wbegin), ld_real8(wend)]
        b, e = nbeg[sl], nend[sl]
        # Fortran quirk: the little-interval loop prints NN(I), not NN(I+328) (I <= 328: a big interval's width)
        out += [I(i + 1, 10) + I(b[i], 10) + I(e[i], 10) + I(nn[i], 10) for i in range(len(b))]
    return out


def vttab_lines(fl=None) -> list[str]:
    """SUBROUTINE VTTAB: FORMAT(I3,' KM/S',10F10.7/(8X,10F10.7))."""
    from .lines.tables import vttab
    vtprof, vtcenter, nvt = vttab(fl)
    out = []
    for iv in range(20):
        vals = [F(float(vtcenter[iv]), 10, 7)] + [F(float(vtprof[i, iv]), 10, 7) for i in range(int(nvt[iv]))]
        rows = _chunks(vals, 10)
        out.append(I(iv + 1, 3) + " KM/S" + "".join(rows[0]))
        out += [X(8) + "".join(r) for r in rows[1:]]
    return out


def selectlines_report(xnfpdf, xnfpdfmax, data_dir, iteff: int, fl=None, device=None,
                       chunk: int = 2_000_000) -> dict:
    """What PROGRAM DFSYNTHE prints for the computed temperature, recomputed from the port's public pieces:
    the SELECTLINES NU/IWLTAB trace of the low-line list, the selected-line counts per list, and per layer MLINES
    (lines that pass the KAPPA0 >= CONTINUUM(NBUFF) test: XLINOP's normal (TYPE 0, 3, < -2), hydrogen (-1, -2)
    and autoionizing (1) lines, and the TAPE12 lines).  Returns dict(nu_lines, counts, mlines, t, p)."""
    import torch
    from . import odf_io
    from .lines.continuum import continuum, layer_setup
    from .lines.f32 import F32, Flavor
    from .lines.selectlines import iter_tape12, select_consts, select_tape19
    fl = fl or Flavor()
    dev = device or torch.device("cpu")
    xt = odf_io.read_xnfpdf(xnfpdf, iteff)
    mx = odf_io.read_xnfpdfmax(xnfpdfmax, iteff)
    # NU trace (lowlines loop of SELECTLINES: NU advances while IWL >= IWLTAB(NU), printing NU, IWLTAB(NU))
    c = select_consts(mx.teff, mx.wledge, fl)
    _, name, count = odf_io.LINE_FILES["low"]
    low = odf_io.open_linelist(__import__("pathlib").Path(data_dir) / name)
    mx_iwl = -(1 << 62)
    for s in range(0, count, chunk):
        mx_iwl = max(mx_iwl, int(np.asarray(low[s:min(count, s + chunk)]["iwl"]).max()))
    nu_final = int(np.searchsorted(c.iwltab, mx_iwl, side="right")) + 1      # Fortran NU after the loop
    nu_lines = [ld_int(nu) + ld_int(c.iwltab[nu - 1]) for nu in range(2, nu_final + 1)]
    # per-layer MLINES
    lay = layer_setup(xt, fl, dev)
    cont = continuum(xt.wledge, lay["ablog"], fl)
    nl = cont.shape[0]
    xnfdop, xnfpel = lay["xnfdop"], lay["xnfpel"]
    tape19 = select_tape19(data_dir, mx.teff, mx.wledge, mx.xnfdopmax, fl)
    ml = torch.zeros(nl, dtype=torch.int64, device=dev)
    if len(tape19):
        typ = tape19["type"]
        nb = torch.from_numpy(tape19["nbuff"].astype(np.int64)).to(dev) - 1
        nel = torch.from_numpy(tape19["nelion"].astype(np.int64)).to(dev) - 1
        gf = torch.from_numpy(tape19["gf"].astype(np.float32)).to(dev)
        cnb = cont[:, nb]
        norm = torch.from_numpy((typ == 0) | (typ == 3) | (typ < -2)).to(dev)
        hyd = torch.from_numpy((typ == -1) | (typ == -2)).to(dev)
        auto = torch.from_numpy(typ == 1).to(dev)
        g_auto = torch.from_numpy((tape19["gammaw"].astype(np.float32) * tape19["gf"].astype(np.float32))
                                  .astype(np.float32)).to(dev)
        k_norm = gf[None, :] * xnfdop[:, nel]
        k_h = gf[None, :] * xnfdop[:, 0:1]
        k_a = g_auto[None, :] * xnfpel[:, nel]
        passed = (norm[None, :] & ~(k_norm < cnb)) | (hyd[None, :] & ~(k_h < cnb)) | (auto[None, :] & ~(k_a < cnb))
        ml += passed.sum(1)
        del cnb, k_norm, k_h, k_a, passed
    counts: dict = {}
    for _, t in iter_tape12(data_dir, mx.teff, mx.wledge, mx.xnfdopmax, fl, chunk, counts):
        if not len(t):
            continue
        nbt = torch.from_numpy(t.nbuff.astype(np.int64)).to(dev) - 1
        nelt = torch.from_numpy(t.nelion.astype(np.int64)).to(dev) - 1
        cg = torch.from_numpy(t.congf.astype(np.float32)).to(dev)
        for j0 in range(0, nl, 5):
            sl = slice(j0, min(nl, j0 + 5))
            k0 = cg[None, :] * xnfdop[sl][:, nelt]
            ml[sl] += (~(k0 < cont[sl][:, nbt])).sum(1)
    counts["nlte"] = len(tape19)
    st = xt.state
    return dict(nu_lines=nu_lines, counts=counts, mlines=ml.cpu().numpy(), t=st["t"][:xt.nrhox].astype(np.float32),
                p=st["p"][:xt.nrhox].astype(np.float32), nrhox=xt.nrhox)


_COUNT_LABELS = (("low", "LOWLINES"), ("hi", "HILINES"), ("diat", "DIATOMICS"), ("tio", "TIOLINES"),
                 ("h2o", "H2OLINES"))


def dfsynthe_stdout(switches: str, xnfpdf, report: dict | None = None, fl=None) -> str:
    """Standard output of PROGRAM DFSYNTHE (dfp00t*.out) for the 57 switches ``switches`` (READ(5,'(57I1)');
    the ones set must have a ``report`` = selectlines_report(...) -- one temperature is what BasicATLAS runs)."""
    import torch
    from . import odf_io
    sw = [int(ch) if ch.isdigit() else 0 for ch in (switches.split("\n")[0] + " " * 57)[:57]]
    out = [" " + "".join(str(s) for s in sw)]
    out += dfintervals_lines(fl)
    out += vttab_lines(fl)
    teff, glog, title = _xnfpdf_header(xnfpdf)
    out.append(" TEFF=" + F(teff, 10, 1) + X(3) + "GRAV=" + F(glog, 6, 3) + X(3) + title)
    log10f = lambda x: float(torch.log10(torch.tensor(np.float32(x), dtype=torch.float32)))
    for it in range(1, 58):
        if not sw[it - 1]:
            t1 = odf_io.read_xnfpdf(xnfpdf, it).state["t"][0]
            out += ["", "", I(it, 3) + " LOGT" + F(log10f(t1), 5, 2)]
            continue
        if report is None:
            raise ValueError("dfsynthe_stdout: temperature {} needs a selectlines_report".format(it))
        r = report
        out += r["nu_lines"]
        n12 = 0
        for key, lab in _COUNT_LABELS:
            if key in r["counts"]:
                out.append(I(r["counts"][key], 10) + " LINES FROM " + lab)
                n12 += int(r["counts"][key])
        out.append(I(r["counts"]["nlte"], 10) + " LINES FROM NLTELINESA")
        out.append(I(n12 + int(r["counts"]["nlte"]), 10) + " LINES TOTAL")
        for j in range(int(r["nrhox"])):
            head = " LOGT" + F(log10f(r["t"][j]), 5, 2) + "   LOGP" + F(log10f(r["p"][j]), 5, 2) + "   VT"
            out += ["", "", head + I(0, 3), head + I(0, 3) + "  SPECTRUM COMPLETED WITH" + I(r["mlines"][j], 8)
                    + " LINES"]
            out += [head + I(v, 3) + "  SPECTRUM BROADENED" for v in (1, 2, 4, 8)]
    return "\n".join(out) + "\n"


def dfsynthe_job(args, switches: str) -> str:
    """compat worker (spawned process): atlas_torch.dfsynthe.pipeline._temperature_job (ODF steps -> parts
    file) plus the DFSYNTHE printout of that temperature, written to ``<part>.out``; returns its path."""
    import torch
    from .ftz import scope
    from .lines.f32 import Flavor
    from .pipeline import _temperature_job
    xnfpdf, xnfpdfmax, data_dir, it, cfgd, part = args
    _temperature_job(args)
    fl = Flavor(cfgd.get("flavor", "stock"), fastmath=cfgd.get("fastmath"))
    dev = torch.device(cfgd.get("device", "cpu"))
    with scope(bool(cfgd.get("ftz", True)) and dev.type == "cpu"):
        rep = selectlines_report(xnfpdf, xnfpdfmax, data_dir, it, fl, dev)
    text = dfsynthe_stdout(switches, xnfpdf, rep, fl)
    p = str(part) + ".out"
    with open(p, "w", encoding="latin-1", newline="") as f:
        f.write(text)
    return p
