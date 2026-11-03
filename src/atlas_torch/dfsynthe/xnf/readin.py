"""READIN (xnfdf.for variant), READMOL and the continuum-edge reader of PROGRAM XNFDF.

READIN is a card interpreter; it is scalar bookkeeping (no layer arithmetic except the BEGIN block), so
it is ported as a Python state machine over an explicit `Model` state that holds the COMMON members it
reads/writes (Fortran shapes, numpy float64). The persistent state (abundances, switches, molecule table,
departure coefficients ...) carries over between successive READIN calls exactly as the COMMONs do.

Implemented control cards: TEFF GRAVITY OPACITY(IFOP/ON/OFF) ITERATIONS MOLECULES ABUNDANCE(SCALE/CHANGE)
PRINT PUNCH READ(FREQUENCIES/DEPARTURE COEFFICIENTS/DECK/DECK6/MOLECULES) LTE NLTE BEGIN SCATTERING END
TITLE CONVECTION TURBULENCE CHANGE(RHOX) SURFACE PRESSURE CORRECTION WAVELENGTH CALL.
Not implemented (need TTAUP, the Kitt Peak Rosseland table): CALCULATE, READ STARTING T-TAU, SCALE MODEL.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ...interp import map1
from .fdata import T as tab
from .fmath import F, LN10_R4, rdiv
from .freeform import CardStream, Free, FortranExit, card_from_line, freeff, freefr, iwordf

KW = 99
MAXMOL = 200

# MIAC codes (base-37 machine independent alphameric codes, first 6 letters)
M = dict(TEFF=1020133, GRAVITY=519223721, OPACITY=1070201044, KAPPA=20688921, ITERATIONS=661856797,
         MOLECULES=930198669, CALCULATE=210518764, ABUNDANCE=74175307, PRINT=30911189, PUNCH=31069574,
         READ=918640, LTE=17173, NLTE=726315, BEGIN=4011517, SCATTERING=1323236444, END=7367,
         TITLE=37966926, CONVECTION=236883734, TURBULENCE=1427151802, CHANGE=223095242,
         SURFACE=1357812572, PRESSURE=1143518210, CORRECTION=237080870, WAVELENGTH=1597906832,
         SCALE=35762836, CALL=153784, ON=569, OFF=20763, IFOP=464662, FREQUENCIES=450075960,
         DEPARTURE=287559136, STARTING=1355094447, DECK=209579, DECK6=7754456, INTENSITY=651354309,
         FLUX=321147)


def exp10(x: float) -> float:
    """READIN statement function EXP10(X)=EXP(X*2.30258509299405E0) (single-precision constant)."""
    return math.exp(x * LN10_R4)


@dataclass
class Model:
    """COMMON state touched by READIN / the BEGIN block (one model)."""
    # /ELEM/
    abund: np.ndarray = None
    atmass: np.ndarray = None
    elem: np.ndarray = None
    xabund: np.ndarray = field(default_factory=lambda: np.zeros(99))
    wtmole: float = 0.0
    # /RHOX/ /TEMP/ /STATE/ /ABROSS/ /RAD/ /TURBPR/ /PZERO/
    nrhox: int = 0
    rhox: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    t: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    p: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    xne: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    xnatom: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    rho: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    ptotal: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    abross: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    tauros: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    accrad: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    prad: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    vturb: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    pturb: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    pradk: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    pradk0: float = 0.0
    pturb0: float = 0.0
    pcon: float = 0.0
    pzero: float = 0.0
    tk: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    hkt: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    hckt: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    tkev: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    tlog: np.ndarray = field(default_factory=lambda: np.zeros(KW))
    # /TEFF/ /FLUX/
    teff: float = 0.0
    grav: float = 0.0
    glog: float = 0.0
    flux: float = 0.0
    # /IF/ /ITER/
    ifcorr: int = 1
    ifpres: int = 1
    ifsurf: int = 0
    ifscat: int = 1
    ifmol: int = 0
    nlteon: int = 0
    ifop: list = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    numits: int = 0
    ifprnt: list = field(default_factory=lambda: [2] * 15)
    ifpnch: list = field(default_factory=lambda: [0] * 15)
    # /JUNK/
    title: list = field(default_factory=lambda: [" "] * 74)
    freqid: list = field(default_factory=lambda: [" "] * 6)
    wlte: str = "LTE "
    xscale: float = 1.0
    # /CONV/ /TURBPR/ switches
    ifconv: int = 0
    mixlth: float = 1.0
    ifturb: int = 0
    trbfdg: float = 0.0
    trbpow: float = 0.0
    trbsnd: float = 0.0
    trbcon: float = 0.0
    # /FRESET/ /MUS/
    freset: np.ndarray = field(default_factory=lambda: np.zeros(500))
    rcoset: np.ndarray = field(default_factory=lambda: np.zeros(500))
    nulo: int = 0
    nuhi: int = 0
    numnu: int = 0
    ifwave: int = 0
    wbegin: float = 0.0
    deltaw: float = 0.0
    nmu: int = 1
    angle: np.ndarray = field(default_factory=lambda: np.zeros(20))
    # departure coefficients (/BHYD/ BHYD,BMIN and the metal B arrays), BLOCKR DATA = 1
    b: dict = field(default_factory=dict)
    # molecules (/XNMOL/ CODE, /IFEQUA/), set by READ MOLECULES
    mol: "Molecules | None" = None

    @classmethod
    def initial(cls) -> "Model":
        m = cls()
        m.abund = tab("blocke", "abund").astype(np.float64).copy()
        m.atmass = tab("blocke", "atmass").astype(np.float64).copy()
        m.elem = tab("blocke", "elem").copy()
        for name, nlev in B_ARRAYS.items():
            m.b[name] = np.ones((KW, nlev))
        m.b["bmin"] = np.ones(KW)
        return m

    def copy(self) -> "Model":
        return copy.deepcopy(self)


B_ARRAYS = {"bhyd": 8, "bhe1": 29, "bhe2": 6, "bc1": 14, "bc2": 6, "bmg1": 11, "bmg2": 6, "bal1": 9, "bal2": 1,
            "bsi1": 11, "bsi2": 10, "bca1": 8, "bca2": 5, "bna1": 8, "bo1": 13, "bo2": 4, "bb1": 7, "bfe1": 15,
            "bk1": 8}
_IFB_TARGET = [None, "bhyd", "bmin", "bhe1", "bhe2", "bc1", "bmg1", "bal1", "bsi1", "bc2", "bal2", "bmg2", "bsi2",
               "bca1", "bca2", "bna1", "bo1", "bb1", "bfe1", "bo2", "bk1"]


# ---------------------------------------------------------------------------------------------------
# READMOL
# ---------------------------------------------------------------------------------------------------
@dataclass
class Molecules:
    code: np.ndarray        # CODE(NUMMOL)
    equil: np.ndarray       # EQUIL(7, NUMMOL)
    locj: np.ndarray        # LOCJ(NUMMOL+1) (1-based component pointers)
    kcomps: np.ndarray      # KCOMPS(NLOC) -> equation numbers (1-based; NEQUA+1 = inverse XNE)
    idequa: np.ndarray      # IDEQUA(1..NEQUA) element id of each equation (index 0 unused)
    nequa: int
    ifequa: np.ndarray      # IFEQUA(1..101)

    @property
    def nummol(self):
        return len(self.code)


def _fortran_f(field_: str, d: int) -> float:
    """Fw.d / Ew.d input conversion of one field (blank = 0, implied decimal point)."""
    s = field_.strip()
    if not s:
        return 0.0
    s = s.replace("D", "E").replace("d", "e")
    if "." in s:
        return float(s)
    # no decimal point: implied d decimals on the mantissa; exponent optional (also "1+05" form)
    mant, exp = s, 0
    for k in range(1, len(s)):
        if s[k] in "Ee":
            mant, exp = s[:k], int(s[k + 1:])
            break
        if s[k] in "+-" and s[k - 1] not in "Ee":
            mant, exp = s[:k], int(s[k:])
            break
    return float(f"{mant}e{exp - d}")


def readmol(text: str) -> Molecules:
    """READMOL: FORMAT(F18.2,F7.3,6E11.4) records until CODE=0."""
    xcode = tab("readmol", "xcode")
    codes, eqs, kcomps, locj = [], [], [], [1]
    ifequa = np.zeros(102, dtype=np.int64)
    for line in text.splitlines():
        line = line.rstrip("\n").ljust(84)
        c = _fortran_f(line[0:18], 2)
        if c == 0.0:
            break
        e = [_fortran_f(line[18:25], 3)] + [_fortran_f(line[25 + 11 * k:36 + 11 * k], 4) for k in range(6)]
        ii = next((i for i in range(8) if c >= xcode[i]), None)
        if ii is None:
            raise FortranExit("READMOL: bad code")
        x = c
        for i in range(ii, 8):
            idv = int(x / xcode[i])
            x = x - float(idv) * xcode[i]
            if idv == 0:
                idv = 100
            ifequa[idv] = 1
            kcomps.append(idv)
        ion = int(x * 100.0 + F(.5))
        if ion >= 1:
            ifequa[100] = 1
            ifequa[101] = 1
            kcomps.extend([101] * ion)
        locj.append(len(kcomps) + 1)
        codes.append(c)
        eqs.append(e)
        if len(codes) > MAXMOL:
            raise FortranExit("TOO MANY MOLECULES")
    ieq = 1
    idequa = np.zeros(32, dtype=np.int64)
    for i in range(1, 101):
        if ifequa[i] == 0:
            continue
        ieq += 1
        ifequa[i] = ieq
        idequa[ieq] = i
    nequa = ieq
    ifequa[101] = nequa + 1
    kc = np.array([ifequa[i] for i in kcomps], dtype=np.int64)
    return Molecules(np.array(codes), np.array(eqs).T.copy(), np.array(locj, dtype=np.int64), kc, idequa,
                     nequa, ifequa)


# ---------------------------------------------------------------------------------------------------
# INTEG / PARCOE (identical to atlas7v)
# ---------------------------------------------------------------------------------------------------
def parcoe(f, x, n):
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    c[0] = 0.
    b[0] = (f[1] - f[0]) / (x[1] - x[0])
    a[0] = f[0] - x[0] * b[0]
    n1 = n - 1
    c[n - 1] = 0.
    b[n - 1] = (f[n - 1] - f[n1 - 1]) / (x[n - 1] - x[n1 - 1])
    a[n - 1] = f[n - 1] - x[n - 1] * b[n - 1]
    if n == 2:
        return a, b, c
    for j in range(1, n1):          # Fortran J=2..N1
        j1 = j - 1
        d = (f[j] - f[j1]) / (x[j] - x[j1])
        c[j] = f[j + 1] / ((x[j + 1] - x[j]) * (x[j + 1] - x[j1])) - f[j] / ((x[j] - x[j1]) * (x[j + 1] - x[j])) \
            + f[j1] / ((x[j] - x[j1]) * (x[j + 1] - x[j1]))
        b[j] = d - (x[j] + x[j1]) * c[j]
        a[j] = f[j1] - x[j1] * d + x[j] * x[j1] * c[j]
    c[1] = 0.
    b[1] = (f[2] - f[1]) / (x[2] - x[1])
    a[1] = f[1] - x[1] * b[1]
    c[2] = 0.
    b[2] = (f[3] - f[2]) / (x[3] - x[2])
    a[2] = f[2] - x[2] * b[2]
    for j in range(1, n1):
        if c[j] == 0.:
            continue
        j1 = j + 1
        wt = abs(c[j1]) / (abs(c[j1]) + abs(c[j]))
        a[j] = a[j1] + wt * (a[j] - a[j1])
        b[j] = b[j1] + wt * (b[j] - b[j1])
        c[j] = c[j1] + wt * (c[j] - c[j1])
    a[n1 - 1] = a[n - 1]
    b[n1 - 1] = b[n - 1]
    c[n1 - 1] = c[n - 1]
    return a, b, c


def integ(x, f, n, start):
    a, b, c = parcoe(f, x, n)
    fint = np.zeros(n)
    fint[0] = start
    for i in range(n - 1):
        fint[i + 1] = fint[i] + (a[i] + b[i] / 2. * (x[i + 1] + x[i]) +
                                 c[i] / 3. * ((x[i + 1] + x[i]) * x[i + 1] + x[i] * x[i])) * (x[i + 1] - x[i])
    return fint


# ---------------------------------------------------------------------------------------------------
# READIN
# ---------------------------------------------------------------------------------------------------
class Readin:
    """Card interpreter; `read(mode)` performs one CALL READIN(MODE) and returns the model state
    (a deep copy) on BEGIN, or None on END with MODE=20 (MODE 1/2 END -> FortranExit)."""

    def __init__(self, stdin: str, molecules_text: str | None = None, state: Model | None = None):
        self.unit5 = CardStream(stdin)
        self.molecules_text = molecules_text
        self.m = state or Model.initial()
        self.free = Free()
        self.card = card_from_line("")

    # helpers ---------------------------------------------------------------------------------------
    def _ff(self):
        return freeff(self.card, self.free)

    def _fr(self):
        return freefr(self.card, self.free, self.unit5)

    def _w(self):
        return iwordf(self.card, self.free)

    def _readcard(self):
        new = self.unit5.read()
        self.card[1:81] = new[1:81]

    def read(self, mode: int = 1) -> Model | None:
        m, fr = self.m, self.free
        self.card[81] = " "
        fr.last = 81
        fr.maxpow = 38
        state = 98
        while True:
            if state == 98:
                fr.more = 0
                fr.letcol = 1
                self._readcard()
                miac = self._w()
                fr.numcol = fr.letcol
            elif state == 97:
                fr.letcol = max(fr.letcol, fr.numcol)
                fr.more = 1
                miac = self._w()
                if fr.iffail == 1:
                    state = 98
                    continue
                fr.more = 0
            state = self._dispatch(miac, mode)
            if state == "return":
                return m.copy()
            if state == "end":
                return None

    def _dispatch(self, miac, mode):
        m, fr = self.m, self.free
        if miac == M["TEFF"]:
            m.teff = self._ff()
            m.flux = 5.6697e-5 / F(12.5664) * m.teff ** 4
            return 97
        if miac == M["GRAVITY"]:
            m.grav = self._ff()
            if m.grav < 10.:
                m.grav = exp10(m.grav)
            m.glog = math.log10(m.grav)
            return 97
        if miac == M["OPACITY"]:
            miac = self._w()
            if miac == M["IFOP"]:
                fr.numcol = fr.letcol
                m.ifop = [int(self._ff()) for _ in range(20)]
                return 98
            if miac in (M["ON"], M["OFF"]):
                iswch = 1 if miac == M["ON"] else 0
                fr.more = 1
                codes = list(tab("readin", "ifop1"))
                while True:
                    miac = self._w()
                    if fr.iffail == 1:
                        return 97
                    if miac not in codes:
                        raise FortranExit("I DO NOT UNDERSTAND " + "".join(self.card[1:81]))
                    m.ifop[codes.index(miac)] = iswch
            raise FortranExit("I DO NOT UNDERSTAND " + "".join(self.card[1:81]))
        if miac == M["ITERATIONS"]:
            m.numits = int(self._ff())
            m.ifpnch = [0] * 15
            m.ifpnch[m.numits - 1] = 1
            return 97
        if miac == M["MOLECULES"]:
            miac = self._w()
            if miac == M["ON"]:
                m.ifmol = 1
            elif miac == M["OFF"]:
                m.ifmol = 0
            else:
                raise FortranExit("I DO NOT UNDERSTAND")
            return 97
        if miac == M["ABUNDANCE"]:
            miac = self._w()
            if miac == M["SCALE"]:
                fr.numcol = fr.letcol
                m.xscale = self._ff()
                return 97
            if miac == M["CHANGE"]:
                fr.more = 1
                while True:
                    iz = int(self._ff())
                    if fr.iffail == 1:
                        return 98
                    m.abund[iz - 1] = self._ff()
            raise FortranExit("I DO NOT UNDERSTAND")
        if miac == M["PRINT"]:
            for i in range(m.numits):
                m.ifprnt[i] = int(self._ff())
            return 97
        if miac == M["PUNCH"]:
            for i in range(m.numits):
                m.ifpnch[i] = int(self._ff())
            return 97
        if miac == M["READ"]:
            return self._read_cmd()
        if miac == M["LTE"]:
            m.nlteon = 0
            m.wlte = "LTE "
            m.b["bhyd"][:40, :6] = 1.
            m.b["bmin"][:40] = 1.
            return 97
        if miac == M["NLTE"]:
            m.nlteon = 1
            m.wlte = "NLTE"
            return 97
        if miac == M["BEGIN"]:
            self._begin(mode)
            return "return"
        if miac == M["SCATTERING"]:
            miac = self._w()
            m.ifscat = {M["ON"]: 1, M["OFF"]: 0}[miac]
            return 97
        if miac == M["END"]:
            if mode != 20:
                raise FortranExit("END")
            m.nrhox = 0
            return "end"
        if miac == M["TITLE"]:
            m.title = self.card[7:81]
            return 98
        if miac == M["CONVECTION"]:
            miac = self._w()
            if miac == M["ON"]:
                m.ifconv = 1
                fr.numcol = fr.letcol
                m.mixlth = self._ff()
            elif miac == M["OFF"]:
                m.ifconv = 0
                m.mixlth = 1.
            else:
                raise FortranExit("I DO NOT UNDERSTAND")
            return 97
        if miac == M["TURBULENCE"]:
            miac = self._w()
            if miac == M["ON"]:
                m.ifturb = 1
                fr.numcol = fr.letcol
                m.trbfdg, m.trbpow, m.trbsnd, m.trbcon = (self._ff() for _ in range(4))
            elif miac == M["OFF"]:
                m.ifturb = 0
                m.trbfdg = m.trbpow = m.trbsnd = m.trbcon = 0.
            else:
                raise FortranExit("I DO NOT UNDERSTAND")
            return 97
        if miac == M["CHANGE"]:
            return self._change_rhox()
        if miac == M["SURFACE"]:
            miac = self._w()
            if miac == M["INTENSITY"]:
                m.nmu = int(self._ff())
                for mu in range(m.nmu):
                    m.angle[mu] = self._ff()
                m.ifsurf = 2
            elif miac == M["FLUX"]:
                m.ifsurf, m.nmu = 1, 1
            elif miac == M["OFF"]:
                m.ifsurf, m.nmu = 0, 1
            else:
                raise FortranExit("I DO NOT UNDERSTAND")
            return 97
        if miac == M["PRESSURE"]:
            m.ifpres = {M["ON"]: 1, M["OFF"]: 0}[self._w()]
            return 97
        if miac == M["CORRECTION"]:
            m.ifcorr = {M["ON"]: 1, M["OFF"]: 0}[self._w()]
            return 97
        if miac == M["WAVELENGTH"]:
            m.wbegin, m.deltaw, wend = self._ff(), self._ff(), self._ff()
            m.ifwave, m.nulo = 1, 1
            m.nuhi = int((wend - m.wbegin) / abs(m.deltaw) + F(.5)) + 1
            m.numnu = m.nuhi
            if m.wbegin < 1e7:
                return 97
            if m.wbegin > 1e20:
                m.wbegin = F(1.e7) / (m.wbegin / 1e25)
                m.deltaw = F(1.e7) / (m.deltaw / 1e25)
                return 97
            m.wbegin = 2.99792458e17 / m.wbegin
            m.deltaw = 2.99792458e17 / m.deltaw
            return 97
        if miac == M["CALL"]:
            return 97
        if miac in (M["CALCULATE"], M["SCALE"]):
            raise NotImplementedError("READIN CALCULATE / SCALE MODEL need TTAUP (not ported)")
        raise FortranExit("I DO NOT UNDERSTAND " + "".join(self.card[1:81]))

    def _read_cmd(self):
        m, fr = self.m, self.free
        miac = self._w()
        fr.numcol = fr.letcol
        if miac == M["FREQUENCIES"]:
            num = int(self._ff())
            m.nulo = int(self._ff())
            m.nuhi = int(self._ff())
            m.numnu = num
            fr.letcol = fr.numcol
            self._w()
            m.freqid = list(fr.word)
            fr.numcol = fr.letcol
            for _ in range(m.numnu):
                nu = int(self._fr())
                v = self._ff()
                if v < 1e7:
                    v = 2.99792458e17 / v
                if v > F(1.e20):
                    v = v * (F(2.99792458e10) / 1e25)
                m.freset[nu - 1] = v
                m.rcoset[nu - 1] = self._ff()
            return 98
        if miac == M["DEPARTURE"]:
            m.nrhox = int(self._ff())
            for j in range(m.nrhox):
                fr.numcol = 1
                self._readcard()
                self._ff()
                for i in range(6):
                    m.b["bhyd"][j, i] = self._ff()
                m.b["bmin"][j] = self._ff()
            m.wlte = "NLTE"
            m.nlteon = 1
            return 98
        if miac == M["STARTING"]:
            raise NotImplementedError("READ STARTING T-TAU needs TTAUP (not ported)")
        if miac in (M["DECK"], M["DECK6"]):
            m.nrhox = int(self._ff())
            for j in range(m.nrhox):
                fr.numcol = 1
                self._readcard()
                m.rhox[j] = self._ff()
                m.t[j] = self._ff()
                fr.more = 1
                m.p[j] = self._ff()
                m.xne[j] = self._ff()
                m.abross[j] = self._ff()
                m.prad[j] = self._ff()
                m.vturb[j] = self._ff()
                fr.more = 0
            n = m.nrhox
            if m.rhox[0] < 0.:
                m.rhox[:n] = [exp10(v) for v in m.rhox[:n]]
            m.pradk0 = 0.
            m.pturb0 = m.pturb[0]
            m.pcon = 0.
            if miac != M["DECK6"]:
                return 98
            self._readcard()
            fr.numcol = 1
            m.pradk0 = self._ff()
            m.accrad[:n] = m.prad[:n]
            m.prad[:n] = integ(m.rhox[:n], m.accrad[:n], n, m.accrad[0] * m.rhox[0])
            m.pradk[:n] = m.prad[:n] + m.pradk0
            return 98
        if miac == M["MOLECULES"]:
            if self.molecules_text is None:
                raise FortranExit("READ MOLECULES without fort.2")
            m.mol = readmol(self.molecules_text)
            return 98
        raise FortranExit("I DO NOT UNDERSTAND " + "".join(self.card[1:81]))

    def _change_rhox(self):
        m, fr = self.m, self.free
        nnew = int(self._ff())
        rhoxa = np.array([self._fr() for _ in range(nnew)])
        n = m.nrhox
        x = torch.from_numpy(m.rhox[:n].copy())
        xn = torch.from_numpy(rhoxa)

        def mp(v):
            return map1(x, torch.from_numpy(np.ascontiguousarray(v[:n])), xn)[0].numpy()
        t, p, xne, ab, vt, pr, bm = (mp(v) for v in (m.t, m.p, m.xne, m.abross, m.vturb, m.prad, m.b["bmin"]))
        m.t[:nnew], m.p[:nnew], m.xne[:nnew], m.abross[:nnew], m.vturb[:nnew], m.prad[:nnew] = t, p, xne, ab, vt, pr
        m.pradk[:nnew] = pr + m.pradk0
        m.b["bmin"][:nnew] = bm
        for i in range(6):
            m.b["bhyd"][:nnew, i] = mp(m.b["bhyd"][:, i])
        m.nrhox = nnew
        m.rhox[:nnew] = rhoxa
        return 97

    def _begin(self, mode):
        m = self.m
        if mode == 1:
            if m.numits == 0 or m.nrhox == 0 or m.numnu == 0 or m.teff == 0. or m.grav == 0.:
                raise FortranExit("READIN BEGIN: missing NUMITS/NRHOX/NUMNU/TEFF/GRAVITY")
        if m.abund[0] < 0.:
            m.abund[0] = exp10(m.abund[0])
        if m.abund[1] < 0.:
            m.abund[1] = exp10(m.abund[1])
        for iz in range(2, 99):
            if m.abund[iz] > 0.:
                m.abund[iz] = math.log10(m.abund[iz])
        for iz in range(2, 99):
            m.xabund[iz] = exp10(m.abund[iz]) * m.xscale
        m.xabund[0] = m.abund[0]
        m.xabund[1] = m.abund[1]
        w = 0.
        for iz in range(99):
            w = w + m.xabund[iz] * m.atmass[iz]
        m.wtmole = w
        n = m.nrhox
        t = torch.from_numpy(m.t[:n].copy())
        tk = 1.38054e-16 * t
        hkt = rdiv(6.6256e-27, tk)
        m.tk[:n] = tk.numpy()
        m.hkt[:n] = hkt.numpy()
        m.hckt[:n] = (hkt * 2.99792458e10).numpy()
        m.tkev[:n] = (8.6171e-5 * t).numpy()
        m.tlog[:n] = torch.log(t).numpy()
        m.xnatom[:n] = m.p[:n] / m.tk[:n] - m.xne[:n]
        m.rho[:n] = m.xnatom[:n] * m.wtmole * 1.660e-24
        m.pturb[:n] = F(.5) * m.rho[:n] * m.vturb[:n] ** 2


# ---------------------------------------------------------------------------------------------------
# continuum edges (PROGRAM XNFDF, unit 17) and the frequency set
# ---------------------------------------------------------------------------------------------------
def read_edges(text: str):
    """Edges read with XFREEF from fort.17 until a 0; sorted by |wavelength| (bubble sort, stable).

    Returns (FRQEDG, WLEDGE, CMEDGE) numpy arrays (IN,) and FREQSET (NUMNU=3*(IN-1),)."""
    from .freeform import xfreef
    unit = CardStream(text)
    fr = Free(more=1, maxpow=99, last=81, numcol=1)
    card = [" "] * 82
    frq, wl, cm, a = [], [], [], []
    while True:
        fr.more = 1
        edge = xfreef(card, fr, unit)
        if edge == 0.:
            break
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
        frq.append(f); wl.append(w); cm.append(c); a.append(abs(w))
    n = len(a)
    a = np.array(a); frq = np.array(frq); wl = np.array(wl); cm = np.array(cm)
    for last in range(2, n + 1):           # DO 17 LAST=2,IN ; DO 16 I=2,IN-LAST+2
        for i in range(1, n - last + 2):
            if a[i] >= a[i - 1]:
                continue
            a[[i - 1, i]] = a[[i, i - 1]]
            frq[[i - 1, i]] = frq[[i, i - 1]]
            wl[[i - 1, i]] = wl[[i, i - 1]]
            cm[[i - 1, i]] = cm[[i, i - 1]]
    fs = []
    f1 = F(1.0000001)
    for i in range(n - 1):
        fs.append(abs(frq[i]) / f1)
        fs.append(2.99792458e17 / (abs(wl[i]) + abs(wl[i + 1])) * 2.)
        fs.append(abs(frq[i + 1]) * f1)
    return frq, wl, cm, np.array(fs)
