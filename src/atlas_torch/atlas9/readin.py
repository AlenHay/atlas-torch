"""READIN (atlas9mem.for, MODE=1): control-card interpreter, plus READMOL, READ KAPPA (unit 1), READ PUNCH
(unit 3) and the BEGIN-time initialisation and printout.

Every card of the atlas9 READIN is handled. Output written to unit 6 is appended to `io.stdout` with the
exact FORMATs. Quirks reproduced:
  * TITLE writes ITITLE(1..148) through a 74-element array: the overflow fills FREQID, WLTE, the padding of
    /JUNK/ and the first 60 bytes of /MUS/ (ANGLE(1..7) and half of ANGLE(8)) -- modelled byte-for-byte.
  * READ PUNCH always reads unit 3 (the OPEN statements are commented out); FILENAME becomes 'fort'.
  * SCALE's "same model" test uses REAL*4 literals 1.001/.999; with GRAVITY nan (BasicATLAS restart
    templates) the model is always rescaled and re-integrated with TTAUP.
  * FREEFF's single-precision powers of ten (see cards.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..fnum import f32, fint
from ..state import State
from . import data, fmath
from . import fformat as ff
from .cards import Free, StdinReader, read_int, read_real
from .errors import FortranExit
from .numerics import integ, map1b
from .rosstab import Rosstab, read_kappa_ros
from .ttaup import ttaup

LN10 = 2.30258509299405      # EXP10 statement function of READIN (D literal)
SIG4PI = None


@dataclass
class IO:
    """The program's units: 5 = stdin cards, 6 = stdout lines, 7 = punch lines; 1/2/3/9 = input files."""
    stdin: StdinReader
    files: dict = field(default_factory=dict)        # unit -> path
    stdout: list = field(default_factory=list)
    punch: list = field(default_factory=list)
    _unit2: list | None = None


# ----------------------------------------------------------------------------- small State helpers

def _sv(s: State, key: str, v):
    cur = s[key]
    if isinstance(cur, np.ndarray):
        a = cur.copy()
        a.reshape(-1)[0] = v
        s[key] = a
    else:
        s[key] = torch.tensor([v], dtype=cur.dtype)


def _arr(s: State, key: str) -> torch.Tensor:
    return s[key].clone()


def flux_of(teff: float) -> float:
    """FLUX=5.6697D-5/12.5664*TEFF**4 (12.5664 is a REAL*4 literal; TEFF**4 = (T*T)*(T*T))."""
    t2 = teff * teff
    return 5.6697e-5 / f32(12.5664) * (t2 * t2)


def _chars(s: State, key: str) -> list[str]:
    a = s[key]
    return [b.decode("latin-1") if b else "\0" for b in np.asarray(a).reshape(-1)]


# ----------------------------------------------------------------------------- READMOL

def readmol(s: State, io: IO) -> None:
    """READMOL: molecule list from unit 2 -> /IFEQUA/ (in place on s)."""
    path = io.files[2]
    lines = Path(path).read_text(encoding="latin-1").split("\n")
    xcode = data.table("readmol", "xcode").astype(np.float64)
    ifequa = np.zeros(101, dtype=np.int64)
    kcomps = s["ifequa.kcomps"].numpy().copy()
    locj = s["ifequa.locj"].numpy().copy()
    code = s["ifequa.code"].numpy().copy()
    equil = s["ifequa.equil"].numpy().copy()
    idequa = s["ifequa.idequa"].numpy().copy()
    kloc = 1
    locj[0] = 1
    li = 0
    jmol = 1
    while jmol <= 161:
        if kloc > 450:
            io.stdout.append("1TOO MANY MOLECULES")
        ln = lines[li] if li < len(lines) else None
        if ln is None:
            raise FortranExit("Fortran runtime error: End of file", 2)
        li += 1
        ln = ln.ljust(80)
        c = read_real(ln[0:18], 2)
        e = [read_real(ln[18:25], 3)] + [read_real(ln[25 + 11 * k:36 + 11 * k], 4) for k in range(5)]
        if c == 0.0:
            break
        ii = None
        for k in range(8):
            if c >= xcode[k]:
                ii = k
                break
        if ii is None:
            raise FortranExit("", 0)
        x = c
        for i in range(ii, 8):
            idd = int(x / xcode[i] + 0.5)
            x = x - float(idd) * xcode[i]
            if idd == 0:
                idd = 100
            ifequa[idd - 1] = 1
            kcomps[kloc - 1] = idd
            kloc += 1
        ion = int(x * 100.0 + 0.5)
        if ion >= 1:
            ifequa[99] = 1
            ifequa[100] = 1
            for _ in range(ion):
                kcomps[kloc - 1] = 101
                kloc += 1
        locj[jmol] = kloc
        code[jmol - 1] = c
        equil[:, jmol - 1] = e
        jmol += 1
    else:
        io.stdout.append("1TOO MANY MOLECULES")
    nummol = jmol - 1
    nloc = kloc - 1
    iequa = 1
    for i in range(100):
        if ifequa[i] == 0:
            continue
        iequa += 1
        ifequa[i] = iequa
        idequa[iequa - 1] = i + 1
    nequa = iequa
    ifequa[100] = nequa + 1
    for k in range(nloc):
        kcomps[k] = ifequa[kcomps[k] - 1]
    io.stdout.append(" MOLECULES  USED" + ff.I(nummol, 4) + "  MAX" + ff.I(160, 4))
    io.stdout.append(" COMPONENTS USED" + ff.I(nloc, 4) + "  MAX" + ff.I(450, 4))
    io.stdout.append(" EQUATIONS  USED" + ff.I(nequa, 4) + "  MAX" + ff.I(25, 4))
    t = lambda a: torch.from_numpy(np.asarray(a))
    s.update({"ifequa.ifequa": t(ifequa), "ifequa.kcomps": t(kcomps), "ifequa.locj": t(locj),
              "ifequa.code": t(code), "ifequa.equil": t(equil), "ifequa.idequa": t(idequa),
              "ifequa.nequa": torch.tensor([nequa]), "ifequa.nummol": torch.tensor([nummol]),
              "ifequa.nloc": torch.tensor([nloc])})


# ----------------------------------------------------------------------------- READ PUNCH (unit 3)

def read_punch(s: State, io: IO) -> None:
    lines = Path(io.files[3]).read_text(encoding="latin-1").split("\n")
    pos = [0]

    def rec() -> str:
        if pos[0] >= len(lines):
            raise FortranExit("Fortran runtime error: End of file", 2)
        ln = lines[pos[0]].rstrip("\r")
        pos[0] += 1
        return ln.ljust(200)

    r = rec()
    teff = read_real(r[5:12], 0)
    glog = read_real(r[21:29], 5)
    wlte = r[30:34]
    r = rec()
    title = list(r[6:80])
    r = rec()
    ifop = [read_int(r[13 + 2 * i:15 + 2 * i]) for i in range(20)]
    r = rec()
    a = r[12:15]
    mixlth = read_real(r[15:21], 2)
    b = r[33:36]
    trb = [read_real(r[36 + 6 * i:42 + 6 * i], 2) for i in range(4)]
    r = rec()
    xscale = read_real(r[16:25], 5)
    abund = s["elem.abund"].clone()
    abund[0] = read_real(r[44:52], 5)
    abund[1] = read_real(r[54:62], 5)
    iz = 2
    while iz < 99:
        r = rec()
        for k in range(6):
            if iz >= 99:
                break
            abund[iz] = read_real(r[17 + 10 * k + 3:17 + 10 * k + 10], 2)
            iz += 1
    _sv(s, "teff.teff", teff)
    _sv(s, "teff.glog", glog)
    s["junk.wlte"] = np.array([wlte.encode("latin-1")], dtype="S4")
    s["junk.title"] = np.array([c.encode("latin-1") for c in title], dtype="S1")
    s["ifop.ifop"] = torch.tensor(ifop, dtype=torch.int64)
    _sv(s, "conv.mixlth", mixlth)
    _sv(s, "turbpr.trbfdg", trb[0]); _sv(s, "turbpr.trbpow", trb[1])
    _sv(s, "turbpr.trbsnd", trb[2]); _sv(s, "turbpr.trbcon", trb[3])
    _sv(s, "junk.xscale", xscale)
    s["elem.abund"] = abund
    grav = fmath.spow(10.0, glog)
    _sv(s, "teff.grav", grav)
    _sv(s, "flux.flux", flux_of(teff))
    if wlte == "LTE ":
        _sv(s, "depart.nlteon", 0)
    if wlte == "NLTE":
        _sv(s, "depart.nlteon", 1)
    if a == "OFF":
        _sv(s, "conv.ifconv", 0)
    if a == "ON ":
        _sv(s, "conv.ifconv", 1)
    if b == "OFF":
        _sv(s, "turbpr.ifturb", 0)
    if b == "ON ":
        _sv(s, "turbpr.ifturb", 1)
    r = rec()
    nrhox = read_int(r[10:13])
    rhox, t, p, xne, abross, accrad, vturb = (_arr(s, k) for k in (
        "rhox.rhox", "temp.t", "state.p", "state.xne", "abross.abross", "rad.accrad", "turbpr.vturb"))
    for j in range(nrhox):
        r = rec()
        rhox[j] = read_real(r[0:15], 8)
        t[j] = read_real(r[15:24], 1)
        for arr, k in ((p, 0), (xne, 1), (abross, 2), (accrad, 3), (vturb, 4)):
            arr[j] = read_real(r[24 + 10 * k:34 + 10 * k], 3)
    _sv(s, "rhox.nrhox", nrhox)
    s.update({"rhox.rhox": rhox, "temp.t": t, "state.p": p, "state.xne": xne, "abross.abross": abross,
              "rad.accrad": accrad, "turbpr.vturb": vturb})
    r = rec()
    pradk0 = read_real(r[5:16], 4)
    _sv(s, "pzero.pradk0", pradk0)
    n = nrhox
    prad = s["rad.prad"].clone()
    prad[:n] = integ(rhox[:n], accrad[:n], accrad[0] * rhox[0])
    pradk = s["pzero.pradk"].clone()
    pradk[:n] = prad[:n] + pradk0
    tauros = s["abross.tauros"].clone()
    tauros[:n] = integ(rhox[:n], abross[:n], abross[0] * rhox[0])
    s.update({"rad.prad": prad, "pzero.pradk": pradk, "abross.tauros": tauros})
    if s.iscalar("depart.nlteon") == 1:
        r = rec()
        n2 = read_int(r[27:30])
        bhyd = s["depart.bhyd"].clone(); bmin = s["depart.bmin"].clone(); rx = s["rhox.rhox"].clone()
        for j in range(n2):
            r = rec()
            rx[j] = read_real(r[0:11], 4)
            for i in range(6):
                bhyd[j, i] = read_real(r[11 + 9 * i:20 + 9 * i], 4)
            bmin[j] = read_real(r[65:74], 4)
        _sv(s, "rhox.nrhox", n2)
        s.update({"depart.bhyd": bhyd, "depart.bmin": bmin, "rhox.rhox": rx})
    rec()   # BEGIN ... ITERATION nn COMPLETED
    s["filename.filename"] = np.array([b"fort".ljust(60)], dtype="S60")


# ----------------------------------------------------------------------------- READIN

def _title_overflow(s: State, card) -> None:
    """TITLE: ITITLE(2*I-1)=CARD(I+6), ITITLE(2*I)=' ' for I=1..74 with ITITLE(74) EQUIVALENCEd to TITLE:
    bytes past TITLE land in FREQID, WLTE, the padding of /JUNK/ (size 96) and then /MUS/ ANGLE."""
    mem = bytearray(96 + 64)
    title = np.asarray(s["junk.title"]).reshape(-1)
    freqid = np.asarray(s["junk.freqid"]).reshape(-1)
    wlte = np.asarray(s["junk.wlte"]).reshape(-1)[0]
    for i in range(74):
        mem[8 + i] = (title[i] or b"\0")[0]
    for i in range(6):
        mem[82 + i] = (freqid[i] or b"\0")[0]
    mem[88:92] = wlte.ljust(4, b"\0")[:4]
    angle = s["mus.angle"].numpy().copy()
    mem[96:96 + 64] = angle[:8].tobytes()
    for i in range(1, 75):
        ch = card[i + 6].encode("latin-1")
        mem[7 + 2 * i - 1] = ch[0]       # ITITLE(2I-1) at offset 8+(2I-1)-1
        mem[7 + 2 * i] = 32              # ITITLE(2I)
    s["junk.title"] = np.array([bytes([mem[8 + i]]) for i in range(74)], dtype="S1")
    s["junk.freqid"] = np.array([bytes([mem[82 + i]]) for i in range(6)], dtype="S1")
    s["junk.wlte"] = np.array([bytes(mem[88:92])], dtype="S4")
    angle[:8] = np.frombuffer(bytes(mem[96:160]), dtype="<f8")
    s["mus.angle"] = torch.from_numpy(angle)


def readin(s: State, io: IO, mode: int = 1) -> State:
    """SUBROUTINE READIN(MODE): returns the new State (s is not modified)."""
    s = State(s)
    fr = Free(io.stdout)
    card = Free.make_card(" " * 80)

    def miac_dispatch_needed():
        pass

    def read_card():
        nonlocal card
        card = Free.make_card(io.stdin.read())

    fr.last = 81
    fr.maxpow = 38
    state = "98"
    miac = 0
    while True:
        if state == "98":
            fr.more = 0
            fr.letcol = 1
            read_card()
            miac = fr.iwordf(card)
            fr.numcol = fr.letcol
            state = "3"
        elif state == "97":
            fr.letcol = max(fr.letcol, fr.numcol)
            fr.more = 1
            miac = fr.iwordf(card)
            if fr.iffail == 1:
                state = "98"
                continue
            fr.more = 0
            state = "3"
        if state != "3":
            continue
        # ---- dispatch (label 3)
        nxt = _dispatch(s, io, fr, card, miac, mode, read_card_cb=lambda: _reread(io, fr, card))
        if nxt == "return":
            _sync_free(s, fr)
            return s
        state = nxt


def _reread(io: IO, fr: Free, card: list) -> None:
    new = Free.make_card(io.stdin.read())
    card[:] = new


def _sync_free(s: State, fr: Free) -> None:
    s.update({"free.numcol": torch.tensor([fr.numcol]), "free.letcol": torch.tensor([fr.letcol]),
              "free.last": torch.tensor([fr.last]), "free.more": torch.tensor([fr.more]),
              "free.iffail": torch.tensor([fr.iffail]), "free.maxpow": torch.tensor([fr.maxpow])})
    s["free.word"] = np.array([w.encode("latin-1") for w in fr.word], dtype="S1")


def _not_understood(io: IO, card):
    io.stdout.append(" I DO NOT UNDERSTAND " + "".join(card[1:81]))
    raise FortranExit("", 0)


def _exp10(x: float) -> float:
    return fmath.sexp(x * LN10)


def _dispatch(s: State, io: IO, fr: Free, card, miac: int, mode: int, read_card_cb) -> str:
    F = lambda: fr.freeff(card)
    kw = s["rhox.rhox"].shape[0]
    if miac == 1020133:                                     # TEFF
        teff = F()
        _sv(s, "teff.teff", teff)
        _sv(s, "flux.flux", flux_of(teff))
        return "97"
    if miac == 519223721:                                   # GRAVITY
        grav = F()
        if grav < 10.0:
            grav = _exp10(grav)
        _sv(s, "teff.grav", grav)
        _sv(s, "teff.glog", fmath.slog10(grav))
        return "97"
    if miac == 1070201044:                                  # OPACITY
        m = fr.iwordf(card)
        if m == 464662:                                     # IFOP
            fr.numcol = fr.letcol
            s["ifop.ifop"] = torch.tensor([fint(F()) for _ in range(20)], dtype=torch.int64)
            return "98"
        if m in (569, 20763):
            iswch = 1 if m == 569 else 0
            fr.more = 1
            ifop1 = data.readin_ifop1()
            ifop = s["ifop.ifop"].clone()
            while True:
                m = fr.iwordf(card)
                if fr.iffail == 1:
                    s["ifop.ifop"] = ifop
                    return "97"
                hit = np.nonzero(ifop1 == m)[0]
                if hit.size == 0:
                    s["ifop.ifop"] = ifop
                    _not_understood(io, card)
                ifop[int(hit[0])] = iswch
        _not_understood(io, card)
    if miac == 20688921:                                    # KAPPA
        _not_understood(io, card)
    if miac == 661856797:                                   # ITERATIONS
        numits = fint(F())
        _sv(s, "iter.numits", numits)
        ifpnch = torch.zeros(15, dtype=torch.int64)
        ifpnch[numits - 1] = 1
        s["iter.ifpnch"] = ifpnch
        return "97"
    if miac == 930198669:                                   # MOLECULES
        m = fr.iwordf(card)
        if m == 569:
            _sv(s, "if.ifmol", 1); return "97"
        if m == 20763:
            _sv(s, "if.ifmol", 0); return "97"
        _not_understood(io, card)
    if miac == 210518764:                                   # CALCULATE
        nrhox = fint(F()); tau1lg = F(); steplg = F()
        _sv(s, "rhox.nrhox", nrhox); _sv(s, "steplg.tau1lg", tau1lg); _sv(s, "steplg.steplg", steplg)
        teff = s.scalar("teff.teff")
        tauros = s["abross.tauros"].clone(); t = s["temp.t"].clone()
        for j in range(nrhox):
            tauros[j] = _exp10(tau1lg + float(j) * steplg)
        x = tauros[:nrhox]
        inner = f32(.75) * (f32(.710) + x - f32(.1331) * fmath.exp(f32(-3.4488) * x))
        t[:nrhox] = teff * fmath.pow(inner, f32(.25))
        s.update({"abross.tauros": tauros, "temp.t": t})
        _label702(s, io)
        return "97"
    if miac == 74175307:                                    # ABUNDANCE
        m = fr.iwordf(card)
        if m == 35762836:                                   # SCALE
            fr.numcol = fr.letcol
            _sv(s, "junk.xscale", F())
            return "97"
        if m in (223095242, 74075686):                      # CHANGE / ABSOLUTE
            fr.more = 1
            abund = s["elem.abund"].clone()
            while True:
                iz = fint(F())
                if fr.iffail == 1:
                    s["elem.abund"] = abund
                    return "98"
                abund[iz - 1] = F()
                if m == 74075686 and iz > 2:
                    abund[iz - 1] = abund[iz - 1] / s.scalar("junk.xscale")
        _not_understood(io, card)
    if miac == 30911189:                                    # PRINT
        a = s["iter.ifprnt"].clone()
        for i in range(s.iscalar("iter.numits")):
            a[i] = fint(F())
        s["iter.ifprnt"] = a
        return "97"
    if miac == 31069574:                                    # PUNCH
        a = s["iter.ifpnch"].clone()
        for i in range(s.iscalar("iter.numits")):
            a[i] = fint(F())
        s["iter.ifpnch"] = a
        return "97"
    if miac == 918640:                                      # READ
        m = fr.iwordf(card)
        fr.numcol = fr.letcol
        if m == 450075960:                                  # READ FREQUENCIES
            num = fint(F()); nulo = fint(F()); nuhi = fint(F())
            _sv(s, "freset.numnu", num); _sv(s, "freset.nulo", nulo); _sv(s, "freset.nuhi", nuhi)
            fr.letcol = fr.numcol
            fr.iwordf(card)
            s["junk.freqid"] = np.array([w.encode() for w in fr.word], dtype="S1")
            fr.numcol = fr.letcol
            freset = s["freset.freset"].clone(); rcoset = s["freset.rcoset"].clone()
            for _ in range(num):
                nu = fint(fr.freefr(card, io.stdin))
                v = F()
                if v < 1.0e7:
                    v = 2.997925e17 / v
                if v > 1.0e20:
                    v = v * (2.997925e10 / 1.0e25)
                freset[nu - 1] = v
                rcoset[nu - 1] = F()
            s.update({"freset.freset": freset, "freset.rcoset": rcoset})
            return "98"
        if m == 287559136:                                  # READ DEPARTURE COEFFICIENTS
            nrhox = fint(F()); _sv(s, "rhox.nrhox", nrhox)
            bhyd = s["depart.bhyd"].clone(); bmin = s["depart.bmin"].clone()
            for j in range(nrhox):
                fr.numcol = 1
                read_card_cb()
                F()
                for i in range(6):
                    bhyd[j, i] = F()
                bmin[j] = F()
            s.update({"depart.bhyd": bhyd, "depart.bmin": bmin})
            s["junk.wlte"] = np.array([b"NLTE"], dtype="S4")
            _sv(s, "depart.nlteon", 1)
            return "98"
        if m == 1355094447:                                 # READ STARTING T-TAU
            nrhox = fint(F()); _sv(s, "rhox.nrhox", nrhox)
            tauros = s["abross.tauros"].clone(); t = s["temp.t"].clone()
            for j in range(nrhox):
                fr.numcol = 1
                read_card_cb()
                tauros[j] = F(); t[j] = F()
            if not (float(tauros[0]) > 0.0):
                for j in range(nrhox):
                    tauros[j] = _exp10(float(tauros[j]))
            s.update({"abross.tauros": tauros, "temp.t": t})
            _label702(s, io)
            return "97"
        if m in (209579, 7754456):                          # READ DECK / DECK6
            nrhox = fint(F()); _sv(s, "rhox.nrhox", nrhox)
            ks = ("rhox.rhox", "temp.t", "state.p", "state.xne", "abross.abross", "rad.prad", "turbpr.vturb")
            arrs = {k: s[k].clone() for k in ks}
            for j in range(nrhox):
                fr.numcol = 1
                read_card_cb()
                arrs["rhox.rhox"][j] = F(); arrs["temp.t"][j] = F()
                fr.more = 1
                for k in ks[2:]:
                    arrs[k][j] = F()
                fr.more = 0
            if not (float(arrs["rhox.rhox"][0]) >= 0.0):
                for j in range(nrhox):
                    arrs["rhox.rhox"][j] = _exp10(float(arrs["rhox.rhox"][j]))
            s.update(arrs)
            _sv(s, "pzero.pradk0", 0.0)
            _sv(s, "pzero.pturb0", float(s["turbpr.pturb"][0]))
            _sv(s, "pzero.pcon", 0.0)
            n = nrhox
            rx, ab = s["rhox.rhox"], s["abross.abross"]
            tauros = s["abross.tauros"].clone()
            tauros[:n] = integ(rx[:n], ab[:n], ab[0] * rx[0])
            s["abross.tauros"] = tauros
            if m != 7754456:
                return "98"
            read_card_cb()
            fr.numcol = 1
            pradk0 = F()
            _sv(s, "pzero.pradk0", pradk0)
            accrad = s["rad.prad"].clone()
            prad = s["rad.prad"].clone()
            prad[:n] = integ(rx[:n], accrad[:n], accrad[0] * rx[0])
            pradk = s["pzero.pradk"].clone()
            pradk[:n] = prad[:n] + pradk0
            s.update({"rad.accrad": accrad, "rad.prad": prad, "pzero.pradk": pradk})
            return "98"
        if m == 20688921:                                   # READ KAPPA (unit 1)
            title, tab = read_kappa_ros(io.files[1])
            io.stdout.append(title)
            s["tabtp.tabkap"] = tab
            _sv(s, "tabtp.iftabk", 1)
            return "98"
        if m == 930198669:                                  # READ MOLECULES (unit 2)
            readmol(s, io)
            return "98"
        if m == 31069574:                                   # READ PUNCH (unit 3)
            read_punch(s, io)
            return "98"
        _not_understood(io, card)
    if miac == 17173:                                       # LTE
        _sv(s, "depart.nlteon", 0)
        s["junk.wlte"] = np.array([b"LTE "], dtype="S4")
        s["depart.bhyd"] = torch.ones_like(s["depart.bhyd"])
        s["depart.bmin"] = torch.ones_like(s["depart.bmin"])
        return "97"
    if miac == 726315:                                      # NLTE
        _sv(s, "depart.nlteon", 1)
        s["junk.wlte"] = np.array([b"NLTE"], dtype="S4")
        return "97"
    if miac == 4011517:                                     # BEGIN
        _begin(s, io, mode)
        return "return"
    if miac == 1323236444:                                  # SCATTERING
        m = fr.iwordf(card)
        if m == 569:
            _sv(s, "if.ifscat", 1); return "97"
        if m == 20763:
            _sv(s, "if.ifscat", 0); return "97"
        if m == 22965179:
            _sv(s, "if.ifscat", 1)
            fr.numcol = fr.letcol
            _sv(s, "if.tauscat", F())
            return "97"
        _not_understood(io, card)
    if miac == 7367:                                        # END
        if mode != 20:
            raise FortranExit("", 0)
        _sv(s, "rhox.nrhox", 0)
        return "return"
    if miac == 37966926:                                    # TITLE
        _title_overflow(s, card)
        return "98"
    if miac == 236883734:                                   # CONVECTION
        m = fr.iwordf(card)
        if m == 569:
            _sv(s, "conv.ifconv", 1)
            fr.numcol = fr.letcol
            _sv(s, "conv.mixlth", F())
            return "97"
        if m == 20763:
            _sv(s, "conv.ifconv", 0)
            _sv(s, "conv.mixlth", 1.0)
            n = s.iscalar("rhox.nrhox")
            for k in ("dltdlp", "heatcp", "dlrdlt", "velsnd", "grdadb", "hscale", "flxcnv", "vconv"):
                a = s[f"conv.{k}"].clone(); a[:n] = 0.0; s[f"conv.{k}"] = a
            return "97"
        if m == 790116:                                     # OVER
            _sv(s, "conv.ifconv", 1)
            fr.numcol = fr.letcol
            _sv(s, "conv.mixlth", F())
            _sv(s, "conv.overwt", F())
            _sv(s, "conv.nconv", fint(F()))
            return "97"
        _not_understood(io, card)
    if miac == 1427151802:                                  # TURBULENCE
        m = fr.iwordf(card)
        if m == 569:
            _sv(s, "turbpr.ifturb", 1)
            fr.numcol = fr.letcol
            for k in ("trbfdg", "trbpow", "trbsnd", "trbcon"):
                _sv(s, f"turbpr.{k}", F())
            return "97"
        if m == 20763:
            _sv(s, "turbpr.ifturb", 0)
            for k in ("trbfdg", "trbpow", "trbsnd", "trbcon"):
                _sv(s, f"turbpr.{k}", 0.0)
            return "97"
        _not_understood(io, card)
    if miac == 223095242:                                   # CHANGE RHOX
        nnew = fint(F())
        rhoxa = torch.tensor([fr.freefr(card, io.stdin) for _ in range(nnew)], dtype=torch.float64)
        n = s.iscalar("rhox.nrhox")
        rx = s["rhox.rhox"][:n]
        ks = ("temp.t", "state.p", "state.xne", "abross.abross", "turbpr.vturb", "rad.prad", "depart.bmin")
        fold = torch.stack([s[k][:n] for k in ks] + [s["depart.bhyd"][:n, i] for i in range(6)])
        new, _ = map1b(rx, fold, rhoxa)
        for i, k in enumerate(ks):
            a = s[k].clone(); a[:nnew] = new[i]; s[k] = a
        pradk = s["pzero.pradk"].clone(); pradk[:nnew] = s["rad.prad"][:nnew] + s.scalar("pzero.pradk0")
        s["pzero.pradk"] = pradk
        bhyd = s["depart.bhyd"].clone()
        for i in range(6):
            bhyd[:nnew, i] = new[len(ks) + i]
        s["depart.bhyd"] = bhyd
        _sv(s, "rhox.nrhox", nnew)
        a = s["rhox.rhox"].clone(); a[:nnew] = rhoxa; s["rhox.rhox"] = a
        return "97"
    if miac == 450075960:                                   # FREQUENCIES n lo hi BIG/LITTLE
        num = fint(F()); nulo = fint(F()); nuhi = fint(F())
        fr.letcol = fr.numcol
        m = fr.iwordf(card)
        _sv(s, "freset.numnu", num); _sv(s, "freset.nulo", nulo); _sv(s, "freset.nuhi", nuhi)
        s["junk.freqid"] = np.array([w.encode() for w in fr.word], dtype="S1")
        if m == 3078:
            off = 5
        elif m == 850035822:
            off = 342
        else:
            _not_understood(io, card)
        fre = s["freset.freset"].clone(); rco = s["freset.rcoset"].clone()
        for nu in range(num):                               # sequential: FRESET(NU+off) may alias
            fre[nu] = 2.997925e17 / float(fre[nu + off])
        for nu in range(1, num - 1):
            rco[nu] = (fre[nu - 1] - fre[nu + 1]) / 2.0
        rco[0] = (2.997925e17 / f32(8.97666) - float(fre[1])) / 2.0
        rco[num - 1] = fre[num - 2] / 2.0
        s.update({"freset.freset": fre, "freset.rcoset": rco})
        return "98"
    if miac == 1357812572:                                  # SURFACE
        m = fr.iwordf(card)
        if m == 651354309:                                  # INTENSITY
            nmu = fint(F()); _sv(s, "mus.nmu", nmu)
            ang = s["mus.angle"].clone()
            for mu in range(nmu):
                ang[mu] = fr.freefr(card, io.stdin)
            s["mus.angle"] = ang
            _sv(s, "if.ifsurf", 2); return "97"
        if m == 321147:
            _sv(s, "if.ifsurf", 1); return "97"
        if m == 20763:
            _sv(s, "if.ifsurf", 0); return "97"
        _not_understood(io, card)
    if miac == 1143518210:                                  # PRESSURE
        m = fr.iwordf(card)
        if m == 569:
            _sv(s, "if.ifpres", 1); return "97"
        if m == 20763:
            _sv(s, "if.ifpres", 0); return "97"
        _not_understood(io, card)
    if miac == 237080870:                                   # CORRECTION
        m = fr.iwordf(card)
        if m == 569:
            _sv(s, "if.ifcorr", 1); return "97"
        if m == 20763:
            _sv(s, "if.ifcorr", 0)
            s["flux.flxerr"] = torch.zeros_like(s["flux.flxerr"])
            s["flux.flxdrv"] = torch.zeros_like(s["flux.flxdrv"])
            return "97"
        _not_understood(io, card)
    if miac == 1597906832:                                  # WAVELENGTH
        wb = F(); dw = F(); we = F()
        _sv(s, "wavey.ifwave", 1)
        _sv(s, "freset.nulo", 1)
        nuhi = int(abs(we - wb) / abs(dw) + f32(.5)) + 1
        _sv(s, "freset.nuhi", nuhi); _sv(s, "freset.numnu", nuhi)
        if wb >= 1.0e7:
            if wb > 1.0e20:
                wb, dw, we = 1.0e7 / (wb / 1.0e25), 1.0e7 / (dw / 1.0e25), 1.0e7 / (we / 1.0e25)
            else:
                wb, dw, we = 2.997925e17 / wb, 2.997925e17 / dw, 2.997925e17 / we
        _sv(s, "wavey.wbegin", wb); _sv(s, "wavey.deltaw", dw)
        return "97"
    if miac == 35762836:                                    # SCALE MODEL
        _scale(s, io, fr, card)
        return "97"
    if miac == 153784:                                      # CALL (DUMMYR)
        return "97"
    if miac == 42274019:                                    # VTURB
        vnew = F()
        n = s.iscalar("rhox.nrhox")
        a = s["turbpr.vturb"].clone(); a[:n] = vnew; s["turbpr.vturb"] = a
        return "97"
    if miac == 1342680354:                                  # SMOOTH
        j1 = fint(F()); j2 = fint(F())
        _sv(s, "tsmooth.j1smooth", j1); _sv(s, "tsmooth.j2smooth", j2)
        w = [F(), F(), F()]
        _sv(s, "tsmooth.wtjm1", w[0]); _sv(s, "tsmooth.wtj", w[1]); _sv(s, "tsmooth.wtjp1", w[2])
        t = s["temp.t"].clone(); ts = s["tsmooth.tsmooth"].clone()
        for j in range(j1 - 1, j2):
            ts[j] = w[0] * t[j - 1] + w[1] * t[j] + w[2] * t[j + 1]
        t[j1 - 1:j2] = ts[j1 - 1:j2]
        s.update({"temp.t": t, "tsmooth.tsmooth": ts})
        return "97"
    _not_understood(io, card)


def _label702(s: State, io: IO) -> None:
    """Label 702 of READIN: starting pressures from TTAUP for a T(tau) model."""
    n = s.iscalar("rhox.nrhox")
    teff = s.scalar("teff.teff")
    t = s["temp.t"]
    xne = s["state.xne"].clone(); xne[:n] = 0.0
    t4 = t[:n] * t[:n]; t4 = t4 * t4
    te2 = teff * teff
    pradk = s["pzero.pradk"].clone()
    pradk[:n] = 2.521e-15 * torch.maximum(t4, torch.full_like(t4, te2 * te2 / 2.0))
    vt = s["turbpr.vturb"].clone(); vt[:n] = 0.0
    pt = s["turbpr.pturb"].clone(); pt[:n] = 0.0
    pradk0 = float(pradk[0])
    _sv(s, "pzero.pradk0", pradk0); _sv(s, "pzero.pcon", 0.0); _sv(s, "pzero.pturb0", 0.0)
    _sv(s, "pzero.pzero", pradk0)
    prad = s["rad.prad"].clone(); prad[:n] = pradk[:n] - pradk0
    s.update({"state.xne": xne, "pzero.pradk": pradk, "turbpr.vturb": vt, "turbpr.pturb": pt, "rad.prad": prad})
    _ttaup_into(s, n, iross=0)


def _ttaup_into(s: State, n: int, iross: int) -> None:
    R = Rosstab(s)
    s.update(R.updates)
    grav = s.scalar("teff.grav")
    ab, ptot, p = ttaup(s["temp.t"], s["abross.tauros"], s["abross.abross"], s["rad.prad"], s["turbpr.pturb"],
                        s["turbpr.vturb"], grav, n, iross, R)
    abross = s["abross.abross"].clone(); abross[:n] = ab
    pp = s["state.p"].clone(); pp[:n] = p
    ptotal = s["ptotal.ptotal"].clone()
    rhox = s["rhox.rhox"].clone()
    rhox[:n] = ptot / grav
    ptotal[:n] = ptot + s.scalar("pzero.pzero")
    s.update({"abross.abross": abross, "state.p": pp, "ptotal.ptotal": ptotal, "rhox.rhox": rhox})


def _scale(s: State, io: IO, fr: Free, card) -> None:
    F = lambda: fr.freeff(card)
    krhox = fint(F()); tau1lg = F(); steplg = F()
    _sv(s, "steplg.krhox", krhox); _sv(s, "steplg.tau1lg", tau1lg); _sv(s, "steplg.steplg", steplg)
    fr.more = 1
    teff1 = F()
    gnew = F()
    if gnew < 10.0:
        gnew = _exp10(gnew)
    fr.more = 0
    taustd = torch.tensor([_exp10(tau1lg + float(j) * steplg) for j in range(krhox)], dtype=torch.float64)
    n = s.iscalar("rhox.nrhox")
    rx, ab = s["rhox.rhox"], s["abross.abross"]
    tauros_full = s["abross.tauros"].clone()
    tauros_full[:n] = integ(rx[:n], ab[:n], ab[0] * rx[0])
    tr = tauros_full[:n]
    ks = ("rhox.rhox", "temp.t", "state.p", "state.xne", "abross.abross", "rad.prad", "turbpr.vturb", "depart.bmin")
    fold = torch.stack([s[k][:n] for k in ks] + [s["depart.bhyd"][:n, i] for i in range(6)])
    new, _ = map1b(tr, fold, taustd)
    s["abross.tauros"] = tauros_full
    pradk0 = s.scalar("pzero.pradk0")
    for i, k in enumerate(ks):
        a = s[k].clone(); a[:krhox] = new[i]; s[k] = a
    pradk = s["pzero.pradk"].clone(); pradk[:krhox] = s["rad.prad"][:krhox] + pradk0
    s["pzero.pradk"] = pradk
    bhyd = s["depart.bhyd"].clone()
    for i in range(6):
        bhyd[:krhox, i] = new[len(ks) + i]
    s["depart.bhyd"] = bhyd
    _sv(s, "rhox.nrhox", krhox)
    n = krhox
    teff, grav = s.scalar("teff.teff"), s.scalar("teff.grav")
    if teff1 == 0.0:
        return
    if teff1 == teff and gnew == grav:
        return
    if (teff1 < teff + 1.0 and teff1 > teff - 1.0 and gnew < grav * f32(1.001) and gnew > grav * f32(.999)):
        return
    tauros = s["abross.tauros"].clone(); tauros[:n] = taustd
    ratio = teff1 / teff
    r2 = ratio * ratio
    r4 = r2 * r2
    t = s["temp.t"].clone(); t[:n] = t[:n] * teff1 / teff
    pt = s["turbpr.pturb"].clone(); pt[:n] = 0.0
    pradk = s["pzero.pradk"].clone(); pradk[:n] = pradk[:n] * r4
    prad = s["rad.prad"].clone(); prad[:n] = prad[:n] * r4
    pradk0 = pradk0 * r4
    _sv(s, "pzero.pradk0", pradk0)
    _sv(s, "pzero.pzero", s.scalar("pzero.pcon") + pradk0 + s.scalar("pzero.pturb0"))
    _sv(s, "teff.teff", teff1)
    _sv(s, "flux.flux", flux_of(teff1))
    _sv(s, "teff.grav", gnew)
    _sv(s, "teff.glog", fmath.slog10(gnew))
    s.update({"abross.tauros": tauros, "temp.t": t, "turbpr.pturb": pt, "pzero.pradk": pradk, "rad.prad": prad})
    _ttaup_into(s, n, iross=0)


def _begin(s: State, io: IO, mode: int) -> None:
    out = io.stdout
    if mode == 1:
        msgs = []
        if s.iscalar("iter.numits") == 0:
            msgs.append(" HOW MANY ITERATIONS")
        if s.iscalar("rhox.nrhox") == 0:
            msgs.append(" HOW MANY RHOX")
        if s.iscalar("freset.numnu") == 0:
            msgs.append(" HOW MANY FREQUENCIES")
        if s.scalar("teff.teff") == 0.0:
            msgs.append(" WHAT TEFF")
        if s.scalar("teff.grav") == 0.0:
            msgs.append(" WHAT GRAVITY")
        out.extend(msgs)
        if msgs:
            raise FortranExit("", 0)
    abund = s["elem.abund"].clone()
    if float(abund[0]) < 0.0:
        abund[0] = _exp10(float(abund[0]))
    if float(abund[1]) < 0.0:
        abund[1] = _exp10(float(abund[1]))
    for iz in range(2, 99):
        if float(abund[iz]) > 0.0:
            abund[iz] = fmath.slog10(float(abund[iz]))
    s["elem.abund"] = abund
    teff, glog = s.scalar("teff.teff"), s.scalar("teff.glog")
    wlte = np.asarray(s["junk.wlte"]).reshape(-1)[0].decode("latin-1").replace("\0", " ")
    title = "".join(_chars(s, "junk.title")).replace("\0", " ")
    xscale = s.scalar("junk.xscale")
    elem = [e.decode("latin-1") for e in np.asarray(s["elem.elem"]).reshape(-1)]
    out += ["1", "", "", "", "",
            " TEFF" + ff.F(teff, 7, 0) + "   LOG G" + ff.F(glog, 8, 4) + ff.X(3) + ff.A(wlte, 4),
            "0TITLE " + title,
            "0XSCALE" + ff.F(xscale, 10, 6) + "".join(ff.X(3) + ff.A(elem[i], 2) + ff.F(float(abund[i]), 8, 5)
                                                     for i in range(2))]
    for r0 in range(2, 99, 10):
        out.append("".join(ff.X(1) + ff.A(elem[i], 2) + ff.F(float(abund[i]), 6, 2) for i in range(r0, min(r0 + 10, 99))))
    xabund = s["xabund.xabund"].clone()
    for iz in range(2, 99):
        xabund[iz] = _exp10(float(abund[iz])) * xscale
    xabund[0] = abund[0]
    xabund[1] = abund[1]
    atmass = s["elem.atmass"]
    wtmole = 0.0
    for iz in range(99):
        wtmole = wtmole + float(xabund[iz]) * float(atmass[iz])
    s["xabund.xabund"] = xabund
    _sv(s, "xabund.wtmole", wtmole)
    n = s.iscalar("rhox.nrhox")
    t = s["temp.t"]
    upd = derived_temperature(s)
    s.update(upd)
    tk = s["temp.tk"]
    xnatom = s["state.xnatom"].clone(); rho = s["state.rho"].clone()
    xnatom[:n] = s["state.p"][:n] / tk[:n] - s["state.xne"][:n]
    rho[:n] = xnatom[:n] * wtmole * 1.660e-24
    s.update({"state.xnatom": xnatom, "state.rho": rho})
    if s.iscalar("turbpr.ifturb") > 0:
        pt = s["turbpr.pturb"].clone()
        v = s["turbpr.vturb"][:n]
        pt[:n] = f32(.5) * rho[:n] * (v * v)
        s["turbpr.pturb"] = pt
    ifop = [int(v) for v in s["ifop.ifop"]]
    names1 = ["0H1", " H2PLUS", " HMINUS", " HRAY", " HE1", " HE2", " HEMINUS", " HERAY", " COOL", " LUKE"]
    names2 = [" HOT", " ELECTRON", " H2RAY", " HLINES", " LINES", " LINESCAT", " XLINES", " XLSCAT", " XCONT",
              " XSCAT"]
    out.append("".join(nm + ff.I(ifop[i], 2) for i, nm in enumerate(names1)))
    out.append("".join(nm + ff.I(ifop[10 + i], 2) for i, nm in enumerate(names2)))
    g = s.iscalar
    out.append("0IFCORR" + ff.I(g("if.ifcorr"), 2) + "  IFPRES" + ff.I(g("if.ifpres"), 2) + "  IFSURF"
               + ff.I(g("if.ifsurf"), 2) + "  IFSCAT" + ff.I(g("if.ifscat"), 2) + "  TAUSCAT"
               + ff.E(s.scalar("if.tauscat"), 10, 2, 1) + "  IFCONV" + ff.I(g("conv.ifconv"), 2) + "  MIXLTH"
               + ff.F(s.scalar("conv.mixlth"), 6, 2) + "  IFMOL" + ff.I(g("if.ifmol"), 2))
    out.append(" IFTURB" + ff.I(g("turbpr.ifturb"), 2) + "  TRBFDG" + ff.F(s.scalar("turbpr.trbfdg"), 6, 2)
               + "  TRBPOW" + ff.F(s.scalar("turbpr.trbpow"), 6, 2) + "  TRBSND"
               + ff.F(s.scalar("turbpr.trbsnd"), 6, 2) + "  TRBCON" + ff.F(s.scalar("turbpr.trbcon"), 6, 2))
    if mode == 1:
        out.append(" NUMITS" + ff.I(g("iter.numits"), 3) + "  IFPRNT"
                   + "".join(ff.I(int(v), 2) for v in s["iter.ifprnt"]) + "  IFPNCH"
                   + "".join(ff.I(int(v), 2) for v in s["iter.ifpnch"]))
        if g("wavey.ifwave") != 0:
            out.append("0WBEGIN" + ff.F(s.scalar("wavey.wbegin"), 11, 4) + "   DELTAW"
                       + ff.F(s.scalar("wavey.deltaw"), 7, 4) + "   NUMNU" + ff.I(g("freset.numnu"), 5))
        else:
            freqid = "".join(_chars(s, "junk.freqid")).replace("\0", " ")
            numnu, nulo, nuhi = g("freset.numnu"), g("freset.nulo"), g("freset.nuhi")
            out.append("0FREQID " + freqid + "   NUMNU" + ff.I(numnu, 4) + "   NULO" + ff.I(nulo, 4)
                       + "   NUHI" + ff.I(nuhi, 4))
            fre, rco = s["freset.freset"], s["freset.rcoset"]
            nn = (numnu + 3) // 4
            nnn = numnu - nn * 3
            if nnn < 1:
                nn = 1
                nnn = 1
            item = lambda nu: ff.I(nu, 6) + ff.E(float(fre[nu - 1]), 13, 6, 1) + ff.E(float(rco[nu - 1]), 13, 6, 1)
            items = [nu for i in range(1, nnn + 1) for nu in range(i, numnu + 1, nn)]
            for k in range(0, len(items), 4):
                out.append("".join(item(nu) for nu in items[k:k + 4]))
            if nn != nnn:
                nnn = nnn + 1
                items = [nu for i in range(nnn, nn + 1) for nu in range(i, numnu + 1, nn)]
                for k in range(0, len(items), 3):
                    out.append("".join(item(nu) for nu in items[k:k + 3]))
                ssum = 0.0
                for nu in range(nulo, nuhi + 1):
                    ssum = ssum + float(rco[nu - 1])
                out.append(ff.E(ssum, 13, 6, 1))
    out += ["1", "", "", "", "",
            "         RHOX         T        P        XNE      ABROSS     PRAD      VTURB"
            + " " * 23 + "BHYD" + " " * 25 + "BMIN"]
    rx, p, xne, ab, prad, vt = (s[k] for k in ("rhox.rhox", "state.p", "state.xne", "abross.abross", "rad.prad",
                                               "turbpr.vturb"))
    bhyd, bmin = s["depart.bhyd"], s["depart.bmin"]
    for j in range(n):
        out.append(ff.I(j + 1, 4) + ff.E(float(rx[j]), 13, 6, 1) + ff.F(float(t[j]), 9, 1)
                   + "".join(ff.E(float(a[j]), 10, 3, 1) for a in (p, xne, ab, prad, vt))
                   + "".join(ff.F(float(bhyd[j, i]), 8, 4) for i in range(6)) + ff.F(float(bmin[j]), 8, 4))


def derived_temperature(s: State) -> dict:
    """TK, HKT, HCKT, TKEV, TLOG from T for J=1..NRHOX (READIN 1516 / TCORR 81 / 2702 pattern)."""
    n = s.iscalar("rhox.nrhox")
    t = s["temp.t"][:n]
    out = {}
    tk = s["temp.tk"].clone(); hkt = s["temp.hkt"].clone(); hckt = s["temp.hckt"].clone()
    tkev = s["temp.tkev"].clone(); tlog = s["temp.tlog"].clone()
    tk[:n] = 1.38054e-16 * t
    hkt[:n] = fmath.rdiv(6.6256e-27, tk[:n])
    hckt[:n] = hkt[:n] * 2.99792458e10
    tkev[:n] = 8.6171e-5 * t
    tlog[:n] = fmath.log(t)
    return {"temp.tk": tk, "temp.hkt": hkt, "temp.hckt": hckt, "temp.tkev": tkev, "temp.tlog": tlog}
