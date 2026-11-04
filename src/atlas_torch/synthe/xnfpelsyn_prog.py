"""PROGRAM XNFPELSYN (xnfpelsyn.for + atlas7v.for, SYNTHE): populations, Doppler widths and continuum
opacity tables of a model atmosphere for SYNTHE / SPECTRV (fort.10 = xnfpelsyn.dat).

    from atlas_torch.synthe.xnfpelsyn_prog import xnfpelsyn
    res = xnfpelsyn(deck_text, continua_text=None, molecules_path=None, out10="fort.10")

`deck_text` is the model deck read by CALL READIN(20) on unit 5 (BasicATLAS' output_synthe.out),
`continua_text` the edge list read by XFREEF on unit 17 (continua.dat), `molecules_path` unit 2
(molecules.dat, READ MOLECULES). Defaults: BasicATLAS' data/synthe_files.

The main program is reproduced statement by statement; the physics (READIN, POPS/NMOLEC/PFSAHA, KAPP) comes
from a backend with the small interface of `TorchPhysics` (default; the torch port of atlas7v.for).
`atlas_torch.testing.xnfpelsyn_oracle_physics` provides the same interface over the Fortran library (tests
only). Layers are a tensor dimension everywhere (no Python loop over J); the frequency loop DO 125 is one
batched KAPP over all FREQSET points.

fort.10 records (gfortran unformatted sequential, 4-byte record markers), written byte-compatibly:
  1  NT, TEFF, GLOG, TITLE(74)
  2  IN, (FRQEDG(I),WLEDGE(I),CMEDGE(I),I=1,IN), IDMOL(100), MOMASS(100)
  3  NUMNU, FREQSET(1..NUMNU)
  4  T,TKEV,TK,HKT,TLOG,HCKT,P,XNE,XNATOM,RHO,RHOX,VTURB,XNFH (kw each), XNFHE(kw,2), XNFH2(kw)
  per layer J: CONTINALL(1..1131,J); CONTABS(1..1131,J); CONTSCAT(1..1131,J); XNFPEL(6,139),DOPPLE(6,139)

Fortran quirks reproduced:
  * REAL*4 literals: 1.0000001 in FREQSET, 1.E15 (FREQ15), 11.091E0 and 49.0414E0 in the CO equilibrium,
    IDMOL DATA (REAL*4 constants widened to REAL*8: 106.01 -> 106.01000213623047, so POPS(IDMOL,1) finds no
    exact CODEMOL match for the ionised molecules and stores 0);
  * CONTINALL/CONTABS/CONTSCAT(NU,J) for NU > NUMNU stay 0 (static arrays); XNFP/DOPPLE for elements
    100..139 not written by POPS/the DOPPLE loops stay 0;
  * XNFH2/XNFPH2/XNFPCO are 0 where T > 9000 K;
  * POPS(106.,1)/POPS(108.,1) (CH/OH, Castelli 2004 correction) into /CHOH/ when IFMOL=1;
  * XNFP(J,6,40)=XNFPH2 and XNFP(J,6,46)=XNFPCO are overwritten by POPS(101.,1)/POPS(608.,1) when IFMOL=1;
  * the main program's IFOP(14..17)=0, IFPRES=1, ITEMP=1 before the populations.
"""
from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..fnum import f32
from ..tables import table
from .freeform import CardStream, Free, xfreef
from .opacity._core import Ctx, exp, log, log10, powi, rdiv, sqrt

KW = 99
MW = 139
MM = MW - 39
NFREQ = 1131
NEDGE = 377
SRC = "xnfpelsyn.for"
DT = torch.float64

# main-program POPS calls before the frequency loop: (COMMON key, CODE, MODE, columns)
POPS_COMMON = [("bhyd.xnfh", 1.00, 12, 1), ("bhe.xnfhe", 2.01, 12, 2), ("bhyd.xnfph", 1.01, 11, 2),
               ("bhe.xnfphe", 2.02, 11, 3), ("bb.xnfpb", 5.00, 11, 1), ("bc.xnfpc", 6.01, 11, 2),
               ("bo.xnfpo", 8.00, 11, 1), ("bna.xnfpna", 11.00, 11, 1), ("bmg.xnfpmg", 12.01, 11, 2),
               ("bal.xnfpal", 13.01, 11, 2), ("bsi.xnfpsi", 14.01, 11, 2), ("bk.xnfpk", 19.00, 11, 1),
               ("bca.xnfpca", 20.01, 11, 2), ("bfe.xnfpfe", 26.00, 11, 1)]
# XNFP element codes 1..28 (29..99: FLOAT(NELEM)+.02D0)
XNFP_CODES = [1.01, 2.02, 3.03, 4.03, 5.03, 6.05, 7.05, 8.05, 9.05, 10.05, 11.05, 12.05, 13.05, 14.05, 15.05,
              16.05, 17.04, 18.04, 19.04, 20.09, 21.09, 22.09, 23.09, 24.09, 25.09, 26.09, 27.09, 28.09]


def _ext() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def default_paths() -> dict:
    d = _ext() / "BasicATLAS" / "data" / "synthe_files"
    return dict(continua=d / "continua.dat", molecules=d / "molecules.dat")


def idmol_momass() -> tuple[np.ndarray, np.ndarray]:
    """DATA IDMOL(mm), MOMASS(mm) of the main program (REAL*8 arrays initialised with REAL*4 constants)."""
    return (np.asarray(table(SRC, "xnfpelsyn", "idmol"), dtype=np.float64).reshape(MM),
            np.asarray(table(SRC, "xnfpelsyn", "momass"), dtype=np.float64).reshape(MM))


# ---------------------------------------------------------------------------------------- physics backend

class TorchPhysics:
    """The torch port of atlas7v.for as the program's physics: READIN, POPS, KAPP on an explicit State."""

    name = "torch"

    def __init__(self):
        self.s = None
        self.readin_state = None        # the State right after READIN (reused by the chain for SPECTRV)

    def readin(self, deck_text: str, molecules_path) -> None:
        from .readin import readin
        self.s = readin(deck_text, molecules_path=molecules_path, mode=20)
        for key in ("choh.xnfpch", "choh.xnfpoh"):      # /CHOH/ (not touched by READIN)
            if key not in self.s:
                self.s[key] = torch.zeros(KW, dtype=DT)
        self.readin_state = type(self.s)({k: (v.clone() if isinstance(v, torch.Tensor) else np.array(v, copy=True))
                                          for k, v in self.s.items()})

    def get(self, key: str):
        return self.s[key]

    def set(self, key: str, value) -> None:
        old = self.s.get(key)
        if isinstance(value, torch.Tensor):
            v = value
        elif isinstance(old, torch.Tensor):
            v = torch.as_tensor(np.asarray(value), dtype=old.dtype).reshape(old.shape)
        else:
            v = torch.as_tensor(np.asarray(value))
        self.s = self.s.updated({key: v})

    def pops(self, code: float, mode: int, number: torch.Tensor) -> torch.Tensor:
        """CALL POPS(CODE,MODE,NUMBER): NUMBER (kw, k) in -> (kw, k) out; State side effects kept."""
        from .pops import pops
        k = number.shape[1]
        out, upd = pops(self.s, code, mode, number)
        if upd:
            self.s = self.s.updated(upd)
        return out[:, :k]

    def kapp(self, freq, freqlg, waveno, ehvkt, stim, bnu):
        """DO 125 loop: CALL KAPP(1,NSTEPS,STEPWT) at every frequency (in order) -> ACONT, SIGMAC [NF, n]."""
        from .kapp import kapp_k
        from .pops import pops
        c = Ctx(self.s, freq, freqlg, waveno, ehvkt, stim, bnu)
        sp, sv, _, _ = kapp_k(c, 1, lambda s, code, mode: pops(s, code, mode)[0])
        return sp["optot.acont"], sp["optot.sigmac"]


# ------------------------------------------------------------------------------------------- edges

def read_edges(continua_text: str, fr: Free):
    """Loop 11: edges read with XFREEF (unit 17) until a 0; loop 14/17 bubble sort by |wavelength|.

    Returns FRQEDG, WLEDGE, CMEDGE (IN,) after sorting."""
    unit = CardStream(continua_text)
    card = [" "] * 83          # CARD(81) of the main program: DATA blank, never read into (L=LAST-1=80)
    frq, wl, cm = [], [], []
    while True:
        edge = xfreef(card, fr, unit)
        if edge == 0.0:
            break
        if abs(edge) < 1e6:                            # 12: wavelength (nm)
            w = edge
            c = 1e7 / edge
            f = 2.99792458e17 / w
        elif abs(edge) < 1e25:                         # 13: frequency (Hz)
            f = edge
            w = 2.99792458e17 / edge
            c = 1e7 / w
        else:                                          # wavenumbers * 1.E25
            c = edge / 1e25
            w = 1e7 / c
            f = 2.99792458e17 / w
        frq.append(f)
        wl.append(w)
        cm.append(c)
    if len(frq) > NEDGE:
        raise ValueError(f"{len(frq)} edges > {NEDGE} (WLEDGE dimension)")
    a = np.abs(np.array(wl))
    # DO 17 LAST=2,IN / DO 16 I=2,IN-LAST+2: complete bubble sort, swaps only on A(I) < A(I-1) = stable sort
    order = np.argsort(a, kind="stable")
    return np.array(frq)[order], np.array(wl)[order], np.array(cm)[order]


def freqset_of(frq: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Loop 18: three frequencies per edge interval."""
    n = len(frq)
    f1 = f32(1.0000001)
    fs = np.empty(3 * (n - 1))
    fs[0::3] = np.abs(frq[:-1]) / f1
    fs[1::3] = 2.99792458e17 / (np.abs(wl[:-1]) + np.abs(wl[1:])) * 2.0
    fs[2::3] = np.abs(frq[1:]) * f1
    if fs.size > NFREQ:
        raise ValueError(f"NUMNU={fs.size} > {NFREQ}")
    return fs


# ------------------------------------------------------------------------------------------ the program

@dataclass
class Result:
    nt: int
    teff: float
    glog: float
    title: bytes
    frqedg: np.ndarray
    wledge: np.ndarray
    cmedge: np.ndarray
    idmol: np.ndarray
    momass: np.ndarray
    freqset: np.ndarray
    state: dict                 # name -> (kw,) / (kw,2) arrays of record 4
    continall: np.ndarray       # (n, 1131)
    contabs: np.ndarray
    contscat: np.ndarray
    xnfpel: np.ndarray          # (n, 6, 139)
    dopple: np.ndarray          # (n, 6, 139)
    timings: dict
    readin_state: object = None  # State after READIN(20) (TorchPhysics only): SPECTRV's READIN gives the same


STATE_KEYS = [("t", "temp.t"), ("tkev", "temp.tkev"), ("tk", "temp.tk"), ("hkt", "temp.hkt"), ("tlog", "temp.tlog"),
              ("hckt", "temp.hckt"), ("p", "state.p"), ("xne", "state.xne"), ("xnatom", "state.xnatom"),
              ("rho", "state.rho"), ("rhox", "rhox.rhox"), ("vturb", "turbpr.vturb"), ("xnfh", "bhyd.xnfh"),
              ("xnfhe", "bhe.xnfhe")]


def _np(v) -> np.ndarray:
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _i(phys, key) -> int:
    return int(_np(phys.get(key)).reshape(-1)[0])


def _f(phys, key) -> float:
    return float(_np(phys.get(key)).reshape(-1)[0])


def _lay(phys, key, n, device):
    return torch.as_tensor(_np(phys.get(key)), dtype=DT, device=device).reshape(KW, -1)[:n]


def freq_setup(freqset: np.ndarray, hkt: torch.Tensor):
    """Loops 125/120 before CALL KAPP: FREQ, FREQLG [NF,1]; EHVKT, STIM, BNU [NF,n]; WAVENO [NF,1]."""
    freq = torch.as_tensor(freqset, dtype=DT, device=hkt.device).reshape(-1, 1)
    freq15 = freq / f32(1.0e15)
    freqlg = log(freq)
    ehvkt = exp(-freq * hkt.reshape(1, -1))
    stim = 1.0 - ehvkt
    bnu = 1.47439e-02 * powi(freq15, 3) * ehvkt / stim
    wave = rdiv(2.99792458e17, freq)
    waveno = rdiv(f32(1.0e7), wave)
    return freq, freqlg, waveno, ehvkt, stim, bnu


def xnfpelsyn(deck_text: str, continua_text: str | None = None, molecules_path=None, out10=None,
              physics=None, device=None, num_threads: int | None = 1) -> Result:
    """Run PROGRAM XNFPELSYN. Returns the fort.10 contents (and writes them to `out10` if given).

    num_threads: torch intra-op threads while the program runs (restored afterwards; None = leave as is). The
    tensors are small ([1131 frequencies x 99 layers]) and the work is many short element-wise ops, so on CPU one
    thread is fastest: measured 1.2 s total (KAPP 0.4 s) with 1 thread vs 3.6-15 s (KAPP 2.6-14 s) with torch's
    default 6 threads on a loaded 12-core host (thread wake-up/spin costs dominate). Results do not depend on it.
    """
    if num_threads is None:
        return _xnfpelsyn(deck_text, continua_text, molecules_path, out10, physics, device)
    nthr = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        return _xnfpelsyn(deck_text, continua_text, molecules_path, out10, physics, device)
    finally:
        torch.set_num_threads(nthr)


def _xnfpelsyn(deck_text, continua_text, molecules_path, out10, physics, device) -> Result:
    tm = {}
    t0 = time.perf_counter()
    paths = default_paths()
    if continua_text is None:
        continua_text = Path(paths["continua"]).read_text()
    if molecules_path is None:
        molecules_path = paths["molecules"]
    phys = TorchPhysics() if physics is None else physics
    device = torch.device("cpu") if device is None else torch.device(device)

    # 10 CALL READIN(20) and the main program's settings
    phys.readin(deck_text, molecules_path)
    tm["readin"] = time.perf_counter() - t0
    ifop = _np(phys.get("if.ifop")).copy()
    ifop[13:17] = 0
    phys.set("if.ifop", ifop)
    fr = Free.from_state({k: phys.get(k) for k in ("free.numcol", "free.letcol", "free.last", "free.more",
                                                   "free.iffail", "free.maxpow", "free.word")})
    fr.more, fr.maxpow, fr.last, fr.numcol = 1, 99, 81, 1
    phys.set("temp.itemp", 0)
    phys.set("if.ifpres", 1)
    phys.set("iter.iter", 1)
    phys.set("iter.numits", 1)
    n = _i(phys, "rhox.nrhox")
    ifmol = _i(phys, "if.ifmol")
    teff, glog = _f(phys, "teff.teff"), _f(phys, "teff.glog")
    title = b"".join(c if isinstance(c, bytes) else str(c).encode("latin-1")
                     for c in _np(phys.get("junk.title")).reshape(-1))
    title = title.ljust(74)[:74]

    # edges and FREQSET
    frq, wl, cm = read_edges(continua_text, fr)
    # loop 17 leaves LAST=IN+1 in /FREE/ (DO variable); the free-format state is not used afterwards
    for k in ("numcol", "letcol", "more", "iffail", "maxpow"):
        phys.set("free." + k, getattr(fr, k))
    phys.set("free.last", len(frq) + 1 if len(frq) >= 2 else 2)
    freqset = freqset_of(frq, wl)
    numnu = len(freqset)
    idmol, momass = idmol_momass()

    # ITEMP=ITEMP+1 and the populations for KAPP
    phys.set("temp.itemp", 1)
    t1 = time.perf_counter()

    def pops_into(key, code, mode, ncol):
        cur = torch.as_tensor(_np(phys.get(key)), dtype=DT).reshape(KW, -1)[:, :ncol].clone()
        v = phys.pops(code, mode, cur)
        full = torch.as_tensor(_np(phys.get(key)), dtype=DT).reshape(KW, -1).clone()
        full[:, :ncol] = torch.as_tensor(_np(v), dtype=DT)[:, :ncol]
        phys.set(key, full.reshape(tuple(np.shape(_np(phys.get(key))))))

    for key, code, mode, ncol in POPS_COMMON:
        pops_into(key, code, mode, ncol)
    if ifmol == 1:
        pops_into("choh.xnfpch", 106.00, 1, 1)
        pops_into("choh.xnfpoh", 108.00, 1, 1)

    # loop 444: H2 and CO equilibria (T <= 9000 K)
    t = _lay(phys, "temp.t", n, device)[:, 0]
    tkev = _lay(phys, "temp.tkev", n, device)[:, 0]
    tlog = _lay(phys, "temp.tlog", n, device)[:, 0]
    xnfh = _lay(phys, "bhyd.xnfh", n, device)[:, 0]
    xnfph1 = _lay(phys, "bhyd.xnfph", n, device)[:, 0]
    xnfpc1 = _lay(phys, "bc.xnfpc", n, device)[:, 0]
    xnfpo1 = _lay(phys, "bo.xnfpo", n, device)[:, 0]
    eq = exp(rdiv(4.478, tkev) - 4.64584e1 + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (
        1.06206e-18 - 3.08720e-23 * t) * t) * t) * t) * t) * t - 1.5 * tlog)
    eqco = exp(rdiv(f32(11.091), tkev) - f32(49.0414) + 14.0306e-4 * t - 26.6341e-8 * (t * t)
               + 35.382e-12 * powi(t, 3) - 26.5424e-16 * powi(t, 4) + 8.32385e-20 * powi(t, 5) - 1.5 * tlog)
    cold = ~(t > 9000.0)
    zero = torch.zeros_like(t)
    xnfh2 = torch.where(cold, xnfh * xnfh * eq, zero)
    xnfph2 = torch.where(cold, xnfph1 * xnfph1 * eq, zero)
    xnfpco = torch.where(cold, xnfpc1 * xnfpo1 * eqco, zero)
    tm["pops"] = time.perf_counter() - t1

    # DO 125 NU=1,NUMNU: KAPP at every frequency
    t2 = time.perf_counter()
    hkt = _lay(phys, "temp.hkt", n, device)[:, 0]
    acont, sigmac = phys.kapp(*freq_setup(freqset, hkt))
    acont = torch.as_tensor(acont, dtype=DT)
    sigmac = torch.as_tensor(sigmac, dtype=DT)
    abtot = acont + sigmac
    continall = np.zeros((n, NFREQ))
    contabs = np.zeros((n, NFREQ))
    contscat = np.zeros((n, NFREQ))
    continall[:, :numnu] = _np(log10(abtot)).T
    contabs[:, :numnu] = _np(log10(acont)).T
    contscat[:, :numnu] = _np(log10(sigmac)).T
    tm["kapp"] = time.perf_counter() - t2

    # record 4 (state after the frequency loop)
    state = {nm: _np(phys.get(key)).astype(np.float64).reshape((KW, 2) if nm == "xnfhe" else (KW,)).copy()
             for nm, key in STATE_KEYS}
    x2 = np.zeros(KW)
    x2[:n] = _np(xnfh2)
    state["xnfh2"] = x2

    # XNFP(J,ION,NELEM): loop 30 zeroes elements 1..99; elements 100..139 are never written (static zeros)
    t3 = time.perf_counter()
    xnfp = torch.zeros(KW, 10, MW, dtype=DT)
    for nelem in range(1, 100):
        code = XNFP_CODES[nelem - 1] if nelem <= 28 else float(np.float32(nelem)) + .02
        xnfp[:, :, nelem - 1] = torch.as_tensor(_np(phys.pops(code, 11, xnfp[:, :, nelem - 1].clone())), dtype=DT)
    xnfp[:n, 5, 39] = xnfph2
    xnfp[:n, 5, 45] = xnfpco
    if ifmol != 0:
        for nelem in range(40, MW + 1):
            col = xnfp[:, 5:6, nelem - 1].clone()
            xnfp[:, 5, nelem - 1] = torch.as_tensor(_np(phys.pops(float(idmol[nelem - 40]), 1, col)),
                                                     dtype=DT)[:, 0]
    for nelem in range(20, 29):          # loop 250
        for off, ion in ((30, 7), (40, 8), (50, 9), (60, 10)):
            xnfp[:, 4, off + nelem - 1] = xnfp[:, ion - 1, nelem - 1]
    xnfpel = xnfp[:n, :6, :]                                            # (n, 6, 139)
    tm["xnfp"] = time.perf_counter() - t3

    # DOPPLE(ION,NELEM) per layer
    tk = _lay(phys, "temp.tk", n, device)[:, 0]
    vt = _lay(phys, "turbpr.vturb", n, device)[:, 0]
    atmass = torch.as_tensor(_np(phys.get("elem.atmass")), dtype=DT).reshape(-1)[:99]
    d1 = sqrt(2.0 * tk[:, None] / atmass[None, :] / 1.660e-24 + (vt * vt)[:, None]) / 2.99792458e10   # (n,99)
    dop = torch.zeros(n, 6, MW, dtype=DT)
    dop[:, :, :99] = d1[:, None, :]
    for nelem in range(20, 29):          # loop 21
        for off in (30, 40, 50, 60):
            dop[:, 4, off + nelem - 1] = d1[:, nelem - 1]
    mom = torch.as_tensor(momass[:60], dtype=DT)
    dop[:, 5, 39:99] = sqrt(2.0 * tk[:, None] / mom[None, :] / 1.660e-24 + (vt * vt)[:, None]) / 2.99792458e10
    tm["total"] = time.perf_counter() - t0

    res = Result(nt=n, teff=teff, glog=glog, title=title, frqedg=frq, wledge=wl, cmedge=cm, idmol=idmol,
                 momass=momass, freqset=freqset, state=state, continall=continall, contabs=contabs,
                 contscat=contscat, xnfpel=_np(xnfpel).copy(), dopple=_np(dop).copy(), timings=tm,
                 readin_state=getattr(phys, "readin_state", None))
    if out10 is not None:
        write_fort10(out10, res)
    return res


# --------------------------------------------------------------------------------------- fort.10 I/O

def _record(fh, *parts) -> None:
    data = b"".join(p if isinstance(p, bytes) else np.ascontiguousarray(p).tobytes() for p in parts)
    m = struct.pack("<i", len(data))
    fh.write(m + data + m)


def write_fort10(path, r: Result) -> None:
    """gfortran unformatted sequential fort.10 (record layout in the module docstring)."""
    edges = np.empty(3 * len(r.frqedg))
    edges[0::3], edges[1::3], edges[2::3] = r.frqedg, r.wledge, r.cmedge
    with open(path, "wb") as fh:
        _record(fh, struct.pack("<idd", r.nt, r.teff, r.glog), r.title)
        _record(fh, struct.pack("<i", len(r.frqedg)), edges, r.idmol, r.momass)
        _record(fh, struct.pack("<i", len(r.freqset)), r.freqset)
        st = [r.state[k] for k, _ in STATE_KEYS[:-1]]
        _record(fh, *st, np.asfortranarray(r.state["xnfhe"]).tobytes(order="F"), r.state["xnfh2"])
        for j in range(r.nt):
            _record(fh, r.continall[j])
            _record(fh, r.contabs[j])
            _record(fh, r.contscat[j])
            _record(fh, r.xnfpel[j].tobytes(order="F"), r.dopple[j].tobytes(order="F"))


def _records(path) -> list[bytes]:
    b = Path(path).read_bytes()
    out, p = [], 0
    while p < len(b):
        (m,) = struct.unpack_from("<i", b, p)
        out.append(b[p + 4:p + 4 + m])
        (m2,) = struct.unpack_from("<i", b, p + 4 + m)
        if m2 != m:
            raise ValueError(f"bad record marker at {p}")
        p += 8 + m
    return out


def read_fort10(path) -> dict:
    """Decode an xnfpelsyn fort.10 into numpy arrays (as SYNTHE/SPECTRV read it)."""
    rec = _records(path)
    nt, teff, glog = struct.unpack_from("<idd", rec[0])
    title = rec[0][20:94]
    nin = struct.unpack_from("<i", rec[1])[0]
    e = np.frombuffer(rec[1], np.float64, 3 * nin, 4).reshape(nin, 3)
    idm = np.frombuffer(rec[1], np.float64, MM, 4 + 24 * nin)
    mom = np.frombuffer(rec[1], np.float64, MM, 4 + 24 * nin + 8 * MM)
    numnu = struct.unpack_from("<i", rec[2])[0]
    fs = np.frombuffer(rec[2], np.float64, numnu, 4)
    s = np.frombuffer(rec[3], np.float64)
    names = [k for k, _ in STATE_KEYS]
    state, p = {}, 0
    for k in names + ["xnfh2"]:
        m = 2 * KW if k == "xnfhe" else KW
        state[k] = s[p:p + m].reshape((KW, 2), order="F") if k == "xnfhe" else s[p:p + m]
        p += m
    lay = rec[4:]
    if len(lay) != 4 * nt:
        raise ValueError(f"{len(lay)} layer records for NT={nt}")
    cont = [np.stack([np.frombuffer(lay[4 * j + i], np.float64) for j in range(nt)]) for i in range(3)]
    xd = [np.frombuffer(lay[4 * j + 3], np.float64) for j in range(nt)]
    xnfpel = np.stack([x[:6 * MW].reshape((6, MW), order="F") for x in xd])
    dopple = np.stack([x[6 * MW:].reshape((6, MW), order="F") for x in xd])
    return dict(nt=nt, teff=teff, glog=glog, title=title, nedge=nin, edges=e, idmol=idm, momass=mom,
                freqset=fs, state=state, continall=cont[0], contabs=cont[1], contscat=cont[2],
                xnfpel=xnfpel, dopple=dopple)


def main(argv=None) -> int:
    """Command line: python -m atlas_torch.synthe.xnfpelsyn_prog < model.deck  (fort.17, fort.2 and fort.10
    in the current directory, like the Fortran executable)."""
    import sys
    deck = sys.stdin.read()
    cont = Path("fort.17").read_text() if Path("fort.17").exists() else None
    mol = Path("fort.2") if Path("fort.2").exists() else None
    xnfpelsyn(deck, cont, mol, out10="fort.10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
