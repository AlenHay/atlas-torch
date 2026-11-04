"""Differentiable XNFPELSYN: populations, Doppler widths and continuum opacity tables of a model atmosphere as
functions of the physically interesting inputs.

Inputs (`params`, every entry optional, torch tensors may carry forward-mode tangents or requires_grad):
  "T", "P", "XNE", "RHOX"  (n,) the model structure per layer (absolute; default: the deck's values)
  "abund"                   {Z or symbol: dex} or a (99,) tensor: BasicATLAS synthe(abun_adjust=...) semantics,
                            N(Z) -> N(Z)*10**dex and all number fractions renormalised (H and He included)
  "mh"                      dex: ABUNDANCE SCALE change, every metal (Z >= 3) times 10**mh, no renormalisation
  "vturb"                   SYNTHE microturbulence [km/s] (used by atlas_torch.diff.lines; carried through here)

The program flow is atlas_torch.synthe.xnfpelsyn_prog._xnfpelsyn; the physics is the port's own torch code
(PFSAHA, KAPP and its opacity kernels) run under atlas_torch.diff.native.native_math(), except
  * NMOLEC: atlas_torch.diff.nmolec (exact iterate + implicit differentiation),
  * POPS: `pops_diff` below (same logic as synthe.pops.pops, but XABUND stays a tensor),
  * READIN's derived quantities (TK, HKT, HCKT, TKEV, TLOG, XNATOM, RHO, XABUND, WTMOLE) recomputed in torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..fnum import f32
from ..synthe import xnfpelsyn_prog as XP
from ..synthe.pfsaha import Layers, pfsaha
from ..synthe.pops import R4_001, R4_01
from . import nmolec as NM
from .native import native_math

DT = torch.float64
KW = 99
MW = 139

SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb "
           "Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta "
           "W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cf Es").split()


def zof(el) -> int:
    if isinstance(el, (int, np.integer)):
        return int(el)
    return SYMBOLS.index(str(el)) + 1


def _np(v):
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


@dataclass
class Base:
    """Everything about a SYNTHE model deck that does not depend on the differentiated parameters."""
    deck_text: str
    molecules_path: Path
    state0: object                  # State after READIN(20) + the xnfpelsyn main-program settings
    n: int
    ifmol: int
    plan: NM.Plan
    freqset: np.ndarray
    frq: np.ndarray
    wl: np.ndarray
    cm: np.ndarray
    idmol: np.ndarray
    momass: np.ndarray
    exact: dict = field(default_factory=dict)    # cache: key of the structure -> NM.Exact

    @classmethod
    def from_deck(cls, deck_path_or_text, molecules_path=None, continua_path=None) -> "Base":
        from ..synthe.freeform import Free
        from ..synthe.readin import readin
        paths = XP.default_paths()
        text = deck_path_or_text
        if "\n" not in str(deck_path_or_text):
            text = Path(deck_path_or_text).read_text()
        mol = Path(molecules_path or paths["molecules"])
        s = readin(text, molecules_path=mol, mode=20)
        for key in ("choh.xnfpch", "choh.xnfpoh"):
            if key not in s:
                s[key] = torch.zeros(KW, dtype=DT)
        ifop = _np(s["if.ifop"]).copy()
        ifop[13:17] = 0
        upd = {"if.ifop": torch.as_tensor(ifop, dtype=s["if.ifop"].dtype).reshape(s["if.ifop"].shape)}
        for k, v in (("temp.itemp", 1), ("if.ifpres", 1), ("iter.iter", 1), ("iter.numits", 1)):
            old = s.get(k)
            upd[k] = torch.full_like(old, v) if isinstance(old, torch.Tensor) else torch.tensor([v])
        s = s.updated(upd)
        fr = Free.from_state({k: s[k] for k in ("free.numcol", "free.letcol", "free.last", "free.more",
                                                "free.iffail", "free.maxpow", "free.word")})
        fr.more, fr.maxpow, fr.last, fr.numcol = 1, 99, 81, 1
        cont = Path(continua_path or paths["continua"]).read_text()
        frq, wl, cm = XP.read_edges(cont, fr)
        freqset = XP.freqset_of(frq, wl)
        idmol, momass = XP.idmol_momass()
        n = int(_np(s["rhox.nrhox"]).reshape(-1)[0])
        ifmol = int(_np(s["if.ifmol"]).reshape(-1)[0])
        if ifmol != 1:
            raise NotImplementedError("differentiable XNFPELSYN needs MOLECULES ON (BasicATLAS' SYNTHE deck)")
        return cls(text, mol, s, n, ifmol, NM.build_plan(s), freqset, frq, wl, cm, idmol, momass)

    # ---------------------------------------------------------------------- deck values
    def deck(self, key: str) -> torch.Tensor:
        return self.state0[key][:self.n].to(DT).clone()


# ---------------------------------------------------------------------------------------- parameters -> State

def abundance_weights(params: dict, dev=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(w_renorm (99,), w_scale (99,)): multiplicative factors on XABUND for "abund" (renormalised) and "mh"."""
    one = torch.ones(KW, dtype=DT, device=dev)
    w = one
    ab = params.get("abund")
    if ab is not None:
        if isinstance(ab, dict):
            d = torch.zeros(KW, dtype=DT, device=dev)
            for el, v in ab.items():
                z = zof(el)
                if z < 3:
                    raise ValueError("abund: H and He are set by renormalisation (BasicATLAS semantics)")
                d = d + torch.nn.functional.one_hot(torch.tensor(z - 1), KW).to(DT) * torch.as_tensor(v, dtype=DT)
        else:
            d = torch.as_tensor(ab, dtype=DT)
        w = torch.pow(10.0, d)
    mh = params.get("mh")
    sc = one
    if mh is not None:
        metal = (torch.arange(KW) >= 2).to(DT)
        sc = torch.pow(10.0, torch.as_tensor(mh, dtype=DT) * metal)
    return w, sc


def inject(base: Base, params: dict):
    """State after READIN with the parameters substituted (READIN's derived quantities recomputed in torch)."""
    s = base.state0
    n = base.n
    upd = {}

    def put(key, rows):
        full = s[key].detach().to(DT)
        upd[key] = torch.cat([rows.to(DT), full[n:]], 0)
    t = params.get("T", base.deck("temp.t"))
    p = params.get("P", base.deck("state.p"))
    xne = params.get("XNE", base.deck("state.xne"))
    rhox = params.get("RHOX", base.deck("rhox.rhox"))
    x0 = s["xabund.xabund"].to(DT)
    w, sc = abundance_weights(params)
    xw = x0 * w
    x = xw / (1.0 + (x0 * (w - 1.0)).sum())
    x = x * sc
    atm = s["elem.atmass"].to(DT).reshape(-1)[:KW]
    wtmole = (x * atm).sum()
    upd["xabund.xabund"] = x
    upd["xabund.wtmole"] = wtmole.reshape(1)
    tk = 1.38054e-16 * t
    hkt = torch.div(torch.full_like(tk, 6.6256e-27), tk)
    xnatom = p / tk - xne
    put("temp.t", t)
    put("temp.tk", tk)
    put("temp.hkt", hkt)
    put("temp.hckt", hkt * 2.99792458e10)
    put("temp.tkev", 8.6171e-5 * t)
    put("temp.tlog", torch.log(t))
    put("state.p", p)
    put("state.xne", xne)
    put("state.xnatom", xnatom)
    put("state.rho", xnatom * wtmole * 1.660e-24)
    put("rhox.rhox", rhox)
    if "state.taustd" in s:
        upd["state.taustd"] = upd["state.xnatom"]
    return s.updated(upd)


# ---------------------------------------------------------------------------------------------------- POPS

def pops_diff(s, code: float, mode: int, number=None) -> torch.Tensor:
    """CALL POPS(CODE,MODE,NUMBER) with IFMOL=1 after NMOLEC (same branch structure as synthe.pops.pops, the
    molecule/ion densities from the State's XNMOL/XNFPMOL, PFSAHA*XNATOM*XABUND otherwise), XABUND a tensor.
    Returns the (99, k) NUMBER array."""
    code = float(code)
    mode = int(mode)
    n = int(_np(s["rhox.nrhox"]).reshape(-1)[0])
    if number is None:
        number = torch.zeros((KW, 10), dtype=DT)
    number = torch.as_tensor(number, dtype=DT)
    if number.ndim == 1:
        number = number[:, None]
    cols = [number[:, i] for i in range(number.shape[1])]
    if code == 0.0:
        return number
    xnatom = s["state.xnatom"][:n].to(DT)
    xabund = s["xabund.xabund"].to(DT)

    def setcol(i, rows):
        cols[i] = torch.cat([rows.to(DT), cols[i][n:]], 0)

    def saha(iz: int, nion: int, nn: int):
        L = Layers.from_state(s, n)
        a = pfsaha(L, iz, nion, mode)
        if a.ndim == 1:
            setcol(0, a)
        else:
            for i in range(a.shape[1]):
                setcol(i, a[:, i])
        for i in range(nn):
            setcol(i, cols[i][:n] * xnatom * xabund[iz - 1])

    codemol = _np(s["xnmol.codemol"] if "xnmol.codemol" in s else s["xnmol.code"]).astype(np.float64)
    nummol = int(_np(s["xnmol.nummol"]).reshape(-1)[0])
    codemol = codemol[:nummol]
    xnmol = s["xnmol.xnmol"][:n].to(DT)
    xnfpmol = s["xnmol.xnfpmol"][:n].to(DT)

    def put(ion: int, jmol: int, m10: int) -> bool:
        if m10 == 2:
            setcol(ion - 1, xnmol[:, jmol])
        elif m10 == 1:
            setcol(ion - 1, xnfpmol[:, jmol])
        elif m10 == 3:
            setcol(ion - 1, xnmol[:, jmol] / xnfpmol[:, jmol])
        else:
            return False
        return True

    def done():
        return torch.stack(cols, 1)

    if code >= 100.0:
        hit = np.nonzero(codemol == code)[0]
        if hit.size == 0:
            setcol(0, torch.zeros(n, dtype=DT))
            return done()
        put(1, int(hit[0]), mode if mode in (1, 2, 3) else -1)
        return done()
    c = code
    nn = int((c - float(int(c))) * 100.0 + 1.5) if mode > 10 else 1
    idc = int(code)
    for i in range(1, nn + 1):
        cplus = c + R4_001
        cminus = c - R4_001
        ion = nn - i + 1
        hit = np.nonzero((codemol > cminus) & (codemol < cplus))[0]
        if hit.size and put(ion, int(hit[0]), mode % 10):
            c = c - R4_01
            continue
        if np.any(np.trunc(codemol).astype(np.int64) == idc):
            setcol(ion - 1, torch.zeros(n, dtype=DT))
            c = c - R4_01
            continue
        ion4 = int((code - float(idc)) * 100.0 + 1.5)
        saha(idc, ion4, 1 if mode == 1 else ion4)
        return done()
    return done()


# ------------------------------------------------------------------------------------------- the program

@dataclass
class XnfResult:
    n: int
    state: object               # attached State (populations, XNE, RHO ...) for downstream use
    t: torch.Tensor
    xne: torch.Tensor
    rho: torch.Tensor
    rhox: torch.Tensor
    hkt: torch.Tensor
    hckt: torch.Tensor
    tk: torch.Tensor
    xnfh: torch.Tensor          # (n,)
    xnfhe: torch.Tensor         # (n, 2)
    xnfh2: torch.Tensor         # (n,)
    xnfpel: torch.Tensor        # (n, 6, 139)
    dopple: torch.Tensor        # (n, 6, 139) without SYNTHE's TURBV
    contabs: torch.Tensor       # (n, NUMNU) log10 ACONT at FREQSET
    contscat: torch.Tensor      # (n, NUMNU) log10 SIGMAC
    nm_iters: np.ndarray = None


def exact_nmolec(base: Base, s) -> NM.Exact:
    """The exact NMOLEC iterate for the (detached) structure of `s`, cached by its bytes."""
    n = base.n
    key = b"".join(_np(s[k][:n]).astype(np.float64).tobytes() for k in ("temp.t", "state.p", "state.xne")) + \
        _np(s["xabund.xabund"]).astype(np.float64).tobytes()
    if key not in base.exact:
        sd = s.updated({k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in s.items()
                        if k.startswith(("temp.", "state.", "xabund.", "rhox."))})
        if len(base.exact) > 16:
            base.exact.clear()
        base.exact[key] = NM.solve_exact(sd, base.plan)
    return base.exact[key]


def xnfpelsyn_diff(base: Base, params: dict | None = None, polish: bool = False) -> XnfResult:
    """XNFPELSYN as a differentiable function of `params` (see module docstring)."""
    params = params or {}
    n = base.n
    s = inject(base, params)
    ex = exact_nmolec(base, s)                         # exact libm, detached
    with native_math():
        att = NM.attach(base.plan, s, ex, polish=polish)
        s = s.updated(NM.finish(base.plan, s, att))
        s = s.updated({"save.pops.itemp1": torch.tensor([1], dtype=torch.int64)})

        def pops_into(s, key, code, mode, ncol):
            cur = s[key].to(DT).reshape(KW, -1)
            v = pops_diff(s, code, mode, cur[:, :ncol])
            full = torch.cat([v[:, :ncol], cur[:, ncol:]], 1)
            return s.updated({key: full.reshape(s[key].shape)})
        for key, code, mode, ncol in XP.POPS_COMMON:
            s = pops_into(s, key, code, mode, ncol)
        s = pops_into(s, "choh.xnfpch", 106.00, 1, 1)
        s = pops_into(s, "choh.xnfpoh", 108.00, 1, 1)
        g = lambda k: s[k][:n].to(DT).reshape(n, -1)[:, 0]          # noqa: E731
        t, tkev, tlog = g("temp.t"), g("temp.tkev"), g("temp.tlog")
        xnfh, xnfph1, xnfpc1, xnfpo1 = g("bhyd.xnfh"), g("bhyd.xnfph"), g("bc.xnfpc"), g("bo.xnfpo")
        from ..atlas9.physics import libm
        from ..synthe.opacity._core import powi, rdiv
        eq = libm.exp(rdiv(4.478, tkev) - 4.64584e1 + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (
            1.06206e-18 - 3.08720e-23 * t) * t) * t) * t) * t) * t - 1.5 * tlog)
        eqco = libm.exp(rdiv(f32(11.091), tkev) - f32(49.0414) + 14.0306e-4 * t - 26.6341e-8 * (t * t)
                        + 35.382e-12 * powi(t, 3) - 26.5424e-16 * powi(t, 4) + 8.32385e-20 * powi(t, 5) - 1.5 * tlog)
        cold = ~(t > 9000.0)
        zero = torch.zeros_like(t)
        xnfh2 = torch.where(cold, xnfh * xnfh * eq, zero)
        xnfph2 = torch.where(cold, xnfph1 * xnfph1 * eq, zero)
        xnfpco = torch.where(cold, xnfpc1 * xnfpo1 * eqco, zero)

        # continuum: KAPP at every FREQSET point (existing torch kernels, POPS -> pops_diff)
        hkt = g("temp.hkt")
        c = XP.Ctx(s, *XP.freq_setup(base.freqset, hkt))
        from ..synthe.kapp import kapp_k
        sp, _, _, _ = kapp_k(c, 1, lambda st, code, mode: pops_diff(st, code, mode))
        acont, sigmac = sp["optot.acont"], sp["optot.sigmac"]
        contabs = torch.log10(acont).T
        contscat = torch.log10(sigmac).T

        # XNFP(J,ION,NELEM): loop 30 (POPS(code,11) for elements 1..99), loop 444's XNFPH2/XNFPCO at (6,40)/(6,46)
        # are overwritten by POPS(IDMOL(NELEM-39),1) for NELEM=40..139 (IFMOL=1), loop 250 copies ions 7..10 of
        # elements 20..28 into ion slot 5 of elements 50..88; elements 100..139 of ions 1..5 stay 0.
        zc = torch.zeros(n, dtype=DT)
        ent = [[zc] * MW for _ in range(10)]                               # ent[ion][nelem-1] (n,)
        for nelem in range(1, 100):
            code = XP.XNFP_CODES[nelem - 1] if nelem <= 28 else float(np.float32(nelem)) + .02
            v = pops_diff(s, code, 11, torch.zeros(KW, 10, dtype=DT))[:n]
            for ion in range(10):
                ent[ion][nelem - 1] = v[:, ion]
        ent[5][39] = xnfph2
        ent[5][45] = xnfpco
        for nelem in range(40, MW + 1):
            ent[5][nelem - 1] = pops_diff(s, float(base.idmol[nelem - 40]), 1, torch.zeros(KW, 1, dtype=DT))[:n, 0]
        for nelem in range(20, 29):
            for off, ion in ((30, 7), (40, 8), (50, 9), (60, 10)):
                ent[4][off + nelem - 1] = ent[ion - 1][nelem - 1]
        xnfpel = torch.stack([torch.stack(ent[i], 1) for i in range(6)], 1)    # (n, 6, 139)

        tk = g("temp.tk")
        vt = s["turbpr.vturb"][:n].to(DT)
        atmass = s["elem.atmass"].to(DT).reshape(-1)[:99]
        d1 = torch.sqrt(2.0 * tk[:, None] / atmass[None, :] / 1.660e-24 + (vt * vt)[:, None]) / 2.99792458e10
        mom = torch.as_tensor(base.momass[:60], dtype=DT)
        dm = torch.sqrt(2.0 * tk[:, None] / mom[None, :] / 1.660e-24 + (vt * vt)[:, None]) / 2.99792458e10
        dz = torch.zeros(n, MW - 99, dtype=DT)
        dop5 = list(torch.cat([d1, dz], 1).unbind(1))
        for nelem in range(20, 29):
            for off in (30, 40, 50, 60):
                dop5[off + nelem - 1] = d1[:, nelem - 1]
        row = torch.cat([d1, dz], 1)
        row6 = torch.cat([d1[:, :39], dm, dz], 1)
        dop = torch.stack([row, row, row, row, torch.stack(dop5, 1), row6], 1)
        xnfhe = s["bhe.xnfhe"][:n].to(DT).reshape(n, -1)[:, :2]
        return XnfResult(n=n, state=s, t=t, xne=g("state.xne"), rho=g("state.rho"), rhox=g("rhox.rhox"),
                         hkt=hkt, hckt=g("temp.hckt"), tk=tk, xnfh=xnfh, xnfhe=xnfhe, xnfh2=xnfh2,
                         xnfpel=xnfpel, dopple=dop, contabs=contabs, contscat=contscat, nm_iters=ex.iters)
