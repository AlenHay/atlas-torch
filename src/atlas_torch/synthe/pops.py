"""POPS, NMOLEC, NELECT, SOLVIT of atlas7v.for (SYNTHE variant): populations, molecular + ionisation
equilibrium.

API (pure functions over the explicit State, see atlas_torch.state):
  pops(s, code, mode, number=None) -> (number, updates)   CALL POPS(CODE,MODE,NUMBER)
  nmolec(s) -> updates                                     CALL NMOLEC
  nelect(s) -> updates                                     CALL NELECT
  solvit(a, b) -> (a, b, ipivot)                           CALL SOLVIT(A,N,B,IPIVOT)

POPS caching (-fno-automatic): POPS keeps ITEMP1 (DATA ITEMP1/0/) as "save.pops.itemp1". With IFPRES=1 it
runs NMOLEC (IFMOL=1) or NELECT (IFMOL=0) only when /TEMP/ ITEMP differs from ITEMP1, then sets
ITEMP1=ITEMP. xnfpelsyn sets ITEMP=1 once, so the first POPS call solves the equilibrium for all layers
(updating XNE, XNATOM, RHO, XNMOL, XNFPMOL, NLTEON in the COMMONs) and every later call only reads the
stored XNMOL/XNFPMOL (molecule/ion codes in the molecule list) or calls PFSAHA with the updated XNE/XNATOM.
`updates` contains everything the call writes to the State (including save.pops.itemp1).

NUMBER is the caller's array (Fortran NUMBER(kw,10)); POPS only overwrites some columns -- and in the
MODE=2/3 path to PFSAHA it rescales columns 2..ION that PFSAHA did not write -- so `number` (shape
(99, k), k >= the columns touched) carries the caller's contents; zeros (99,10) if None. The returned array
is a modified copy.

NMOLEC: the layer loop is an inherent recurrence (layer J starts its Newton iteration from the converged
solution of layer J-1 scaled by P(J)/P(J-1), and XNE(J) = XNE(J-1)*P(J)/P(J-1) enters the ion equilibrium
constants of layer J); because the convergence test is loose (|dX/X| < 1E-3) a different start changes the
result at the 1e-6 level, so it cannot be batched over layers without losing the Fortran's answer. Hence:
everything that does not depend on the recurrence is vectorised over all layers up front (equilibrium
constants of the neutral molecules, the NLTE partition-function corrections CPF -- which PFSAHA evaluates
with the *input* XNE(J), not the updated one --, PFSAHA's T-only stage for every ion), the layer loop runs
sequentially, and each Newton step is vectorised over molecules/equations: TERM products column by
column, the Jacobian assembled with an ordered scatter-add (np.add.at accumulates in index order = the
Fortran's summation order for every DEQ/EQ element), SOLVIT's row operations as whole-matrix updates. The
Newton solve itself runs in float64 numpy on the CPU (scalar-sized linear algebra); inputs and outputs are
torch tensors on the State's device. The Fortran's Newton loop has no iteration cap; a safety cap
(MAXIT) raises instead of looping forever.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..atlas9.physics import libm
from ..fnum import f32
from .freeform import FortranExit
from .pfsaha import Layers, SahaPlan, nions_of, pfsaha

KW = 99
DT = torch.float64
MAXIT = 100000

R4_001 = f32(.001)
R4_69 = f32(.69)
R4_01 = f32(.01)
R4_0005 = f32(.0005)
R4_00001 = f32(.00001)
R4_05 = f32(.05)

# CPF elements of NMOLEC: (IZ, State key of the departure coefficients B(J,1))
_CPF = ((1, "bhyd.bhyd"), (6, "bc.bc1"), (8, "bo.bo1"), (12, "bmg.bmg1"), (13, "bal.bal1"), (14, "bsi.bsi1"),
        (20, "bca.bca1"))


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _i(s, k, default=0) -> int:
    return int(_np(s[k]).reshape(-1)[0]) if k in s else default


def _f(s, k) -> float:
    return float(_np(s[k]).reshape(-1)[0])


# ------------------------------------------------------------------------------------------------ SOLVIT

def solvit(a, b):
    """SUBROUTINE SOLVIT(A,N,B,IPIVOT): Gauss-Jordan with full pivoting (first maximum |A(J,K)| in the
    Fortran's row-major scan). a: (N,N) [row, column]; returns (A destroyed, B = solution, IPIVOT)."""
    a = np.array(_np(a), dtype=np.float64, copy=True)
    b = np.array(_np(b), dtype=np.float64, copy=True)
    n = b.shape[0]
    piv = np.zeros(n, dtype=bool)
    ipivot = np.zeros(n, dtype=np.int64)
    for _ in range(n):
        cand = np.abs(a)
        cand[piv, :] = -1.0
        cand[:, piv] = -1.0
        idx = int(np.argmax(cand))
        if not cand.flat[idx] > 0.0:
            raise FloatingPointError("SOLVIT: no positive pivot (singular or NaN matrix)")
        irow, icol = divmod(idx, n)
        piv[icol] = True
        ipivot[icol] += 1
        if irow != icol:
            a[[irow, icol], :] = a[[icol, irow], :]
            b[irow], b[icol] = b[icol], b[irow]
        pivot = a[icol, icol]
        a[icol, icol] = 1.0
        a[icol, :] = a[icol, :] / pivot
        b[icol] = b[icol] / pivot
        t = a[:, icol].copy()
        rows = np.arange(n) != icol
        a[rows, icol] = 0.0
        a[rows, :] = a[rows, :] - a[icol, :][None, :] * t[rows, None]
        b[rows] = b[rows] - b[icol] * t[rows]
    return a, b, ipivot


# ------------------------------------------------------------------------------------------ molecule table

@dataclass
class MolTab:
    """COMMON /IFEQUA/ + /XNMOL/ codes in Python form (1-based equation numbers)."""
    nummol: int
    nequa: int
    idequa: np.ndarray          # IDEQUA(1..MAXEQ) at index 0..MAXEQ-1
    code: np.ndarray            # (nummol,)
    equil: np.ndarray           # (7, nummol)
    comps: list                 # per molecule: list of K (1..NEQUA+1)

    @classmethod
    def from_state(cls, s) -> "MolTab":
        nummol = _i(s, "xnmol.nummol")
        nequa = _i(s, "ifequa.nequa")
        locj = _np(s["ifequa.locj"]).astype(np.int64)
        kc = _np(s["ifequa.kcomps"]).astype(np.int64)
        code = _np(s["xnmol.code"] if "xnmol.code" in s else s["xnmol.codemol"]).astype(np.float64)[:nummol]
        comps = [list(kc[locj[j] - 1:locj[j + 1] - 1]) for j in range(nummol)]
        return cls(nummol, nequa, _np(s["ifequa.idequa"]).astype(np.int64), code,
                   _np(s["ifequa.equil"]).astype(np.float64)[:, :nummol], comps)

    def ide(self, k: int) -> int:
        """IDEQUA(K) (K 1-based)."""
        return int(self.idequa[k - 1])


class _Events:
    """Ordered scatter-add program for the Newton system of NMOLEC (loops 61-99).

    Target vector V = [EQ(1..NE), DEQ(1..NE*NE) column-major]. Source values per Newton step:
    TERM(JMOL) (src 0), D(JMOL, comp) (src 1: TERM/XN(K), or -TERM/XN(NE) for K=NEQUA1),
    D2(JMOL, comp) (src 2: TERM/XN(K), negative-ion correction). Events are listed in the Fortran's
    execution order, so np.add.at reproduces each element's summation sequence."""

    def __init__(self, mt: MolTab):
        ne, ne1 = mt.nequa, mt.nequa + 1
        self.ne = ne
        mols = [j for j in range(mt.nummol) if len(mt.comps[j]) > 1]
        self.mols = np.array(mols, dtype=np.int64)
        maxc = max((len(mt.comps[j]) for j in mols), default=1)
        self.maxc = maxc
        nm = len(mols)
        compk = np.zeros((nm, maxc), dtype=np.int64)          # K (1-based), 0 = padding
        for r, j in enumerate(mols):
            compk[r, :len(mt.comps[j])] = mt.comps[j]
        self.compk = compk
        self.valid = compk > 0
        self.is101 = compk == ne1
        # XN index used by D: K, or NE for K=NEQUA1
        self.kd = np.where(self.is101, ne, compk)
        tgt, src, idx, sgn = [], [], [], []

        def ev(t, s, i, g):
            tgt.append(t); src.append(s); idx.append(i); sgn.append(g)

        def deq(m, k):
            return ne + (m - 1) + (k - 1) * ne
        for r, j in enumerate(mols):
            cj = mt.comps[j]
            ev(0, 0, r, 1.0)                                    # EQ(1)=EQ(1)+TERM
            for p, k in enumerate(cj):
                kk = ne if k == ne1 else k
                ev(kk - 1, 0, r, 1.0)                           # EQ(K)=EQ(K)+TERM
                ev(deq(1, kk), 1, r * maxc + p, 1.0)            # DEQ(1,K)=DEQ(1,K)+D
                for m in cj:
                    mm = ne if m == ne1 else m
                    ev(deq(mm, kk), 1, r * maxc + p, 1.0)       # DEQ(M,K)=DEQ(M,K)+D
            klast = cj[-1]
            if (mt.ide(klast) if klast <= len(mt.idequa) else 0) != 100:
                continue
            for p, k in enumerate(cj):                          # negative ions: charge equation
                if k == ne1:
                    raise NotImplementedError("NMOLEC negative-ion correction with an NEQUA1 component")
                if k == ne:
                    ev(k - 1, 0, r, -1.0)
                    ev(k - 1, 0, r, -1.0)
                for m in cj:
                    if m != ne:
                        continue
                    ev(deq(m, k), 2, r * maxc + p, -1.0)
                    ev(deq(m, k), 2, r * maxc + p, -1.0)
        self.tgt = np.array(tgt, dtype=np.int64)
        self.src = np.array(src, dtype=np.int64)
        self.idx = np.array(idx, dtype=np.int64)
        self.sgn = np.array(sgn, dtype=np.float64)

    def terms(self, equilj: np.ndarray, xn: np.ndarray):
        """TERM per listed molecule, D and D2 per (molecule, comp) (xn 1-based, length NE+1)."""
        term = equilj[self.mols].copy()
        ne = self.ne
        for p in range(self.maxc):
            k = self.compk[:, p]
            v = self.valid[:, p]
            mul = term * xn[np.where(v & ~self.is101[:, p], k, 1)]
            div = term / xn[ne]
            term = np.where(v, np.where(self.is101[:, p], div, mul), term)
        xk = xn[np.where(self.valid, self.kd, 1)]
        d = np.where(self.is101, -term[:, None] / xk, term[:, None] / xk)
        d2 = term[:, None] / xn[np.where(self.valid, self.compk.clip(max=ne), 1)]
        return term, d, d2

    def apply(self, v: np.ndarray, term, d, d2):
        srcs = (term, d.reshape(-1), d2.reshape(-1))
        vals = np.empty(self.tgt.shape[0])
        for s in range(3):
            m = self.src == s
            vals[m] = self.sgn[m] * srcs[s][self.idx[m]]
        np.add.at(v, self.tgt, vals)


def _h2_equil(t, tkev, tlog):
    """EQUILJ of H2 (CODE=101.), NMOLEC."""
    return libm.exp(libm.rdiv(4.478, tkev) - 4.64584e1 + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (
        1.06206e-18 - 3.08720e-23 * t) * t) * t) * t) * t) * t - 1.5 * tlog)


def _bcol(s, key: str, n: int) -> torch.Tensor:
    return s[key][:n, 0].to(DT)


# ------------------------------------------------------------------------------------------------ NMOLEC

def nmolec(s, maxit: int = MAXIT, hook=None) -> dict:
    """SUBROUTINE NMOLEC. Returns the State entries it writes: xnmol.xnmol, xnmol.xnfpmol, state.xne,
    state.xnatom (+ its READIN alias state.taustd if present), state.rho, if.nlteon (=0).
    hook(j, it, deq, eq), if given, sees every Newton system (1-based layer J, iteration) before SOLVIT."""
    mt = MolTab.from_state(s)
    n = _i(s, "rhox.nrhox")
    ne, ne1 = mt.nequa, mt.nequa + 1
    nummol = mt.nummol
    dev = s["temp.t"].device
    L = Layers.from_state(s, n)
    t, tkev, tlog, tk, p = L.t, L.tkev, L.tlog, L.tk, L.p
    xabund = _np(s["xabund.xabund"]).astype(np.float64)
    wtmole = _f(s, "xabund.wtmole")
    atmass = _np(s["elem.atmass"]).astype(np.float64)

    # ---- loop 31/32: XAB, starting XN, XNE(1)
    xab = np.zeros(ne + 1)
    idl = 0
    for k in range(2, ne + 1):
        idl = mt.ide(k)
        if idl < 100:
            xab[k] = max(xabund[idl - 1], 1e-20)
    if idl == 100:
        xab[ne] = 0.0
    pn, tkn = _np(p).astype(np.float64), _np(tk).astype(np.float64)
    xntot = pn[0] / tkn[0]
    xn = np.zeros(ne + 1)
    xn[1] = xntot / 2.0
    x = xn[1] / 10.0
    for k in range(2, ne + 1):
        xn[k] = x * xab[k]
    if idl == 100:
        xn[ne] = x
    xne = _np(L.xne).astype(np.float64).copy()
    xne[0] = x

    # ---- CPF (NLTE corrections), all layers: PFSAHA(J,IZ,1,3) with NLTEON=-1 and 0 and the XNE(J) seen at
    #      that point of the layer loop (X for J=1, the input XNE(J) for J>1)
    Lc = L.with_(xne=torch.as_tensor(xne, dtype=DT, device=dev))
    rows = [(iz, min(1 + 2, nions_of(iz))) for iz, _ in _CPF]
    pf_m1 = SahaPlan(Lc.with_(nlteon=-1), rows).parts(slice(None), Lc.with_(nlteon=-1))[0][:, :, 0]
    pf_0 = SahaPlan(Lc.with_(nlteon=0), rows).parts(slice(None), Lc.with_(nlteon=0))[0][:, :, 0]
    cpf = {}
    for r, (iz, key) in enumerate(_CPF):
        cpf[iz] = pf_m1[:, r] / pf_0[:, r] * _bcol(s, key, n)

    # ---- EQUILJ of the molecules with EQUIL(1)/=0 (T only), all layers
    equil_mol = torch.zeros((n, nummol), dtype=DT, device=dev)
    ion_mols = []                                       # (jmol, ID, NCOMP) for EQUIL(1)=0, NCOMP>1
    one_mols = []
    hot = t > 10000.0
    for j in range(nummol):
        ncomp = len(mt.comps[j])
        eq = mt.equil[:, j]
        if eq[0] != 0.0:
            code = float(mt.code[j])
            ion = int((code - float(int(code))) * 100.0 + 0.5)
            if code == 101.0:
                e = _h2_equil(t, tkev, tlog)
            else:
                e = libm.exp(libm.rdiv(float(eq[0]), tkev) - float(eq[1]) + (float(eq[2]) + (-float(eq[3]) + (
                    float(eq[4]) + (-float(eq[5]) + float(eq[6]) * t) * t) * t) * t) * t
                    - 1.5 * float(ncomp - ion - ion - 1) * tlog)
            for k in mt.comps[j]:
                idk = mt.ide(k)
                if idk in cpf:
                    e = e * cpf[idk]
            equil_mol[:, j] = torch.where(hot, torch.zeros_like(e), e)
        elif ncomp == 1:
            one_mols.append(j)
            equil_mol[:, j] = 1.0
        else:
            ion_mols.append((j, int(mt.code[j]), ncomp))
    equil_np = _np(equil_mol).astype(np.float64).copy()

    # ---- PFSAHA(J,ID,NCOMP,12) for the ions: T-only stage for all layers now, XNE stage per layer
    L0 = L.with_(nlteon=0)
    if ion_mols:
        irows = []
        for _, iz, nc in ion_mols:
            n2 = min(nc + 2, nions_of(iz))
            if nc > n2:
                raise NotImplementedError(f"NMOLEC ion {iz} with NCOMP={nc} > NION2={n2}")
            irows.append((iz, n2))
        iplan = SahaPlan(L0, irows)
        ir = np.arange(len(ion_mols))
        inc = np.array([nc for _, _, nc in ion_mols])
        ij = np.array([j for j, _, _ in ion_mols])

    ev = _Events(mt)
    xnz = np.zeros((n, ne + 1))
    xnmol = np.zeros((n, nummol))
    xnatom = np.zeros(n)
    rho = np.zeros(n)
    for j in range(n):
        xntot = pn[j] / tkn[j]
        if j > 0:
            ratio = pn[j] / pn[j - 1]
            xne[j] = xne[j - 1] * ratio
            xn[1:] = xn[1:] * ratio
        equilj = equil_np[j].copy()
        if ion_mols:
            Lj = L0.index(slice(j, j + 1)).with_(xne=torch.tensor([xne[j]], dtype=DT, device=dev))
            frac = _np(iplan.fractions(slice(j, j + 1), Lj)[0][0]).astype(np.float64)    # (R, M)
            fr = frac[ir, inc - 1] / frac[ir, 0]
            equilj[ij] = [fr[r] * libm.ipow_var(xne[j], int(inc[r]) - 1) for r in range(len(ion_mols))]
        # ---- Newton iteration (labels 50-105)
        eqold = np.zeros(ne + 1)
        it = 0
        while True:
            it += 1
            if it > maxit:
                raise FloatingPointError(f"NMOLEC: no convergence in layer {j + 1} after {maxit} iterations")
            v = np.zeros(ne + ne * ne)
            e1 = -xntot
            for k in range(2, ne + 1):
                e1 = e1 + xn[k]
                v[ne + (k - 1) * ne] = 1.0                      # DEQ(1,K)
                v[k - 1] = xn[k] - xab[k] * xn[1]               # EQ(K)
                v[ne + (k - 1) + (k - 1) * ne] = 1.0            # DEQ(K,K)
                v[ne + (k - 1)] = -xab[k]                       # DEQ(K,1)
            v[0] = e1
            if mt.ide(ne) >= 100:
                v[ne - 1] = -xn[ne]
                v[ne + ne * ne - 1] = -1.0
            term, d, d2 = ev.terms(equilj, xn)
            ev.apply(v, term, d, d2)
            deq = v[ne:].reshape(ne, ne, order="F")
            if hook is not None:
                hook(j + 1, it, deq.copy(), v[:ne].copy())
            _, sol, _ = solvit(deq, v[:ne])
            iferr = False
            scale = 100.0
            for k in range(1, ne + 1):
                ek = sol[k - 1]
                if abs(ek / xn[k]) > R4_001:
                    iferr = True
                if eqold[k] * ek < 0.0:
                    ek = ek * R4_69
                xneq = xn[k] - ek
                if abs(xneq) < xn[k] / 100.0:
                    xn[k] = xn[k] / scale
                    if eqold[k] * ek < 0.0:
                        scale = math.sqrt(scale)
                else:
                    xn[k] = abs(xneq)
                eqold[k] = ek
            if not iferr:
                break
        xnz[j, 1:] = xn[1:]
        xnatom[j] = xn[1]
        rho[j] = xnatom[j] * wtmole * 1.660e-24
        if mt.ide(ne) == 100:
            xne[j] = xn[ne]
        for jm in range(nummol):
            vm = equilj[jm]
            for k in mt.comps[jm]:
                vm = vm / xn[ne] if k == ne1 else vm * xn[k]
            xnmol[j, jm] = vm

    # ---- loops 121-123: XNZ / partition functions, all layers (NLTEON=0, final XNE)
    td = t
    Lf = L0.with_(xne=torch.as_tensor(xne, dtype=DT, device=dev))
    xnz_t = torch.as_tensor(xnz, dtype=DT, device=dev)
    bkey = dict(_CPF)
    for k in range(2, ne + 1):
        idk = mt.ide(k)
        if idk == 100:
            xnz_t[:, k] = xnz_t[:, k] / 2.0 / 2.4148e15 / td / libm.sqrt(td)
            continue
        pf = pfsaha(Lf, idk, 1, 3)
        if idk in bkey:
            pf = pf / _bcol(s, bkey[idk], n)
        xnz_t[:, k] = xnz_t[:, k] / pf / 1.8786e20 / libm.sqrt(libm.ipow_const(float(atmass[idk - 1]) * td, 3))
    # ---- loop 140: XNFPMOL
    xnmol_t = torch.as_tensor(xnmol, dtype=DT, device=dev)
    xnfpmol_t = torch.zeros_like(xnmol_t)
    for jm in range(nummol):
        eq = mt.equil[:, jm]
        comps = mt.comps[jm]
        if eq[0] != 0.0:
            vm = libm.exp(libm.rdiv(float(eq[0]), tkev))
            amass = 0.0
            for k in comps:
                if k == ne1:
                    vm = vm / xnz_t[:, ne]
                    continue
                idk = mt.ide(k)
                if idk < 100:
                    amass = amass + float(atmass[idk - 1])
                vm = vm * xnz_t[:, k]
            xnfpmol_t[:, jm] = vm * 1.8786e20 * libm.sqrt(libm.ipow_const(amass * td, 3))
        else:
            pf = pfsaha(Lf, int(mt.code[jm]), len(comps), 3)
            xnfpmol_t[:, jm] = xnmol_t[:, jm] / pf

    out = {}
    xm = s["xnmol.xnmol"].clone().to(DT)
    xf = s["xnmol.xnfpmol"].clone().to(DT)
    xm[:n, :nummol] = xnmol_t
    xf[:n, :nummol] = xnfpmol_t
    out["xnmol.xnmol"] = xm
    out["xnmol.xnfpmol"] = xf
    out.update(_layer_updates(s, n, xne, xnatom, rho))
    out["if.nlteon"] = _like_int(s, "if.nlteon", 0)
    return out


def _like_int(s, key, value):
    ref = s.get(key)
    if isinstance(ref, torch.Tensor):
        return torch.full_like(ref, value)
    return torch.tensor([value], dtype=torch.int64)


def _layer_updates(s, n, xne, xnatom, rho) -> dict:
    out = {}
    for key, val in (("state.xne", xne), ("state.xnatom", xnatom), ("state.rho", rho)):
        a = s[key].clone().to(DT)
        a[:n] = torch.as_tensor(np.asarray(val)[:n], dtype=DT, device=a.device)
        out[key] = a
    if "state.taustd" in s:                     # READIN's EQUIVALENCE (TAUSTD(1),XNATOM(1))
        out["state.taustd"] = out["state.xnatom"]
    return out


# ------------------------------------------------------------------------------------------------ NELECT

_NELEMZ = (1, 2, 6, 11, 12, 13, 14, 19, 20, 26)
_NIONZ = (1, 2, 2, 2, 2, 2, 2, 2, 2, 2)


def nelect(s) -> dict:
    """SUBROUTINE NELECT (POPS with IFMOL=0): electron density by fixed-point iteration, layer by layer
    (layer J starts from XNE(J-1)*P(J)/P(J-1), a recurrence). Returns state.xne/xnatom/rho entries."""
    n = _i(s, "rhox.nrhox")
    L = Layers.from_state(s, n)
    dev = L.t.device
    xabund = _np(s["xabund.xabund"]).astype(np.float64)
    wtmole = _f(s, "xabund.wtmole")
    pn, tkn = _np(L.p).astype(np.float64), _np(L.tk).astype(np.float64)
    plan = SahaPlan(L, [(iz, min(ni + 2, nions_of(iz))) for iz, ni in zip(_NELEMZ, _NIONZ)])
    nz = len(_NELEMZ)
    n2 = [min(ni + 2, nions_of(iz)) for iz, ni in zip(_NELEMZ, _NIONZ)]
    xne = _np(L.xne).astype(np.float64).copy()
    xnatom = np.zeros(n)
    rho = np.zeros(n)
    xne[0] = pn[0] / tkn[0] / 2.0
    xsave = [0.0] * nz                       # SAVEd X(I)
    for j in range(n):
        if j > 0:
            xne[j] = xne[j - 1] * pn[j] / pn[j - 1]
        xntot = pn[j] / tkn[j]
        xnatom[j] = xntot - xne[j]
        mask = [True] * nz
        for _ in range(200):
            Lj = L.index(slice(j, j + 1)).with_(xne=torch.tensor([xne[j]], dtype=DT, device=dev))
            f = _np(plan.fractions(slice(j, j + 1), Lj)[0][0]).astype(np.float64)       # (10, M)
            xnenew = 0.0
            for i in range(nz):
                if not mask[i]:
                    continue
                elec = 0.0
                for ion in range(2, n2[i] + 1):
                    elec = elec + f[i, ion - 1] * float(ion - 1)
                xsave[i] = elec * xnatom[j] * xabund[_NELEMZ[i] - 1]
                xnenew = xnenew + xsave[i]
            xnenew = (xnenew + xne[j]) / 2.0
            error = abs((xne[j] - xnenew) / xnenew)
            xne[j] = xnenew
            xnatom[j] = xntot - xne[j]
            if error < R4_0005:
                break
            if j == 0:
                continue
            x1 = R4_00001 * xne[j]
            if error < R4_05:
                x1 = x1 * 10.0
            for i in range(nz):
                if xsave[i] < x1:
                    mask[i] = False
        else:
            raise FortranExit("XNE DOES NOT CONVERGE")
        rho[j] = xnatom[j] * wtmole * 1.660e-24
    return _layer_updates(s, n, xne, xnatom, rho)


# -------------------------------------------------------------------------------------------------- POPS

def pops(s, code: float, mode: int, number=None):
    """CALL POPS(CODE,MODE,NUMBER). Returns (number (99, k) tensor, State updates)."""
    code = float(code)
    mode = int(mode)
    n = _i(s, "rhox.nrhox")
    if number is None:
        number = torch.zeros((KW, 10), dtype=DT, device=s["temp.t"].device)
    number = torch.as_tensor(number, dtype=DT).clone()
    if number.ndim == 1:
        number = number[:, None]
    upd = {}
    ifmol, ifpres = _i(s, "if.ifmol"), _i(s, "if.ifpres")
    itemp, itemp1 = _i(s, "temp.itemp"), _i(s, "save.pops.itemp1")
    if ifpres == 1 and itemp != itemp1:
        upd.update(nmolec(s) if ifmol == 1 else nelect(s))
        s = s.updated(upd) if hasattr(s, "updated") else {**s, **upd}
    upd["save.pops.itemp1"] = torch.tensor([itemp], dtype=torch.int64)
    if code == 0.0:
        return number, upd
    xnatom = s["state.xnatom"][:n].to(DT)
    xabund = _np(s["xabund.xabund"]).astype(np.float64)

    def saha(iz: int, nion: int, nn: int):
        """PFSAHA(J,IZ,NION,MODE,NUMBER) for all J, then NUMBER(J,I)=NUMBER(J,I)*XNATOM(J)*XABUND(IZ),
        I=1..nn (columns PFSAHA did not write keep the caller's contents)."""
        L = Layers.from_state(s, n)
        a = pfsaha(L, iz, nion, mode)
        if a.ndim == 1:
            number[:n, 0] = a
        else:
            number[:n, :a.shape[1]] = a
        for i in range(nn):
            number[:n, i] = number[:n, i] * xnatom * float(xabund[iz - 1])

    if ifmol != 1:
        if code >= 1e2:
            raise FortranExit("MOLECULES OFF")
        iz = int(code)
        nion = int((code - float(iz)) * 100.0 + 1.5)
        saha(iz, nion, nion if mode >= 10 else 1)
        return number, upd

    codemol = _np(s["xnmol.codemol"] if "xnmol.codemol" in s else s["xnmol.code"]).astype(np.float64)
    nummol = _i(s, "xnmol.nummol")
    codemol = codemol[:nummol]
    xnmol = s["xnmol.xnmol"][:n].to(DT)
    xnfpmol = s["xnmol.xnfpmol"][:n].to(DT)

    def put(ion: int, jmol: int, m10: int) -> bool:
        if m10 == 2:
            number[:n, ion - 1] = xnmol[:, jmol]
        elif m10 == 1:
            number[:n, ion - 1] = xnfpmol[:, jmol]
        elif m10 == 3:
            number[:n, ion - 1] = xnmol[:, jmol] / xnfpmol[:, jmol]
        else:
            return False
        return True

    if code >= 100.0:
        hit = np.nonzero(codemol == code)[0]
        if hit.size == 0:
            number[:n, 0] = 0.0
            return number, upd
        put(1, int(hit[0]), mode if mode in (1, 2, 3) else -1)
        return number, upd

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
        # label 310
        if np.any(np.trunc(codemol).astype(np.int64) == idc):
            number[:n, ion - 1] = 0.0
            c = c - R4_01
            continue
        # label 400
        ion4 = int((code - float(idc)) * 100.0 + 1.5)
        saha(idc, ion4, 1 if mode == 1 else ion4)
        return number, upd
    return number, upd
