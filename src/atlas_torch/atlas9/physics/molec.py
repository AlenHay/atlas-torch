"""READMOL, NMOLEC, MOLEC, SOLVIT (atlas9mem.for / kappa9.for): molecular + ionisation equilibrium.

NMOLEC solves, layer by layer, the Newton system for the number densities of the NEQUA "components"
(XNATOM, one per element, XNE). The Fortran starts layer J from the converged solution of layer J-1
scaled by P(J)/P(J-1) (and XNE(J) = XNE(J-1)*P(J)/P(J-1) enters the ion equilibrium constants of layer J),
so the layer loop is an inherent recurrence and stays sequential. Everything inside a layer is vectorised:
equilibrium constants of all molecules, the TERM products, the Jacobian assembly (an ordered scatter-add:
np.add.at accumulates in index order, i.e. exactly the Fortran's sequential summation order per element),
SOLVIT's row operations. The per-layer Newton loop follows the Fortran convergence logic exactly
(RATIO>1E-4 test, sign-flip damping *.69, the /SCALE step limiter and its SQRT update, NCHEMITER cap).

Variant switch: atlas9 (BasicATLAS patch) stops after NCHEMITER=20000 and prints CHEMFAIL; kappa9 has no
cap. The Newton solve runs in float64 numpy on the CPU (scalar-sized linear algebra); inputs/outputs are
torch tensors on the State's device.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import libm
from .. import native as _native
from ._data import r4
from .pfsaha import Layers, SahaPlan, _nions, pfsaha

R4_1E4 = r4(".0001")
R4_464584 = r4("46.4584")
R4_69 = r4(".69")
R4_1001 = r4("1.001")
R4_999 = r4(".999")
R4_XCODE = [r4(x) for x in ("1.E14", "1.E12", "1.E10", "1.E8", "1.E6", "1.E4", "1.E2", "1.E0")]


# ------------------------------------------------------------------------------------------------ READMOL

def readmol(lines) -> dict[str, np.ndarray]:
    """SUBROUTINE READMOL: parse the molecule list (unit 2, BasicATLAS data/atlas_files/molecules.dat)
    into COMMON /IFEQUA/ (keys "ifequa.*"). `lines`: path or iterable of text lines."""
    if isinstance(lines, (str, bytes)) or hasattr(lines, "read_text"):
        lines = open(lines).read().split("\n")
    MAXMOL, MAX1, MAXEQ, MAXLOC = 160, 161, 25, 450
    ifequa = np.zeros(101, np.int64)
    kcomps = np.zeros(MAXLOC, np.int64)
    locj = np.zeros(MAX1, np.int64)
    code = np.zeros(MAXMOL)
    equil = np.zeros((6, MAXMOL))
    idequa = np.zeros(MAXEQ, np.int64)

    def fld(s, a, b):
        t = s[a:b].strip()
        return 0.0 if not t else float(t.replace("D", "E").replace("d", "e"))

    kloc = 1
    locj[0] = 1
    it = iter(lines)
    jmol = 0
    for jmol in range(1, MAX1 + 1):
        s = next(it, "").ljust(80)
        c = fld(s, 0, 18)
        e = [fld(s, 18, 25)] + [fld(s, 25 + 11 * i, 36 + 11 * i) for i in range(5)]
        if c == 0.0:
            break
        for ii in range(8):
            if c >= R4_XCODE[ii]:
                break
        else:
            raise ValueError(f"bad molecule code {c}")
        x = c
        for i in range(ii, 8):
            idd = int(x / R4_XCODE[i] + 0.5)
            x = x - float(idd) * R4_XCODE[i]
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
    nummol = jmol - 1
    nloc = kloc - 1
    iequa = 1
    for i in range(1, 101):
        if ifequa[i - 1] == 0:
            continue
        iequa += 1
        ifequa[i - 1] = iequa
        idequa[iequa - 1] = i
    nequa = iequa
    ifequa[100] = nequa + 1
    for k in range(nloc):
        kcomps[k] = ifequa[kcomps[k] - 1]
    return {"ifequa.ifequa": ifequa, "ifequa.kcomps": kcomps, "ifequa.locj": locj, "ifequa.code": code,
            "ifequa.equil": equil, "ifequa.idequa": idequa, "ifequa.nequa": np.array([nequa]),
            "ifequa.nummol": np.array([nummol]), "ifequa.nloc": np.array([nloc])}


# ------------------------------------------------------------------------------------------------ SOLVIT

def solvit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SUBROUTINE SOLVIT(A,N,B,IPIVOT): Gaussian elimination with partial pivoting, exactly as the Fortran
    (in place on copies; returns the solution). a: (N,N) with a[i,j] = A(i+1,j+1)."""
    a = np.array(a, dtype=np.float64, order="F", copy=True)
    b = np.array(b, dtype=np.float64, copy=True)
    n = a.shape[0]
    ipiv = np.zeros(n, np.int64)
    for i in range(n - 1):
        col = np.abs(a[i:, i])
        m = i
        # IF(ABS(A(K,I)).GT.ABS(A(M,I)))M=K, sequentially (strict >, running maximum)
        m = i + int(np.argmax(col))       # first maximum == running strict-> search
        ipiv[i] = m
        if m != i:
            t = a[i, i + 1:].copy()
            a[i, i + 1:] = a[m, i + 1:]
            a[m, i + 1:] = t
        pivot = 1.0 / a[m, i]
        a[m, i] = a[i, i]
        a[i, i] = pivot
        a[i + 1:, i] = a[i + 1:, i] * pivot
        for j in range(i + 1, n):
            c = a[i, j]
            if c == 0.0:
                continue
            a[i + 1:, j] = a[i + 1:, j] - a[i + 1:, i] * c
    a[n - 1, n - 1] = 1.0 / a[n - 1, n - 1]
    for i in range(n - 1):
        m = ipiv[i]
        if m != i:
            b[m], b[i] = b[i], b[m]
        c = b[i]
        b[i + 1:] = b[i + 1:] - a[i + 1:, i] * c
    j1 = n - 1
    for _ in range(n - 1):
        j = j1
        j1 -= 1
        b[j] = b[j] * a[j, j]
        c = b[j]
        b[:j1 + 1] = b[:j1 + 1] - a[:j1 + 1, j] * c
    b[0] = b[0] * a[0, 0]
    return b


# ------------------------------------------------------------------------------------------------ NMOLEC

@dataclass
class MolTables:
    nequa: int
    nummol: int
    idequa: np.ndarray       # (NEQUA,) element ids (1-based eq index -> id), idequa[0]=0
    code: np.ndarray         # (NUMMOL,)
    equil: np.ndarray        # (6, NUMMOL)
    ncomp: np.ndarray        # (NUMMOL,)
    comps: list              # per molecule: list of K (1-based equation index, NEQUA+1 = 1/XNE)
    comp_pad: np.ndarray     # (NUMMOL, maxc) 0-based K-1, -1 padding
    is_inv: np.ndarray       # (NUMMOL, maxc) component is NEQUA1 (divide by XN(NEQUA))
    events: tuple            # assembled Jacobian scatter events (see _events)

    @classmethod
    def from_state(cls, s) -> "MolTables":
        """Tables of /IFEQUA/ (cached on the arrays' contents: READMOL runs once per model)."""
        key = tuple(np.asarray(s[k].cpu().numpy() if isinstance(s[k], torch.Tensor) else s[k]).tobytes()
                    for k in ("ifequa.nequa", "ifequa.nummol", "ifequa.locj", "ifequa.kcomps", "ifequa.equil",
                              "ifequa.code", "ifequa.idequa"))
        t = _MOLTABLES.get(key)
        if t is None:
            if len(_MOLTABLES) > 16:
                _MOLTABLES.clear()
            t = _MOLTABLES[key] = cls._build(s)
        return t

    @classmethod
    def _build(cls, s) -> "MolTables":
        g = lambda k: np.asarray(s[k].cpu().numpy() if isinstance(s[k], torch.Tensor) else s[k]).reshape(-1)
        nequa, nummol = int(g("ifequa.nequa")[0]), int(g("ifequa.nummol")[0])
        locj, kcomps = g("ifequa.locj").astype(np.int64), g("ifequa.kcomps").astype(np.int64)
        equil = np.asarray(s["ifequa.equil"].cpu().numpy() if isinstance(s["ifequa.equil"], torch.Tensor) else s["ifequa.equil"])
        comps = [list(kcomps[locj[j] - 1:locj[j + 1] - 1]) for j in range(nummol)]
        maxc = max(len(c) for c in comps)
        pad = -np.ones((nummol, maxc), np.int64)
        inv = np.zeros((nummol, maxc), bool)
        for j, c in enumerate(comps):
            for i, k in enumerate(c):
                inv[j, i] = k == nequa + 1
                pad[j, i] = (nequa if k == nequa + 1 else k) - 1
        t = cls(nequa, nummol, g("ifequa.idequa").astype(np.int64)[:nequa], g("ifequa.code")[:nummol],
                equil[:, :nummol], np.array([len(c) for c in comps]), comps, pad, inv, ())
        t.events = _events(t)
        return t


_MOLTABLES: dict = {}


def _events(t: MolTables):
    """Static list of the Jacobian/residual updates of one Newton iteration in Fortran order.

    Each event e adds  sign_e * TERM[jmol_e] (/ XN[div_e] if div_e >= 0)  to the flat target tgt_e of the
    concatenated vector [EQ(1..N), DEQ(1..N*N) column-major]. Returns arrays (tgt, jmol, div, sign).
    """
    n = t.nequa
    n1 = n + 1
    tgt, jm, dv, sg = [], [], [], []

    def ev(target, j, div, sign):
        tgt.append(target); jm.append(j); dv.append(div); sg.append(sign)

    def deq(m, k):          # DEQ(M,K) (1-based) -> flat index
        return n + (m - 1) + (k - 1) * n

    for j in range(t.nummol):
        comps = t.comps[j]
        if len(comps) == 1:
            continue
        ev(0, j, -1, 1.0)                                       # EQ(1)=EQ(1)+TERM
        for k in comps:
            if k >= n1:
                kk, sign = n, -1.0                              # D=-TERM/XN(NEQUA)
            else:
                kk, sign = k, 1.0                               # D=TERM/XN(K)
            ev(kk - 1, j, -1, 1.0)                              # EQ(K)=EQ(K)+TERM
            ev(deq(1, kk), j, kk - 1, sign)                     # DEQ(K1)=DEQ(K1)+D, K1=NEQUA*(K-1)+1
            for m in comps:
                mm = n if m == n1 else m
                ev(deq(mm, kk), j, kk - 1, sign)                # DEQ(MK)=DEQ(MK)+D
        # correction to the charge equation for negative ions
        klast = comps[-1]
        if klast <= n and t.idequa[klast - 1] == 100:
            for k in comps:
                # D=TERM/XN(K) with K the raw component index (K=NEQUA+1 cannot occur here)
                if k == n:
                    ev(k - 1, j, -1, -1.0)                      # EQ(K)=EQ(K)-TERM-TERM
                    ev(k - 1, j, -1, -1.0)
                for m in comps:
                    if m != n:
                        continue
                    ev(deq(m, k), j, k - 1, -1.0)               # DEQ(MK)=DEQ(MK)-D-D
                    ev(deq(m, k), j, k - 1, -1.0)
    return (np.array(tgt, np.int64), np.array(jm, np.int64), np.array(dv, np.int64), np.array(sg))


def _terms(t: MolTables, equilj: np.ndarray, xn: np.ndarray) -> np.ndarray:
    """TERM(JMOL)=EQUILJ(JMOL)*prod XN(K) (/XN(NEQUA) for 1/XNE components), component order."""
    term = equilj.copy()
    for c in range(t.comp_pad.shape[1]):
        k = t.comp_pad[:, c]
        valid = k >= 0
        x = xn[np.maximum(k, 0)]
        term = np.where(valid & t.is_inv[:, c], term / x, np.where(valid, term * x, term))
    return term


def _equilj_mol(t: MolTables, s, n: int) -> np.ndarray:
    """EQUILJ for the molecules with EQUIL(1)/=0 (independent of XNE): (n_layers, NUMMOL), else NaN."""
    T = s["temp.t"][:n].cpu().numpy()
    tkev = s["temp.tkev"][:n].cpu().numpy()
    tlog = s["temp.tlog"][:n].cpu().numpy()
    out = np.full((n, t.nummol), np.nan)
    E = t.equil
    for j in range(t.nummol):
        if E[0, j] == 0.0:
            continue
        cd = t.code[j]
        ion = int((cd - float(int(cd))) * 100.0 + 0.5)
        if cd == 101.0:
            v = libm.exp(torch.from_numpy(
                E[0, j] / tkev - R4_464584 + (16.3660e-4 + (-49.3992e-8 + (111.822e-12 + (-149.567e-16 + (
                    106.206e-20 - 30.8720e-24 * T) * T) * T) * T) * T) * T - 1.5 * tlog)).numpy()
        else:
            v = libm.exp(torch.from_numpy(
                E[0, j] / tkev - E[1, j] + (E[2, j] + (-E[3, j] + (E[4, j] - E[5, j] * T) * T) * T) * T
                - 1.5 * float(t.ncomp[j] - ion - ion - 1) * tlog)).numpy()
        out[:, j] = np.where(T > 10000.0, 0.0, v)
    return out


def _ion_groups(t: MolTables):
    """Molecules with EQUIL(1)=0 and NCOMP>1 (atomic ions): {element id: [(jmol, ncomp), ...]}."""
    g = {}
    for j in range(t.nummol):
        if t.equil[0, j] == 0.0 and t.ncomp[j] > 1:
            g.setdefault(int(t.code[j]), []).append((j, int(t.ncomp[j])))
    return g


def _newton_solver(t: MolTables, xab: np.ndarray, cap):
    """The per-layer Newton iteration of NMOLEC (labels 41-110) as a function newton(xn, equilj, xntot) ->
    (iterations, chemfail) that updates xn in place: the native transcription (native/nmolec.c) when built,
    else the reference loop below (identical results)."""
    N = t.nequa
    tgt, ejm, ediv, esg = t.events
    last_el = bool(t.idequa[N - 1] >= 100)
    lib = _native.load()
    if lib is not None:
        cp = np.ascontiguousarray(t.comp_pad, dtype=np.int64)
        inv = np.ascontiguousarray(t.is_inv, dtype=np.uint8)
        tg, jm_, dv = (np.ascontiguousarray(a, dtype=np.int64) for a in (tgt, ejm, ediv))
        sg = np.ascontiguousarray(esg, dtype=np.float64)
        xab_c = np.ascontiguousarray(xab, dtype=np.float64)
        work = np.empty(N + N * N + 2 * N + t.nummol)
        iwork = np.empty(N, dtype=np.int32)
        flag = np.zeros(1, dtype=np.int32)
        ptr = lambda a: a.ctypes.data
        maxc = cp.shape[1]

        def newton(xn, equilj, xntot):
            eqj = np.ascontiguousarray(equilj, dtype=np.float64)
            assert xn.dtype == np.float64 and xn.flags.c_contiguous
            it = lib.nmolec_newton(N, t.nummol, maxc, ptr(cp), ptr(inv), len(tg), ptr(tg), ptr(jm_), ptr(dv), ptr(sg),
                                   ptr(xab_c), float(xntot), int(last_el), ptr(eqj), ptr(xn),
                                   0 if cap is None else int(cap), R4_1E4, R4_69, ptr(work), ptr(iwork), ptr(flag))
            return int(it), bool(flag[0])
        return newton
    has_div = ediv >= 0
    ediv0 = np.maximum(ediv, 0)
    nflat = N + N * N

    def newton(xn, equilj, xntot):
        eqold = np.zeros(N)
        it = 0
        chemfail = False
        while True:
            it += 1
            flat = np.zeros(nflat)
            eq = flat[:N]
            deq = flat[N:].reshape((N, N), order="F")     # view: deq[m-1, k-1] = DEQ(M,K)
            eq[0] = -xntot
            for k in range(2, N + 1):
                eq[0] = eq[0] + xn[k - 1]
                deq[0, k - 1] = 1.0
                eq[k - 1] = xn[k - 1] - xab[k - 1] * xn[0]
                deq[k - 1, k - 1] = 1.0
                deq[k - 1, 0] = -xab[k - 1]
            if last_el:
                eq[N - 1] = -xn[N - 1]
                deq[N - 1, N - 1] = -1.0
            term = _terms(t, equilj, xn)
            vals = term[ejm]
            vals = np.where(has_div, vals / xn[ediv0], vals) * esg
            np.add.at(flat, tgt, vals)
            sol = solvit(deq, eq)
            iferr = 0
            scale = 100.0
            for k in range(N):
                e = sol[k]
                ratio = abs(e / xn[k])
                if ratio > R4_1E4:
                    iferr = 1
                if eqold[k] * e < 0.0:
                    e = e * R4_69
                xneq = xn[k] - e
                xn100 = xn[k] / 100.0
                if abs(xneq) < xn100:
                    xn[k] = xn[k] / scale
                    if eqold[k] * e < 0.0:
                        scale = math.sqrt(scale)
                else:
                    xn[k] = abs(xneq)
                eqold[k] = e
            if cap is not None and it >= cap:
                chemfail = True
                break
            if iferr != 1:
                break
        return it, chemfail
    return newton


def nmolec(s, mode: int, variant: str = "atlas9", max_iter: int | None = None) -> dict:
    """SUBROUTINE NMOLEC(MODE). Returns the updated COMMON entries:
    state.xne/xnatom/rho, xnmol.xnmol, xnsave.xnsave (IFEDNS=0), edens.edens (IFEDNS=1), and temp.t/tk/tkev
    (the IFEDNS=1 branch perturbs T by *1.001, /1.001*.999, /.999 around PFSAHA calls; the round trip is
    not exact in floating point and the Fortran keeps the drifted values)."""
    dev = s["temp.t"].device
    n = s.iscalar("rhox.nrhox")
    t = MolTables.from_state(s)
    N, nummol = t.nequa, t.nummol
    cap = (20000 if variant == "atlas9" else None) if max_iter is None else max_iter
    npf = lambda k: s[k].detach().cpu().numpy().copy()
    P, TK, T = npf("state.p"), npf("temp.tk"), npf("temp.t")
    xne = npf("state.xne")
    xnatom, rho = npf("state.xnatom"), npf("state.rho")
    xabund = npf("xabund.xabund")
    wtmole = s.scalar("xabund.wtmole")
    atmass = npf("elem.atmass")
    xnsave = npf("xnsave.xnsave")
    xnmol = npf("xnmol.xnmol")
    ifedns = s.iscalar("edens.ifedns")
    potion = npf("potion.potion")
    L_all = Layers.from_state(s, n)

    xab = np.zeros(N)
    idd = 0
    for k in range(2, N + 1):
        idd = int(t.idequa[k - 1])
        if idd < 100:
            xab[k - 1] = max(xabund[idd - 1], 1e-20)
    if idd == 100:
        xab[N - 1] = 0.0
    xn = np.zeros(N)
    xntot = P[0] / TK[0]
    xn[0] = xntot / 2.0
    if T[0] < 4000.0:
        xn[0] = xntot
    x = xn[0] / 10.0
    for k in range(2, N + 1):
        xn[k - 1] = x * xab[k - 1]
    if idd == 100:
        xn[N - 1] = x
    xne[0] = x

    eq_mol = _equilj_mol(t, s, n)
    groups = _ion_groups(t)
    rows, ion_mols = [], []
    for iz, lst in groups.items():
        for jm, nc in lst:
            row = (iz, min(nc + 2, _nions(iz, variant)))
            if row not in rows:
                rows.append(row)
            ion_mols.append((jm, rows.index(row), nc))
    plan = SahaPlan(L_all, rows, potion, variant) if rows else None
    xnz = np.zeros((n, N))
    eqiters = np.zeros(n, np.int64)
    newton = _newton_solver(t, xab, cap)
    chemfail = False
    for j in range(n):
        xntot = P[j] / TK[j]
        if j > 0:
            ratio = P[j] / P[j - 1]
            xne[j] = xne[j - 1] * ratio
            xn = xn * ratio
        if ifedns == 1:
            xn = xnsave[j, :N].copy()
        equilj = eq_mol[j].copy()
        for jm in range(nummol):
            if t.equil[0, jm] == 0.0 and t.ncomp[jm] == 1:
                equilj[jm] = 1.0
        if plan is not None:
            frac = plan.fractions_layer(j, float(xne[j]))                          # (R, M)
            for jm, r, nc in ion_mols:
                equilj[jm] = frac[r, nc - 1] / frac[r, 0] * libm.ipow_var(xne[j], nc - 1)
        it, failed = newton(xn, equilj, xntot)
        chemfail = chemfail or failed
        eqiters[j] = it
        xnz[j] = xn
        xnatom[j] = xn[0]
        rho[j] = xnatom[j] * wtmole * 1.660e-24
        if t.idequa[N - 1] == 100:
            xne[j] = xn[N - 1]
        xnmol[j, :nummol] = _terms(t, equilj, xn)
    out = {}
    if ifedns == 1:
        tt, ttk, ttkev = (torch.from_numpy(a) for a in (T.copy(), TK.copy(), npf("temp.tkev")))
        edens = _edens_molecular(t, s, n, xnmol, rho, potion, variant, tt, ttk, ttkev, xne)
        out["edens.edens"] = edens
        out["temp.t"], out["temp.tk"], out["temp.tkev"] = tt, ttk, ttkev
    else:
        xnsave[:n, :N] = xnz
        out["xnsave.xnsave"] = torch.from_numpy(xnsave)
        if mode not in (2, 12):
            _partition_divide(t, s, n, xnz, xnmol, atmass, potion, variant, xne)
    out.update({"state.xne": torch.from_numpy(xne), "state.xnatom": torch.from_numpy(xnatom),
                "state.rho": torch.from_numpy(rho), "xnmol.xnmol": torch.from_numpy(xnmol)})
    out = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
    out["_nmolec.iterations"] = torch.from_numpy(eqiters)
    out["_nmolec.chemfail"] = chemfail
    return out


def _partition_divide(t, s, n, xnz, xnmol, atmass, potion, variant, xne):
    """Label 120-140: number densities -> number densities / partition functions (in place)."""
    N = t.nequa
    T = s["temp.t"][:n]
    Tn = T.cpu().numpy()
    L = Layers.from_state(s, n).replace(xne=torch.from_numpy(xne[:n].copy()).to(T.device))
    # PFSAHA(J,ID,NION,3,FRAC) = PART(NION) does not depend on NION2: one plan, rows (ID, max NION needed)
    need = {}
    for k in range(2, N + 1):
        idd = int(t.idequa[k - 1])
        if idd != 100:
            need[idd] = max(need.get(idd, 1), 1)
    for jm in range(t.nummol):
        if t.equil[0, jm] == 0.0:
            idd = int(t.code[jm])
            need[idd] = max(need.get(idd, 1), int(t.ncomp[jm]))
    rows = sorted(need.items())
    part_all = SahaPlan(L, rows, potion, variant).parts(slice(None), L)[0].cpu().numpy()   # (n, R, M)
    rix = {iz: r for r, (iz, _) in enumerate(rows)}

    def part_of(iz, nion):
        return part_all[:, rix[iz], nion - 1]
    for k in range(2, N + 1):
        idd = int(t.idequa[k - 1])
        if idd == 100:
            xnz[:, k - 1] = xnz[:, k - 1] / 2.0 / 2.4148e15 / Tn / np.sqrt(Tn)
            continue
        frac = part_of(idd, 1)
        xnz[:, k - 1] = xnz[:, k - 1] / frac / 1.8786e20 / np.sqrt(libm.ipow_const(atmass[idd - 1] * Tn, 3))
    tkev = s["temp.tkev"][:n].cpu().numpy()
    for jm in range(t.nummol):
        nc = int(t.ncomp[jm])
        if t.equil[0, jm] != 0.0:
            v = libm.exp(torch.from_numpy(t.equil[0, jm] / tkev)).numpy()
            amass = 0.0
            for k in t.comps[jm]:
                if k == N + 1:
                    v = v / xnz[:, N - 1]
                    continue
                idd = int(t.idequa[k - 1])
                if idd < 100:
                    amass = amass + atmass[idd - 1]
                v = v * xnz[:, k - 1]
            xnmol[:n, jm] = v * 1.8786e20 * np.sqrt(libm.ipow_const(amass * Tn, 3))
        else:
            idd = int(t.code[jm])
            xnmol[:n, jm] = xnmol[:n, jm] / part_of(idd, nc)


def _edens_molecular(*args):
    with np.errstate(all="ignore"):                # IEEE inf/nan as in the Fortran (and torch), no warnings
        return _edens_molecular_impl(*args)


def _edens_molecular_impl(t, s, n, xnmol, rho, potion, variant, T, TK, TKEV, xne):
    """Label 160-181 (IFEDNS=1): internal energy per gram EDENS, with the T perturbations kept in T/TK/TKEV
    (full-length tensors, modified in place).

    The PFSAHA calls (one per perturbed temperature) depend only on the perturbation chain of T/TK/TKEV,
    which is fixed by the molecule structure and the masks XNMOL>0 -- not on any PFSAHA result. So the chain is
    replayed first (recording the perturbed temperatures of every call), the partition functions of all calls
    of one element are then evaluated in one layer-batched SahaPlan (elementwise, identical per layer), and
    finally EDENS is accumulated in the Fortran's molecule order. The chain and the accumulation are [n]-sized
    IEEE operations done in numpy (cheaper per operation than torch at this size)."""
    N = t.nequa
    Pn = s["state.p"][:n].detach().cpu().numpy()
    edens_full = s["edens.edens"].clone()
    xm = np.array(xnmol[:n], dtype=np.float64)
    L0 = Layers.from_state(s, n).replace(xne=torch.from_numpy(xne[:n].copy()))
    Tn, TKn, TKEVn = (a[:n].detach().cpu().numpy().copy() for a in (T, TK, TKEV))
    E = t.equil
    xntot = Pn / TKn
    edens = 1.5 * xntot * TKn                         # before the chain perturbs TK
    # --- pass 1: the T perturbation chain and the PFSAHA requests (element, NION, MODE, T state)
    rec_t, rec_tk, rec_tkev = [], [], []
    calls = []                                        # (idd, nion)
    plan_mol = []                                     # per molecule: (kind, data)

    def perturb(mask, step):
        _perturb_inplace(Tn, TKn, TKEVn, mask, step)

    def request(idd, nion, mask, step):
        perturb(mask, step)
        rec_t.append(Tn.copy()); rec_tk.append(TKn.copy()); rec_tkev.append(TKEVn.copy())
        calls.append((idd, nion))
        return len(calls) - 1

    for jm in range(t.nummol):
        nc = int(t.ncomp[jm])
        pos = xm[:, jm] > 0.0
        if E[0, jm] != 0.0:
            tplus, tminus = Tn * R4_1001, Tn * R4_999          # T as it is at this point of the chain
            reqs = []
            skip_update = False
            if t.code[jm] != 101.0:
                for k in t.comps[jm]:
                    if k == N:
                        continue
                    if k > N:
                        skip_update = True        # GO TO 180: positive molecular ions get no EDENS term
                        break
                    idd = int(t.idequa[k - 1])
                    cu = request(idd, 1, pos, "up")
                    cd = request(idd, 1, pos, "down")
                    perturb(pos, "back")
                    reqs.append((cu, cd))
            plan_mol.append(("mol", jm, pos, tplus, tminus, reqs, skip_update, TKn.copy(), TKEVn.copy()))
        else:
            idd = int(t.code[jm])
            cu = request(idd, nc, pos, "up")
            cd = request(idd, nc, pos, "down")
            perturb(pos, "back")
            plan_mol.append(("ion", jm, pos, idd, nc, cu, cd, TKn.copy(), TKEVn.copy()))
    # the drifted temperatures stay (the Fortran perturbs T/TK/TKEV in place)
    for full, v in ((T, Tn), (TK, TKn), (TKEV, TKEVn)):
        full[:n] = torch.from_numpy(v)
    # --- pass 2: the partition functions of all requests, batched per element
    res = [None] * len(calls)
    eions = {}
    by_el = {}
    for c, (idd, nion) in enumerate(calls):
        by_el.setdefault(idd, []).append(c)
    cat = lambda rec, cs: torch.from_numpy(np.concatenate([rec[c] for c in cs]))
    for idd, cs in by_el.items():
        k = len(cs)
        # PART(ION) of each ion is independent of the other ions of the row: ions 1..max NION suffice
        nion2 = max(calls[c][1] for c in cs)
        Lb = Layers(cat(rec_t, cs), cat(rec_tk, cs), cat(rec_tkev, cs), L0.hckt.repeat(k), L0.tlog.repeat(k),
                    L0.xne.repeat(k), L0.p.repeat(k), L0.bhyd.repeat(k, 1), L0.nlteon)
        plan = SahaPlan(Lb, [(idd, nion2)], potion, variant)
        part = plan.parts(slice(None), Lb)[0][:, 0, :].reshape(k, n, -1).cpu().numpy()
        for i, c in enumerate(cs):
            res[c] = part[i, :, calls[c][1] - 1]
        acc, ev = 0.0, [0.0]                                # PFSAHA MODE 5: ANSWER(ION+7)=IP(ION)+ANSWER(ION+6)
        for m in range(plan.M):
            acc = float(plan.ip[0, m]) + acc
            ev.append(acc)
        eions[idd] = ev
    # --- pass 3: EDENS in molecule order
    exp = lambda a: libm.exp(torch.from_numpy(a)).numpy()
    c101 = [r4(v) for v in ("0.582145", "16.3760", "-49.4684", "112.049", "-149.953", "106.531", "30.9791")]
    pfplus = np.zeros(n)
    pfmin = np.zeros(n)
    for item in plan_mol:
        if item[0] == "mol":
            _, jm, pos, tplus, tminus, reqs, skip_update, tk_j, tkev_j = item
            pp = exp(-E[1, jm] + (E[2, jm] + (-E[3, jm] + (E[4, jm] - E[5, jm] * tplus) * tplus) * tplus) * tplus) + 1e-30
            pm = exp(-E[1, jm] + (E[2, jm] + (-E[3, jm] + (E[4, jm] - E[5, jm] * tminus) * tminus) * tminus) * tminus) + 1e-30
            if t.code[jm] == 101.0:
                tm10, tp10 = tminus / 10000.0, tplus / 10000.0
                c = c101

                def h2(x):
                    return exp(c[0] + (c[1] + (c[2] + (c[3] + (c[4] + (c[5] - c[6] * x) * x) * x) * x) * x) * x) + 1e-30
                pm, pp = h2(tm10), h2(tp10)
            pfplus = np.where(pos, pp, pfplus)
            pfmin = np.where(pos, pm, pfmin)
            for cu, cd in reqs:
                pfplus = np.where(pos, pfplus * res[cu], pfplus)
                pfmin = np.where(pos, pfmin * res[cd], pfmin)
            if skip_update:
                continue
            upd = edens + xm[:, jm] * tk_j * (np.float64(-E[0, jm]) / tkev_j + (pfplus - pfmin) / (pfplus + pfmin) * 2.0 * 500.0)
            edens = np.where(pos, upd, edens)
        else:
            _, jm, pos, idd, nc, cu, cd, tk_j, tkev_j = item
            a, b = res[cu], res[cd]
            a = np.maximum(a, b)
            upd = edens + xm[:, jm] * tk_j * (np.float64(eions[idd][nc - 1]) / tkev_j + (a - b) / (a + b) * 2.0 * 500.0)
            edens = np.where(pos, upd, edens)
    edens_full[:n] = torch.from_numpy(edens / np.asarray(rho[:n], dtype=np.float64))
    return edens_full


def _perturb_inplace(T, TK, TKEV, mask, step):
    """Apply one of the Fortran's in-place temperature perturbations to the masked layers (numpy arrays).
    step 'up': X=X*1.001; 'down': X=X/1.001*.999; 'back': X=X/.999."""
    for a in (T, TK, TKEV):
        if step == "up":
            v = a * R4_1001
        elif step == "down":
            v = a / R4_1001 * R4_999
        else:
            v = a / R4_999
        np.copyto(a, np.where(mask, v, a))


# ------------------------------------------------------------------------------------------------- MOLEC

def molec(s, codout: float, mode: int, variant: str = "atlas9") -> torch.Tensor:
    """SUBROUTINE MOLEC(CODOUT,MODE,NUMBER) for IFPRES=1 / IFPOP!=2-independent path (label 200 on):
    returns NUMBER as (NRHOX, nn) (columns 1..nn written)."""
    n = s.iscalar("rhox.nrhox")
    t = MolTables.from_state(s)
    xnmol = s["xnmol.xnmol"]
    code = t.code
    if codout >= 100.0:
        for jm in range(t.nummol):
            if code[jm] == codout:
                return xnmol[:n, jm:jm + 1].clone()
        raise RuntimeError(f"BETTER LUCK NEXT TIME {codout}")
    c = codout
    nn = 1
    if mode == 11:
        nn = int((c - float(int(c))) * 100.0 + 1.5)
    out = torch.zeros((n, nn), dtype=torch.float64, device=xnmol.device)
    r4_01 = r4(".01")
    for i in range(1, nn + 1):
        ion = nn - i + 1
        hit = None
        for jm in range(t.nummol):
            if code[jm] + r4(".001") > c and code[jm] - r4(".001") < c:
                hit = jm
                break
        if hit is not None:
            out[:, ion - 1] = xnmol[:n, hit]
        else:
            idd = int(codout)
            if any(int(code[jm]) == idd for jm in range(t.nummol)):
                out[:, ion - 1] = 0.0
            else:
                # label 400: element not in the equations -> PFSAHA
                ion = int((codout - float(idd)) * 100.0 + 1.5)
                nn2 = 1 if mode == 1 else ion
                L = Layers.from_state(s, n)
                r = pfsaha(L, idd, ion, mode, s["potion.potion"], variant)
                r = r if r.dim() == 2 else r[:, None]
                res = torch.zeros((n, max(nn, r.shape[1])), dtype=torch.float64, device=xnmol.device)
                res[:, :r.shape[1]] = r
                res[:, :nn2] = r[:, :nn2] * s["state.xnatom"][:n, None] * float(s["xabund.xabund"][idd - 1])
                return res
        c = c - r4_01
    return out
