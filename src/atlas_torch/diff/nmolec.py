"""NMOLEC (atlas7v.for, SYNTHE variant) with implicit differentiation at convergence.

Forward value: the Fortran's own iterate. `solve_exact` replays the Newton iteration of
atlas_torch.synthe.pops.nmolec operation for operation (exact libm, numpy float64), layer by layer with the
Fortran's start values (layer J starts from the converged layer J-1 scaled by P(J)/P(J-1)) and convergence test
(|dX/X| < 1E-3 for every unknown). Its result is bit-identical to the exact path (checked in the tests).

Derivatives: implicit function theorem on the converged equations, never through the iterations. Per layer J the
unknowns are X_J = XN(1..NEQUA) (total atoms, neutral atoms of the equation elements, electrons) and the
equations are NMOLEC's EQ(X; theta_J) = 0 (particle conservation per element, charge conservation), where theta_J
holds T(J), P(J) (XNTOT = P/kT), the abundances XAB(K), the equilibrium constants EQUILJ(T(J), XNE_guess(J)) and
the NLTE factors CPF(T(J), XNE_deck(J)). Two Fortran details make EQUILJ depend on more than T(J):
  * ion "molecules" (EQUIL(1)=0) get EQUILJ = F(NCOMP)/F(1) * XNE**(NCOMP-1) from PFSAHA evaluated with the
    *start* XNE(J) = XNE*(J-1) * P(J)/P(J-1) (Debye lowering only; the XNE power cancels the Saha factor);
  * CPF uses the input (deck) XNE(J) of SYNTHE's model.
So layer J's equations depend on layer J-1's converged electron density: the system is block lower bidiagonal
and its tangent solve is a forward substitution, dX_J = -(dEQ/dX_J)^-1 (dEQ/dtheta_J dtheta_J + dEQ/dg_J dg_J).

Implementation ("one Newton step at the converged point"): with X* the exact iterate (detached) and
J* = dEQ/dX at X* (analytic, detached),
    Delta_J = -solve(J*_J, EQ_J(X*_J; theta_J(params), g_J))           (a differentiable function of params)
    X_J     = X*_J + (Delta_J - Delta_J.detach())
The primal of X_J is exactly X*_J; its tangent / gradient is the IFT one. g_J is built from the *attached* X of
layer J-1, so the forward substitution happens automatically (sequential over layers, small torch ops).
Plain differentiable ops only: works in forward mode (torch.autograd.forward_ad) and reverse mode.

`polish=True` (option) additionally converges each layer's Newton iteration to machine precision before
attaching (smooth forward function; differs from the exact path by NMOLEC's loose convergence, see RESULTS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..atlas9.physics import libm
from ..synthe import pops as P
from ..synthe.pfsaha import Layers, SahaPlan, nions_of, pfsaha

DT = torch.float64


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


@dataclass
class Plan:
    """Static structure of NMOLEC's system (from the molecule table of the State)."""
    mt: P.MolTab
    ne: int
    xab_idx: np.ndarray        # (ne+1,) element index (0-based, -1 none) of XAB(K)
    ide_ne: int
    mols: np.ndarray           # molecules with > 1 component (TERM contributors)
    A: torch.Tensor            # (ne, nm) EQ(row) += sum A[row, m] * TERM(m)
    C: torch.Tensor            # (nm, ne) TERM(m) = EQUILJ(m) * prod_k XN(k)**C[m, k]
    ion_mols: list             # (jmol, IZ, NCOMP) with EQUIL(1)=0, NCOMP>1
    one_mols: list
    cpf_rows: list


def build_plan(s) -> Plan:
    mt = P.MolTab.from_state(s)
    ne, ne1 = mt.nequa, mt.nequa + 1
    xab_idx = np.full(ne + 1, -1, dtype=np.int64)
    idl = 0
    for k in range(2, ne + 1):
        idl = mt.ide(k)
        if idl < 100:
            xab_idx[k] = idl - 1
    ev = P._Events(mt)
    mols = ev.mols
    nm = len(mols)
    A = np.zeros((ne, nm))
    m0 = ev.src == 0
    np.add.at(A, (ev.tgt[m0], ev.idx[m0]), ev.sgn[m0])
    C = np.zeros((nm, ne))
    for r, j in enumerate(mols):
        for k in mt.comps[j]:
            if k == ne1:
                C[r, ne - 1] -= 1.0
            else:
                C[r, k - 1] += 1.0
    ion_mols, one_mols = [], []
    for j in range(mt.nummol):
        if mt.equil[0, j] == 0.0:
            if len(mt.comps[j]) == 1:
                one_mols.append(j)
            else:
                ion_mols.append((j, int(mt.code[j]), len(mt.comps[j])))
    return Plan(mt, ne, xab_idx, mt.ide(ne), mols, torch.as_tensor(A, dtype=DT), torch.as_tensor(C, dtype=DT),
                ion_mols, one_mols, [(iz, min(1 + 2, nions_of(iz))) for iz, _ in P._CPF])


# ---------------------------------------------------------------------------------------------- pieces in torch

def layers_of(s, n) -> Layers:
    return Layers.from_state(s, n)


def cpf_factors(plan: Plan, s, L: Layers, n: int) -> dict:
    """CPF(IZ) = PFSAHA(NLTEON=-1)/PFSAHA(NLTEON=0) * B(J,1), with the XNE NMOLEC sees there: X (the start
    value) in layer 1, the input XNE(J) in the other layers."""
    xne = L.xne.clone()
    xntot0 = L.p[0] / L.tk[0]
    xne = torch.cat([(xntot0 / 2.0 / 10.0).reshape(1), xne[1:]])
    Lc = L.with_(xne=xne)
    pf_m1 = SahaPlan(Lc.with_(nlteon=-1), plan.cpf_rows).parts(slice(None), Lc.with_(nlteon=-1))[0][:, :, 0]
    pf_0 = SahaPlan(Lc.with_(nlteon=0), plan.cpf_rows).parts(slice(None), Lc.with_(nlteon=0))[0][:, :, 0]
    return {iz: pf_m1[:, r] / pf_0[:, r] * P._bcol(s, key, n) for r, (iz, key) in enumerate(P._CPF)}


def equil_neutral(plan: Plan, L: Layers, cpf: dict) -> torch.Tensor:
    """EQUILJ (n, nummol) of every molecule with EQUIL(1) /= 0 (T only), 1 for the one-component entries, 0 for
    the ion molecules (filled per layer by `equil_ions`). Same expressions as synthe.pops.nmolec."""
    mt = plan.mt
    t, tkev, tlog = L.t, L.tkev, L.tlog
    cols = []
    hot = t > 10000.0
    for j in range(mt.nummol):
        eq = mt.equil[:, j]
        ncomp = len(mt.comps[j])
        if eq[0] != 0.0:
            code = float(mt.code[j])
            ion = int((code - float(int(code))) * 100.0 + 0.5)
            if code == 101.0:
                e = P._h2_equil(t, tkev, tlog)
            else:
                e = libm.exp(libm.rdiv(float(eq[0]), tkev) - float(eq[1]) + (float(eq[2]) + (-float(eq[3]) + (
                    float(eq[4]) + (-float(eq[5]) + float(eq[6]) * t) * t) * t) * t) * t
                    - 1.5 * float(ncomp - ion - ion - 1) * tlog)
            for k in mt.comps[j]:
                idk = mt.ide(k)
                if idk in cpf:
                    e = e * cpf[idk]
            cols.append(torch.where(hot, torch.zeros_like(e), e))
        elif ncomp == 1:
            cols.append(torch.ones_like(t))
        else:
            cols.append(torch.zeros_like(t))
    return torch.stack(cols, 1)


class IonPlan:
    def __init__(self, plan: Plan, L0: Layers):
        self.plan = plan
        irows = []
        for _, iz, nc in plan.ion_mols:
            n2 = min(nc + 2, nions_of(iz))
            irows.append((iz, n2))
        self.iplan = SahaPlan(L0, irows) if irows else None
        self.ir = np.arange(len(plan.ion_mols))
        self.inc = np.array([nc for _, _, nc in plan.ion_mols], dtype=np.int64)
        self.ij = np.array([j for j, _, _ in plan.ion_mols], dtype=np.int64)

    def values(self, L0: Layers, idx: slice, xne_guess: torch.Tensor) -> torch.Tensor:
        """EQUILJ of the ion molecules (len(idx), R) for layers idx with start electron densities xne_guess."""
        Lj = L0.index(idx).with_(xne=xne_guess)
        frac = self.iplan.fractions(idx, Lj)[0]                    # (m, R, M)
        fr = frac[:, self.ir, self.inc - 1] / frac[:, self.ir, 0]
        pw = torch.stack([libm.ipow_var(xne_guess, int(nc) - 1) for nc in self.inc], 1)
        return fr * pw


def residual(plan: Plan, xn: torch.Tensor, xab: torch.Tensor, xntot: torch.Tensor, equil: torch.Tensor):
    """EQ(X) of NMOLEC for a batch: xn (m, ne) = XN(1..NE), xab (ne,) XAB(1..NE) (XAB(1) unused), xntot (m,),
    equil (m, nummol). Returns (m, ne)."""
    ne = plan.ne
    m = xn.shape[0]
    cols = [-xntot + xn[:, 1:].sum(1)]
    for k in range(2, ne + 1):
        cols.append(xn[:, k - 1] - xab[k - 1] * xn[:, 0])
    if plan.ide_ne >= 100:
        cols[ne - 1] = -xn[:, ne - 1]
    base = torch.stack(cols, 1)
    if plan.mols.size == 0:
        return base
    eqm = equil[:, torch.as_tensor(plan.mols)]
    term = eqm * _prodpow(xn, plan.C)
    return base + term @ plan.A.T.to(xn.device)


def _prodpow(xn: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """prod_k xn[:, k]**C[m, k] for small integer C (repeated multiplication/division)."""
    out = torch.ones(xn.shape[0], C.shape[0], dtype=xn.dtype, device=xn.device)
    Cn = C.cpu().numpy()
    for k in range(C.shape[1]):
        col = Cn[:, k]
        nz = np.nonzero(col)[0]
        for r in nz:
            p = int(col[r])
            v = xn[:, k] ** abs(p)
            out[:, r] = out[:, r] * v if p > 0 else out[:, r] / v
    return out


def jacobian(plan: Plan, xn: torch.Tensor, xab: torch.Tensor, equil: torch.Tensor) -> torch.Tensor:
    """dEQ/dXN (m, ne, ne) at xn (detached analytic Jacobian)."""
    ne = plan.ne
    m = xn.shape[0]
    J = torch.zeros(m, ne, ne, dtype=DT, device=xn.device)
    J[:, 0, 1:] = 1.0
    for k in range(2, ne + 1):
        J[:, k - 1, k - 1] = 1.0
        J[:, k - 1, 0] = -xab[k - 1]
    if plan.ide_ne >= 100:
        J[:, ne - 1, :] = 0.0
        J[:, ne - 1, ne - 1] = -1.0
    if plan.mols.size:
        eqm = equil[:, torch.as_tensor(plan.mols)]
        term = eqm * _prodpow(xn, plan.C)                                   # (m, nm)
        dterm = term[:, :, None] * plan.C[None].to(xn.device) / xn[:, None, :]   # (m, nm, ne)
        J = J + torch.einsum("rm,bmk->brk", plan.A.to(xn.device), dterm)
    return J


# ------------------------------------------------------------------------------------------- exact solve

@dataclass
class Exact:
    xn: np.ndarray          # (n, ne) converged XN(1..NE) of every layer (the Fortran's iterate)
    guess: np.ndarray       # (n,) XNE(J) used as start (and in the ion EQUILJ)
    iters: np.ndarray       # (n,) Newton iterations


def solve_exact(s, plan: Plan, maxit: int = P.MAXIT) -> Exact:
    """The Newton iteration of synthe.pops.nmolec (exact libm; must be called outside native_math())."""
    mt = plan.mt
    n = P._i(s, "rhox.nrhox")
    ne = plan.ne
    L = Layers.from_state(s, n)
    xabund = _np(s["xabund.xabund"]).astype(np.float64)
    xab = np.zeros(ne + 1)
    idl = 0
    for k in range(2, ne + 1):
        idl = mt.ide(k)
        if idl < 100:
            xab[k] = max(xabund[idl - 1], 1e-20)
    if idl == 100:
        xab[ne] = 0.0
    pn, tkn = _np(L.p).astype(np.float64), _np(L.tk).astype(np.float64)
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
    dev = L.t.device
    cpf = cpf_factors(plan, s, L, n)
    equil_np = _np(equil_neutral(plan, L, cpf)).astype(np.float64).copy()
    L0 = L.with_(nlteon=0)
    ip = IonPlan(plan, L0) if plan.ion_mols else None
    ev = P._Events(mt)
    out = np.zeros((n, ne))
    guess = np.zeros(n)
    iters = np.zeros(n, dtype=np.int64)
    for j in range(n):
        xntot = pn[j] / tkn[j]
        if j > 0:
            ratio = pn[j] / pn[j - 1]
            xne[j] = xne[j - 1] * ratio
            xn[1:] = xn[1:] * ratio
        guess[j] = xne[j]
        equilj = equil_np[j].copy()
        if ip is not None:
            Lj = L0.index(slice(j, j + 1)).with_(xne=torch.tensor([xne[j]], dtype=DT, device=dev))
            frac = _np(ip.iplan.fractions(slice(j, j + 1), Lj)[0][0]).astype(np.float64)
            fr = frac[ip.ir, ip.inc - 1] / frac[ip.ir, 0]
            equilj[ip.ij] = [fr[r] * libm.ipow_var(xne[j], int(ip.inc[r]) - 1) for r in range(len(ip.ir))]
        eqold = np.zeros(ne + 1)
        it = 0
        while True:
            it += 1
            if it > maxit:
                raise FloatingPointError(f"NMOLEC: no convergence in layer {j + 1}")
            v = np.zeros(ne + ne * ne)
            e1 = -xntot
            for k in range(2, ne + 1):
                e1 = e1 + xn[k]
                v[ne + (k - 1) * ne] = 1.0
                v[k - 1] = xn[k] - xab[k] * xn[1]
                v[ne + (k - 1) + (k - 1) * ne] = 1.0
                v[ne + (k - 1)] = -xab[k]
            v[0] = e1
            if mt.ide(ne) >= 100:
                v[ne - 1] = -xn[ne]
                v[ne + ne * ne - 1] = -1.0
            term, d, d2 = ev.terms(equilj, xn)
            ev.apply(v, term, d, d2)
            deq = v[ne:].reshape(ne, ne, order="F")
            _, sol, _ = P.solvit(deq, v[:ne])
            iferr = False
            scale = 100.0
            for k in range(1, ne + 1):
                ek = sol[k - 1]
                if abs(ek / xn[k]) > P.R4_001:
                    iferr = True
                if eqold[k] * ek < 0.0:
                    ek = ek * P.R4_69
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
        out[j] = xn[1:]
        iters[j] = it
        if mt.ide(ne) == 100:
            xne[j] = xn[ne]
    return Exact(out, guess, iters)


# ------------------------------------------------------------------------------------------- attached solve

def attach(plan: Plan, s, ex: Exact, polish: bool = False, polish_steps: int = 30) -> dict:
    """Converged NMOLEC unknowns as functions of the State's (possibly dual / requires_grad) inputs.

    Call inside native_math(). Returns dict with 'xn' (n, ne) attached (primal = ex.xn, or the polished root),
    'xne', 'xnatom' (n,), and 'equil' (n, nummol) attached, 'guess' (n,) attached."""
    mt = plan.mt
    n = ex.xn.shape[0]
    ne = plan.ne
    L = Layers.from_state(s, n)
    dev = L.t.device
    xabund = s["xabund.xabund"].to(DT)
    xab_cols = [torch.zeros((), dtype=DT, device=dev)]
    for k in range(2, ne + 1):
        i = int(plan.xab_idx[k])
        xab_cols.append(torch.clamp(xabund[i], min=1e-20) if i >= 0 else torch.zeros((), dtype=DT, device=dev))
    xab = torch.stack(xab_cols)                                          # XAB(1..NE)
    xntot = L.p / L.tk
    cpf = cpf_factors(plan, s, L, n)
    equil = equil_neutral(plan, L, cpf)                                  # (n, nummol), ions 0 for now
    L0 = L.with_(nlteon=0)
    ip = IonPlan(plan, L0) if plan.ion_mols else None
    xs = torch.as_tensor(ex.xn, dtype=DT, device=dev)
    if polish:
        xs = _polish(plan, s, ex, xab.detach(), xntot.detach(), equil.detach(), ip, L0, polish_steps)
    rows, eq_rows, guesses = [], [], []
    prev = None
    xab_d = xab.detach()
    for j in range(n):
        if j == 0:
            g = xntot[0] / 2.0 / 10.0
        else:
            base = prev[ne - 1] if plan.ide_ne == 100 else guesses[-1]
            g = base * (L.p[j] / L.p[j - 1])
        guesses.append(g)
        eqj = equil[j]
        if ip is not None:
            vals = ip.values(L0, slice(j, j + 1), g.reshape(1))[0]
            eqj = eqj.index_put((torch.as_tensor(ip.ij, device=dev),), vals)
        eq_rows.append(eqj)
        xj = xs[j:j + 1]
        F = residual(plan, xj, xab, xntot[j:j + 1], eqj.unsqueeze(0))       # (1, ne) attached
        Jm = jacobian(plan, xj, xab_d, eqj.detach().unsqueeze(0))           # (1, ne, ne) detached
        delta = -torch.linalg.solve(Jm, F.unsqueeze(-1)).squeeze(-1)[0]
        xa = xj[0] + (delta - delta.detach())
        rows.append(xa)
        prev = xa
    xn_att = torch.stack(rows)
    out = {"xn": xn_att, "equil": torch.stack(eq_rows), "guess": torch.stack(guesses), "xab": xab}
    out["xnatom"] = xn_att[:, 0]
    out["xne"] = xn_att[:, ne - 1] if plan.ide_ne == 100 else L.xne
    return out


def _polish(plan, s, ex, xab, xntot, equil, ip, L0, steps, sweeps: int = 8):
    """Undamped Newton to machine precision from the exact iterate (detached), batched over layers, with the
    ion EQUILJ evaluated at self-consistent start electron densities g_J = XNE(J-1)*P(J)/P(J-1) of the
    polished solution (block Gauss-Seidel sweeps over the weak g dependence until g is stationary)."""
    n, ne = ex.xn.shape
    dev = xab.device
    x = torch.as_tensor(ex.xn, dtype=DT, device=dev)
    g = torch.as_tensor(ex.guess, dtype=DT, device=dev)
    p = L0.p.detach()
    for _ in range(sweeps):
        eq = equil.clone()
        if ip is not None:
            with torch.no_grad():
                eq[:, torch.as_tensor(ip.ij, device=dev)] = ip.values(L0, slice(None), g).detach()
        for _ in range(steps):
            F = residual(plan, x, xab, xntot, eq)
            Jm = jacobian(plan, x, xab, eq)
            dx = torch.linalg.solve(Jm, F.unsqueeze(-1)).squeeze(-1)
            x = x - dx
            if float((dx.abs() / x.abs()).max()) < 1e-15:
                break
        if plan.ide_ne != 100:
            break
        gn = torch.cat([g[:1], x[:-1, ne - 1] * (p[1:] / p[:-1])])
        dg = float(((gn - g).abs() / g.abs()).max())
        g = gn
        if dg == 0.0:
            break
    return x.detach()


# ------------------------------------------------------------------------------------------- post-processing

def finish(plan: Plan, s, att: dict) -> dict:
    """Loops 121-140 of NMOLEC: XNZ partition-function scaling and XNMOL / XNFPMOL, plus XNE, XNATOM, RHO
    (State-shaped (99, ...) tensors as the exact path writes them)."""
    mt = plan.mt
    n = att["xn"].shape[0]
    ne, ne1 = plan.ne, plan.ne + 1
    L = Layers.from_state(s, n)
    xn = att["xn"]
    equil = att["equil"]
    xne = att["xne"]
    atmass = _np(s["elem.atmass"]).astype(np.float64)
    wtmole = s["xabund.wtmole"].reshape(-1)[0].to(DT)
    td = L.t
    Lf = L.with_(nlteon=0, xne=xne)
    bkey = dict(P._CPF)
    xnz = [None] * (ne + 1)
    for k in range(2, ne + 1):
        idk = mt.ide(k)
        if idk == 100:
            xnz[k] = xn[:, k - 1] / 2.0 / 2.4148e15 / td / libm.sqrt(td)
            continue
        pf = pfsaha(Lf, idk, 1, 3)
        if idk in bkey:
            pf = pf / P._bcol(s, bkey[idk], n)
        xnz[k] = xn[:, k - 1] / pf / 1.8786e20 / libm.sqrt(libm.ipow_const(float(atmass[idk - 1]) * td, 3))
    xnmol_cols, xnfpmol_cols = [], []
    for jm in range(mt.nummol):
        vm = equil[:, jm]
        for k in mt.comps[jm]:
            vm = vm / xn[:, ne - 1] if k == ne1 else vm * xn[:, k - 1]
        xnmol_cols.append(vm)
    xnmol = torch.stack(xnmol_cols, 1)
    for jm in range(mt.nummol):
        eq = mt.equil[:, jm]
        comps = mt.comps[jm]
        if eq[0] != 0.0:
            vm = libm.exp(libm.rdiv(float(eq[0]), L.tkev))
            amass = 0.0
            for k in comps:
                if k == ne1:
                    vm = vm / xnz[ne]
                    continue
                idk = mt.ide(k)
                if idk < 100:
                    amass = amass + float(atmass[idk - 1])
                vm = vm * xnz[k]
            xnfpmol_cols.append(vm * 1.8786e20 * libm.sqrt(libm.ipow_const(amass * td, 3)))
        else:
            pf = pfsaha(Lf, int(mt.code[jm]), len(comps), 3)
            xnfpmol_cols.append(xnmol[:, jm] / pf)
    xnfpmol = torch.stack(xnfpmol_cols, 1)

    def full(key, rows):
        base = s[key].detach().clone().to(DT)
        return torch.cat([rows, base[n:]], 0) if rows.dim() == 1 else \
            torch.cat([torch.cat([rows, base[:n, rows.shape[1]:]], 1), base[n:]], 0)
    xnatom = att["xnatom"]
    rho = xnatom * wtmole * 1.660e-24
    out = {"xnmol.xnmol": full("xnmol.xnmol", xnmol), "xnmol.xnfpmol": full("xnmol.xnfpmol", xnfpmol),
           "state.xne": full("state.xne", xne), "state.xnatom": full("state.xnatom", xnatom),
           "state.rho": full("state.rho", rho)}
    if "state.taustd" in s:
        out["state.taustd"] = out["state.xnatom"]
    return out
