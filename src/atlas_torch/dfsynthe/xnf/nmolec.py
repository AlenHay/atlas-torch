"""NMOLEC / SOLVIT / NELECT / POPS of xnfdf.for.

Batch layout: per-layer tensors have shape (B, J) (B models of NRHOX=J layers). NMOLEC and NELECT
start layer J from the converged solution of layer J-1 scaled by P(J)/P(J-1) (Fortran logic), so they
loop over J and vectorise over models; the Newton iteration runs with per-model convergence masks and
exactly the Fortran's update/damping rules (no iteration cap in the Fortran; a generous safety cap here).
Everything after the layer loop (partition-function normalisation of XNZ, XNFPMOL, POPS) is batched
over (B, J).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .fmath import F, fsqrt, powi, rdiv
from .pfsaha import Layers, pfsaha
from .readin import Molecules

DT = torch.float64
MAXIT = 5000

_CPF = ((1, "bhyd"), (6, "bc1"), (8, "bo1"), (12, "bmg1"), (13, "bal1"), (14, "bsi1"), (20, "bca1"))


def solvit(a: torch.Tensor, b: torch.Tensor):
    """SUBROUTINE SOLVIT(A,N,B,IPIVOT): Gauss-Jordan with full pivoting, batched (A: (B,N,N) [row,col])."""
    a = a.clone()
    b = b.clone()
    nb, n, _ = a.shape
    ar = torch.arange(nb)
    ipiv = torch.zeros(nb, n, dtype=torch.bool)
    for _ in range(n):
        cand = torch.where(ipiv[:, :, None] | ipiv[:, None, :], torch.full_like(a, -1.0), a.abs())
        flat = cand.reshape(nb, n * n)
        idx = torch.argmax(flat, dim=1)          # first maximum in (row, column) order = the Fortran scan
        amax = flat[ar, idx]
        if bool((amax <= 0).any()):
            raise FloatingPointError("SOLVIT: singular matrix (no positive pivot)")
        irow = idx // n
        icol = idx % n
        ipiv[ar, icol] = True
        rr, rc = a[ar, irow].clone(), a[ar, icol].clone()
        a[ar, icol] = rr
        a[ar, irow] = rc
        br, bc = b[ar, irow].clone(), b[ar, icol].clone()
        b[ar, icol] = br
        b[ar, irow] = bc
        pivot = a[ar, icol, icol].clone()
        a[ar, icol, icol] = 1.
        a[ar, icol] = a[ar, icol] / pivot[:, None]
        b[ar, icol] = b[ar, icol] / pivot
        tcol = a[ar, :, icol].clone()                       # T for every row L1
        prow = a[ar, icol].clone()                          # row ICOLUM
        notp = torch.ones(nb, n, dtype=torch.bool)
        notp[ar, icol] = False
        a2 = a.clone()
        a2[ar, :, icol] = 0.
        a2 = a2 - prow[:, None, :] * tcol[:, :, None]
        a = torch.where(notp[:, :, None], a2, a)
        b = torch.where(notp, b - b[ar, icol][:, None] * tcol, b)
    return b


@dataclass
class MolTab:
    """Python-side view of /XNMOL/ + /IFEQUA/ for the loops."""
    code: list
    equil: list          # 7-tuples
    comps: list          # per molecule: list of equation numbers (1-based; nequa+1 = 1/XNE)
    nequa: int
    idequa: list         # idequa[k] for k=0..nequa+1 (0 for unset)

    @classmethod
    def of(cls, m: Molecules) -> "MolTab":
        comps = [list(m.kcomps[m.locj[j] - 1:m.locj[j + 1] - 1]) for j in range(m.nummol)]
        ide = [int(v) for v in m.idequa[:m.nequa + 2]]
        return cls([float(c) for c in m.code], [tuple(float(x) for x in m.equil[:, j]) for j in range(m.nummol)],
                   comps, m.nequa, ide)


def _h2_equil(t, tkev, tlog):
    return torch.exp(rdiv(F(4.478), tkev) - 4.64584e1 + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (
        1.06206e-18 - 3.08720e-23 * t) * t) * t) * t) * t) * t - 1.5 * tlog)


def nmolec(L: Layers, mol: Molecules, xabund: torch.Tensor, atmass: np.ndarray, wtmole: torch.Tensor,
           maxit: int = MAXIT):
    """Returns dict with xnmol, xnfpmol (B,J,NUMMOL), xne, xnatom, rho (B,J), xnz (B,J,NEQUA)."""
    mt = MolTab.of(mol)
    nb, nj = L.t.shape
    ne = mt.nequa
    ne1 = ne + 1
    nmol = len(mt.code)
    ide = mt.idequa
    xab = [None] * (ne + 1)
    idl = None
    for k in range(2, ne + 1):
        idl = ide[k]
        if idl < 100:
            xab[k] = torch.clamp(xabund[:, idl - 1], min=1e-20)
    if idl == 100:
        xab[ne] = torch.zeros(nb, dtype=DT)
    p, tk = L.p, L.tk
    xntot = p[:, 0] / tk[:, 0]
    xn = [None] * (ne + 1)
    xn[1] = xntot / 2.
    x = xn[1] / 10.
    for k in range(2, ne + 1):
        xn[k] = x * xab[k]
    if idl == 100:
        xn[ne] = x
    xne = L.xne.clone()
    xne[:, 0] = x
    xnatom = torch.zeros_like(p)
    rho = torch.zeros_like(p)
    xnz = torch.zeros(nb, nj, ne + 1, dtype=DT)
    xnmol = torch.zeros(nb, nj, nmol, dtype=DT)
    equilj_all = torch.zeros(nb, nj, nmol, dtype=DT)
    for j in range(nj):
        Lj = Layers(L.t[:, j], L.tkev[:, j], L.tk[:, j], L.hckt[:, j], L.tlog[:, j], p[:, j], xne[:, j],
                    {k: v[:, j] for k, v in L.b.items()}, 0)
        pfs = {iz: pfsaha(Lj.with_(nlteon=-1), iz, 1, 3)[:, 0] for iz, _ in _CPF}
        bpf = {iz: pfsaha(Lj.with_(nlteon=0), iz, 1, 3)[:, 0] for iz, _ in _CPF}
        cpf = {iz: pfs[iz] / bpf[iz] * Lj.b[bn][:, 0] for iz, bn in _CPF}
        xnt = p[:, j] / tk[:, j]
        if j > 0:
            ratio = p[:, j] / p[:, j - 1]
            xne[:, j] = xne[:, j - 1] * ratio
            for k in range(1, ne + 1):
                xn[k] = xn[k] * ratio
        Lj = Lj.with_(xne=xne[:, j], nlteon=0)
        tj, tkevj, tlogj = Lj.t, Lj.tkev, Lj.tlog
        equilj = []
        for jm in range(nmol):
            code, eq, comps = mt.code[jm], mt.equil[jm], mt.comps[jm]
            ncomp = len(comps)
            if eq[0] != 0.:
                ion = int((code - float(int(code))) * 100. + F(.5))
                if code == 101.:
                    e = _h2_equil(tj, tkevj, tlogj)
                else:
                    e = torch.exp(rdiv(eq[0], tkevj) - eq[1] + (eq[2] + (-eq[3] + (eq[4] + (-eq[5] + eq[6] * tj) * tj) * tj)
                                                           * tj) * tj - 1.5 * float(ncomp - ion - ion - 1) * tlogj)
                for k in comps:
                    idk = ide[k] if k <= ne else 0
                    for iz, _ in _CPF:
                        if idk == iz:
                            e = e * cpf[iz]
                e = torch.where(tj > 10000., torch.zeros_like(e), e)
            elif ncomp == 1:
                e = torch.ones_like(tj)
            else:
                frac = pfsaha(Lj, int(code), ncomp, 12)
                e = frac[:, ncomp - 1] / frac[:, 0] * powi(Lj.xne, ncomp - 1)
            equilj.append(e)
        equilj_all[:, j] = torch.stack(equilj, -1)
        # Newton iteration ------------------------------------------------------------------------
        eqold = [torch.zeros(nb, dtype=DT) for _ in range(ne + 1)]
        active = torch.ones(nb, dtype=torch.bool)
        it = 0
        while True:
            it += 1
            if it > maxit:
                raise FloatingPointError(f"NMOLEC: no convergence at layer {j + 1} after {maxit} iterations")
            zero = torch.zeros(nb, dtype=DT)
            eqv = [None] * (ne + 1)
            deq = {}

            def add(r, c, v):
                deq[(r, c)] = deq[(r, c)] + v if (r, c) in deq else zero + v
            eqv[1] = -xnt
            for k in range(2, ne + 1):
                eqv[1] = eqv[1] + xn[k]
                deq[(1, k)] = zero + 1.
                eqv[k] = xn[k] - xab[k] * xn[1]
                deq[(k, k)] = zero + 1.
                deq[(k, 1)] = -xab[k] + zero
            if ide[ne] >= 100:
                eqv[ne] = -xn[ne]
                deq[(ne, ne)] = zero - 1.
            for jm in range(nmol):
                comps = mt.comps[jm]
                if len(comps) == 1:
                    continue
                term = equilj[jm]
                for k in comps:
                    term = term / xn[ne] if k == ne1 else term * xn[k]
                eqv[1] = eqv[1] + term
                for k in comps:
                    if k < ne1:
                        d = term / xn[k]
                    else:
                        k = ne
                        d = -term / xn[k]
                    eqv[k] = eqv[k] + term
                    add(1, k, d)
                    for m in comps:
                        add(ne if m == ne1 else m, k, d)
                kl = comps[-1]
                if (ide[kl] if kl <= ne1 else 0) != 100:
                    continue
                for k in comps:
                    d = term / xn[k]
                    if k == ne:
                        eqv[k] = eqv[k] - term - term
                    for m in comps:
                        if m != ne:
                            continue
                        add(m, k, -d)
                        add(m, k, -d)
            A = torch.zeros(nb, ne, ne, dtype=DT)
            for (r, c), v in deq.items():
                A[:, r - 1, c - 1] = v
            rhs = torch.stack(eqv[1:], -1)
            sol = solvit(A, rhs)
            iferr = torch.zeros(nb, dtype=torch.bool)
            scale = torch.full((nb,), 100., dtype=DT)
            for k in range(1, ne + 1):
                ek = sol[:, k - 1]
                rat = torch.abs(ek / xn[k])
                iferr = iferr | (rat > F(.001))
                flip = eqold[k] * ek < 0.
                ek = torch.where(flip, ek * F(.69), ek)
                xneq = xn[k] - ek
                xn100 = xn[k] / 100.
                small = torch.abs(xneq) < xn100
                newx = torch.where(small, xn[k] / scale, torch.abs(xneq))
                scale = torch.where(active & small & (eqold[k] * ek < 0.), fsqrt(scale), scale)
                xn[k] = torch.where(active, newx, xn[k])
                eqold[k] = torch.where(active, ek, eqold[k])
            active = active & iferr
            if not bool(active.any()):
                break
        for k in range(1, ne + 1):
            xnz[:, j, k] = xn[k]
        xnatom[:, j] = xn[1]
        rho[:, j] = xn[1] * wtmole * 1.660e-24
        if ide[ne] == 100:
            xne[:, j] = xn[ne]
        for jm in range(nmol):
            v = equilj[jm]
            for k in mt.comps[jm]:
                v = v / xn[ne] if k == ne1 else v * xn[k]
            xnmol[:, j, jm] = v
    # partition-function normalisation (loops 121/123) and XNFPMOL (loop 140), all layers at once
    Lf = L.with_(xne=xne, nlteon=0)
    t = L.t
    for k in range(2, ne + 1):
        idk = ide[k]
        if idk == 100:
            xnz[:, :, k] = xnz[:, :, k] / 2. / 2.4148e15 / t / fsqrt(t)
            continue
        pf = pfsaha(Lf, idk, 1, 3)[..., 0]
        for iz, bn in _CPF:
            if idk == iz:
                pf = pf / L.b[bn][..., 0]
        xnz[:, :, k] = xnz[:, :, k] / pf / 1.8786e20 / fsqrt(powi(float(atmass[idk - 1]) * t, 3))
    xnfpmol = torch.zeros_like(xnmol)
    for jm in range(nmol):
        comps = mt.comps[jm]
        eq = mt.equil[jm]
        if eq[0] != 0.:
            v = torch.exp(rdiv(eq[0], L.tkev))
            amass = 0.
            for k in comps:
                if k == ne1:
                    v = v / xnz[:, :, ne]
                    continue
                if ide[k] < 100:
                    amass = amass + float(atmass[ide[k] - 1])
                v = v * xnz[:, :, k]
            xnfpmol[:, :, jm] = v * 1.8786e20 * fsqrt(powi(amass * t, 3))
        else:
            pf = pfsaha(Lf, int(mt.code[jm]), len(comps), 3)[..., 0]
            xnfpmol[:, :, jm] = xnmol[:, :, jm] / pf
    return dict(xnmol=xnmol, xnfpmol=xnfpmol, xne=xne, xnatom=xnatom, rho=rho, xnz=xnz[:, :, 1:],
                equilj=equilj_all)


def nelect(L: Layers, xabund: torch.Tensor, wtmole: torch.Tensor):
    """SUBROUTINE NELECT (IFMOL=0 path of POPS): electron density by iteration, layer by layer."""
    nelemz = (1, 2, 6, 11, 12, 13, 14, 19, 20, 26)
    nionz = (1, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    p, tk = L.p, L.tk
    nb, nj = p.shape
    xne = L.xne.clone()
    xnatom = torch.zeros_like(p)
    rho = torch.zeros_like(p)
    xne[:, 0] = p[:, 0] / tk[:, 0] / 2.
    for j in range(nj):
        if j > 0:
            xne[:, j] = xne[:, j - 1] * p[:, j] / p[:, j - 1]
        xntot = p[:, j] / tk[:, j]
        xnat = xntot - xne[:, j]
        mask = [torch.ones(nb, dtype=torch.bool) for _ in nelemz]
        active = torch.ones(nb, dtype=torch.bool)
        xj = xne[:, j].clone()
        for it in range(200):
            Lj = Layers(L.t[:, j], L.tkev[:, j], L.tk[:, j], L.hckt[:, j], L.tlog[:, j], p[:, j], xj,
                        {k: v[:, j] for k, v in L.b.items()}, L.nlteon)
            xnenew = torch.zeros(nb, dtype=DT)
            x = []
            for i, (iz, ni) in enumerate(zip(nelemz, nionz)):
                elec = pfsaha(Lj, iz, ni, 4)[:, 0]
                xi = elec * xnat * xabund[:, iz - 1]
                x.append(xi)
                xnenew = torch.where(mask[i], xnenew + xi, xnenew)
            xnenew = (xnenew + xj) / 2.
            err = torch.abs((xj - xnenew) / xnenew)
            xj = torch.where(active, xnenew, xj)
            xnat = torch.where(active, xntot - xj, xnat)
            done = err < F(.0005)
            if j > 0:
                x1 = F(.00001) * xj
                x1 = torch.where(err < F(.05), x1 * 10., x1)
                for i in range(len(nelemz)):
                    mask[i] = torch.where(active & ~done, mask[i] & ~(x[i] < x1), mask[i])
            active = active & ~done
            if not bool(active.any()):
                break
        else:
            raise FloatingPointError("XNE DOES NOT CONVERGE")
        xne[:, j] = xj
        xnatom[:, j] = xnat
        rho[:, j] = xnat * wtmole * F(1.660e-24)
    return dict(xne=xne, xnatom=xnatom, rho=rho)


class Pops:
    """SUBROUTINE POPS(CODE,MODE,NUMBER) given the solved state. call() returns {ION: (B,J) tensor}
    for the columns NUMBER(.,ION) the Fortran writes (others left untouched by the caller)."""

    def __init__(self, L: Layers, xabund: torch.Tensor, xnatom: torch.Tensor, ifmol: int,
                 mol: Molecules | None = None, xnmol=None, xnfpmol=None):
        self.L = L
        self.xabund = xabund
        self.xnatom = xnatom
        self.ifmol = ifmol
        self.codes = [float(c) for c in mol.code] if mol is not None else []
        self.xnmol = xnmol
        self.xnfpmol = xnfpmol

    def _saha(self, iz, nion, mode, nn, number=None):
        """NUMBER(J,I)=NUMBER(J,I)*XNATOM*XABUND for I=1..NN after PFSAHA; for MODE<10 PFSAHA only writes
        column 1, the others are the caller's previous contents (`number`, zeros if None) rescaled."""
        a = pfsaha(self.L, iz, nion, mode)
        fac = self.xnatom * self.xabund[:, iz - 1][:, None]
        out = {}
        for i in range(nn):
            col = a[..., i] if i < a.shape[-1] else (
                number[..., i] if number is not None else torch.zeros_like(fac))
            out[i + 1] = col * fac
        return out

    def __call__(self, code: float, mode: int, number: torch.Tensor | None = None) -> dict:
        """number: the caller's NUMBER array (B,J,10), only needed for the MODE 2/3 PFSAHA branch."""
        if self.ifmol == 0:
            if code == 0.:
                return {}
            if code >= 1e2:
                raise RuntimeError("MOLECULES OFF")
            iz = int(code)
            nion = int((code - float(iz)) * 100. + 1.5)
            nnnn = 1 if mode < 10 else nion
            return self._saha(iz, nion, mode, nnnn, number)
        if code == 0.:
            return {}
        codes = self.codes
        if code >= 100.:
            jm = next((k for k, c in enumerate(codes) if c == code), None)
            if jm is None:
                return {1: torch.zeros_like(self.xnatom)}
            if mode == 1:
                return {1: self.xnfpmol[..., jm]}
            if mode == 2:
                return {1: self.xnmol[..., jm]}
            if mode == 3:
                return {1: self.xnmol[..., jm] / self.xnfpmol[..., jm]}
            return {}
        out = {}
        c = code
        nn = int((c - float(int(c))) * 100. + 1.5) if mode > 10 else 1
        for i in range(1, nn + 1):
            cplus = c + F(.001)
            cminus = c - F(.001)
            ion = nn - i + 1
            jm = next((k for k, cm in enumerate(codes) if cminus < cm < cplus), None)
            done = False
            if jm is not None:
                m10 = mode % 10
                if m10 == 2:
                    out[ion] = self.xnmol[..., jm]; done = True
                elif m10 == 1:
                    out[ion] = self.xnfpmol[..., jm]; done = True
                elif m10 == 3:
                    out[ion] = self.xnmol[..., jm] / self.xnfpmol[..., jm]; done = True
            if not done:
                idc = int(code)
                if any(int(cm) == idc for cm in codes):
                    out[ion] = torch.zeros_like(self.xnatom)
                else:
                    ion4 = int((code - float(idc)) * 100. + 1.5)
                    nn4 = 1 if mode == 1 else ion4
                    out.update(self._saha(idc, ion4, mode, nn4, number))
                    return out
            c = c - F(.01)
        return out
