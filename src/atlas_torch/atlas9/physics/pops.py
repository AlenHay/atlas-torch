"""POPS, NELECT and the main program's population block (atlas9mem.for / kappa9.for).

POPS(CODE,MODE,NUMBER): with molecules on (IFMOL=1) the first call after a temperature change (ITEMP !=
ITEMP1, IFPRES=1) runs NMOLEC, later calls only look the species up (MOLEC); with molecules off NELECT
iterates the electron density and PFSAHA gives the populations. ITEMP1 (POPS' SAVEd local) is explicit
State: "save.pops.itemp1".

NELECT, like NMOLEC, starts layer J from the converged XNE(J-1) and is therefore a sequential layer loop;
the per-layer work (the PFSAHA calls for its 10 electron donors) is one vectorised SahaPlan evaluation.
"""
from __future__ import annotations

import numpy as np
import torch

from . import libm
from ._data import r4
from .molec import R4_1001, R4_999, molec, nmolec
from .pfsaha import Layers, SahaPlan, _nions, pfsaha

NELEMZ = (1, 2, 6, 11, 12, 13, 14, 19, 20, 26)
NIONZ = (2, 3, 2, 2, 2, 2, 2, 2, 2, 2)


def _itemp1(s) -> int:
    v = s.get("save.pops.itemp1")
    return 0 if v is None else int(v.reshape(-1)[0])


def nelect(s, variant: str = "atlas9") -> dict:
    """SUBROUTINE NELECT: electron density by fixed-point iteration over the 10 donors (IFMOL=0)."""
    n = s.iscalar("rhox.nrhox")
    dev = s["temp.t"].device
    npf = lambda k: s[k].detach().cpu().numpy().copy()
    P, TK = npf("state.p"), npf("temp.tk")
    xne, xnatom, rho = npf("state.xne"), npf("state.xnatom"), npf("state.rho")
    xabund = npf("xabund.xabund")
    wtmole = s.scalar("xabund.wtmole")
    ifedns = s.iscalar("edens.ifedns")
    it, numits = s.iscalar("iter.iter"), s.iscalar("iter.numits")
    potion = s["potion.potion"]
    if it < numits or ifedns != 1:
        xne[0] = P[0] / TK[0] / 2.0
    L_all = Layers.from_state(s, n)
    rows = [(iz, min(nz + 2, _nions(iz, variant))) for iz, nz in zip(NELEMZ, NIONZ)]
    plan = SahaPlan(L_all, rows, potion, variant)
    nz = len(NELEMZ)
    T = s["temp.t"].clone()
    TKt = s["temp.tk"].clone()
    TKEV = s["temp.tkev"].clone()
    edens = s["edens.edens"].clone()
    r4_5e4, r4_1e5, r4_05 = r4(".0005"), r4(".00001"), r4(".05")
    x = np.zeros(nz)
    for j in range(n):
        if j > 0 and ifedns == 0:
            xne[j] = xne[j - 1] * P[j] / P[j - 1]
        xntot = P[j] / TK[j]
        xnatom[j] = xntot - xne[j]
        mask = np.ones(nz, bool)
        Lj = L_all.index(slice(j, j + 1))
        for _ in range(200):
            f = plan.fractions(slice(j, j + 1), Lj.replace(xne=torch.tensor([xne[j]], dtype=torch.float64, device=dev)))[0][0]
            f = f.cpu().numpy()
            xnenew = 0.0
            for i in range(nz):
                if not mask[i]:
                    continue
                elec = 0.0
                for ion in range(2, rows[i][1] + 1):
                    elec = elec + f[i, ion - 1] * float(ion - 1)
                x[i] = elec * xnatom[j] * xabund[NELEMZ[i] - 1]
                xnenew = xnenew + x[i]
            xnenew = (xnenew + xne[j]) / 2.0
            error = abs((xne[j] - xnenew) / xnenew)
            xne[j] = xnenew
            xnatom[j] = xntot - xne[j]
            if error < r4_5e4:
                break
            if j == 0:
                continue
            x1 = r4_1e5 * xne[j]
            if error < r4_05:
                x1 = x1 * 10.0
            mask &= ~(x < x1)
        else:
            raise RuntimeError("XNE DOES NOT CONVERGE")
        rho[j] = xnatom[j] * wtmole * 1.660e-24
        if ifedns == 0:
            continue
        # internal energy (IFEDNS=1), with the Fortran's in-place T perturbations of layer J
        edj = 1.5 * xntot * float(TKt[j])
        for i in range(nz):
            iz, nion = NELEMZ[i], NIONZ[i]
            for a in (T, TKt, TKEV):
                a[j] = a[j] * R4_1001
            Lp = Layers.from_state(s.updated({"temp.t": T, "temp.tk": TKt, "temp.tkev": TKEV,
                                              "state.xne": torch.from_numpy(xne)}), n).index(slice(j, j + 1))
            pfp, eion = pfsaha(Lp, iz, nion, 5, potion, variant)
            for a in (T, TKt, TKEV):
                a[j] = a[j] / R4_1001 * R4_999
            Lm = Layers.from_state(s.updated({"temp.t": T, "temp.tk": TKt, "temp.tkev": TKEV,
                                              "state.xne": torch.from_numpy(xne)}), n).index(slice(j, j + 1))
            pfm, _ = pfsaha(Lm, iz, nion, 5, potion, variant)
            for a in (T, TKt, TKEV):
                a[j] = a[j] / R4_999
            L0 = Layers.from_state(s.updated({"temp.t": T, "temp.tk": TKt, "temp.tkev": TKEV,
                                              "state.xne": torch.from_numpy(xne)}), n).index(slice(j, j + 1))
            xnumb = pfsaha(L0, iz, nion, 12, potion, variant)[0]
            for ion in range(1, nion + 1):
                xn_ = float(xnumb[ion - 1]) * xnatom[j] * xabund[iz - 1]
                a_, b_ = float(pfp[0, ion - 1]), float(pfm[0, ion - 1])
                edj = edj + xn_ * float(TKt[j]) * (float(eion[0, ion - 1]) / float(TKEV[j]) + (a_ - b_) / (a_ + b_) * 2.0 * 500.0)
        edens[j] = edj / rho[j]
    out = {"state.xne": torch.from_numpy(xne).to(dev), "state.xnatom": torch.from_numpy(xnatom).to(dev),
           "state.rho": torch.from_numpy(rho).to(dev)}
    if ifedns == 1:
        out.update({"edens.edens": edens, "temp.t": T, "temp.tk": TKt, "temp.tkev": TKEV})
    return out


def pops(s, code: float, mode: int, variant: str = "atlas9", number=None):
    """SUBROUTINE POPS(CODE,MODE,NUMBER). Returns (number (NRHOX, ncols) or None, updates dict).

    `updates` holds what NELECT/NMOLEC wrote (when they ran) and save.pops.itemp1; NUMBER is the array the
    caller stores (columns 1..ncols of e.g. XNFPC)."""
    upd = {}
    ifmol, ifpres = s.iscalar("if.ifmol"), s.iscalar("if.ifpres")
    itemp = s.iscalar("temp.itemp")
    st = s
    if ifpres == 1 and itemp != _itemp1(s):
        upd = nmolec(s, mode, variant) if ifmol == 1 else nelect(s, variant)
        upd = {k: v for k, v in upd.items() if not k.startswith("_")}
        st = s.updated(upd)
    upd["save.pops.itemp1"] = torch.tensor([itemp])
    if code == 0.0:
        return None, upd
    n = st.iscalar("rhox.nrhox")
    if ifmol == 1:
        return molec(st, code, mode, variant), upd
    if code >= 100.0:
        raise RuntimeError("MOLECULES OFF")
    iz = int(code)
    nion = int((code - float(iz)) * 100.0 + 1.5)
    r = pfsaha(Layers.from_state(st, n), iz, nion, mode, st["potion.potion"], variant)
    r = r if r.dim() == 2 else r[:, None]
    nnnn = nion if mode >= 10 else 1
    out = r.clone()
    out[:, :nnnn] = r[:, :nnnn] * st["state.xnatom"][:n, None] * float(st["xabund.xabund"][iz - 1])
    return out, upd


def _store(s, key, number, ncols=None):
    a = s[key].clone()
    n = number.shape[0]
    if a.dim() == 1:
        a[:n] = number[:, 0]
    else:
        k = min(a.shape[1], number.shape[1]) if ncols is None else ncols
        a[:n, :k] = number[:, :k]
    return a


_POPS11 = [(1.01, "ions.xnfph"), (2.02, "ions.xnfphe"), (6.03, "xnfp.xnfpc"), (7.04, "xnfp.xnfpn"),
           (8.05, "xnfp.xnfpo"), (10.05, "xnfp.xnfpne"), (12.01, "xnfp.xnfpmg"), (13.00, "xnfp.xnfpal"),
           (14.01, "xnfp.xnfpsi"), (20.01, "xnfp.xnfpca"), (26.00, "xnfp.xnfpfe")]
_XNF = [(6.05, "xnf.xnfc"), (7.05, "xnf.xnfn"), (8.05, "xnf.xnfo"), (10.05, "xnf.xnfne"), (12.05, "xnf.xnfmg"),
        (14.05, "xnf.xnfsi"), (16.05, "xnf.xnfs"), (26.04, "xnf.xnffe")]


def electrons(s, variant: str = "atlas9") -> dict:
    """POPS(0.D0,1,XNE): NELECT / NMOLEC for the current T, P (only if ITEMP changed)."""
    return pops(s, 0.0, 1, variant)[1]


def populations(s, variant: str = "atlas9", include_electrons: bool = True) -> dict:
    """The ATLAS9 main program's population block for one iteration (see package docstring)."""
    upd = {}
    st = s
    if include_electrons and st.iscalar("if.ifpres") == 1:
        upd["edens.ifedns"] = torch.tensor([0])
        st = st.updated(upd)
        u = electrons(st, variant)
        upd.update(u)
        st = st.updated(u)

    def call(code, mode, key):
        nonlocal st
        num, u = pops(st, code, mode, variant)
        upd.update(u)
        st = st.updated(u)
        a = _store(st, key, num)
        upd[key] = a
        st = st.updated({key: a})

    for code, key in _POPS11:
        call(code, 11, key)
    ifmol = st.iscalar("if.ifmol")
    if ifmol == 0:
        call(1.01, 12, "ions.xnfh")
        call(2.02, 12, "ions.xnfhe")
        for code, key in _XNF:
            call(code, 12, key)
    if ifmol == 1:
        call(106.00, 11, "xnfp.xnfpch")
        call(108.00, 11, "xnfp.xnfpoh")
        for code, key in _XNF:
            call(code, 11, key)
        # loop 444: number densities = (n/PF) * PF
        n = st.iscalar("rhox.nrhox")
        L = Layers.from_state(st, n)
        pot = st["potion.potion"]
        xnfh, xnfhe = st["ions.xnfh"].clone(), st["ions.xnfhe"].clone()
        xnfph, xnfphe = st["ions.xnfph"], st["ions.xnfphe"]
        part = pfsaha(L, 1, 1, 3, pot, variant)
        xnfh[:n, 0] = xnfph[:n, 0] * part
        xnfh[:n, 1] = xnfph[:n, 1]
        part = pfsaha(L, 2, 2, 13, pot, variant)
        xnfhe[:n, 0] = xnfphe[:n, 0] * part[:, 0]
        xnfhe[:n, 1] = xnfphe[:n, 1] * part[:, 1]
        xnfhe[:n, 2] = xnfphe[:n, 2]
        upd["ions.xnfh"], upd["ions.xnfhe"] = xnfh, xnfhe
        for iz, key in ((6, "xnf.xnfc"), (7, "xnf.xnfn"), (8, "xnf.xnfo"), (10, "xnf.xnfne"), (12, "xnf.xnfmg"),
                        (14, "xnf.xnfsi"), (16, "xnf.xnfs"), (26, "xnf.xnffe")):
            nion = 5 if iz == 26 else 6
            part = pfsaha(L, iz, nion, 13, pot, variant)
            a = st[key].clone()
            a[:n, :nion] = a[:n, :nion] * part
            upd[key] = a
    return upd
