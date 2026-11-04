"""PROGRAM XNFDF (DFSYNTHE): populations, Doppler widths and continuum opacities on a (T, P) grid.

    from atlas_torch.dfsynthe.xnfdf import run
    res = run(stdin_text, continua_text=..., molecules_text=..., out10="xnfpdf.dat", out11="xnfpdfmax.dat",
              ftz=True)    # ftz: stock flush-to-zero environment (default); False = IEEE (ifx -O0 / oracle)

Every READIN/BEGIN block of the input is one "model" (a column of NRHOX pressures at one temperature in
the DFSYNTHE grid). All models (layers) are computed together: shape (B models, J layers) for all layer
quantities, (B, J, ...) for populations; NMOLEC loops over J (each layer starts from the previous
layer's solution, as the Fortran does). The writers reproduce ifx unformatted sequential records
byte for byte.

Fortran quirks reproduced:
- default REAL literals (e.g. CUTOFF=.001, 1.0000001, POPS C=C-.01) are single precision;
- IDMOL(60) DATA are REAL*4 literals widened to REAL*8, so POPS(IDMOL,1) finds no molecule with an
  exact match for the ionised molecules (106.01, 107.01, ...): their XNFPEL(6,.) is 0;
- POPS(106.,11)/POPS(108.,11) (XNFPCH/XNFPOH) assign nothing (MODE 11 has no molecule branch);
- XNFPH2/XNFPCO are only set where T<=9000 (stale elsewhere) and only reach XNFP(.,10,.), which is
  never written;
- the REAL*4 state record always has 25 entries (stale beyond NRHOX, kept across models).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .xnf.fdata import T as tab
from .xnf.fmath import F, fsqrt, powi, rdiv
from .xnf.freeform import FortranExit
from .xnf.kapp import KappInput, kapp, prepare
from .xnf.nmolec import Pops, nelect, nmolec
from .xnf.pfsaha import Layers
from .xnf.readin import Model, Readin, read_edges
from .xnf.xnfio import FortranWriter

DT = torch.float64
KW = 99


def _ext() -> Path:
    import os
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def default_inputs():
    d = _ext() / "BasicATLAS" / "data" / "dfsynthe_files"
    return dict(continua_text=(d / "continua.dat").read_text(), molecules_text=(d / "molecules.dat").read_text())


def read_models(stdin: str, molecules_text: str) -> list[Model]:
    r = Readin(stdin, molecules_text)
    out = []
    while True:
        try:
            m = r.read(1)
        except FortranExit as e:
            if "END" in str(e) or "end of file" in str(e):
                break
            raise
        out.append(m)
    return out


def _stack(ms, key, nj):
    return torch.from_numpy(np.stack([getattr(m, key)[:nj] for m in ms]))


def pow10(x: torch.Tensor) -> torch.Tensor:
    """10.**X (REAL*4 base, REAL*8 exponent) = libm pow(10, x)."""
    if x.device.type == "cpu":
        f = np.frompyfunc(lambda v: math.pow(10.0, v), 1, 1)
        return torch.from_numpy(f(x.numpy()).astype(np.float64))
    return torch.pow(10.0, x)


@dataclass
class Batch:
    models: list
    nj: int
    L: Layers = None
    pops: dict = field(default_factory=dict)
    res: dict = field(default_factory=dict)


def populations(ms: list[Model], nj: int):
    """READIN BEGIN state -> NMOLEC/NELECT + the main program's POPS calls (batched over models)."""
    m0 = ms[0]
    L = Layers(_stack(ms, "t", nj), _stack(ms, "tkev", nj), _stack(ms, "tk", nj), _stack(ms, "hckt", nj),
               _stack(ms, "tlog", nj), _stack(ms, "p", nj), _stack(ms, "xne", nj),
               {k: torch.from_numpy(np.stack([m.b[k][:nj] for m in ms])) for k in m0.b}, m0.nlteon)
    xabund = torch.from_numpy(np.stack([m.xabund for m in ms]))
    wtmole = torch.tensor([m.wtmole for m in ms], dtype=DT)
    if m0.ifmol == 1:
        r = nmolec(L, m0.mol, xabund, m0.atmass, wtmole)
        L = L.with_(xne=r["xne"], nlteon=0)
        pops = Pops(L, xabund, r["xnatom"], 1, m0.mol, r["xnmol"], r["xnfpmol"])
    else:
        r = nelect(L, xabund, wtmole)
        L = L.with_(xne=r["xne"])
        pops = Pops(L, xabund, r["xnatom"], 0)
    return L, pops, r, xabund


def _cols(d: dict, n: int, like: torch.Tensor):
    z = torch.zeros_like(like)
    return torch.stack([d.get(i, z) for i in range(1, n + 1)], -1)


def run(stdin: str, continua_text: str | None = None, molecules_text: str | None = None,
        out10: str | Path | None = None, out11: str | Path | None = None, progress=None,
        ftz: bool = True) -> dict:
    """PROGRAM XNFDF.  ftz=True (default): stock floating-point environment (flush-to-zero + denormals-are-zero,
    see ftz.py; byte-identical to the ifx -O0 -ftz build); ftz=False: IEEE gradual underflow (byte-identical to
    the ifx -O0 build / oracle)."""
    from .ftz import scope
    with scope(ftz):
        return _run(stdin, continua_text, molecules_text, out10, out11, progress)


def _run(stdin, continua_text, molecules_text, out10, out11, progress) -> dict:
    if continua_text is None or molecules_text is None:
        d = default_inputs()
        continua_text = continua_text or d["continua_text"]
        molecules_text = molecules_text or d["molecules_text"]
    frq, wl, cm, freqset = read_edges(continua_text)
    nin = len(frq)
    numnu = len(freqset)
    if numnu > 1131 or nin > 377:
        raise ValueError("too many edges for the xnfdf arrays")
    models = read_models(stdin, molecules_text)
    idmol = tab("xnfdf", "idmol").astype(np.float64)
    momass = tab("xnfdf", "momass").astype(np.float64)
    keys = [(m.nrhox, tuple(m.ifop), m.ifmol, m.nlteon) for m in models]
    for k in keys:
        if k[0] > 25:
            raise NotImplementedError("NRHOX > 25 overflows the REAL*4 arrays of PROGRAM XNFDF")
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    results = [None] * len(models)
    for k, idx in groups.items():
        out = _run_group([models[i] for i in idx], freqset, idmol, momass, progress)
        for j, i in enumerate(idx):
            results[i] = {kk: v[j] for kk, v in out.items()}
    if out10 is not None:
        _write(out10, out11, models, results, frq, wl, cm, freqset, idmol, momass)
    return dict(models=models, results=results, edges=(frq, wl, cm), freqset=freqset)


def group_state(ms: list[Model]) -> dict:
    """Everything PROGRAM XNFDF computes before the frequency loop, for models sharing NRHOX/IFOP/IFMOL:
    populations (NMOLEC/NELECT + the main program's POPS calls), XNFH2 and the KAPP inputs/caches."""
    nj = ms[0].nrhox
    ifop = list(ms[0].ifop)
    ifop[13:17] = [0, 0, 0, 0]                 # IFOP(14..17)=0 in the main program
    L, pops, r, xabund = populations(ms, nj)
    xnatom, rho = r["xnatom"], r["rho"]
    like = L.t
    P = {}
    for name, code, mode, n in [("xnfh", 1.00, 12, 1), ("xnfhe", 2.01, 12, 2), ("xnfph", 1.01, 11, 2),
                                ("xnfphe", 2.02, 11, 3), ("xnfpb", 5.00, 11, 1), ("xnfpc", 6.01, 11, 2),
                                ("xnfpo", 8.00, 11, 1), ("xnfpna", 11.00, 11, 1), ("xnfpmg", 12.01, 11, 2),
                                ("xnfpal", 13.01, 11, 2), ("xnfpsi", 14.01, 11, 2), ("xnfpk", 19.00, 11, 1),
                                ("xnfpca", 20.01, 11, 2), ("xnfpfe", 26.00, 11, 1)]:
        P[name] = _cols(pops(code, mode), n, like)
    t, tkev, tlog = L.t, L.tkev, L.tlog
    hkt = _stack(ms, "hkt", nj)
    eq = torch.exp(rdiv(F(4.478), tkev) - 4.64584e1 + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (
        1.06206e-18 - 3.08720e-23 * t) * t) * t) * t) * t) * t - 1.5 * tlog)
    cold = ~(t > 9000.)
    xnfh2 = torch.where(cold, powi(P["xnfh"][..., 0], 2) * eq, torch.zeros_like(t))
    # opacity inputs (LUKEOP / HOTOP populations: POPS with MODE 1 / 11)
    lk = {nm: _cols(pops(code, 1), 1, like)[..., 0] for nm, code in
          [("lk_n", 7.0), ("lk_o", 8.0), ("lk_mg", 12.01), ("lk_si", 14.01), ("lk_ca", 20.01)]}
    hot = torch.cat([_cols(pops(6.03, 11), 4, like), _cols(pops(7.04, 11), 5, like),
                     _cols(pops(8.05, 11), 6, like), _cols(pops(10.05, 11), 6, like)], -1)
    ki = KappInput(t, tkev, hkt, L.hckt, tlog, L.xne, rho, L.b,
                   dict(xnfph=P["xnfph"], xnfhe=P["xnfhe"], xnfphe=P["xnfphe"], xnfpc=P["xnfpc"],
                        xnfpmg=P["xnfpmg"], xnfpal=P["xnfpal"], xnfpsi=P["xnfpsi"], xnfpfe=P["xnfpfe"],
                        hot=hot, **lk), tuple(ifop))
    c = prepare(ki)
    return dict(nj=nj, L=L, pops=pops, r=r, xabund=xabund, P=P, lk=lk, hot=hot, hkt=hkt, xnfh2=xnfh2, ki=ki, c=c)


def _run_group(ms, freqset, idmol, momass, progress):
    g = group_state(ms)
    nj, nb = g["nj"], len(ms)
    L, pops, r, P, hkt, xnfh2, ki, c = g["L"], g["pops"], g["r"], g["P"], g["hkt"], g["xnfh2"], g["ki"], g["c"]
    xnatom, rho = r["xnatom"], r["rho"]
    t, tkev, tlog = L.t, L.tkev, L.tlog
    numnu = len(freqset)
    contin = torch.zeros(nb, nj, 1131, dtype=DT)
    for nu in range(numnu):
        acont, sigmac = kapp(float(freqset[nu]), ki, c)
        contin[:, :, nu] = torch.log10(acont + sigmac)
        if progress and nu % 100 == 0:
            progress(nu, numnu)
    # XNFP(J,ION,NELEM) for ION=1..10
    xnfp = torch.zeros(nb, nj, 10, 99, dtype=DT)

    def put(nelem, d, ion0=1):
        for ion, v in d.items():
            xnfp[:, :, ion0 + ion - 2, nelem - 1] = v
    codes = {1: 1.01, 2: 2.02, 3: 3.03, 4: 4.03, 5: 5.03, 6: 6.05, 7: 7.05, 8: 8.05, 9: 9.05, 10: 10.05,
             11: 11.05, 12: 12.05, 13: 13.05, 14: 14.05, 15: 15.05, 16: 16.05, 17: 17.04, 18: 18.04, 19: 19.04,
             20: 20.09, 21: 21.09, 22: 22.09, 23: 23.09, 24: 24.09, 25: 25.09, 26: 26.09, 27: 27.09, 28: 28.09}
    for nelem in range(1, 100):
        code = codes[nelem] if nelem <= 28 else float(np.float32(nelem)) + .02
        put(nelem, pops(code, 11))
    # XNFP(J,10,40)=XNFPH2, XNFP(J,10,46)=XNFPCO: ION 10 is never written to the files (not reproduced)
    if ms[0].ifmol == 1:
        for nelem in range(40, 100):
            d = pops(float(idmol[nelem - 40]), 1)
            if 1 in d:
                xnfp[:, :, 5, nelem - 1] = d[1]
    for nelem in range(20, 29):
        xnfp[:, :, 4, 30 + nelem - 1] = xnfp[:, :, 6, nelem - 1]
        xnfp[:, :, 4, 40 + nelem - 1] = xnfp[:, :, 7, nelem - 1]
        xnfp[:, :, 4, 50 + nelem - 1] = xnfp[:, :, 8, nelem - 1]
        xnfp[:, :, 4, 60 + nelem - 1] = xnfp[:, :, 9, nelem - 1]
    xnfpel = xnfp[:, :, :6, :]                                   # (B,J,6,99)
    # DOPPLE(ION,NELEM)
    atmass = torch.from_numpy(ms[0].atmass.astype(np.float64))
    tk = _stack(ms, "tk", nj)
    vt = _stack(ms, "vturb", nj)
    d1 = fsqrt(2. * tk[..., None] / atmass / 1.660e-24 + powi(vt, 2)[..., None]) / 2.99792458e10   # (B,J,99)
    dop = d1[:, :, None, :].expand(nb, nj, 6, 99).clone()
    for nelem in range(20, 29):
        for off in (30, 40, 50, 60):
            dop[:, :, 4, off + nelem - 1] = d1[:, :, nelem - 1]
    mom = torch.from_numpy(momass)
    dop[:, :, 5, 39:] = fsqrt(2. * tk[..., None] / mom / 1.660e-24) / 2.99792458e10
    # XNFDOPMAX(NU3,ION,NELEM): max over layers of XNFPEL/DOPPLE/FREQSET(NU+1)/RHO/CONTMIN/CUTOFF (REAL*4)
    cutoff = F(.001)
    nu3 = numnu // 3
    cmin = torch.minimum(torch.minimum(contin[:, :, 0:numnu:3], contin[:, :, 1:numnu:3]), contin[:, :, 2:numnu:3])
    cmin = pow10(cmin)                                            # (B,J,nu3)
    fmid = torch.from_numpy(np.ascontiguousarray(freqset[1:numnu:3]))
    acc = torch.zeros(nb, nu3, 6, 99, dtype=torch.float32)
    ratio = xnfpel / dop                                          # (B,J,6,99)
    for j in range(nj):                                           # same division order as the Fortran
        v = ratio[:, j, None, :, :] / fmid[None, :, None, None]
        v.div_(rho[:, j, None, None, None])
        v.div_(cmin[:, j, :, None, None])
        v.div_(cutoff)
        torch.maximum(acc, v.to(torch.float32), out=acc)
    xmax = torch.zeros(nb, 377, 6, 99, dtype=torch.float32)
    xmax[:, :nu3] = acc
    return dict(t=t, tkev=tkev, tk=tk, hkt=hkt, tlog=tlog, hckt=L.hckt, p=L.p, xne=L.xne, xnatom=xnatom, rho=rho,
                rhox=_stack(ms, "rhox", nj), vturb=vt, xnfh=P["xnfh"][..., 0], xnfhe=P["xnfhe"], xnfh2=xnfh2,
                xnfpel=xnfpel, dopple=dop, continall=contin, xnfdopmax=xmax,
                **({k: r[k] for k in ("xnmol", "xnfpmol")} if "xnmol" in r else {}))   # NMOLEC printout


def _title_bytes(title):
    return b"".join(ch.encode("latin-1") + b" " * 7 for ch in title)


def _write(out10, out11, models, results, frq, wl, cm, freqset, idmol, momass):
    nin = len(frq)
    edges = np.empty(3 * nin)
    edges[0::3], edges[1::3], edges[2::3] = frq, wl, cm
    st_names = ("t", "tkev", "tk", "hkt", "tlog", "hckt", "p", "xne", "xnatom", "rho", "rhox", "vturb", "xnfh")
    state4 = {k: np.zeros(25, np.float32) for k in st_names}
    state4["xnfhe"] = np.zeros((25, 2), np.float32)
    state4["xnfh2"] = np.zeros(25, np.float32)
    with FortranWriter(out10) as f10, FortranWriter(out11) as f11:
        for m, r in zip(models, results):
            nt = m.nrhox
            head = (np.int32(nt).tobytes(), np.float64(m.t[0]).tobytes(), np.float64(0.).tobytes(),
                    _title_bytes(m.title))
            f10.record(*head)
            f11.record(*head)
            f10.record(np.int32(nin).tobytes(), edges, idmol, momass)
            f11.record(np.int32(nin).tobytes(), edges)
            f10.record(np.int32(len(freqset)).tobytes(), freqset)
            f11.record(np.int32(len(freqset)).tobytes(), freqset)
            for k in st_names + ("xnfh2",):
                state4[k][:nt] = r[k].numpy().astype(np.float32)
            state4["xnfhe"][:nt] = r["xnfhe"].numpy().astype(np.float32)
            f10.record(*[state4[k] for k in st_names], state4["xnfhe"], state4["xnfh2"])
            xp = r["xnfpel"].numpy().astype(np.float32)          # (J,6,99)
            dp = r["dopple"].numpy().astype(np.float32)
            ca = r["continall"].numpy().astype(np.float32)        # (J,1131)
            for j in range(nt):
                f10.record(np.asfortranarray(xp[j]), np.asfortranarray(dp[j]), ca[j])
            f11.record(np.asfortranarray(r["xnfdopmax"].numpy()))
