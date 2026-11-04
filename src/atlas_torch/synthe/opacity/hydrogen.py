"""Hydrogen continuous opacities of atlas7v.for: HOP (+XKARSAS, COULFF), H2PLOP, HMINOP (+LINTER), H2RAOP.

(HRAYOP lives in atlas_torch.synthe.hrayop.)  Each routine has
  * a batched kernel `<name>_k(ctx, ...)` -> [NF, n] tensors (used by KAPP over many frequencies), and
  * the Fortran-call API `<name>(s: State) -> dict` (one frequency, COMMON/SAVE entries it writes).
SAVEd caches reproduced: HMINOP (ITEMP1, XHMIN, THETA), H2RAOP (ITEMP1, XNH2). HMINOP's ISTART/FFLOG
cache is a pure function of DATA (computed once at import).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ...interp import map1
from ...state import State
from ._core import (Ctx, acc, chain, exp, f32, itemp_tensor, log, log10f, powi, put_rows, rdiv, save_get,  # noqa: F401
                    save_itemp1, sqrt, table)

# =========================================================================================== XKARSAS
_XK = None


def _xk_tables():
    global _XK
    if _XK is None:
        fn = table("xkarsas", "freqn").astype(np.float32)        # (29,15) REAL*4
        xn = table("xkarsas", "xn").astype(np.float32)           # (29,15)
        xl = table("xkarsas", "xl").astype(np.float32)           # (29,6,6)
        ek = table("xkarsas", "ekarsas").astype(np.float32)      # (29,)
        # One trailing 0 per table: the search falls through to I=30 when FREQLG == FREQN(29,N) exactly, and
        # FREQN(30,N)/XN(30,N)/XL(30,L+1,N) are then read through Fortran storage order; past the last column
        # that is the word after the array, which is zero alignment padding in every build (stock gfortran -O3
        # xnfpelsyn.exe: .rodata equiv.108/109/107; gf-O0 exe and oracle .so: .data), and the -O3 code does
        # load it (xkarsas_ fall-through path), so X = XN(29,15) at FREQLG = FREQN(29,15) (WAVENO 487.456).
        z = np.zeros(1, np.float32)
        _XK = (np.concatenate([fn.ravel(order="F"), z]), np.concatenate([xn.ravel(order="F"), z]),
               np.concatenate([xl.ravel(order="F"), z]), ek)
    return _XK


def xkarsas_freqlg(freq, zeff2: float = 1.0) -> np.ndarray:
    """FREQLG=LOG10(SNGL(FREQ/ZEFF2)) (REAL*4) for a batch of frequencies."""
    f = torch.as_tensor(freq, dtype=torch.float64).reshape(-1)
    return log10f((f / zeff2).cpu().numpy().astype(np.float32))


def xkarsas(freq, zeff2: float, n: int, l: int, flg: np.ndarray | None = None) -> torch.Tensor:
    """FUNCTION XKARSAS(FREQ,ZEFF2,N,L) (IMPLICIT REAL*4; Karzas & Latter 1961) for a batch of FREQ.

    The interpolation runs in REAL*4 exactly as compiled (no FMA, left-to-right). The table search falls
    through to I=30 only when FREQLG equals FREQN(29,N) exactly; that out-of-bounds read is reproduced
    through Fortran storage order (FREQN(30,N) = FREQN(1,N+1); past the end of a table: 0, see _xk_tables).
    """
    freq = torch.as_tensor(freq, dtype=torch.float64)
    shape, dev = freq.shape, freq.device
    if flg is None:
        flg = xkarsas_freqlg(freq, zeff2)
    flg = np.asarray(flg, dtype=np.float32).reshape(-1)
    with np.errstate(all="ignore"):          # lanes below threshold (returned as 0) may divide by 0
        x, ret0 = _xkarsas_x(flg, n, l)
    x = torch.from_numpy(np.asarray(x, dtype=np.float32).astype(np.float64)).to(dev)
    r = exp(x * 2.30258509299405) / zeff2
    r = torch.where(torch.from_numpy(np.asarray(ret0)).to(dev), torch.zeros_like(r), r)
    return r.reshape(shape)


def _xkarsas_x(flg: np.ndarray, n: int, l: int):
    """REAL*4 interpolated log10 cross-section X and the `RETURN with 0` mask."""
    fn, xn, xl, ek = _xk_tables()
    f4 = np.float32
    if l >= n or n > 6:
        if n <= 15:
            col = 29 * (n - 1)
            fcol = lambda i: fn[col + i - 1]                   # noqa: E731
            ret0 = flg < fn[col + 28]
            I = np.full(flg.shape, 30)
            for i in range(29, 1, -1):
                I = np.where(flg > fn[col + i - 1], i, I)
            ycol = lambda i: xn[col + i - 1]                   # noqa: E731
            a, b = fcol(I), fcol(I - 1)
            x = (((flg - a) / (b - a)) * (ycol(I - 1) - ycol(I))) + ycol(I)
        else:
            nn = n * n
            fl29 = f4(math.log10(f32(109677.576) * 2.99792458e10 / float(nn)))
            ret0 = flg < fl29
            fn15 = np.zeros(29, dtype=np.float32)          # FREQN15(1) is never written (stays 0)
            fn15[28] = fl29
            for i in range(2, 29):
                fn15[i - 1] = f4(math.log10(float(f4(f4(ek[i - 1] + f4(f4(1.0) / f4(nn))) * f4(109677.576)))
                                            * 2.99792458e10))
            I = np.full(flg.shape, 29)
            for i in range(28, 1, -1):
                I = np.where(flg > fn15[i - 1], i, I)
            ycol = lambda i: xn[29 * 14 + i - 1]                                         # noqa: E731
            a, b = fn15[I - 1], fn15[I - 2]
            x = (((flg - a) / (b - a)) * (ycol(I - 1) - ycol(I))) + ycol(I)
    else:
        col = 29 * (n - 1)
        ret0 = flg < fn[col + 28]
        I = np.full(flg.shape, 30)
        for i in range(29, 1, -1):
            I = np.where(flg > fn[col + i - 1], i, I)
        fcol = lambda i: fn[col + i - 1]                       # noqa: E731
        lc = 29 * l + 29 * 6 * (n - 1)                                                   # XL(I,L+1,N)
        ycol = lambda i: xl[lc + i - 1]                        # noqa: E731
        a, b = fcol(I), fcol(I - 1)
        x = (((flg - a) / (b - a)) * (ycol(I - 1) - ycol(I))) + ycol(I)
    return x, ret0


# ============================================================================================ COULFF
_A = None


def _coulff_tables():
    global _A
    if _A is None:
        _A = (torch.as_tensor(table("coulff", "a"), dtype=torch.float64),
              torch.as_tensor(table("coulff", "z4log"), dtype=torch.float64))
    return _A


def coulff_k(c: Ctx, nz: int) -> torch.Tensor:
    """FUNCTION COULFF(J,NZ) for all layers and frequencies: free-free Gaunt factor (Karzas & Latter)."""
    A, z4 = _coulff_tables()
    A, z4 = A.to(c.device), z4.to(c.device)
    tlog = c.lay("temp.tlog")
    gamlog = f32(10.39638) - tlog / f32(1.15129) + float(z4[nz - 1])
    igam = torch.clamp(torch.trunc(gamlog + 7.0), max=10).clamp(min=1).to(torch.int64)
    hvktlg = (c.freqlg - tlog) / f32(1.15129) - f32(20.63764)
    ihvkt = torch.clamp(torch.trunc(hvktlg + 9.0), max=11).clamp(min=1).to(torch.int64)
    p = gamlog - (igam - 7).to(torch.float64)
    q = hvktlg - (ihvkt - 9).to(torch.float64)
    ig, ih = torch.broadcast_tensors(igam, ihvkt)
    a11, a12 = A[ig - 1, ih - 1], A[ig - 1, ih]
    a21, a22 = A[ig, ih - 1], A[ig, ih]
    return (1.0 - p) * ((1.0 - q) * a11 + q * a12) + p * ((1.0 - q) * a21 + q * a22)


# =============================================================================================== HOP
# (threshold of IF(WAVENO.LT.thr)GO TO 30, N, statistical weight, level energy) for N = 15 .. 1
_HOP_LEVELS = [
    (f32(487.456), 15, 450.0, 109191.313), (f32(559.579), 14, 392.0, 109119.188),
    (f32(648.980), 13, 338.0, 109029.789), (f32(761.649), 12, 288.0, 108917.117),
    (f32(906.426), 11, 242.0, 108772.336), (f32(1096.776), 10, 200.0, 108581.992),
    (f32(1354.044), 9, 162.0, 108324.719), (f32(1713.713), 8, 128.0, 107965.051),
    (f32(2238.320), 7, 98.0, 107440.444), (f32(3046.604), 6, 72.0, 106632.160),
    (f32(4387.113), 5, 50.0, 105291.651), (f32(6854.871), 4, 32.0, 102823.893),
    (f32(12186.462), 3, 18.0, 97492.302), (f32(27419.659), 2, 8.0, 82259.105),
    (109678.764, 1, 2.0, None),
]


def hop_k(c: Ctx, shyd_prev: torch.Tensor):
    """SUBROUTINE HOP: H I bound-free (N=1..15 + N>=16 continuum) and free-free -> (AHYD, SHYD)."""
    freq, stim, bnu, ehvkt = c.freq, c.stim, c.bnu, c.ehvkt
    hckt = c.lay("temp.hckt")
    freq3 = rdiv(f32(2.815e29), freq) / freq / freq
    h = freq3 * 2.0 / 2.0 / (109677.576 * hckt) * (
        exp(-torch.clamp(109678.764 - c.waveno, min=109250.336) * hckt) - exp(-109678.764 * hckt)) * stim
    s = h * bnu
    masks = chain(c.waveno, [t for t, *_ in _HOP_LEVELS])
    flg = xkarsas_freqlg(freq)
    for m, (_, n, g, e) in zip(masks, _HOP_LEVELS):
        if not bool(m.any()):
            break
        x = xkarsas(freq, 1.0, n, n, flg).reshape(-1, 1)
        if n >= 7:
            a = x * g * exp(-e * hckt) * stim
            ds = a * bnu
        else:
            b = c.lay("bhyd.bhyd", n)
            a = (x * g * 1.0 * (b - ehvkt)) if e is None else (x * g * exp(-e * hckt) * (b - ehvkt))
            ds = a * bnu * stim / (b - ehvkt)
        h = acc(m, h, a)
        s = acc(m, s, ds)
    xnfph1, rho = c.lay("bhyd.xnfph", 1), c.lay("state.rho")
    h = h * xnfph1 / rho
    s = s * xnfph1 / rho
    a = (rdiv(f32(3.6919e8), sqrt(c.lay("temp.t"))) * coulff_k(c, 1) / freq * c.lay("state.xne") / freq
         * c.lay("bhyd.xnfph", 2) / freq * stim / rho)
    h = h + a
    s = s + a * bnu
    shyd = torch.where(h > 0.0, s / h, shyd_prev)
    return h, shyd


def hop(s: State) -> dict:
    c = Ctx.from_state(s)
    ahyd, shyd = hop_k(c, c.lay("bhyd.shyd"))
    return {"bhyd.ahyd": put_rows(s["bhyd.ahyd"], ahyd, c.n), "bhyd.shyd": put_rows(s["bhyd.shyd"], shyd, c.n)}


# ============================================================================================ H2PLOP
def h2plop_k(c: Ctx, ah2p_prev: torch.Tensor) -> torch.Tensor:
    """SUBROUTINE H2PLOP: H2+ bound-free/free-free (returns early, AH2P untouched, for FREQ>3.28805D15)."""
    lg = c.freqlg
    fr = -3.0233e3 + (f32(3.7797e2) + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * lg) * lg) * lg) * lg
    freq15 = c.freq / 1.0e15
    es = -7.342e-3 + (-2.409 + (1.028 + (-4.230e-01 + (1.224e-01 - 1.351e-02 * freq15) * freq15) * freq15)
                      * freq15) * freq15
    a = (exp(-es / c.lay("temp.tkev") + fr + log(c.lay("bhyd.xnfph", 1))) * 2.0 * c.lay("bhyd.bhyd", 1)
         * c.lay("bhyd.xnfph", 2) / c.lay("state.rho") * c.stim)
    return torch.where(c.freq > 3.28805e15, ah2p_prev, a)


def h2plop(s: State) -> dict:
    c = Ctx.from_state(s)
    return {"bhyd.ah2p": put_rows(s["bhyd.ah2p"], h2plop_k(c, c.lay("bhyd.ah2p")), c.n)}


# ============================================================================================ LINTER
def linter_idx(xold: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
    """IOLD (1-based) reached by LINTER's walk for one point XNEW (NNEW=1 call; IOLD starts at 2)."""
    nold = xold.shape[-1]
    io = torch.full(xnew.shape, 2, dtype=torch.int64, device=xnew.device)
    alive = torch.ones(xnew.shape, dtype=torch.bool, device=xnew.device)
    for i in range(2, nold):
        alive = alive & ~(xnew < xold[i - 1])
        io = torch.where(alive, torch.full_like(io, i + 1), io)
    return io


def linter(xold: torch.Tensor, yold: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
    """SUBROUTINE LINTER(XOLD,YOLD,NOLD,XNEW,YNEW,1) for many independent points.

    xold [K]; yold [B, K]; xnew [B or 1, M] -> [B', M] (B' = broadcast of B and xnew's batch).
    Each point is a separate NNEW=1 call (as HMINOP calls it); for increasing XNEW this equals one
    NNEW=M call, since the walk only moves forward.
    """
    io = linter_idx(xold, xnew)
    K = yold.shape[-1]
    B = max(yold.shape[0], xnew.shape[0])
    io = io.expand(B, xnew.shape[1])
    yo = yold.expand(B, K)
    y1, y2 = torch.gather(yo, 1, io - 2), torch.gather(yo, 1, io - 1)
    x1, x2 = xold[io - 2], xold[io - 1]
    return y1 + (y2 - y1) / (x2 - x1) * (xnew - x1)


# ============================================================================================ HMINOP
_HM = None


def _hminop_tables():
    """DATA tables + the ISTART block (WFFLOG, FFLOG), computed once with glibc log."""
    global _HM
    if _HM is None:
        ff = table("hminop", "ff")                 # FF(11,22)
        thetaff = table("hminop", "thetaff")
        wavek = table("hminop", "wavek")
        wfflog = np.array([math.log(91.134 / w) for w in wavek])
        fflog = np.empty((22, 11))
        for iw in range(22):
            for it in range(11):
                fflog[iw, it] = math.log(ff[it, iw] / thetaff[it] * 5040.0 * 1.380658e-16)
        t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
        _HM = dict(wfflog=t(wfflog), fflog=t(fflog), thetaff=t(thetaff),
                   wbf=t(table("hminop", "wbf")), bf=t(table("hminop", "bf")))
    return _HM


def hminop_cache(c: Ctx, itemp1: int, xhmin_s: torch.Tensor, theta_s: torch.Tensor):
    """ITEMP cache of HMINOP: (itemp1, XHMIN(kw), THETA(kw)) after the call."""
    itemp = c.itemp()
    if itemp == itemp1:
        return itemp1, xhmin_s, theta_s
    n = c.n
    t, tkev = c.lay("temp.t"), c.lay("temp.tkev")
    theta = rdiv(5040.0, t)
    xhmin = (exp(rdiv(0.754209, tkev)) / (2.0 * 2.4148e15 * t * sqrt(t)) * c.lay("bhyd.bmin") * c.lay("bhyd.bhyd", 1)
             * c.lay("bhyd.xnfph", 1) * c.lay("state.xne"))
    return itemp, put_rows(xhmin_s, xhmin, n), put_rows(theta_s, theta, n)


def hminop_k(c: Ctx, xhmin: torch.Tensor, theta: torch.Tensor):
    """SUBROUTINE HMINOP after its ITEMP block: H- bound-free (Mathisen) + free-free (Bell & Berrington).
    xhmin/theta: the cached kw arrays. Returns (AHMIN, SHMIN) [NF, n]."""
    T = {k: v.to(c.device) for k, v in _hminop_tables().items()}
    n = c.n
    wave = rdiv(2.99792458e17, c.freq)                                  # [NF,1]
    wavelog = log(wave)
    fftt = torch.cat([exp(linter(T["wfflog"], T["fflog"][:, it].reshape(1, -1), wavelog))
                      for it in range(11)], dim=1)                  # [NF,11]
    bf, _ = map1(T["wbf"], T["bf"], wave.reshape(-1))
    hminbf = torch.where(c.freq > 1.82365e14, bf.reshape(-1, 1), torch.zeros_like(c.freq))
    fftheta = linter(T["thetaff"], fftt, theta[:n].reshape(1, n))  # [NF,n]
    xnfph1, bhyd1, xne, rho = c.lay("bhyd.xnfph", 1), c.lay("bhyd.bhyd", 1), c.lay("state.xne"), c.lay("state.rho")
    bmin = c.lay("bhyd.bmin")
    hminff = fftheta * xnfph1 * 2.0 * bhyd1 * xne / rho * 1.0e-26
    h = hminbf * f32(1.0e-18) * (1.0 - c.ehvkt / bmin) * xhmin[:n].reshape(1, n) / rho
    ahmin = h + hminff
    shmin = (h * c.bnu * c.stim / (bmin - c.ehvkt) + hminff * c.bnu) / ahmin
    return ahmin, shmin


def hminop(s: State) -> dict:
    c = Ctx.from_state(s)
    it1, xhmin, theta = hminop_cache(c, save_itemp1(s, "hminop"), save_get(s, "hminop", "xhmin", (99,)),
                                     save_get(s, "hminop", "theta", (99,)))
    ahmin, shmin = hminop_k(c, xhmin, theta)
    return {"bhyd.ahmin": put_rows(s["bhyd.ahmin"], ahmin, c.n), "bhyd.shmin": put_rows(s["bhyd.shmin"], shmin, c.n),
            "save.hminop.itemp1": itemp_tensor(it1), "save.hminop.xhmin": xhmin, "save.hminop.theta": theta}


# ============================================================================================ H2RAOP
def h2raop_cache(c: Ctx, itemp1: int, xnh2_s: torch.Tensor):
    itemp = c.itemp()
    if itemp == itemp1:
        return itemp1, xnh2_s
    t = c.lay("temp.t")
    q = c.lay("bhyd.xnfph", 1) * 2.0 * c.lay("bhyd.bhyd", 1)
    poly = (1.8031e-3 + (-5.0239e-7 + (8.1424e-11 - 5.0501e-15 * t) * t) * t) * t
    xnh2 = q * q * exp(rdiv(4.477, c.lay("temp.tkev")) - 4.6628e1 + poly - 1.5 * c.lay("temp.tlog")) / c.lay("state.rho")
    return itemp, put_rows(xnh2_s, xnh2, c.n)


def h2raop_k(c: Ctx, xnh2: torch.Tensor) -> torch.Tensor:
    """SUBROUTINE H2RAOP: H2 Rayleigh scattering -> SIGH2 [NF,n]."""
    wave = rdiv(2.99792458e18, torch.clamp(c.freq, max=2.922e15))
    ww = wave * wave
    sig = (8.14e-13 + rdiv(1.28e-6, ww) + rdiv(f32(1.61), ww * ww)) / (ww * ww)
    return sig * xnh2[: c.n].reshape(1, c.n)


def h2raop(s: State) -> dict:
    c = Ctx.from_state(s)
    it1, xnh2 = h2raop_cache(c, save_itemp1(s, "h2raop"), save_get(s, "h2raop", "xnh2", (99,)))
    return {"bhyd.sigh2": put_rows(s["bhyd.sigh2"], h2raop_k(c, xnh2), c.n),
            "save.h2raop.itemp1": itemp_tensor(it1), "save.h2raop.xnh2": xnh2}
