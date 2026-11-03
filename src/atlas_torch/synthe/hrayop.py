"""HRAYOP (atlas7v.for): Rayleigh scattering by neutral hydrogen, Gavrila (1967) cross-sections.

Template port: reads COMMON entries from the State, returns the entries it writes. DATA tables are
loaded from the fetched Fortran source (atlas_torch.kurucz_data), never transcribed.
"""
from __future__ import annotations

import torch

from ..fnum import f32, fint
from ..interp import map1
from ..kurucz_data import table
from ..state import State

SRC = "atlas7v.for"
NU0 = 3.288051e15  # 3.288051D15


def _tables():
    t = {k: torch.as_tensor(table(SRC, "hrayop", k), dtype=torch.float64)
         for k in ("gavrilam", "gavrilamab", "gavrilambc", "gavrilamcd", "gavrilalymancont", "fgavrilalymancont")}
    return t


_T = None


def hrayop_xsect(freq: float) -> float:
    """Cross-section per H atom at frequency `freq` (scalar FREQ of COMMON /FREQ/)."""
    global _T
    if _T is None:
        _T = _tables()
    T = _T
    if freq < 3.288051e13:
        return 6.65e-25 * float(T["gavrilam"][0]) ** 2 * (freq / 3.288051e13) ** 4
    if freq <= 2.4190611e15:
        i = min(fint(freq / 3.288051e13) + 1, 74)
        gm = T["gavrilam"]
        g = float(gm[i - 2]) + (float(gm[i - 1]) - float(gm[i - 2])) / 3.288051e13 * (freq - (i - 1) * 3.288051e13)
        return 6.65e-25 * g**2
    if freq < f32(.77) * NU0:
        return 0.0
    if freq <= f32(.885) * NU0:
        i = fint((freq - f32(.755) * NU0) / 1.644026e13) + 1
        i = min(i + 1, 27)
        t = T["gavrilamab"]
        g = float(t[i - 2]) + (float(t[i - 1]) - float(t[i - 2])) / 1.644026e13 * (
            freq - (f32(.755) * NU0 + ((i - 1) - 1) * 1.664026e13))
        return 6.65e-25 * g**2
    if freq < f32(.890) * NU0:
        return 0.0
    if freq <= f32(.936) * NU0:
        i = fint((freq - f32(.890) * 3.28851e15) / 0.657610e13) + 1
        i = min(i + 1, 24)
        t = T["gavrilambc"]
        g = float(t[i - 2]) + (float(t[i - 1]) - float(t[i - 2])) / 0.657610e13 * (
            freq - (f32(.890) * NU0 + ((i - 1) - 1) * 0.657610e13))
        return 6.65e-25 * g**2
    if freq < f32(.938) * NU0:
        return 0.0
    if freq <= f32(.959) * NU0:
        i = fint((freq - f32(.938) * NU0) / 0.3288051e13) + 1
        i = min(i + 1, 22)
        t = T["gavrilamcd"]
        g = float(t[i - 2]) + (float(t[i - 1]) - float(t[i - 2])) / 0.3288051e13 * (
            freq - (f32(.938) * NU0 + ((i - 1) - 1) * 0.3288051e13))
        return 6.65e-25 * g**2
    if freq < f32(.961) * NU0:
        return 0.0
    if freq <= f32(1.000) * NU0:
        return 6.65e-25 * float(T["gavrilalymancont"][0])
    g, _ = map1(T["fgavrilalymancont"], T["gavrilalymancont"], torch.tensor([freq / NU0], dtype=torch.float64))
    return 6.65e-25 * float(g[0])


def hrayop(s: State) -> dict[str, torch.Tensor]:
    """SIGH(J) = XSECT * XNFPH(J,1) * 2 * BHYD(J,1) / RHO(J), J=1..NRHOX."""
    n = s.iscalar("rhox.nrhox")
    xsect = hrayop_xsect(s.scalar("freq.freq"))
    sigh = s["bhyd.sigh"].clone()
    sigh[:n] = xsect * s["bhyd.xnfph"][:n, 0] * 2.0 * s["bhyd.bhyd"][:n, 0] / s["state.rho"][:n]
    return {"bhyd.sigh": sigh}
