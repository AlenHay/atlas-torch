"""KAPP (kappa9.for; identical in atlas9mem.for): combination of the opacity sources into /OPTOT/.

N = 1 (after the source routines filled /OPS/, which KAPP zeroes first):
    A      = AH2P+AHE1+AHE2+AHEMIN+ACOOL+ALUKE+AHOT
    ACONT  = A+AHYD+AHMIN+AXCONT
    SCONT  = BNU, or (A*BNU+AHYD*SHYD+AHMIN*SHMIN+AXCONT*SXCONT)/ACONT if ACONT > 0
    ALINE  = AHLINE+ALINES+AXLINE
    SLINE  = BNU, or (AHLINE*SHLINE+ALINES*BNU+AXLINE*SXLINE)/ALINE if ALINE > 0
    SIGMAC = SIGH+SIGHE+SIGEL+SIGH2+SIGX
    SIGMAL = SIGLIN+SIGXL
N > 1 (only LINOP/LINSOP re-run): ALINE, SLINE (only where ALINE > 0; else the previous SLINE stays), SIGMAL.
Sums are left to right as written. Tensors broadcast (layers on any dims).
"""
from __future__ import annotations

import torch


def kapp_first(ops: dict, bnu: torch.Tensor) -> dict:
    a = ops["ah2p"] + ops["ahe1"] + ops["ahe2"] + ops["ahemin"] + ops["acool"] + ops["aluke"] + ops["ahot"]
    acont = a + ops["ahyd"] + ops["ahmin"] + ops["axcont"]
    scont = torch.where(acont > 0.0,
                        (a * bnu + ops["ahyd"] * ops["shyd"] + ops["ahmin"] * ops["shmin"]
                         + ops["axcont"] * ops["sxcont"]) / torch.where(acont > 0.0, acont, 1.0), bnu)
    aline = ops["ahline"] + ops["alines"] + ops["axline"]
    sline = torch.where(aline > 0.0,
                        (ops["ahline"] * ops["shline"] + ops["alines"] * bnu + ops["axline"] * ops["sxline"])
                        / torch.where(aline > 0.0, aline, 1.0), bnu)
    sigmac = ops["sigh"] + ops["sighe"] + ops["sigel"] + ops["sigh2"] + ops["sigx"]
    sigmal = ops["siglin"] + ops["sigxl"]
    return {"acont": acont, "scont": scont, "aline": aline, "sline": sline, "sigmac": sigmac, "sigmal": sigmal}


def kapp_next(ops: dict, bnu: torch.Tensor, sline_prev: torch.Tensor) -> dict:
    aline = ops["ahline"] + ops["alines"] + ops["axline"]
    sline = torch.where(aline > 0.0,
                        (ops["ahline"] * ops["shline"] + ops["alines"] * bnu + ops["axline"] * ops["sxline"])
                        / torch.where(aline > 0.0, aline, 1.0), sline_prev)
    sigmal = ops["siglin"] + ops["sigxl"]
    return {"aline": aline, "sline": sline, "sigmal": sigmal}
