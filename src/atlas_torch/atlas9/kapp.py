"""KAPP (atlas9mem.for): combination of the opacity sources of COMMON /OPS/ into /OPTOT/.

The sources themselves (HOP, HMINOP, ..., XCONOP) are the continuum physics behind `Physics.continuum`; this
module is KAPP's own arithmetic (loop 100 for N<=1, loop 201 for the line steps N>1), elementwise on [..., n]
tensors, left-to-right sums as in the Fortran.
"""
from __future__ import annotations

import torch

OPS = ("ahyd", "ah2p", "ahmin", "sigh", "ahe1", "ahe2", "ahemin", "sighe", "acool", "aluke", "ahot", "sigel",
       "sigh2", "ahline", "alines", "siglin", "axline", "sigxl", "axcont", "sigx", "shyd", "shmin", "shline",
       "sxline", "sxcont")


def kapp_combine(ops: dict, bnu: torch.Tensor) -> dict:
    """KAPP loop 100: ACONT, SCONT, ALINE, SLINE, SIGMAC, SIGMAL from the 25 /OPS/ arrays (dict name->tensor)."""
    o = ops
    a = o["ah2p"] + o["ahe1"] + o["ahe2"] + o["ahemin"] + o["acool"] + o["aluke"] + o["ahot"]
    acont = a + o["ahyd"] + o["ahmin"] + o["axcont"]
    scont = torch.where(acont > 0, (a * bnu + o["ahyd"] * o["shyd"] + o["ahmin"] * o["shmin"]
                                    + o["axcont"] * o["sxcont"]) / acont, bnu)
    aline = o["ahline"] + o["alines"] + o["axline"]
    sline = torch.where(aline > 0, (o["ahline"] * o["shline"] + o["alines"] * bnu + o["axline"] * o["sxline"])
                        / aline, bnu)
    sigmac = o["sigh"] + o["sighe"] + o["sigel"] + o["sigh2"] + o["sigx"]
    sigmal = o["siglin"] + o["sigxl"]
    return {"acont": acont, "scont": scont, "aline": aline, "sline": sline, "sigmac": sigmac, "sigmal": sigmal}


def kapp_lines(ops: dict, bnu: torch.Tensor, sline_prev: torch.Tensor) -> dict:
    """KAPP label 200 (N>1): only the line part is recomputed; SLINE keeps its value when ALINE<=0."""
    o = ops
    aline = o["ahline"] + o["alines"] + o["axline"]
    sline = torch.where(aline > 0, (o["ahline"] * o["shline"] + o["alines"] * bnu + o["axline"] * o["sxline"])
                        / aline, sline_prev)
    return {"aline": aline, "sline": sline, "sigmal": o["siglin"] + o["sigxl"]}
