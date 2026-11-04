"""KAPP (atlas7v.for, SYNTHE variant): total continuous opacity /OPTOT/ from the IFOP-selected routines.

    kapp(s, n=1, pops=None)            one Fortran CALL KAPP(N,NSTEPS,STEPWT): State -> entries written
    kapp_batch(s, freqs, n=1, pops)    xnfpelsyn's frequency loop: /FREQ/ set per frequency exactly as
                                       xnfpelsyn does (loop 120), then KAPP; all arrays [NF, NRHOX]

Routines on the xnfpelsyn path (IFOP(1..13); xnfpelsyn zeroes IFOP(14..17)): HOP, H2PLOP, HMINOP, HRAYOP,
HE1OP, HE2OP, HEMIOP, HERAOP, COOLOP (C1OP MG1OP AL1OP SI1OP FE1OP H2COLLOP CHOP OHOP), LUKEOP (N1OP O1OP
MG2OP SI2OP CA2OP), HOTOP, ELECOP, H2RAOP; XCONOP/XSOP (IFOP 19/20) are empty dummies. The line opacities
(HLINOP, LINOP, LINSOP, XLINOP, XLISOP; IFOP 14-18 with N>0) are not ported: NotImplementedError.

Cross-call state (SAVE, -fno-automatic) is carried as save.<unit>.* State entries: the ITEMP caches of
HMINOP, H2COLLOP, H2RAOP, LUKEOP, N1OP, MG2OP, SI2OP, CA2OP, HOTOP and the FREQ1 of CHOP/OHOP. In a batch
the frequencies are processed in order, each seeing the caches the previous one left (ITEMP is constant
within a batch, so the ITEMP caches are refreshed at most once). LUKEOP/HOTOP need POPS populations when
their ITEMP cache is stale: from `pops(s, code, mode)` or pre-filled save.lukeop.* / save.hotop.*.

Fortran side effects kept: H2COLLOP's WAVENO=FREQ/2.99792458D10 (returned as freq.waveno), FE1OP's
FLXRAD=BSI1(J,1) (flux.flxrad).
"""
from __future__ import annotations

import torch

from ..state import State
from .hrayop import hrayop_xsect
from .opacity._core import Ctx, itemp_tensor, put_rows, save_get, save_itemp1, scalar1
from .opacity.cool import coolop_k
from .opacity.helium import he1op_k, he2op_k, hemiop_k, heraop_k
from .opacity.hot import elecop_k, hot_saves, hotop_cache, hotop_k
from .opacity.hydrogen import h2plop_k, h2raop_cache, h2raop_k, hminop_cache, hminop_k, hop_k
from .opacity.luke import luke_saves, lukeop_k, saves_to_state

# arrays KAPP zeroes (loop 91), with their COMMON block
ZEROED = {
    "bhyd": ["ahyd", "ahmin", "ah2p", "ahline", "sigh", "sigh2", "shyd", "shmin", "shline"],
    "bhe": ["ahe1", "ahe2", "ahemin", "sighe", "she1", "she2"],
    "bc": ["ac1", "sc1"], "bmg": ["amg1", "smg1"], "bal": ["aal1", "sal1"], "bsi": ["asi1", "ssi1"],
    "bfe": ["afe1", "sfe1"],
    "ops": ["acool", "aluke", "ahot", "alines", "axline", "axcont", "sigel", "siglin", "sigxl", "sigx", "sxline",
            "sxcont"],
}
OPTOT = ["acont", "scont", "aline", "sline", "sigmac", "sigmal"]


def _saves(s: State) -> dict:
    return {
        "hminop": (save_itemp1(s, "hminop"), save_get(s, "hminop", "xhmin", (99,)), save_get(s, "hminop", "theta", (99,))),
        "h2raop": (save_itemp1(s, "h2raop"), save_get(s, "h2raop", "xnh2", (99,))),
        "h2collop": (save_itemp1(s, "h2collop"), save_get(s, "h2collop", "xnh2", (99,))),
        "chop": float(save_get(s, "chop", "freq1").reshape(-1)[0]),
        "ohop": float(save_get(s, "ohop", "freq1").reshape(-1)[0]),
        "luke": luke_saves(s),
        "hotop": hot_saves(s),
    }


def kapp_k(c: Ctx, n: int = 1, pops=None):
    """KAPP for the frequency batch of `c` -> (arrays {"block.var": [NF,n]}, State entries for the SAVEs,
    freq.waveno [NF,1] after the call, flux.flxrad rows [1,n] or None)."""
    s = c.s
    ifop = [int(v) for v in s["if.ifop"].reshape(-1)]
    on = lambda k: ifop[k - 1] == 1                                       # noqa: E731
    nf, nl = c.nf, c.n
    sp = {f"{b}.{v}": c.zeros() for b, vs in ZEROED.items() for v in vs}
    sv = _saves(s)
    out_saves: dict = {}
    waveno, flxrad = c.waveno, None
    exp_ = lambda t: t.expand(nf, nl)                                     # noqa: E731
    if n > 1:
        if on(15) or on(16):
            raise NotImplementedError("KAPP N>1 with LINOP/LINSOP")
        sp = {f"{b}.{v}": exp_(c.lay(f"{b}.{v}")) for b, vs in ZEROED.items() for v in vs}
        optot = {k: exp_(c.lay(f"optot.{k}")) for k in OPTOT}
        aline = sp["bhyd.ahline"] + sp["ops.alines"] + sp["ops.axline"]
        sline = torch.where(aline > 0.0, (sp["bhyd.ahline"] * sp["bhyd.shline"] + sp["ops.alines"] * c.bnu
                                          + sp["ops.axline"] * sp["ops.sxline"]) / aline, optot["sline"])
        optot.update(aline=aline, sline=sline, sigmal=sp["ops.siglin"] + sp["ops.sigxl"])
        return {**{f"optot.{k}": v for k, v in optot.items()}}, out_saves, waveno, flxrad
    if n > 0 and any(on(k) for k in range(14, 19)):
        raise NotImplementedError("KAPP: line opacities (IFOP 14-18) are not ported")
    if on(1):
        sp["bhyd.ahyd"], sp["bhyd.shyd"] = hop_k(c, sp["bhyd.shyd"])
    if on(2):
        sp["bhyd.ah2p"] = h2plop_k(c, sp["bhyd.ah2p"])
    if on(3):
        it1, xhmin, theta = hminop_cache(c, *sv["hminop"])
        out_saves.update({"save.hminop.itemp1": itemp_tensor(it1), "save.hminop.xhmin": xhmin,
                          "save.hminop.theta": theta})
        sp["bhyd.ahmin"], sp["bhyd.shmin"] = hminop_k(c, xhmin, theta)
    if on(4):
        xs = torch.tensor([hrayop_xsect(float(f)) for f in c.freq.reshape(-1)], dtype=torch.float64,
                          device=c.device).reshape(-1, 1)
        sp["bhyd.sigh"] = exp_(xs * c.lay("bhyd.xnfph", 1) * 2.0 * c.lay("bhyd.bhyd", 1) / c.lay("state.rho"))
    if on(5):
        sp["bhe.ahe1"], sp["bhe.she1"] = he1op_k(c)
    if on(6):
        sp["bhe.ahe2"], sp["bhe.she2"] = he2op_k(c)
    if on(7):
        sp["bhe.ahemin"] = exp_(hemiop_k(c))
    if on(8):
        sp["bhe.sighe"] = exp_(heraop_k(c))
    if on(9):
        prev = {"sc1": sp["bc.sc1"], "smg1": sp["bmg.smg1"], "sal1": sp["bal.sal1"], "sfe1": sp["bfe.sfe1"]}
        csv = {"h2collop": sv["h2collop"], "chop": sv["chop"], "ohop": sv["ohop"]}
        cp, nsv, waveno, flxrad = coolop_k(c, prev, csv)
        for k, b in (("ac1", "bc"), ("sc1", "bc"), ("amg1", "bmg"), ("smg1", "bmg"), ("aal1", "bal"),
                     ("sal1", "bal"), ("asi1", "bsi"), ("ssi1", "bsi"), ("afe1", "bfe"), ("sfe1", "bfe"),
                     ("acool", "ops")):
            sp[f"{b}.{k}"] = exp_(cp[k])
        out_saves.update({"save.h2collop.itemp1": itemp_tensor(nsv["h2collop"][0]),
                          "save.h2collop.xnh2": nsv["h2collop"][1],
                          "save.chop.freq1": torch.tensor([nsv["chop"]], dtype=torch.float64),
                          "save.ohop.freq1": torch.tensor([nsv["ohop"]], dtype=torch.float64),
                          "save.coolop.ah2coll": cp["ah2coll"]})
        c = c.with_waveno(waveno)
    if on(10):
        sp["ops.aluke"], new = lukeop_k(c, sv["luke"], pops)
        out_saves.update(saves_to_state(new))
    if on(11):
        it1, xnfp, q = hotop_cache(c, *sv["hotop"], pops=pops)
        sp["ops.ahot"] = hotop_k(c, xnfp, q)
        out_saves.update({"save.hotop.itemp1": itemp_tensor(it1), "save.hotop.xnfp": xnfp,
                          "save.hotop.xnfsumqq": q})
    if on(12):
        sp["ops.sigel"] = exp_(elecop_k(c))
    if on(13):
        it1, xnh2 = h2raop_cache(c, *sv["h2raop"])
        sp["bhyd.sigh2"] = h2raop_k(c, xnh2)
        out_saves.update({"save.h2raop.itemp1": itemp_tensor(it1), "save.h2raop.xnh2": xnh2})
    # IFOP(19) XCONOP, IFOP(20) XSOP: dummy routines
    g = sp.__getitem__
    bnu = c.bnu
    a = g("bhyd.ah2p") + g("bhe.ahemin") + g("ops.aluke") + g("ops.ahot")
    acont = a + g("bhyd.ahyd") + g("bhyd.ahmin") + g("ops.axcont") + g("bhe.ahe1") + g("bhe.ahe2") + g("ops.acool")
    num = (a * bnu + g("bhyd.ahyd") * g("bhyd.shyd") + g("bhyd.ahmin") * g("bhyd.shmin")
           + g("ops.axcont") * g("ops.sxcont") + g("bhe.ahe1") * g("bhe.she1") + g("bhe.ahe2") * g("bhe.she2")
           + g("bc.ac1") * g("bc.sc1") + g("bmg.amg1") * g("bmg.smg1") + g("bal.aal1") * g("bal.sal1")
           + g("bsi.asi1") * g("bsi.ssi1") + g("bfe.afe1") * g("bfe.sfe1"))
    scont = torch.where(acont > 0.0, num / acont, bnu.expand_as(acont))
    aline = g("bhyd.ahline") + g("ops.alines") + g("ops.axline")
    sline = torch.where(aline > 0.0, (g("bhyd.ahline") * g("bhyd.shline") + g("ops.alines") * bnu
                                      + g("ops.axline") * g("ops.sxline")) / aline, bnu.expand_as(aline))
    sigmac = g("bhyd.sigh") + g("bhe.sighe") + g("ops.sigel") + g("bhyd.sigh2") + g("ops.sigx")
    sigmal = g("ops.siglin") + g("ops.sigxl")
    sp.update({"optot.acont": acont, "optot.scont": scont, "optot.aline": aline, "optot.sline": sline,
               "optot.sigmac": sigmac, "optot.sigmal": sigmal})
    return sp, out_saves, waveno, flxrad


def kapp(s: State, n: int = 1, pops=None) -> dict:
    """CALL KAPP(N,NSTEPS,STEPWT) at the State's /FREQ/ -> COMMON/SAVE entries written (+ arg:nsteps,
    arg:stepwt)."""
    c = Ctx.from_state(s)
    sp, sv, waveno, flxrad = kapp_k(c, n, pops)
    nl = c.n
    out = {k: put_rows(s[k], v[0], nl) for k, v in sp.items()}
    out.update(sv)
    out["freq.waveno"] = scalar1(waveno.reshape(-1)[0], s["freq.waveno"])
    if flxrad is not None:
        out["flux.flxrad"] = put_rows(s["flux.flxrad"], flxrad, nl)
    if "save.coolop.ah2coll" in out:
        out["save.coolop.ah2coll"] = put_rows(save_get(s, "coolop", "ah2coll", (99,)), sv["save.coolop.ah2coll"][0],
                                              nl)
    out["arg:nsteps"] = torch.tensor([1])
    out["arg:stepwt"] = torch.tensor([1.0])
    return out


def kapp_batch(s: State, freqs, n: int = 1, pops=None) -> dict:
    """xnfpelsyn's `DO 125 NU` loop over `freqs` (in order): /FREQ/ set as xnfpelsyn does, then KAPP.

    Returns {"block.var": [NF, NRHOX]} for /OPTOT/ and every per-species array KAPP touches, "freq.waveno"
    [NF] (after KAPP), the SAVE entries after the last frequency (save.<unit>.*) and "flux.flxrad" (kw).
    """
    c = Ctx.from_freqs(s, freqs)
    sp, sv, waveno, flxrad = kapp_k(c, n, pops)
    out = dict(sp)
    out.update({k: v for k, v in sv.items() if k != "save.coolop.ah2coll"})
    if "save.coolop.ah2coll" in sv:
        out["save.coolop.ah2coll"] = put_rows(save_get(s, "coolop", "ah2coll", (99,)), sv["save.coolop.ah2coll"][-1],
                                              c.n)
    out["freq.waveno"] = waveno.reshape(-1)
    for k in ("freq", "freqlg"):
        out[f"freq.{k}"] = getattr(c, k).reshape(-1)
    out["freq.ehvkt"], out["freq.stim"], out["freq.bnu"] = c.ehvkt, c.stim, c.bnu
    if flxrad is not None:
        out["flux.flxrad"] = put_rows(s["flux.flxrad"], flxrad, c.n)
    return out
