"""Exact-port continuous opacities (atlas_torch.synthe KAPP as XNFPELSYN calls it, per routine) and v2 Populations
built from the exact port's populations, to compare v2 opacities with and without EOS differences."""
from __future__ import annotations

import numpy as np
import torch

# v2 process -> exact-port KAPP array it is part of
GROUPS = {
    "bhyd.ahyd": ("H I bf", "H I ff"), "bhyd.ahmin": ("H- bf", "H- ff"), "bhyd.ah2p": ("H2+",),
    "bhe.ahe1": ("He I bf", "He I ff"), "bhe.ahe2": ("He II bf", "He II ff"), "bhe.ahemin": ("He- ff",),
    "bc.ac1": ("C I bf",), "bmg.amg1": ("Mg I bf",), "bal.aal1": ("Al I bf",), "bsi.asi1": ("Si I bf",),
    "bfe.afe1": ("Fe I bf",), "choh": ("CH", "OH"), "h2coll": ("H2-H2 CIA", "H2-He CIA"),
    "ops.aluke": ("N I bf", "O I bf", "Mg II bf", "Si II bf", "Ca II bf"),
    "ops.ahot": ("C N O Ne ions bf", "metal ff"),
    "bhyd.sigh": ("H I Rayleigh",), "bhe.sighe": ("He I Rayleigh",), "bhyd.sigh2": ("H2 Rayleigh",),
    "ops.sigel": ("electrons",),
}


def oracle_kapp(deck_text: str) -> dict:
    """Run the exact-port XNFPELSYN on a deck; return freqset (F,), the KAPP arrays per routine (L, F) (plus 'choh' and
    'h2coll' split out of ACOOL), 'acont', 'sigmac', and the fort.10 tables contabs/contscat/continall (L, F)."""
    from atlas_torch.synthe.kapp import kapp_k
    from atlas_torch.synthe.opacity._core import Ctx
    from atlas_torch.synthe.opacity.cool import h2collop_cache, h2collop_k
    from atlas_torch.synthe.pops import pops
    from atlas_torch.synthe.xnfpelsyn_prog import TorchPhysics, xnfpelsyn

    class Physics(TorchPhysics):
        def kapp(self, freq, freqlg, waveno, ehvkt, stim, bnu):
            c = Ctx(self.s, freq, freqlg, waveno, ehvkt, stim, bnu)
            sp, _, _, _ = kapp_k(c, 1, lambda s, code, mode: pops(s, code, mode)[0])
            _, xnh2 = h2collop_cache(c, 0, torch.zeros(99, dtype=torch.float64))
            self.sp = dict(sp)
            self.sp["h2coll"] = h2collop_k(c, xnh2)[0]
            return sp["optot.acont"], sp["optot.sigmac"]

    phys = Physics()
    r = xnfpelsyn(deck_text, physics=phys)
    n, nf = r.nt, len(r.freqset)
    sp = {k: v.detach().cpu().numpy().T for k, v in phys.sp.items()}
    sp["choh"] = sp["ops.acool"] - sum(sp[k] for k in ("bc.ac1", "bmg.amg1", "bal.aal1", "bsi.asi1", "bfe.afe1",
                                                         "h2coll"))
    out = {k: sp[k] for k in GROUPS}
    out.update(acont=sp["optot.acont"], sigmac=sp["optot.sigmac"], freqset=np.asarray(r.freqset),
               contabs=r.contabs[:, :nf], contscat=r.contscat[:, :nf], continall=r.continall[:, :nf], n_layers=n)
    return out


def populations_from_oracle(orc: dict, molecules):
    """v2 Populations holding the exact port's n, n/U, n_e, rho (from _oracle.oracle_populations)."""
    from atlas_torch.v2.constants import KURUCZ
    from atlas_torch.v2.eos import Populations
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)    # noqa: E731
    n, nou = orc["n"], orc["n_over_u"]
    u = np.where(nou > 0, n / np.where(nou > 0, nou, 1.0), 1.0)
    nl = n.shape[0]
    n_mol = np.zeros((nl, len(molecules)))
    nou_mol = np.zeros((nl, len(molecules)))
    for k, code in enumerate(orc["mol_codes"]):
        if code >= 100.0:
            try:
                i = molecules.index(code)
            except KeyError:
                continue
            n_mol[:, i], nou_mol[:, i] = orc["xnmol"][:, k], orc["xnfpmol"][:, k]
    ntot = orc["p"] / (KURUCZ.k * orc["t"])
    return Populations(T=t(orc["t"]), P=t(orc["p"]), n_total=t(ntot), n_e=t(orc["xne"]), n_nuclei=t(orc["xnatom"]),
                       rho=t(orc["rho"]), mu=t(orc["rho"] / (KURUCZ.amu * ntot)),
                       debye=torch.zeros(nl, dtype=torch.float64), n_atom=t(n), u_atom=t(u), n_mol=t(n_mol), n_over_u_mol=t(nou_mol), molecules=molecules)


# wavelength bands (nm) for the statistics: H Lyman continuum, Balmer continuum / UV, Paschen continuum / visible,
# near IR to the H- threshold, IR
BANDS = {"lt91": (0.0, 91.18), "91_365": (91.18, 364.7), "365_821": (364.7, 820.6), "821_1644": (820.6, 1643.9),
         "gt1644": (1643.9, np.inf)}
# atlas7v.for C1OP, MG1OP, AL1OP, SI1OP start their level sums at 1E-30 cm^2 (times n/U of the neutral atom)
FLOOR_1E30 = {"bc.ac1": (6, 1), "bmg.amg1": (12, 1), "bal.aal1": (13, 1), "bsi.asi1": (14, 1)}
SIGNIFICANT = 1e-3       # share of the total extinction above which a process's relative deviation is reported


def _stats(rel: np.ndarray, mask: np.ndarray, lam: np.ndarray) -> dict:
    out = {}
    for band, (lo, hi) in {"all": (0.0, np.inf), **BANDS}.items():
        m = mask & ((lam >= lo) & (lam < hi))[None, :]
        if not m.any():
            continue
        r = np.where(m, rel, 0.0)
        j, f = np.unravel_index(np.argmax(r), r.shape)
        out[band] = {"max": float(r.max()), "rms": float(np.sqrt(np.mean(rel[m] ** 2))), "layer": int(j) + 1,
                     "lambda_nm": float(lam[f])}
    return out


def deviations(op, ok: dict, add_floor: bool = False, pops=None) -> dict:
    """Per exact-port routine group (GROUPS): relative deviation where the group holds >= SIGNIFICANT of the total
    extinction, and its impact |v2 - port| / total on the total; per wavelength band. Totals: |delta log10| of
    CONTABS, CONTSCAT, CONTINALL. add_floor: add the 1E-30 floor of FLOOR_1E30 to v2 (needs pops)."""
    lam = 2.99792458e17 / ok["freqset"]
    total = ok["acont"] + ok["sigmac"]
    comp = op.absorbers | op.scatterers
    out = {"groups": {}, "totals": {}}
    for g, procs in GROUPS.items():
        if not all(p in comp for p in procs):
            continue
        v = sum(comp[p] for p in procs).detach().cpu().numpy()
        if add_floor and g in FLOOR_1E30:
            z, s = FLOOR_1E30[g]
            v = v + 1e-30 * (pops.n_over_u_atom[..., z - 1, s - 1] / pops.rho).numpy()[:, None]
        o = ok[g]
        with np.errstate(all="ignore"):
            rel = np.abs(v / o - 1.0)
        sig = (o > SIGNIFICANT * total) & (o > 0)
        out["groups"][g] = {"rel": _stats(rel, sig, lam),
                            "impact": _stats(np.abs(v - o) / total, np.ones_like(sig), lam)}
    a, s = op.absorption.detach().cpu().numpy(), op.scattering.detach().cpu().numpy()
    for k, v in (("contabs", np.log10(a)), ("contscat", np.log10(s)), ("continall", np.log10(a + s))):
        out["totals"][k] = _stats(np.abs(v - ok[k]), np.ones_like(v, dtype=bool), lam)
    return out


TIERS = {
    "A_kurucz_equivalent": dict(pops="exact_port", const="KURUCZ", compat="all"),
    "B_v2_physics": dict(pops="exact_port", const="KURUCZ", compat="none"),
    "C_codata": dict(pops="exact_port", const="CODATA2018", compat="none"),
    "D_v2_default": dict(pops="v2_eos", const="CODATA2018", compat="none"),
}
STOCK_FORT10 = {"sun": "work/stocksun_500_510/xnfpelsyn.dat"}


def _round(x):
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return float(f"{x:.3g}") if isinstance(x, float) else x


def model_validation(name: str) -> dict:
    """Tiers A-D against the exact port; per KuruczCompat switch, its effect on CONTINALL (tier A with the switch
    off); for models with a stock XNFPELSYN fort.10, tiers A and D against it."""
    from _oracle import EXT, model_path, oracle_populations
    from atlas_torch.synthe.xnfpelsyn_prog import read_fort10
    from atlas_torch.v2 import constants
    from atlas_torch.v2.eos import ChemicalEquilibrium
    from atlas_torch.v2.io import format_atlas_deck, read_atlas_deck, to_synthe_deck
    from atlas_torch.v2.opacity.continuum import ContinuousOpacity, KuruczCompat
    deck = to_synthe_deck(read_atlas_deck(model_path(name)))
    text = format_atlas_deck(deck)
    ok = oracle_kapp(text)
    eq = ChemicalEquilibrium()
    pops = {"exact_port": populations_from_oracle(oracle_populations(text), eq.molecules),
            "v2_eos": eq.solve(deck.atmosphere, deck.composition)}
    nu = torch.as_tensor(ok["freqset"])
    compat = {"all": KuruczCompat.all(), "none": KuruczCompat()}
    out, ops = {"tiers": {}, "switches": {}}, {}
    for tier, t in TIERS.items():
        p = pops[t["pops"]]
        ops[tier] = op = ContinuousOpacity(const=getattr(constants, t["const"]),
                                           compat=compat[t["compat"]])(p, nu, parts=True)
        out["tiers"][tier] = deviations(op, ok, add_floor=tier.startswith("A"), pops=p)
    lam = 2.99792458e17 / ok["freqset"]
    ref = np.log10(ops["A_kurucz_equivalent"].total.numpy())
    for flag in KuruczCompat.__dataclass_fields__:
        cfg = KuruczCompat(**{k: k != flag for k in KuruczCompat.__dataclass_fields__})
        op = ContinuousOpacity(const=constants.KURUCZ, compat=cfg)(pops["exact_port"], nu)
        d = np.abs(np.log10(op.total.numpy()) - ref)
        out["switches"][flag] = _stats(d, np.ones_like(d, dtype=bool), lam)
    if name in STOCK_FORT10:
        st = read_fort10(EXT / STOCK_FORT10[name])
        n, nf = ok["n_layers"], len(ok["freqset"])
        out["stock_fort10"] = {}
        for tier in ("A_kurucz_equivalent", "D_v2_default"):
            op = ops[tier]
            v = {"contabs": op.absorption, "contscat": op.scattering, "continall": op.total}
            out["stock_fort10"][tier] = {k: _stats(np.abs(np.log10(x.numpy()) - st[k][:n, :nf]),
                                                   np.ones((n, nf), dtype=bool), lam) for k, x in v.items()}
        out["stock_fort10"]["exact_port"] = {k: _stats(np.abs(ok[k] - st[k][:n, :nf]), np.ones((n, nf), dtype=bool),
                                                       lam) for k in ("contabs", "contscat", "continall")}
    return _round(out)
