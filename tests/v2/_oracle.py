"""Exact-port populations (atlas_torch.synthe: READIN + POPS/NMOLEC/PFSAHA as XNFPELSYN calls them) and
deviation statistics of v2 populations against them."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from atlas_torch.v2.eos.partition import n_stages

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
MODELS = ("sun", "t4500g25", "t5000g45m20", "t5500g40alpha", "t6500g40m10", "t9000g40")


def model_path(name: str) -> Path:
    return EXT / "runs" / "stock" / "atlas" / name / "output_summary.out"


def oracle_populations(deck_text: str) -> dict:
    """xne, xnatom, rho (L,); n, n_over_u (L,99,10) per element/stage from POPS(code,12/11); molecule codes
    with XNMOL and XNFPMOL (L, nmol)."""
    from atlas_torch.synthe.xnfpelsyn_prog import TorchPhysics, default_paths
    phys = TorchPhysics()
    phys.readin(deck_text, default_paths()["molecules"])
    for k, v in (("temp.itemp", 1), ("if.ifpres", 1), ("iter.iter", 1), ("iter.numits", 1)):
        phys.set(k, v)
    n = int(phys.get("rhox.nrhox").reshape(-1)[0])
    num = np.zeros((n, 99, 10))
    nou = np.zeros((n, 99, 10))
    for z in range(1, 100):
        ns = n_stages(z)
        code = z + (ns - 1) / 100.0
        num[:, z - 1, :ns] = phys.pops(code, 12, torch.zeros(99, 10, dtype=torch.float64))[:n, :ns].numpy()
        nou[:, z - 1, :ns] = phys.pops(code, 11, torch.zeros(99, 10, dtype=torch.float64))[:n, :ns].numpy()
    s = phys.s
    nmol = int(s["xnmol.nummol"].reshape(-1)[0])
    g = lambda k: s[k].numpy()[:n]
    return dict(xne=g("state.xne"), xnatom=g("state.xnatom"), rho=g("state.rho"), p=g("state.p"), t=g("temp.t"),
                n=num, n_over_u=nou, mol_codes=s["xnmol.code"].numpy()[:nmol],
                xnmol=s["xnmol.xnmol"].numpy()[:n, :nmol], xnfpmol=s["xnmol.xnfpmol"].numpy()[:n, :nmol])


def _rel(a, b):
    return np.abs(a / np.where(b == 0, np.nan, b) - 1.0)


def nmolec_start_ne(orc: dict) -> np.ndarray:
    """n_e at which NMOLEC evaluates the Debye lowering of layer J: P/kT/20 for J=1, else the converged
    n_e(J-1) * P(J)/P(J-1)."""
    from atlas_torch.v2.constants import KURUCZ
    ntot = orc["p"] / (KURUCZ.k * orc["t"])
    return np.concatenate([ntot[:1] / 20.0, orc["xne"][:-1] * orc["p"][1:] / orc["p"][:-1]])


def deviations(pops, orc: dict, min_fraction: float = 1e-6, first_layer: int = 0, skip_elements=()) -> dict:
    """Max |v2/oracle - 1| over layers >= first_layer: n_e, n_nuclei, rho, mu; atoms/ions (and n/U) over stages
    holding more than `min_fraction` of their element; molecules holding more than `min_fraction` of their
    rarest constituent. Worst species listed with (code, layer)."""
    from atlas_torch.v2.constants import KURUCZ
    out = {}
    sl = slice(first_layer, None)
    orc = {k: (v[sl] if k != "mol_codes" else v) for k, v in orc.items()}
    npy = lambda t: t.detach().cpu().numpy()[sl]
    out["n_e"] = float(np.max(_rel(npy(pops.n_e), orc["xne"])))
    out["n_e_layer"] = int(np.argmax(_rel(npy(pops.n_e), orc["xne"]))) + 1 + first_layer
    out["n_nuclei"] = float(np.max(_rel(npy(pops.n_nuclei), orc["xnatom"])))
    out["rho"] = float(np.max(_rel(npy(pops.rho), orc["rho"])))
    mu_o = orc["rho"] / (KURUCZ.amu * orc["p"] / (KURUCZ.k * orc["t"]))
    out["mu"] = float(np.max(_rel(npy(pops.mu), mu_o)))
    n_v, nu_v = npy(pops.n_atom), npy(pops.n_over_u_atom)
    elem = orc["n"].sum(-1, keepdims=True)
    sig = (orc["n"] > min_fraction * elem) & (orc["n"] > 0)
    sig[:, [z - 1 for z in skip_elements], :] = False
    for key, v, o in (("atoms", n_v, orc["n"]), ("atoms_n_over_u", nu_v, orc["n_over_u"])):
        r = np.where(sig, _rel(v, o), 0.0)
        j, z, s = np.unravel_index(np.nanargmax(r), r.shape)
        out[key] = float(np.nanmax(r))
        out[key + "_worst"] = f"{z + 1}.{s:02d} layer {j + 1 + first_layer}"
        per = np.nanmax(r, axis=0)
        top = np.argsort(per.ravel())[::-1][:8]
        out[key + "_top"] = {f"{t // 10 + 1}.{t % 10:02d}": float(per.ravel()[t]) for t in top}
    mols = pops.molecules
    codes = orc["mol_codes"]
    rows = []
    for k, c in enumerate(codes):
        if c < 100:
            continue
        i = mols.index(c)
        zs = np.flatnonzero(mols.stoich[i])
        if any(z + 1 in skip_elements for z in zs):
            continue
        ab = orc["n"][:, zs, :].sum(-1).min(-1)
        keep = orc["xnmol"][:, k] > min_fraction * ab
        if not keep.any():
            continue
        rn = np.where(keep, _rel(npy(pops.n_mol)[:, i], orc["xnmol"][:, k]), 0.0)
        ru = np.where(keep, _rel(npy(pops.n_over_u_mol)[:, i], orc["xnfpmol"][:, k]), 0.0)
        rows.append((float(np.nanmax(rn)), float(np.nanmax(ru)), float(c), int(np.nanargmax(rn)) + 1 + first_layer))
    rows.sort(reverse=True)
    out["molecules"] = max(r[0] for r in rows)
    out["molecules_n_over_u"] = max(r[1] for r in rows)
    out["molecules_top"] = {f"{c:.2f}": [rn, ru, j] for rn, ru, c, j in rows[:8]}
    return out


def spectators() -> tuple[int, ...]:
    """Elements NMOLEC leaves out of the equilibrium (no atom/ion line in molecules.dat), plus Li, Be, B, which
    enter only as neutral components of LiH/BeH/BH."""
    from atlas_torch.v2.io import read_molecules_dat
    listed = {int(r.code) for r in read_molecules_dat() if r.code < 100}
    return tuple(z for z in range(1, 100) if z not in listed) + (3, 4, 5)


TIERS = {
    "A_fortran_equivalent": dict(network="kurucz", debye="kurucz", const="KURUCZ", lagged_lowering=True),
    "B_consistent_lowering": dict(network="kurucz", debye="kurucz", const="KURUCZ", lagged_lowering=False),
    "C_codata": dict(network="kurucz", debye="kurucz", const="CODATA2018", lagged_lowering=False),
    "D_v2_default": dict(network="full", debye="exact", const="CODATA2018", lagged_lowering=False),
}


def solve_tier(deck, orc: dict, tier: str):
    from atlas_torch.v2 import constants
    from atlas_torch.v2.eos import ChemicalEquilibrium
    t = TIERS[tier]
    eq = ChemicalEquilibrium(network=t["network"], debye=t["debye"], const=getattr(constants, t["const"]))
    lag = torch.as_tensor(nmolec_start_ne(orc)) if t["lagged_lowering"] else None
    return eq.solve(deck.atmosphere, deck.composition, debye_n_e=lag)


def tier_deviations(deck, orc: dict) -> dict:
    """Deviations of every tier, all layers and layers >= 2; tiers A-C without the spectator elements (NMOLEC
    puts them in afterwards with the final n_e)."""
    out = {}
    for tier in TIERS:
        pops = solve_tier(deck, orc, tier)
        skip = spectators() if tier != "D_v2_default" else ()
        out[tier] = {"all_layers": deviations(pops, orc, skip_elements=skip),
                     "layers_2_on": deviations(pops, orc, first_layer=1, skip_elements=skip)}
    return out
