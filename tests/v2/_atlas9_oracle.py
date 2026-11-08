"""Exact-port ATLAS9 (atlas_torch.atlas9, byte-identical to atlas9mem.for) as the oracle of the v2 structure modules:
the stock run's control cards and files, the iteration States at the main program's hook points, and the
quantities of one iteration (populations, continuum at the 337 frequencies, frequency integrals)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def stock_dir(model: str) -> Path:
    return EXT / "runs" / "stock" / "atlas" / model


def stock_inputs(model: str) -> tuple[str, dict]:
    """(control cards, {unit: path}) of the stock BasicATLAS run (atlas_control.com, atlas_control_start.com)."""
    d = stock_dir(model)
    text = (d / "atlas_control.com").read_text()
    cards = text.split('<<"EOF"', 1)[1].split("\n", 1)[1].rsplit("\nEOF", 1)[0] + "\n"
    files = {int(u): Path(p) for p, u in re.findall(r"ln -s (\S+) fort\.(\d+)", (d / "atlas_control_start.com").read_text())}
    return cards, files


def exact_states(model: str, iterations=(1,), at=("frequencies", "end"), max_iterations: int | None = None) -> dict:
    """{(iteration, hook name): State} of the exact port run on the stock inputs (iterations counted from 1 over
    all ITERATIONS batches)."""
    from atlas_torch.atlas9.main import run
    from atlas_torch.atlas9.program import float64_default, make_physics
    cards, files = stock_inputs(model)
    want = set(iterations)
    count = {"n": 1}
    out = {}

    def hook(name, s):
        if name in at and count["n"] in want:
            out[(count["n"], name)] = s
        if name == "end":
            count["n"] += 1
    with float64_default():
        run(cards, files, make_physics("torch"), max_iterations=max_iterations or max(want), hook=hook,
            keep_states=False)
    return out


def vec(s, key: str) -> np.ndarray:
    n = s.iscalar("rhox.nrhox")
    return s[key][:n].detach().cpu().numpy().astype(np.float64)


def populations(s) -> dict:
    """n, n/U (L, 99, 10) per element/stage from POPS(code, 12/11) of the State, and n_e, XNATOM, rho, T, P."""
    from atlas_torch.atlas9.physics import pops
    from atlas_torch.v2.eos.partition import n_stages
    n = s.iscalar("rhox.nrhox")
    num = np.zeros((n, 99, 10))
    nou = np.zeros((n, 99, 10))
    for z in range(1, 100):
        ns = n_stages(z)
        code = z + (ns - 1) / 100.0
        for mode, dst in ((12, num), (11, nou)):
            r = pops(s, code, mode)
            arr = r[0] if isinstance(r, tuple) else r
            dst[:, z - 1, :ns] = np.asarray(arr)[:n, :ns]
    return dict(n=num, n_over_u=nou, xne=vec(s, "state.xne"), xnatom=vec(s, "state.xnatom"), rho=vec(s, "state.rho"),
                t=vec(s, "temp.t"), p=vec(s, "state.p"), xnfpch=vec(s, "xnfp.xnfpch"), xnfpoh=vec(s, "xnfp.xnfpoh"))


def v2_populations(orc: dict, molecules):
    """v2 Populations holding the oracle's n, n/U, n_e, rho, and n/U of CH and OH (other molecules: 0)."""
    from atlas_torch.v2.constants import KURUCZ
    from atlas_torch.v2.eos import Populations
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)    # noqa: E731
    from atlas_torch.v2.eos import PartitionFunctions
    from atlas_torch.v2.eos.saha import debye_lowering, kurucz_charge_density
    n, nou = orc["n"], orc["n_over_u"]
    nl = n.shape[0]
    ntot = orc["p"] / (KURUCZ.k * orc["t"])
    # POPS returns n/U only for the elements of the molecular network (main program loop 444): U from PFSAHA's
    # tables at the Kurucz Debye lowering
    de = debye_lowering(t(orc["t"]), kurucz_charge_density(t(orc["xne"]), t(ntot)), KURUCZ)
    u_pf = PartitionFunctions(KURUCZ)(t(orc["t"]), de).numpy()
    u = np.where((n > 0) & (nou > 0), n / np.where(nou > 0, nou, 1.0), u_pf)
    n = np.where((n > 0) | (nou <= 0), n, nou * u)
    zm = np.zeros((nl, len(molecules)))
    nou_mol = zm.copy()
    for code, key in ((106.0, "xnfpch"), (108.0, "xnfpoh")):
        if key in orc:
            nou_mol[:, molecules.index(code)] = orc[key]
    return Populations(T=t(orc["t"]), P=t(orc["p"]), n_total=t(ntot), n_e=t(orc["xne"]), n_nuclei=t(orc["xnatom"]),
                       rho=t(orc["rho"]), mu=t(orc["rho"] / (KURUCZ.amu * ntot)), debye=torch.zeros(nl, dtype=torch.float64),
                       n_atom=t(n), u_atom=t(u), n_mol=t(zm), n_over_u_mol=t(nou_mol), molecules=molecules)


def continuum(s) -> dict:
    """KAPP(0) /OPS/ arrays (F, L) at the 337 frequencies of the State, and the frequencies (F,)."""
    from atlas_torch.atlas9.main import frequencies
    from atlas_torch.atlas9.physics import Atlas9Physics
    fq = frequencies(s)
    ops = Atlas9Physics("atlas9").continuum(s, fq)
    out = {k: v.detach().cpu().numpy() for k, v in ops.items()}
    out["freq"] = fq["freq"].detach().cpu().numpy()
    out["rco"] = fq["rco"].detach().cpu().numpy()
    return out
