"""v2 populations in the array layout XNFPELSYN writes to fort.10 for SYNTHE/SPECTRV (xnfpelsyn.for)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ...tables import table
from ..constants import decimal32
from ..eos.equilibrium import Populations

N_SLOTS = 139


@lru_cache(maxsize=None)
def idmol() -> np.ndarray:
    """DATA IDMOL of xnfpelsyn.for: molecule code of XNFPEL slot 40..139 (0 = unused)."""
    return decimal32(table("xnfpelsyn.for", "xnfpelsyn", "idmol")).reshape(-1)


def xnfpel(pops: Populations) -> torch.Tensor:
    """XNFPEL(ION, NELEM) per layer, (..., L, 6, 139): n/U of stages 1-6 of Z=1..99; Ca-Ni stages 7-10 in
    stage-5 slots 30+Z, 40+Z, 50+Z, 60+Z (loop 250); molecule IDMOL(k) in stage-6 slot 39+k (loop 201).
    Molecules missing from the EOS set are 0."""
    nou = pops.n_over_u_atom
    out = torch.zeros(nou.shape[:-2] + (6, N_SLOTS), dtype=nou.dtype, device=nou.device)
    out[..., :, :99] = nou[..., :, :6].transpose(-1, -2)
    for z in range(20, 29):
        for off, stage in ((30, 7), (40, 8), (50, 9), (60, 10)):
            out[..., 4, off + z - 1] = nou[..., z - 1, stage - 1]
    codes = pops.molecules.codes
    for k, c in enumerate(idmol()):
        hit = np.flatnonzero(np.abs(codes - c) < 1e-3) if c else []
        if len(hit):
            out[..., 5, 39 + k] = pops.n_over_u_mol[..., int(hit[0])]
    return out


def synthe_state(pops: Populations) -> dict[str, torch.Tensor]:
    """Record-4 quantities of fort.10 that come from the EOS: XNE, XNATOM, RHO, XNFH (H I), XNFHE (He I, He II),
    XNFH2 (H2), each (..., L[, 2])."""
    h2 = pops.molecules.index(101.0)
    return {"xne": pops.n_e, "xnatom": pops.n_nuclei, "rho": pops.rho, "xnfh": pops.n_atom[..., 0, 0],
            "xnfhe": pops.n_atom[..., 1, :2], "xnfh2": pops.n_mol[..., h2]}


def continuum_tables(opacity) -> dict[str, torch.Tensor]:
    """Fort.10 continuum records of XNFPELSYN: CONTINALL, CONTABS, CONTSCAT = log10 of the total, absorption and
    scattering coefficients (cm^2 g^-1), each (..., L, F), from a v2 Opacity."""
    return {"continall": torch.log10(opacity.total), "contabs": torch.log10(opacity.absorption),
            "contscat": torch.log10(opacity.scattering)}
