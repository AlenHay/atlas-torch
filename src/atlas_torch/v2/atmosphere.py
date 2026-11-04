"""Model atmosphere structure on a depth grid; batch dims lead, depth is the last axis."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import torch

from .constants import CODATA2018, PhysicalConstants


@dataclass(frozen=True)
class Atmosphere:
    """Depth structure, every tensor (..., L) except teff/logg (...).

    rhox   column mass, g cm^-2 (the ATLAS depth variable)
    T      temperature, K
    P      gas pressure, dyn cm^-2
    n_e    electron number density from the model, cm^-3 (input estimate; the EOS recomputes it)
    vturb  microturbulent velocity, cm s^-1
    extra  further deck columns by name (ABROSS, ACCRAD, FLXRAD, VCONV, VELSND, ...)
    """
    rhox: torch.Tensor
    T: torch.Tensor
    P: torch.Tensor
    n_e: torch.Tensor | None = None
    vturb: torch.Tensor | None = None
    teff: torch.Tensor | None = None
    logg: torch.Tensor | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return self.T.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        return self.T.shape[:-1]

    def n_total(self, const: PhysicalConstants = CODATA2018) -> torch.Tensor:
        """Total particle density P/kT, cm^-3."""
        return self.P / (const.k * self.T)

    def to(self, device) -> "Atmosphere":
        mv = lambda v: v.to(device) if isinstance(v, torch.Tensor) else v
        kw = {f.name: mv(getattr(self, f.name)) for f in fields(self) if f.name != "extra"}
        return Atmosphere(**kw, extra={k: mv(v) for k, v in self.extra.items()})

    def with_(self, **kw) -> "Atmosphere":
        return replace(self, **kw)

    @staticmethod
    def stack(models: list["Atmosphere"]) -> "Atmosphere":
        """Batch models with the same number of layers along a new leading dim."""
        def st(name):
            vals = [getattr(m, name) for m in models]
            return None if any(v is None for v in vals) else torch.stack(vals)
        keys = set.intersection(*(set(m.extra) for m in models))
        return Atmosphere(**{f.name: st(f.name) for f in fields(Atmosphere) if f.name != "extra"},
                          extra={k: torch.stack([m.extra[k] for m in models]) for k in keys})
