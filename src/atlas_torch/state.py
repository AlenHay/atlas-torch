"""Explicit Fortran state for the port.

Kurucz codes communicate through COMMON blocks and SAVEd locals (compiled with -fno-automatic). The port
makes that state explicit: a `State` maps "block.var" (COMMON members, lower case, exactly as in the
Fortran) and "save.<unit>.<var>" (SAVEd/DATA locals that carry information between calls) to torch tensors
with the Fortran declared shape (dimension order as in Fortran: A(kw,8) -> shape (99, 8); index 0 is
Fortran index 1). Golden traces (tools/trace) use the same keys, so a routine's oracle test is
"State.from_trace(entry) -> port -> compare with exit".

Ported routines are pure: `def hrayop(s: State) -> dict[str, Tensor]` returns only the entries it writes;
`s.updated(changes)` gives the next state. Scalars are 0-d or shape-(1,) tensors as in the trace.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

DTYPE = torch.float64


def _to_tensor(a) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a
    a = np.asarray(a)
    if a.dtype.kind in "SUV":       # CHARACTER data stays numpy
        return a
    if a.dtype.kind == "f":
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))
    if a.dtype.kind in "iub":
        return torch.from_numpy(np.ascontiguousarray(a).astype(np.int64))
    return torch.from_numpy(np.ascontiguousarray(a))


class State(dict):
    """dict "block.var" -> tensor; copy-on-write updates."""

    @classmethod
    def from_trace(cls, d: Mapping[str, np.ndarray]) -> "State":
        s = cls()
        for k, v in d.items():
            if k.startswith("arg:") or k.startswith("static:"):
                continue
            s[k] = _to_tensor(v)
        return s

    def updated(self, changes: Mapping[str, torch.Tensor]) -> "State":
        n = State(self)
        n.update(changes)
        return n

    def scalar(self, key: str) -> float:
        v = self[key]
        return float(v.reshape(-1)[0]) if isinstance(v, torch.Tensor) else float(np.asarray(v).reshape(-1)[0])

    def iscalar(self, key: str) -> int:
        v = self[key]
        return int(v.reshape(-1)[0])
