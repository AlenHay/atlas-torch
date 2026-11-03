"""Shared fixtures: external (never committed) data, Fortran oracles and golden traces."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
ORACLE_DIR = EXT / "oracle"
TRACE_DIR = EXT / "traces"


def _need(p: Path):
    if not p.exists():
        pytest.skip(f"missing external artifact {p} (run M0 scripts)")
    return p


@pytest.fixture(scope="session")
def ext():
    return _need(EXT)


@pytest.fixture(scope="session")
def oracle():
    """oracle('atlas7v') -> fresh Oracle over $ATLAS_EXT/oracle/libatlas7v.so (fresh SAVE state per call)."""
    from atlas_torch.testing.fortran_oracle import Oracle

    def make(name):
        return Oracle(_need(ORACLE_DIR / f"lib{name}.so"))
    return make


@pytest.fixture(scope="session")
def golden():
    """golden('sun_500_510/xnfpelsyn', 'pops', hit, 'entry'|'exit') -> dict of numpy arrays from the gdb trace."""
    def load(stage, routine, hit=0, phase="entry"):
        p = _need(TRACE_DIR / stage / f"{routine.lower()}_{hit}_{phase}.npz")
        with np.load(p) as d:
            return {k: d[k] for k in d.files}
    return load
