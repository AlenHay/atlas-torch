"""KAPREADTS port: byte-identical kappa.ros from the stock KAPPA9 outputs (solar and alpha mixes)."""
import os
from pathlib import Path

import pytest

from atlas_torch.kappa9.kapreadts import kapreadts

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


@pytest.mark.parametrize("mix", ["solar", "alpha"])
def test_kapreadts_bytes(mix, tmp_path):
    w = EXT / "work" / f"kappa9_{mix}_stock"
    if not (w / "kappa.ros").exists():
        pytest.skip("no KAPPA9 reference work dir")
    kapreadts([w / f"kapk{v}.dat" for v in (0, 1, 2, 4, 8)], tmp_path / "kappa.ros")
    assert (tmp_path / "kappa.ros").read_bytes() == (w / "kappa.ros").read_bytes()
