"""Oracle tests for the KAPPA9 main program's own pieces (kappa9.for PROGRAM KAPPA9).

1. golden: the T grid (EXP10 with the REAL*4 constant), the first pressure and the first frequency of the
   integration (shifted /FRESET/) equal the stock run's trace.
2. the kapk writer (FORMATs 75 / 77) reproduces every stock kapk file byte for byte from its integers.
3. the frequency set: 337 frequencies NU = 90..426, RCO(90) recomputed from the prepended FRESET(89).
"""
import os
from pathlib import Path

import numpy as np
import pytest

from atlas_torch.io.formats import read_kapk
from atlas_torch.kappa9 import kappa9_prog as kp
from atlas_torch.kappa9.readin import readin
from atlas_torch.testing import kappa9_oracle_physics as k9o

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
STAGE = "odf_solar/kappa9"
RUN = EXT / "traces" / "odf_solar" / "kw"


def test_grid_golden(golden):
    g = kp.grid_temperatures()
    p = kp.grid_pressures()
    ent = golden(STAGE, "kapp", 0, "entry")
    assert np.array_equal(ent["state.p"][:57], np.full(57, p[0].item()))
    ent = golden(STAGE, "linop", 0, "entry")
    assert np.array_equal(ent["temp.t"][:57], g["t"].numpy())
    rd = readin((RUN / "stdin.txt").read_text(), RUN / "fort.2")
    freq, rco = kp.frequency_set(rd)
    assert float(ent["freq.freq"][0]) == freq[0]
    assert freq.size == 337 and np.all(np.diff(freq) < 0)
    fres = rd["freset.freset"].numpy()
    assert rco[0] == (3.364687e16 - fres[1]) * 0.5          # RCOSET(90) = (FRESET(89)-FRESET(91))*.5
    assert np.array_equal(rco[1:], rd["freset.rcoset"].numpy()[1:337])


@pytest.mark.parametrize("mix", ["solar", "alpha"])
@pytest.mark.parametrize("v", [0, 1, 2, 4, 8])
def test_kapk_writer_roundtrip(mix, v):
    p = EXT / "work" / f"kappa9_{mix}_stock" / f"kapk{v}.dat"
    if not p.exists():
        pytest.skip(f"missing {p}")
    assert kp.kapk_text(read_kapk(p)) == p.read_text()


def test_kapk_writer_overflow():
    kab = np.full((42, 57), -12345)
    kab[0, 0] = 99999
    txt = kp.kapk_text(kab)
    assert "*****," in txt and "99999," in txt
