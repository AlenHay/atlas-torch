"""EOS against the exact port on the six reference models (2500-26600 K, [M/H] = 0, -1, -2, alpha):
deviation tiers written to results/v2/eos_validation.json.

A  same species list, Debye approximation, lagged lowering and constants as NMOLEC (left: its |dX/X| < 1e-3
   convergence test); B  self-consistent lowering; C  CODATA 2018 constants; D  v2 default (all elements and
   stages, n_c = sum q^2 n)."""
import json
from pathlib import Path

import pytest

from _oracle import MODELS, TIERS, model_path, oracle_populations, tier_deviations

OUT = Path(__file__).resolve().parents[2] / "results" / "v2" / "eos_validation.json"


@pytest.mark.stage
def test_reference_models():
    from atlas_torch.v2.io import format_atlas_deck, read_atlas_deck, to_synthe_deck
    res = {"tiers": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in TIERS.items()}, "models": {}}
    for name in MODELS:
        p = model_path(name)
        if not p.exists():
            pytest.skip(f"missing {p}")
        deck = to_synthe_deck(read_atlas_deck(p))
        orc = oracle_populations(format_atlas_deck(deck))
        res["models"][name] = d = tier_deviations(deck, orc)
        a = d["A_fortran_equivalent"]["all_layers"]
        assert a["n_e"] < 1e-4 and a["n_nuclei"] < 1e-6, name
        assert a["atoms"] < 2e-3 and a["molecules"] < 2e-3, name
        v2 = d["D_v2_default"]["layers_2_on"]
        assert v2["n_e"] < 0.03 and v2["rho"] < 1e-3, name
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
