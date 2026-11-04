"""Continuous opacity against the exact port (KAPP as XNFPELSYN calls it) on the solar, a hot (9000 K) and a cool
(4500 K, log g 2.5) model: results/v2/continuum_validation.json.

A  exact-port populations, Kurucz constants, every KuruczCompat switch on (left: REAL*4 literals and table values,
   Kurucz's gamma^2 constant in the Gaunt factor, the 1E-30 floor, threshold-window and gap-boundary points);
B  v2 physics; C  CODATA 2018 constants; D  v2 populations (the whole v2 default chain)."""
import json
from pathlib import Path

import pytest

from _opacity_oracle import TIERS, model_validation
from _oracle import model_path

OUT = Path(__file__).resolve().parents[2] / "results" / "v2" / "continuum_validation.json"
MODELS = ("sun", "t9000g40", "t4500g25")


@pytest.mark.stage
def test_reference_models():
    res = {"tiers": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in TIERS.items()}, "models": {}}
    for name in MODELS:
        if not model_path(name).exists():
            pytest.skip(f"missing {model_path(name)}")
        res["models"][name] = d = model_validation(name)
        a = d["tiers"]["A_kurucz_equivalent"]
        for g, v in a["groups"].items():
            if g not in ("bhyd.ahyd", "bhyd.sigh") and "all" in v["rel"]:
                assert v["rel"]["all"]["max"] < 2e-4 or v["impact"]["all"]["max"] < 1e-6, (name, g)
        assert a["totals"]["contabs"]["all"]["rms"] < 1e-4, name
        assert a["totals"]["continall"]["all"]["rms"] < 1e-4, name
        for tier, tol in (("C_codata", 3e-3), ("D_v2_default", 1e-2)):
            for band in ("365_821", "821_1644", "gt1644"):
                assert d["tiers"][tier]["totals"]["continall"][band]["rms"] < tol, (name, tier, band)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
