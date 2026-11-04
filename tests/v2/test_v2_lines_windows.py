"""Line opacity against SYNTHE on the stage inputs of the stock solar model (work dirs stocksun_<window> of
scripts/synthe_workdir.sh), per line class, and its effect on the spectrum (stock SPECTRV fed with the v2 line
opacity instead of SYNTHE's fort.9). Results in results/v2/line_opacity_validation.json:

  kurucz          KURUCZ constants, SYNTHE's line selection, isotopic fractions and numerical choices
                  (KURUCZ_OPTIONS): gated by tolerances.json stages.synthe
  default         v2 defaults (exact Voigt function, no wing truncation, wide selection, CODATA constants; line
                  centres on the grid points as SYNTHE): recorded, not gated (differences attributed by toggles)
  toggles         spectrum deviation with one KURUCZ_OPTIONS choice switched to v2 physics at a time
                  (line_center=exact: positions not moved to the grid, not a default)
  v2_eos          default line opacity from the v2 EOS populations instead of XNFPELSYN's
"""
import json
from pathlib import Path

import pytest
import torch

from _lines_oracle import (CLASSES, MODEL, WINDOWS, class_references, continuum_from_xnfpelsyn, deviation_stats,
                           flux_stats, gas_from_xnfpelsyn, read_fort9, spectrum_with, workdir)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v2" / "line_opacity_validation.json"
KINDS = {"atomic": ("ATOMIC",), "molecular": ("MOLECULAR",), "hydrogen": ("HYDROGEN",),
         "helium": ("HELIUM_I", "HELIUM_II", "AUTOIONIZING")}
TOGGLES = {"const=CODATA2018": dict(const="CODATA2018"), "line_center=exact": dict(line_center="exact"),
           "voigt=exact": dict(voigt="exact"), "doppler_mass=line": dict(doppler_mass="line"),
           "max_wing_points=None": dict(max_wing_points=None), "cutoff_rule=local": dict(cutoff_rule="local"),
           "helium_ii=voigt": dict(helium_ii="voigt"), "isotopes=physical": "isotopes", "selection=wide": "wide"}


def _gates():
    tol = json.loads((ROOT / "results" / "v2" / "tolerances.json").read_text())["stages"]["synthe"]
    return {k: tol[k]["tol"] for k in ("normflux:abs", "normflux:p99", "normflux:rms_abs")}


@pytest.mark.stage
@pytest.mark.parametrize("window", list(WINDOWS))
def test_window(window):
    from atlas_torch.v2.constants import CODATA2018, KURUCZ
    from atlas_torch.v2.opacity.lines import (KURUCZ_OPTIONS, GasState, LineKind, LineOpacity, LineOpacityOptions,
                                              LineSelection, LogWavelengthGrid, read_basicatlas_lines)
    wd = workdir(window)
    if not (wd / "pre_spectrv").exists():
        pytest.skip(f"missing {wd} (scripts/synthe_workdir.sh)")
    torch.set_num_threads(4)
    wb, we = WINDOWS[window]
    grid = LogWavelengthGrid.window(wb, we)
    gas = gas_from_xnfpelsyn(wd / "xnfpelsyn.dat")
    L = gas.T.shape[0]
    cont = continuum_from_xnfpelsyn(wd / "xnfpelsyn.dat", grid)
    c = cont.numpy()
    split = class_references(window)
    refs = {k: read_fort9(split / k / "fort.9", L) for k in CLASSES}
    full = read_fort9(wd / "fort.9", L)
    base = spectrum_with(window, full)
    kinds = {k: tuple(LineKind[n] for n in v) for k, v in KINDS.items()}
    lists = {}

    def lines(rule, kurucz_iso):
        if (rule, kurucz_iso) not in lists:
            lists[(rule, kurucz_iso)] = read_basicatlas_lines(wb, we, LineSelection(rule), kurucz_isotopes=kurucz_iso)
        return lists[(rule, kurucz_iso)]

    def run(ll, opts, g=gas, per_class=True):
        op = LineOpacity(ll, grid, LineOpacityOptions(**opts))
        if not per_class:
            v = op(g, cont).numpy()
            return v, None
        res, tot = {}, 0.0
        for k in CLASSES:
            v = op(g, cont, kinds=kinds[k]).numpy()
            tot = tot + v
            res[k] = deviation_stats(v, refs[k], c) if (refs[k].max() > 0 or v.max() > 0) else None
        res["total"] = deviation_stats(tot, full, c)
        return tot, res

    kopt = dict(const=KURUCZ, **KURUCZ_OPTIONS)
    out = {"lines_read": {}}
    tot, res = run(lines("kurucz", True), kopt)
    res["spectrum"] = flux_stats(base, spectrum_with(window, tot))
    out["kurucz"] = res
    tot, res = run(lines("wide", False), {})
    res["spectrum"] = flux_stats(base, spectrum_with(window, tot))
    out["default"] = res
    out["toggles"] = {}
    for name, t in TOGGLES.items():
        ll = lines("wide" if t == "wide" else "kurucz", t != "isotopes")
        o = dict(kopt)
        if isinstance(t, dict):
            o.update({k: (CODATA2018 if v == "CODATA2018" else v) for k, v in t.items()})
        v, _ = run(ll, o, per_class=False)
        s = flux_stats(base, spectrum_with(window, v))
        out["toggles"][name] = {k: s[k] for k in ("normflux:abs", "normflux:p99", "normflux:rms_abs")}
    from atlas_torch.v2.eos import ChemicalEquilibrium
    from atlas_torch.v2.io import read_atlas_deck
    deck = read_atlas_deck(MODEL)
    pops = ChemicalEquilibrium().solve(deck.atmosphere, deck.composition)
    tot, res = run(lines("wide", False), {}, g=GasState.from_populations(deck.atmosphere, pops))
    res["spectrum"] = flux_stats(base, spectrum_with(window, tot))
    out["v2_eos"] = res
    out["lines_read"] = {f"{r}{'_kurucz_isotopes' if k else ''}": len(v) for (r, k), v in lists.items()}

    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    allres[window] = out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(allres, indent=1, sort_keys=True) + "\n")
    gates = _gates()
    for mode in ("kurucz",):
        fails = {k: (out[mode]["spectrum"][k], t) for k, t in gates.items() if not out[mode]["spectrum"][k] <= t}
        assert not fails, (mode, fails)
    for k in ("atomic", "molecular", "hydrogen"):
        r = out["kurucz"][k]
        if r is not None and r["points_compared"]:
            assert r["rel:median"] < 3e-3 and abs(r["total_ratio"] - 1.0) < 5e-3, (k, r)
