"""ODF stage checks (slow): the v2 ODF gas against xnfpdf.dat, the v2 line spectrum against the exact port's
DFSYNTHE BUFFER (B1), and one block of ODF steps against the stock p00big/p00lit tables (B7), from the same stock
xnfpdf.dat inputs. Numbers to results/v2/odf_validation.json."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from _odf_oracle import exact_port_buffer, odf_gas_from_xnfpdf

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
OUT = Path(__file__).resolve().parents[2] / "results" / "v2" / "odf_validation.json"


def _update(key, value):
    doc = json.loads(OUT.read_text()) if OUT.exists() else {}
    doc[key] = value
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _stats(d):
    a = np.abs(d)
    return {"max": float(a.max()), "rms": float(np.sqrt((d ** 2).mean())), "p99": float(np.quantile(a, 0.99)),
            "median": float(np.median(d))}


@pytest.mark.stage
def test_line_spectrum_vs_exact_port_buffer():
    """B1: 5012 K (iteff 21), 500-510 nm, 25 pressures, SpectrumOptions.kurucz() and Kurucz constants on the stock
    XNFDF gas. The packed-list lines (TAPE12) against the exact port's BUFFER minus its XLINOP part, where the
    TAPE12 lines exceed the floor and XLINOP's part; the NLTE-list part (where it dominates) and the floored total
    recorded (nlte_wings='walk': XLINOP's wings across the grid)."""
    from atlas_torch.v2.constants import KURUCZ
    from atlas_torch.v2.odf import DF_GRID, DfLineSource, LineSpectrum, SpectrumOptions, nlte_lines
    xnfpdf = EXT / "runs" / "stock" / "odf" / "solar" / "xnfpdf.dat"
    if not xnfpdf.exists():
        pytest.skip(f"missing {xnfpdf}")
    lo = int(math.ceil(math.log(500.0) / DF_GRID.step)) - DF_GRID.first
    hi = int(math.floor(math.log(510.0) / DF_GRID.step)) - DF_GRID.first
    gas = odf_gas_from_xnfpdf(xnfpdf, 21)
    opt = SpectrumOptions.kurucz()
    packed = LineSpectrum(gas, DfLineSource(temperature_limits=True), None, opt, KURUCZ)
    lp, cont = packed(lo, hi)
    nlte = LineSpectrum(gas, None, nlte_lines(), opt, KURUCZ)
    ln, _ = nlte(lo, hi)
    ref = exact_port_buffer(21, lo, hi)
    floor = ref["floor"]
    t12 = ref["lines"] - ref["xlinop"]
    sel = (t12 > floor) & (t12 > ref["xlinop"])          # TAPE12-dominated points (REAL*4 difference otherwise)
    zero = float((lp.numpy()[sel] == 0).mean())
    d_packed = np.log10(np.maximum(lp.numpy()[sel], floor[sel])) - np.log10(t12[sel])
    seln = (ref["xlinop"] > floor) & (ref["xlinop"] > t12)
    d_nlte = np.log10(np.maximum(ln.numpy()[seln], 1e-300)) - np.log10(ref["xlinop"][seln])
    tot = np.log10(np.maximum((lp + ln).numpy(), floor)) - np.log10(np.maximum(ref["lines"], floor))
    res = {"T": 5012.0, "window_nm": [500.0, 510.0], "points": int(floor.size),
           "floor_dex_max": float(np.abs(np.log10(1e-3 * cont.numpy()) - np.log10(floor)).max()),
           "packed_lists": dict(_stats(d_packed), share=float(sel.mean()), zero_share=zero),
           "nlte_list": dict(_stats(d_nlte), share=float(seln.mean())),
           "floored_total": _stats(tot), "counts": packed.counts}
    _update("B1_line_spectrum_5012K_500_510nm", res)
    assert res["floor_dex_max"] < 1e-6
    # 0.2 <= a <= 1.4 wings (MAIN: ATAB to 5 Doppler widths, then the asymptotic form with the previous line's AA)
    # give the p99 tail at the intermediate pressures
    assert abs(res["packed_lists"]["median"]) < 0.002 and res["packed_lists"]["p99"] < 0.1
    assert abs(res["nlte_list"]["median"]) < 0.002 and res["nlte_list"]["p99"] < 0.1


@pytest.mark.stage
def test_odf_gas_vs_xnfpdf():
    """B8: odf_gas(OdfConfig.kurucz()) against XNFDF's xnfpdf.dat at seven temperatures: log10 continuum (ABLOG;
    XNFDF's process set, OdfConfig.processes), n/U per NELION slot (XNFPEL) within 8 dex of each layer's largest,
    Doppler widths (DOPPLE). Bounds: the ODF gates (rms 0.01 dex on the continuum floor, 0.1 dex on line
    strengths)."""
    from atlas_torch.dfsynthe.odf_io import read_xnfpdf
    from atlas_torch.v2.io.odf_directory import read_abundance_cards
    from atlas_torch.v2.odf import OdfConfig, odf_gas
    from atlas_torch.v2.opacity.odf import ODF_LOG_T
    src = EXT / "runs" / "stock" / "odf" / "solar"
    if not (src / "xnfpdf.dat").exists():
        pytest.skip(f"missing {src}")
    cfg = OdfConfig.kurucz()
    its = (1, 6, 11, 21, 31, 45, 57)
    rows = odf_gas(read_abundance_cards(src / "xnfdf.com"), cfg, ODF_LOG_T[[i - 1 for i in its]], cfg.nodes)
    res = {}
    for g, it in zip(rows, its):
        x = read_xnfpdf(src / "xnfpdf.dat", it)
        n = x.nrhox
        dc = g.log_cont.numpy() - x.ablog.reshape(n, -1)[:, :3 * (x.nedge - 1)].astype(np.float64)
        a, b = g.n_over_u.numpy(), x.xnfpel.astype(np.float64)
        top = np.log10(np.maximum(b, 1e-300)) > np.log10(b.max(axis=1, keepdims=True)) - 8.0
        sel = top & (a > 0) & (b > 0)
        dn = np.log10(a[sel]) - np.log10(b[sel])
        dd = g.doppler.numpy()[x.dopple > 0] / x.dopple[x.dopple > 0].astype(np.float64) - 1.0
        res[str(round(float(g.T[0])))] = {
            "ablog_max": float(np.abs(dc).max()), "ablog_rms": float(np.sqrt((dc ** 2).mean())),
            "xnfpel_max": float(np.abs(dn).max()), "xnfpel_missing": int((top & ~sel).sum()),
            "dopple_rel_max": float(np.abs(dd).max())}
    _update("B8_gas_vs_xnfpdf", res)
    for r in res.values():
        assert r["ablog_rms"] < 0.01 and r["ablog_max"] < 0.03 and r["xnfpel_max"] < 0.1 and r["xnfpel_missing"] == 0
        assert r["dopple_rel_max"] < 1e-6


@pytest.mark.stage
def test_odf_block_vs_stock():
    """B7: OdfConfig.kurucz() steps of one block (490-550 nm: BIG 168-174, LITTLE 427-457) at 5012 K on the stock
    XNFDF gas against the stock p00big/p00lit, all vturbs, gated on results/v2/tolerances.json stages.odf
    (odf_v*:rms_abs, odf_v*:p999)."""
    import resource
    import time
    from atlas_torch.scoring import gate_v2
    from atlas_torch.v2.io.odf_bdf import read_bdf
    from atlas_torch.v2.odf import DfLineSource, OdfConfig, nlte_lines, segments, temperature_steps
    src = EXT / "runs" / "stock" / "odf" / "solar"
    if not (src / "xnfpdf.dat").exists():
        pytest.skip(f"missing {src}")
    cfg = OdfConfig.kurucz()
    seg = [s for s in segments(cfg.segment_points) if s.big == (168, 174)][0]
    gas = odf_gas_from_xnfpdf(src / "xnfpdf.dat", 21)
    t0, st = time.time(), {}
    out = temperature_steps(gas, cfg, DfLineSource(temperature_limits=True), nlte_lines(), st, blocks=[seg])
    m = {}
    for kind, key, sl in (("big", "big", slice(*seg.big)), ("little", "lit", slice(*seg.little))):
        for j, v in enumerate(cfg.vturbs):
            d = out[kind][j, sl].numpy() - read_bdf(src / f"p00{key}{v}.bdf")[sl, 20].astype(np.float64) / 1000.0
            a = np.abs(d)
            for k, x in (("abs", a.max()), ("rms_abs", np.sqrt((d ** 2).mean())), ("p999", np.quantile(a, 0.999))):
                m[f"odf_v{v}:{k}"] = max(m.get(f"odf_v{v}:{k}", 0.0), float(x))
            m[f"{kind}_v{v}_rms_per_layer"] = np.sqrt((d ** 2).mean(axis=(0, 2))).round(4).tolist()
    tols = json.loads((Path(__file__).resolve().parents[2] / "results/v2/tolerances.json").read_text())["stages"]["odf"]
    tols = {k: t for k, t in tols.items() if k.startswith("odf_v*:")}
    fails = gate_v2(m, tols)
    _update("B7_block_5012K_490_550nm", {"metrics": m, "fails": {k: list(v) for k, v in fails.items()},
                                         "seconds": round(time.time() - t0, 1), "counts": st,
                                         "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024})
    assert not fails, fails


@pytest.mark.stage
def test_lyman_alpha_block_10000K():
    """B9: OdfConfig.kurucz() (Lyman alpha with the impact profile, SpectrumOptions.lyman_alpha) on the stock XNFDF gas
    at 10000 K, the block holding Lyman alpha (BIG 110-113, 116-132 nm), against the stock p00big0: mean step
    difference per BIG interval. The remaining deficit is HPROF4's quasi-molecular H2/H2+ wings (not modelled);
    without Lyman alpha the 120-124 nm interval is ~1.9 dex low."""
    from atlas_torch.v2.io.odf_bdf import read_bdf
    from atlas_torch.v2.odf import DfLineSource, OdfConfig, nlte_lines, segments, temperature_steps
    src = EXT / "runs" / "stock" / "odf" / "solar"
    if not (src / "xnfpdf.dat").exists():
        pytest.skip(f"missing {src}")
    cfg = OdfConfig.kurucz(vturbs=(0,))
    seg = [s for s in segments(cfg.segment_points) if s.big[0] <= 111 < s.big[1]][0]
    out = temperature_steps(odf_gas_from_xnfpdf(src / "xnfpdf.dat", 31), cfg, DfLineSource(temperature_limits=True),
                            nlte_lines(), blocks=[seg])
    sl = slice(*seg.big)
    d = out["big"][0, sl].numpy() - read_bdf(src / "p00big0.bdf")[sl, 30].astype(np.float64) / 1000.0
    res = {f"big_{sl.start + k}": {"mean": float(d[k].mean()), "rms": float(np.sqrt((d[k] ** 2).mean())),
                                   "max": float(np.abs(d[k]).max())} for k in range(d.shape[0])}
    _update("B9_lyman_alpha_block_10000K", res)
    assert abs(res["big_111"]["mean"]) < 0.15
