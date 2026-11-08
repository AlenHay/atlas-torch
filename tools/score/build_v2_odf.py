"""Build the v2 ODF products of a BasicATLAS mixture (resumable; run at nice 19).

usage: nice -n 19 python tools/score/build_v2_odf.py --mix solar [--config kurucz|v2] [--temperatures 56,55,...]
       [--estimate] [--threads 4]
The composition comes from the stock xnfdf.com of runs/stock/odf/<mix>. Per-temperature steps go to
$ATLAS_EXT/runs/v2/odf/<config>/<mix>/.parts/t{index}.pt (hottest first unless --temperatures is given, already
present ones skipped); when all 57 exist, p00big{v}.bdf and p00lit{v}.bdf (one vturb at a time), kappa.ros
(v2 Rosseland tables from the new BIG ODFs with the matching RosselandConfig), xnfdf.com and the abundance blocks
are written to $ATLAS_EXT/runs/v2/odf/<config>/<mix>/. --estimate: time two segments at each requested temperature
and print the extrapolated cost without writing parts.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="solar")
    ap.add_argument("--config", default="kurucz", choices=["kurucz", "v2"])
    ap.add_argument("--temperatures", default=None, help="comma-separated indices 0..56")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    from atlas_torch.v2.io.odf_directory import read_abundance_cards, write_odf_directory
    from atlas_torch.v2.odf import OdfConfig, assemble, build_odf
    from atlas_torch.v2.opacity.odf import ODF_LOG_T
    from atlas_torch.v2.opacity.rosseland import RosselandConfig, RosselandOpacity

    torch.set_num_threads(a.threads)
    cfg = OdfConfig.kurucz() if a.config == "kurucz" else OdfConfig()
    comp = read_abundance_cards(EXT / "runs/stock/odf" / a.mix / "xnfdf.com")
    out = EXT / "runs/v2/odf" / a.config / a.mix
    idx = [int(i) for i in a.temperatures.split(",")] if a.temperatures else list(range(len(ODF_LOG_T)))[::-1]
    if a.estimate:
        return estimate(comp, cfg, idx)
    res = build_odf(comp, cfg, idx, workdir=out / ".parts", verbose=True)
    if "big" not in res:
        print(f"{len(list((out / '.parts').glob('t*.pt')))} of {len(ODF_LOG_T)} temperatures built")
        return
    rcfg = RosselandConfig.kurucz() if a.config == "kurucz" else RosselandConfig()
    table = RosselandOpacity(res["big"], rcfg).tabulate(comp)
    write_odf_directory(out, comp, big=res["big"], rosseland=table)
    for v in cfg.vturbs:
        assemble(out / ".parts", "little", (v,)).write(out)
    (out / "provenance_v2.json").write_text(json.dumps({"config": cfg.describe(), "rosseland": rcfg.describe(),
                                                        "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    print(f"written {out}")


def estimate(comp, cfg, idx):
    """B6: wall time, profile points and peak RSS of two segments per temperature, extrapolated to the grid."""
    from atlas_torch.v2.odf import DfLineSource, LineSpectrum, nlte_lines, odf_gas, segments
    from atlas_torch.v2.opacity.odf import ODF_LOG_T
    seg = segments(cfg.segment_points)
    pick = [seg[len(seg) // 3], seg[2 * len(seg) // 3]]
    src, nl = DfLineSource(temperature_limits=cfg.temperature_limits), nlte_lines()
    rows = {}
    for i in idx:
        t0 = time.time()
        g = odf_gas(comp, cfg, ODF_LOG_T[i:i + 1], cfg.nodes)[0]
        tg = time.time() - t0
        spec = LineSpectrum(g, src, nl, cfg.spectrum, cfg.const)
        t1 = time.time()
        for s in pick:
            spec(s.begin, s.end)
        dt = time.time() - t1
        pts = sum(s.end - s.begin + 1 for s in pick)
        rows[i] = {"T": round(float(10 ** ODF_LOG_T[i])), "gas_s": round(tg, 1), "segments_s": round(dt, 1),
                   "points_sampled": pts, "profile_points": spec.counts["points"], "line_pairs": spec.counts["pairs"],
                   "extrapolated_s": round(tg + dt * 3507859 / pts, 0),
                   "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}
        print(json.dumps(rows[i]), flush=True)
    return rows


if __name__ == "__main__":
    main()
