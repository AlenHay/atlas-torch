"""M7 study report: summary tables (results/m7/gpu_summary.json, markdown on stdout) and plots
results/plots/m7_accuracy.png, m7_speed.png, m7_compile.png from results/m7/gpu_float32_compile.json.

  tools/bench/gpu_run.sh cpu tools/bench/gpu_report.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_common as G  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ACC_MODES = ["exact", "native", "gpu64", "cpu32", "gpu32", "cpu32p", "gpu32p"]
MODE_LABEL = {"fortran": "Fortran", "exact": "CPU exact", "native": "CPU f64 native", "gpu64": "GPU f64",
              "cpu32": "CPU f32", "gpu32": "GPU f32", "cpu32p": "CPU f32-prec", "gpu32p": "GPU f32-prec",
              "gpu_lines": "GPU lines", "exact_noftz": "CPU exact, no FTZ"}
# categorical slots of the reference palette, fixed order (dataviz references/palette.md)
SERIES = {"fortran": "#52514e", "exact": "#2a78d6", "native": "#1baf7a", "gpu64": "#eb6834", "gpu32": "#e87ba4",
          "gpu_lines": "#eda100", "cpu32": "#6250d6"}
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def load():
    d = json.loads((G.RESULTS / "gpu_float32_compile.json").read_text())
    for p in sorted((G.SCRATCH / "synthe").glob("w*/fortran.json")):     # Fortran per-program timings
        f = json.loads(p.read_text())
        d[f"synthe/{p.parent.name}/fortran"] = {k: {"seconds": v, "status": "ok"} for k, v in f["fortran_seconds"].items()}
    return d


def rows(d):
    """[(label, family, {mode: {"ratio", "pass", "seconds", "status"}})] in display order."""
    out = []
    for w in range(4):
        for prog in ("xnfpelsyn", "synthe", "spectrv", "chain"):
            cells = {}
            for m in ACC_MODES + ["fortran"]:
                r = d.get(f"synthe/w{w}/{m}", {}).get(prog)
                if r:
                    g = r.get("gate", {})
                    cells[m] = {"ratio": g.get("worst", [None, None])[1], "pass": g.get("pass"),
                                "seconds": r.get("seconds"), "status": r.get("status")}
            out.append((f"SYNTHE w{w} {prog}", "synthe", cells))
    for mix in ("solar", "alpha"):
        cells = {}
        for m in ACC_MODES + ["fortran"]:
            r = d.get(f"kappa9/{mix}/{m}")
            if r:
                g = r.get("gate", {})
                cells[m] = {"ratio": (g.get("worst") or [None, None])[1], "pass": g.get("pass"),
                            "seconds": r.get("seconds"), "status": r.get("status")}
        if cells:
            out.append((f"KAPPA9 {mix}", "kappa9", cells))
    for model in ("sun", "t4500g25", "t6500g40m10", "t9000g40", "t5000g45m20", "t5500g40alpha"):
        cells = {}
        for m in ACC_MODES + ["fortran"]:
            r = d.get(f"atlas/{model}/{m}")
            if r and "stopped_after" not in r:
                g = r.get("gate", {})
                cells[m] = {"ratio": (g.get("worst") or [None, None])[1], "pass": g.get("pass"),
                            "seconds": r.get("seconds"), "status": r.get("status")}
        if cells:
            out.append((f"ATLAS9 {model}", "atlas9", cells))
    for t in ("t21", "t33"):
        cells = {}
        for m in ACC_MODES + ["fortran", "gpu_lines", "exact_noftz"]:
            r = d.get(f"dfsynthe/solar/{t}/{m}")
            if r:
                mt = r.get("metrics", {})
                cells[m] = {"ratio": mt.get("worst_ratio_vs_table_gates"), "pass": mt.get("within_table_gates"),
                            "seconds": r.get("seconds"), "status": r.get("status"),
                            "max_steps": max((mt.get(f"odf_v{v}:abs", 0) for v in (0, 1, 2, 4, 8)), default=None)}
        if cells:
            T = {"t21": "5012 K", "t33": "12589 K"}[t]
            out.append((f"DFSYNTHE {T}", "dfsynthe", cells))
    return out


def fmt_ratio(c):
    if c is None:
        return "—"
    if c.get("status") not in (None, "ok"):
        return {"fail_overflow": "overflow", "crash": "crash", "killed": "killed"}.get(c["status"], c["status"])
    r = c.get("ratio")
    if r is None:
        return "—"
    if r == 0:
        return "0"
    if math.isinf(r):
        return "∞"
    return f"{r:.2g}"


def markdown(rs):
    lines = ["| stage / program | " + " | ".join(MODE_LABEL[m] for m in ACC_MODES) + " |",
             "|---|" + "---|" * len(ACC_MODES)]
    for label, _, cells in rs:
        vals = []
        for m in ACC_MODES:
            c = cells.get(m)
            s = fmt_ratio(c)
            if c and c.get("pass") is True:
                s += " ✓"
            elif c and c.get("pass") is False:
                s += " ✗"
            vals.append(s)
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def speed_markdown(rs):
    ms = ["fortran", "exact", "native", "gpu64", "gpu32", "gpu_lines", "exact_noftz"]
    lines = ["| stage / program | " + " | ".join(MODE_LABEL[m] for m in ms) + " |", "|---|" + "---|" * len(ms)]
    for label, _, cells in rs:
        v = [(f"{cells[m]['seconds']:.2f}" if cells.get(m, {}).get("seconds") else "—") for m in ms]
        if all(x == "—" for x in v):
            continue
        lines.append(f"| {label} | " + " | ".join(v) + " |")
    return "\n".join(lines)


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_accuracy(rs, path):
    modes = ACC_MODES
    n = len(rs)
    M = np.full((n, len(modes)), np.nan)
    for i, (_, _, cells) in enumerate(rs):
        for j, m in enumerate(modes):
            c = cells.get(m)
            if not c:
                continue
            if c.get("status") not in (None, "ok"):
                M[i, j] = 3.0
                continue
            r = c.get("ratio")
            if r is None:
                continue
            M[i, j] = -3.0 if r == 0 else float(np.clip(np.log10(r) if np.isfinite(r) else 3.0, -3, 3))
    # diverging: blue (well inside the gate) -> neutral gray at ratio 1 -> red (outside)
    cmap = LinearSegmentedColormap.from_list("div", ["#256abf", "#86b6ef", "#f0efec", "#ec835a", "#c0392b"])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(10.5, 0.32 * n + 1.6), dpi=130)
    fig.patch.set_facecolor(SURF)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-3, vmax=3, aspect="auto")
    for i, (_, _, cells) in enumerate(rs):
        for j, m in enumerate(modes):
            c = cells.get(m)
            s = fmt_ratio(c)
            if c and c.get("pass") is True:
                s += " ✓"
            elif c and c.get("pass") is False:
                s += " ✗"
            dark = not np.isnan(M[i, j]) and abs(M[i, j]) > 2.2
            ax.text(j, i, s, ha="center", va="center", fontsize=7, color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(modes)), [MODE_LABEL[m] for m in modes], fontsize=8, color=INK)
    ax.set_yticks(range(n), [r[0] for r in rs], fontsize=7.5, color=INK)
    ax.xaxis.tick_top()
    for i in range(1, n):
        if rs[i][1] != rs[i - 1][1]:
            ax.axhline(i - 0.5, color=INK2, lw=0.8)
    ax.set_xticks(np.arange(-0.5, len(modes)), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURF, lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_ticks([-3, -2, -1, 0, 1, 2, 3], labels=["≤1e-3 / 0", "0.01", "0.1", "1 (gate)", "10", "100", "≥1e3"])
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Worst error / frozen gate per stage and mode (✓ within all gates, ✗ outside; 0 = identical "
                 "to stock; ∞ = nonzero error on a zero-width gate)\nDFSYNTHE rows: single temperature vs the 57-temperature "
                 "table gates (orientation only); — = not run",
                 fontsize=9, color=INK, loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def plot_speed(rs, path):
    pick = [r for r in rs if r[0] in ("SYNTHE w0 xnfpelsyn", "SYNTHE w0 synthe", "SYNTHE w0 spectrv",
                                      "SYNTHE w1 synthe", "SYNTHE w3 synthe", "SYNTHE w3 chain", "KAPPA9 solar",
                                      "ATLAS9 sun", "DFSYNTHE 5012 K", "DFSYNTHE 12589 K")]
    ms = ["fortran", "exact", "native", "gpu64", "gpu32", "gpu_lines"]
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(pick) + 1.4), dpi=130)
    fig.patch.set_facecolor(SURF)
    _style(ax)
    h = 0.8 / len(ms)
    for j, m in enumerate(ms):
        ys, xs = [], []
        for i, (_, _, cells) in enumerate(pick):
            s = cells.get(m, {}).get("seconds")
            if s:
                ys.append(i - 0.4 + (j + 0.5) * h)
                xs.append(s)
        ax.barh(ys, xs, height=h * 0.85, color=SERIES[m], label=MODE_LABEL[m], edgecolor=SURF, linewidth=0.6)
        for y, x in zip(ys, xs):
            ax.text(x * 1.06, y, f"{x:.3g}", va="center", fontsize=6, color=INK2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(pick)), [p[0] for p in pick], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("wall time [s], one CPU thread for CPU work (log scale)", fontsize=8, color=INK2)
    ax.legend(fontsize=7.5, frameon=False, ncol=len(ms), loc="upper center", bbox_to_anchor=(0.45, 1.06))
    ax.set_title("Time per program and mode (SYNTHE: best of 2; GPU = AMD RDNA3 8 GB, ROCm 6.4)", fontsize=9,
                 color=INK, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def plot_compile(d, path):
    ks = ["spectrv_josh", "atlas9_josh", "synthe_voigt", "dfsynthe_chunk"]
    names = {"spectrv_josh": "SPECTRV josh_batch", "atlas9_josh": "ATLAS9 josh", "synthe_voigt": "SYNTHE voigt",
             "dfsynthe_chunk": "DFSYNTHE line chunk"}
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=130)
    fig.patch.set_facecolor(SURF)
    _style(ax)
    devs = [("cpu", "CPU inductor (torch 2.15)", "#2a78d6"), ("cuda", "GPU inductor/triton (torch 2.9 ROCm)", "#eb6834")]
    h = 0.38
    for j, (dv, lab, col) in enumerate(devs):
        for i, k in enumerate(ks):
            r = d.get(f"compile/{k}/{dv}", {})
            y = i - 0.2 + j * 0.4
            if r.get("status") == "ok" and r.get("speedup"):
                sp = r["speedup"]
                ax.barh(y, sp, height=h * 0.9, color=col, label=lab if i == 0 else None, edgecolor=SURF)
                same = r.get("compiled_vs_eager", {})
                tag = "bit-identical" if same.get("identical") else f"max rel {same.get('max_rel', 0):.1g}"
                ax.text(sp * 1.05, y, f"×{sp:.2f}  ({tag})", va="center", fontsize=7, color=INK2)
            else:
                ax.barh(y, 0, height=h * 0.9, color=col, label=lab if i == 0 else None)
                err = (r.get("error") or "").split("\n")[0].strip()
                if "earlyoom" in err:
                    err = "killed by earlyoom during compilation"
                elif "AssertionError" in err:
                    err = "dynamo AssertionError (np.broadcast_shapes in map1b)"
                ax.text(0.055, y, f"{r.get('status', 'not run')}: {err[:60]}", va="center", fontsize=7, color=INK2)
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 20)
    ax.set_yticks(range(len(ks)), [names[k] for k in ks], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("speedup of torch.compile over eager on the same device (log; dashed = 1)", fontsize=8,
                  color=INK2)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("torch.compile on captured hot-kernel inputs (native torch math)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main():
    d = load()
    rs = rows(d)
    G.PLOTS.mkdir(parents=True, exist_ok=True)
    plot_accuracy(rs, G.PLOTS / "m7_accuracy.png")
    plot_speed(rs, G.PLOTS / "m7_speed.png")
    plot_compile(d, G.PLOTS / "m7_compile.png")
    G.save_json(G.RESULTS / "gpu_summary.json", {"rows": [{"label": a, "family": b, "cells": c} for a, b, c in rs]})
    acc, spd = markdown(rs), speed_markdown(rs)
    md = G.RESULTS / "RESULTS_section_gpu.md"
    if md.exists():
        import re
        t = md.read_text()
        for tag, body in (("TABLE_ACCURACY", acc), ("TABLE_SPEED", spd)):
            t = re.sub(rf"<!-- {tag} -->.*?<!-- /{tag} -->", lambda _: f"<!-- {tag} -->\n{body}\n<!-- /{tag} -->", t,
                       flags=re.S)
        md.write_text(t)
    print(acc)
    print()
    print(spd)


if __name__ == "__main__":
    main()
