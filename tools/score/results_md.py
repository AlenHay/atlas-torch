"""Rewrite the auto-generated scoreboard block of RESULTS.md (text outside the markers is hand-written)."""
from pathlib import Path

BEGIN, END = "<!-- SCOREBOARD:BEGIN -->", "<!-- SCOREBOARD:END -->"


def write(repo: Path, rec: dict, tols: dict):
    p = repo / "RESULTS.md"
    txt = p.read_text() if p.exists() else f"# Results\n\n{BEGIN}\n{END}\n"
    if BEGIN not in txt:
        txt += f"\n{BEGIN}\n{END}\n"
    L = [f"Last `make score`: {rec['time']} at commit `{rec['commit']}`{' (dirty)' if rec.get('dirty') else ''}"
         + (f" ({rec['note']})" if rec.get("note") else ""), ""]
    if "tests" in rec:
        t = rec["tests"]
        L.append(f"Fast tier: {t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped ({rec['test_seconds']} s)")
        L.append("")
        L.append("| variant | routines ported | passing |")
        L.append("|---|---|---|")
        for v, r in sorted(rec.get("routines", {}).items()):
            L.append(f"| {v} | {r['ported']} | {r['passing']} |")
        L.append("")
    L.append("| stage | status | worst quantity (error / tolerance) | time s | peak RSS MB |")
    L.append("|---|---|---|---|---|")
    for s, v in rec["stages"].items():
        worst = ""
        if "metrics" in v and s in tols:
            ratios = {k: (v["metrics"].get(k, float("inf")) / t["tol"] if t["tol"] > 0 else
                          (0.0 if v["metrics"].get(k, 1) == 0 else float("inf")), k) for k, t in tols[s].items()}
            r, k = max(ratios.values())
            worst = f"{k}: {v['metrics'].get(k, float('nan')):.3g} / {tols[s][k]['tol']:.3g}"
        L.append(f"| {s} | {v['status']} | {worst} | {v.get('seconds', '')} | {v.get('peak_rss_mb', '')} |")
    block = BEGIN + "\n" + "\n".join(L) + "\n" + END
    pre, rest = txt.split(BEGIN, 1)
    post = rest.split(END, 1)[1]
    p.write_text(pre + block + post)
