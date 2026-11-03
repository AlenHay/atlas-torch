"""Execute BasicATLAS's examples/*.ipynb code cells against a backend (M6).

usage: python scripts/run_examples.py <backend> [notebook ...]
  backend = stock       : BasicATLAS's own atlas.py driving the stock Fortran build
            torch       : atlas_torch.basicatlas (drop-in API, torch backend)
Cells run in order with HOME=$ATLAS_EXT/runs/<backend>/examples/<notebook> (the notebooks write to ~/...),
cwd = the BasicATLAS examples/ dir (they read ../data/...), matplotlib Agg, IPython magics translated
(`rm -rf` line magics run via shutil; %matplotlib skipped). Outputs stay in that HOME.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
BA = EXT / "BasicATLAS"


def cells(nb):
    for c in json.loads(Path(nb).read_text())["cells"]:
        if c["cell_type"] == "code":
            yield "".join(c["source"])


def translate(src, home):
    out = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("%"):
            continue
        if s.startswith("rm -rf "):
            out.append(f"__import__('shutil').rmtree(__import__('os').path.expanduser({s[7:].strip()!r}), ignore_errors=True)")
            continue
        out.append(ln)
    return "\n".join(out)


def main():
    backend = sys.argv[1]
    names = sys.argv[2:] or ["sun_model", "output", "custom_abun"]
    import matplotlib
    matplotlib.use("Agg")
    if backend == "stock":
        sys.path.insert(0, str(BA))
    elif backend == "torch":
        sys.path.insert(0, str(REPO / "src" / "atlas_torch" / "compat"))   # provides `atlas`
        sys.path.insert(1, str(REPO / "src"))
    else:
        raise SystemExit("backend?")
    report = {}
    for name in names:
        home = EXT / "runs" / backend / "examples" / name
        if home.exists():
            shutil.rmtree(home)
        home.mkdir(parents=True)
        os.environ["HOME"] = str(home)
        os.chdir(BA / "examples")
        g = {"__name__": "__main__"}
        t0 = time.time()
        for i, src in enumerate(cells(BA / "examples" / f"{name}.ipynb")):
            code = translate(src, home)
            print(f"--- {name} cell {i}\n{code}", flush=True)
            exec(compile(code, f"{name}[{i}]", "exec"), g)
            import matplotlib.pyplot as plt
            plt.close("all")
        report[name] = time.time() - t0
        print(f"=== {name} done in {report[name]:.0f}s", flush=True)
    (EXT / "runs" / backend / "examples" / "timing.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
