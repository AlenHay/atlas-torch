"""Gradients demo (atlas_torch.diff, M8): derivatives of a SYNTHE spectrum, and of the converged ATLAS9 model.

    scripts/dx '.venv/bin/python examples/gradients_demo.py [--model RUN_DIR] [--window 500 501] [--atlas]'

1. d(F/Fcont)/d[Fe], d/d[Mg], d/d vturb on a short window by forward-mode AD (one JVP each), checked against a
   central finite difference of the same differentiable function, and a Newton step on [Fe]: given a "target"
   spectrum made at [Fe]=+0.05, one Gauss-Newton step from [Fe]=0 recovers the offset.
2. With --atlas: d(T(tau))/d(Teff, log g, [M/H]) of the converged model by implicit differentiation (FD Jacobian
   of one ATLAS9 iteration: ~5-30 min depending on the machine; cached next to the model) and the chained
   d(F/Fcont)/dTeff.
Writes gradients_demo.png if matplotlib is available.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ext = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ext / "runs/stock/atlas/sun"), help="ATLAS run dir (with output_synthe.out)")
    ap.add_argument("--window", nargs=2, type=float, default=[500.0, 501.0])
    ap.add_argument("--atlas", action="store_true", help="also the ATLAS9 implicit derivatives (slow)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    from atlas_torch.diff import SyntheDiff

    t0 = time.perf_counter()
    sd = SyntheDiff(a.model, *a.window)
    print(f"setup (line lists, decisions): {time.perf_counter() - t0:.1f} s, {sd.wave.numel()} points")
    t0 = time.perf_counter()
    J = sd.jacobian({}, ["abund:Fe", "abund:Mg", "vturb"])
    print(f"3 JVPs: {time.perf_counter() - t0:.1f} s")
    base = J["_primal"]["norm"]
    for k in ("abund:Fe", "abund:Mg", "vturb"):
        g = J[k]["norm"]
        print(f"  max |d(F/Fc)/d {k}| = {g.abs().max():.4f}")
    # FD check of one derivative
    h = 1e-5
    with torch.no_grad():
        fd = (sd({"abund": {"Fe": h}})["norm"] - sd({"abund": {"Fe": -h}})["norm"]) / (2 * h)
    g = J["abund:Fe"]["norm"]
    print(f"  [Fe]: max|AD-FD|/max|FD| = {float((g - fd).abs().max() / fd.abs().max()):.2e}")
    # one Gauss-Newton step on [Fe]
    with torch.no_grad():
        target = sd({"abund": {"Fe": 0.05}})["norm"]
    r = target - base
    step = float((g * r).sum() / (g * g).sum())
    print(f"  Gauss-Newton step from [Fe]=0 towards a [Fe]=+0.05 spectrum: {step:+.4f} dex")

    chained = None
    if a.atlas:
        from atlas_torch.diff import ChainDiff
        ch = ChainDiff(a.model, a.window, cache_dir=ext / "scratch" / "m8")
        t0 = time.perf_counter()
        ch.prepare(log=print)
        print(f"ATLAS9 fixed point + FD Jacobian: {time.perf_counter() - t0:.0f} s")
        tg = ch.structure_tangents()
        n = ch.fp.n
        for th in ("teff", "logg", "mh"):
            print(f"  dT/d{th} at layers 1, 36, {n}:", np.round(tg[th]["T"][[0, 35, n - 1]], 3))
        chained = ch.spectrum_derivatives(("teff",))["teff"]["norm"]
        print(f"  max |d(F/Fc)/dTeff| = {chained.abs().max():.3e} per K")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = 2 if chained is not None else 1
        fig, ax = plt.subplots(k + 1, 1, figsize=(10, 3 * (k + 1)), sharex=True)
        w = sd.wave.numpy()
        ax[0].plot(w, base.numpy(), lw=0.7)
        ax[0].set_ylabel("F / Fcont")
        ax[1].plot(w, J["abund:Fe"]["norm"].numpy(), lw=0.7, label="d/d[Fe]")
        ax[1].plot(w, J["abund:Mg"]["norm"].numpy(), lw=0.7, label="d/d[Mg]")
        ax[1].plot(w, J["vturb"]["norm"].numpy(), lw=0.7, label="d/d vturb [per km/s]")
        ax[1].legend(fontsize=8)
        if chained is not None:
            ax[2].plot(w, chained.numpy(), lw=0.7)
            ax[2].set_ylabel("d/dTeff [1/K]")
        ax[-1].set_xlabel("vacuum wavelength [nm]")
        fig.tight_layout()
        fig.savefig("gradients_demo.png", dpi=120)
        print("wrote gradients_demo.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
