"""Finite-difference validation of atlas_torch.diff (used by tests/stage/test_diff_*.py and
examples/gradients_demo.py; `python -m atlas_torch.diff.validate --help`).

Statistics reported for a derivative d (AD) against a central difference f (FD) over a spectrum:
  max_rel  = max_k |d_k - f_k| / max_k |f_k|     (worst point, normalised by the largest derivative)
  p99_rel  = 99th percentile of |d_k - f_k| / max|f|
  rms_rel  = ||d - f||_2 / ||f||_2
Two kinds of checks:
  * "diff":  FD of the differentiable function itself with the base point's discrete decisions frozen
             (SYNTHE cutoffs, Balmer-wing walks, RT interval choices). Tests the derivatives. Remaining
             outliers are points next to a kink of Kurucz's blended-parabola interpolation (C0, not C1); they
             disappear as the step shrinks (`refine` re-checks the worst points with a 10x smaller step).
  * "exact": FD of the exact path (the byte-identical port: REAL*4 tables, decisions re-taken, abundances
             through the deck's printed cards) with steps large enough to average over its staircase.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import time
from pathlib import Path

import numpy as np
import torch

D = torch.float64


def stats(ad, fd) -> dict:
    ad = np.asarray(ad.detach().cpu() if isinstance(ad, torch.Tensor) else ad, dtype=np.float64)
    fd = np.asarray(fd.detach().cpu() if isinstance(fd, torch.Tensor) else fd, dtype=np.float64)
    e = np.abs(ad - fd)
    scale = max(np.max(np.abs(fd)), 1e-300)
    return {"max_rel": float(e.max() / scale), "p99_rel": float(np.percentile(e, 99) / scale),
            "rms_rel": float(np.sqrt((e ** 2).mean()) / max(np.sqrt((fd ** 2).mean()), 1e-300)),
            "max_abs_derivative": float(scale), "argmax": int(e.argmax())}


# ------------------------------------------------------------------------------------- parameter steps

def perturbed(sd, name: str, h: float):
    """params with parameter `name` moved by h (names as in SyntheDiff.jacobian)."""
    if name.startswith("abund:"):
        return {"abund": {name[6:]: torch.tensor(h, dtype=D)}}
    if name == "mh":
        return {"mh": torch.tensor(h, dtype=D)}
    if name == "vturb":
        return {"vturb": torch.tensor(sd.vturb0 + h, dtype=D)}
    q, j = name[:-1].split("[")
    key = {"T": "temp.t", "P": "state.p", "XNE": "state.xne", "RHOX": "rhox.rhox"}[q]
    v = sd.base.deck(key)
    v[int(j)] = v[int(j)] + h
    return {q: v}


def default_step(sd, name: str) -> float:
    """Steps small enough to avoid most interpolation kinks, large enough for float64 round-off."""
    if name.startswith("abund:") or name == "mh":
        return 1e-5
    if name == "vturb":
        return 1e-5
    q, j = name[:-1].split("[")
    key = {"T": "temp.t", "P": "state.p", "XNE": "state.xne", "RHOX": "rhox.rhox"}[q]
    return 1e-7 * float(sd.base.deck(key)[int(j)])


def check_diff(sd, names, key: str = "norm", steps: dict | None = None, refine: bool = True) -> dict:
    """AD (forward mode) vs central FD of the differentiable spectrum, decisions frozen."""
    out = {}
    for name in names:
        h = (steps or {}).get(name, default_step(sd, name))
        t0 = time.perf_counter()
        J = sd.jacobian({}, [name])
        ad = J[name][key]
        t_ad = time.perf_counter() - t0
        with torch.no_grad():
            fp = sd(perturbed(sd, name, h))[key]
            fm = sd(perturbed(sd, name, -h))[key]
        fd = (fp - fm) / (2 * h)
        st = stats(ad, fd)
        st.update(step=h, seconds_ad=t_ad)
        if refine and st["max_rel"] > 1e-4:
            # worst points: recheck with a 10x smaller step (kinks of the C0 interpolants move out of range)
            k = st["argmax"]
            with torch.no_grad():
                f2 = (sd(perturbed(sd, name, h / 10))[key] - sd(perturbed(sd, name, -h / 10))[key]) / (h / 5)
            fd_ref = fd.clone()
            better = (f2 - ad).abs() < (fd - ad).abs()
            fd_ref[better] = f2[better]
            st["refined"] = stats(ad, fd_ref)
            st["refined"]["worst_point_ad_vs_fd_h/10"] = float(abs(ad[k] - f2[k]) / max(abs(float(fd.abs().max())), 1e-300))
        out[name] = st
    return out


# ------------------------------------------------------------------------------------- the exact path

def _deck_with(deck_text: str, abund: dict | None = None, mh: float = 0.0) -> str:
    """SYNTHE deck with ABUNDANCE CHANGE values moved by abund {Z: dex} (printed to 0.01 dex, as BasicATLAS'
    cards; the H/He renormalisation is below their 5-decimal print precision for the steps used) and
    ABUNDANCE SCALE times 10**mh."""
    from .xnf import zof
    lines = deck_text.split("\n")
    ab = {zof(k): v for k, v in (abund or {}).items()}
    for i, ln in enumerate(lines):
        if ln.startswith("ABUNDANCE SCALE") and mh:
            m = re.match(r"(ABUNDANCE SCALE\s+)(\S+)(.*)", ln)
            lines[i] = "ABUNDANCE SCALE {:>9.6f}{}".format(float(m.group(2)) * 10 ** mh, m.group(3))
        if "ABUNDANCE CHANGE" in ln and ab:
            def rep(mm):
                z = int(mm.group(1))
                v = float(mm.group(2))
                if z in ab and z >= 3:
                    v = v + ab[z]
                    return "{:>3d}{:>7.2f}".format(z, v)
                return mm.group(0)
            head, tail = ln.split("ABUNDANCE CHANGE", 1)
            if not head.startswith("ABUNDANCE SCALE"):
                lines[i] = head + "ABUNDANCE CHANGE" + re.sub(r"\s*(\d+)\s+(-?\d+\.\d+)", rep, tail)
    return "\n".join(lines)


def exact_spectrum(deck_text: str, linedir, workdir, turbv: float | None = None) -> dict:
    """The exact path (atlas_torch.synthe: xnfpelsyn -> synthe -> spectrv, stock libm emulation) on the line
    lists of `linedir` (fort.12/fort.19/fort.93, e.g. a SyntheDiff work directory); TURBV patched in fort.93
    when given. Returns flux, cont (SPECTRV HNU)."""
    from ..synthe import spectrv_prog as sp
    from ..synthe.chain import FORT25, basicatlas_dir
    from ..synthe.synthe_prog import SyntheProgram
    from ..synthe.xnfpelsyn_prog import xnfpelsyn
    wd = Path(workdir)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    for f in ("fort.12", "fort.19", "fort.93", "fort.20"):
        src = Path(linedir) / f
        if src.exists():
            shutil.copy(src, wd / f)
    if turbv is not None:
        b = bytearray((wd / "fort.93").read_bytes())
        struct.pack_into("<f", b, 4 + 20, float(turbv))
        (wd / "fort.93").write_bytes(bytes(b))
    s_files = basicatlas_dir() / "data" / "synthe_files"
    xr = xnfpelsyn(deck_text, out10=wd / "fort.10")
    os.symlink(s_files / "he1tables.dat", wd / "fort.18")
    SyntheProgram(wd).run(write=True, delete_inputs=False)
    (wd / "fort.25").write_text(FORT25)
    res = sp.spectrv(xr.readin_state, sp.load_inputs(wd), libm="libmvec")
    q = res.q
    out = {"flux": q[:, 0].copy(), "cont": q[:, 1].copy()}
    shutil.rmtree(wd)
    return out


def check_exact(sd, cases: dict, key: str = "norm", scratch=None) -> dict:
    """AD of `sd` (profile='exact') vs central FD of the exact path. cases: {name: step} with names
    "abund:<El>" (dex, multiple of 0.01), "mh" (dex), "vturb" (km/s)."""
    from .xnf import zof
    scratch = Path(scratch or (sd.wd.parent / "exact_fd"))
    deck = sd.deck.read_text()
    out = {}
    for name, h in cases.items():
        J = sd.jacobian({}, [name])
        ad = J[name][key]
        ev = []
        for sg in (1, -1):
            if name.startswith("abund:"):
                r = exact_spectrum(_deck_with(deck, {zof(name[6:]): sg * h}), sd.wd, scratch)
            elif name == "mh":
                r = exact_spectrum(_deck_with(deck, mh=sg * h), sd.wd, scratch)
            elif name == "vturb":
                r = exact_spectrum(deck, sd.wd, scratch, turbv=sd.vturb0 + sg * h)
            else:
                raise KeyError(name)
            ev.append(r["flux"] / r["cont"] if key == "norm" else r[key])
        fd = (ev[0] - ev[1]) / (2 * h)
        st = stats(ad, fd)
        st["step"] = h
        out[name] = st
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
                                           / "runs/stock/atlas/sun"))
    ap.add_argument("--window", nargs=2, type=float, default=[500.0, 510.0])
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None, help="JSON file for the results")
    ap.add_argument("--what", nargs="+", default=["diff_smooth", "diff_exact", "exact"])
    a = ap.parse_args(argv)
    from .spectrum import SyntheDiff
    torch.set_num_threads(int(os.environ.get("ATLAS_TORCH_THREADS", "4")))
    res = {"model": a.model, "window": a.window, "load": os.getloadavg()}
    names = ["abund:Fe", "abund:Mg", "abund:C", "mh", "vturb", "T[30]", "T[45]", "P[40]", "RHOX[40]", "XNE[40]"]
    for what in a.what:
        t0 = time.perf_counter()
        if what in ("diff_smooth", "diff_exact"):
            sd = SyntheDiff(a.model, *a.window, workdir=a.workdir, profile=what.split("_")[1], polish=True)
            sel = names if what == "diff_smooth" else ["abund:Fe", "abund:Mg", "abund:C", "mh"]
            res[what] = check_diff(sd, sel)
        elif what == "exact":
            sd = SyntheDiff(a.model, *a.window, workdir=a.workdir, profile="exact", polish=False)
            res[what] = check_exact(sd, {"abund:Fe": 0.05, "abund:Mg": 0.05, "mh": 0.02, "vturb": 0.05})
        res[what + "_seconds"] = time.perf_counter() - t0
        print(json.dumps({what: res[what]}, indent=1), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
