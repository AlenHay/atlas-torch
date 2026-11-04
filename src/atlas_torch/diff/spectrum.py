"""Differentiable SYNTHE spectrum of a model atmosphere: SyntheDiff and spectrum_and_jacobian.

    from atlas_torch.diff import SyntheDiff
    sd = SyntheDiff("path/to/atlas_run", 500., 510.)           # model dir (output_synthe.out) or deck path
    out = sd({})                                                # base spectrum (torch tensors)
    out["flux"], out["cont"], out["norm"], out["wave"]         # HNU (SPECTRV's flux units), nm
    J = sd.jacobian({"abund": {"Fe": 0.0}, "vturb": 1.5}, wrt=["abund:Fe", "vturb"])
    J["abund:Fe"]["norm"]                                       # d(flux/cont)/d[Fe] per dex, (LENGTH,)

Line lists (fort.12/fort.19/fort.93) come from the exact path's line programs (atlas_torch.synthe.lines,
byte-identical to BasicATLAS), prepared once in a work directory. Everything that depends on the parameters is
computed by atlas_torch.diff.{xnf, nmolec, lines, hydrogen, rt}. Derivatives are forward-mode (one JVP per
parameter direction, torch.autograd.forward_ad); see atlas_torch.diff.__init__.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.autograd.forward_ad as fwad

from . import xnf as XN
from .hydrogen import HydrogenLines
from .lines import LineSet, Window
from .rt import Spectrv

D = torch.float64


def _ext() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def _deck_path(model) -> Path:
    p = Path(model)
    if p.is_dir():
        p = p / "output_synthe.out"
    if not p.is_file():
        raise FileNotFoundError(f"SYNTHE model deck not found: {p} (run BasicATLAS synthe() once, or pass the deck)")
    return p


def prepare_workdir(workdir, wlbeg, wlend, resolu=600000.0, vturb=1.5, linelist="BasicATLAS", c12c13=None,
                    airorvac="VAC", data_dir=None) -> Path:
    """Line lists of a SYNTHE batch (synbeg -> rgfalllinesnew -> rmolecasc x 21 -> rh2ofast) in `workdir`,
    unless already there for the same card."""
    from ..synthe.chain import MOLECULAR_LISTS, _linelist_path, basicatlas_dir, c12c13_card
    from ..synthe.lines import prepare_lines
    wd = Path(workdir)
    stamp = wd / "diff_lines.txt"
    card = f"{wlbeg!r} {wlend!r} {resolu!r} {vturb!r} {linelist} {c12c13!r} {airorvac}"
    if stamp.exists() and stamp.read_text() == card and (wd / "fort.93").exists():
        return wd
    ba = Path(data_dir) if data_dir is not None else basicatlas_dir()
    s_files, d_files = ba / "data" / "synthe_files", ba / "data" / "dfsynthe_files"
    wd.mkdir(parents=True, exist_ok=True)
    iso = None
    if c12c13 is not None:
        a, b = c12c13_card(float(c12c13)).split()
        iso = (float(a), float(b))
    prepare_lines(wd, wlbeg, wlend, resolu, vturb, _linelist_path(linelist, s_files),
                  [s_files / m for m in MOLECULAR_LISTS], d_files / "h2ofastfix.bin", c12c13=iso, airorvac=airorvac)
    stamp.write_text(card)
    return wd


class SyntheDiff:
    """A SYNTHE window of one model atmosphere as a differentiable function of `params` (see diff.xnf)."""

    def __init__(self, model, wlbeg: float, wlend: float, resolu: float = 600000.0, vturb: float = 1.5,
                 workdir=None, linelist: str = "BasicATLAS", profile: str = "exact", polish: bool = False,
                 airorvac: str = "VAC", device=None):
        t0 = time.perf_counter()
        self.deck = _deck_path(model)
        self.profile, self.polish = profile, polish
        self.vturb0 = float(vturb)
        if workdir is None:
            tag = f"{self.deck.parent.name}_{wlbeg:g}_{wlend:g}_{resolu:g}_{vturb:g}"
            workdir = _ext() / "scratch" / "m8" / "work" / tag
        self.wd = prepare_workdir(workdir, wlbeg, wlend, resolu, vturb, linelist, airorvac=airorvac)
        self.base = XN.Base.from_deck(self.deck)
        self.win = Window.from_fort93(self.wd / "fort.93")
        self.lines = LineSet(self.wd, self.win, device=device)
        from ..synthe.lineopac import fortio
        l19 = fortio.read_fort19(self.wd / "fort.19", self.win.n19) if self.win.n19 > 0 else \
            np.zeros(0, fortio.FORT19)
        self.hlines = HydrogenLines(l19, self.win, self.lines.tabs, mode=profile)
        self.rt = Spectrv(self.base, self.win, device=device)
        x0 = XN.xnfpelsyn_diff(self.base, {}, polish=polish)
        self.lines.decide(x0, {"wl": self.base.wl}, vturb=self.vturb0)
        if len(self.hlines.lines):                    # records the Balmer-wing walk decisions
            self.hlines.buffer(x0, self.lines.layer_quantities(x0, self.vturb0), self.lines.cont32)
        self.wave = torch.as_tensor(self.win.wave(), dtype=D)
        self(None)                                    # records the RT decisions of the base point
        self.timings = {"setup": time.perf_counter() - t0}

    # ------------------------------------------------------------------------------------- forward
    def __call__(self, params: dict | None = None, redecide: bool = False) -> dict:
        """Spectrum for `params`. redecide=True re-takes SYNTHE's cutoff decisions at these parameters (what the
        exact path does; piecewise), else the base point's decisions are kept (smooth in params)."""
        params = dict(params or {})
        tm = {}
        t = time.perf_counter()
        x = XN.xnfpelsyn_diff(self.base, params, polish=self.polish)
        tm["xnfpelsyn"] = time.perf_counter() - t
        vturb = params.get("vturb", self.vturb0)
        if redecide:
            xd = XN.XnfResult(**{k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in x.__dict__.items()})
            self.lines.decide(xd, {"wl": self.base.wl}, vturb=float(torch.as_tensor(vturb).detach()))
            self.hlines.reset()
            self.rt.reset()
        t = time.perf_counter()
        q = self.lines.layer_quantities(x, vturb)
        buf = self.lines.buffer(x, vturb, mode=self.profile)
        tm["lines"] = time.perf_counter() - t
        t = time.perf_counter()
        if len(self.hlines.lines):
            buf = buf + self.hlines.buffer(x, q, self.lines.cont32)
        tm["hlines"] = time.perf_counter() - t
        t = time.perf_counter()
        asy = self.lines.asynth(x, buf)
        flux, cont = self.rt(x, asy)
        tm["spectrv"] = time.perf_counter() - t
        return {"wave": self.wave, "flux": flux, "cont": cont, "norm": flux / cont, "timings": tm, "xnf": x}

    # ------------------------------------------------------------------------------------- derivatives
    def jvp(self, params: dict, tangents: dict) -> dict:
        """Directional derivative of the spectrum along `tangents` {param: tangent} (param as in params:
        "T"/"P"/"XNE"/"RHOX" (n,) arrays, "vturb", "mh" scalars, "abund" {el: scalar}). Returns primal and
        tangent of flux, cont, norm."""
        base = self.base
        with fwad.dual_level():
            p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in params.items()}
            for k, tg in tangents.items():
                if k == "abund":
                    ab = dict(p.get("abund", {}))
                    for el, te in tg.items():
                        ab[el] = fwad.make_dual(torch.as_tensor(float(ab.get(el, 0.0)), dtype=D),
                                                torch.as_tensor(te, dtype=D))
                    p["abund"] = ab
                elif k in ("T", "P", "XNE", "RHOX"):
                    key = {"T": "temp.t", "P": "state.p", "XNE": "state.xne", "RHOX": "rhox.rhox"}[k]
                    v = torch.as_tensor(p[k], dtype=D) if k in p else base.deck(key)
                    p[k] = fwad.make_dual(v.clone(), torch.as_tensor(tg, dtype=D).clone())
                else:
                    v = p.get(k, self.vturb0 if k == "vturb" else 0.0)
                    p[k] = fwad.make_dual(torch.as_tensor(float(v), dtype=D), torch.as_tensor(tg, dtype=D))
            out = self(p)
            res = {"timings": out["timings"]}
            for k in ("flux", "cont", "norm"):
                u = fwad.unpack_dual(out[k])
                res[k] = u.primal.detach()
                res["d" + k] = u.tangent.detach() if u.tangent is not None else torch.zeros_like(u.primal)
        return res

    def jacobian(self, params: dict | None = None, wrt=("vturb",)) -> dict:
        """{name: {"flux"|"cont"|"norm": derivative (LENGTH,)}} for scalar parameters: "vturb", "mh",
        "abund:<El>" (per dex), "T[j]"/"P[j]"/"XNE[j]"/"RHOX[j]" (one layer, per unit), or a vector parameter
        "T"/"P"/... with the tangent given as ("T", array)."""
        params = dict(params or {})
        out = {}
        for w in wrt:
            if isinstance(w, tuple):
                name, tg = w
                r = self.jvp(params, {name: tg})
            elif w.startswith("abund:"):
                r = self.jvp(params, {"abund": {w.split(":", 1)[1]: 1.0}})
            elif "[" in w:
                name, j = w[:-1].split("[")
                tg = torch.zeros(self.base.n, dtype=D)
                tg[int(j)] = 1.0
                r = self.jvp(params, {name: tg})
            else:
                r = self.jvp(params, {w: 1.0})
            out[w if isinstance(w, str) else w[0]] = {k: r["d" + k] for k in ("flux", "cont", "norm")}
            out.setdefault("_primal", {k: r[k] for k in ("flux", "cont", "norm")})
        return out

    def structure_jacobian(self, params: dict | None = None, quantities=("T", "P"), layers=None,
                           key: str = "norm") -> dict:
        """d spectrum / d structure per layer: {q: tensor (n_layers_requested, LENGTH)} (one JVP per layer and
        quantity; ~1 forward pass each)."""
        params = dict(params or {})
        layers = range(self.base.n) if layers is None else layers
        out = {}
        for qn in quantities:
            rows = []
            for j in layers:
                tg = torch.zeros(self.base.n, dtype=D)
                tg[j] = 1.0
                rows.append(self.jvp(params, {qn: tg})["d" + key])
            out[qn] = torch.stack(rows)
        return out


def spectrum_and_jacobian(model, window, params: dict | None = None, wrt=("vturb",), **kw):
    """One-call API: (spectrum dict, jacobian dict) for `model` (ATLAS run dir or SYNTHE deck) on
    window=(wlbeg, wlend) [nm]; `wrt` as in SyntheDiff.jacobian; extra keyword arguments go to SyntheDiff."""
    sd = SyntheDiff(model, window[0], window[1], **kw)
    J = sd.jacobian(params, wrt)
    prim = J.pop("_primal", None)
    spec = {"wave": sd.wave, **(prim or {k: v.detach() for k, v in sd(params).items() if k in ("flux", "cont", "norm")})}
    return spec, J
