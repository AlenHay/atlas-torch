"""TEST-ONLY physics backend for the KAPPA9 port: populations and continuous opacities from the Fortran.

This adapter is NOT part of the torch port. It exists so that the KAPPA9 program (atlas_torch.kappa9.kappa9_prog:
READIN usage, T/P grid, LINOP, KAPP assembly, Rosseland integration, kapk writer) can be run and tested end to
end before the torch physics library (atlas_torch.atlas9.physics, variant 'kappa9') is available. It drives the
population and continuum routines of the oracle library libkappa9.so (ifx -g -O0) through ctypes on the
State produced by the port:

* load(rd)            copy the READIN State (abundances, IFOP, /IF/, /IFEQUA/ molecule tables, /DEPART/,
                      deck /TEMP/ incl. the stale HCKT, ...) into the library's COMMONs;
* populations(k, g)   KAPPA9's POPS block for pressure K (ITEMP = K): POPS(0,1,XNE), the 11 POPS(...,11, XNFP*),
                      molecules ON: POPS(106/108) + POPS(...,11,XNF*) and the PFSAHA products, exactly as the
                      main program; returns the population COMMONs for the NRHOX layers;
* continuum(f)        KAPP's N=1 opacity calls (all IFOP sources except LINOP/LINSOP, which the port computes)
                      after zeroing /OPS/; returns the 25 /OPS/ arrays.

The library needs the Intel runtime (libifcoremt, libimf, libsvml, libintlc, libifport); it is preloaded from
$ATLAS_IFX_LIBDIR, /opt/intel/oneapi/compiler/*/lib or the oneAPI dev-container image.
"""
from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path

import numpy as np
import torch

from .fortran_oracle import Oracle

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
LIB = EXT / "oracle" / "libkappa9.so"
_IFX_LIBS = ("libintlc.so.5", "libimf.so", "libsvml.so", "libirng.so", "libifcoremt.so.5", "libifport.so.5")
_loaded = False

# KAPP's N=1 opacity sources: IFOP index (1-based) -> routine (LINOP=15 / LINSOP=16 are the port's)
_SOURCES = [(1, "hop"), (2, "h2plop"), (3, "hminop"), (4, "hrayop"), (5, "he1op"), (6, "he2op"), (7, "hemiop"),
            (8, "heraop"), (9, "coolop"), (10, "lukeop"), (11, "hotop"), (12, "elecop"), (13, "h2raop"),
            (14, "hlinop"), (17, "xlinop"), (18, "xlisop"), (19, "xconop"), (20, "xsop")]
OPS = ["ahyd", "ah2p", "ahmin", "sigh", "ahe1", "ahe2", "ahemin", "sighe", "acool", "aluke", "ahot", "sigel",
       "sigh2", "ahline", "alines", "siglin", "axline", "sigxl", "axcont", "sigx", "shyd", "shmin", "shline",
       "sxline", "sxcont"]
POP_KEYS = {"state": ["xne", "xnatom", "rho"], "ions": ["xnfph", "xnfphe", "xnfh", "xnfhe"],
            "xnfp": ["xnfpc", "xnfpn", "xnfpo", "xnfpne", "xnfpal", "xnfpmg", "xnfpsi", "xnfpca", "xnfpfe",
                     "xnfpch", "xnfpoh"],
            "xnf": ["xnfc", "xnfn", "xnfo", "xnfne", "xnfmg", "xnfsi", "xnfs", "xnffe"]}


def ifx_libdir() -> str | None:
    cands = [os.environ.get("ATLAS_IFX_LIBDIR", "")]
    cands += sorted(glob.glob("/opt/intel/oneapi/compiler/*/lib"))
    cands += sorted(glob.glob(str(Path.home() / ".local/share/containers/storage/overlay/*/diff/opt/intel/oneapi/compiler/*/lib")))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "libifcoremt.so.5")):
            return c
    return None


def preload_ifx_runtime() -> None:
    global _loaded
    if _loaded:
        return
    d = ifx_libdir()
    if d is None:
        raise FileNotFoundError("Intel Fortran runtime (libifcoremt.so.5) not found; set ATLAS_IFX_LIBDIR")
    for n in _IFX_LIBS:
        p = os.path.join(d, n)
        if os.path.exists(p):
            ctypes.CDLL(p, mode=os.RTLD_GLOBAL)
    _loaded = True


def available() -> bool:
    return LIB.exists() and ifx_libdir() is not None


def _np(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


class OracleKappa9Physics:
    """Populations + continuum of libkappa9.so for one 57-layer (or fewer) pressure column at a time."""

    name = "oracle-adapter"

    def __init__(self, so_path: str | Path = LIB):
        preload_ifx_runtime()
        self.o = Oracle(so_path)
        self.nrhox = 0

    def set(self, block: str, var: str, value) -> None:
        views = self.o.common(block)
        dst = views[var]
        src = _np(value)
        if dst.ndim == 0 or dst.shape == ():
            dst[...] = src.reshape(-1)[0]
        else:
            src = src.astype(dst.dtype)
            dst[tuple(slice(0, n) for n in src.shape)] = src

    def load(self, rd) -> None:
        """Copy every READIN entry that is a COMMON member of the library."""
        lay = self.o.commons_layout
        for key, v in rd.items():
            if "." not in key or key.startswith("save."):
                continue
            blk, var = key.split(".", 1)
            if blk in lay and var in lay[blk] and not (isinstance(v, np.ndarray) and v.dtype.kind == "S"):
                self.set(blk, var, v)

    def populations(self, k: int, grid: dict) -> dict:
        """grid: t, tk, hkt, tkev, tlog, p (layers). Runs the KAPPA9 POPS block with ITEMP = k."""
        o = self.o
        n = int(grid["t"].shape[0])
        self.nrhox = n
        self.set("rhox", "nrhox", n)
        for v in ("t", "tk", "hkt", "tkev", "tlog"):
            self.set("temp", v, grid[v])
        self.set("temp", "itemp", k)
        self.set("state", "p", grid["p"])
        ions, xnfp, xnf = o.common("ions"), o.common("xnfp"), o.common("xnf")
        st = o.common("state")
        ifmol = int(o.common("if")["ifmol"].reshape(-1)[0])

        def pops(code, mode, arr):
            o.call("pops", float(code), int(mode), _farr(arr))

        pops(0.0, 1, st["xne"])
        for code, a in [(1.01, ions["xnfph"]), (2.02, ions["xnfphe"]), (6.03, xnfp["xnfpc"]), (7.04, xnfp["xnfpn"]),
                        (8.05, xnfp["xnfpo"]), (10.05, xnfp["xnfpne"]), (12.01, xnfp["xnfpmg"]), (13.00, xnfp["xnfpal"]),
                        (14.01, xnfp["xnfpsi"]), (20.01, xnfp["xnfpca"]), (26.00, xnfp["xnfpfe"])]:
            pops(code, 11, a)
        heavy = [(6.05, "xnfc"), (7.05, "xnfn"), (8.05, "xnfo"), (10.05, "xnfne"), (12.05, "xnfmg"),
                 (14.05, "xnfsi"), (16.05, "xnfs"), (26.04, "xnffe")]
        if ifmol == 0:
            pops(1.01, 12, ions["xnfh"])
            pops(2.02, 12, ions["xnfhe"])
            for code, v in heavy:
                pops(code, 12, xnf[v])
        else:
            pops(106.00, 11, xnfp["xnfpch"])
            pops(108.00, 11, xnfp["xnfpoh"])
            for code, v in heavy:
                pops(code, 11, xnf[v])
            part = np.zeros((99, 6), order="F")
            for j in range(1, n + 1):
                jj = j - 1
                o.call("pfsaha", j, 1, 1, 3, part)
                ions["xnfh"][jj, 0] = ions["xnfph"][jj, 0] * part[jj, 0]
                ions["xnfh"][jj, 1] = ions["xnfph"][jj, 1]
                o.call("pfsaha", j, 2, 2, 13, part)
                ions["xnfhe"][jj, 0] = ions["xnfphe"][jj, 0] * part[jj, 0]
                ions["xnfhe"][jj, 1] = ions["xnfphe"][jj, 1] * part[jj, 1]
                ions["xnfhe"][jj, 2] = ions["xnfphe"][jj, 2]
                for iz, nion, v in [(6, 6, "xnfc"), (7, 6, "xnfn"), (8, 6, "xnfo"), (10, 6, "xnfne"), (12, 6, "xnfmg"),
                                    (14, 6, "xnfsi"), (16, 6, "xnfs"), (26, 5, "xnffe")]:
                    o.call("pfsaha", j, iz, nion, 13, part)
                    for i in range(nion):
                        xnf[v][jj, i] = xnf[v][jj, i] * part[jj, i]
        out = {}
        for blk, vs in POP_KEYS.items():
            c = o.common(blk)
            for v in vs:
                out[f"{blk}.{v}"] = torch.from_numpy(np.array(c[v][:n], copy=True))
        return out

    def continuum(self, freq: float, freqlg: float, ehvkt, stim, bnu) -> dict:
        """KAPP N=1 opacity sources (except LINOP/LINSOP) at one frequency for the current column."""
        o = self.o
        f = o.common("freq")
        f["freq"][...] = freq
        f["freqlg"][...] = freqlg
        n = self.nrhox
        f["ehvkt"][:n] = _np(ehvkt)
        f["stim"][:n] = _np(stim)
        f["bnu"][:n] = _np(bnu)
        ops = o.common("ops")
        ops["dummy"][:n, :] = 0.0
        ifop = o.common("ifop")["ifop"]
        for i, r in _SOURCES:
            if ifop[i - 1] == 1:
                o.call(r)
        return {v: torch.from_numpy(np.array(ops[v][:n], copy=True)) for v in OPS}


def _farr(a: np.ndarray) -> np.ndarray:
    if not a.flags.f_contiguous:
        raise ValueError("non-contiguous COMMON view")
    return a


_READIN_SCRIPT = r"""
import sys, numpy as np
sys.path.insert(0, sys.argv[3])
from atlas_torch.testing.kappa9_oracle_physics import preload_ifx_runtime
preload_ifx_runtime()
from atlas_torch.testing.fortran_oracle import Oracle
o = Oracle(sys.argv[1])
o.call("readin", 1)
out = {}
for blk in o.commons_layout:
    for k, v in o.common(blk).items():
        out[blk + "." + k] = np.array(v, copy=True)
np.savez(sys.argv[2], **out)
"""


def fortran_readin(stdin_text: str, molecules_path: str | Path, so_path: str | Path = LIB) -> dict:
    """Run the library's READIN(1) on `stdin_text` (unit 5) with unit 2 = molecules_path in a subprocess
    (fresh Fortran runtime, private cwd); returns every COMMON member after READIN as numpy arrays."""
    import subprocess
    import sys
    import tempfile
    src = str(Path(__file__).resolve().parents[2])
    with tempfile.TemporaryDirectory(prefix="k9readin_") as d:
        os.symlink(os.path.abspath(molecules_path), os.path.join(d, "fort.2"))
        out = os.path.join(d, "commons.npz")
        p = subprocess.run([sys.executable, "-c", _READIN_SCRIPT, str(so_path), out, src], cwd=d,
                           input=stdin_text.encode("latin-1"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("Fortran READIN failed: " + p.stderr.decode("latin-1")[-2000:])
        with np.load(out) as z:
            return {k: z[k] for k in z.files}


_MAIN_SCRIPT = r"""
import sys, ctypes
sys.path.insert(0, sys.argv[2])
from atlas_torch.testing.kappa9_oracle_physics import preload_ifx_runtime
preload_ifx_runtime()
ctypes.CDLL(sys.argv[1]).MAIN__()
"""


def fortran_kappa9_main(stdin_text: str, odf_path: str | Path, molecules_path: str | Path,
                        so_path: str | Path = LIB) -> str:
    """The whole Fortran KAPPA9 program of the oracle library (MAIN__, subprocess, private cwd with
    fort.9 = ODF and fort.2 = molecules): returns its fort.7 (kapk text)."""
    import subprocess
    import sys
    import tempfile
    src = str(Path(__file__).resolve().parents[2])
    with tempfile.TemporaryDirectory(prefix="k9main_") as d:
        os.symlink(os.path.abspath(odf_path), os.path.join(d, "fort.9"))
        os.symlink(os.path.abspath(molecules_path), os.path.join(d, "fort.2"))
        p = subprocess.run([sys.executable, "-c", _MAIN_SCRIPT, str(so_path), src], cwd=d,
                           input=stdin_text.encode("latin-1"), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        f7 = os.path.join(d, "fort.7")
        if p.returncode != 0 or not os.path.exists(f7):
            raise RuntimeError("Fortran KAPPA9 failed: " + p.stderr.decode("latin-1")[-2000:])
        return Path(f7).read_text()
