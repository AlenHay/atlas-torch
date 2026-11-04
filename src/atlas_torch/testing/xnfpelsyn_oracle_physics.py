"""TEST-ONLY physics backend for PROGRAM XNFPELSYN: READIN / POPS / KAPP of the Fortran oracle library.

This adapter is NOT part of the torch port. It gives `atlas_torch.synthe.xnfpelsyn_prog.xnfpelsyn(...,
physics=OracleXnfpelsynPhysics())` the Fortran routines of /home/deck/atlas-ext/oracle/libxnfpelsyn.so
(gfortran -g -O0 -fno-automatic: xnfpelsyn.for + atlas7v.for) behind the same interface as the torch
backend (`TorchPhysics`), so the main-program logic (edges, FREQSET, POPS call sequence, H2/CO equilibria,
log10 tables, XNFPEL/DOPPLE, fort.10 writer) is tested against the Fortran independently of the physics
port:

* readin(deck, molecules)   CALL READIN(20) in a subprocess (deck on unit 5 = stdin, fort.2 = molecules in a
                            private cwd); every COMMON member is copied back into this process' Oracle;
* get(key) / set(key, v)    COMMON members ("block.var");
* pops(code, mode, number)  CALL POPS(CODE,MODE,NUMBER) on a (kw, k) Fortran array;
* kapp(freq, ...)           /FREQ/ set from the program's loop-120 values, CALL KAPP(1,NSTEPS,STEPWT) per
                            frequency, returns /OPTOT/ ACONT, SIGMAC [NF, n].
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .fortran_oracle import Oracle

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
LIB = EXT / "oracle" / "libxnfpelsyn.so"

_READIN_SCRIPT = r"""
import sys, numpy as np
sys.path.insert(0, sys.argv[3])
from atlas_torch.testing.fortran_oracle import Oracle
o = Oracle(sys.argv[1])
o.call("readin", 20)
out = {}
for blk in o.commons_layout:
    for k, v in o.common(blk).items():
        out[blk + "." + k] = np.array(v, copy=True)
np.savez(sys.argv[2], **out)
"""


def available() -> bool:
    return LIB.exists()


def gfortran13_libdir() -> str | None:
    """Directory of the build container's libgfortran.so.5 (GCC 13, the runtime the oracle libraries were
    built for). The host's newer libgfortran (GCC 16) rejects READIN's legacy Hollerith FORMAT 1517
    ("Missing comma between descriptors") at run time, so READIN runs against GCC 13's runtime."""
    import glob
    cands = [os.environ.get("ATLAS_GFORTRAN_LIBDIR", "")]
    cands += sorted(glob.glob(str(Path.home() / ".local/share/containers/storage/overlay/*/diff/usr/lib/"
                                                 "x86_64-linux-gnu")))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "libgfortran.so.5")):
            return c
    return None


def _env() -> dict:
    env = dict(os.environ)
    d = gfortran13_libdir()
    if d is not None:      # only libgfortran (+ libquadmath) from the container; glibc stays the host's
        shim = Path(tempfile.gettempdir()) / f"xnfpelsyn_gf13_{os.getuid()}"
        shim.mkdir(exist_ok=True)
        for n in ("libgfortran.so.5", "libquadmath.so.0"):
            if os.path.exists(os.path.join(d, n)) and not (shim / n).exists():
                os.symlink(os.path.join(d, n), shim / n)
        env["LD_LIBRARY_PATH"] = str(shim) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def fortran_readin(deck_text: str, molecules_path, so_path=LIB) -> dict:
    """READIN(20) of the library on `deck_text` in a fresh process -> {block.var: array} after the call."""
    src = str(Path(__file__).resolve().parents[2])
    with tempfile.TemporaryDirectory(prefix="xnfreadin_") as d:
        os.symlink(os.path.abspath(molecules_path), os.path.join(d, "fort.2"))
        out = os.path.join(d, "commons.npz")
        p = subprocess.run([sys.executable, "-c", _READIN_SCRIPT, str(so_path), out, src], cwd=d,
                           input=deck_text.encode("latin-1"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           env=_env())
        if p.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("Fortran READIN failed: " + p.stderr.decode("latin-1")[-2000:])
        with np.load(out) as z:
            return {k: z[k] for k in z.files}


def fortran_main(deck_text: str, continua_path, molecules_path, workdir, so_path=LIB) -> Path:
    """The whole Fortran program of the oracle library (MAIN__) in `workdir` -> its fort.10 (the -O0
    reference for byte comparisons)."""
    w = Path(workdir)
    w.mkdir(parents=True, exist_ok=True)
    for name, src in (("fort.17", continua_path), ("fort.2", molecules_path)):
        if (w / name).exists() or (w / name).is_symlink():
            (w / name).unlink()
        os.symlink(os.path.abspath(src), w / name)
    script = ("import ctypes, sys; argv = (ctypes.c_char_p * 2)(b'xnfpelsyn', None); "
              "ctypes.CDLL(sys.argv[1]).main(1, argv)")     # gfortran's C main -> set_args/options -> MAIN__
    p = subprocess.run([sys.executable, "-c", script, str(so_path)], cwd=w, input=deck_text.encode("latin-1"),
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=_env())
    if not (w / "fort.10").exists():
        raise RuntimeError("Fortran XNFPELSYN failed: " + p.stderr.decode("latin-1")[-2000:])
    return w / "fort.10"


class OracleXnfpelsynPhysics:
    """libxnfpelsyn.so READIN/POPS/KAPP with the TorchPhysics interface (tests only)."""

    name = "oracle-adapter"

    def __init__(self, so_path=LIB):
        self.so_path = so_path
        self.o = Oracle(so_path)

    def _view(self, key: str) -> np.ndarray:
        blk, var = key.split(".", 1)
        return self.o.common(blk)[var]

    def readin(self, deck_text: str, molecules_path) -> None:
        for key, v in fortran_readin(deck_text, molecules_path, self.so_path).items():
            dst = self._view(key)
            dst[...] = v.reshape(dst.shape)

    def get(self, key: str) -> np.ndarray:
        return np.array(self._view(key), copy=True)

    def set(self, key: str, value) -> None:
        dst = self._view(key)
        v = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        dst[...] = v.astype(dst.dtype).reshape(dst.shape)

    def pops(self, code: float, mode: int, number) -> torch.Tensor:
        a = number.detach().cpu().numpy() if isinstance(number, torch.Tensor) else np.asarray(number)
        a = np.asfortranarray(a, dtype=np.float64).copy(order="F")
        self.o.call("pops", float(code), int(mode), a)
        return torch.from_numpy(np.ascontiguousarray(a))

    def kapp(self, freq, freqlg, waveno, ehvkt, stim, bnu):
        n = int(self._view("rhox.nrhox").reshape(-1)[0])
        f = self.o.common("freq")
        op = self.o.common("optot")
        npy = lambda x: x.detach().cpu().numpy()                     # noqa: E731
        freq, freqlg, waveno = npy(freq).reshape(-1), npy(freqlg).reshape(-1), npy(waveno).reshape(-1)
        ehvkt, stim, bnu = npy(ehvkt), npy(stim), npy(bnu)
        nf = freq.size
        acont = np.empty((nf, n))
        sigmac = np.empty((nf, n))
        for i in range(nf):
            f["freq"][...] = freq[i]
            f["freqlg"][...] = freqlg[i]
            f["ehvkt"][:n] = ehvkt[i]
            f["stim"][:n] = stim[i]
            f["bnu"][:n] = bnu[i]
            f["waveno"][...] = waveno[i]
            self.o.call("kapp", 1, 0, 0.0)
            acont[i] = op["acont"][:n]
            sigmac[i] = op["sigmac"][:n]
        return torch.from_numpy(acont), torch.from_numpy(sigmac)
