"""Test support for the ifx-built oracle libxnfdf.so (ifx -g -O0 -save).

The Intel runtime (libimf, libsvml, libintlc, libifcoremt, libifport) is not on the host library path
(ifx lives in a container image); preload it RTLD_GLOBAL from $ATLAS_IFX_LIB or the podman overlay so
that `Oracle(libxnfdf.so)` resolves its dependencies by soname.
"""
from __future__ import annotations

import ctypes
import glob
import os
from functools import lru_cache
from pathlib import Path

_LIBS = ("libintlc.so.5", "libimf.so", "libsvml.so", "libirng.so", "libifport.so.5", "libifcoremt.so.5")


@lru_cache(maxsize=None)
def ifx_lib_dir() -> str | None:
    d = os.environ.get("ATLAS_IFX_LIB")
    if d and Path(d, "libimf.so").exists():
        return d
    pats = [str(Path.home() / ".local/share/containers/storage/overlay/*/diff/opt/intel/oneapi/compiler/*/lib/libimf.so"),
            "/opt/intel/oneapi/compiler/*/lib/libimf.so"]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return str(Path(hits[-1]).parent)
    return None


@lru_cache(maxsize=None)
def ensure_ifx_runtime() -> bool:
    d = ifx_lib_dir()
    if d is None:
        return False
    pending = [Path(d, n) for n in _LIBS if Path(d, n).exists()]
    for _ in range(len(pending) + 1):          # resolve inter-library dependencies by retrying
        left = []
        for p in pending:
            try:
                ctypes.CDLL(str(p), mode=os.RTLD_LAZY | os.RTLD_GLOBAL)
            except OSError:
                left.append(p)
        if not left:
            return True
        pending = left
    raise OSError(f"cannot preload Intel runtime from {d}: {pending}")


def imf_function(name: str, nargs: int = 1):
    """A libimf scalar math function (for diagnosing libm ulp differences)."""
    ensure_ifx_runtime()
    lib = ctypes.CDLL(str(Path(ifx_lib_dir(), "libimf.so")))
    f = getattr(lib, name)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double] * nargs
    return f


# ---------------------------------------------------------------------------------------------------
# helpers for tests/oracle/test_xnfdf_*.py
# ---------------------------------------------------------------------------------------------------
def ext_dir() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))


def oracle_path() -> Path:
    return ext_dir() / "oracle" / "libxnfdf.so"


def available() -> bool:
    return oracle_path().exists() and ifx_lib_dir() is not None


def new_oracle():
    """Fresh private copy of libxnfdf.so (SAVE state from the load-time image)."""
    ensure_ifx_runtime()
    from ...testing.fortran_oracle import Oracle
    return Oracle(oracle_path())


def card_array(card) -> "np.ndarray":
    """CARD(1..81) REAL*8 array as 80A1 leaves it: each character followed by 7 blanks."""
    import numpy as np
    chars = card[1:82] if isinstance(card, list) else list(card[:81].ljust(81))
    raw = b"".join(c.encode("latin-1") + b" " * 7 for c in chars)
    return np.frombuffer(raw, dtype="<f8").copy(order="F")


_BLOCK_OF = {"bhyd": "bhyd", "bmin": "bhyd", "bhe1": "bhe", "bhe2": "bhe", "bc1": "bc", "bc2": "bc", "bmg1": "bmg",
             "bmg2": "bmg", "bal1": "bal", "bal2": "bal", "bsi1": "bsi", "bsi2": "bsi", "bca1": "bca", "bca2": "bca",
             "bna1": "bna", "bo1": "bo", "bo2": "bo", "bb1": "bb", "bfe1": "bfe", "bk1": "bk"}


def push_model(o, m, itemp: int = 1, nlteon: int = 0):
    """Load one READIN/BEGIN model (readin.Model) into the oracle COMMONs."""
    o.set_common("temp", t=m.t, tkev=m.tkev, tk=m.tk, hkt=m.hkt, tlog=m.tlog, hckt=m.hckt, itemp=itemp)
    o.set_common("state", p=m.p, xne=m.xne, xnatom=m.xnatom, rho=m.rho)
    o.set_common("elem", abund=m.abund, atmass=m.atmass, xabund=m.xabund, wtmole=m.wtmole)
    o.set_common("rhox", rhox=m.rhox, nrhox=m.nrhox)
    o.set_common("if", nlteon=nlteon, ifmol=m.ifmol, ifpres=1, ifop=list(m.ifop))
    o.set_common("iter", iter=0, numits=1)          # ITER<NUMITS: no printout from NMOLEC/NELECT
    for name, arr in m.b.items():
        o.set_common(_BLOCK_OF[name], **{name: arr})


def push_molecules(o, mol):
    """READMOL result into /IFEQUA/ and /XNMOL/ (CODE, NUMMOL)."""
    o.set_common("ifequa", ifequa=mol.ifequa[1:102], kcomps=mol.kcomps, locj=mol.locj, equil=mol.equil,
                 idequa=mol.idequa[1:31], nequa=mol.nequa, nloc=len(mol.kcomps))
    o.set_common("xnmol", codemol=mol.code, nummol=mol.nummol)


def push_kapp_state(o, m, g: dict, b: int, itemp: int):
    """Everything KAPP(1) and its sources read, for model b of a xnfdf.group_state() batch."""
    import numpy as np
    r, P = g["r"], g["P"]
    push_model(o, m, itemp=itemp)
    push_molecules(o, m.mol)
    o.saved("pops")["itemp1"][...] = itemp            # POPS inside LUKEOP/HOTOP must not re-run NMOLEC
    o.set_common("state", xne=r["xne"][b].numpy(), xnatom=r["xnatom"][b].numpy(), rho=r["rho"][b].numpy())
    o.set_common("xnmol", xnmol=r["xnmol"][b].numpy(), xnfpmol=r["xnfpmol"][b].numpy())
    o.set_common("if", ifop=list(g["ki"].ifop))
    col = lambda k: P[k][b].numpy()
    o.set_common("bhyd", xnfph=col("xnfph"), xnfh=col("xnfh")[:, 0])
    o.set_common("bhe", xnfphe=col("xnfphe"), xnfhe=col("xnfhe"))
    for blk, k in (("bc", "xnfpc"), ("bmg", "xnfpmg"), ("bal", "xnfpal"), ("bsi", "xnfpsi"), ("bfe", "xnfpfe"),
                   ("bca", "xnfpca"), ("bo", "xnfpo"), ("bna", "xnfpna"), ("bk", "xnfpk"), ("bb", "xnfpb")):
        o.set_common(blk, **{k: col(k)})
    del np


def push_freq(o, freq: float, hkt):
    """/FREQ/ as PROGRAM XNFDF sets it before CALL KAPP."""
    import math
    import numpy as np
    f15 = freq / 1e15
    ehvkt = np.exp(-freq * hkt)
    stim = 1. - ehvkt
    bnu = 1.47439e-02 * f15 ** 3 * ehvkt / stim
    o.set_common("freq", freq=freq, freqlg=math.log(freq), ehvkt=ehvkt, stim=stim, bnu=bnu,
                 waveno=1e7 / (2.99792458e17 / freq))


from contextlib import contextmanager


@contextmanager
def oracle_libm():
    """Run the port with the libm the ctypes oracle binds to (glibc exp/log/log10; python's libm comes
    first in the global symbol scope, so the oracle's exp@plt resolves there, not to libimf).
    Elementwise scalar calls: slow, test-only. Proves that remaining port/oracle differences are libm ulps."""
    import numpy as np
    import torch
    m = ctypes.CDLL("libm.so.6")

    def scalar(name):
        f = getattr(m, name)
        f.restype = ctypes.c_double
        f.argtypes = [ctypes.c_double]
        return f
    orig = {k: getattr(torch, k) for k in ("exp", "log", "log10")}

    def wrap(name):
        f, o = scalar(name), orig[name]

        def g(x, *a, **k):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.float64 or a or k:
                return o(x, *a, **k)
            xs = x.detach().cpu().numpy().ravel()
            return torch.from_numpy(np.fromiter((f(float(v)) for v in xs), np.float64, xs.size).reshape(x.shape))
        return g
    try:
        for k in orig:
            setattr(torch, k, wrap(k))
        yield
    finally:
        for k, v in orig.items():
            setattr(torch, k, v)
