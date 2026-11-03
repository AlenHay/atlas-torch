"""TEST SUPPORT (not used by the port): Intel runtime discovery for the ifx-built oracle libdfsynthe.so and the
Intel libimf transcendentals (to reproduce the -O0 Fortran bit for bit)."""
from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
LIB = EXT / "oracle" / "libdfsynthe.so"
TRACE = EXT / "traces" / "odf_solar" / "dfsynthe"
TRACE_RUN = EXT / "traces" / "odf_solar" / "dw"
STOCK = EXT / "runs" / "stock" / "odf" / "solar"
DATA = EXT / "BasicATLAS" / "data" / "dfsynthe_files"
_IFX_LIBS = ("libintlc.so.5", "libimf.so", "libsvml.so", "libirng.so", "libifcoremt.so.5", "libifport.so.5")
_loaded = False


def ifx_libdir() -> str | None:
    cands = [os.environ.get("ATLAS_IFX_LIBDIR", "")]
    cands += sorted(glob.glob("/opt/intel/oneapi/compiler/*/lib"))
    cands += sorted(glob.glob(str(Path.home() / ".local/share/containers/storage/overlay/*/diff/opt/intel/oneapi/"
                                                  "compiler/*/lib")))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "libifcoremt.so.5")):
            return c
    return None


def available() -> bool:
    return LIB.exists() and ifx_libdir() is not None


def preload() -> str:
    global _loaded
    d = ifx_libdir()
    if d is None:
        raise FileNotFoundError("Intel runtime not found; set ATLAS_IFX_LIBDIR")
    if not _loaded:
        for n in _IFX_LIBS:
            p = os.path.join(d, n)
            if os.path.exists(p):
                ctypes.CDLL(p, mode=os.RTLD_GLOBAL)
        _loaded = True
    return d


def oracle():
    from ...testing.fortran_oracle import Oracle
    preload()
    return Oracle(LIB)


def intel_flavor():
    """Flavor reproducing the ifx -O0 debug executable (golden traces) bit for bit (Intel libimf via ctypes)."""
    from .f32 import Flavor, IntelLibm
    return Flavor("O0", IntelLibm(preload()))


def glibc_flavor():
    """Flavor matching the oracle library inside a Python process (libm symbols bind to glibc)."""
    from .f32 import Flavor, GlibcLibm
    return Flavor("O0", GlibcLibm())


def oracle_xlinop(a: dict):
    """Run the oracle's XLINOP for layers 1..NL on the TAPE19 records a['tape19'] (NLTE_DTYPE) in a fresh
    process (the Intel runtime keeps unit 19 connected process-wide).  a: state, txnxn, xnfpel, dopple, xnfdop,
    xnfph, cont [NL, LENGTH], extab, scratch dir.  Returns the buffers [NL, LENBUFF]."""
    import concurrent.futures
    import multiprocessing
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as ex:
        return ex.submit(_oracle_xlinop_job, a).result()


def _oracle_xlinop_job(a: dict):
    import struct
    import tempfile

    import numpy as np

    from . import tables as T
    o = oracle()
    o.call("tabvoigt", np.array([200.0], np.float32), 2001)
    o.set_common("extab", extab=a["extab"])
    o.set_common("nlines", wlbeg=T.WLBEG, wlend=T.WLEND, resolu=T.RESOLU, ratio=T.RATIO, ratiolg=T.RATIOLG,
                 wbegin=T.WBEGIN, length=T.LENGTH, ixwlbeg=T.IXWLBEG)
    st = a["state"]
    o.set_common("state", p=st["p"], xne=st["xne"], xnatom=st["xnatom"], rho=st["rho"])
    o.set_common("temp", t=st["t"], tkev=st["tkev"], tk=st["tk"], hkt=st["hkt"], tlog=st["tlog"], hckt=st["hckt"],
                 itemp=1)
    o.set_common("txnxn", txnxn=a["txnxn"], xnfh2=st["xnfh2"])
    o.set_common("bhe", xnfhe=st["xnfhe"])
    o.set_common("rhox", nrhox=25)
    nl = a["cont"].shape[0]
    out = []
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(dir=a.get("scratch")) as td:
        os.chdir(td)
        try:
            with open("fort.19", "wb") as f:              # TAPE19 record: WL, GNNNGGG (ELO dropped)
                for r in a["tape19"]:
                    rb = r.tobytes()
                    f.write(struct.pack("<i", 52) + rb[:8] + rb[12:56] + struct.pack("<i", 52))
            for j in range(nl):
                o.set_common("buffer", buffer=np.zeros(T.LENBUFF, np.float32))
                c = np.zeros(T.LENBUFF, np.float32)
                c[: T.LENGTH] = a["cont"][j]
                o.set_common("contin", continuum=c)
                o.set_common("xnfdop", xnfpel=a["xnfpel"][j], dopple=a["dopple"][j], xnfdop=a["xnfdop"][j])
                xp = np.zeros((25, 2), np.float32)
                xp[: j + 1] = a["xnfph"][: j + 1]
                o.set_common("bhyd", xnfph=xp)
                o.call("xlinop", j + 1, len(a["tape19"]))
                out.append(o.common("buffer")["buffer"].copy())
        finally:
            os.chdir(cwd)
    return np.stack(out)
