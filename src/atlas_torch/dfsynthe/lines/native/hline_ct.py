"""ctypes mirrors of the hline.c structs (hsetup_t, hdepth_t, hline_t) and their construction from the torch
Hydrogen state (lines/hydrogen.py) and tables."""
from __future__ import annotations

import ctypes

import numpy as np
import torch

c_f, c_d, c_i, c_l, c_p = ctypes.c_float, ctypes.c_double, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p


class HSetup(ctypes.Structure):
    _fields_ = [("n", c_i), ("m", c_i), ("index", c_i), ("nfin", c_i),
                *[(k, c_f) for k in ("y1num", "y1wht", "freqnm", "dbeta", "c1con", "c2con", "radamp", "resont",
                                     "vdw", "stark")],
                ("finest", c_f * 16), ("finswt", c_f * 16)]

    @classmethod
    def of(cls, s: dict) -> "HSetup":
        o = cls()
        o.n, o.m, o.index = int(s["n"]), int(s["m"]), int(s["index"])
        fe, fw = np.asarray(s["finest"], np.float32), np.asarray(s["finswt"], np.float32)
        assert len(fe) == len(fw) <= 16
        o.nfin = len(fe)
        for k in ("y1num", "y1wht", "freqnm", "dbeta", "c1con", "c2con", "radamp", "resont", "vdw", "stark"):
            setattr(o, k, float(np.float32(s[k])))
        for i in range(len(fe)):
            o.finest[i], o.finswt[i] = float(fe[i]), float(fw[i])
        return o


class HDepth(ctypes.Structure):
    _fields_ = [*[(k, c_p) for k in ("pp", "fo", "y1b", "y1s", "t3nhe", "t3nh2", "c1d", "c2d", "gcon1", "gcon2",
                                     "xne", "xnfph1", "xnfph2")],
                ("h0tab", c_p), ("nh0", c_l), ("extab", c_p), ("nextab", c_l), ("cutoffh2", c_p),
                ("cutoffh2plus", c_p), ("propbm", c_p), ("sc", c_p), ("sd", c_p), ("spp", c_p), ("sbeta", c_p),
                ("c52", c_f), ("fastmath", c_i), ("f_logf", c_p), ("f_expf4", c_p), ("f_expf1", c_p),
                ("f_iexpf", c_p), ("f_powf", c_p), ("rcptab", c_p), ("kf", c_f * 16), ("kf2", c_f * 2)]

# order of HDepth.kf (hline.c KF_*) and kf2, from Hydrogen._fast_consts
KF = ("l", "res", "pi", "c2", "nf22", "nsp22", "r22", "c22", "a22", "b22", "f22", "nf15", "nsp15", "r15", "c15",
      "a15")
KF2 = ("b15", "f15")


class HLine(ctypes.Structure):
    _fields_ = [("nl", c_i), ("path", c_i), ("do_red", c_i), ("nb", c_l), ("length", c_l),
                *[(k, c_d) for k in ("wl", "wlbeg", "w0r", "w0b", "alphahyd_n", "ratio", "redcut", "wlminus1",
                                     "wlminus2", "bluecut", "wlplus1", "wlplus2")],
                *[(k, c_p) for k in ("keep", "k0", "dopph", "wcon", "wtail", "k0red", "k0blue", "s_main", "s_red",
                                     "s_blue", "cont")],
                ("cont_stride", c_l), ("cont_cols", c_l), ("buf", c_p), ("buf_stride", c_l), ("emit", c_i),
                *[(k, c_p) for k in ("sidx", "sp", "first", "change")],
                ("stats", c_l * 3)]


def interp_math(fl):
    """Stock flavour: addresses of libimf logf, __svml_expf4 and __svml_expf1 (the functions dfsynthe.exe uses in
    the hydrogen-wing log interpolation, see xlinop.py XLinop._h_interp); None otherwise or without them."""
    if not getattr(fl, "fastmath", False):
        return None
    from .. import imf, svml
    a = (imf.address("logf"), svml._vaddr("__svml_expf4"), svml._vaddr("__svml_expf1"))
    return a if all(a) else None


def depth_struct(hyd, tabs: dict, fl) -> HDepth:
    """HDepth pointing at contiguous float32 copies (kept alive on the returned object)."""
    from ..hydrogen import _tab, sofbet_tables, Hydrogen
    c32 = lambda t: (t if isinstance(t, torch.Tensor) else torch.from_numpy(np.asarray(t))).to(
        "cpu", torch.float32).contiguous()  # noqa: E731
    keep = {}
    for k in ("pp", "fo", "y1b", "y1s", "t3nhe", "t3nh2", "c1d", "c2d", "gcon1", "gcon2", "xne"):
        keep[k] = c32(getattr(hyd, k))
    keep["xnfph1"], keep["xnfph2"] = c32(hyd.xnfph[:, 0]), c32(hyd.xnfph[:, 1])
    keep["h0tab"], keep["extab"] = c32(hyd.h0tab), c32(hyd.extab)
    keep["cutoffh2"] = c32(_tab("hprof4", "cutoffh2").astype(np.float32))
    keep["cutoffh2plus"] = c32(_tab("hprof4", "cutoffh2plus").astype(np.float32))
    t = sofbet_tables()
    keep["propbm"] = c32(np.ascontiguousarray(t["propbm"]))            # (ip 5, j 15, index 7)
    keep["sc"], keep["sd"] = c32(np.ascontiguousarray(t["c"])), c32(np.ascontiguousarray(t["d"]))
    keep["spp"], keep["sbeta"] = c32(t["pp"]), c32(t["beta"])
    assert keep["propbm"].shape == (5, 15, 7) and keep["sc"].shape == (5, 7)
    assert keep["cutoffh2"].numel() == 91 and keep["cutoffh2plus"].numel() == 111
    o = HDepth()
    for k, v in keep.items():
        setattr(o, k, v.data_ptr())
    o.nh0, o.nextab = keep["h0tab"].numel(), keep["extab"].numel()
    o.c52 = float(Hydrogen._c52())
    o.fastmath = int(bool(fl.fastmath))
    m = interp_math(fl)
    if m is not None:
        o.f_logf, o.f_expf4, o.f_expf1 = m
    if fl.fastmath:
        from .. import imf
        from ..hydrogen import Hydrogen
        from ..rcpps import _table_np
        K = Hydrogen._fast_consts()
        for i, k in enumerate(KF):
            o.kf[i] = float(K[k])
        for i, k in enumerate(KF2):
            o.kf2[i] = float(K[k])
        keep["rcptab"] = torch.from_numpy(_table_np().astype(np.uint32).view(np.int32).copy())
        o.rcptab = keep["rcptab"].data_ptr()
        if fl.libm is None:
            o.f_logf = imf.address("logf") or None
            o.f_iexpf = imf.address("expf") or None
            f = imf._fn("powf", False, 2)
            o.f_powf = ctypes.cast(f, ctypes.c_void_p).value if f is not None else None
    o._keep = keep
    return o
