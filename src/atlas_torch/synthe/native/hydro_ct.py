"""ctypes mirrors of the synline.c hydrogen structs (shs_t, shd_t, shl_t) and their construction from the torch
HProf4 state (synthe/lineopac/hydrogen.py) of a SYNTHE layer block."""
from __future__ import annotations

import ctypes

import numpy as np
import torch

c_f, c_d, c_i, c_l, c_p = ctypes.c_float, ctypes.c_double, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p
F32 = np.float32


class SHS(ctypes.Structure):
    _fields_ = [("n", c_i), ("m", c_i), ("nfin", c_i),
                *[(k, c_f) for k in ("freqnm", "dbeta", "stark", "vdw", "vdw2", "radamp", "resont", "radfq_pi",
                                     "fqrad")],
                ("wavenm", c_d), ("finest", c_f * 16), ("finswt", c_f * 16),
                *[(k, c_p) for k in ("c1", "c2", "g1", "gnot")]]


class SHD(ctypes.Structure):
    _fields_ = [*[(k, c_p) for k in ("fo", "t3nhe", "t3nh2", "xnfph1", "xnfph2", "gcon1", "gcon2", "pp", "dopph",
                                     "extab")],
                ("nextab", c_l), *[(k, c_p) for k in ("cutoffh2", "cutoffh2plus", "prob", "sc", "sd", "spp",
                                                     "sbeta")],
                ("c4000", c_f)]


class SHL(ctypes.Structure):
    _fields_ = [("nl", c_i), ("path", c_i), ("do_red", c_i),
                *[(k, c_p) for k in ("lay", "k0", "k0red", "k0blue", "wcon", "wtail")],
                ("nbuff", c_l), ("length", c_l),
                *[(k, c_d) for k in ("wl", "wbegin", "ratio", "alphahyd", "redcut", "wlm1", "wlm2", "bluecut", "wlp1",
                                     "wlp2")],
                *[(k, c_p) for k in ("s_main", "s_red", "s_blue", "buffer", "cutcont")]]


def _c32(t):
    return (t if isinstance(t, torch.Tensor) else torch.as_tensor(np.asarray(t))).to("cpu", torch.float32).contiguous()


def setup_struct(hp, n: int, m: int) -> SHS:
    """SHS for line (n, m) of the block's HProf4 `hp` (per-layer Stark set-up kept alive on the object)."""
    c = hp.line(n, m)
    o = SHS()
    o.n, o.m = n, m
    fe, fw = np.asarray(c["finest"], F32), np.asarray(c["finswt"], F32)
    assert len(fe) == len(fw) <= 16
    o.nfin = len(fe)
    for i in range(len(fe)):
        o.finest[i], o.finswt[i] = float(fe[i]), float(fw[i])
    o.freqnm, o.dbeta, o.stark = float(c["freqnm"]), float(c["dbeta"]), float(c["stark"])
    o.vdw, o.vdw2 = float(c["vdw"]), float(F32(2) * c["vdw"])
    o.radamp, o.resont = float(c["radamp"]), float(c["resont"])
    o.radfq_pi = float(c["radamp"] * c["freqnm"] / F32(3.14159))
    o.fqrad = float(c["freqnm"] * c["radamp"])
    o.wavenm = float(c["wavenm"])
    keep = {k: _c32(c[k]) for k in ("c1", "c2", "g1", "gnot")}
    for k, v in keep.items():
        setattr(o, k, v.data_ptr())
    o._keep = keep
    return o


def depth_struct(hp, dopph: torch.Tensor, extab: torch.Tensor) -> SHD:
    from ..lineopac.hydrogen import _folded_pow10, _t
    assert dopph.dtype == torch.float32 and dopph.is_contiguous() and dopph.device.type == "cpu"
    keep = {k: _c32(getattr(hp, k)) for k in ("fo", "t3nhe", "t3nh2", "xnfph1", "xnfph2", "gcon1", "gcon2", "pp")}
    keep["extab"] = _c32(extab)
    keep["cutoffh2"] = _c32(_t("hprof4", "cutoffh2").astype(F32))
    keep["cutoffh2plus"] = _c32(_t("hprof4", "cutoffh2plus").astype(F32))
    sb = hp.sofbet
    keep.update(prob=_c32(sb.prob), sc=_c32(sb.c), sd=_c32(sb.d), spp=_c32(sb.pp), sbeta=_c32(sb.beta))
    assert keep["prob"].numel() == 525 and keep["sc"].numel() == 35 and keep["sbeta"].numel() == 15
    assert keep["cutoffh2"].numel() == 91 and keep["cutoffh2plus"].numel() == 111
    o = SHD()
    for k, v in keep.items():
        setattr(o, k, v.data_ptr())
    o.dopph = dopph.data_ptr()
    o.nextab = keep["extab"].numel()
    o.c4000 = float(_folded_pow10(float(F32(-11.07) - F32(14.))))
    o._keep = keep
    o._dopph = dopph
    return o
