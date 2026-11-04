"""TEST-ONLY stand-in for the atlas9-variant physics (populations + continuous opacities): runs the original
Fortran routines (POPS/NELECT/NMOLEC/MOLEC/PFSAHA and the KAPP(0) continuum sources) from the oracle library
libatlas9.so through ctypes on the port's State, so the torch ATLAS9 iteration can run end to end before the
torch physics port lands. Implements atlas_torch.atlas9.physics.Physics.

The library instance is persistent: SAVEd state of the Fortran physics (NMOLEC starting guesses, ITEMP
caches, /XNMOL/, /XNSAVE/) evolves across calls exactly as in the Fortran program, provided the calls come
in the program's order. Before every call the COMMONs of the State are copied into the library; afterwards
the blocks the physics writes are read back.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ..atlas9.commons import layout
from ..atlas9.kapp import OPS
from .fortran_oracle import Oracle

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
_SKIP = {"free"}                      # READIN scratch, irrelevant to the physics


class OraclePhysics:
    def __init__(self, so_path: str | Path | None = None):
        self.o = Oracle(so_path or EXT / "oracle" / "libatlas9.so")
        self.calls = {"electrons": 0, "populations": 0, "continuum": 0}

    # ------------------------------------------------------------------ COMMON sync
    def push(self, s) -> None:
        for blk, members in layout().items():
            if blk in _SKIP:
                continue
            vals = {}
            for name, shape, dt in members:
                key = f"{blk}.{name}"
                if key not in s:
                    continue
                v = s[key]
                v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                if v.shape == (1,):
                    v = v[0]
                vals[name] = v
            self.o.set_common(blk, **vals)

    def pull(self, s, blocks) -> dict:
        out = {}
        for blk in blocks:
            c = self.o.common(blk)
            for name, shape, dt in layout()[blk]:
                key = f"{blk}.{name}"
                a = np.array(c[name], copy=True).reshape(shape)
                if dt.startswith("S"):
                    out[key] = a.astype(dt)
                elif dt == "f4":
                    out[key] = torch.from_numpy(a.astype(np.float32))
                elif dt == "i8":
                    out[key] = torch.from_numpy(a.astype(np.int64))
                else:
                    out[key] = torch.from_numpy(a.astype(np.float64))
        return out

    def _changed(self, s) -> dict:
        """Every canonical COMMON entry the Fortran changed (e.g. NMOLEC also leaves T/TK/TKEV with the roundoff
        of its own x*1.001/1.001*.999/.999 perturbations when IFEDNS=1)."""
        out = {}
        for k, v in self.pull(s, [b for b in layout() if b not in _SKIP]).items():
            old = s.get(k)
            if old is None:
                out[k] = v
                continue
            a = old.detach().cpu().numpy() if isinstance(old, torch.Tensor) else np.asarray(old)
            b = v.numpy() if isinstance(v, torch.Tensor) else v
            if a.shape != b.shape or not np.array_equal(a.astype(b.dtype) if b.dtype.kind != "S" else a, b,
                                                        equal_nan=b.dtype.kind == "f"):
                out[k] = v
        return out

    # ------------------------------------------------------------------ Physics interface
    def electrons(self, s) -> dict:
        """POPS(0.D0,1,XNE)."""
        self.calls["electrons"] += 1
        self.push(s)
        self.o.call("pops", 0.0, 1, self.o.common("state")["xne"])
        return self._changed(s)

    def populations(self, s) -> dict:
        """The main program's POPS/PFSAHA block (between POPS(0.D0,1,XNE) and PUTOUT(1))."""
        self.calls["populations"] += 1
        self.push(s)
        o = self.o
        ions, xnf, xnfp = o.common("ions"), o.common("xnf"), o.common("xnfp")
        pops = lambda code, mode, arr: o.call("pops", float(code), int(mode), arr)
        pops(1.01, 11, ions["xnfph"]); pops(2.02, 11, ions["xnfphe"]); pops(6.03, 11, xnfp["xnfpc"])
        pops(7.04, 11, xnfp["xnfpn"]); pops(8.05, 11, xnfp["xnfpo"]); pops(10.05, 11, xnfp["xnfpne"])
        pops(12.01, 11, xnfp["xnfpmg"]); pops(13.00, 11, xnfp["xnfpal"]); pops(14.01, 11, xnfp["xnfpsi"])
        pops(20.01, 11, xnfp["xnfpca"]); pops(26.00, 11, xnfp["xnfpfe"])
        ifmol = int(o.common("if")["ifmol"][0])
        if ifmol == 0:
            pops(1.01, 12, ions["xnfh"]); pops(2.02, 12, ions["xnfhe"]); pops(6.05, 12, xnf["xnfc"])
            pops(7.05, 12, xnf["xnfn"]); pops(8.05, 12, xnf["xnfo"]); pops(10.05, 12, xnf["xnfne"])
            pops(12.05, 12, xnf["xnfmg"]); pops(14.05, 12, xnf["xnfsi"]); pops(16.05, 12, xnf["xnfs"])
            pops(26.04, 12, xnf["xnffe"])
        if ifmol == 1:
            pops(106.00, 11, xnfp["xnfpch"]); pops(108.00, 11, xnfp["xnfpoh"])
            for code, arr in ((6.05, "xnfc"), (7.05, "xnfn"), (8.05, "xnfo"), (10.05, "xnfne"), (12.05, "xnfmg"),
                              (14.05, "xnfsi"), (16.05, "xnfs"), (26.04, "xnffe")):
                pops(code, 11, xnf[arr])
            part = np.zeros((99, 31), order="F")
            n = int(o.common("rhox")["nrhox"][0])
            for j in range(1, n + 1):
                J = j - 1
                o.call("pfsaha", j, 1, 1, 3, part)
                ions["xnfh"][J, 0] = ions["xnfph"][J, 0] * part[J, 0]
                ions["xnfh"][J, 1] = ions["xnfph"][J, 1]
                o.call("pfsaha", j, 2, 2, 13, part)
                ions["xnfhe"][J, 0] = ions["xnfphe"][J, 0] * part[J, 0]
                ions["xnfhe"][J, 1] = ions["xnfphe"][J, 1] * part[J, 1]
                ions["xnfhe"][J, 2] = ions["xnfphe"][J, 2]
                for iz, nion, arr in ((6, 6, "xnfc"), (7, 6, "xnfn"), (8, 6, "xnfo"), (10, 6, "xnfne"),
                                      (12, 6, "xnfmg"), (14, 6, "xnfsi"), (16, 6, "xnfs"), (26, 5, "xnffe")):
                    o.call("pfsaha", j, iz, nion, 13, part)
                    for i in range(nion):
                        xnf[arr][J, i] = xnf[arr][J, i] * part[J, i]
        return self._changed(s)

    def continuum(self, s, fq: dict) -> dict:
        """KAPP(0,...) per frequency; returns the 25 /OPS/ arrays [F, n] (KAPP's own combination is the port's)."""
        self.calls["continuum"] += 1
        self.push(s)
        o = self.o
        fc, ops = o.common("freq"), o.common("ops")
        n = int(o.common("rhox")["nrhox"][0])
        F = fq["freq"].shape[0]
        out = {k: np.zeros((F, n)) for k in OPS}
        for f in range(F):
            fc["freq"][0] = float(fq["freq"][f]); fc["freqlg"][0] = float(fq["freqlg"][f])
            fc["ehvkt"][:n] = fq["ehvkt"][f].numpy(); fc["stim"][:n] = fq["stim"][f].numpy()
            fc["bnu"][:n] = fq["bnu"][f].numpy()
            o.call("kapp", 0, 0, 0.0)
            for k in OPS:
                out[k][f] = ops[k][:n]
        return {k: torch.from_numpy(v) for k, v in out.items()}


# ---------------------------------------------------------------------- test helpers (oracle tests)

def run_fortran(o: Oracle, s, routine: str, *args, unit_statics: dict | None = None, restype=None):
    """Copy the State's COMMONs (and `unit_statics` {unit: {var: array}}) into the oracle library, call
    `routine`, and return (result, {key: value}) with every canonical COMMON after the call plus the
    statics of the units in `unit_statics` as "save.<unit>.<var>"."""
    ph = OraclePhysics.__new__(OraclePhysics)
    ph.o = o
    ph.push(s)
    for unit, vals in (unit_statics or {}).items():
        sv = o.saved(unit)
        for k, v in vals.items():
            v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            dst = sv[k]
            if dst.shape == (1,) and v.size == 1:
                dst[0] = v.reshape(-1)[0]
            else:
                dst[tuple(slice(0, d) for d in v.shape)] = v.astype(dst.dtype)
    r, outs = o.call(routine, *args, restype=restype)
    blocks = [b for b in layout() if b not in _SKIP]
    res = ph.pull(s, blocks)
    for unit in (unit_statics or {}):
        for k, v in o.saved(unit).items():
            res[f"save.{unit}.{k}"] = torch.from_numpy(np.array(v, copy=True))
    return r, res


_SNAP: dict = {}


def _rw_sections(so_path: str):
    from elftools.elf.elffile import ELFFile
    out = []
    with open(so_path, "rb") as fh:
        for sec in ELFFile(fh).iter_sections():
            if sec.name in (".data", ".bss"):
                out.append((sec.header.sh_addr, sec.header.sh_size))
    return out


def mem_snapshot(o: Oracle) -> list[bytes]:
    """zlib-compressed bytes of the library's .data/.bss (all COMMONs and SAVEd locals of the Fortran)."""
    import ctypes
    import zlib
    return [zlib.compress(ctypes.string_at(o.base + a, n), 1) for a, n in _rw_sections(o.so_path)]


def mem_restore(o: Oracle, snap: list[bytes]) -> None:
    import ctypes
    import zlib
    for (a, n), b in zip(_rw_sections(o.so_path), snap):
        ctypes.memmove(o.base + a, zlib.decompress(b), n)


def iteration_snapshots(niter: int = 1) -> dict:
    """Run the torch ATLAS9 on the traced solar inputs with OraclePhysics for `niter` iterations and return
    {(iteration, hook name): State} (cached per process). Test fixture data for the routine oracle tests."""
    if niter in _SNAP:
        return _SNAP[niter]
    from ..atlas9.main import run
    w = EXT / "traces" / "sun_atlas9" / "w"
    files = {1: w / "fort.1", 2: w / "fort.2", 3: w / "fort.3", 9: w / "fort.9"}
    snaps = {}
    counter = {"it": 0}

    ph = OraclePhysics()

    def hook(name, st):
        if name == "populations":
            counter["it"] += 1
        snaps[(counter["it"], name)] = st
        if name in ("frequencies", "convec", "tcorr3"):
            snaps[(counter["it"], name, "mem")] = mem_snapshot(ph.o)

    r = run((w / "stdin.txt").read_text(), files, ph, max_iterations=niter, hook=hook)
    snaps["result"] = r
    _SNAP[niter] = snaps
    return snaps


def fortran_frequency_integration(o: Oracle, s, workdir: str | Path | None = None) -> dict:
    """The main program's frequency loop (DO 25 NU, IFOP(15)=1 line path, IFSURF=0) driven from Python with the
    Fortran KAPP/JOSH/LINOP/TCORR/RADIAP/ROSS via ctypes, starting from State `s` (as at hook 'frequencies',
    i.e. after TCORR/ROSS/RADIAP(1)). Returns the accumulated COMMONs and statics (full-precision reference).
    LINOP opens 'fort.9' relative to the working directory: the call runs inside `workdir` (default: the
    traced solar run directory)."""
    old = os.getcwd()
    os.chdir(workdir or EXT / "traces" / "sun_atlas9" / "w")
    try:
        return _fortran_frequency_integration(o, s)
    finally:
        os.chdir(old)


def _fortran_frequency_integration(o: Oracle, s) -> dict:
    import ctypes
    import math
    ph = OraclePhysics.__new__(OraclePhysics)
    ph.o = o
    ph.push(s)
    tc = o.saved("tcorr")
    for k in ("rjmins", "rdabh", "rdiagj", "ddel", "oldt1"):
        if f"save.tcorr.{k}" in s:
            tc[k][:] = s[f"save.tcorr.{k}"].numpy()
    o.saved("radiap")["h"][:] = s["save.radiap.h"].numpy()
    c = {b: o.common(b) for b in ("freq", "freset", "rhox", "temp", "ops", "optot", "taushj", "abtot", "cont",
                                  "if", "teff", "pzero", "wavey")}
    n = int(c["rhox"]["nrhox"][0])
    ifcorr = int(c["if"]["ifcorr"][0]); ifscat = int(c["if"]["ifscat"][0]); ifsurf = int(c["if"]["ifsurf"][0])
    nlte = int(o.common("depart")["nlteon"][0])
    assert ifsurf == 0 and nlte == 0 and int(c["wavey"]["ifwave"][0]) == 0
    hkt = c["temp"]["hkt"]
    f32 = lambda v: float(np.float32(v))
    R9995, R998, R0001 = f32(0.9995), f32(0.998), f32(0.0001)
    N = ctypes.c_int(0); NSTEPS = ctypes.c_int(0); STEPWT = ctypes.c_double(0.0)

    def josh():
        o.call("josh", ifscat, ifsurf, float(c["freq"]["freq"][0]))

    def integrals(rcowt):
        if ifcorr == 1:
            o.call("tcorr", 2, rcowt)
        o.call("radiap", 2, rcowt)
        o.call("ross", 2, rcowt)

    tau, sn, hn, jn, jm = (c["taushj"][k] for k in ("taunu", "snu", "hnu", "jnu", "jmins"))
    ab, al = c["abtot"]["abtot"], c["abtot"]["alpha"]
    cont = c["cont"]
    for nu in range(int(c["freset"]["nulo"][0]), int(c["freset"]["nuhi"][0]) + 1):
        freq = float(c["freset"]["freset"][nu - 1]); rco = float(c["freset"]["rcoset"][nu - 1])
        c["freq"]["freq"][0] = freq
        c["freq"]["freqlg"][0] = math.log(freq)
        for j in range(n):
            e = math.exp(-freq * float(hkt[j]))
            c["freq"]["ehvkt"][j] = e
            c["freq"]["stim"][j] = 1.0 - e
            f15 = freq / 1.0e15
            c["freq"]["bnu"][j] = 1.47439e-2 * (f15 * f15 * f15) * e / (1.0 - e)
        N.value = 0
        o.call("kapp", N, NSTEPS, STEPWT)
        josh()
        for a, b in ((ab, "abtotc"), (al, "alphac"), (tau, "taunuc"), (sn, "snuc"), (hn, "hnuc"), (jn, "jnuc"),
                     (jm, "jminsc")):
            cont[b][:n] = a[:n]
        cont["residc"][:n] = 0.0
        sumwt = 0.0
        cont["residc"][0] = 0.0
        N.value = 1
        state = 72
        while True:
            if state == 24:
                N.value += 1
                if cont["residc"][0] > R9995:
                    state = 79
                elif any(cont["residc"][j] < R998 for j in range(n)):
                    state = 72
                else:
                    state = 79
            if state == 79:
                STEPWT.value = 1.0 - sumwt
                N.value += 1
                if STEPWT.value < R0001:
                    STEPWT.value = 0.0
                N.value = NSTEPS.value + 1
                cont["residc"][0] = 1.0
                for a, b in ((ab, "abtotc"), (al, "alphac"), (tau, "taunuc"), (sn, "snuc"), (hn, "hnuc"),
                             (jn, "jnuc"), (jm, "jminsc")):
                    a[:n] = cont[b][:n]
            elif state == 72:
                o.call("linop", N, NSTEPS, STEPWT)
                c["optot"]["aline"][:n] = c["ops"]["alines"][:n]
                josh()
                cont["residc"][0] = 1.0
            for j in range(n):
                cont["residc"][j] = 1.0
                if cont["hnuc"][j] > 0.0:
                    cont["residc"][j] = hn[j] / cont["hnuc"][j]
            sumwt = sumwt + STEPWT.value
            rcowt = rco * STEPWT.value
            if STEPWT.value != 0.0:
                integrals(rcowt)
            if N.value < NSTEPS.value:
                state = 24
                continue
            if N.value == NSTEPS.value:
                state = 79
                continue
            break
    out = ph.pull(s, [b for b in layout() if b not in _SKIP])
    for k in ("rjmins", "rdabh", "rdiagj"):
        out[f"save.tcorr.{k}"] = torch.from_numpy(np.array(tc[k], copy=True))
    out["save.radiap.h"] = torch.from_numpy(np.array(o.saved("radiap")["h"], copy=True))
    return out
