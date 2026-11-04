"""The KAPPA9 program: Rosseland mean opacity tables (kapk{v}.dat) from an ODF, batched over the T/P grid.

Fortran (kappa9.for, PROGRAM KAPPA9), as run by BasicATLAS for v = 0,1,2,4,8 km/s:
  READIN(1) (control cards; READ MOLECULES from unit 2; the model deck only leaves HCKT behind, see readin.py)
  frequency set: the 337 BIG frequencies are shifted to NU = 90..426 and 89 extra frequencies prepended
      (only RCOSET(90) = (FRESET(89)-FRESET(91))*.5 of that survives: the integration runs over NU = 90..NUHI);
  57 temperatures T = EXP10(TABT) with the statement function EXP10(X) = EXP(X*2.302585092994E0) (REAL*4
      constant), TK, HKT, TKEV, TLOG (HCKT is *not* updated);
  for each of the 42 pressures K (ITEMP = K, P = EXP10(TABP(K)) on all layers): POPS block; then for every
      frequency: EHVKT, STIM, BNU; KAPP(N) for N = 1..NSTEPS (12 ODF sub-steps, LINOP); per step
          ABTOT = ACONT+ALINE+SIGMAC+SIGMAL,  DBDT = BNU*FREQ*HKT/T/STIM,
          ABROSS = ABROSS + DBDT/ABTOT*(RCO*STEPWT)
      ABROSS = (4.*5.6697D-5/3.14159)*T**3/ABROSS,  KABROS = INT(LOG10(ABROSS)*1000.)  -> WRITE(7,77)
  (the opacity-component printout after the step loop is dead code: N always equals NSTEPS there).

Port: the T x P grid (57 x 42 = 2394 points) is one batch of layers for LINOP, KAPP and the Rosseland sums;
the frequency/step accumulation keeps the Fortran's sequential order (bit-exact sums). Populations and the
continuum sources are evaluated per pressure column (the Fortran's NMOLEC chains its starting guess over the
57 temperatures of a column). They do not depend on VTURB, so run_workdir computes them once for the five ODFs.

Physics backend (populations + continuum sources): `physics=` or $ATLAS_TORCH_KAPPA9_PHYSICS
  'torch'  atlas_torch.atlas9.physics (variant 'kappa9'), the torch port -- TorchKappa9Physics: populations per
           pressure column, then the continuum once per frequency on all 2394 grid points;
  'oracle' TEST-ONLY adapter over the Fortran routines of libkappa9.so (atlas_torch.testing.kappa9_oracle_physics),
           per column (Fortran COMMONs hold at most kw = 99 layers); ~20x faster today;
  'auto'   (default) torch when atlas_torch.atlas9.physics provides populations/continuum, else the oracle
           adapter (with a warning).
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from ..fnum import f32
from ..tables import table
from . import linop as _linop
from .kapp import kapp_first
from .kapreadts import kapreadts
from .readin import readin

SRC = "kappa9.for"
VTURBS = (0, 1, 2, 4, 8)
NT, NP = 57, 42
_LN10_F32 = f32(2.302585092994)        # EXP10 statement function of the main program (REAL*4 constant)
PHYSICS_ENV = "ATLAS_TORCH_KAPPA9_PHYSICS"


# --------------------------------------------------------------------------------------------- grid / freqs

def tabt() -> np.ndarray:
    return table(SRC, "KAPPA9", "TABT").astype(np.float64)


def tabp() -> np.ndarray:
    return table(SRC, "KAPPA9", "TABP").astype(np.float64)


def exp10(x):
    return torch.exp(torch.as_tensor(x, dtype=torch.float64) * _LN10_F32)


def grid_temperatures(device=None) -> dict:
    t = exp10(torch.as_tensor(tabt(), device=device))
    tk = 1.38054e-16 * t
    return {"t": t, "tk": tk, "hkt": 6.6256e-27 / tk, "tkev": 8.6171e-5 * t, "tlog": torch.log(t)}


def grid_pressures(device=None) -> torch.Tensor:
    return exp10(torch.as_tensor(tabp(), device=device))


def frequency_set(rd) -> tuple[np.ndarray, np.ndarray]:
    """(FREQ, RCO) for NU = 90..NUHI after the main program's shift of /FRESET/ (IFWAVE = 0)."""
    if int(rd["wavey.ifwave"].reshape(-1)[0]) != 0:
        raise NotImplementedError("IFWAVE=1 frequency sets are not ported")
    freset = rd["freset.freset"].numpy().copy()
    rcoset = rd["freset.rcoset"].numpy().copy()
    numnu = int(rd["freset.numnu"].reshape(-1)[0])
    for nu in range(1, numnu + 1):                       # DO 561 (top down: no aliasing)
        freset[numnu + 90 - nu - 1] = freset[numnu + 1 - nu - 1]
        rcoset[numnu + 90 - nu - 1] = rcoset[numnu + 1 - nu - 1]
    numnu += 89
    nuhi = numnu
    for nu in range(1, 90):
        freset[nu - 1] = 3.364687e16 + (89 - nu) * 0.5e15
        rcoset[nu - 1] = f32(0.5e15)
    rcoset[0] = f32(0.25e15)
    rcoset[88] = (freset[87] - freset[89]) * 0.5
    rcoset[89] = (freset[88] - freset[90]) * 0.5
    return freset[89:nuhi].copy(), rcoset[89:nuhi].copy()


def freq_arrays(freq: torch.Tensor, g: dict) -> dict:
    """/FREQ/ per frequency (dim 0) and temperature (dim 1): EHVKT, STIM, BNU; and DBDT of the Rosseland sum."""
    f = freq.view(-1, 1)
    freq15 = f / 1.0e15
    ehvkt = torch.exp(-f * g["hkt"].view(1, -1))
    stim = 1.0 - ehvkt
    bnu = 1.47439e-02 * (freq15 * freq15 * freq15) * ehvkt / stim
    dbdt = bnu * f * g["hkt"].view(1, -1) / g["t"].view(1, -1) / stim
    return {"freqlg": torch.log(freq), "ehvkt": ehvkt, "stim": stim, "bnu": bnu, "dbdt": dbdt}


# --------------------------------------------------------------------------------------------- physics

class TorchKappa9Physics:
    """Adapter to atlas_torch.atlas9.physics (variant 'kappa9'): populations(state) and continuum(state, freq).

    Expected contract (the physics package's announced entry points): `populations(state, variant=...)`
    returns the population COMMON entries ("state.xne", "state.xnatom", "state.rho", "ions.xnfph", "xnfp.*",
    "xnf.*", ...) for the layers of `state` as KAPPA9's POPS block leaves them, and
    `continuum(state, freq, variant=...)` the 25 "ops.*" arrays of KAPP's N=1 opacity sources (all IFOP
    sources except LINOP/LINSOP, which this program computes). Raises NotImplementedError until both exist.

    batched = True: populations are evaluated per pressure column (NMOLEC chains its starting guess over the
    57 temperatures of a column), then the continuum is evaluated once per frequency on all 42 x 57 points
    (K-major layers; per-layer READIN leftovers such as the deck's HCKT(J) are tiled over the columns)."""

    name = "torch"
    batched = True

    # atlas9.physics.libm.EXACT (default True) evaluates exp/log elementwise with Python's math module, which
    # raises OverflowError where the Fortran gets +Inf and discards the value (NMOLEC's EQUILJ for layers with
    # T > 10000 K, which KAPPA9's grid reaches; ATLAS9 models do not). Native torch math (EXACT = False) is used
    # for the physics calls of this program (scoped, restored afterwards); it agrees with the Fortran to ~1e-14
    # and gives byte-identical kapk tables. Set exact_libm = True to use the package default.
    exact_libm = False

    def _call(self, fn, *a, **kw):
        try:
            from ..atlas9.physics import libm
        except ImportError:
            return fn(*a, **kw)
        old = libm.EXACT
        libm.EXACT = self.exact_libm
        try:
            return fn(*a, **kw)
        finally:
            libm.EXACT = old

    def __init__(self):
        try:
            from ..atlas9 import physics as ph   # noqa: F401
        except ImportError as e:
            raise NotImplementedError(f"atlas_torch.atlas9.physics not available ({e})") from e
        if not (hasattr(ph, "populations") and hasattr(ph, "continuum")):
            raise NotImplementedError("atlas_torch.atlas9.physics does not provide populations/continuum yet")
        self.ph = ph
        self.rd = None
        self.cur = None
        self.cols = []
        self.state = None

    def load(self, rd) -> None:
        """Program-start State (atlas9.commons: zeroed COMMONs + BLOCK DATA images, kw = 99 shapes) overlaid
        with the READIN result."""
        from ..state import State
        try:
            from ..atlas9.commons import initial_state
            base = initial_state()
        except ImportError:
            base = State()
        for k, shape in (("xnmol.xnmol", (99, 160)), ("xnsave.xnsave", (99, 25))):   # KAPPA9 COMMONs (.bss)
            base.setdefault(k, torch.zeros(shape, dtype=torch.float64))
        for k, v in rd.items():
            cur = base.get(k)
            if isinstance(cur, torch.Tensor) and isinstance(v, torch.Tensor) and cur.shape != v.shape \
                    and cur.numel() >= v.numel() and cur.dim() == v.dim():
                c = cur.clone()
                c[tuple(slice(0, n) for n in v.shape)] = v.to(c.dtype)
                v = c
            base[k] = v
        self.rd = rd
        self.cur = State(base)
        self.cols = []
        self.state = None

    def populations(self, k: int, grid: dict) -> dict:
        """Column K: /TEMP/ (HCKT stays READIN's: Fortran quirk), ITEMP = K, P, NRHOX; the State (incl. SAVE
        entries such as POPS' ITEMP1) carries over from the previous column as in the Fortran."""
        n = int(grid["t"].shape[0])
        s = self.cur
        upd = {}
        for v in ("t", "tk", "hkt", "tkev", "tlog"):
            a = s[f"temp.{v}"].clone()
            a[:n] = grid[v]
            upd[f"temp.{v}"] = a
        a = s["state.p"].clone()
        a[:n] = grid["p"]
        upd["state.p"] = a
        upd["temp.itemp"] = torch.tensor([k])
        upd["rhox.nrhox"] = torch.tensor([n])
        s = s.updated(upd)
        out = self._call(self.ph.populations, s, variant="kappa9")
        self.cur = s.updated(out)
        self.cols.append((n, self.cur))
        return out

    def finish_columns(self) -> None:
        """One State over all grid points: per-layer arrays (first dim kw = 99) are the columns' first n layers
        concatenated (K-major); everything else is taken from the last column."""
        from ..state import State
        n = self.cols[0][0]
        last = self.cols[-1][1]
        s = State(last)
        for key, v in last.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == 99 and key not in (
                    "freset.freset", "freset.rcoset") and all(isinstance(c.get(key), torch.Tensor)
                                                             and c[key].shape == v.shape for _, c in self.cols):
                s[key] = torch.cat([c[key][:n] for _, c in self.cols], 0)
        s["rhox.nrhox"] = torch.tensor([n * len(self.cols)])
        self.state = s

    def continuum(self, freq, freqlg, ehvkt, stim, bnu) -> dict:
        s = self.state.updated({"freq.freq": torch.tensor([float(freq)]), "freq.freqlg": torch.tensor([float(freqlg)]),
                                "freq.ehvkt": ehvkt, "freq.stim": stim, "freq.bnu": bnu})
        n = s.iscalar("rhox.nrhox")
        out = self._call(self.ph.continuum, s, None, variant="kappa9", derive_freq=False)
        return {k[4:]: v[:n] for k, v in out.items() if k.startswith("ops.")}


def get_physics(physics=None):
    if physics is not None and not isinstance(physics, str):
        return physics
    name = (physics or os.environ.get(PHYSICS_ENV, "auto")).strip().lower()
    if name in ("torch", "auto"):
        try:
            return TorchKappa9Physics()
        except NotImplementedError:
            if name == "torch":
                raise
    if name in ("oracle", "auto"):
        from ..testing.kappa9_oracle_physics import OracleKappa9Physics
        if name == "auto":
            warnings.warn("KAPPA9: torch physics (atlas_torch.atlas9.physics) not available; using the TEST-ONLY "
                          "Fortran oracle adapter for populations and continuum opacities", stacklevel=2)
        return OracleKappa9Physics()
    raise ValueError(f"{PHYSICS_ENV} must be torch, oracle or auto, not {name!r}")


# --------------------------------------------------------------------------------------------- continuum

_CONT_KEYS = ("acont", "sigmac", "ahline", "axline", "sigxl", "siglin", "alines")


def continuum_tables(rd, physics=None, device=None) -> dict:
    """Velocity-independent part: per frequency and grid point the KAPP N=1 quantities that enter ABTOT.
    Returns tensors [NF, 42, 57] plus the frequency data."""
    phys = get_physics(physics)
    phys.load(rd)
    g = grid_temperatures(device)
    p = grid_pressures(device)
    freq_np, rco_np = frequency_set(rd)
    freq = torch.as_tensor(freq_np, device=device)
    fa = freq_arrays(freq, g)
    nf = freq.numel()
    out = {k: torch.zeros((nf, NP, NT), dtype=torch.float64, device=device) for k in _CONT_KEYS}

    def store(inu, sl, ops, bnu):
        ops = {kk: torch.as_tensor(v, dtype=torch.float64, device=device) for kk, v in ops.items()}
        first = kapp_first(ops, bnu)
        for kk in ("acont", "sigmac"):
            out[kk][inu][sl] = first[kk].reshape(out[kk][inu][sl].shape)
        for kk in ("ahline", "axline", "sigxl", "siglin", "alines"):
            out[kk][inu][sl] = ops[kk].reshape(out[kk][inu][sl].shape)

    batched = getattr(phys, "batched", False)
    from .printout import NmolecCapture, column_record
    printout_columns = []                           # what NMOLEC prints per column (kapm40k2.out, printout.py)
    for k in range(1, NP + 1):
        gk = dict(g)
        gk["p"] = p[k - 1].expand(NT).clone()
        with NmolecCapture() as cap:
            phys.populations(k, gk)                 # ITEMP = K: the column's POPS block
        printout_columns.append(column_record(phys, cap, NT))
        if batched:
            continue
        for inu in range(nf):                       # per column (Fortran COMMONs: at most kw = 99 layers)
            store(inu, k - 1, phys.continuum(float(freq_np[inu]), float(fa["freqlg"][inu]), fa["ehvkt"][inu],
                                             fa["stim"][inu], fa["bnu"][inu]), fa["bnu"][inu])
    if batched:                                     # all 42 x 57 grid points as layers, one call per frequency
        phys.finish_columns()
        tile = lambda a: a.view(1, NT).expand(NP, NT).reshape(NP * NT)
        for inu in range(nf):
            bnu = tile(fa["bnu"][inu])
            store(inu, slice(None), phys.continuum(float(freq_np[inu]), float(fa["freqlg"][inu]),
                                                   tile(fa["ehvkt"][inu]), tile(fa["stim"][inu]), bnu), bnu)
    out.update({"freq": freq, "rco": torch.as_tensor(rco_np, device=device), "dbdt": fa["dbdt"], "grid": g, "p": p,
                "physics": getattr(phys, "name", type(phys).__name__),
                "printout_columns": printout_columns if all(c is not None for c in printout_columns) else None})
    return out


# --------------------------------------------------------------------------------------------- Rosseland

def rosseland(cont: dict, kap: torch.Tensor, rd) -> torch.Tensor:
    """KABROS [42, 57] (int64) for one ODF (kap: int16 [328, 57, 25, 12])."""
    g, p = cont["grid"], cont["p"]
    dev = g["t"].device
    tauscat = float(rd["if.tauscat"].reshape(-1)[0])
    ifop = rd["ifop.ifop"]
    lines_on = int(ifop[14]) == 1
    if int(ifop[15]) == 1:
        pass  # LINSOP is a dummy routine
    nf = cont["freq"].numel()
    L = NP * NT
    tlog = g["tlog"].view(1, NT).expand(NP, NT).reshape(L)
    pp = p.view(NP, 1).expand(NP, NT).reshape(L)
    if tauscat > 0.0:
        raise NotImplementedError("TAUSCAT > 0 needs TAUROS of the grid points (not ported)")
    wt = _linop.step_weights().to(dev)
    if lines_on:
        co = _linop.setup(tlog, pp)
        wlbeg, wlend = _linop.wl_bounds(rd["biglit.wavedf"].numpy())
        iwl = torch.tensor(_linop.iwl_sequence(cont["freq"].tolist(), wlbeg, wlend), device=dev)
        a = _linop.line_opacity(kap.to(dev), iwl, co).to(torch.float64)        # [NF, 12, L]
        fscat = co["fscat"]
        alines = (1.0 - fscat) * a
        siglin = fscat * a
        nsteps = torch.where(iwl > 0, 12, 1)
    else:
        alines = torch.zeros((nf, 1, L), dtype=torch.float64, device=dev)
        siglin = torch.zeros_like(alines)
        nsteps = torch.ones(nf, dtype=torch.int64, device=dev)
    c = {k: cont[k].reshape(nf, 1, L) for k in _CONT_KEYS}
    # steps N >= 2 (and the 12-step frequencies at N = 1) take ALINES/SIGLIN from LINOP; a frequency without
    # ODF data (WAVE < WLBEG(1)) keeps KAPP's N = 1 values (ALINES = SIGLIN = 0 from the /OPS/ zeroing).
    has = (nsteps == 12).view(nf, 1, 1)
    alines = torch.where(has, alines, c["alines"])
    siglin = torch.where(has, siglin, c["siglin"])
    aline = c["ahline"] + alines + c["axline"]
    sigmal = siglin + c["sigxl"]
    abtot = c["acont"] + aline + c["sigmac"] + sigmal
    dbdt = cont["dbdt"].view(nf, 1, 1, NT).expand(nf, 1, NP, NT).reshape(nf, 1, L)
    S = alines.shape[1]
    stepwt = torch.where(has.view(nf, 1), wt.view(1, -1)[:, :S], torch.ones((), dtype=torch.float64, device=dev))
    rcowt = (cont["rco"].view(nf, 1) * stepwt).view(nf, S, 1)                 # RCOWT = RCO*STEPWT
    term = dbdt / abtot * rcowt                                               # [NF, 12, L]
    abross = torch.zeros(L, dtype=torch.float64, device=dev)
    ns = nsteps.tolist()
    for inu in range(nf):                                                     # Fortran order: NU, then N
        for n in range(ns[inu]):
            abross = abross + term[inu, n]
    t = g["t"].view(1, NT).expand(NP, NT).reshape(L)
    abross = (4.0 * 5.6697e-5 / f32(3.14159)) * (t * t * t) / abross
    kab = torch.trunc(torch.log10(abross) * 1000.0).to(torch.int64)
    return kab.view(NP, NT)


# --------------------------------------------------------------------------------------------- writer

def _f63(x: float) -> str:
    s = f"{x:6.3f}"
    if len(s) > 6:
        s = "-" + s[2:] if s.startswith("-0.") else s
    return s if len(s) <= 6 else "******"


def _i5(v: int) -> str:
    s = f"{int(v):5d}"
    return s if len(s) <= 5 else "*****"


def kapk_header(tt: np.ndarray | None = None, tp: np.ndarray | None = None) -> str:
    """WRITE(7,75) TABT, TABP."""
    tt = tabt() if tt is None else tt
    tp = tabp() if tp is None else tp
    f = [_f63(float(x)) for x in tt]
    g = [_f63(float(x)) for x in tp]
    L = ["      DATA TABT/    " + "".join(v + "," for v in f[0:7])]
    i = 7
    for r in range(1, 6):
        L.append(f"     {r}" + "".join(v + "," for v in f[i:i + 9])); i += 9
    L.append("     6" + "".join(v + "," for v in f[i:i + 4]) + f[i + 4] + "/")
    L.append("      DATA TABP/    " + "".join(v + "," for v in g[0:7]))
    i = 7
    for r in range(1, 4):
        L.append(f"     {r}" + "".join(v + "," for v in g[i:i + 9])); i += 9
    L.append("     4" + "".join(v + "," for v in g[i:i + 7]) + g[i + 7] + "/")
    return "\n".join(L) + "\n"


def kapk_block(k: int, kab: np.ndarray, tabp_k: float) -> str:
    """WRITE(7,77) K1,K2,TABP(K),(KABROS(J),J=1,10),TABP(K),... for pressure K (1-based)."""
    tp = _f63(tabp_k)
    k1, k2 = k // 10, k - (k // 10) * 10
    L = ["      DATA KTAB" + f"{k1:1d}{k2:1d}" + "/" + " " * 54 + tp]
    for r in range(1, 6):
        L.append(f"     {r}" + "".join(_i5(v) + "," for v in kab[(r - 1) * 10:r * 10]) + " " * 6 + tp)
    L.append("     6" + "".join(_i5(v) + "," for v in kab[50:56]) + _i5(kab[56]) + "/" + " " * 24 + tp)
    return "\n".join(L) + "\n"


def kapk_text(kabros: torch.Tensor | np.ndarray) -> str:
    kab = np.asarray(kabros.cpu() if isinstance(kabros, torch.Tensor) else kabros)
    tp = tabp()
    return kapk_header() + "".join(kapk_block(k + 1, kab[k], float(tp[k])) for k in range(NP))


# --------------------------------------------------------------------------------------------- drivers

def control_from_com(com_path: str | Path) -> str:
    """The heredoc (KAPPA9 stdin) of a BasicATLAS kappa9v{v}.com command file."""
    lines = Path(com_path).read_text().splitlines()
    out, inside = [], False
    for ln in lines:
        if not inside and "<<EOF" in ln:
            inside = True
            continue
        if inside:
            if ln.strip() == "EOF":
                break
            out.append(ln)
    if not inside:
        raise ValueError(f"no <<EOF heredoc in {com_path}")
    return "\n".join(out) + "\n"


def default_molecules() -> Path:
    from ..compat.settings import basicatlas_dir
    return Path(basicatlas_dir()) / "data" / "dfsynthe_files" / "molecules.dat"


def _deck_key(control: str, molecules: Path) -> str:
    body = "\n".join(ln for ln in control.splitlines() if not re.match(r"^\s*VTURB\b", ln))
    return hashlib.sha256(body.encode() + b"\0" + Path(molecules).read_bytes()).hexdigest()


_CONT_CACHE: dict = {}


def _continuum_cached(control: str, molecules: Path, physics=None, device=None):
    key = (_deck_key(control, molecules), str(device),
           physics if physics is None or isinstance(physics, str) else id(physics), os.environ.get(PHYSICS_ENV))
    if key not in _CONT_CACHE:
        rd = readin(control, molecules)
        _CONT_CACHE.clear()
        _CONT_CACHE[key] = (rd, continuum_tables(rd, physics, device))
    return _CONT_CACHE[key]


def kappa9(odf_path: str | Path, vturb: int | None = None, control: str | None = None,
           molecules: str | Path | None = None, physics=None, device=None, cont: dict | None = None) -> str:
    """One KAPPA9 run: returns the kapk text (fort.7) for the ODF `odf_path` (p00big{v}.bdf).

    control: the stdin cards; default: kappa9v{vturb}.com next to the ODF (BasicATLAS layout).
    cont: precomputed continuum_tables (velocity independent) to reuse."""
    odf_path = Path(odf_path)
    if control is None:
        if vturb is None:
            m = re.search(r"big(\d+)\.bdf$", odf_path.name)
            vturb = int(m.group(1)) if m else None
        if vturb is None:
            raise ValueError("need control cards or vturb to locate kappa9v{v}.com")
        control = control_from_com(odf_path.parent / f"kappa9v{int(vturb)}.com")
    molecules = Path(molecules) if molecules is not None else default_molecules()
    if cont is None:
        rd, cont = _continuum_cached(control, molecules, physics, device)
    else:
        rd = readin(control, molecules)
    kap = _linop.odf_kap(odf_path)
    return kapk_text(rosseland(cont, kap, rd))


def run_workdir(odf_dir: str | Path, workdir: str | Path, molecules: str | Path | None = None,
                physics=None, device=None) -> dict:
    """BasicATLAS-style KAPPA9 work dir: kapk{0,1,2,4,8}.dat, stdin_v{v}.txt and kappa.ros (KAPREADTS)."""
    odf_dir, workdir = Path(odf_dir), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    molecules = Path(molecules) if molecules is not None else default_molecules()
    t0 = time.perf_counter()
    info = {"timing": {}}
    controls = {}
    for v in VTURBS:
        com = odf_dir / f"kappa9v{v}.com"
        controls[v] = control_from_com(com) if com.exists() else (workdir / f"stdin_v{v}.txt").read_text()
        (workdir / f"stdin_v{v}.txt").write_text(controls[v])
    cont_by_key = {}
    for v in VTURBS:
        key = _deck_key(controls[v], molecules)
        if key not in cont_by_key:
            t1 = time.perf_counter()
            rd = readin(controls[v], molecules)
            cont_by_key[key] = (rd, continuum_tables(rd, physics, device))
            info["timing"][f"continuum_v{v}"] = time.perf_counter() - t1
            info["physics"] = cont_by_key[key][1]["physics"]
        rd, cont = cont_by_key[key]
        t1 = time.perf_counter()
        kab = rosseland(cont, _linop.odf_kap(odf_dir / f"p00big{v}.bdf"), readin(controls[v], molecules))
        (workdir / f"kapk{v}.dat").write_text(kapk_text(kab))
        info["timing"][f"rosseland_v{v}"] = time.perf_counter() - t1
    kapreadts([workdir / f"kapk{v}.dat" for v in VTURBS], workdir / "kappa.ros")
    info["timing"]["total"] = time.perf_counter() - t0
    return info


def kappa9_program(workdir, stdin_text, unit_files):
    """compat.runners program interface: fort.9 = ODF, fort.2 = molecules.dat -> fort.7 (kapk text)."""
    workdir = Path(workdir)
    odf = Path(unit_files.get(9, workdir / "fort.9"))
    mol = Path(unit_files.get(2, workdir / "fort.2"))
    rd, cont = _continuum_cached(stdin_text, mol)
    kab = rosseland(cont, _linop.odf_kap(odf), rd)
    (workdir / "fort.7").write_text(kapk_text(kab))
    from .printout import kappa9_stdout
    return kappa9_stdout(stdin_text, readin(stdin_text, mol), cont, kab)      # kapm40k2.out


PROGRAMS = {"kappa9": kappa9_program}
