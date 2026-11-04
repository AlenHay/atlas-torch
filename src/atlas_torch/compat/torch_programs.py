"""Thin adapters: the torch/Python ports of BasicATLAS's programs behind the :mod:`atlas_torch.compat.runners`
program interface ``fn(workdir, stdin_text, unit_files) -> stdout_text``.

ATLAS9 (``atlas9mem``) lives in :mod:`atlas_torch.atlas9.program` and KAPPA9 in
:mod:`atlas_torch.kappa9.kappa9_prog`; this module wires the rest. Each adapter reads the units the Fortran
program reads (``unit_files[N]``, else ``workdir/fort.N``) and writes the units it writes as ``workdir/fort.N``.

Standard output: reproduced where something downstream parses it (``synbeg`` -- validate_run() reads the echo of
the SYNBEG card from ``synbeg.out``; ``synthe`` and ``spectrv`` -- their ports produce it) and for the DFSYNTHE
chain, whose printouts BasicATLAS's meta()/validate_run() parse (``xnfdf.out``, ``kapm40k2.out``: READIN's
abundance listing and 0XSCALE): ``xnfdf``, ``dfsynthe`` (dfp00t*.out) -- atlas_torch.dfsynthe.printouts -- and
``kappa9`` (atlas_torch.kappa9.printout) write their whole printouts; ``dfsortp``/``separatedf``/``kapreadts`` print
nothing, as the Fortran.  The other ports do not reproduce their printouts (``xnfpelsyn.out``,
``rgfalllinesnew.out``, the ``rmolecasc``/``rh2ofast`` listings); they return a one-line note instead.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

__all__ = ["PROGRAMS"]


def _unit(workdir: Path, unit_files: dict, n: int) -> Path:
    return Path(unit_files[n]) if n in unit_files else workdir / "fort.{}".format(n)


def _note(name: str) -> str:
    return " {} (atlas_torch): printout not reproduced\n".format(name.upper())


# ------------------------------------------------------------------------------------------------ SYNTHE chain
def xnfpelsyn_program(workdir, stdin_text, unit_files):
    """xnfpelsyn.exe < model: fort.2 molecules, fort.17 continua -> fort.10."""
    from atlas_torch.synthe.xnfpelsyn_prog import xnfpelsyn
    wd = Path(workdir)
    xnfpelsyn(stdin_text, _unit(wd, unit_files, 17).read_text(), _unit(wd, unit_files, 2), out10=wd / "fort.10")
    return _note("xnfpelsyn")


def synbeg_stdout(p: dict, airvac: str) -> str:
    """FORMATs 2 and 3 of synbeg.for (NREAD=0)."""
    from atlas_torch.atlas9 import fformat as ff
    import numpy as np
    f32 = lambda v: float(np.float32(v))                     # TURBV, CUTOFF are REAL*4
    line1 = (" " + airvac.ljust(3)[:3] + "   WLBEG=" + ff.F(p["wlbeg"], 10, 4) + "   WLEND=" + ff.F(p["wlend"], 10, 4)
             + "   RESOLU=" + ff.F(p["resolu"], 10, 1) + "   TURBV=" + ff.F(f32(p["turbv"]), 5, 2)
             + "   IFNLTE=" + ff.I(p["ifnlte"], 1) + "   LINOUT=" + ff.I(p["linout"], 6)
             + "   CUTOFF=" + ff.F(f32(p["cutoff"]), 7, 5) + "   IFPRED=" + ff.I(p["ifpred"], 1))
    line2 = "   NREAD" + ff.I(p.get("nread", 0), 3)
    ratio = p["ratio"]
    wlbeg_pt = p["wbegin"]
    wllast = p["wllast"]
    line3 = ("0LENGTH=" + ff.I(p["length"], 6) + "   RATIO=" + ff.F(ratio, 12, 10) + "  WBEGIN," + ff.F(wlbeg_pt, 10, 4)
             + "   DWLBEG=" + ff.F(f32(wlbeg_pt * ratio - wlbeg_pt), 7, 5) + "   WLLAST=" + ff.F(wllast, 10, 4)
             + "   DWLLAST=" + ff.F(f32(wllast - wllast / ratio), 7, 5))
    return line1 + "\n" + line2 + "\n" + line3 + "\n"


def synbeg_program(workdir, stdin_text, unit_files):
    """synbeg.exe <<card: fort.93 (+ empty fort.12/14/19/20); echoes the card like the Fortran."""
    import math
    from atlas_torch.synthe.lines.synbeg import parse_synbeg_card, run_synbeg
    card = (stdin_text.split("\n") or [""])[0]
    c = parse_synbeg_card(card)
    if c.get("nread", 0) > 0:
        raise NotImplementedError("synbeg: NREAD > 0 (depth-dependent cards) is not supported by the port")
    p = run_synbeg(Path(workdir), card=card)
    # WBEGIN / WLLAST as synbeg.for computes them (DLOG/DEXP of the REAL*8 card values)
    ratiolg = p["ratiolg"]
    ix = int(math.log(p["wlbeg"]) / ratiolg)
    wbegin = math.exp(ix * ratiolg)
    if wbegin < p["wlbeg"]:
        wbegin = math.exp((ix + 1) * ratiolg)
    ix = int(math.log(p["wlend"]) / ratiolg)
    wllast = math.exp(ix * ratiolg)
    if wllast >= p["wlend"]:
        wllast = math.exp((ix - 1) * ratiolg)
    return synbeg_stdout(dict(p, wbegin=wbegin, wllast=wllast, nread=c.get("nread", 0)), c["airvac"])


def rgfalllinesnew_program(workdir, stdin_text, unit_files):
    """rgfalllinesnew.exe: fort.11 atomic line list; appends fort.12/fort.19, updates fort.93."""
    from atlas_torch.synthe.lines.rgfalllines import run_rgfalllines
    wd = Path(workdir)
    run_rgfalllines(wd, _unit(wd, unit_files, 11))
    return _note("rgfalllinesnew")


def rmolecasc_program(workdir, stdin_text, unit_files):
    """rmolecasc.exe: fort.11 molecular list (+ c12c13.dat in workdir); appends fort.12, updates fort.93."""
    from atlas_torch.synthe.lines.rmolecasc import run_rmolecasc
    wd = Path(workdir)
    run_rmolecasc(wd, _unit(wd, unit_files, 11))
    return _note("rmolecasc")


def rh2ofast_program(workdir, stdin_text, unit_files):
    """rh2ofast.exe: fort.11 h2ofastfix.bin; appends fort.12, updates fort.93."""
    from atlas_torch.synthe.lines.rh2ofast import run_rh2ofast
    wd = Path(workdir)
    run_rh2ofast(wd, _unit(wd, unit_files, 11))
    return _note("rh2ofast")


def synthe_program(workdir, stdin_text, unit_files):
    """synthe.exe: fort.10 (xnfpelsyn.dat), fort.18 (he1tables), fort.12/14/19/20/93 -> fort.9, fort.8, fort.14,
    fort.15, progress.dat; stdout as the Fortran."""
    from atlas_torch.synthe.synthe_prog import run_synthe
    return run_synthe(Path(workdir))["stdout"]


def spectrv_program(workdir, stdin_text, unit_files):
    """spectrv.exe < model: fort.2 molecules, fort.25 card, fort.9 (synthe), fort.10 (xnfpelsyn), mask.bin ->
    fort.7 (spectrum.bin)."""
    from atlas_torch.synthe import spectrv_prog as sp
    from atlas_torch.synthe.readin import readin
    wd = Path(workdir)
    state = readin(stdin_text, molecules_path=_unit(wd, unit_files, 2), mode=20)
    res = sp.spectrv(state, sp.load_inputs(wd))
    res.write(wd / "fort.7")
    if res.fort17:
        (wd / "fort.17").write_text(res.fort17)
    return res.stdout


# ------------------------------------------------------------------------------------------------ DFSYNTHE chain
def xnfdf_program(workdir, stdin_text, unit_files):
    """xnfdf.exe: fort.2 molecules, fort.17 continua -> fort.10 (xnfpdf.dat), fort.11 (xnfpdfmax.dat)."""
    from atlas_torch.dfsynthe import printouts, xnfdf
    wd = Path(workdir)
    continua = _unit(wd, unit_files, 17).read_text()
    res = xnfdf.run(stdin_text, continua_text=continua, molecules_text=_unit(wd, unit_files, 2).read_text(),
                    out10=wd / "fort.10", out11=wd / "fort.11")
    return printouts.xnfdf_stdout(res, stdin_text, continua)


_DF_OUT_UNITS = {0: 15, 1: 16, 2: 17, 4: 18, 8: 20}
_df_slots = None
_df_lock = threading.Lock()


def _df_semaphore():
    """DFSYNTHE temperatures need ~2.5 GB each: bound the concurrent ones (compat dfsynthe(parallel=True)
    starts one thread per temperature)."""
    global _df_slots
    with _df_lock:
        if _df_slots is None:
            from atlas_torch.dfsynthe.pipeline import default_workers
            _df_slots = threading.BoundedSemaphore(max(1, default_workers(57)))
        return _df_slots


def dfsynthe_program(workdir, stdin_text, unit_files):
    """dfsynthe.exe <<switches: one temperature (the '1' in the 57 switches). Inputs fort.10 (xnfpdf.dat),
    fort.22 (xnfpdfmax.dat) and the line lists fort.11/21/31/41/43/51 (taken from the directory of fort.11's
    target, i.e. BasicATLAS's dfsynthe_files); outputs fort.15/16/17/18/20 (vturb 0/1/2/4/8).

    Runs in a spawned worker process (one torch thread, stock flush-to-zero environment scoped there), as
    atlas_torch.dfsynthe.pipeline does; the worker also writes the printout (dfsynthe.printouts.dfsynthe_job)."""
    import concurrent.futures
    import multiprocessing
    import tempfile
    import numpy as np
    from atlas_torch.dfsynthe import odf_io
    from atlas_torch.dfsynthe.printouts import dfsynthe_job
    wd = Path(workdir)
    sw = re.sub(r"\s", "", stdin_text)
    if len(sw) < 57 or sw[:57].count("1") != 1:
        raise ValueError("dfsynthe: expected 57 switches with one '1', got {!r}".format(stdin_text[:80]))
    iteff = sw.index("1") + 1
    data_dir = Path(os.path.realpath(_unit(wd, unit_files, 11))).parent
    cfgd = dict(flavor="stock", fastmath=True, ftz=True, device="cpu")
    with _df_semaphore(), tempfile.TemporaryDirectory(dir=wd) as tmp:
        part = Path(tmp) / "t.npz"
        args = (str(_unit(wd, unit_files, 10)), str(_unit(wd, unit_files, 22)), str(data_dir), iteff, cfgd, str(part))
        with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as ex:
            out = ex.submit(dfsynthe_job, args, stdin_text).result()
        z = np.load(part)
        for v, unit in _DF_OUT_UNITS.items():
            odf_io.write_odf_bin(wd / "fort.{}".format(unit), z["idf{}".format(v)], z["v{}".format(v)])
        with open(out, encoding="latin-1", newline="") as f:
            return f.read()


def dfsortp_program(workdir, stdin_text, unit_files):
    """dfsortp.exe: fort.1 (one temperature/vturb ODF) -> fort.2 (sorted, ascii)."""
    from atlas_torch.dfsynthe.dfsortp import dfsortp
    wd = Path(workdir)
    dfsortp(_unit(wd, unit_files, 1), wd / "fort.2")
    return ""


def separatedf_program(workdir, stdin_text, unit_files):
    """separatedf.exe: fort.10..66 (the 57 sorted temperatures) -> fort.2 (p00big), fort.3 (p00lit)."""
    from atlas_torch.dfsynthe.separatedf import separatedf
    wd = Path(workdir)
    separatedf([_unit(wd, unit_files, 10 + i) for i in range(57)], wd / "fort.2", wd / "fort.3")
    return ""


def kapreadts_program(workdir, stdin_text, unit_files):
    """kapreadts.exe: fort.11..15 (kapk0/1/2/4/8.dat) -> fort.2 (kappa.ros)."""
    from atlas_torch.kappa9.kapreadts import kapreadts
    wd = Path(workdir)
    kapreadts([_unit(wd, unit_files, 11 + i) for i in range(5)], wd / "fort.2")
    return ""


PROGRAMS = {
    "xnfpelsyn": xnfpelsyn_program, "synbeg": synbeg_program, "rgfalllinesnew": rgfalllinesnew_program,
    "rmolecasc": rmolecasc_program, "rh2ofast": rh2ofast_program, "synthe": synthe_program,
    "spectrv": spectrv_program, "xnfdf": xnfdf_program, "dfsynthe": dfsynthe_program, "dfsortp": dfsortp_program,
    "separatedf": separatedf_program, "kapreadts": kapreadts_program,
}
