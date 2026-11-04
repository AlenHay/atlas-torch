"""The SYNTHE chain of BasicATLAS (templates.synthe_control) with the torch/Python programs only.

    from atlas_torch.synthe.chain import run_synthe_chain
    res = run_synthe_chain("output_synthe.out", 500., 510., 600000., 1.5, "work/synthe_1", "BasicATLAS")
    res["spectrum"]   # -> <workdir>/spectrum.bin (spectrv fort.7, the file BasicATLAS reads)

What synthe_control does, step by step, and what runs here instead of the executable:
  xnfpelsyn.exe < model     -> atlas_torch.synthe.xnfpelsyn_prog.xnfpelsyn   (fort.2 molecules, fort.17
                                continua -> xnfpelsyn.dat)
  synbeg.exe <<card          \\
  rgfalllinesnew.exe          |  atlas_torch.synthe.lines.prepare_lines      (fort.93, 12, 14, 19, 20;
  [c12c13.dat]                |   same 21 molecular lists in the same order, c12c13.dat honoured)
  rmolecasc.exe x 21          |
  rh2ofast.exe               /
  synthe.exe                -> atlas_torch.synthe.synthe_prog.run_synthe    (fort.10 = xnfpelsyn.dat,
                                fort.18 = he1tables.dat -> fort.9, fort.8, fort.14, fort.15)
  spectrv.exe < model       -> atlas_torch.synthe.spectrv_prog.spectrv      (fort.25 card, fort.9,
                                fort.10, optional mask.bin -> fort.7 = spectrum.bin)
SPECTRV starts with CALL READIN(20) on the same deck as XNFPELSYN (same routine, same molecules file, fresh
program), so the READIN State of the xnfpelsyn step is reused instead of reading the deck twice.

One call is one BasicATLAS batch (`synthe_<n>` directory): the caller splits wide ranges by the SYNBEG
buffer size exactly as BasicATLAS' synthe() does. The stdout listings of xnfpelsyn/spectrv (*.out) are not
reproduced (synthe's stdout is written to synthe.out). As BasicATLAS, the work directory is cleaned at the
end (fort.*, xnfpelsyn.dat) unless keep=True.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

MOLECULAR_LISTS = ["chmasseron_corrected.asc", "mgh_exomol.asc", "nh.asc", "ohupdate.asc", "sihax.asc", "h2.asc",
                   "h2xx.asc", "hdxx.asc", "c2ax.asc", "c2ba.asc", "c2dabrookek.asc", "c2ea.asc",
                   "cnaxbrookek.asc", "cnbxbrookek.asc", "cnxx12brooke.asc", "coax.asc", "coxx.asc", "sioax.asc",
                   "sioex.asc", "sioxx.asc", "tiototo.asc"]
FORT25 = ("0.0       0.        1.        0.        0.        0.        0.        0.\n0.\n"
          "RHOXJ     R1        R101      PH1       PC1       PSI1      PRDDOP    PRDPOW\n")


def basicatlas_dir() -> Path:
    return Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext")) / "BasicATLAS"


def _linelist_path(linelist, s_files: Path) -> Path:
    p = Path(linelist)
    if not p.suffix and not p.exists():
        p = s_files / f"{linelist}.dat"            # BasicATLAS: linelist name -> data/synthe_files/<name>.dat
    p = Path(os.path.realpath(p))
    if not p.is_file():
        raise ValueError(f"Linelist {linelist} not found")
    return p


def c12c13_card(ratio: float) -> str:
    """BasicATLAS' c12c13.dat for a 12C/13C ratio: 'log10(C12) log10(C13)' with Python's float repr."""
    c13 = 1 / (ratio + 1)
    c12 = 1 - c13
    return "{} {}".format(np.log10(c12), np.log10(c13))


def run_synthe_chain(model_deck_path, wlbeg: float, wlend: float, resolu: float, vturb: float, workdir,
                     linelist="BasicATLAS", c12c13=None, airorvac: str = "VAC", mask=None, *, keep: bool = False,
                     device=None, physics=None, data_dir=None, progress=None, spectrv_libm: str = "libmvec") -> dict:
    """Run xnfpelsyn -> line lists -> synthe -> spectrv in `workdir`; returns paths and per-program timings.

    model_deck_path: the SYNTHE model deck (BasicATLAS output_synthe.out). wlbeg/wlend [nm], resolu, vturb
    [km/s] as on the SYNBEG card (IFNLTE=0, LINOUT=-1, CUTOFF=0.0001, IFPRED=1, NREAD=0 as BasicATLAS).
    linelist: name in data/synthe_files (e.g. 'BasicATLAS') or a path. c12c13: 12C/13C ratio or None
    (rmolecasc defaults). airorvac: 'VAC' or 'AIR'. mask: None, a path to a mask.bin, or a boolean/int array
    (one byte per wavelength point, nonzero = compute; written as mask.bin for SPECTRV's BasicATLAS patch).
    physics: xnfpelsyn physics backend (default: the torch port). keep: keep the intermediate fort.* files.
    spectrv_libm: SPECTRV's libm mode (atlas_torch.synthe.spectrv_prog.spectrv): "libmvec" (default) evaluates
    the vectorised EXP loop as the stock gfortran -O3 spectrv.exe does (glibc libmvec), "auto" as gf-O0.
    """
    from .lines import prepare_lines
    from .xnfpelsyn_prog import xnfpelsyn

    ba = Path(data_dir) if data_dir is not None else basicatlas_dir()
    s_files, d_files = ba / "data" / "synthe_files", ba / "data" / "dfsynthe_files"
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    deck_path = Path(model_deck_path)
    deck_text = deck_path.read_text()
    ll = _linelist_path(linelist, s_files)
    if airorvac not in ("VAC", "AIR"):
        raise ValueError("airorvac must be 'VAC' or 'AIR'")
    tm: dict = {}
    say = progress or (lambda msg: None)

    # xnfpelsyn.exe < model > xnfpelsyn.out ; mv fort.10 xnfpelsyn.dat
    say("xnfpelsyn")
    t = time.perf_counter()
    xr = xnfpelsyn(deck_text, (s_files / "continua.dat").read_text(), s_files / "molecules.dat",
                   out10=wd / "xnfpelsyn.dat", physics=physics, device=device)
    tm["xnfpelsyn"] = time.perf_counter() - t

    # synbeg / rgfalllinesnew / c12c13.dat / rmolecasc x 21 / rh2ofast
    say("line lists")
    t = time.perf_counter()
    c12c13_file = wd / "c12c13.dat"
    if c12c13 is None or isinstance(c12c13, bool):
        c12c13_file.unlink(missing_ok=True)              # rm -f c12c13.dat
        iso = None
    else:
        card = c12c13_card(float(c12c13))
        c12c13_file.write_text(card + "\n")               # echo "..." > c12c13.dat
        a, b = card.split()
        iso = (float(a), float(b))                       # READ(991,*) C12, C13 (REAL*8)
    lr = prepare_lines(wd, wlbeg, wlend, resolu, vturb, ll, [s_files / m for m in MOLECULAR_LISTS],
                       d_files / "h2ofastfix.bin", c12c13=iso, airorvac=airorvac)
    tm["lines"] = time.perf_counter() - t

    # synthe.exe: fort.10 = xnfpelsyn.dat, fort.18 = he1tables.dat
    say("synthe")
    t = time.perf_counter()
    for name, src in (("fort.10", wd / "xnfpelsyn.dat"), ("fort.18", s_files / "he1tables.dat")):
        dst = wd / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.abspath(src), dst)
    from .synthe_prog import run_synthe
    sr = run_synthe(wd, device=device, stdout_path=wd / "synthe.out")
    tm["synthe"] = time.perf_counter() - t

    # spectrv.exe < model: fort.25 card, mask.bin (BasicATLAS patch), fort.7 -> spectrum.bin
    say("spectrv")
    t = time.perf_counter()
    (wd / "fort.25").write_text(FORT25)
    mpath = wd / "mask.bin"
    if mask is None:
        mpath.unlink(missing_ok=True)
    elif isinstance(mask, (str, os.PathLike)):
        if Path(mask).resolve() != mpath.resolve():
            mpath.write_bytes(Path(mask).read_bytes())
    else:
        mpath.write_bytes((np.asarray(mask) != 0).astype(np.int8).tobytes())
    from . import spectrv_prog as sp
    state = xr.readin_state
    if state is None:                                    # non-torch xnfpelsyn backend: READIN for SPECTRV
        from .readin import readin
        state = readin(deck_text, molecules_path=s_files / "molecules.dat", mode=20)
    res = sp.spectrv(state, sp.load_inputs(wd), device=device, libm=spectrv_libm)
    spectrum = wd / "spectrum.bin"
    res.write(spectrum)                                  # mv fort.7 spectrum.bin
    tm["spectrv"] = time.perf_counter() - t

    if not keep:                                         # rm fort.* ; rm xnfpelsyn.dat
        for f in wd.glob("fort.*"):
            f.unlink()
        (wd / "xnfpelsyn.dat").unlink(missing_ok=True)
    return dict(spectrum=spectrum, workdir=wd, timings=tm, lines=lr, synthe=sr, xnfpelsyn=xr)
