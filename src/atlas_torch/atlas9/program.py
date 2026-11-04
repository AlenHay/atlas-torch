"""The ``atlas9mem`` program for :mod:`atlas_torch.compat.runners` (BasicATLAS's ``atlas9mem.exe``).

What BasicATLAS feeds the executable (``atlas_control_start.com`` / ``atlas_control.com``) and what this runner
reads instead:

  stdin    the control cards: READ KAPPA, READ PUNCH, MOLECULES ON|OFF, READ MOLECULES, FREQUENCIES, VTURB,
           CONVECTION, TITLE, SCALE, ABUNDANCE ..., then ``ITERATIONS 15 PRINT ... / PUNCH ... / BEGIN`` batches
           (niter // 15 of them, plus one with the remainder)
  fort.1   kappa.ros (READ KAPPA)        fort.2  molecules.dat (READ MOLECULES)
  fort.3   the restart deck (READ PUNCH) fort.9  the ODF p00big{v}.bdf (IFOP(15)=1 line opacity)

Outputs, as the Fortran writes them:
  return value   unit 6 (``output_main.out``): READIN listing, iteration tables, the patch's stop messages
  fort.7         unit 7: one punched model per iteration, each ended by ``==========``

Termination: STOP (REACHED GOLD TOLERANCES, MODEL DIVERGED, INVALID STRUCTURE, HYDROFAIL ...) returns normally.
Running out of control cards is a gfortran runtime error in the Fortran (exit status 2 and a message on stderr,
which makes BasicATLAS raise); here fort.7 is written and a ValueError is raised, as the stock executable run
through :class:`atlas_torch.compat.runners.FortranRunner` would.

Physics (populations + continuous opacities): ``physics=`` or ``$ATLAS_TORCH_ATLAS9_PHYSICS``:
  ``torch``  (default) :class:`atlas_torch.atlas9.physics.Atlas9Physics` -- the torch port;
  ``oracle`` TEST-ONLY: the Fortran routines through ctypes (:mod:`atlas_torch.testing.atlas9_oracle_physics`).
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

import torch

__all__ = ["atlas9mem", "atlas9mem_program", "make_physics", "PROGRAMS"]

UNITS = (1, 2, 3, 9)


@contextlib.contextmanager
def float64_default():
    """The port allocates scalars with torch.tensor([...]): run with float64 as the default dtype."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def make_physics(physics=None):
    physics = physics or os.environ.get("ATLAS_TORCH_ATLAS9_PHYSICS", "torch").strip().lower() or "torch"
    if not isinstance(physics, str):
        return physics
    if physics == "torch":
        from .physics import Atlas9Physics
        return Atlas9Physics("atlas9")
    if physics == "oracle":
        from ..testing.atlas9_oracle_physics import OraclePhysics
        return OraclePhysics()
    raise ValueError("ATLAS9 physics must be 'torch' or 'oracle', not {!r}".format(physics))


def atlas9mem(stdin_text: str, files: dict, physics=None, on_iteration=None):
    """Run the program; returns (stdout text, fort.7 text, RunResult without per-iteration States).
    ``files``: {unit: path} for units 1, 2, 3, 9 (only those the control cards read are needed)."""
    from .main import run
    with float64_default():
        r = run(stdin_text, {u: Path(p) for u, p in files.items()}, make_physics(physics),
                on_iteration=on_iteration, keep_states=False)
    stdout = "".join(line + "\n" for line in r.stdout)
    punch = "".join(line + "\n" for line in r.punch)
    return stdout, punch, r


def atlas9mem_program(workdir, stdin_text, unit_files):
    """compat.runners interface: units 1/2/3/9 from ``unit_files`` (else ``workdir/fort.N``) -> stdout, fort.7."""
    workdir = Path(workdir)
    files = {}
    for u in UNITS:
        p = Path(unit_files[u]) if u in unit_files else workdir / "fort.{}".format(u)
        if p.exists():
            files[u] = p
    t0 = time.perf_counter()
    stdout, punch, r = atlas9mem(stdin_text, files)
    # unit 7 is opened on first write: no punch, no file (as gfortran)
    if punch:
        with open(workdir / "fort.7", "w", encoding="latin-1", newline="") as f:
            f.write(punch)
    if os.environ.get("ATLAS_TORCH_TIMING"):
        print("atlas9mem (torch): {:.1f} s".format(time.perf_counter() - t0))
    if r.exit_code != 0:
        raise ValueError("atlas9mem (torch) terminated with status {}: {}".format(r.exit_code, r.exit_message))
    return stdout


PROGRAMS = {"atlas9mem": atlas9mem_program}
