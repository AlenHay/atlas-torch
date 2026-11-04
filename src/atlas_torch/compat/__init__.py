"""BasicATLAS-compatible front end for atlas-torch (milestone M6).

Put this directory on ``sys.path`` (``scripts/run_examples.py`` does) and ``import atlas`` resolves to
:mod:`atlas_torch.compat.atlas`, a reimplementation of BasicATLAS's public API whose Fortran program calls
go through :mod:`atlas_torch.compat.runners`.
"""
