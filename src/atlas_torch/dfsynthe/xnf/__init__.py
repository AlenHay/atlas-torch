"""Building blocks of the PROGRAM XNFDF port (xnfdf.for, DFSYNTHE variant of the physics library).

Entry point: atlas_torch.dfsynthe.xnfdf.run(). Modules:
  fdata     DATA tables (EQUIVALENCE-assembled), PFGROUND compiled from source
  fmath     Fortran numeric semantics (REAL*4 literals, correctly rounded X**N, true scalar/tensor division)
  freeform  FREEFF / IWORDF / FREEFR / XFREEF card readers
  readin    READIN, READMOL, INTEG/PARCOE, continuum edges
  pfsaha    PFSAHA, PFIRON
  nmolec    NMOLEC, SOLVIT, NELECT, POPS
  kapp      KAPP(N=1) and the continuum opacity sources
  xnfio     fort.10 / fort.11 unformatted writers and readers
  oracle_support  ctypes-oracle helpers for the tests (Intel runtime preload, COMMON loaders)
"""
