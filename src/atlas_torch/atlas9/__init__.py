"""ATLAS9 (atlas9mem.for as patched by BasicATLAS): structure-iteration machinery.

Modules (one per Fortran unit or small group):
  data      DATA tables of the atlas9 variant (from the fetched source, never transcribed)
  commons   canonical COMMON layout + initial (BLOCK DATA) State
  numerics  DERIV, INTEG, PARCOE, EXPI, batched MAP1
  fformat   Fortran formatted-output emulation (E/F/I/A edit descriptors as gfortran prints them)
  cards     READIN's free-format card parser (IWORDF/FREEFF/FREEFR)
  readin    READIN (+ READMOL, READ KAPPA, READ PUNCH)
  rosstab   ROSSTAB (kappa.ros table interpolation)
  ttaup     TTAUP (hydrostatic T-tau integration)
  linop     LINOP (ODF line opacity, 12 sub-steps) + ODF reader
  kapp      KAPP's combination of the opacity sources into /OPTOT/
  josh      JOSH (transfer: Feautrier-like matrix iteration on the XTAU grid)
  radiap, ross, high, turb, convec, tcorr   the per-iteration integrals and corrections
  putout    PUTOUT (patched iteration table on stdout + punch on unit 7)
  physics   the interface to the populations / continuous-opacity physics (ported by another agent)
  main      the ATLAS9 main program (iteration loop, frequency integration with opacity sampling)
  program   the ``atlas9mem`` program for compat.runners (stdin cards + fort.1/2/3/9 -> stdout, fort.7)
  batch     several models concurrently (exact; worker processes) + notes on a model-axis batch
"""
