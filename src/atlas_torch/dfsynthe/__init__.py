"""DFSYNTHE family (opacity distribution functions) in torch.

Entry points:
  pipeline.run_odf(out_dir, xnfdf_stdin=... | xnfdf_com=...)   whole chain -> xnfpdf.dat, xnfpdfmax.dat,
                                                                p00big{v}.bdf, p00lit{v}.bdf, kappa.ros
  xnfdf.run(stdin, ...)                                        PROGRAM XNFDF
  dfsynthe.run_temperature(xnfpdf, xnfpdfmax, data_dir, it)   one DFSYNTHE temperature
  compat.torch_programs (xnfdf/dfsynthe/dfsortp/separatedf runners for compat.atlas.dfsynthe)
Stock semantics by default: flush-to-zero environment (ftz.py) and the ifx -fp-model fast Lyman-alpha term
(lines/hydrogen.py, Hydrogen._lya_h2plus_fast).
"""
