"""SYNTHE-variant (atlas7v.for) continuous opacities called by KAPP on the xnfpelsyn path.

Modules: hydrogen (HOP, XKARSAS, COULFF, H2PLOP, HMINOP, LINTER, H2RAOP), helium (HE1OP, HE2OP, HEMIOP,
HERAOP), cool (COOLOP, C1OP, MG1OP, AL1OP, SI1OP, FE1OP, CHOP, OHOP, H2COLLOP), luke (LUKEOP, N1OP, O1OP,
MG2OP, SI2OP, CA2OP, SEATON), hot (HOTOP, ELECOP). HRAYOP: atlas_torch.synthe.hrayop. Driver:
atlas_torch.synthe.kapp. Conventions (batching, libm, SAVE state) in _core.
"""
