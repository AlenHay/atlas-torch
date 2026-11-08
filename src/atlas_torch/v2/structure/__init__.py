"""Model-atmosphere structure (ATLAS9 physics, Kurucz 1970): hydrostatic, radiative and convective equilibrium on
a fixed Rosseland optical-depth grid with ODF line opacity.

    from atlas_torch.v2.structure import ModelSpec, StructureConfig, StructureSolver
    from atlas_torch.v2.io.atlas_control import read_restart
    spec = ModelSpec.from_control("atlas_control.com", odf_dir)
    solver = StructureSolver(StructureConfig.kurucz(), ModelSpec.stack([spec]))
    result = solver.run(solver.start([read_restart("restart.dat")]))
    result.write("run_dir")                    # output_main.out, output_last_iteration.out, output_summary.out
"""
from .config import ModelBatch, ModelSpec, StructureConfig
from .grid import RosselandGrid
from .model import ModelResult, StructureSolver, StructureState, compute_model

__all__ = ["ModelBatch", "ModelSpec", "StructureConfig", "RosselandGrid", "ModelResult", "StructureSolver",
           "StructureState", "compute_model"]
