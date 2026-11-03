"""Torch stage runners used by `make score` (stage-isolated: each consumes the stock Fortran inputs).

STAGES[kind](stage, reference_set) -> path of the candidate output dir, comparable with the stock output by
atlas_torch.scoring. A kind is registered here once its torch implementation exists.
"""
STAGES: dict = {}
