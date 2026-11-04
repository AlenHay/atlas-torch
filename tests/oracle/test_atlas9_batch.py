"""Plumbing of the multi-model driver (atlas9.batch.run_models): jobs run in spawned workers, results come back in
job order, and a failing model is reported without stopping the others. (The models themselves are checked end
to end in tests/stage/test_atlas9_program.py.)"""
from atlas_torch.atlas9.batch import AtlasJob, run_models
from atlas_torch.compat.settings import Settings


def test_run_models_reports_per_model_errors(tmp_path):
    dirs = [tmp_path / "a", tmp_path / "b"]
    for d in dirs:
        d.mkdir()                                   # atlas() refuses an existing output directory
    res = run_models([AtlasJob(str(d), Settings()) for d in dirs], workers=2)
    assert [r["output_dir"] for r in res] == [str(d) for d in dirs]
    for r, d in zip(res, dirs):
        assert r["error"] == "ValueError: Directory {} already exists".format(d)
    assert run_models([]) == []
