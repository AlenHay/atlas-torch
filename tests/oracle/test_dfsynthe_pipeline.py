"""DFSYNTHE pipeline (atlas_torch.dfsynthe.pipeline) bookkeeping, fast:

1. the in-memory DFSORTP + SEPARATEDF of run_odf (parts files -> load_steps -> write_bdf) equals DFSORTP's
   reordering, and (stage: writes 285 ascii files) is byte-identical to the file route of BasicATLAS
   (fort.15.. records -> dfsortp ascii -> separatedf) on random ODF steps;
2. the KAPPA9 control cards built from an XNFDF deck equal BasicATLAS's kappa9v{v}.com heredocs (stock runs);
3. parse_xnfdf_com / abundance_cards on the stock command files.
"""
from pathlib import Path

import numpy as np
import pytest

from atlas_torch.dfsynthe import odf_io, pipeline
from atlas_torch.dfsynthe.dfsynthe import write_temperature
from atlas_torch.dfsynthe.separatedf import separatedf

STOCK = pipeline._ext() / "runs" / "stock" / "odf"


@pytest.fixture
def big_tmp():
    """Large scratch files (> 1 GB of ascii) go to $ATLAS_EXT/scratch, not the shared /tmp tmpfs."""
    import shutil
    import tempfile
    base = pipeline._ext() / "scratch"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="test_dfsynthe_pipeline_", dir=base))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _fake_parts(parts, rng):
    j = np.arange(1, odf_io.KW + 1)
    idf = np.arange(1, odf_io.NFREQ + 1)
    out = {}
    for it in range(1, pipeline.NTEMP + 1):
        d = {}
        for v in odf_io.VTURBS:
            d[f"v{v}"] = rng.integers(-3000, 9000, (odf_io.KW * odf_io.NFREQ, 12)).astype(np.int16)
            d[f"idf{v}"] = (v * 100000000 + idf[None, :] * 10000 + it * 100 + j[:, None]).reshape(-1).astype(np.int32)
        np.savez(parts / f"t{it:02d}.npz", **d)
        out[it] = d
    return out


def test_load_steps_equals_dfsortp(tmp_path):
    rng = np.random.default_rng(3)
    parts = tmp_path / "parts"
    parts.mkdir()
    data = _fake_parts(parts, rng)
    steps = pipeline.load_steps(parts)
    from atlas_torch.dfsynthe.dfsortp import dfsortp_arrays
    for it in (1, 30, 57):
        for v in odf_io.VTURBS:
            idf, st = dfsortp_arrays(data[it][f"idf{v}"], data[it][f"v{v}"])
            assert np.array_equal(steps[v][it - 1], st.reshape(odf_io.NFREQ, odf_io.KW, 12))
            assert np.array_equal(idf.reshape(odf_io.NFREQ, odf_io.KW) % 100, np.tile(np.arange(1, 26), (1540, 1)))


@pytest.mark.stage
def test_bdf_in_memory_equals_file_route(big_tmp):
    tmp_path = big_tmp
    rng = np.random.default_rng(7)
    parts = tmp_path / "parts"
    parts.mkdir()
    data = _fake_parts(parts, rng)
    mem = tmp_path / "mem"
    mem.mkdir()
    pipeline.write_bdf(pipeline.load_steps(parts), mem)
    ref = tmp_path / "ref"
    for it, d in data.items():
        dft = str(int(float(odf_io.DFTS[it - 1])))
        res = {v: (d[f"idf{v}"], d[f"v{v}"]) for v in odf_io.VTURBS}
        write_temperature(res, ref, dft)
    for v in odf_io.VTURBS:
        files = [ref / f"dfp00t{int(float(t))}vt{v}sortp.asc" for t in odf_io.DFTS]
        separatedf(files, ref / f"p00big{v}.bdf", ref / f"p00lit{v}.bdf")
        for p in ("big", "lit"):
            assert (mem / f"p00{p}{v}.bdf").read_bytes() == (ref / f"p00{p}{v}.bdf").read_bytes(), (p, v)


def test_load_steps_rejects_wrong_idfout(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    _fake_parts(parts, np.random.default_rng(1))
    z = dict(np.load(parts / "t05.npz"))
    z["idf0"] = z["idf0"] + 1
    np.savez(parts / "t05.npz", **z)
    with pytest.raises(ValueError):
        pipeline.load_steps(parts)


@pytest.mark.parametrize("mix", ["solar", "alpha"])
def test_kappa9_controls_match_basicatlas(mix):
    d = STOCK / mix
    if not (d / "xnfdf.com").exists() or not (d / "kappa9v0.com").exists():
        pytest.skip("stock ODF run missing")
    pytest.importorskip("atlas_torch.compat.atlas")
    from atlas_torch.kappa9.kappa9_prog import control_from_com
    stdin, cps = pipeline.parse_xnfdf_com(d / "xnfdf.com")
    assert {"fort.2", "fort.17"} <= set(cps)
    ctl = pipeline.kappa9_controls_from_xnfdf(stdin)
    for v in odf_io.VTURBS:
        assert ctl[v] == control_from_com(d / f"kappa9v{v}.com"), v
    cards = pipeline.abundance_cards(stdin)
    assert cards.startswith("ABUNDANCE SCALE") and cards.count("\n") == 17
