"""ODF and Rosseland-table files and the ODF as an opacity (v2/io/odf_bdf.py, v2/io/kappa_ros.py,
v2/opacity/odf.py, v2/opacity/frequencies.py) against the stock files and the exact port (LINOP, READIN)."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.v2.constants import decimal32
from atlas_torch.v2.io.kappa_ros import read_kapk, read_kappa_ros, write_kapk, write_kappa_ros
from atlas_torch.v2.io.odf_bdf import read_bdf, write_bdf
from atlas_torch.v2.opacity.frequencies import KAPPA9_LOWER_EDGE_HZ, WBIG_NM, FrequencyQuadrature
from atlas_torch.v2.opacity.odf import STEP_FRACTIONS, IntervalSet, OpacityDistribution

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
ODF = EXT / "runs" / "stock" / "odf" / "solar"
KAPK = EXT / "work" / "kappa9_solar_stock"
C_RATIO = 2.997925 / 2.99792458          # READIN's c literal / exact c


def need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p


def test_bdf_roundtrip_byte_identical(tmp_path):
    src = need(ODF / "p00big0.bdf")
    a = read_bdf(src)
    assert a.shape == (328, 57, 25, 12) and a.dtype == np.int16
    write_bdf(tmp_path / "x.bdf", a)
    assert (tmp_path / "x.bdf").read_bytes() == src.read_bytes()


def test_odf_write_roundtrip(tmp_path):
    odf = OpacityDistribution.read(need(ODF), "big", vturbs=(2,))
    odf.write(tmp_path)
    assert (tmp_path / "p00big2.bdf").read_bytes() == (ODF / "p00big2.bdf").read_bytes()


def test_kappa_ros_roundtrip(tmp_path):
    src = need(ODF / "kappa.ros")
    d = read_kappa_ros(src)
    assert d["log_kappa"].shape == (5, 57, 42) and d["extra"].shape == (57, 42, 4)
    write_kappa_ros(tmp_path / "k.ros", d["log_t"], d["log_p"], d["log_kappa"], title=d["title"])
    assert (tmp_path / "k.ros").read_text() == src.read_text()


@pytest.mark.parametrize("v", [0, 8])
def test_kapk_roundtrip_truncate(tmp_path, v):
    src = need(KAPK / f"kapk{v}.dat")
    k = read_kapk(src)
    d = read_kappa_ros(need(ODF / "kappa.ros"))
    write_kapk(tmp_path / "k.dat", d["log_t"], d["log_p"], k.T / 1000.0, rounding="truncate")
    assert (tmp_path / "k.dat").read_text() == src.read_text()


def test_intervals_equal_atlas9_and_dfsynthe():
    from atlas_torch.atlas9 import data
    big = IntervalSet.big()
    assert len(big) == 328 and len(IntervalSet.little()) == 1212
    w = data.wavedf()
    # WAVEDF(329,2), WAVEDF(1213,1) = 1e7 in ATLAS9 (LINOP extends the last interval); DFINTERVALS ends at 1e4 nm
    assert np.array_equal(decimal32(w[:328, 1]), big.edges_nm[:328]) and big.edges_nm[328] == 10000.0
    assert np.array_equal(decimal32(w[:1212, 0]), IntervalSet.little().edges_nm[:1212]) and w[1212, 0] == 1e7
    assert big.interval_of(torch.tensor([8.0, 8.97666, 500.0, 10000.0, 2e5])).tolist() == [-1, 0, 169, 327, 327]
    assert big.interval_of(torch.tensor([2e5]), extend_last=False).tolist() == [-1]


def test_atlas_big_quadrature_vs_readin():
    """READIN labels 2210-2212 (exact port, kappa9 READIN of the stock control cards): same formula; the
    differences are READIN's c = 2.997925e17 nm/s (scaled out), the REAL*4 WBIG literals (5e-8 in nu) and
    REAL*4 RCOSET (amplified by the frequency differences: < 3e-6 in the weights)."""
    from atlas_torch.atlas9 import linop
    from atlas_torch.kappa9 import kappa9_prog as kp
    from atlas_torch.kappa9.readin import readin
    rd = readin(kp.control_from_com(need(ODF / "kappa9v0.com")),
                need(EXT / "BasicATLAS/data/dfsynthe_files/molecules.dat"))
    fre, rco = rd["freset.freset"].numpy()[:337], rd["freset.rcoset"].numpy()[:337]
    q = FrequencyQuadrature.atlas_big()
    assert np.abs(q.nu.numpy() * C_RATIO / fre - 1).max() < 6e-8
    assert np.abs(q.weight.numpy() * C_RATIO / rco - 1).max() < 3e-6
    nu9, rco9 = kp.frequency_set(rd)
    q9 = FrequencyQuadrature.atlas_big(lower_edge_nm=2.99792458e17 / KAPPA9_LOWER_EDGE_HZ)
    assert np.abs(q9.nu.numpy() * C_RATIO / nu9 - 1).max() < 6e-8
    assert np.abs(q9.weight.numpy() * C_RATIO / rco9 - 1).max() < 3e-6
    # LINOP's interval of each BIG frequency (forward search from interval 1)
    iwl = linop.interval_index([float(x) for x in WBIG_NM])
    assert q.interval.tolist() == [i - 1 for i in iwl]


def test_samples():
    q = FrequencyQuadrature.intervals(IntervalSet.big(), uv_to_nm=5.0, ir_to_nm=1e5, n_extension=4)
    s = q.samples()
    assert len(q.nu) == 328 + 8
    assert len(s) == 4 + (328 + 4) * 12
    assert torch.allclose(s.weight.sum(), q.weight.sum(), rtol=1e-14)
    for sl in s.chunks(1000):
        f = s.frequency[sl]
        assert sl.start == 0 or bool(s.frequency[sl.start - 1] != f[0])
    assert np.isclose(STEP_FRACTIONS.sum(), 1.0)


def test_odf_kappa_vs_linop():
    """OpacityDistribution.kappa at the stock solar model's T, P vs the exact port's LINOP ALINES (REAL*4
    coefficients and EXP: ~1e-5)."""
    from atlas_torch.atlas9 import linop
    from atlas_torch.v2.io import read_atlas_deck
    atm = read_atlas_deck(need(EXT / "runs/stock/atlas/sun/output_summary.out")).atmosphere
    odf = OpacityDistribution.read(need(ODF), "big", vturbs=(2,))
    k = odf.kappa(atm.T, atm.P)
    it, ip, co, fs = linop.coefficients(torch.log(atm.T), atm.P, torch.zeros_like(atm.T), 0.0)
    al, _ = linop.line_opacity(linop.read_odf(str(ODF / "p00big2.bdf")), torch.arange(1, 329), it, ip, co, fs)
    ref = al.permute(2, 0, 1).flip(-1)
    assert float(((k - ref).abs() / ref).max()) < 2e-5


def test_odf_vturb_interpolation_and_stack():
    odf = OpacityDistribution.read(need(ODF), "big", vturbs=(1, 2))
    T, P = torch.tensor([[5000.0, 7000.0]]), torch.tensor([[1e4, 1e5]])
    a = odf.select_vturb(1).log_kappa_at(T, P)
    b = odf.select_vturb(2).log_kappa_at(T, P)
    m = odf.log_kappa_at(T, P, vturb=torch.tensor([[1.5e5, 2.5e5]]))
    assert torch.allclose(m[0, 0], 0.5 * (a[0, 0] + b[0, 0])) and torch.allclose(m[0, 1], b[0, 1])
    st = OpacityDistribution.stack([odf.select_vturb(1), odf.select_vturb(2)])
    s = st.log_kappa_at(T.expand(2, 2), P.expand(2, 2), table_index=torch.tensor([1, 0]))
    assert torch.equal(s[0], b[0]) and torch.equal(s[1], a[0])
    assert odf.in_domain(torch.tensor([1000.0, 5000.0]), torch.tensor([1.0, 1e9])).tolist() == [False, False]


def test_odf_directory_basicatlas(tmp_path):
    """write_odf_directory: BasicATLAS meta() reads the same mixture as from the stock directory, and
    validate_run() accepts the xnfdf.com / xnfdf.out and kappa9v0.com / kapm40k2.out pairs."""
    import shutil
    import sys
    from atlas_torch.v2.io.odf_directory import read_abundance_cards, write_odf_directory
    stock = need(EXT / "runs" / "stock" / "odf" / "alpha")
    sys.path.insert(0, str(need(EXT / "BasicATLAS")))
    try:
        import atlas as basicatlas
    finally:
        sys.path.pop(0)
    write_odf_directory(tmp_path, read_abundance_cards(stock / "xnfdf.com"))
    shutil.copy(stock / "kappa9v0.com", tmp_path / "kappa9v0.com")
    assert basicatlas.meta(str(tmp_path)) == basicatlas.meta(str(stock))
    assert basicatlas.validate_run(str(tmp_path), silent=True)
