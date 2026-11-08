"""Rosseland mean opacity and tables (v2/opacity/rosseland.py): analytic limits, ROSSTAB, autograd."""
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_torch.v2.constants import CODATA2018, KURUCZ
from atlas_torch.v2.opacity import planck_derivative
from atlas_torch.v2.opacity.frequencies import FrequencyQuadrature
from atlas_torch.v2.opacity.odf import IntervalSet, OpacityDistribution
from atlas_torch.v2.opacity.rosseland import RosselandConfig, RosselandOpacity, RosselandTable, rosseland_mean

EXT = Path(os.environ.get("ATLAS_EXT", Path.home() / "atlas-ext"))
ODF = EXT / "runs" / "stock" / "odf" / "solar"


def need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p


def dense(lo=1e11, hi=1e18, n=200000):
    """Midpoint rule on a log-uniform frequency grid."""
    e = torch.logspace(math.log10(lo), math.log10(hi), n + 1, dtype=torch.float64)
    return 0.5 * (e[1:] + e[:-1]), e[1:] - e[:-1]


@pytest.mark.parametrize("T", [2000.0, 5772.0, 50000.0])
def test_planck_derivative_integral(T):
    nu, w = dense()
    t = torch.tensor([T], dtype=torch.float64)
    s = (planck_derivative(nu, t) * w).sum(-1)
    ref = 4.0 * CODATA2018.sigma * T ** 3 / math.pi
    assert abs(float(s[0]) / ref - 1) < 1e-6


def test_planck_derivative_matches_autograd():
    t = torch.tensor([4000.0, 9000.0], dtype=torch.float64, requires_grad=True)
    nu = torch.tensor([1e13, 5e14, 3e15], dtype=torch.float64)
    from atlas_torch.v2.opacity import planck
    (g,) = torch.autograd.grad(planck(nu, t).sum(), t)
    assert torch.allclose(g, planck_derivative(nu, t).sum(-1), rtol=1e-12)


def test_grey():
    q = FrequencyQuadrature.atlas_big()
    s = q.samples()
    T = torch.tensor([[3000.0, 6000.0, 20000.0]], dtype=torch.float64)
    k = torch.full(T.shape + (len(s),), 0.37, dtype=torch.float64)
    assert torch.allclose(rosseland_mean(k, s, T), torch.full_like(T, 0.37), rtol=1e-14)
    nu, w = dense()
    kd = torch.full(T.shape + nu.shape, 0.37, dtype=torch.float64)
    assert torch.allclose(rosseland_mean(kd, (nu, w), T, normalization="analytic"), torch.full_like(T, 0.37),
                          rtol=1e-6)


def test_rosseland_mean_gradcheck():
    nu, w = torch.logspace(13, 16, 7, dtype=torch.float64), torch.rand(7, dtype=torch.float64) * 1e14
    k = (torch.rand(2, 3, 7, dtype=torch.float64) + 0.1).requires_grad_()
    T = (torch.tensor([[4000.0, 6000.0, 9000.0]] * 2, dtype=torch.float64)).requires_grad_()
    for norm in ("quadrature", "analytic"):
        assert torch.autograd.gradcheck(lambda a, b: rosseland_mean(a, (nu, w), b, normalization=norm), (k, T))


def test_table_rosstab():
    """RosselandTable.kappa vs the exact port's ROSSTAB on the stock kappa.ros: with ROSSTAB's REAL*4 TABT/TABP
    nodes < 1e-6 dex (EXP10 there uses a REAL*4 ln 10); with the decimal nodes the abscissae shift by up to
    5e-8 and kappa_R by < 5e-6 dex."""
    from dataclasses import replace
    from atlas_torch.atlas9 import data
    from atlas_torch.atlas9.commons import initial_state
    from atlas_torch.atlas9.rosstab import Rosstab, read_kappa_ros
    tab = RosselandTable.read(need(ODF / "kappa.ros"))
    _, tk = read_kappa_ros(ODF / "kappa.ros")
    assert np.array_equal(tab.log_kappa.numpy(), tk.permute(2, 0, 1).numpy())
    s = initial_state()
    s["tabtp.tabkap"] = tk
    s["tabtp.iftabk"] = torch.tensor([1])
    R = Rosstab(s)
    g = torch.Generator().manual_seed(1)
    lt = 3.25 + 2.15 * torch.rand(300, generator=g, dtype=torch.float64)
    lp = -6.0 + 15.0 * torch.rand(300, generator=g, dtype=torch.float64)
    v = torch.tensor([0.0, 1e5, 1.5e5, 2e5, 3e5, 8e5, 9e5], dtype=torch.float64)[torch.randint(0, 7, (300,), generator=g)]
    T, P = 10.0 ** lt, 10.0 ** lp
    ref = torch.log10(torch.stack([R(float(a), float(b), float(c)) for a, b, c in zip(T, P, v)]))
    f32 = replace(tab, log_t=data.rosstab_grid()["tabt"], log_p=data.rosstab_grid()["tabp"])
    assert float((torch.log10(f32.kappa(T, P, v)) - ref).abs().max()) < 1e-6
    assert float((torch.log10(tab.kappa(T, P, v)) - ref).abs().max()) < 5e-6


@pytest.mark.stage
def test_rosseland_opacity_smoke_and_derivatives():
    """Two layers through EOS + continuum + ODF; log derivatives by forward mode vs central differences. (T, P)
    avoid the nodes of the piecewise-linear tables, where the derivative is one-sided: CH/OH (2000 + 500 j K), H2
    CIA (1000 j K), the ODF's bilinear (log T, log P) grid (log P = -2, -1.5, ... 10)."""
    from atlas_torch.v2.io import read_atlas_deck
    comp = read_atlas_deck(need(EXT / "runs/stock/atlas/sun/output_summary.out")).composition
    odf = OpacityDistribution.read(need(ODF), "big", vturbs=(2,))
    ross = RosselandOpacity(odf, RosselandConfig())
    T = torch.tensor([5130.0, 8130.0], dtype=torch.float64)
    P = torch.tensor([1.3e4, 1.3e5], dtype=torch.float64)
    k = ross(T, P, comp)
    assert k.shape == (2,) and bool((k > 0).all())
    dt, dp = ross.log_derivatives(T, P, comp)
    h = 1e-5
    f = lambda t, p: torch.log(ross(t, p, comp))   # noqa: E731
    fdt = (f(T * math.exp(h), P) - f(T * math.exp(-h), P)) / (2 * h)
    fdp = (f(T, P * math.exp(h)) - f(T, P * math.exp(-h))) / (2 * h)
    assert torch.allclose(dt, fdt, rtol=1e-5, atol=1e-6) and torch.allclose(dp, fdp, rtol=1e-5, atol=1e-6)


def test_kurucz_config():
    c = RosselandConfig.kurucz()
    assert c.const is KURUCZ and c.normalization == "analytic" and c.kapk_rounding == "truncate"
    q = c.frequencies(IntervalSet.big())
    assert abs(float(q.weight[0]) - (3.364687e16 - float(q.nu[1])) / 2) < 1.0
    assert "dfsynthe_files" in c.describe()["molecules_path"]


@pytest.mark.stage
@pytest.mark.parametrize("mix", ["solar", "alpha"])
def test_kappa9_tables_from_stock_odf(mix):
    """B4: RosselandConfig.kurucz() on the stock p00big{v}.bdf against the stock KAPPA9 tables and against the
    exact-port KAPPA9 with HCKT updated per temperature (stock evaluates PFSAHA with the control deck's HCKT: up to
    0.076 dex at log T 3.3-3.6). Gates (results/v2/tolerances.json stages.kappa_ros) on the HCKT-corrected tables;
    numbers to results/v2/rosseland_validation.json."""
    import json
    from atlas_torch.scoring import gate_v2
    from atlas_torch.v2.io.kappa_ros import read_kapk
    from atlas_torch.v2.io.odf_directory import read_abundance_cards
    repo = Path(__file__).resolve().parents[2]
    src = need(EXT / "runs" / "stock" / "odf" / mix)
    tab = RosselandOpacity(OpacityDistribution.read(src, "big"), RosselandConfig.kurucz()).tabulate(
        read_abundance_cards(src / "kappa9v0.com"))
    tols = json.loads((repo / "results/v2/tolerances.json").read_text())["stages"]["kappa_ros"]
    res = {}
    for name, ref in (("stock", EXT / "work" / f"kappa9_{mix}_stock"),
                      ("stock_hckt_fixed", EXT / "runs/v2/kappa_ros/kappa9_hckt_fixed" / mix)):
        if not (ref / "kapk0.dat").exists():
            continue
        m, all_d = {}, []
        for j, v in enumerate((0, 1, 2, 4, 8)):
            d = tab.log_kappa[j].numpy() - read_kapk(ref / f"kapk{v}.dat").T / 1000.0
            m[f"kapk_v{v}:abs"] = float(np.abs(d).max())
            m[f"kapk_v{v}:rms_abs"] = float(np.sqrt((d ** 2).mean()))
            all_d.append(d)
        all_d = np.stack(all_d)                       # kappa.ros holds the five kapk tables
        m["logkappa_ros:abs"], m["logkappa_ros:rms_abs"] = float(np.abs(all_d).max()), float(np.sqrt((all_d ** 2).mean()))
        res[name] = {"metrics": m, "fails": {k: list(v) for k, v in gate_v2(m, tols).items()}}
    out = repo / "results" / "v2" / "rosseland_validation.json"
    doc = json.loads(out.read_text()) if out.exists() else {}
    doc.setdefault("kurucz_config_vs_kappa9", {})[mix] = res
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    if "stock_hckt_fixed" in res:
        assert not res["stock_hckt_fixed"]["fails"], res["stock_hckt_fixed"]["fails"]


@pytest.mark.stage
def test_abundance_gradient():
    """d log10 kappa_R / d log10 A(Fe) through the EOS (implicit function theorem) and the continuum, by autograd,
    against a central difference (stock ODF, three (T, P) points)."""
    from atlas_torch.v2.composition import Composition
    from atlas_torch.v2.io import read_atlas_deck
    comp = read_atlas_deck(need(EXT / "runs/stock/atlas/sun/output_summary.out")).composition
    ross = RosselandOpacity(OpacityDistribution.read(need(ODF), "big", vturbs=(2,)), RosselandConfig())
    T = torch.tensor([4000.0, 6000.0, 9000.0], dtype=torch.float64)
    P = torch.tensor([1e3, 1e5, 1e4], dtype=torch.float64)
    base = torch.log10(comp.abundance)

    def f(dfe):
        lg = base.clone()
        lg[25] = lg[25] + dfe
        return torch.log10(ross(T, P, Composition(10.0 ** lg, comp.scale)))
    x = torch.zeros((), dtype=torch.float64, requires_grad=True)
    g = torch.stack([torch.autograd.grad(f(x)[i], x, retain_graph=True)[0] for i in range(3)])
    h = 1e-4
    fd = (f(torch.tensor(h)) - f(torch.tensor(-h))) / (2 * h)
    assert torch.allclose(g, fd, rtol=1e-4, atol=1e-7)
