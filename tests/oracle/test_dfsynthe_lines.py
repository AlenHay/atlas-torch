"""DFSYNTHE MAIN line loop (labels 301-359: TAPE12 line profiles) and continuum.

1. golden: layer-1 continuum equals the traced CONTINUUM; layer-1 spectrum = traced buffer after XLINOP + the
   port's TAPE12 accumulation + MAX(.,CONTINUUM) equals the traced DFCALC-entry buffer, bit for bit.
2. sequential reference: a direct scalar transcription of the Fortran loop (one line after the other, linear
   first-crossing scans, REAL*4 numpy) on real lines with damping constants spread over all four ADAMP branches
   (incl. the stale-AA quirk of the 0.2..1.4 branch, carried across layers) == the vectorised port, bit for bit.
"""
import numpy as np
import pytest
import torch

from atlas_torch.dfsynthe import odf_io
from atlas_torch.dfsynthe.lines import oracle_support as S
from atlas_torch.dfsynthe.lines import tables as T
from atlas_torch.dfsynthe.lines.accumulate import LineAccumulator
from atlas_torch.dfsynthe.lines.continuum import continuum, layer_setup
from atlas_torch.dfsynthe.lines.f32 import Flavor, lit
from atlas_torch.dfsynthe.lines.selectlines import selectlines

pytestmark = pytest.mark.skipif(not (S.available() and S.DATA.exists() and S.STOCK.exists()),
                                reason="needs libdfsynthe.so, Intel runtime, line lists and the stock solar run")
STAGE = "odf_solar/dfsynthe"
f4 = np.float32


@pytest.fixture(scope="module")
def setup():
    fl = S.intel_flavor()
    xt = odf_io.read_xnfpdf(S.STOCK / "xnfpdf.dat", 21)
    mx = odf_io.read_xnfpdfmax(S.STOCK / "xnfpdfmax.dat", 21)
    t12, t19, _ = selectlines(S.DATA, mx.teff, mx.wledge, mx.xnfdopmax, fl)
    lay = layer_setup(xt, fl, torch.device("cpu"))
    return fl, xt, t12, lay


def _t12(t12, sl=slice(None)):
    return {k: torch.from_numpy(np.ascontiguousarray(getattr(t12, k)[sl]).astype(np.int64)
                                if getattr(t12, k).dtype.kind == "i" else np.ascontiguousarray(getattr(t12, k)[sl]))
            for k in ("nbuff", "congf", "nelion", "gamr", "gams", "gamw")}


def test_layer1_golden(golden, setup):
    fl, xt, t12, lay = setup
    cont = continuum(xt.wledge, lay["ablog"][:1], fl)
    g = golden(STAGE, "xlinop", 0, "entry")
    assert int(g["arg:j"][0]) == 1
    assert np.array_equal(cont[0].numpy().view(np.int32), g["contin.continuum"][: T.LENGTH].view(np.int32))
    tabs = T.device_tables(torch.device("cpu"), fl)
    acc = LineAccumulator(tabs, lay["xne"][:1], lay["txnxn"][:1], lay["dopple"][:1], lay["xnfdop"][:1], cont)
    buf = torch.from_numpy(golden(STAGE, "xlinop", 0, "exit")["buffer.buffer"].copy())
    acc.accumulate(buf, _t12(t12))
    got = torch.maximum(buf[: T.LENGTH], cont[0]).numpy()
    ref = golden(STAGE, "dfcalc", 0, "entry")["buffer.buffer"][: T.LENGTH]
    assert np.array_equal(got.view(np.int32), ref.view(np.int32))


# ------------------------------------------------------------------ sequential reference (Fortran transcription)
def _reference(t, xne, txnxn, dopple, xnfdop, cont, tabs, aa0=f4(0)):
    h0, h1, atab, freq1000 = (tabs[k].numpy() for k in ("h0tab", "h1tab", "atab", "freq1000"))
    nl = cont.shape[0]
    buf = np.zeros((nl, T.LENBUFF), f4)
    C = lambda s: f4(lit(s))   # noqa: E731
    aa = aa0
    for j in range(nl):
        for i in range(len(t["nbuff"])):
            nb, nel = int(t["nbuff"][i]), int(t["nelion"][i])
            contmin = cont[j, nb - 1]
            k0 = f4(t["congf"][i] * xnfdop[j, nel - 1])
            if k0 < contmin:
                continue
            d = dopple[j, nel - 1]
            adamp = f4(f4(f4(t["gamr"][i] + f4(t["gams"][i] * xne[j])) + f4(t["gamw"][i] * txnxn[j])) / d)
            dv = f4(np.float64(f4(f4(1) / d)) / T.RESOLU)
            ks = np.arange(1, T.MAXPROF + 1)
            kd = (ks.astype(f4) * dv).astype(f4)
            if adamp <= C(".2"):
                center = f4(k0 * f4(f4(1) - f4(C("1.128") * adamp)))
                tabstep = f4(np.float64(T.VSTEPS) / (np.float64(d) * T.RESOLU))
                n10 = int(np.float64(f4(f4(10) * d)) * T.RESOLU)
                prof = np.zeros(T.MAXPROF + 1, f4)
                tabi = f4(1.5)
                nstep = None
                for k in range(1, n10 + 1):
                    tabi = f4(tabi + tabstep)
                    it = int(tabi) - 1
                    prof[k] = f4(k0 * f4(h0[it] + f4(adamp * h1[it])))
                    if prof[k] < contmin:
                        nstep = k
                        break
                if nstep is None:
                    x = f4(prof[n10] * f4(f4(n10) * f4(n10)))
                    maxstep = min(int(f4(np.sqrt(np.float64(f4(x / contmin)))) + f4(1)), T.MAXPROF)
                    if f4(t["congf"][i] * freq1000[nb // 1000]) < C(".0001") or nel % 6 == 0:
                        maxstep = min(maxstep, n10 * 10)
                    kk = np.arange(n10 + 1, maxstep + 1)
                    prof[n10 + 1:maxstep + 1] = (x / (kk.astype(f4) * kk.astype(f4))).astype(f4)
                    nstep = maxstep
            else:
                if adamp > 100.0:
                    pc = f4(f4(f4(k0 / adamp) * C(".79788")) / C("1.4142"))
                    center = pc
                    aa = f4(adamp * adamp)
                    p = (f4(pc * aa) / (aa + kd * kd)).astype(f4)
                    start = 1
                elif adamp > C("1.4"):
                    aa = f4(adamp * adamp)
                    profc = f4(f4(k0 * adamp) * C(".79788"))
                    center = f4(f4(f4(f4(k0 / adamp) * C(".79788")) / C("1.4142")) *
                                f4(f4(1) + f4(f4(f4(C(".75") / aa) - f4(.5)) / aa)))
                    p = _asym(profc, aa, kd)
                    start = 1
                else:
                    ia = int(f4(f4(adamp * f4(200)) + f4(1.5))) - 1
                    center = f4(k0 * atab[ia, 0])
                    n5 = int(np.float64(f4(f4(5) * d)) * T.RESOLU)
                    iv = (((ks[:n5].astype(f4) * dv).astype(f4) * f4(200)).astype(f4) + f4(1.5)).astype(f4)
                    core = (k0 * atab[ia, iv.astype(np.int64) - 1]).astype(f4)
                    profc = f4(f4(k0 * adamp) * C(".79788"))
                    p = np.concatenate([core, _asym(profc, aa, kd[n5:])])      # AA: stale state
                    start = 1
                below = np.nonzero(p < contmin)[0]
                nstep = int(below[0]) + 1 if below.size else T.MAXPROF
                prof = np.concatenate([[f4(0)], p]).astype(f4)
            buf[j, nb - 1] = f4(buf[j, nb - 1] + center)
            r = min(T.LENGTH - nb, nstep) if nb < T.LENGTH else 0
            b = min(nb - 1, nstep) if nb > 1 else 0
            buf[j, nb:nb + r] = (buf[j, nb:nb + r] + prof[1:r + 1]).astype(f4)
            if b:
                buf[j, nb - 1 - b:nb - 1] = (buf[j, nb - 1 - b:nb - 1] + prof[1:b + 1][::-1]).astype(f4)
    return buf


def _asym(profc, aa, kd):
    vv = (kd * kd).astype(f4)
    u = ((aa + vv) * f4(lit("1.4142"))).astype(f4)
    aau = (aa / u).astype(f4)
    vvu = (vv / u).astype(f4)
    inner = ((((aau - f4(10) * vvu) * aau + f4(5) * vvu * vvu) / u + vvu - aau / f4(3)) * f4(3) / u + f4(1))
    return ((profc / u).astype(f4) * inner.astype(f4)).astype(f4)


@pytest.mark.parametrize("seed", [0, 1])
def test_accumulate_vs_sequential_reference(setup, seed):
    fl, xt, t12, lay = setup
    rng = np.random.default_rng(seed)
    s0 = int(rng.integers(0, len(t12) - 400))
    t = {k: np.array(getattr(t12, k)[s0:s0 + 120]) for k in ("nbuff", "congf", "nelion", "gamr", "gams", "gamw")}
    t["gamw"] = (t["gamw"] * (10.0 ** rng.uniform(-1, 6, len(t["gamw"])))).astype(f4)   # spread ADAMP
    t["congf"] = (t["congf"] * (10.0 ** rng.uniform(0, 4, len(t["congf"])))).astype(f4)
    layers = [0, 12, 24]
    cont = (np.full((3, T.LENGTH), 1e-3, f4) * (10.0 ** rng.uniform(-1, 1, (3, 1)))).astype(f4)
    xne, txnxn = lay["xne"][layers].numpy(), lay["txnxn"][layers].numpy()
    dop, xfd = lay["dopple"][layers].numpy(), lay["xnfdop"][layers].numpy()
    tabs = T.device_tables(torch.device("cpu"), Flavor("O0"))
    ref = _reference(t, xne, txnxn, dop, xfd, cont, tabs)
    acc = LineAccumulator(tabs, *(torch.from_numpy(a) for a in (xne, txnxn, dop, xfd, cont)), max_elements=200_000)
    buf = torch.zeros(3 * T.LENBUFF, dtype=torch.float32)
    tt = {k: torch.from_numpy(v.astype(np.int64) if v.dtype.kind == "i" else v) for k, v in t.items()}
    acc.accumulate(buf, tt, chunk=37)
    assert np.array_equal(buf.view(3, -1).numpy().view(np.int32), ref.view(np.int32))
