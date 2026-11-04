"""JOSH (atlas9mem.for): golden traced calls (continuum + first two ODF steps of the first frequency) and
perturbed opacity/source inputs vs the Fortran subroutine; batched problems give the same as single ones."""
import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from atlas_torch.atlas9.josh import josh
from atlas_torch.state import State

STAGE = "sun_atlas9/atlas9"
OUT = (("abtot", "abtot.abtot"), ("alpha", "abtot.alpha"), ("taunu", "taushj.taunu"), ("snu", "taushj.snu"),
       ("hnu", "taushj.hnu"), ("jnu", "taushj.jnu"), ("jmins", "taushj.jmins"))
IN = ("acont", "scont", "aline", "sline", "sigmac", "sigmal")


def _port(e, n):
    T = lambda k: torch.as_tensor(np.asarray(e[f"optot.{k}"])[:n]).reshape(-1, n)
    return josh(*(T(k) for k in IN), torch.as_tensor(np.asarray(e["rhox.rhox"])[:n]),
                int(np.asarray(e["arg:ifscat"]).reshape(-1)[0]), int(np.asarray(e["arg:ifsurf"]).reshape(-1)[0]))


def test_josh_golden(golden):
    for hit in range(3):
        e, x = golden(STAGE, "josh", hit, "entry"), golden(STAGE, "josh", hit, "exit")
        n = int(e["rhox.nrhox"][0])
        r = _port(e, n)
        for k, key in OUT:
            assert np.array_equal(r[k][0].numpy(), x[key][:n]), (hit, k)
        assert float(r["knu1"][0]) == x["pzero.knu"][0]


@settings(max_examples=40, deadline=None)
@given(hit=st.integers(0, 2), seed=st.integers(0, 2**31 - 1), lscale=st.floats(-3.0, 4.0),
       ifscat=st.sampled_from([1, 1, 0]))
def test_josh_vs_fortran(oracle, golden, hit, seed, lscale, ifscat):
    """Scale the traced opacities (lines by 10**lscale, random +-30% per layer) and sources."""
    e = dict(golden(STAGE, "josh", hit, "entry"))
    n = int(e["rhox.nrhox"][0])
    rng = np.random.default_rng(seed)
    for k in IN:
        a = e[f"optot.{k}"].copy()
        a[:n] *= rng.uniform(0.7, 1.3, n) * (10.0 ** lscale if k == "aline" else 1.0)
        e[f"optot.{k}"] = a
    e["optot.aline"][:n] = np.maximum(e["optot.aline"][:n], 1e-30 * (10.0 ** lscale))
    e["arg:ifscat"] = np.array([ifscat])
    o = oracle("atlas9")
    s = State.from_trace(e)
    for blk in ("optot", "rhox", "taushj", "abtot", "pzero"):
        o.set_common(blk, **{k.split(".")[1]: v.numpy() for k, v in s.items() if k.startswith(blk + ".")})
    o.call("josh", ifscat, 0, float(e["arg:freq"][0]))
    r = _port(e, n)
    for k, key in OUT:
        b, c = key.split(".")
        assert np.array_equal(r[k][0].numpy(), o.common(b)[c][:n]), k
    assert float(r["knu1"][0]) == o.common("pzero")["knu"][0]


def test_josh_batched_equals_single(golden):
    es = [golden(STAGE, "josh", h, "entry") for h in range(3)]
    n = int(es[0]["rhox.nrhox"][0])
    singles = [_port(e, n) for e in es]
    stack = lambda k: torch.cat([torch.as_tensor(e[f"optot.{k}"][:n]).reshape(1, n) for e in es])
    b = josh(*(stack(k) for k in IN), torch.as_tensor(es[0]["rhox.rhox"][:n]), 1, 0)
    for i, r in enumerate(singles):
        for k, _ in OUT:
            assert np.array_equal(b[k][i].numpy(), r[k][0].numpy()), k
