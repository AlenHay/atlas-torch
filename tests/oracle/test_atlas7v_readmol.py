"""Oracle tests for SUBROUTINE READMOL (atlas7v.for): the molecule list (unit 2, FORMAT(F18.2,F7.3,6E11.4))
-> COMMON /IFEQUA/ (components, equation numbers) and /XNMOL/ codes + equilibrium constants.
  golden:   READMOL on BasicATLAS' synthe_files/molecules.dat == the traced READMOL exit (COMMONs + the SAVEd
            locals C, ..., KLOC, X), bitwise
  property: randomly thinned / perturbed molecule lists written with other legal F/E field spellings (D and
            sign-led exponents, implied decimal points, blank padding) vs the Fortran READMOL reading them
            as fort.2, bitwise.
"""
import os
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from atlas_torch.synthe.readin import default_molecules_path, readmol
from test_atlas7v_pfsaha import STAGE

KEYS = ("ifequa.ifequa", "ifequa.kcomps", "ifequa.locj", "ifequa.equil", "ifequa.idequa", "ifequa.nequa",
        "ifequa.nloc", "xnmol.code", "xnmol.nummol")


def test_readmol_golden(golden):
    x = golden(STAGE, "readmol", 0, "exit")
    r = readmol()
    for k in KEYS:
        assert np.array_equal(r[k].numpy().reshape(-1).astype(np.float64), x[k].reshape(-1).astype(np.float64)), k
    assert np.array_equal(r["xnmol.codemol"].numpy(), r["xnmol.code"].numpy())


def _respell(rng, field: str, w: int, d: int) -> str:
    """Another legal Fortran spelling of the same numeric field (width w, d decimals)."""
    v = field.strip()
    if not v:
        return field
    r = rng.random()
    if "E" in v and r < 0.3:
        v = v.replace("E", "D")
    elif "E" in v and r < 0.5:
        v = v.replace("E", "")                       # sign-led exponent: 4.4942+01
    elif "E" not in v and "." in v and r < 0.4:      # implied decimal point: 2.429 -> 2429 (d=3)
        ip, fp = v.split(".")
        fp = (fp + "0" * d)[:d]
        if len(fp) == d:
            v = (ip + fp).lstrip("0") or "0"
    v = v.rjust(w) if rng.random() < 0.7 else v.ljust(w)
    return v[:w] if len(v) <= w else field


def make_molecules(seed: int) -> str:
    rng = np.random.default_rng(seed)
    lines = default_molecules_path().read_text(encoding="latin-1").split("\n")
    out = []
    for ln in lines:
        if not ln.strip():
            continue
        if ln[:18].strip() in ("0.", "0") :
            break
        if rng.random() < 0.15:
            continue
        f = [ln[0:18], ln[18:25]] + [ln[25 + 11 * i:36 + 11 * i] for i in range(6)]
        f = [x.ljust(w) for x, w in zip(f, (18, 7) + (11,) * 6)]
        if f[1].strip() and rng.random() < 0.5:
            f[1] = f"{float(f[1]) * rng.uniform(0.9, 1.1):7.3f}"
        f = [f[0], _respell(rng, f[1], 7, 3)] + [_respell(rng, x, 11, 4) for x in f[2:]]
        out.append("".join(f))
    out.append("         0.")
    return "\n".join(out) + "\n"


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 2**31 - 1))
def test_readmol_vs_fortran(oracle, tmp_path, seed):
    text = make_molecules(seed)        # tmp_path reused across examples: files are overwritten each time
    (tmp_path / "mol.dat").write_text(text)
    (tmp_path / "fort.2").write_text(text)
    o = oracle("atlas7v")
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        o.call("readmol")
    finally:
        os.chdir(cwd)
    r = readmol(tmp_path / "mol.dat")
    ref = {**{"ifequa." + k: v for k, v in o.common("ifequa").items()},
           **{"xnmol." + k: v for k, v in o.common("xnmol").items()}}
    for k in KEYS:
        assert np.array_equal(r[k].numpy().reshape(-1).astype(np.float64),
                              np.asarray(ref[k]).reshape(-1).astype(np.float64)), (k, seed)
