"""DATA tables (incl. EQUIVALENCE-assembled ones: IONPOTS' POTION, PFSAHA's NNN, PFIRON's PFTAB, XKARSAS'
FREQN/XN/XL, CHOP/OHOP cross-sections (DATA statements without a blank: 'DATAC10/'), H2COLLOP, HOTOP, HMINOP)
read from the fetched sources, validated bit-for-bit against the compiled oracle libraries' SAVE images."""
import numpy as np
import pytest

from atlas_torch.atlas9.physics import _testing as T
from atlas_torch.atlas9.physics._data import equivalenced, potion, table

EQUIV = [("pfsaha", "nnn"), ("pfiron", "pftab"), ("xkarsas", "freqn"), ("xkarsas", "xn"), ("xkarsas", "xl"),
         ("hminop", "ff"), ("chop", "crossch"), ("ohop", "crossoh"), ("h2collop", "h2he"), ("h2collop", "h2h2"),
         ("hotop", "a")]
PLAIN = [("pfsaha", k) for k in ("ehyd", "ghyd", "ehe1", "ghe1", "ehe2", "ghe2", "ec1", "gc1", "ec2", "gc2",
                                 "emg1", "gmg1", "emg2", "gmg2", "eal1", "gal1", "esi1", "gsi1", "esi2", "gsi2",
                                 "ena1", "gna1", "eb1", "gb1", "ek1", "gk1", "eo1", "go1", "locz", "scale")] + [
    ("pfiron", "potlo"), ("pfiron", "potlolog"), ("xkarsas", "ekarsas"), ("coulff", "a"), ("coulff", "z4log"),
    ("coulbf1s", "gaunt1s"), ("hminop", "wbf"), ("hminop", "bf"), ("hminop", "wavek"), ("hminop", "thetaff"),
    ("hrayop", "gavrilam"), ("hrayop", "gavrilamab"), ("hrayop", "gavrilambc"), ("hrayop", "gavrilamcd"),
    ("hrayop", "gavrilalymancont"), ("hrayop", "fgavrilalymancont"), ("he1op", "g"), ("he1op", "chi"),
    ("he1op", "hefreq"), ("crosshe", "x505"), ("crosshe", "x50"), ("crosshe", "x20"), ("crosshe", "x10"),
    ("he12s1s", "freq1s"), ("he12s1s", "x1s"), ("he12s3s", "freq3s"), ("he12s3s", "x3s"), ("he12p1p", "freq1p"),
    ("he12p1p", "x1p"), ("he12p3p", "freq3p"), ("he12p3p", "x3p"), ("mg1op", "peach"), ("mg1op", "freqmg"),
    ("mg1op", "flog"), ("mg1op", "tlg"), ("si2op", "peach"), ("si2op", "freqsi"), ("si2op", "flog"),
    ("si2op", "tlg"), ("si1op", "elev"), ("si1op", "glev"), ("fe1op", "g"), ("fe1op", "e"), ("fe1op", "wno"),
    ("chop", "partch"), ("ohop", "partoh"), ("si1seaton3p4p3s", "flam3p4p3s"), ("si1seaton3p4p3s", "x3p4p3s"),
    ("si1seaton3p4p1d", "flam3p4p1d"), ("si1seaton3p4p1d", "x3p4p1d"), ("si1seaton3p3d3p", "flam3p3d3p"),
    ("si1seaton3p3d3p", "x3p3d3p")]


@pytest.mark.parametrize("variant", ["atlas9", "kappa9"])
def test_tables_match_oracle_image(variant):
    if variant == "kappa9" and not T.preload_intel():
        pytest.skip("Intel runtime not found")
    o = T.make_oracle(variant)
    assert np.array_equal(potion(variant), o.common("potion")["potion"])
    for unit, var in EQUIV + PLAIN:
        a = equivalenced(variant, unit, var) if (unit, var) in EQUIV else table(variant, unit, var)
        e = o.saved(unit)[var]
        assert np.array_equal(np.asarray(a).reshape(-1, order="F"), np.asarray(e).reshape(-1, order="F")), (unit, var)


def test_pfground_cases_parse():
    from atlas_torch.atlas9.physics._data import pfground_cases
    c = pfground_cases("atlas9")
    assert c[43] == c[431] and "(-HCK/T*97427.)" in c[54].replace(" ", "")      # fall-through + missing EXP
