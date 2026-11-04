"""Partition functions against PFSAHA (exact port) at the solar model's layers, same lowering."""
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def layers(solar_deck):
    from atlas_torch.synthe.pfsaha import Layers
    from atlas_torch.synthe.readin import readin
    from atlas_torch.v2.io import format_atlas_deck
    s = readin(format_atlas_deck(solar_deck))
    return Layers.from_state(s)


def _lowering(L, const):
    from atlas_torch.v2.eos.saha import debye_lowering, kurucz_charge_density
    return debye_lowering(L.t, kurucz_charge_density(L.xne, L.p / L.tk), const)


def _worst(U, L):
    from atlas_torch.synthe.pfsaha import pfsaha
    from atlas_torch.v2.eos.partition import n_stages
    out = {}
    for z in range(1, 100):
        ns = n_stages(z)
        ref = pfsaha(L, z, ns, 13)
        d = (U[:, z - 1, :ns] / ref - 1).abs().amax(0)
        for s in range(ns):
            out[(z, s + 1)] = float(d[s])
    return out


def test_partition_functions_kurucz_constants(layers):
    from atlas_torch.v2.constants import KURUCZ
    from atlas_torch.v2.eos.partition import PartitionFunctions
    dev = _worst(PartitionFunctions(KURUCZ)(layers.t, _lowering(layers, KURUCZ)), layers)
    # N VII: PFGROUND label 43 falls through to label 431 (O I levels) in the Fortran
    assert dev.pop((7, 7)) > 0.5
    assert max(dev.values()) < 2e-6           # Ca-Ni: PFIRON is a REAL*4 routine


def test_partition_functions_codata(layers):
    from atlas_torch.v2.constants import CODATA2018
    from atlas_torch.v2.eos.partition import PartitionFunctions
    dev = _worst(PartitionFunctions(CODATA2018)(layers.t, _lowering(layers, CODATA2018)), layers)
    dev.pop((7, 7))
    assert max(dev.values()) < 2e-3           # Rydberg 13.595 vs 13.6057 eV in the hydrogenic tail


def test_level_and_ground_data_from_source():
    from atlas_torch.v2.eos.partition import ground_levels, level_blocks
    lev = level_blocks()
    assert set(lev) == {(1, 1), (2, 1), (2, 2), (6, 1), (6, 2), (8, 1), (11, 1), (12, 1), (12, 2), (13, 1),
                        (14, 1), (14, 2), (5, 1), (19, 1)}
    h = lev[(1, 1)]
    np.testing.assert_allclose(h.g, [2, 8, 18, 32, 50, 72])
    assert h.tail_g == 2.0 and abs(h.tail_start_cm - 109677.576 / 6.5 ** 2) < 1e-9
    assert lev[(6, 1)].tail_g is None
    gr = ground_levels()
    assert (56, 2) in gr and (8, 1) not in gr and (9, 1) not in gr
    np.testing.assert_array_equal(gr[(7, 7)].g, [2.0])            # label 43's own value, not the fall-through


def test_monotone_and_lowering_dependence():
    from atlas_torch.v2.eos.partition import PartitionFunctions
    pf = PartitionFunctions()
    T = torch.linspace(3000.0, 30000.0, 50)
    u1 = pf(T, torch.full_like(T, 0.01))
    u2 = pf(T, torch.full_like(T, 0.1))
    assert torch.isfinite(u1).all() and (u1 >= 1.0 - 1e-12).all()
    assert (u2[:, 25, 0] <= u1[:, 25, 0] + 1e-12).all()          # Fe I: larger lowering, fewer bound levels
    assert torch.allclose(u1[:, 0, 1], torch.ones(50))           # H II
