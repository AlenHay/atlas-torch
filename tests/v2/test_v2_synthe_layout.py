"""v2 populations in XNFPELSYN's fort.10 layout against the exact port's XNFPEL."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def both(solar_deck):
    from atlas_torch.synthe.xnfpelsyn_prog import xnfpelsyn
    from atlas_torch.v2.eos import ChemicalEquilibrium
    from atlas_torch.v2.io import format_atlas_deck
    from atlas_torch.v2.io.synthe_layout import synthe_state, xnfpel
    ref = xnfpelsyn(format_atlas_deck(solar_deck))
    pops = ChemicalEquilibrium().solve(solar_deck.atmosphere, solar_deck.composition)
    return xnfpel(pops).numpy(), synthe_state(pops), ref


def test_slot_pattern(both):
    x, _, ref = both
    o = ref.xnfpel
    assert not ((o > 0) & (x == 0)).any()
    from atlas_torch.v2.io.synthe_layout import idmol
    extra = {round(float(idmol()[b - 39]), 2) for _, a, b in np.argwhere((o == 0) & (x > 1e-20))}
    assert extra == {106.01, 108.01, 114.01, 10101.01}     # IDMOL REAL*4 codes miss the ionised molecules


def test_xnfpel_values(both):
    """Layers >= 2, entries > 1e-20 cm^-3; without K I, Na I, Mg I, Al I and their molecules (NMOLEC's lagged
    Debye lowering) and LiH, BeH, BH (NMOLEC treats Li, Be, B as neutral only)."""
    from atlas_torch.v2.eos.molecules import decode_code
    from atlas_torch.v2.io.synthe_layout import idmol
    x, _, ref = both
    o = ref.xnfpel
    sig = (o > 1e-20) & (x > 0)
    sig[0] = False
    for z in (11, 12, 13, 19):
        sig[:, 0, z - 1] = False
    for k, c in enumerate(idmol()):
        if c and set(decode_code(c)[0]) & {3, 4, 5, 11, 12, 13, 19}:
            sig[:, 5, 39 + k] = False
    rel = np.abs(x[sig] / o[sig] - 1)
    assert np.median(rel) < 0.01 and np.percentile(rel, 99) < 0.05 and rel.max() < 0.05


def test_state_record(both):
    _, s, ref = both
    n = s["xne"].shape[0]
    for key, tol in (("xne", 0.02), ("xnatom", 1e-3), ("rho", 1e-3), ("xnfh", 1e-3), ("xnfhe", 0.02)):
        r = ref.state[key][:n]
        assert np.max(np.abs(s[key].numpy() / r - 1)) < tol, key
    cold = ref.state["t"][:n] <= 9000.0                    # XNFH2 is cut at 9000 K in XNFPELSYN, 10^4 K in NMOLEC
    assert np.max(np.abs(s["xnfh2"].numpy()[cold] / ref.state["xnfh2"][:n][cold] - 1)) < 0.02
