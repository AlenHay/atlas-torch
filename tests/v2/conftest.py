import pytest

from _oracle import model_path, oracle_populations


@pytest.fixture(scope="session")
def solar_deck():
    from atlas_torch.v2.io import read_atlas_deck, to_synthe_deck
    p = model_path("sun")
    if not p.exists():
        pytest.skip(f"missing {p}")
    return to_synthe_deck(read_atlas_deck(p))


@pytest.fixture(scope="session")
def solar_oracle(solar_deck):
    from atlas_torch.v2.io import format_atlas_deck
    return oracle_populations(format_atlas_deck(solar_deck))
