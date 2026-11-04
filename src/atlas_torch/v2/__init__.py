"""atlas_torch.v2: physics-first model-atmosphere code (Kurucz ATLAS/SYNTHE physics, own data structures).

    from atlas_torch.v2 import read_atlas_deck, ChemicalEquilibrium
    deck = read_atlas_deck("output_summary.out")
    pops = ChemicalEquilibrium().solve(deck.atmosphere, deck.composition)
"""
from .atmosphere import Atmosphere
from .composition import Composition
from .constants import CODATA2018, KURUCZ, PhysicalConstants
from .eos import ChemicalEquilibrium, Populations
from .io import read_atlas_deck

__all__ = ["Atmosphere", "Composition", "CODATA2018", "KURUCZ", "PhysicalConstants", "ChemicalEquilibrium",
           "Populations", "read_atlas_deck"]
