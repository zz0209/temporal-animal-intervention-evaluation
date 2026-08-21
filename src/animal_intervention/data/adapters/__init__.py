"""Dataset-specific raw-to-canonical adapters."""

from .baboon import GuineaBaboonsAdapter
from .barn_swallow import BarnSwallowsAdapter
from .base import BaseAdapter
from .free_ranging_sheep import FreeRangingSheepAdapter
from .oxford import OxfordWildbirdAdapter
from .radolfzell import RadolfzellGreatTitsAdapter
from .sheep import DomesticSheepAdapter
from .songbirds import ExperimentalSongbirdsAdapter
from .wytham import WythamGreatTitsAdapter
from .vampire_bats import WildVampireBatsAdapter


ADAPTERS: dict[str, type[BaseAdapter]] = {
    "oxford_wildbird_network": OxfordWildbirdAdapter,
    "guinea_baboons_sociopatterns": GuineaBaboonsAdapter,
    "barn_swallows_encounternet": BarnSwallowsAdapter,
    "domestic_sheep_sirtrack": DomesticSheepAdapter,
    "wytham_great_tits_divorce": WythamGreatTitsAdapter,
    "radolfzell_great_tits_ontogeny": RadolfzellGreatTitsAdapter,
    "experimental_wild_songbirds": ExperimentalSongbirdsAdapter,
    "wild_vampire_bats_proximity": WildVampireBatsAdapter,
    "free_ranging_sheep_fission_fusion": FreeRangingSheepAdapter,
}

__all__ = ["ADAPTERS", "BaseAdapter"]
