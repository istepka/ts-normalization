"""The LTS color palette every figure draws from."""

import json
from pathlib import Path

import matplotlib as mpl

PALETTE = json.loads((Path(__file__).parent / "color_palette.json").read_text())
IDENTITY = PALETTE["identity"]
CATEGORICAL = PALETTE["categorical"]
CATEGORICAL_ORDER = PALETTE["categoricalOrder"]

PRIMARY = IDENTITY["primary"]
SECONDARY = IDENTITY["secondary"]
ACCENT = IDENTITY["accent"]
TEXT = IDENTITY["text"]
TEAL = CATEGORICAL["companionTeal"]


def apply_palette() -> None:
    """Make the categorical order matplotlib's default color cycle."""
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CATEGORICAL_ORDER)


def categorical_colors(count: int) -> list[str]:
    """The first `count` categorical colors in their canonical order."""
    if count > len(CATEGORICAL_ORDER):
        raise ValueError(f"palette holds {len(CATEGORICAL_ORDER)} categorical colors")
    return CATEGORICAL_ORDER[:count]
