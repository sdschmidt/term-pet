"""Stat configuration and generation."""

import random

from pydantic import BaseModel, Field

from tpet.models.rarity import Rarity

DEFAULT_STAT_NAMES: list[str] = ["HUMOR", "PATIENCE", "CHAOS", "WISDOM", "SNARK"]


class StatConfig(BaseModel):
    """Configuration for pet stat generation."""

    names: list[str] = Field(
        default_factory=lambda: list(DEFAULT_STAT_NAMES),
        description=(
            "Ordered list of stat names used during random fallback generation. "
            "Default: HUMOR, PATIENCE, CHAOS, WISDOM, SNARK."
        ),
    )
    pool_size: int = Field(
        default=5,
        description="Number of stats to generate per pet during random fallback. Must be <= len(names).",
    )


def generate_stats(config: StatConfig, rarity: Rarity) -> dict[str, int]:
    """Generate random stats based on rarity range.

    Args:
        config: Stat configuration with names and pool size.
        rarity: Determines the value range for generated stats.

    Returns:
        Dict mapping stat names to integer values.
    """
    count = min(config.pool_size, len(config.names))
    chosen_names = random.sample(config.names, count)
    lo, hi = rarity.stat_range
    return {name: random.randint(lo, hi) for name in chosen_names}
