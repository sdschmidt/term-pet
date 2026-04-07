"""Tests for the Stats model."""

from tpet.models.rarity import Rarity
from tpet.models.stats import StatConfig, generate_stats

DEFAULT_STAT_NAMES = ["HUMOR", "PATIENCE", "CHAOS", "WISDOM", "SNARK"]


class TestStatConfig:
    """Tests for StatConfig model."""

    def test_defaults(self) -> None:
        config = StatConfig()
        assert config.names == DEFAULT_STAT_NAMES
        assert config.pool_size == 5

    def test_custom_names(self) -> None:
        config = StatConfig(names=["SPEED", "POWER"], pool_size=2)
        assert config.names == ["SPEED", "POWER"]
        assert config.pool_size == 2


class TestGenerateStats:
    """Tests for stat generation."""

    def test_generates_correct_count(self) -> None:
        config = StatConfig()
        stats = generate_stats(config, Rarity.COMMON)
        assert len(stats) == config.pool_size

    def test_uses_stat_names(self) -> None:
        config = StatConfig()
        stats = generate_stats(config, Rarity.COMMON)
        for name in stats:
            assert name in config.names

    def test_values_in_rarity_range(self) -> None:
        for rarity in Rarity:
            config = StatConfig()
            stats = generate_stats(config, rarity)
            lo, hi = rarity.stat_range
            for value in stats.values():
                assert lo <= value <= hi, f"{rarity}: {value} not in [{lo}, {hi}]"

    def test_pool_size_limits_stats(self) -> None:
        config = StatConfig(names=["A", "B", "C", "D", "E"], pool_size=3)
        stats = generate_stats(config, Rarity.RARE)
        assert len(stats) == 3

    def test_pool_size_capped_at_names_length(self) -> None:
        config = StatConfig(names=["A", "B"], pool_size=5)
        stats = generate_stats(config, Rarity.COMMON)
        assert len(stats) == 2
