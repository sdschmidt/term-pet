"""Stat bar rendering using Rich."""

from rich.text import Text

BAR_WIDTH = 10
FILLED_CHAR = "\u2588"  # Full block
EMPTY_CHAR = "\u2591"  # Light shade
STAT_NAME_WIDTH = 10


def render_stat_bars(stats: dict[str, int], color: str) -> list[Text]:
    """Render stat bars as Rich Text objects.

    Args:
        stats: Mapping of stat name to value (0-99).
        color: Rich color for filled portion.

    Returns:
        List of Text objects, one per stat.
    """
    bars: list[Text] = []
    for name, value in stats.items():
        filled = round(value / 99 * BAR_WIDTH)
        empty = BAR_WIDTH - filled

        text = Text()
        text.append(f"  {name:<{STAT_NAME_WIDTH}}", style="bold")
        text.append(FILLED_CHAR * filled, style=color)
        text.append(EMPTY_CHAR * empty, style="dim")
        text.append(f"  {value:>2}", style="bold")
        bars.append(text)
    return bars
