"""Full card display renderer using Rich."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from tpet.renderer.statbars import render_stat_bars

if TYPE_CHECKING:
    from rich.console import RenderableType

    from tpet.config import TpetConfig
    from tpet.models.pet import PetProfile

logger = logging.getLogger(__name__)


def _build_halfblock_art(pet: PetProfile, config: TpetConfig) -> RenderableType | None:
    """Build a halfblock art renderable from PNG frames for embedding in a card.

    Args:
        pet: Pet profile.
        config: Application configuration.

    Returns:
        Rich renderable with the halfblock art, or None if unavailable.
    """
    from rich.ansi import AnsiDecoder
    from rich.containers import Lines

    from tpet.art.process import render_halfblock_from_png
    from tpet.art.storage import get_png_frame_path

    data_dir = config.pet_data_dir
    png_path = get_png_frame_path(data_dir, pet.name, 0)
    if not png_path.exists():
        return None

    ansi_str = render_halfblock_from_png(png_path, max_terminal_rows=20, max_terminal_cols=40)
    decoder = AnsiDecoder()
    art_lines = list(decoder.decode(ansi_str))
    if not art_lines:
        return None

    art_width = max(len(line) for line in art_lines)
    return Align.center(
        Lines(art_lines),
        width=art_width,
    )


def render_card(pet: PetProfile, *, show_backstory: bool = False, config: TpetConfig | None = None) -> Panel:
    """Render a full pet card as a Rich Panel.

    Args:
        pet: Pet profile to render.
        show_backstory: Whether to include the backstory section.
        config: Optional config for art mode awareness.

    Returns:
        Rich Panel containing the full card.
    """
    rarity_color = pet.rarity.color
    parts: list[RenderableType | str] = []

    from tpet.config import ArtMode

    # Frame title with rarity and creature type
    title = Text()
    title.append(f" {pet.rarity.stars} {pet.rarity.value}", style=f"bold {rarity_color}")
    title.append("  ")
    title.append(f"{pet.creature_type.upper()} ", style=f"bold {rarity_color}")

    # Art display
    if config is not None and config.art_mode == ArtMode.SIXEL_ART:
        # Try halfblock art from PNG frames (embedded inside the card)
        art_renderable = _build_halfblock_art(pet, config)
        if art_renderable is not None:
            parts.append(art_renderable)
            parts.append("")
    elif pet.ascii_art:
        parts.append(Align.center(Text(pet.ascii_art[0], style=pet.accent_color)))
        parts.append("")

    # Name
    name_text = Text(f"  {pet.name}", style="bold white")
    parts.append(Align.center(name_text))
    parts.append("")

    # Personality quote
    personality_lines = _wrap_text(pet.personality, width=34)
    for line in personality_lines:
        parts.append(Text(f"  {line}", style="italic"))
    parts.append("")

    # Backstory
    if show_backstory and pet.backstory:
        parts.append(Text("  backstory", style="dim"))
        backstory_lines = _wrap_text(pet.backstory, width=34)
        for line in backstory_lines:
            parts.append(Text(f"  {line}", style="white"))
        parts.append("")

    # Stat bars
    for bar in render_stat_bars(pet.stats, color=rarity_color):
        parts.append(bar)
    parts.append("")

    # Last comment speech bubble
    if pet.last_comment:
        parts.append(Text("  last said", style="dim"))
        comment_lines = _wrap_text(pet.last_comment, width=34)
        max_line_width = max(len(line) for line in comment_lines) if comment_lines else 0
        inner_width = max(max_line_width, 8)
        parts.append(Text(f"  \u256d{'\u2500' * (inner_width + 2)}\u256e", style="dim"))
        for line in comment_lines:
            padded = line.ljust(inner_width)
            parts.append(Text(f"  \u2502 {padded} \u2502", style="dim"))
        parts.append(Text(f"  \u2570{'\u2500' * (inner_width + 2)}\u256f", style="dim"))

    group = Group(*parts)
    return Panel(
        group,
        title=title,
        title_align="center",
        border_style=rarity_color,
        width=42,
        padding=(1, 0),
    )


def _wrap_text(text: str, width: int) -> list[str]:
    """Simple word-wrap for text.

    Args:
        text: Text to wrap.
        width: Maximum line width.

    Returns:
        List of wrapped lines.
    """
    words = text.split()
    lines: list[str] = []
    current_line = ""
    for word in words:
        if current_line and len(current_line) + 1 + len(word) > width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = f"{current_line} {word}" if current_line else word
    if current_line:
        lines.append(current_line)
    return lines
