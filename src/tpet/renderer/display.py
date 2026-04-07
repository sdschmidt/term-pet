"""Rich Live display layout for the running pet."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from rich.ansi import AnsiDecoder
from rich.columns import Columns
from rich.containers import Lines
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tpet.art.process import render_halfblock_from_png

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import RenderableType

    from tpet.config import BubblePlacement
    from tpet.models.pet import PetProfile

_decoder = AnsiDecoder()

# Fraction of terminal height devoted to art
_ART_HEIGHT_FRACTION = 0.35


def _compute_art_constraints(art_width_pct: int = 40) -> tuple[int, int]:
    """Compute max terminal rows/cols for art based on current terminal size.

    The art width is a percentage of the terminal width, independent of
    the bubble width.

    Args:
        art_width_pct: Percentage of terminal width allocated to the art (1-100).

    Returns:
        (max_rows, max_cols) for art display.
    """
    try:
        cols, rows = os.get_terminal_size()
    except OSError:
        cols, rows = 80, 24

    max_rows = max(4, int(rows * _ART_HEIGHT_FRACTION))
    # Art panel gets art_width_pct of terminal width (minus panel border overhead)
    max_cols = max(8, int(cols * art_width_pct / 100) - 4)
    return max_rows, max_cols


def _arrange_layout(
    art_panel: RenderableType,
    bubble: RenderableType,
    placement: BubblePlacement,
) -> RenderableType:
    """Arrange art panel and speech bubble according to placement setting.

    Args:
        art_panel: The pet art renderable.
        bubble: The speech bubble renderable.
        placement: Where to position the bubble relative to the art.

    Returns:
        Combined Rich renderable.
    """
    from tpet.config import BubblePlacement

    if placement == BubblePlacement.RIGHT:
        return Columns([art_panel, bubble], padding=(0, 1))

    # Top or bottom: stack vertically using a borderless Table
    grid = Table.grid(padding=0)
    grid.add_column()
    if placement == BubblePlacement.TOP:
        grid.add_row(bubble)
        grid.add_row(art_panel)
    else:
        grid.add_row(art_panel)
        grid.add_row(bubble)
    return grid


def build_display_layout(
    pet: PetProfile,
    frame_index: int,
    comment: str | None,
    placement: BubblePlacement | None = None,
) -> RenderableType:
    """Build the display layout with pet art and speech bubble.

    Args:
        pet: Pet profile.
        frame_index: Current animation frame index.
        comment: Current comment to display, or None.
        placement: Bubble position relative to art. Defaults to BOTTOM.

    Returns:
        Rich renderable for the display.
    """
    from tpet.config import BubblePlacement

    if placement is None:
        placement = BubblePlacement.BOTTOM
    rarity_color = pet.rarity.color

    # Pet art panel
    frame_idx = min(frame_index, len(pet.ascii_art) - 1)
    art = pet.ascii_art[frame_idx] if pet.ascii_art else "(no art)"
    art_text = Text(art, style=pet.accent_color)
    title = Text(f" {pet.name} ", style=f"bold {rarity_color}")

    art_panel = Panel(art_text, title=title, title_align="center", border_style=rarity_color, padding=(1, 2))

    # Speech bubble
    bubble = _build_bubble(pet, comment)

    return _arrange_layout(art_panel, bubble, placement)


def build_halfblock_layout(
    pet: PetProfile,
    png_path: Path,
    comment: str | None,
    max_width_pct: int = 40,
    placement: BubblePlacement | None = None,
) -> RenderableType:
    """Build layout with halfblock ANSI art and speech bubble.

    Loads a raw PNG frame, scales to fit the terminal at runtime, converts
    to halfblock ANSI, and renders alongside the bubble.

    Args:
        pet: Pet profile.
        png_path: Path to the raw PNG frame file.
        comment: Current comment to display, or None.
        max_width_pct: Percentage of terminal width allocated to the art (1-100).
        placement: Bubble position relative to art. Defaults to BOTTOM.

    Returns:
        Rich renderable for the display.
    """
    from tpet.config import BubblePlacement

    if placement is None:
        placement = BubblePlacement.BOTTOM
    rarity_color = pet.rarity.color

    max_rows, max_cols = _compute_art_constraints(max_width_pct)

    # Render PNG to halfblock ANSI at runtime scale
    art_text = render_halfblock_from_png(png_path, max_rows, max_cols)

    # Decode ANSI escape sequences into Rich Text objects (preserves colors)
    art_lines = list(_decoder.decode(art_text))
    title = Text(f" {pet.name} ", style=f"bold {rarity_color}")

    # Measure art content width to constrain the panel
    art_width = max(len(line) for line in art_lines) if art_lines else 10
    art_panel_width = art_width + 2  # +2 for left/right border characters

    art_panel = Panel(
        Lines(art_lines),
        title=title,
        title_align="center",
        border_style=rarity_color,
        padding=(0, 0),
        width=art_panel_width,
    )

    # Speech bubble auto-sizes to content
    bubble = _build_bubble(pet, comment)

    return _arrange_layout(art_panel, bubble, placement)


def build_bubble_only(pet: PetProfile, comment: str | None) -> RenderableType:
    """Build just the speech bubble panel for sixel mode display.

    In sixel mode, the art is rendered directly to the terminal as sixel
    graphics, so Rich only handles the speech bubble portion.

    Args:
        pet: Pet profile.
        comment: Current comment to display, or None.

    Returns:
        Rich renderable containing just the speech bubble.
    """
    return _build_bubble(pet, comment, show_name=True)


def _build_bubble(
    pet: PetProfile,
    comment: str | None,
    show_name: bool = False,
) -> Panel:
    """Build the speech bubble panel.

    Auto-sizes to content width via expand=False.

    Args:
        pet: Pet profile.
        comment: Current comment to display, or None.
        show_name: Include pet name in title.

    Returns:
        Rich Panel containing the speech bubble.
    """
    title_text = f"{pet.name}" if show_name else ""
    content = Text(comment.rstrip()) if comment else Text("...", style="dim italic")
    return Panel(
        content,
        title=title_text,
        title_align="left",
        border_style="dim",
        padding=(0, 1),
        expand=False,
    )
