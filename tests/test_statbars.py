"""Tests for stat bar rendering."""

from rich.console import Console

from tpet.renderer.statbars import render_stat_bars


class TestRenderStatBars:
    """Tests for stat bar rendering."""

    def test_renders_correct_count(self) -> None:
        stats = {"DEBUG": 80, "CHAOS": 50}
        renderables = render_stat_bars(stats, color="cyan")
        assert len(renderables) == 2

    def test_stat_text_contains_name_and_value(self) -> None:
        stats = {"WISDOM": 42}
        renderables = render_stat_bars(stats, color="yellow")
        console = Console(file=None, force_terminal=True, width=80)
        with console.capture() as capture:
            for r in renderables:
                console.print(r)
        output = capture.get()
        assert "WISDOM" in output
        assert "42" in output

    def test_empty_stats(self) -> None:
        renderables = render_stat_bars({}, color="dim")
        assert len(renderables) == 0

    def test_zero_value(self) -> None:
        stats = {"LUCK": 0}
        renderables = render_stat_bars(stats, color="green")
        assert len(renderables) == 1

    def test_max_value(self) -> None:
        stats = {"POWER": 99}
        renderables = render_stat_bars(stats, color="bright_magenta")
        assert len(renderables) == 1
