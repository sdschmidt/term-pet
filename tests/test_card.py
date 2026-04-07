"""Tests for the card renderer."""

from datetime import UTC, datetime

from rich.console import Console

from tpet.models.pet import PetProfile
from tpet.models.rarity import Rarity
from tpet.renderer.card import render_card


def _make_pet() -> PetProfile:
    return PetProfile(
        name="Knurling",
        creature_type="axolotl",
        rarity=Rarity.RARE,
        personality="An obsessively methodical axolotl who traces bugs with supernatural focus.",
        backstory="Born in a debugging dimension.",
        ascii_art=[
            "  }~(______)~{\n  }~(\u25c9 .. \u25c9)~{\n    ( .--. )\n    (_/  \\_)",
            "  }~(______)~{\n  }~(\u25c9 .. \u25c9)~{\n    ( .--. )\n    (_/  \\_)",
        ],
        stats={"DEBUGGING": 95, "PATIENCE": 41, "CHAOS": 62, "WISDOM": 40, "SNARK": 15},
        accent_color="cyan",
        created_at=datetime(2026, 4, 2, tzinfo=UTC),
        last_comment="Finder launches naked. Defaulting to en_US.UTF-8 saves the day.",
    )


class TestRenderCard:
    """Tests for card rendering."""

    def test_renders_without_error(self) -> None:
        pet = _make_pet()
        panel = render_card(pet)
        console = Console(file=None, force_terminal=True, width=50)
        with console.capture() as capture:
            console.print(panel)
        output = capture.get()
        assert len(output) > 0

    def test_contains_pet_name(self) -> None:
        pet = _make_pet()
        panel = render_card(pet)
        console = Console(file=None, force_terminal=True, width=50)
        with console.capture() as capture:
            console.print(panel)
        assert "Knurling" in capture.get()

    def test_contains_rarity(self) -> None:
        pet = _make_pet()
        panel = render_card(pet)
        console = Console(file=None, force_terminal=True, width=50)
        with console.capture() as capture:
            console.print(panel)
        assert "RARE" in capture.get()

    def test_contains_creature_type(self) -> None:
        pet = _make_pet()
        panel = render_card(pet)
        console = Console(file=None, force_terminal=True, width=50)
        with console.capture() as capture:
            console.print(panel)
        assert "AXOLOTL" in capture.get().upper()

    def test_contains_stats(self) -> None:
        pet = _make_pet()
        panel = render_card(pet)
        console = Console(file=None, force_terminal=True, width=50)
        with console.capture() as capture:
            console.print(panel)
        output = capture.get()
        assert "DEBUGGING" in output
        assert "95" in output

    def test_contains_last_comment(self) -> None:
        pet = _make_pet()
        panel = render_card(pet)
        console = Console(file=None, force_terminal=True, width=50)
        with console.capture() as capture:
            console.print(panel)
        assert "Finder launches naked" in capture.get()

    def test_no_last_comment(self) -> None:
        pet = _make_pet()
        pet.last_comment = None
        panel = render_card(pet)
        console = Console(file=None, force_terminal=True, width=50)
        with console.capture() as capture:
            console.print(panel)
        assert "Knurling" in capture.get()
