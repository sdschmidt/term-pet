"""Tests for the live display renderer."""

from rich.console import Console

from tpet.models.pet import PetProfile
from tpet.models.rarity import Rarity
from tpet.renderer.display import build_bubble_only, build_display_layout


def _make_pet() -> PetProfile:
    return PetProfile(
        name="Knurling",
        creature_type="axolotl",
        rarity=Rarity.RARE,
        personality="Methodical axolotl.",
        backstory="Born debugging.",
        ascii_art=[
            "  (o.o)  \n  /| |\\  \n   | |   ",
            "  (o_o)  \n  /| |\\  \n   | |   ",
            "  (O.O)  \n  /| |\\  \n   | |   ",
            "  (-_-)  \n  /| |\\  \n   | |   ",
        ],
        stats={"DEBUGGING": 95},
        accent_color="cyan",
    )


class TestBuildDisplayLayout:
    def test_renders_without_error(self) -> None:
        pet = _make_pet()
        layout = build_display_layout(pet, frame_index=0, comment="Hello world")
        console = Console(file=None, force_terminal=True, width=60)
        with console.capture() as capture:
            console.print(layout)
        output = capture.get()
        assert len(output) > 0

    def test_contains_pet_art(self) -> None:
        pet = _make_pet()
        layout = build_display_layout(pet, frame_index=0, comment=None)
        console = Console(file=None, force_terminal=True, width=60)
        with console.capture() as capture:
            console.print(layout)
        assert "(o.o)" in capture.get()

    def test_contains_comment_bubble(self) -> None:
        pet = _make_pet()
        layout = build_display_layout(pet, frame_index=0, comment="Fix the bug!")
        console = Console(file=None, force_terminal=True, width=60)
        with console.capture() as capture:
            console.print(layout)
        assert "Fix the bug!" in capture.get()

    def test_different_frames(self) -> None:
        pet = _make_pet()
        console = Console(file=None, force_terminal=True, width=60)
        with console.capture() as c0:
            console.print(build_display_layout(pet, frame_index=0, comment=None))
        with console.capture() as c3:
            console.print(build_display_layout(pet, frame_index=3, comment=None))
        assert "(o.o)" in c0.get()
        assert "(-_-)" in c3.get()

    def test_no_comment_shows_name(self) -> None:
        pet = _make_pet()
        layout = build_display_layout(pet, frame_index=0, comment=None)
        console = Console(file=None, force_terminal=True, width=60)
        with console.capture() as capture:
            console.print(layout)
        assert "Knurling" in capture.get()


class TestBuildBubbleOnly:
    def test_bubble_renders_comment(self) -> None:
        pet = _make_pet()
        bubble = build_bubble_only(pet, comment="Hello from sixel!")
        console = Console(file=None, force_terminal=True, width=60)
        with console.capture() as capture:
            console.print(bubble)
        assert "Hello from sixel!" in capture.get()

    def test_bubble_shows_pet_name(self) -> None:
        pet = _make_pet()
        bubble = build_bubble_only(pet, comment="Test")
        console = Console(file=None, force_terminal=True, width=60)
        with console.capture() as capture:
            console.print(bubble)
        assert "Knurling" in capture.get()

    def test_bubble_no_comment_shows_ellipsis(self) -> None:
        pet = _make_pet()
        bubble = build_bubble_only(pet, comment=None)
        console = Console(file=None, force_terminal=True, width=60)
        with console.capture() as capture:
            console.print(bubble)
        assert "..." in capture.get()

    def test_bubble_renders_without_error(self) -> None:
        pet = _make_pet()
        bubble = build_bubble_only(pet, comment=None)
        console = Console(file=None, force_terminal=True, width=60)
        with console.capture() as capture:
            console.print(bubble)
        assert len(capture.get()) > 0
