"""Tests for the PetProfile model."""

from datetime import UTC, datetime

from tpet.models.pet import PetProfile
from tpet.models.rarity import Rarity


class TestPetProfile:
    """Tests for PetProfile model."""

    def test_create_profile(self) -> None:
        pet = PetProfile(
            name="TestPet",
            creature_type="axolotl",
            rarity=Rarity.RARE,
            personality="A snarky axolotl.",
            backstory="Born in a test lab.",
            ascii_art=[
                "  (o.o)  ",
                "  (o_o)  ",
                "  (O.O)  ",
                "  (-_-)  ",
            ],
            stats={"DEBUGGING": 80, "CHAOS": 65},
            accent_color="cyan",
        )
        assert pet.name == "TestPet"
        assert pet.rarity == Rarity.RARE
        assert len(pet.ascii_art) == 4
        assert pet.project_path is None
        assert pet.last_comment is None
        assert pet.comment_history == []

    def test_created_at_default(self) -> None:
        pet = PetProfile(
            name="T",
            creature_type="t",
            rarity=Rarity.COMMON,
            personality="t",
            backstory="t",
            ascii_art=["x"],
            stats={},
            accent_color="white",
        )
        assert isinstance(pet.created_at, datetime)

    def test_comment_history_append(self) -> None:
        pet = PetProfile(
            name="T",
            creature_type="t",
            rarity=Rarity.COMMON,
            personality="t",
            backstory="t",
            ascii_art=["x"],
            stats={},
            accent_color="white",
        )
        pet.comment_history.append("hello")
        pet.last_comment = "hello"
        assert pet.last_comment == "hello"
        assert len(pet.comment_history) == 1

    def test_serialization_roundtrip(self) -> None:
        pet = PetProfile(
            name="TestPet",
            creature_type="axolotl",
            rarity=Rarity.RARE,
            personality="Snarky.",
            backstory="Born in a lab.",
            ascii_art=["(o.o)", "(o_o)"],
            stats={"WIT": 90},
            accent_color="cyan",
            created_at=datetime(2026, 4, 2, tzinfo=UTC),
        )
        data = pet.model_dump(mode="json")
        restored = PetProfile.model_validate(data)
        assert restored.name == pet.name
        assert restored.rarity == pet.rarity
        assert restored.stats == pet.stats
