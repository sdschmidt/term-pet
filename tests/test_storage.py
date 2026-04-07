"""Tests for profile storage."""

from datetime import UTC, datetime
from pathlib import Path

from tpet.models.pet import PetProfile
from tpet.models.rarity import Rarity
from tpet.profile.storage import get_profile_path, load_profile, save_profile


def _make_pet() -> PetProfile:
    return PetProfile(
        name="TestPet",
        creature_type="axolotl",
        rarity=Rarity.RARE,
        personality="Snarky axolotl.",
        backstory="Born in a test lab.",
        ascii_art=["(o.o)", "(o_o)", "(O.O)", "(-_-)"],
        stats={"DEBUGGING": 80, "CHAOS": 65, "WISDOM": 72},
        accent_color="cyan",
        created_at=datetime(2026, 4, 2, 12, 0, 0, tzinfo=UTC),
    )


class TestProfileStorage:
    """Tests for profile save/load."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "profile.yaml"
        save_profile(_make_pet(), path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "TestPet" in content
        assert "axolotl" in content

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "profile.yaml"
        original = _make_pet()
        save_profile(original, path)
        loaded = load_profile(path)
        assert loaded is not None
        assert loaded.name == original.name
        assert loaded.rarity == original.rarity
        assert loaded.stats == original.stats
        assert loaded.ascii_art == original.ascii_art

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.yaml"
        assert load_profile(path) is None

    def test_project_profile_path(self, tmp_project_dir: Path) -> None:
        path = tmp_project_dir / ".tpet" / "profile.yaml"
        pet = _make_pet()
        pet.project_path = str(tmp_project_dir)
        save_profile(pet, path)
        loaded = load_profile(path)
        assert loaded is not None
        assert loaded.project_path == str(tmp_project_dir)


class TestGetProfilePath:
    """Tests for get_profile_path."""

    def test_global_path(self, tmp_config_dir: Path) -> None:
        path = get_profile_path(tmp_config_dir)
        assert path == tmp_config_dir / "profile.yaml"

    def test_project_path(self, tmp_config_dir: Path, tmp_project_dir: Path) -> None:
        path = get_profile_path(tmp_config_dir, str(tmp_project_dir))
        assert path == tmp_project_dir / ".tpet" / "profile.yaml"
