"""YAML profile persistence."""

import logging
from pathlib import Path

from tpet.io import load_yaml, save_yaml
from tpet.models.pet import PetProfile

logger = logging.getLogger(__name__)


def save_profile(pet: PetProfile, path: Path) -> None:
    """Save a pet profile to YAML.

    Args:
        pet: Pet profile to save.
        path: File path to write to.
    """
    save_yaml(pet, path)
    logger.info("Saved profile to %s", path)


def load_profile(path: Path) -> PetProfile | None:
    """Load a pet profile from YAML.

    Args:
        path: File path to read from.

    Returns:
        Loaded profile or None if file doesn't exist or is invalid.
    """
    return load_yaml(path, PetProfile)


def get_profile_path(config_dir: Path, project_path: str | None = None) -> Path:
    """Get the profile path for global or project pet.

    Args:
        config_dir: Global config directory.
        project_path: Optional project directory for project-specific pet.

    Returns:
        Path to the profile YAML file.
    """
    if project_path:
        return Path(project_path) / ".tpet" / "profile.yaml"
    return config_dir / "profile.yaml"


def resolve_profile(config_dir: Path, project_path: str | None = None) -> tuple[Path, PetProfile | None]:
    """Resolve the active pet profile with global fallback.

    Tries the project-specific profile first.  If the project has no pet,
    falls back to the global profile.

    Args:
        config_dir: Global config directory (e.g. ``~/.config/tpet``).
        project_path: Optional project directory for project-specific lookup.

    Returns:
        Tuple of ``(profile_path, pet_or_none)``.
    """
    # Try project-specific pet first
    if project_path:
        project_profile = Path(project_path) / ".tpet" / "profile.yaml"
        pet = load_profile(project_profile)
        if pet is not None:
            logger.info("Using project pet from %s", project_profile)
            return project_profile, pet

    # Fall back to global pet
    global_profile = config_dir / "profile.yaml"
    return global_profile, load_profile(global_profile)
