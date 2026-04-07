"""Shared I/O utilities for YAML persistence of Pydantic models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def save_yaml(model: BaseModel, path: Path, *, exclude: set[str] | None = None) -> None:
    """Save a Pydantic model to a YAML file.

    Creates parent directories as needed.

    Args:
        model: Pydantic model instance to serialize.
        path: File path to write to.
        exclude: Optional set of field names to exclude from serialization.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.model_dump(mode="json", exclude=exclude or set())
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")


def load_yaml[T: BaseModel](path: Path, model_type: type[T]) -> T | None:
    """Load a Pydantic model from a YAML file with error handling.

    Returns None (instead of raising) when the file is missing, malformed,
    or fails validation — callers can fall back to defaults.

    Args:
        path: File path to read from.
        model_type: Pydantic model class to validate against.

    Returns:
        Parsed model instance, or None on any failure.
    """
    if not path.exists():
        logger.debug("File not found: %s", path)
        return None

    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if data is None:
            return None
        return model_type.model_validate(data)
    except yaml.YAMLError:
        logger.exception("Malformed YAML in %s", path)
        return None
    except ValidationError:
        logger.exception("Validation failed for %s", path)
        return None
    except OSError:
        logger.exception("Could not read %s", path)
        return None
