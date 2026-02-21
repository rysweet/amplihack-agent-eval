"""YAML level loader.

Discovers and parses YAML level definitions from the levels/ directory.
Each YAML file defines one evaluation level with questions, scoring, and metadata.

Public API:
    load_level(level_id) -> LevelDefinition
    load_all_levels() -> list[LevelDefinition]
    validate_level(level) -> list[str]
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .schema import LevelDefinition, QuestionTemplate, ScoringConfig

logger = logging.getLogger(__name__)

# Directory containing YAML level files (same directory as this module)
_LEVELS_DIR = Path(__file__).parent


def _parse_scoring(raw: dict) -> ScoringConfig:
    """Parse a scoring config dict from YAML into a ScoringConfig."""
    return ScoringConfig(
        pass_threshold=float(raw.get("pass_threshold", 0.7)),
        dimensions=list(raw.get("dimensions", ["factual_accuracy"])),
        weights=dict(raw.get("weights", {})),
        grader_votes=int(raw.get("grader_votes", 3)),
    )


def _parse_question(raw: dict) -> QuestionTemplate:
    """Parse a question dict from YAML into a QuestionTemplate."""
    return QuestionTemplate(
        id=str(raw["id"]),
        text=str(raw["text"]),
        category=str(raw.get("category", "general")),
        scoring_dimensions=list(raw.get("scoring_dimensions", ["factual_accuracy"])),
        expected_answer=raw.get("expected_answer"),
        rubric=raw.get("rubric"),
    )


def _parse_level(data: dict) -> LevelDefinition:
    """Parse a full YAML dict into a LevelDefinition."""
    scoring = _parse_scoring(data.get("scoring", {}))
    questions = [_parse_question(q) for q in data.get("questions", [])]

    return LevelDefinition(
        id=str(data["id"]),
        name=str(data["name"]),
        description=str(data["description"]),
        category=str(data.get("category", "memory")),
        difficulty=int(data.get("difficulty", 1)),
        questions=questions,
        scoring=scoring,
        prerequisites=list(data.get("prerequisites", [])),
        data_source=str(data.get("data_source", "progressive_levels")),
        min_turns=int(data.get("min_turns", 50)),
        grading_mode=str(data.get("grading_mode", "hybrid")),
    )


def load_level(level_id: str, levels_dir: Path | None = None) -> LevelDefinition:
    """Load a single level definition from its YAML file.

    Args:
        level_id: Level identifier (e.g. "L01"). The loader searches for
            files matching ``*{level_id}*.yaml`` in the levels directory.
        levels_dir: Override directory to search. Defaults to the package
            levels/ directory.

    Returns:
        Parsed LevelDefinition.

    Raises:
        FileNotFoundError: If no YAML file matches the level_id.
        ValueError: If the YAML cannot be parsed into a valid level.
    """
    search_dir = levels_dir or _LEVELS_DIR
    # Find YAML file matching level_id (e.g. L01_simple_recall.yaml)
    candidates = list(search_dir.glob(f"*{level_id}*.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No YAML file found for level '{level_id}' in {search_dir}"
        )
    yaml_path = candidates[0]
    logger.debug("Loading level %s from %s", level_id, yaml_path)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict in {yaml_path}, got {type(data).__name__}")

    level = _parse_level(data)

    # Sanity check: parsed ID should match requested ID
    if level.id != level_id:
        logger.warning(
            "Requested level %s but YAML contains id=%s in %s",
            level_id, level.id, yaml_path,
        )

    return level


def load_all_levels(levels_dir: Path | None = None) -> list[LevelDefinition]:
    """Load all YAML level definitions from the levels directory.

    Args:
        levels_dir: Override directory. Defaults to the package levels/ directory.

    Returns:
        List of LevelDefinition objects sorted by ID.
    """
    search_dir = levels_dir or _LEVELS_DIR
    yaml_files = sorted(search_dir.glob("L*.yaml"))
    levels: list[LevelDefinition] = []

    for yaml_path in yaml_files:
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                levels.append(_parse_level(data))
            else:
                logger.warning("Skipping %s: not a YAML dict", yaml_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", yaml_path, e)

    return sorted(levels, key=lambda lv: lv.id)


def validate_level(level: LevelDefinition) -> list[str]:
    """Validate a level definition and return a list of errors.

    Args:
        level: LevelDefinition to validate.

    Returns:
        List of error strings. Empty list means valid.
    """
    return level.validate()


__all__ = [
    "load_level",
    "load_all_levels",
    "validate_level",
]
