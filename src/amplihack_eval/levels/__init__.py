"""Level definitions re-exported for convenience.

Provides both the original Python-defined levels (progressive_levels),
the new YAML-driven level definitions (schema + loader), and
hive mind scoring for multi-agent shared-memory evaluation.
"""

from __future__ import annotations

from ..data.progressive_levels import (
    ADVANCED_LEVELS,
    ALL_LEVELS,
    NOVEL_SKILL_LEVELS,
    TEACHER_STUDENT_LEVELS,
    TRANSFER_LEVELS,
    TestArticle,
    TestLevel,
    TestQuestion,
    get_level_by_id,
)
from .hive_mind_scoring import (
    HiveMindDimensionScore,
    HiveMindEvalReport,
    HiveMindQuestionResult,
    score_hive_mind_scenario,
    score_single_response,
)
from .loader import load_all_levels, load_level, validate_level
from .schema import LevelDefinition, QuestionTemplate, ScoringConfig

__all__ = [
    # Original Python-defined levels
    "TestArticle",
    "TestQuestion",
    "TestLevel",
    "ALL_LEVELS",
    "TEACHER_STUDENT_LEVELS",
    "ADVANCED_LEVELS",
    "NOVEL_SKILL_LEVELS",
    "TRANSFER_LEVELS",
    "get_level_by_id",
    # YAML-driven schema
    "LevelDefinition",
    "QuestionTemplate",
    "ScoringConfig",
    # YAML loader
    "load_level",
    "load_all_levels",
    "validate_level",
    # Hive mind scoring
    "HiveMindDimensionScore",
    "HiveMindEvalReport",
    "HiveMindQuestionResult",
    "score_hive_mind_scenario",
    "score_single_response",
]
