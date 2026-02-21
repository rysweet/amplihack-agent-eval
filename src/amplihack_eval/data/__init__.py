"""Data generation for evaluation scenarios.

Provides deterministic data generators for long-horizon memory tests
and progressive difficulty level definitions (L1-L12).
"""

from __future__ import annotations

from .long_horizon import (
    GradingRubric,
    GroundTruth,
    Question,
    Turn,
    generate_dialogue,
    generate_questions,
)
from .progressive_levels import (
    ALL_LEVELS,
    ADVANCED_LEVELS,
    NOVEL_SKILL_LEVELS,
    TEACHER_STUDENT_LEVELS,
    TRANSFER_LEVELS,
    TestArticle,
    TestLevel,
    TestQuestion,
    get_level_by_id,
)

__all__ = [
    # long_horizon
    "Turn",
    "Question",
    "GradingRubric",
    "GroundTruth",
    "generate_dialogue",
    "generate_questions",
    # progressive_levels
    "TestArticle",
    "TestQuestion",
    "TestLevel",
    "ALL_LEVELS",
    "TEACHER_STUDENT_LEVELS",
    "ADVANCED_LEVELS",
    "NOVEL_SKILL_LEVELS",
    "TRANSFER_LEVELS",
    "get_level_by_id",
]
