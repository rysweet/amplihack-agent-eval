"""Level definitions and scoring re-exported for convenience."""

from __future__ import annotations

from ..data.progressive_levels import (
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
from .L13_tool_selection import (
    ToolSelectionScore,
    ToolTrajectory,
    aggregate_scores as aggregate_tool_scores,
    score_scenario as score_tool_scenario,
    score_tool_chain,
    score_tool_efficiency,
    score_tool_selection,
)
from .L14_selective_forgetting import (
    ForgettingResult,
    aggregate_forgetting_scores,
    score_current_value_accuracy,
    score_forgetting_scenario,
    score_stale_data_penalty,
    score_update_awareness,
)
from .L15_adversarial_recall import (
    AdversarialRecallScore,
    aggregate_adversarial_scores,
    score_adversarial_scenario,
    score_confidence_calibration,
    score_fact_boundary_awareness,
    score_hallucination_resistance,
)
from .L16_decision_from_memory import (
    DecisionScore,
    aggregate_decision_scores,
    score_decision_quality,
    score_decision_scenario,
    score_fact_usage,
    score_reasoning_quality,
)

__all__ = [
    # Progressive levels (L1-L12)
    "TestArticle",
    "TestQuestion",
    "TestLevel",
    "ALL_LEVELS",
    "TEACHER_STUDENT_LEVELS",
    "ADVANCED_LEVELS",
    "NOVEL_SKILL_LEVELS",
    "TRANSFER_LEVELS",
    "get_level_by_id",
    # L13: Tool Selection
    "ToolTrajectory",
    "ToolSelectionScore",
    "score_tool_selection",
    "score_tool_efficiency",
    "score_tool_chain",
    "score_tool_scenario",
    "aggregate_tool_scores",
    # L14: Selective Forgetting
    "ForgettingResult",
    "score_current_value_accuracy",
    "score_stale_data_penalty",
    "score_update_awareness",
    "score_forgetting_scenario",
    "aggregate_forgetting_scores",
    # L15: Adversarial Recall
    "AdversarialRecallScore",
    "score_hallucination_resistance",
    "score_fact_boundary_awareness",
    "score_confidence_calibration",
    "score_adversarial_scenario",
    "aggregate_adversarial_scores",
    # L16: Decision From Memory
    "DecisionScore",
    "score_decision_quality",
    "score_reasoning_quality",
    "score_fact_usage",
    "score_decision_scenario",
    "aggregate_decision_scores",
]
