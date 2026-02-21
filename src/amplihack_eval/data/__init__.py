"""Data generation for evaluation scenarios.

Provides deterministic data generators for long-horizon memory tests,
progressive difficulty level definitions (L1-L12), and new scenario
data for L13-L16 (tool use, forgetting, adversarial, decision).
"""

from __future__ import annotations

from .adversarial_scenarios import (
    ALL_ADVERSARIAL_SCENARIOS,
    COMMON_KB,
    AdversarialScenario,
    KnowledgeBaseFact,
    get_adversarial_scenario_by_id,
    get_adversarial_scenarios_by_category,
    get_adversarial_scenarios_by_difficulty,
)
from .decision_scenarios import (
    ALL_DECISION_SCENARIOS,
    ContextFact,
    DecisionScenario,
    get_decision_scenario_by_id,
    get_decision_scenarios_by_difficulty,
    get_decision_scenarios_by_domain,
)
from .forgetting_scenarios import (
    ALL_FORGETTING_SCENARIOS,
    FactUpdate,
    ForgettingScenario,
    get_forgetting_scenario_by_id,
    get_forgetting_scenarios_by_domain,
)
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
from .tool_use_scenarios import (
    ALL_TOOL_USE_SCENARIOS,
    ALL_TOOLS,
    ToolDefinition,
    ToolUseScenario,
    get_scenario_by_id,
    get_scenarios_by_difficulty,
    get_scenarios_by_domain,
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
    # tool_use_scenarios (L13)
    "ToolDefinition",
    "ToolUseScenario",
    "ALL_TOOLS",
    "ALL_TOOL_USE_SCENARIOS",
    "get_scenario_by_id",
    "get_scenarios_by_domain",
    "get_scenarios_by_difficulty",
    # forgetting_scenarios (L14)
    "FactUpdate",
    "ForgettingScenario",
    "ALL_FORGETTING_SCENARIOS",
    "get_forgetting_scenario_by_id",
    "get_forgetting_scenarios_by_domain",
    # adversarial_scenarios (L15)
    "KnowledgeBaseFact",
    "AdversarialScenario",
    "COMMON_KB",
    "ALL_ADVERSARIAL_SCENARIOS",
    "get_adversarial_scenario_by_id",
    "get_adversarial_scenarios_by_category",
    "get_adversarial_scenarios_by_difficulty",
    # decision_scenarios (L16)
    "ContextFact",
    "DecisionScenario",
    "ALL_DECISION_SCENARIOS",
    "get_decision_scenario_by_id",
    "get_decision_scenarios_by_domain",
    "get_decision_scenarios_by_difficulty",
]
