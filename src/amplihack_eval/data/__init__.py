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
from .hive_mind_scenarios import (
    ALL_HIVE_MIND_SCENARIOS,
    HiveMindQuestion,
    HiveMindScenario,
)
from .hive_mind_scenarios import (
    SCENARIO_ADVERSARIAL as HIVE_SCENARIO_ADVERSARIAL,
)
from .hive_mind_scenarios import (
    SCENARIO_ARCH as HIVE_SCENARIO_ARCH,
)
from .hive_mind_scenarios import (
    SCENARIO_INCIDENT as HIVE_SCENARIO_INCIDENT,
)
from .hive_mind_scenarios import (
    SCENARIO_INFRA as HIVE_SCENARIO_INFRA,
)
from .hive_mind_scenarios import (
    SCENARIO_RESEARCH as HIVE_SCENARIO_RESEARCH,
)
from .hive_mind_scenarios import (
    get_questions_by_difficulty as get_hive_questions_by_difficulty,
)
from .hive_mind_scenarios import (
    get_scenario_by_id as get_hive_scenario_by_id,
)
from .hive_mind_scenarios import (
    get_scenarios_by_difficulty as get_hive_scenarios_by_difficulty,
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
from .security_analyst_scenario import (
    LEVEL_DISTRIBUTION as SECURITY_LEVEL_DISTRIBUTION,
)
from .security_analyst_scenario import (
    LEVEL_NAMES as SECURITY_LEVEL_NAMES,
)
from .security_analyst_scenario import (
    generate_dialogue as generate_security_dialogue,
)
from .security_analyst_scenario import (
    generate_questions as generate_security_questions,
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
    # hive_mind_scenarios
    "HiveMindQuestion",
    "HiveMindScenario",
    "ALL_HIVE_MIND_SCENARIOS",
    "HIVE_SCENARIO_INFRA",
    "HIVE_SCENARIO_ARCH",
    "HIVE_SCENARIO_INCIDENT",
    "HIVE_SCENARIO_RESEARCH",
    "HIVE_SCENARIO_ADVERSARIAL",
    "get_hive_scenario_by_id",
    "get_hive_scenarios_by_difficulty",
    "get_hive_questions_by_difficulty",
    # security_analyst_scenario
    "SECURITY_LEVEL_NAMES",
    "SECURITY_LEVEL_DISTRIBUTION",
    "generate_security_dialogue",
    "generate_security_questions",
]
