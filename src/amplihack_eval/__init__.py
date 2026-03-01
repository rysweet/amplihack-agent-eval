"""amplihack-agent-eval: Evaluation framework for goal-seeking AI agents.

Provides memory recall, tool use, planning, and reasoning evaluation
across progressive difficulty levels (L1-L16), plus hive mind
multi-agent evaluation scenarios.

Levels:
    L1-L6: Core memory recall (single source through incremental learning)
    L7: Teacher-student knowledge transfer
    L8-L10: Advanced reasoning (metacognition, causal, counterfactual)
    L11: Novel skill acquisition
    L12: Far transfer
    L13: Tool use selection
    L14: Selective forgetting
    L15: Adversarial recall (hallucination resistance)
    L16: Decision from memory

Hive Mind:
    Multi-agent shared-memory evaluation with 5 scenario types

Public API:
    EvalRunner: Main evaluation runner
    AgentAdapter: Interface to make any agent evaluable
    AgentResponse: Response from an agent including trajectory
    LevelResult: Result of running a single YAML-defined level
    SuiteResult: Result of running a suite of YAML levels
    run_level: Run a single YAML level against an agent
    run_suite: Run multiple YAML levels with prerequisite checking
    HiveMindGroupAdapter: Multi-agent shared-memory adapter
    HiveMindScenario: Hive mind evaluation scenario data
    score_hive_mind_scenario: Score a hive mind evaluation
"""

from __future__ import annotations

__version__ = "0.1.0"

from .adapters.base import AgentAdapter, AgentResponse, ToolCall
from .adapters.hive_mind_adapter import HiveMindGroupAdapter, InMemorySharedStore
from .core.grader import GradeResult, grade_answer
from .core.runner import (
    CategoryBreakdown,
    DimensionScore,
    EvalReport,
    EvalResult,
    EvalRunner,
    LevelResult,
    SuiteResult,
    run_level,
    run_suite,
)
from .data.hive_mind_scenarios import (
    ALL_HIVE_MIND_SCENARIOS,
    HiveMindQuestion,
    HiveMindScenario,
)
from .levels.hive_mind_scoring import (
    HiveMindEvalReport,
    score_hive_mind_scenario,
)

__all__ = [
    # Adapters
    "AgentAdapter",
    "AgentResponse",
    "ToolCall",
    # Core
    "EvalRunner",
    "EvalResult",
    "EvalReport",
    "CategoryBreakdown",
    "DimensionScore",
    "LevelResult",
    "SuiteResult",
    "GradeResult",
    "grade_answer",
    "run_level",
    "run_suite",
    # Hive Mind
    "HiveMindGroupAdapter",
    "InMemorySharedStore",
    "HiveMindQuestion",
    "HiveMindScenario",
    "ALL_HIVE_MIND_SCENARIOS",
    "HiveMindEvalReport",
    "score_hive_mind_scenario",
]
