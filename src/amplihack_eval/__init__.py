"""amplihack-agent-eval: Evaluation framework for goal-seeking AI agents.

Provides memory recall, tool use, planning, and reasoning evaluation
across progressive difficulty levels (L1-L12).

Public API:
    EvalRunner: Main evaluation runner
    AgentAdapter: Interface to make any agent evaluable
    AgentResponse: Response from an agent including trajectory
    LevelResult: Result of running a single YAML-defined level
    SuiteResult: Result of running a suite of YAML levels
    run_level: Run a single YAML level against an agent
    run_suite: Run multiple YAML levels with prerequisite checking
"""

from __future__ import annotations

__version__ = "0.1.0"

from .adapters.base import AgentAdapter, AgentResponse, ToolCall
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
]
