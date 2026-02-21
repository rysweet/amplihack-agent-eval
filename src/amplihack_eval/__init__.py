"""amplihack-agent-eval: Evaluation framework for goal-seeking AI agents.

Provides memory recall, tool use, planning, and reasoning evaluation
across progressive difficulty levels (L1-L12).

Public API:
    EvalRunner: Main evaluation runner
    AgentAdapter: Interface to make any agent evaluable
    AgentResponse: Response from an agent including trajectory
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
    "GradeResult",
    "grade_answer",
]
