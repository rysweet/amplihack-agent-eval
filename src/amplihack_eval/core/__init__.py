"""Core evaluation logic: runner and grader."""

from __future__ import annotations

from .grader import GradeResult, grade_answer
from .runner import (
    CategoryBreakdown,
    DimensionScore,
    EvalReport,
    EvalResult,
    EvalRunner,
)

__all__ = [
    "EvalRunner",
    "EvalResult",
    "EvalReport",
    "CategoryBreakdown",
    "DimensionScore",
    "GradeResult",
    "grade_answer",
]
