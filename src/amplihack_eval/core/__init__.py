"""Core evaluation logic: runner and grader."""

from __future__ import annotations

from .grader import GradeResult, grade_answer
from .runner import (
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
