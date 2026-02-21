"""Specialized grading agent with a specific perspective.

Each GraderAgent evaluates answers from one of three perspectives:
- factual: Is the answer factually correct?
- reasoning: Is the reasoning sound and well-structured?
- completeness: Does the answer cover all aspects of the question?

Multiple GraderAgents produce independent grades that are aggregated
via multi-vote (median) to reduce grading noise.

Philosophy:
- Hybrid grading: deterministic keyword matching first, LLM for judgment
- Each perspective uses a tailored system prompt
- Aggregation via median score is robust to outlier grades
- JSON-serializable for logging

Public API:
    PerspectiveGrade: Grade from a single perspective
    AggregateGrade: Multi-vote aggregation across grader agents
    GraderAgent: Specialized grading agent with a specific perspective
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from ..data.long_horizon import GradingRubric, Question

logger = logging.getLogger(__name__)


PERSPECTIVES = ("factual", "reasoning", "completeness")

_PERSPECTIVE_PROMPTS: dict[str, str] = {
    "factual": (
        "You are a FACTUAL ACCURACY grader. Your sole focus is whether the answer "
        "contains correct facts. Ignore style, completeness, and reasoning quality. "
        "Grade ONLY on whether stated facts match the expected answer.\n\n"
        "Scoring:\n"
        "- 1.0: All facts correct\n"
        "- 0.8-0.9: Minor factual imprecision (e.g., rounding)\n"
        "- 0.5-0.7: Some correct facts mixed with errors\n"
        "- 0.2-0.4: Mostly incorrect\n"
        "- 0.0-0.1: Completely wrong or fabricated"
    ),
    "reasoning": (
        "You are a REASONING QUALITY grader. Your sole focus is whether the agent's "
        "reasoning process is sound. Ignore minor factual errors if the reasoning "
        "approach is correct. Grade on logical structure, appropriate use of evidence, "
        "and sound inference.\n\n"
        "Scoring:\n"
        "- 1.0: Flawless reasoning chain\n"
        "- 0.8-0.9: Sound reasoning with minor gaps\n"
        "- 0.5-0.7: Partially sound, some logical leaps\n"
        "- 0.2-0.4: Weak reasoning, major gaps\n"
        "- 0.0-0.1: No reasoning or completely flawed logic"
    ),
    "completeness": (
        "You are a COMPLETENESS grader. Your sole focus is whether the answer covers "
        "all aspects of the question. Ignore minor factual errors if all topics are "
        "addressed. Grade on coverage, detail, and whether key points are missing.\n\n"
        "Scoring:\n"
        "- 1.0: All aspects of the question addressed thoroughly\n"
        "- 0.8-0.9: Most aspects covered, minor omissions\n"
        "- 0.5-0.7: Partial coverage, some key points missing\n"
        "- 0.2-0.4: Major aspects of the question not addressed\n"
        "- 0.0-0.1: Answer does not address the question"
    ),
}


@dataclass
class PerspectiveGrade:
    """Grade from a single grader perspective."""

    perspective: str
    score: float  # 0.0 to 1.0
    reasoning: str
    question_id: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "perspective": self.perspective,
            "score": round(self.score, 4),
            "reasoning": self.reasoning,
            "question_id": self.question_id,
        }


@dataclass
class AggregateGrade:
    """Multi-vote aggregation across grader agents."""

    question_id: str
    question_text: str
    expected_answer: str
    actual_answer: str
    perspective_grades: list[PerspectiveGrade]
    overall_score: float  # Median of perspective scores
    agreement: float  # 1 - stddev of perspective scores (higher = more agreement)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text[:200],
            "expected_answer": self.expected_answer[:200],
            "actual_answer": self.actual_answer[:200],
            "perspective_grades": [g.to_dict() for g in self.perspective_grades],
            "overall_score": round(self.overall_score, 4),
            "agreement": round(self.agreement, 4),
        }


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM response text."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _deterministic_grade(rubric: GradingRubric, actual_answer: str) -> float | None:
    """Quick deterministic grade using rubric keywords.

    Returns a score if the rubric has enough information for deterministic
    grading, or None if LLM grading is needed.
    """
    if not rubric.required_keywords and not rubric.incorrect_patterns:
        return None

    answer_lower = actual_answer.lower()

    # Instant 0 for incorrect patterns
    if rubric.incorrect_patterns:
        for pat in rubric.incorrect_patterns:
            if re.search(re.escape(pat.lower()), answer_lower):
                return 0.0

    # Keyword matching
    if rubric.required_keywords:
        matched = sum(
            1 for kw in rubric.required_keywords
            if re.search(re.escape(kw.lower()), answer_lower)
        )
        ratio = matched / len(rubric.required_keywords)
    else:
        ratio = 0.5

    # Paraphrase bonus
    if rubric.acceptable_paraphrases:
        hits = sum(
            1 for p in rubric.acceptable_paraphrases
            if re.search(re.escape(p.lower()), answer_lower)
        )
        ratio = min(1.0, ratio + hits * 0.1)

    return round(ratio, 4)


class GraderAgent:
    """Specialized grading agent with a specific perspective.

    Each instance grades from one perspective (factual, reasoning, or
    completeness). Multiple instances are used in parallel to produce
    multi-vote grades.

    Args:
        perspective: One of "factual", "reasoning", "completeness"
        model: LLM model identifier (default: from GRADER_MODEL env var)

    Example::

        grader = GraderAgent("factual")
        grade = grader.grade(question, answer="Paris is the capital.")
        print(f"Score: {grade.score}, Reasoning: {grade.reasoning}")
    """

    PERSPECTIVES = PERSPECTIVES

    def __init__(self, perspective: str, model: str = ""):
        if perspective not in PERSPECTIVES:
            raise ValueError(
                f"Invalid perspective '{perspective}'. Must be one of: {PERSPECTIVES}"
            )
        self.perspective = perspective
        self.model = model or os.environ.get("GRADER_MODEL", "claude-sonnet-4-5-20250929")
        self._system_prompt = _PERSPECTIVE_PROMPTS[perspective]

    def grade(
        self,
        question: Question,
        answer: str,
        rubric: GradingRubric | None = None,
    ) -> PerspectiveGrade:
        """Grade an answer from this agent's perspective.

        Uses hybrid grading: deterministic first (if rubric available and
        perspective is factual), LLM for judgment dimensions.

        Args:
            question: The question with expected answer
            answer: Agent's actual answer
            rubric: Optional grading rubric for deterministic scoring

        Returns:
            PerspectiveGrade with score and reasoning
        """
        if not answer or not answer.strip():
            return PerspectiveGrade(
                perspective=self.perspective,
                score=0.0,
                reasoning="No answer provided",
                question_id=question.question_id,
            )

        # Try deterministic grading for factual perspective
        effective_rubric = rubric or question.rubric
        if self.perspective == "factual" and effective_rubric:
            det_score = _deterministic_grade(effective_rubric, answer)
            if det_score is not None:
                return PerspectiveGrade(
                    perspective=self.perspective,
                    score=det_score,
                    reasoning=f"Deterministic grade from rubric keywords",
                    question_id=question.question_id,
                )

        # LLM grading
        return self._grade_with_llm(question, answer)

    def _grade_with_llm(self, question: Question, answer: str) -> PerspectiveGrade:
        """Grade using LLM with this agent's perspective-specific prompt."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return PerspectiveGrade(
                perspective=self.perspective,
                score=0.0,
                reasoning="No ANTHROPIC_API_KEY available",
                question_id=question.question_id,
            )

        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(api_key=api_key)

        prompt = (
            f"Question: {question.text}\n\n"
            f"Expected Answer: {question.expected_answer}\n\n"
            f"Agent's Answer: {answer}\n\n"
            f"Category: {question.category}\n\n"
            f"Grade the agent's answer from YOUR perspective (0.0 to 1.0).\n"
            f'Return ONLY JSON: {{"score": 0.85, "reasoning": "Brief explanation"}}'
        )

        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=500,
                system=self._system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = message.content[0].text
            result = _extract_json(raw)

            return PerspectiveGrade(
                perspective=self.perspective,
                score=float(result.get("score", 0.0)),
                reasoning=str(result.get("reasoning", "")),
                question_id=question.question_id,
                raw_response=raw,
            )

        except Exception as e:
            logger.warning(
                "GraderAgent(%s) failed for %s: %s",
                self.perspective,
                question.question_id,
                e,
            )
            return PerspectiveGrade(
                perspective=self.perspective,
                score=0.0,
                reasoning=f"Grading error: {e}",
                question_id=question.question_id,
            )

    @staticmethod
    def aggregate_grades(
        grades: list[PerspectiveGrade],
        question: Question,
        actual_answer: str,
    ) -> AggregateGrade:
        """Multi-vote aggregation across grader agents.

        Takes the median score across all perspective grades. Agreement is
        computed as 1 - stddev (higher = more inter-grader agreement).

        Args:
            grades: List of PerspectiveGrade from different grader agents
            question: The question being graded
            actual_answer: The agent's answer

        Returns:
            AggregateGrade with overall score and agreement metric
        """
        if not grades:
            return AggregateGrade(
                question_id=question.question_id,
                question_text=question.text,
                expected_answer=question.expected_answer,
                actual_answer=actual_answer,
                perspective_grades=[],
                overall_score=0.0,
                agreement=0.0,
            )

        scores = [g.score for g in grades]
        median_score = statistics.median(scores)

        if len(scores) >= 2:
            stddev = statistics.stdev(scores)
        else:
            stddev = 0.0
        agreement = max(0.0, 1.0 - stddev)

        return AggregateGrade(
            question_id=question.question_id,
            question_text=question.text,
            expected_answer=question.expected_answer,
            actual_answer=actual_answer,
            perspective_grades=grades,
            overall_score=median_score,
            agreement=agreement,
        )


__all__ = [
    "GraderAgent",
    "PerspectiveGrade",
    "AggregateGrade",
    "PERSPECTIVES",
]
