"""Scoring module for hive mind evaluation scenarios.

Scores hive mind performance across five dimensions:
- cross_domain_accuracy: Can agents answer questions requiring other agents' knowledge?
- knowledge_coverage: What % of the total knowledge space does each agent access?
- collaboration_efficiency: How many propagation rounds needed for adequate coverage?
- adversarial_resilience: Does the hive amplify or suppress bad data?
- no_regression: Do agents maintain accuracy on their own domain questions?

Philosophy:
- Keyword-based scoring: no LLM needed, deterministic and reproducible
- Comparison-ready: baseline (without hive) vs hive results show the delta
- JSON-serializable for logging and cross-run comparison
- Agent-agnostic: works with any AgentAdapter-based hive

Public API:
    HiveMindDimensionScore: Score on a single dimension
    HiveMindQuestionResult: Result for a single question
    HiveMindEvalReport: Complete report for a scenario
    score_hive_mind_scenario: Score a hive mind evaluation scenario
    score_single_response: Score a single response against expected keywords
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..adapters.base import AgentResponse
from ..data.hive_mind_scenarios import HiveMindQuestion, HiveMindScenario

logger = logging.getLogger(__name__)


@dataclass
class HiveMindDimensionScore:
    """Score on a single dimension for the hive evaluation.

    Attributes:
        dimension: Name of the scoring dimension
        score: Score from 0.0 to 1.0
        details: Human-readable explanation
    """

    dimension: str
    score: float
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": round(self.score, 4),
            "details": self.details,
        }


@dataclass
class HiveMindQuestionResult:
    """Result for a single hive mind question.

    Attributes:
        question_id: ID of the question
        question_text: The question text
        difficulty: single_domain | cross_domain | synthesis
        required_domains: Domains needed for the answer
        hive_answer: Answer from the hive agent
        baseline_answer: Answer without hive (for comparison)
        hive_score: Score of the hive answer (0.0-1.0)
        baseline_score: Score of the baseline answer (0.0-1.0)
        keywords_found: Which expected keywords were found in the hive answer
        keywords_missing: Which expected keywords were missing
    """

    question_id: str
    question_text: str
    difficulty: str
    required_domains: list[str]
    hive_answer: str
    baseline_answer: str
    hive_score: float
    baseline_score: float
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "difficulty": self.difficulty,
            "required_domains": self.required_domains,
            "hive_score": round(self.hive_score, 4),
            "baseline_score": round(self.baseline_score, 4),
            "improvement": round(self.hive_score - self.baseline_score, 4),
            "keywords_found": self.keywords_found,
            "keywords_missing": self.keywords_missing,
        }


@dataclass
class HiveMindEvalReport:
    """Complete evaluation report for a hive mind scenario.

    Attributes:
        scenario_id: Which scenario was evaluated
        dimensions: Scores on each evaluation dimension
        question_results: Per-question results
        overall_score: Weighted average across all dimensions
        hive_vs_baseline_delta: Average improvement of hive over baseline
        per_difficulty_scores: Breakdown by difficulty level
    """

    scenario_id: str
    dimensions: list[HiveMindDimensionScore]
    question_results: list[HiveMindQuestionResult]
    overall_score: float
    hive_vs_baseline_delta: float
    per_difficulty_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "overall_score": round(self.overall_score, 4),
            "hive_vs_baseline_delta": round(self.hive_vs_baseline_delta, 4),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "per_difficulty_scores": {
                k: round(v, 4) for k, v in self.per_difficulty_scores.items()
            },
            "num_questions": len(self.question_results),
            "question_results": [qr.to_dict() for qr in self.question_results],
        }


def score_single_response(answer: str, expected_keywords: list[str]) -> float:
    """Score a single response against expected keywords.

    Case-insensitive keyword matching. Score is the fraction of
    expected keywords found in the answer.

    Args:
        answer: The agent's answer text
        expected_keywords: Keywords that should appear in a correct answer

    Returns:
        Score from 0.0 to 1.0
    """
    if not expected_keywords:
        return 1.0 if answer.strip() else 0.0

    if not answer or not answer.strip():
        return 0.0

    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def _score_cross_domain_accuracy(
    question_results: list[HiveMindQuestionResult],
) -> HiveMindDimensionScore:
    """Score cross-domain accuracy: can agents answer questions requiring other agents' knowledge?"""
    cross_domain_results = [
        qr for qr in question_results
        if qr.difficulty in ("cross_domain", "synthesis")
    ]

    if not cross_domain_results:
        return HiveMindDimensionScore(
            dimension="cross_domain_accuracy",
            score=0.0,
            details="No cross-domain questions found.",
        )

    avg_score = sum(qr.hive_score for qr in cross_domain_results) / len(cross_domain_results)

    return HiveMindDimensionScore(
        dimension="cross_domain_accuracy",
        score=avg_score,
        details=(
            f"Average score on {len(cross_domain_results)} cross-domain/synthesis questions: "
            f"{avg_score:.2%}"
        ),
    )


def _score_knowledge_coverage(
    scenario: HiveMindScenario,
    coverage_stats: dict[str, Any] | None,
) -> HiveMindDimensionScore:
    """Score knowledge coverage: what % of total knowledge does each agent access?"""
    if coverage_stats is None:
        return HiveMindDimensionScore(
            dimension="knowledge_coverage",
            score=0.0,
            details="No coverage statistics provided.",
        )

    per_agent = coverage_stats.get("per_agent", {})
    if not per_agent:
        return HiveMindDimensionScore(
            dimension="knowledge_coverage",
            score=0.0,
            details="No per-agent coverage data.",
        )

    coverages = [
        agent_stats.get("coverage_pct", 0.0) / 100.0
        for agent_stats in per_agent.values()
    ]
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0

    return HiveMindDimensionScore(
        dimension="knowledge_coverage",
        score=avg_coverage,
        details=(
            f"Average agent coverage: {avg_coverage:.2%} across "
            f"{len(coverages)} agents."
        ),
    )


def _score_collaboration_efficiency(
    propagation_rounds: int,
    max_rounds: int,
    total_facts_propagated: int,
    total_facts_available: int,
) -> HiveMindDimensionScore:
    """Score collaboration efficiency: how quickly does knowledge propagate?

    Higher score = fewer rounds needed to achieve good coverage.
    """
    if max_rounds == 0 or total_facts_available == 0:
        return HiveMindDimensionScore(
            dimension="collaboration_efficiency",
            score=0.0,
            details="No propagation data available.",
        )

    # Efficiency: lower rounds = better, scaled by coverage achieved
    round_efficiency = 1.0 - (propagation_rounds / max_rounds)
    coverage_achieved = min(1.0, total_facts_propagated / total_facts_available)

    # Combined score: efficient propagation that achieves good coverage
    score = (round_efficiency * 0.4) + (coverage_achieved * 0.6)
    score = max(0.0, min(1.0, score))

    return HiveMindDimensionScore(
        dimension="collaboration_efficiency",
        score=score,
        details=(
            f"Propagation: {propagation_rounds}/{max_rounds} rounds, "
            f"{total_facts_propagated} facts propagated "
            f"({coverage_achieved:.0%} coverage)."
        ),
    )


def _score_adversarial_resilience(
    question_results: list[HiveMindQuestionResult],
    scenario: HiveMindScenario,
) -> HiveMindDimensionScore:
    """Score adversarial resilience: does the hive suppress bad data?

    Only applicable to scenarios with a 'misleading' agent domain.
    For other scenarios, returns 1.0 (no adversarial component).
    """
    if "misleading" not in scenario.agent_domains:
        return HiveMindDimensionScore(
            dimension="adversarial_resilience",
            score=1.0,
            details="No adversarial agent in this scenario. Score defaults to 1.0.",
        )

    # For adversarial scenarios, check if synthesis questions (which
    # explicitly mention contradictions) score well
    synthesis_results = [
        qr for qr in question_results
        if qr.difficulty == "synthesis"
    ]

    if not synthesis_results:
        return HiveMindDimensionScore(
            dimension="adversarial_resilience",
            score=0.5,
            details="No synthesis questions to evaluate adversarial resilience.",
        )

    avg_score = sum(qr.hive_score for qr in synthesis_results) / len(synthesis_results)

    return HiveMindDimensionScore(
        dimension="adversarial_resilience",
        score=avg_score,
        details=(
            f"Average score on {len(synthesis_results)} synthesis questions "
            f"with adversarial data: {avg_score:.2%}"
        ),
    )


def _score_no_regression(
    question_results: list[HiveMindQuestionResult],
) -> HiveMindDimensionScore:
    """Score no-regression: do agents maintain accuracy on their own domain?

    Compares hive_score vs baseline_score for single-domain questions.
    Hive should not degrade performance on questions within an agent's own domain.
    """
    single_domain = [
        qr for qr in question_results
        if qr.difficulty == "single_domain"
    ]

    if not single_domain:
        return HiveMindDimensionScore(
            dimension="no_regression",
            score=1.0,
            details="No single-domain questions to check for regression.",
        )

    regressions = 0
    for qr in single_domain:
        if qr.hive_score < qr.baseline_score - 0.05:  # 5% tolerance
            regressions += 1

    regression_rate = regressions / len(single_domain)
    score = 1.0 - regression_rate

    regression_detail = f"{regressions}/{len(single_domain)} questions regressed"
    if regressions == 0:
        regression_detail = "No regressions detected"

    return HiveMindDimensionScore(
        dimension="no_regression",
        score=score,
        details=f"{regression_detail} (5% tolerance).",
    )


def score_hive_mind_scenario(
    scenario: HiveMindScenario,
    responses: dict[str, dict[str, AgentResponse]],
    baseline_responses: dict[str, dict[str, AgentResponse]],
    coverage_stats: dict[str, Any] | None = None,
    propagation_rounds: int = 0,
    max_propagation_rounds: int = 3,
    total_facts_propagated: int = 0,
) -> HiveMindEvalReport:
    """Score a hive mind evaluation scenario.

    Compares hive-connected agent responses against baseline (isolated)
    agent responses across five scoring dimensions.

    Args:
        scenario: The scenario being evaluated
        responses: agent_id -> {question_id -> AgentResponse} from hive agents
        baseline_responses: Same structure, from agents without hive
        coverage_stats: Optional coverage statistics from HiveMindGroupAdapter
        propagation_rounds: Number of gossip rounds executed
        max_propagation_rounds: Maximum configured propagation rounds
        total_facts_propagated: Total facts propagated during gossip

    Returns:
        HiveMindEvalReport with all dimension scores and per-question results
    """
    question_results: list[HiveMindQuestionResult] = []

    # Build a question lookup for fast access
    questions_by_id = {q.question_id: q for q in scenario.questions}

    # Total facts in the scenario
    total_facts = sum(len(facts) for facts in scenario.agent_domains.values())

    # Score each question
    for question in scenario.questions:
        # Find the best response across all agents for this question
        best_hive_answer = ""
        best_hive_score = 0.0
        best_baseline_answer = ""
        best_baseline_score = 0.0

        for agent_id in responses:
            agent_responses = responses[agent_id]
            if question.question_id in agent_responses:
                resp = agent_responses[question.question_id]
                s = score_single_response(resp.answer, question.expected_keywords)
                if s > best_hive_score:
                    best_hive_score = s
                    best_hive_answer = resp.answer

        for agent_id in baseline_responses:
            agent_responses = baseline_responses[agent_id]
            if question.question_id in agent_responses:
                resp = agent_responses[question.question_id]
                s = score_single_response(resp.answer, question.expected_keywords)
                if s > best_baseline_score:
                    best_baseline_score = s
                    best_baseline_answer = resp.answer

        # Determine which keywords were found/missing
        answer_lower = best_hive_answer.lower()
        found = [kw for kw in question.expected_keywords if kw.lower() in answer_lower]
        missing = [kw for kw in question.expected_keywords if kw.lower() not in answer_lower]

        question_results.append(HiveMindQuestionResult(
            question_id=question.question_id,
            question_text=question.text,
            difficulty=question.difficulty,
            required_domains=question.required_domains,
            hive_answer=best_hive_answer,
            baseline_answer=best_baseline_answer,
            hive_score=best_hive_score,
            baseline_score=best_baseline_score,
            keywords_found=found,
            keywords_missing=missing,
        ))

    # Score all dimensions
    cross_domain = _score_cross_domain_accuracy(question_results)
    coverage = _score_knowledge_coverage(scenario, coverage_stats)
    efficiency = _score_collaboration_efficiency(
        propagation_rounds=propagation_rounds,
        max_rounds=max_propagation_rounds,
        total_facts_propagated=total_facts_propagated,
        total_facts_available=total_facts,
    )
    resilience = _score_adversarial_resilience(question_results, scenario)
    no_regression = _score_no_regression(question_results)

    dimensions = [cross_domain, coverage, efficiency, resilience, no_regression]

    # Overall score: weighted average of dimensions
    # Weight adversarial_resilience lower for non-adversarial scenarios
    weights = {
        "cross_domain_accuracy": 0.30,
        "knowledge_coverage": 0.20,
        "collaboration_efficiency": 0.15,
        "adversarial_resilience": 0.15,
        "no_regression": 0.20,
    }
    overall = sum(
        d.score * weights.get(d.dimension, 0.2)
        for d in dimensions
    )

    # Hive vs baseline delta
    hive_avg = (
        sum(qr.hive_score for qr in question_results) / len(question_results)
        if question_results else 0.0
    )
    baseline_avg = (
        sum(qr.baseline_score for qr in question_results) / len(question_results)
        if question_results else 0.0
    )
    delta = hive_avg - baseline_avg

    # Per-difficulty breakdown
    per_difficulty: dict[str, float] = {}
    for difficulty in ("single_domain", "cross_domain", "synthesis"):
        diff_results = [qr for qr in question_results if qr.difficulty == difficulty]
        if diff_results:
            per_difficulty[difficulty] = (
                sum(qr.hive_score for qr in diff_results) / len(diff_results)
            )

    return HiveMindEvalReport(
        scenario_id=scenario.scenario_id,
        dimensions=dimensions,
        question_results=question_results,
        overall_score=overall,
        hive_vs_baseline_delta=delta,
        per_difficulty_scores=per_difficulty,
    )


__all__ = [
    "HiveMindDimensionScore",
    "HiveMindQuestionResult",
    "HiveMindEvalReport",
    "score_hive_mind_scenario",
    "score_single_response",
]
