"""Multi-seed holdout evaluation for long-horizon memory.

Runs the long-horizon eval across multiple random seeds to measure
inter-seed variance and flag noisy questions.

Philosophy:
- A single seed can produce unrepresentative results due to data ordering
- Running across 4+ seeds reveals which scores are stable vs noisy
- Questions with >10pp inter-seed variance are flagged for investigation
- Mean +/- stddev gives confidence intervals for each category

Public API:
    MultiSeedConfig: Configuration for multi-seed evaluation
    MultiSeedReport: Aggregate report across seeds
    run_multi_seed_eval: Execute multi-seed evaluation
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..adapters.base import AgentAdapter
from .runner import EvalReport, EvalRunner

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 123, 456, 789]
NOISY_THRESHOLD_PP = 10.0  # 10 percentage points


@dataclass
class QuestionVariance:
    """Per-question variance across seeds."""

    question_id: str
    question_text: str
    category: str
    scores_by_seed: dict[int, float]
    mean_score: float
    stddev: float
    is_noisy: bool  # True if stddev > NOISY_THRESHOLD_PP / 100


@dataclass
class CategoryStats:
    """Per-category statistics across seeds."""

    category: str
    mean_score: float
    stddev: float
    min_score: float
    max_score: float
    scores_by_seed: dict[int, float]
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    margin_of_error: float = 0.0


@dataclass
class MultiSeedReport:
    """Aggregate report across multiple seeds."""

    seeds: list[int]
    num_turns: int
    num_questions: int
    total_time_s: float
    overall_mean: float
    overall_stddev: float
    category_stats: list[CategoryStats]
    noisy_questions: list[QuestionVariance]
    all_question_variances: list[QuestionVariance]
    per_seed_reports: dict[int, EvalReport]
    overall_ci_95_lower: float = 0.0
    overall_ci_95_upper: float = 0.0
    overall_margin_of_error: float = 0.0
    repeats_per_seed: int = 1
    intra_seed_stddev: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "seeds": self.seeds,
            "num_turns": self.num_turns,
            "num_questions": self.num_questions,
            "total_time_s": round(self.total_time_s, 2),
            "overall_mean": round(self.overall_mean, 4),
            "overall_stddev": round(self.overall_stddev, 4),
            "overall_ci_95_lower": round(self.overall_ci_95_lower, 4),
            "overall_ci_95_upper": round(self.overall_ci_95_upper, 4),
            "overall_margin_of_error": round(self.overall_margin_of_error, 4),
            "repeats_per_seed": self.repeats_per_seed,
            "intra_seed_stddev": round(self.intra_seed_stddev, 4),
            "category_stats": [
                {
                    "category": cs.category,
                    "mean_score": round(cs.mean_score, 4),
                    "stddev": round(cs.stddev, 4),
                    "min_score": round(cs.min_score, 4),
                    "max_score": round(cs.max_score, 4),
                    "scores_by_seed": {str(k): round(v, 4) for k, v in cs.scores_by_seed.items()},
                    "ci_95_lower": round(cs.ci_95_lower, 4),
                    "ci_95_upper": round(cs.ci_95_upper, 4),
                    "margin_of_error": round(cs.margin_of_error, 4),
                }
                for cs in self.category_stats
            ],
            "noisy_questions": [
                {
                    "question_id": q.question_id,
                    "question_text": q.question_text[:80],
                    "category": q.category,
                    "mean_score": round(q.mean_score, 4),
                    "stddev": round(q.stddev, 4),
                    "scores_by_seed": {str(k): round(v, 4) for k, v in q.scores_by_seed.items()},
                }
                for q in self.noisy_questions
            ],
            "num_noisy_questions": len(self.noisy_questions),
            "total_questions_evaluated": len(self.all_question_variances),
        }


def _safe_stddev(values: list[float]) -> float:
    """Compute sample standard deviation, returning 0 for <2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _ci_95(mean: float, stddev: float, n: int) -> tuple[float, float, float]:
    """Compute 95% confidence interval for a mean.

    Args:
        mean: Sample mean (expected in [0, 1] range)
        stddev: Sample standard deviation
        n: Sample size

    Returns:
        (lower, upper, margin_of_error) tuple.
        Bounds are clamped to [0.0, 1.0]. Returns (mean, mean, 0.0) for n < 2.
    """
    if n < 2:
        return (mean, mean, 0.0)
    moe = 1.96 * stddev / math.sqrt(n)
    return (max(0.0, mean - moe), min(1.0, mean + moe), moe)


def run_multi_seed_eval(
    agent_factory: Callable[[], AgentAdapter],
    num_turns: int = 100,
    num_questions: int = 20,
    seeds: list[int] | None = None,
    grader_model: str = "",
    grader_votes: int = 3,
    repeats_per_seed: int = 1,
) -> MultiSeedReport:
    """Run long-horizon eval across multiple seeds.

    Args:
        agent_factory: Callable that returns a fresh AgentAdapter instance.
            Each seed gets its own agent to avoid cross-contamination.
        num_turns: Number of dialogue turns per seed
        num_questions: Number of quiz questions per seed
        seeds: Random seeds to use (default: [42, 123, 456, 789])
        grader_model: Model for LLM grading
        grader_votes: Number of grading votes per question
        repeats_per_seed: Number of repeat runs per seed. When > 1, each seed
            is run N times. The median-scoring run is kept as the representative
            report for that seed. Intra-seed stddev is tracked.

    Returns:
        MultiSeedReport with variance analysis and confidence intervals
    """
    seeds = seeds or DEFAULT_SEEDS
    start_time = time.time()

    per_seed_reports: dict[int, EvalReport] = {}
    intra_seed_stddevs: list[float] = []

    for seed in seeds:
        if repeats_per_seed > 1:
            # Run multiple times per seed, keep the median
            repeat_reports: list[EvalReport] = []
            for rep in range(repeats_per_seed):
                logger.info("=== Running seed %d, repeat %d/%d ===", seed, rep + 1, repeats_per_seed)
                agent = agent_factory()
                try:
                    evaluator = EvalRunner(
                        num_turns=num_turns,
                        num_questions=num_questions,
                        seed=seed,
                        grader_votes=grader_votes,
                    )
                    report = evaluator.run(agent, grader_model=grader_model)
                    repeat_reports.append(report)
                    logger.info(
                        "Seed %d repeat %d: overall=%.2f%%",
                        seed, rep + 1, report.overall_score * 100,
                    )
                finally:
                    agent.close()

            # Pick the report closest to the median score
            scores = [r.overall_score for r in repeat_reports]
            median_score = statistics.median(scores)
            best_report = min(repeat_reports, key=lambda r: abs(r.overall_score - median_score))
            per_seed_reports[seed] = best_report

            # Track intra-seed variance
            intra_seed_stddevs.append(_safe_stddev(scores))
        else:
            logger.info("=== Running seed %d ===", seed)
            agent = agent_factory()
            try:
                evaluator = EvalRunner(
                    num_turns=num_turns,
                    num_questions=num_questions,
                    seed=seed,
                    grader_votes=grader_votes,
                )
                report = evaluator.run(agent, grader_model=grader_model)
                per_seed_reports[seed] = report
                logger.info("Seed %d complete: overall=%.2f%%", seed, report.overall_score * 100)
            finally:
                agent.close()

    total_time = time.time() - start_time

    # Compute overall mean/stddev across seeds
    overall_scores = [r.overall_score for r in per_seed_reports.values()]
    overall_mean = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    overall_stddev = _safe_stddev(overall_scores)

    # Compute overall 95% CI
    n_seeds = len(overall_scores)
    overall_ci_lower, overall_ci_upper, overall_moe = _ci_95(overall_mean, overall_stddev, n_seeds)

    # Compute mean intra-seed stddev
    intra_seed_stddev = (
        sum(intra_seed_stddevs) / len(intra_seed_stddevs) if intra_seed_stddevs else 0.0
    )

    # Compute per-category stats
    all_categories: set[str] = set()
    for report in per_seed_reports.values():
        for cb in report.category_breakdown:
            all_categories.add(cb.category)

    category_stats: list[CategoryStats] = []
    for cat in sorted(all_categories):
        cat_scores: dict[int, float] = {}
        for seed, report in per_seed_reports.items():
            for cb in report.category_breakdown:
                if cb.category == cat:
                    cat_scores[seed] = cb.avg_score
                    break

        values = list(cat_scores.values())
        cat_mean = sum(values) / len(values) if values else 0.0
        cat_stddev = _safe_stddev(values)
        cat_ci_lower, cat_ci_upper, cat_moe = _ci_95(cat_mean, cat_stddev, len(values))

        category_stats.append(
            CategoryStats(
                category=cat,
                mean_score=cat_mean,
                stddev=cat_stddev,
                min_score=min(values) if values else 0.0,
                max_score=max(values) if values else 0.0,
                scores_by_seed=cat_scores,
                ci_95_lower=cat_ci_lower,
                ci_95_upper=cat_ci_upper,
                margin_of_error=cat_moe,
            )
        )

    # Compute per-question variance
    question_scores: dict[str, dict[int, float]] = {}
    question_meta: dict[str, tuple[str, str]] = {}  # qid -> (text, category)

    for seed, report in per_seed_reports.items():
        for result in report.results:
            qid = result.question_id
            if qid not in question_scores:
                question_scores[qid] = {}
                question_meta[qid] = (result.question_text, result.category)
            question_scores[qid][seed] = result.overall_score

    all_variances: list[QuestionVariance] = []
    noisy: list[QuestionVariance] = []

    for qid, seed_scores in sorted(question_scores.items()):
        text, cat = question_meta[qid]
        values = list(seed_scores.values())
        mean_val = sum(values) / len(values) if values else 0.0
        std_val = _safe_stddev(values)
        is_noisy = std_val > (NOISY_THRESHOLD_PP / 100.0)

        qv = QuestionVariance(
            question_id=qid,
            question_text=text,
            category=cat,
            scores_by_seed=seed_scores,
            mean_score=mean_val,
            stddev=std_val,
            is_noisy=is_noisy,
        )
        all_variances.append(qv)
        if is_noisy:
            noisy.append(qv)

    logger.info(
        "Multi-seed eval complete: mean=%.2f%% +/- %.2f%%, %d noisy questions out of %d",
        overall_mean * 100,
        overall_stddev * 100,
        len(noisy),
        len(all_variances),
    )

    return MultiSeedReport(
        seeds=seeds,
        num_turns=num_turns,
        num_questions=num_questions,
        total_time_s=total_time,
        overall_mean=overall_mean,
        overall_stddev=overall_stddev,
        category_stats=category_stats,
        noisy_questions=noisy,
        all_question_variances=all_variances,
        per_seed_reports=per_seed_reports,
        overall_ci_95_lower=overall_ci_lower,
        overall_ci_95_upper=overall_ci_upper,
        overall_margin_of_error=overall_moe,
        repeats_per_seed=repeats_per_seed,
        intra_seed_stddev=intra_seed_stddev,
    )


def print_multi_seed_report(report: MultiSeedReport) -> None:
    """Print human-readable multi-seed comparison report."""
    print("\n" + "=" * 70)
    print("MULTI-SEED LONG-HORIZON MEMORY EVALUATION")
    print("=" * 70)
    print(f"Seeds: {report.seeds}")
    print(f"Turns: {report.num_turns} | Questions: {report.num_questions}")
    if report.repeats_per_seed > 1:
        print(f"Repeats per seed: {report.repeats_per_seed}")
    print(f"Total time: {report.total_time_s:.1f}s")

    if report.overall_margin_of_error > 0:
        moe_pp = report.overall_margin_of_error * 100
        lo_pct = report.overall_ci_95_lower * 100
        hi_pct = report.overall_ci_95_upper * 100
        print(
            f"\nOVERALL: {report.overall_mean:.2%} "
            f"+/- {moe_pp:.1f}pp (95% CI: {lo_pct:.1f}% -- {hi_pct:.1f}%)"
        )
    else:
        print(f"\nOVERALL: {report.overall_mean:.2%} +/- {report.overall_stddev:.2%}")

    print("\nPER-SEED SCORES:")
    for seed, seed_report in sorted(report.per_seed_reports.items()):
        print(f"  Seed {seed}: {seed_report.overall_score:.2%}")

    print("\nCATEGORY BREAKDOWN (mean +/- stddev):")
    print("-" * 70)
    print(f"{'Category':<25} {'Mean':>8} {'StdDev':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    for cs in report.category_stats:
        print(
            f"{cs.category:<25} {cs.mean_score:>7.2%} {cs.stddev:>7.2%} "
            f"{cs.min_score:>7.2%} {cs.max_score:>7.2%}"
        )
    print("-" * 70)

    if report.noisy_questions:
        print(f"\nNOISY QUESTIONS ({len(report.noisy_questions)} with >10pp inter-seed variance):")
        for qv in sorted(report.noisy_questions, key=lambda q: -q.stddev):
            seeds_str = ", ".join(f"s{s}={v:.0%}" for s, v in sorted(qv.scores_by_seed.items()))
            print(f"  [{qv.stddev:.2%}] {qv.question_text[:55]}")
            print(f"    {seeds_str}")
    else:
        print("\nNo noisy questions detected (all within 10pp variance).")


__all__ = [
    "MultiSeedReport",
    "QuestionVariance",
    "CategoryStats",
    "run_multi_seed_eval",
    "print_multi_seed_report",
    "_ci_95",
]
