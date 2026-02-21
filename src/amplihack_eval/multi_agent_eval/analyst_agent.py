"""Analyst agent that identifies patterns in eval results and proposes improvements.

Provides deep analysis of evaluation results including:
- Failure patterns (which question types consistently fail?)
- Regression risks (which improvements might hurt other areas?)
- Bottleneck identification (retrieval? synthesis? grading?)
- Improvement priorities (highest-impact fixes)
- Cross-run comparison (how did the agent change between runs?)

Philosophy:
- Data-driven: all analysis is grounded in actual eval results
- Actionable: every insight comes with a concrete improvement suggestion
- JSON-serializable for logging and reporting
- Works without LLM for basic statistics; uses LLM for deeper insights

Public API:
    AnalysisReport: Deep analysis of a single evaluation run
    ComparisonReport: Comparison of multiple evaluation runs
    Improvement: Concrete improvement suggestion with expected impact
    AnalystAgent: Analyzes eval results and proposes improvements
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from ..core.runner import EvalReport, EvalResult

logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """A recurring failure pattern identified in eval results."""

    pattern_name: str
    description: str
    affected_categories: list[str]
    affected_question_ids: list[str]
    frequency: float  # 0.0 to 1.0 (fraction of questions affected)
    severity: float  # 0.0 to 1.0 (average score on affected questions)
    example_question: str = ""
    example_expected: str = ""
    example_actual: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "description": self.description,
            "affected_categories": self.affected_categories,
            "num_affected": len(self.affected_question_ids),
            "frequency": round(self.frequency, 4),
            "severity": round(self.severity, 4),
            "example_question": self.example_question[:200],
        }


@dataclass
class Improvement:
    """Concrete improvement suggestion with expected impact."""

    title: str
    description: str
    target_component: str  # "retrieval", "synthesis", "grading", "data", "prompt"
    expected_impact: float  # Estimated score improvement (0.0 to 1.0)
    confidence: float  # How confident we are (0.0 to 1.0)
    effort: str  # "low", "medium", "high"
    addresses_patterns: list[str]  # Pattern names this would fix

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "target_component": self.target_component,
            "expected_impact": round(self.expected_impact, 4),
            "confidence": round(self.confidence, 4),
            "effort": self.effort,
            "addresses_patterns": self.addresses_patterns,
        }


@dataclass
class AnalysisReport:
    """Deep analysis of evaluation results."""

    overall_score: float
    num_questions: int
    failure_patterns: list[FailurePattern]
    category_scores: dict[str, float]
    bottleneck_component: str
    bottleneck_reasoning: str
    improvement_priorities: list[Improvement]
    raw_llm_analysis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 4),
            "num_questions": self.num_questions,
            "failure_patterns": [fp.to_dict() for fp in self.failure_patterns],
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "bottleneck_component": self.bottleneck_component,
            "bottleneck_reasoning": self.bottleneck_reasoning,
            "improvement_priorities": [imp.to_dict() for imp in self.improvement_priorities],
        }


@dataclass
class ComparisonReport:
    """Comparison of multiple evaluation runs."""

    run_labels: list[str]
    overall_scores: dict[str, float]
    category_trends: dict[str, dict[str, float]]  # category -> {run_label: score}
    regressions: list[dict[str, Any]]
    improvements: list[dict[str, Any]]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_labels": self.run_labels,
            "overall_scores": {k: round(v, 4) for k, v in self.overall_scores.items()},
            "category_trends": {
                cat: {k: round(v, 4) for k, v in scores.items()}
                for cat, scores in self.category_trends.items()
            },
            "num_regressions": len(self.regressions),
            "num_improvements": len(self.improvements),
            "regressions": self.regressions,
            "improvements": self.improvements,
            "summary": self.summary,
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


class AnalystAgent:
    """Analyzes eval results and proposes improvements.

    Combines statistical analysis (no LLM needed) with optional LLM-powered
    deeper insight generation for failure pattern identification and
    improvement suggestions.

    Args:
        model: LLM model identifier (default: from GRADER_MODEL env var)

    Example::

        analyst = AnalystAgent()
        analysis = analyst.analyze(eval_report)
        for imp in analysis.improvement_priorities:
            print(f"{imp.title}: expected +{imp.expected_impact:.0%}")
    """

    def __init__(self, model: str = ""):
        self.model = model or os.environ.get("GRADER_MODEL", "claude-sonnet-4-5-20250929")

    def analyze(self, report: EvalReport) -> AnalysisReport:
        """Deep analysis of evaluation results.

        Identifies failure patterns, bottleneck components, and improvement
        priorities. Uses statistical analysis first, then optionally enriches
        with LLM insights.

        Args:
            report: Complete evaluation report

        Returns:
            AnalysisReport with patterns, bottleneck, and improvements
        """
        # Step 1: Compute category scores
        category_scores = {
            cb.category: cb.avg_score
            for cb in report.category_breakdown
        }

        # Step 2: Identify failure patterns (statistical)
        failure_patterns = self._identify_failure_patterns(report)

        # Step 3: Identify bottleneck component
        bottleneck, bottleneck_reason = self._identify_bottleneck(report, failure_patterns)

        # Step 4: Generate improvement suggestions
        improvements = self._suggest_improvements_statistical(
            report, failure_patterns, bottleneck
        )

        # Step 5: Optionally enrich with LLM analysis
        raw_analysis = ""
        llm_improvements = self._suggest_improvements_llm(report, failure_patterns)
        if llm_improvements:
            improvements.extend(llm_improvements)
            # Sort by expected impact descending
            improvements.sort(key=lambda imp: imp.expected_impact, reverse=True)

        return AnalysisReport(
            overall_score=report.overall_score,
            num_questions=report.num_questions,
            failure_patterns=failure_patterns,
            category_scores=category_scores,
            bottleneck_component=bottleneck,
            bottleneck_reasoning=bottleneck_reason,
            improvement_priorities=improvements,
            raw_llm_analysis=raw_analysis,
        )

    def compare_reports(
        self,
        reports: list[EvalReport],
        labels: list[str] | None = None,
    ) -> ComparisonReport:
        """Compare multiple eval runs to identify trends.

        Args:
            reports: List of evaluation reports (in chronological order)
            labels: Optional labels for each run (e.g., ["v1", "v2", "v3"])

        Returns:
            ComparisonReport with trends, regressions, and improvements
        """
        if not reports:
            return ComparisonReport(
                run_labels=[],
                overall_scores={},
                category_trends={},
                regressions=[],
                improvements=[],
                summary="No reports to compare.",
            )

        if labels is None:
            labels = [f"run_{i}" for i in range(len(reports))]

        # Overall scores per run
        overall_scores = {
            label: report.overall_score
            for label, report in zip(labels, reports)
        }

        # Category trends
        all_categories: set[str] = set()
        for report in reports:
            for cb in report.category_breakdown:
                all_categories.add(cb.category)

        category_trends: dict[str, dict[str, float]] = {}
        for cat in sorted(all_categories):
            category_trends[cat] = {}
            for label, report in zip(labels, reports):
                for cb in report.category_breakdown:
                    if cb.category == cat:
                        category_trends[cat][label] = cb.avg_score
                        break

        # Detect regressions and improvements between consecutive runs
        regressions: list[dict[str, Any]] = []
        improvements_list: list[dict[str, Any]] = []

        for i in range(1, len(reports)):
            prev_label = labels[i - 1]
            curr_label = labels[i]

            for cat in all_categories:
                prev_score = category_trends.get(cat, {}).get(prev_label, 0.0)
                curr_score = category_trends.get(cat, {}).get(curr_label, 0.0)
                delta = curr_score - prev_score

                if delta < -0.05:  # 5pp regression threshold
                    regressions.append({
                        "category": cat,
                        "from_run": prev_label,
                        "to_run": curr_label,
                        "from_score": round(prev_score, 4),
                        "to_score": round(curr_score, 4),
                        "delta": round(delta, 4),
                    })
                elif delta > 0.05:  # 5pp improvement threshold
                    improvements_list.append({
                        "category": cat,
                        "from_run": prev_label,
                        "to_run": curr_label,
                        "from_score": round(prev_score, 4),
                        "to_score": round(curr_score, 4),
                        "delta": round(delta, 4),
                    })

        # Build summary
        overall_trend = ""
        if len(reports) >= 2:
            first_score = reports[0].overall_score
            last_score = reports[-1].overall_score
            delta = last_score - first_score
            direction = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
            overall_trend = (
                f"Overall score {direction} from {first_score:.2%} to {last_score:.2%} "
                f"({delta:+.2%}) across {len(reports)} runs."
            )

        summary_parts = [overall_trend]
        if regressions:
            summary_parts.append(
                f"{len(regressions)} category regression(s) detected (>5pp drop)."
            )
        if improvements_list:
            summary_parts.append(
                f"{len(improvements_list)} category improvement(s) detected (>5pp gain)."
            )

        return ComparisonReport(
            run_labels=labels,
            overall_scores=overall_scores,
            category_trends=category_trends,
            regressions=regressions,
            improvements=improvements_list,
            summary=" ".join(summary_parts),
        )

    def suggest_improvements(self, analysis: AnalysisReport) -> list[Improvement]:
        """Concrete improvement suggestions with expected impact.

        Returns the improvement priorities from the analysis, which are
        already computed during analyze().

        Args:
            analysis: AnalysisReport from analyze()

        Returns:
            List of Improvement suggestions sorted by expected impact
        """
        return sorted(
            analysis.improvement_priorities,
            key=lambda imp: imp.expected_impact,
            reverse=True,
        )

    def _identify_failure_patterns(self, report: EvalReport) -> list[FailurePattern]:
        """Identify recurring failure patterns in eval results.

        Uses statistical analysis to cluster failures by category and
        identify common patterns.
        """
        patterns: list[FailurePattern] = []

        # Pattern 1: Low-scoring categories
        for cb in report.category_breakdown:
            if cb.avg_score < 0.5:
                affected_ids = [
                    r.question_id for r in report.results
                    if r.category == cb.category and r.overall_score < 0.5
                ]
                example = next(
                    (r for r in report.results
                     if r.category == cb.category and r.overall_score < 0.5),
                    None,
                )
                patterns.append(FailurePattern(
                    pattern_name=f"weak_{cb.category}",
                    description=f"Category '{cb.category}' scores below 50% on average",
                    affected_categories=[cb.category],
                    affected_question_ids=affected_ids,
                    frequency=len(affected_ids) / max(1, cb.num_questions),
                    severity=cb.avg_score,
                    example_question=example.question_text if example else "",
                    example_expected=example.expected_answer if example else "",
                    example_actual=example.actual_answer if example else "",
                ))

        # Pattern 2: High-variance categories (inconsistent performance)
        for cb in report.category_breakdown:
            if cb.max_score - cb.min_score > 0.5 and cb.num_questions >= 3:
                affected_ids = [
                    r.question_id for r in report.results
                    if r.category == cb.category
                ]
                patterns.append(FailurePattern(
                    pattern_name=f"inconsistent_{cb.category}",
                    description=(
                        f"Category '{cb.category}' has high variance "
                        f"(min={cb.min_score:.2f}, max={cb.max_score:.2f})"
                    ),
                    affected_categories=[cb.category],
                    affected_question_ids=affected_ids,
                    frequency=1.0,
                    severity=cb.avg_score,
                ))

        # Pattern 3: Zero-score questions
        zero_results = [r for r in report.results if r.overall_score == 0.0]
        if zero_results:
            cats = list({r.category for r in zero_results})
            patterns.append(FailurePattern(
                pattern_name="total_failure",
                description=f"{len(zero_results)} questions scored 0.0 (complete failure)",
                affected_categories=cats,
                affected_question_ids=[r.question_id for r in zero_results],
                frequency=len(zero_results) / max(1, len(report.results)),
                severity=0.0,
                example_question=zero_results[0].question_text if zero_results else "",
                example_expected=zero_results[0].expected_answer if zero_results else "",
                example_actual=zero_results[0].actual_answer if zero_results else "",
            ))

        return patterns

    def _identify_bottleneck(
        self,
        report: EvalReport,
        patterns: list[FailurePattern],
    ) -> tuple[str, str]:
        """Identify the bottleneck component causing most failures.

        Returns:
            Tuple of (component_name, reasoning)
        """
        # Heuristic: look at dimension averages across all categories
        dim_scores: dict[str, list[float]] = {}
        for cb in report.category_breakdown:
            for dim, avg in cb.dimension_averages.items():
                dim_scores.setdefault(dim, []).append(avg)

        if dim_scores:
            dim_averages = {
                dim: statistics.mean(scores)
                for dim, scores in dim_scores.items()
            }
            worst_dim = min(dim_averages, key=lambda d: dim_averages[d])
            worst_score = dim_averages[worst_dim]

            # Map dimension to component
            dim_to_component = {
                "factual_accuracy": "retrieval",
                "specificity": "retrieval",
                "temporal_awareness": "synthesis",
                "source_attribution": "synthesis",
                "confidence_calibration": "prompt",
            }
            component = dim_to_component.get(worst_dim, "unknown")
            reasoning = (
                f"Dimension '{worst_dim}' has the lowest average score ({worst_score:.2f}), "
                f"suggesting the {component} component is the bottleneck."
            )
            return component, reasoning

        # Fallback: use failure patterns
        if patterns:
            most_severe = min(patterns, key=lambda p: p.severity)
            return "unknown", f"Most severe pattern: {most_severe.pattern_name}"

        return "unknown", "Insufficient data for bottleneck identification"

    def _suggest_improvements_statistical(
        self,
        report: EvalReport,
        patterns: list[FailurePattern],
        bottleneck: str,
    ) -> list[Improvement]:
        """Generate improvement suggestions from statistical analysis."""
        improvements: list[Improvement] = []

        # Improvement for each failure pattern
        for fp in patterns:
            if fp.pattern_name.startswith("weak_"):
                improvements.append(Improvement(
                    title=f"Improve {fp.affected_categories[0]} performance",
                    description=(
                        f"Category scores {fp.severity:.0%} on average. "
                        f"Focus on {len(fp.affected_question_ids)} underperforming questions."
                    ),
                    target_component=bottleneck,
                    expected_impact=min(0.3, (0.7 - fp.severity)),
                    confidence=0.6,
                    effort="medium",
                    addresses_patterns=[fp.pattern_name],
                ))
            elif fp.pattern_name.startswith("inconsistent_"):
                improvements.append(Improvement(
                    title=f"Stabilize {fp.affected_categories[0]} performance",
                    description=(
                        f"High variance in this category. Consider adding more "
                        f"deterministic grading rules or improving retrieval consistency."
                    ),
                    target_component="retrieval",
                    expected_impact=0.1,
                    confidence=0.5,
                    effort="low",
                    addresses_patterns=[fp.pattern_name],
                ))
            elif fp.pattern_name == "total_failure":
                improvements.append(Improvement(
                    title="Fix zero-score questions",
                    description=(
                        f"{len(fp.affected_question_ids)} questions scoring 0.0 "
                        f"suggest fundamental issues with retrieval or understanding."
                    ),
                    target_component="retrieval",
                    expected_impact=0.2,
                    confidence=0.7,
                    effort="high",
                    addresses_patterns=["total_failure"],
                ))

        return improvements

    def _suggest_improvements_llm(
        self,
        report: EvalReport,
        patterns: list[FailurePattern],
    ) -> list[Improvement]:
        """Use LLM for deeper improvement suggestions."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return []

        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(api_key=api_key)

        # Build context for the LLM
        pattern_text = "\n".join(
            f"- {fp.pattern_name}: {fp.description} (severity={fp.severity:.2f})"
            for fp in patterns
        )

        cat_text = "\n".join(
            f"- {cb.category}: avg={cb.avg_score:.2f}, min={cb.min_score:.2f}, max={cb.max_score:.2f}"
            for cb in report.category_breakdown
        )

        # Sample of worst results
        worst = sorted(report.results, key=lambda r: r.overall_score)[:5]
        worst_text = "\n".join(
            f"- [{r.overall_score:.2f}] Q: {r.question_text[:60]}... "
            f"Expected: {r.expected_answer[:40]}... Got: {r.actual_answer[:40]}..."
            for r in worst
        )

        prompt = f"""Analyze these evaluation results and suggest specific improvements.

Overall score: {report.overall_score:.2%}
Questions: {report.num_questions}

Category breakdown:
{cat_text}

Identified failure patterns:
{pattern_text}

Worst-performing questions:
{worst_text}

Suggest 2-3 concrete, actionable improvements. For each:
1. What specific change to make
2. Which component to change (retrieval, synthesis, grading, prompt, data)
3. Expected impact (0.0 to 1.0 score improvement)
4. Confidence in this suggestion (0.0 to 1.0)
5. Effort level (low, medium, high)

Return ONLY JSON: {{"improvements": [{{"title": "...", "description": "...", "target_component": "...", "expected_impact": 0.1, "confidence": 0.7, "effort": "medium"}}]}}"""

        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = message.content[0].text
            result = _extract_json(raw)
            items = result.get("improvements", [])

            return [
                Improvement(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    target_component=item.get("target_component", "unknown"),
                    expected_impact=float(item.get("expected_impact", 0.0)),
                    confidence=float(item.get("confidence", 0.0)),
                    effort=item.get("effort", "medium"),
                    addresses_patterns=[],
                )
                for item in items
                if item.get("title")
            ]

        except Exception as e:
            logger.warning("LLM improvement suggestion failed: %s", e)
            return []


__all__ = [
    "AnalystAgent",
    "AnalysisReport",
    "ComparisonReport",
    "FailurePattern",
    "Improvement",
]
