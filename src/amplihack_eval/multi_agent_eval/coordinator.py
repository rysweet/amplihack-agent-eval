"""Orchestrates the multi-agent evaluation pipeline.

The EvalCoordinator manages the full evaluation lifecycle:
1. Data agent generates dialogue (via EvalRunner)
2. Feed content to the agent under test
3. Grader agents score in parallel (multi-vote across perspectives)
4. Adversary generates targeted hard questions
5. Analyst identifies patterns in failures

Philosophy:
- Coordinator orchestrates but does not grade/analyze itself
- Each agent works independently (testable in isolation)
- All results are JSON-serializable for logging
- No amplihack dependencies -- standalone package

Public API:
    EvalConfig: Configuration for multi-agent evaluation
    EvalCoordinator: Orchestrates the multi-agent evaluation pipeline
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..adapters.base import AgentAdapter
from ..core.runner import EvalReport, EvalResult, EvalRunner, CategoryBreakdown, DimensionScore
from ..data.long_horizon import GroundTruth, Question
from .adversary_agent import AdversaryAgent
from .analyst_agent import AnalysisReport, AnalystAgent
from .grader_agent import AggregateGrade, GraderAgent

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for multi-agent evaluation.

    Args:
        num_turns: Number of dialogue turns for data generation
        num_questions: Number of standard quiz questions
        seed: Random seed for reproducibility
        grader_perspectives: Which grading perspectives to use
        enable_adversary: Whether to run adversarial question generation
        adversarial_questions: Number of adversarial questions to generate
        grader_model: Model for grading LLM calls
    """

    num_turns: int = 100
    num_questions: int = 20
    seed: int = 42
    grader_perspectives: list[str] = field(
        default_factory=lambda: ["factual", "reasoning", "completeness"]
    )
    enable_adversary: bool = True
    adversarial_questions: int = 10
    grader_model: str = ""


class EvalCoordinator:
    """Orchestrates the multi-agent evaluation pipeline.

    Creates and manages specialized grader agents, adversary agent, and
    analyst agent. Coordinates the evaluation flow and produces a
    comprehensive report.

    Args:
        grader_agents: Number of grading perspectives (1-3)
        enable_adversary: Whether to include adversarial question round

    Example::

        coordinator = EvalCoordinator(grader_agents=3, enable_adversary=True)
        config = EvalConfig(num_turns=100, num_questions=20)
        report = coordinator.run_eval(agent, config)
        print(f"Overall: {report.overall_score:.2%}")
    """

    def __init__(
        self,
        grader_agents: int = 3,
        enable_adversary: bool = True,
    ):
        self._num_graders = max(1, min(grader_agents, len(GraderAgent.PERSPECTIVES)))
        self._enable_adversary = enable_adversary
        self._graders: list[GraderAgent] = []
        self._adversary: AdversaryAgent | None = None
        self._analyst = AnalystAgent()

    def run_eval(
        self,
        agent: AgentAdapter,
        config: EvalConfig,
    ) -> EvalReport:
        """Full evaluation pipeline.

        Flow:
        1. Generate dialogue data
        2. Feed to agent under test (learning phase)
        3. Quiz agent on standard questions
        4. Grade answers with multi-perspective graders
        5. (Optional) Adversary generates targeted questions
        6. (Optional) Agent answers adversarial questions
        7. (Optional) Grade adversarial answers
        8. Analyst identifies patterns
        9. Return comprehensive report

        Args:
            agent: The agent to evaluate
            config: Evaluation configuration

        Returns:
            EvalReport with all results including adversarial and analysis
        """
        overall_start = time.time()

        # Initialize agents
        self._init_agents(config)

        # Step 1: Generate data
        logger.info("Step 1: Generating dialogue data (%d turns)", config.num_turns)
        runner = EvalRunner(
            num_turns=config.num_turns,
            num_questions=config.num_questions,
            seed=config.seed,
        )
        ground_truth, questions = runner.generate()
        logger.info("Generated %d turns, %d questions", len(ground_truth.turns), len(questions))

        # Step 2: Learning phase
        logger.info("Step 2: Learning phase")
        learning_time = runner.run_dialogue(agent, ground_truth)

        # Step 3: Standard questioning with multi-perspective grading
        logger.info("Step 3: Questioning with multi-perspective grading")
        results, grading_time = self._question_and_grade(agent, questions)

        # Step 4: Adversarial round (optional)
        adversarial_results: list[EvalResult] = []
        if self._enable_adversary and self._adversary:
            logger.info("Step 4: Adversarial question generation")
            adversarial_results = self._run_adversarial_round(
                agent, ground_truth, results, config
            )

        # Step 5: Build report
        all_results = results + adversarial_results
        report = self._build_report(
            all_results=all_results,
            ground_truth=ground_truth,
            config=config,
            learning_time=learning_time,
            grading_time=grading_time,
        )

        # Step 6: Analysis
        logger.info("Step 6: Analyst reviewing results")
        analysis = self._analyst.analyze(report)
        report.memory_stats["analysis"] = analysis.to_dict()
        report.memory_stats["bottleneck"] = analysis.bottleneck_component
        report.memory_stats["num_failure_patterns"] = len(analysis.failure_patterns)

        total_time = time.time() - overall_start
        logger.info(
            "Multi-agent eval complete: overall=%.2f%% in %.1fs",
            report.overall_score * 100,
            total_time,
        )

        return report

    def _init_agents(self, config: EvalConfig) -> None:
        """Initialize all evaluation agents."""
        perspectives = config.grader_perspectives[:self._num_graders]
        self._graders = [
            GraderAgent(perspective=p, model=config.grader_model)
            for p in perspectives
        ]
        logger.info("Initialized %d grader agents: %s", len(self._graders), perspectives)

        if self._enable_adversary:
            self._adversary = AdversaryAgent(model=config.grader_model)

    def _question_and_grade(
        self,
        agent: AgentAdapter,
        questions: list[Question],
    ) -> tuple[list[EvalResult], float]:
        """Ask questions and grade with multi-perspective graders.

        Returns:
            Tuple of (eval_results, total_grading_time)
        """
        results: list[EvalResult] = []
        total_grade_time = 0.0

        for i, question in enumerate(questions):
            logger.info("Question %d/%d: %s", i + 1, len(questions), question.text[:60])

            # Get agent's answer
            try:
                response = agent.answer(question.text)
                answer = response.answer
            except Exception as e:
                logger.warning("Agent failed to answer: %s", e)
                answer = f"Error: {e}"

            # Grade with all perspectives
            grade_start = time.time()
            perspective_grades = [
                grader.grade(question, answer, question.rubric)
                for grader in self._graders
            ]

            # Aggregate grades
            aggregate = GraderAgent.aggregate_grades(
                perspective_grades, question, answer
            )
            grade_time = time.time() - grade_start
            total_grade_time += grade_time

            # Convert to EvalResult
            dimension_scores = [
                DimensionScore(
                    dimension=pg.perspective,
                    score=pg.score,
                    reasoning=pg.reasoning,
                )
                for pg in perspective_grades
            ]

            result = EvalResult(
                question_id=question.question_id,
                question_text=question.text,
                category=question.category,
                expected_answer=question.expected_answer,
                actual_answer=answer if isinstance(answer, str) else str(answer),
                dimensions=dimension_scores,
                overall_score=aggregate.overall_score,
                grading_time_s=grade_time,
            )
            results.append(result)

            logger.info(
                "  Score: %.2f (agreement: %.2f) | %s",
                aggregate.overall_score,
                aggregate.agreement,
                answer[:60] if isinstance(answer, str) else str(answer)[:60],
            )

        return results, total_grade_time

    def _run_adversarial_round(
        self,
        agent: AgentAdapter,
        ground_truth: GroundTruth,
        standard_results: list[EvalResult],
        config: EvalConfig,
    ) -> list[EvalResult]:
        """Generate and evaluate adversarial questions.

        Args:
            agent: The agent to test
            ground_truth: Ground truth from data generation
            standard_results: Results from standard questioning
            config: Evaluation configuration

        Returns:
            List of EvalResult from adversarial questions
        """
        if not self._adversary:
            return []

        # Convert EvalResult to dicts for the adversary
        result_dicts = [
            {
                "question_text": r.question_text,
                "expected_answer": r.expected_answer,
                "actual_answer": r.actual_answer,
                "score": r.overall_score,
                "category": r.category,
            }
            for r in standard_results
        ]

        # Generate adversarial questions
        adv_questions = self._adversary.generate_adversarial_questions(
            ground_truth=ground_truth,
            previous_results=result_dicts,
            num_questions=config.adversarial_questions,
        )

        if not adv_questions:
            logger.info("No adversarial questions generated")
            return []

        logger.info("Generated %d adversarial questions", len(adv_questions))

        # Also generate forgetting probes
        forget_probes = self._adversary.generate_forgetting_probes(
            ground_truth=ground_truth,
            num_questions=max(1, config.adversarial_questions // 3),
        )
        all_adv_questions = adv_questions + forget_probes

        # Question and grade
        results, _ = self._question_and_grade(agent, all_adv_questions)
        return results

    def _build_report(
        self,
        all_results: list[EvalResult],
        ground_truth: GroundTruth,
        config: EvalConfig,
        learning_time: float,
        grading_time: float,
    ) -> EvalReport:
        """Build an EvalReport from all results."""
        # Category breakdown
        categories: dict[str, list[EvalResult]] = {}
        for r in all_results:
            categories.setdefault(r.category, []).append(r)

        breakdown: list[CategoryBreakdown] = []
        for cat, cat_results in sorted(categories.items()):
            scores = [r.overall_score for r in cat_results]
            dim_avgs: dict[str, list[float]] = {}
            for r in cat_results:
                for d in r.dimensions:
                    dim_avgs.setdefault(d.dimension, []).append(d.score)

            breakdown.append(CategoryBreakdown(
                category=cat,
                num_questions=len(cat_results),
                avg_score=sum(scores) / len(scores),
                min_score=min(scores),
                max_score=max(scores),
                dimension_averages={k: sum(v) / len(v) for k, v in dim_avgs.items()},
            ))

        overall = (
            sum(r.overall_score for r in all_results) / len(all_results)
            if all_results else 0.0
        )

        total_facts = sum(len(t.facts) for t in ground_truth.turns)

        # Separate standard vs adversarial counts for metadata
        standard_count = sum(
            1 for r in all_results
            if not r.category.startswith("adversarial") and r.category != "forgetting_probe"
        )
        adversarial_count = len(all_results) - standard_count

        return EvalReport(
            num_turns=config.num_turns,
            num_questions=len(all_results),
            total_facts_delivered=total_facts,
            learning_time_s=learning_time,
            questioning_time_s=grading_time,
            grading_time_s=grading_time,
            overall_score=overall,
            category_breakdown=breakdown,
            results=all_results,
            memory_stats={
                "standard_questions": standard_count,
                "adversarial_questions": adversarial_count,
                "grader_perspectives": [g.perspective for g in self._graders],
                "adversary_enabled": self._enable_adversary,
            },
        )


__all__ = ["EvalCoordinator", "EvalConfig"]
