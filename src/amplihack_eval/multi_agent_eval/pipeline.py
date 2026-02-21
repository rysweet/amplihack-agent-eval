"""Full multi-agent evaluation pipeline with iterative adversarial hardening.

Flow:
1. Standard eval (data -> learn -> question -> grade)
2. Adversary generates targeted questions based on results
3. Agent answers adversarial questions
4. Graders score adversarial answers
5. Analyst produces comprehensive report
6. Optional: loop back to step 2 for iterative hardening

The pipeline wraps EvalCoordinator with support for multiple adversarial
rounds and cross-run comparison.

Philosophy:
- Each round makes the evaluation harder and more targeted
- Analyst tracks whether the agent's weak spots shift between rounds
- All results accumulated for comprehensive final report
- JSON-serializable for logging

Public API:
    PipelineConfig: Configuration for the full pipeline
    PipelineReport: Comprehensive report with all rounds
    MultiAgentEvalPipeline: Full multi-agent evaluation pipeline
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..adapters.base import AgentAdapter
from ..core.runner import EvalReport
from .analyst_agent import AnalystAgent, ComparisonReport
from .coordinator import EvalConfig, EvalCoordinator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full multi-agent evaluation pipeline.

    Args:
        eval_config: Base evaluation configuration
        adversarial_rounds: Number of adversarial hardening rounds (0 = standard only)
        agent_factory: Factory function to create fresh agent instances
            (if None, agent is reused across rounds without reset)
        reset_between_rounds: Whether to reset agent state between rounds
    """

    eval_config: EvalConfig = field(default_factory=EvalConfig)
    adversarial_rounds: int = 1
    agent_factory: Callable[[], AgentAdapter] | None = None
    reset_between_rounds: bool = False


@dataclass
class RoundResult:
    """Results from a single evaluation round."""

    round_number: int
    report: EvalReport
    is_adversarial: bool
    round_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "overall_score": round(self.report.overall_score, 4),
            "num_questions": self.report.num_questions,
            "is_adversarial": self.is_adversarial,
            "round_time_s": round(self.round_time_s, 2),
        }


@dataclass
class PipelineReport:
    """Comprehensive report from the full multi-agent evaluation pipeline.

    Contains all round results, cross-round comparison, and final analysis.
    """

    rounds: list[RoundResult]
    comparison: ComparisonReport | None
    final_overall_score: float
    total_questions_asked: int
    total_time_s: float
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_overall_score": round(self.final_overall_score, 4),
            "total_questions_asked": self.total_questions_asked,
            "total_time_s": round(self.total_time_s, 2),
            "num_rounds": len(self.rounds),
            "rounds": [r.to_dict() for r in self.rounds],
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "config": self.config,
        }


class MultiAgentEvalPipeline:
    """Full multi-agent evaluation pipeline.

    Wraps EvalCoordinator to support multiple adversarial rounds and
    cross-run comparison. Each round generates increasingly targeted
    adversarial questions based on the agent's evolving performance.

    Args:
        grader_agents: Number of grading perspectives (1-3)

    Example::

        pipeline = MultiAgentEvalPipeline(grader_agents=3)
        config = PipelineConfig(
            eval_config=EvalConfig(num_turns=100, num_questions=20),
            adversarial_rounds=2,
        )
        report = pipeline.run(agent, config)
        print(f"Final score: {report.final_overall_score:.2%}")
        print(f"Total questions: {report.total_questions_asked}")
    """

    def __init__(self, grader_agents: int = 3):
        self._num_graders = grader_agents
        self._analyst = AnalystAgent()

    def run(
        self,
        agent: AgentAdapter,
        config: PipelineConfig,
    ) -> PipelineReport:
        """Execute the full evaluation pipeline.

        Args:
            agent: The agent to evaluate (or initial agent if factory provided)
            config: Pipeline configuration

        Returns:
            PipelineReport with all rounds and comparison
        """
        pipeline_start = time.time()
        rounds: list[RoundResult] = []
        all_reports: list[EvalReport] = []
        labels: list[str] = []

        # Round 0: Standard evaluation (with adversary enabled on round 0)
        logger.info("=== Round 0: Standard evaluation ===")
        round_start = time.time()

        coordinator = EvalCoordinator(
            grader_agents=self._num_graders,
            enable_adversary=config.eval_config.enable_adversary,
        )

        report = coordinator.run_eval(agent, config.eval_config)
        round_time = time.time() - round_start

        round_result = RoundResult(
            round_number=0,
            report=report,
            is_adversarial=False,
            round_time_s=round_time,
        )
        rounds.append(round_result)
        all_reports.append(report)
        labels.append("round_0")

        logger.info(
            "Round 0 complete: %.2f%% (%d questions) in %.1fs",
            report.overall_score * 100,
            report.num_questions,
            round_time,
        )

        # Additional adversarial rounds
        for round_num in range(1, config.adversarial_rounds + 1):
            logger.info("=== Round %d: Adversarial hardening ===", round_num)
            round_start = time.time()

            # Optionally get fresh agent
            current_agent = agent
            if config.agent_factory:
                current_agent = config.agent_factory()
            elif config.reset_between_rounds:
                agent.reset()

            # Run with adversary, using results from previous round
            adv_config = EvalConfig(
                num_turns=config.eval_config.num_turns,
                num_questions=config.eval_config.num_questions,
                seed=config.eval_config.seed + round_num,  # Different seed each round
                grader_perspectives=config.eval_config.grader_perspectives,
                enable_adversary=True,
                adversarial_questions=config.eval_config.adversarial_questions,
                grader_model=config.eval_config.grader_model,
            )

            adv_coordinator = EvalCoordinator(
                grader_agents=self._num_graders,
                enable_adversary=True,
            )

            adv_report = adv_coordinator.run_eval(current_agent, adv_config)
            round_time = time.time() - round_start

            round_result = RoundResult(
                round_number=round_num,
                report=adv_report,
                is_adversarial=True,
                round_time_s=round_time,
            )
            rounds.append(round_result)
            all_reports.append(adv_report)
            labels.append(f"round_{round_num}")

            # Close factory-created agents
            if config.agent_factory and current_agent is not agent:
                current_agent.close()

            logger.info(
                "Round %d complete: %.2f%% (%d questions) in %.1fs",
                round_num,
                adv_report.overall_score * 100,
                adv_report.num_questions,
                round_time,
            )

        # Cross-round comparison
        comparison = None
        if len(all_reports) >= 2:
            comparison = self._analyst.compare_reports(all_reports, labels)

        # Compute totals
        total_questions = sum(r.report.num_questions for r in rounds)
        final_score = rounds[-1].report.overall_score
        total_time = time.time() - pipeline_start

        return PipelineReport(
            rounds=rounds,
            comparison=comparison,
            final_overall_score=final_score,
            total_questions_asked=total_questions,
            total_time_s=total_time,
            config={
                "num_turns": config.eval_config.num_turns,
                "num_questions": config.eval_config.num_questions,
                "adversarial_rounds": config.adversarial_rounds,
                "grader_agents": self._num_graders,
            },
        )


__all__ = [
    "MultiAgentEvalPipeline",
    "PipelineConfig",
    "PipelineReport",
    "RoundResult",
]
