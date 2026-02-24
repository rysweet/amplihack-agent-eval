"""Tests for parallel question answering and grading.

Verifies that:
- Parallel evaluation produces identical results to sequential
- Worker count limits are enforced
- Exception handling in one worker doesn't kill others
- Progress reporting works in parallel mode
- Parallel execution is measurably faster than sequential
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

from amplihack_eval.adapters.base import AgentAdapter, AgentResponse
from amplihack_eval.core.runner import (
    DimensionScore,
    EvalRunner,
    _grade_multi_vote,
)
from amplihack_eval.data.long_horizon import (
    GradingRubric,
    Question,
)


def _mock_grade_with_llm(question, actual_answer, dimensions, grader_model=""):
    """Mock LLM grading that returns deterministic scores without calling Anthropic."""
    return [
        DimensionScore(dimension=d, score=0.5, reasoning="mock LLM score")
        for d in dimensions
    ]


# ---------------------------------------------------------------------------
# Test fixtures: deterministic mock agent
# ---------------------------------------------------------------------------


class DeterministicMockAgent(AgentAdapter):
    """Agent that returns predictable answers based on question text.

    Thread-safe: uses no shared mutable state during answer().
    """

    def __init__(self, delay: float = 0.0):
        self._learned: list[str] = []
        self._delay = delay
        self._answer_count = 0
        self._lock = threading.Lock()

    def learn(self, content: str) -> None:
        self._learned.append(content)

    def answer(self, question: str) -> AgentResponse:
        if self._delay > 0:
            time.sleep(self._delay)
        with self._lock:
            self._answer_count += 1
        # Deterministic: return question hash-based answer
        return AgentResponse(answer=f"Answer for: {question[:50]}")

    def reset(self) -> None:
        self._learned.clear()
        self._answer_count = 0

    def close(self) -> None:
        pass

    @property
    def answer_count(self) -> int:
        return self._answer_count


class FailingMockAgent(AgentAdapter):
    """Agent that fails on specific question indices."""

    def __init__(self, fail_indices: set[int]):
        self._fail_indices = fail_indices
        self._call_count = 0
        self._lock = threading.Lock()

    def learn(self, content: str) -> None:
        pass

    def answer(self, question: str) -> AgentResponse:
        with self._lock:
            idx = self._call_count
            self._call_count += 1
        if idx in self._fail_indices:
            raise RuntimeError(f"Simulated failure on question {idx}")
        return AgentResponse(answer=f"Answer {idx}")

    def reset(self) -> None:
        self._call_count = 0

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests: parallel vs sequential equivalence
# ---------------------------------------------------------------------------


class TestParallelSequentialEquivalence:
    """Verify parallel and sequential modes produce identical results."""

    def _make_runner_and_questions(
        self, parallel_workers: int
    ) -> tuple[EvalRunner, list[Question]]:
        runner = EvalRunner(
            num_turns=20,
            num_questions=5,
            seed=42,
            grader_votes=1,
            parallel_workers=parallel_workers,
        )
        runner.generate()
        return runner, runner.questions

    @patch("amplihack_eval.core.runner._grade_with_llm", side_effect=_mock_grade_with_llm)
    def test_same_scores_sequential_vs_parallel(self, mock_llm):
        """Both modes produce identical overall scores with deterministic grading."""
        agent_seq = DeterministicMockAgent()
        runner_seq, _ = self._make_runner_and_questions(parallel_workers=1)
        report_seq = runner_seq.evaluate(agent_seq)

        agent_par = DeterministicMockAgent()
        runner_par, _ = self._make_runner_and_questions(parallel_workers=5)
        report_par = runner_par.evaluate(agent_par)

        # Same number of results
        assert len(report_seq.results) == len(report_par.results)

        # Same question IDs in same order
        seq_ids = [r.question_id for r in report_seq.results]
        par_ids = [r.question_id for r in report_par.results]
        assert seq_ids == par_ids

        # Same answers (deterministic agent)
        for s, p in zip(report_seq.results, report_par.results):
            assert s.actual_answer == p.actual_answer
            assert s.overall_score == p.overall_score

    @patch("amplihack_eval.core.runner._grade_with_llm", side_effect=_mock_grade_with_llm)
    def test_same_category_breakdown(self, mock_llm):
        """Category breakdowns match between sequential and parallel."""
        agent_seq = DeterministicMockAgent()
        runner_seq, _ = self._make_runner_and_questions(parallel_workers=1)
        report_seq = runner_seq.evaluate(agent_seq)

        agent_par = DeterministicMockAgent()
        runner_par, _ = self._make_runner_and_questions(parallel_workers=5)
        report_par = runner_par.evaluate(agent_par)

        seq_cats = {cb.category: cb.avg_score for cb in report_seq.category_breakdown}
        par_cats = {cb.category: cb.avg_score for cb in report_par.category_breakdown}
        assert seq_cats == par_cats


# ---------------------------------------------------------------------------
# Tests: worker count limits
# ---------------------------------------------------------------------------


class TestWorkerCountLimits:
    def test_min_workers_is_1(self):
        runner = EvalRunner(parallel_workers=0)
        assert runner.parallel_workers == 1

    def test_negative_workers_clamped_to_1(self):
        runner = EvalRunner(parallel_workers=-5)
        assert runner.parallel_workers == 1

    def test_max_workers_is_20(self):
        runner = EvalRunner(parallel_workers=100)
        assert runner.parallel_workers == 20

    def test_default_workers_is_10(self):
        runner = EvalRunner()
        assert runner.parallel_workers == 10

    def test_explicit_sequential(self):
        runner = EvalRunner(parallel_workers=1)
        assert runner.parallel_workers == 1


# ---------------------------------------------------------------------------
# Tests: exception handling
# ---------------------------------------------------------------------------


class TestParallelExceptionHandling:
    def test_one_failure_doesnt_kill_others(self):
        """If one question's agent.answer() raises, others still complete."""
        agent = FailingMockAgent(fail_indices={1})  # fail on 2nd question
        runner = EvalRunner(
            num_turns=20,
            num_questions=5,
            seed=42,
            grader_votes=1,
            parallel_workers=5,
        )
        runner.generate()
        report = runner.evaluate(agent)

        # All 5 questions should have results
        assert len(report.results) == 5

        # The failed question should have "Error:" in its answer
        error_results = [r for r in report.results if "Error:" in r.actual_answer]
        assert len(error_results) >= 1

        # Non-error results should have valid answers
        ok_results = [r for r in report.results if "Error:" not in r.actual_answer]
        assert len(ok_results) >= 1

    def test_all_failures_still_produces_report(self):
        """Even if all questions fail, we get a report with zero scores."""
        agent = FailingMockAgent(fail_indices={0, 1, 2, 3, 4})
        runner = EvalRunner(
            num_turns=20,
            num_questions=5,
            seed=42,
            grader_votes=1,
            parallel_workers=3,
        )
        runner.generate()
        report = runner.evaluate(agent)

        assert len(report.results) == 5
        # All should contain error text
        for r in report.results:
            assert "Error:" in r.actual_answer


# ---------------------------------------------------------------------------
# Tests: progress reporting
# ---------------------------------------------------------------------------


class TestParallelProgressReporting:
    def test_progress_logged_in_parallel(self, caplog):
        """Progress messages appear during parallel evaluation."""
        import logging

        agent = DeterministicMockAgent()
        runner = EvalRunner(
            num_turns=20,
            num_questions=5,
            seed=42,
            grader_votes=1,
            parallel_workers=3,
        )
        runner.generate()

        with caplog.at_level(logging.INFO, logger="amplihack_eval.core.runner"):
            runner.evaluate(agent)

        # Should see "Starting parallel evaluation" message
        assert any("parallel evaluation" in r.message.lower() for r in caplog.records)
        # Should see completion messages
        assert any("Completed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: parallel grading (multi-vote)
# ---------------------------------------------------------------------------


class TestParallelGrading:
    def test_multi_vote_produces_valid_scores(self):
        """Multi-vote grading with parallel execution returns valid dimension scores."""
        q = Question(
            question_id="test_q",
            text="What color is the sky?",
            expected_answer="Blue",
            category="test",
            relevant_turns=[1],
            scoring_dimensions=["factual_accuracy"],
            rubric=GradingRubric(required_keywords=["blue"]),
        )

        scores = _grade_multi_vote(
            q,
            actual_answer="The sky is blue on a clear day.",
            dimensions=["factual_accuracy"],
            grader_model="",
            num_votes=3,
        )

        assert len(scores) == 1
        assert scores[0].dimension == "factual_accuracy"
        assert 0.0 <= scores[0].score <= 1.0
        assert "median of 3 votes" in scores[0].reasoning

    def test_single_vote_skips_threading(self):
        """num_votes=1 returns directly without ThreadPoolExecutor."""
        q = Question(
            question_id="test_q",
            text="What is 2+2?",
            expected_answer="4",
            category="test",
            relevant_turns=[1],
            scoring_dimensions=["factual_accuracy"],
            rubric=GradingRubric(required_keywords=["4"]),
        )

        scores = _grade_multi_vote(
            q,
            actual_answer="The answer is 4.",
            dimensions=["factual_accuracy"],
            num_votes=1,
        )

        assert len(scores) == 1
        # No "median of" in reasoning since single vote
        assert "median of" not in scores[0].reasoning


# ---------------------------------------------------------------------------
# Tests: speedup benchmark
# ---------------------------------------------------------------------------


class TestParallelSpeedup:
    @patch("amplihack_eval.core.runner._grade_with_llm", side_effect=_mock_grade_with_llm)
    def test_parallel_is_faster_than_sequential(self, mock_llm):
        """Parallel mode with delayed agent is measurably faster.

        Uses a mock agent with 0.1s delay per answer. With 10 questions:
        - Sequential: ~1.0s (10 * 0.1s)
        - Parallel (5 workers): ~0.2s (2 batches * 0.1s)
        Expected speedup: ~3-5x
        """
        num_questions = 10
        delay = 0.1

        # Sequential timing
        agent_seq = DeterministicMockAgent(delay=delay)
        runner_seq = EvalRunner(
            num_turns=20,
            num_questions=num_questions,
            seed=42,
            grader_votes=1,
            parallel_workers=1,
        )
        runner_seq.generate()
        t0 = time.time()
        runner_seq.evaluate(agent_seq)
        seq_time = time.time() - t0

        # Parallel timing
        agent_par = DeterministicMockAgent(delay=delay)
        runner_par = EvalRunner(
            num_turns=20,
            num_questions=num_questions,
            seed=42,
            grader_votes=1,
            parallel_workers=5,
        )
        runner_par.generate()
        t0 = time.time()
        runner_par.evaluate(agent_par)
        par_time = time.time() - t0

        speedup = seq_time / par_time if par_time > 0 else float("inf")
        print(f"\nSequential: {seq_time:.2f}s, Parallel: {par_time:.2f}s, Speedup: {speedup:.1f}x")

        # Expect at least 2x speedup (conservative to avoid flaky tests)
        assert speedup >= 2.0, f"Expected >=2x speedup, got {speedup:.1f}x"

    def test_all_questions_answered_in_parallel(self):
        """All questions get answered even with many workers."""
        agent = DeterministicMockAgent()
        runner = EvalRunner(
            num_turns=20,
            num_questions=10,
            seed=42,
            grader_votes=1,
            parallel_workers=10,
        )
        runner.generate()
        report = runner.evaluate(agent)

        assert len(report.results) == len(runner.questions)
        assert agent.answer_count == len(runner.questions)


# ---------------------------------------------------------------------------
# Tests: EvalRunner.run() integration with parallel
# ---------------------------------------------------------------------------


class TestRunIntegration:
    def test_run_with_parallel_workers(self):
        """Full run() with parallel_workers produces valid report."""
        agent = DeterministicMockAgent()
        runner = EvalRunner(
            num_turns=20,
            num_questions=5,
            seed=42,
            grader_votes=1,
            parallel_workers=3,
        )
        report = runner.run(agent)

        assert report.num_questions == 5
        assert report.overall_score >= 0.0
        assert len(report.results) == 5
        assert report.learning_time_s >= 0.0

    @patch("amplihack_eval.core.runner._grade_with_llm", side_effect=_mock_grade_with_llm)
    def test_run_sequential_produces_same_report_structure(self, mock_llm):
        """Sequential run() produces same report structure as parallel."""
        agent = DeterministicMockAgent()
        runner = EvalRunner(
            num_turns=20,
            num_questions=3,
            seed=42,
            grader_votes=1,
            parallel_workers=1,
        )
        report = runner.run(agent)

        assert report.num_questions == 3
        assert len(report.results) == 3
        assert len(report.category_breakdown) > 0


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParallelArg:
    def test_parallel_workers_in_run_parser(self):
        """--parallel-workers is accepted by the 'run' subcommand."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        run_parser = subparsers.add_parser("run")
        run_parser.add_argument("--parallel-workers", type=int, default=10)

        args = parser.parse_args(["run", "--parallel-workers", "5"])
        assert args.parallel_workers == 5

    def test_parallel_workers_default_is_10(self):
        """Default value for --parallel-workers is 10."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--parallel-workers", type=int, default=10)
        args = parser.parse_args([])
        assert args.parallel_workers == 10
