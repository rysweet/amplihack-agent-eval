"""Evaluation runner for long-horizon memory stress tests.

Philosophy:
- 1000-turn dialogue tests memory at scale (not just short-horizon recall)
- Deterministic data generation, reproducible results
- Hybrid deterministic + LLM grading: rubric keywords scored without LLM,
  judgment dimensions (confidence, source attribution) use LLM
- Multi-vote grading for stability: grade N times, take median per dimension
- Agent-agnostic: works with any AgentAdapter implementation
- Parallel execution: question answering and grading run concurrently for ~6x speedup

Public API:
    EvalRunner: Main evaluation class (renamed from LongHorizonMemoryEval)
    EvalResult: Per-question result with scores
    EvalReport: Aggregate report with breakdown by category

Usage:
    amplihack-eval run --turns 100 --questions 20
    amplihack-eval run --turns 1000 --questions 100
    amplihack-eval run --turns 100 --questions 20 --parallel-workers 10
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from ..adapters.base import AgentAdapter
from ..data.long_horizon import (
    GradingRubric,
    GroundTruth,
    Question,
    generate_dialogue,
    generate_questions,
)

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """Score on a single dimension for a single question."""

    dimension: str
    score: float  # 0.0 to 1.0
    reasoning: str = ""


@dataclass
class EvalResult:
    """Result for a single question."""

    question_id: str
    question_text: str
    category: str
    expected_answer: str
    actual_answer: str
    dimensions: list[DimensionScore]
    overall_score: float  # Average of dimension scores
    grading_time_s: float = 0.0


@dataclass
class CategoryBreakdown:
    """Aggregate scores for a question category."""

    category: str
    num_questions: int
    avg_score: float
    min_score: float
    max_score: float
    dimension_averages: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Complete evaluation report."""

    num_turns: int
    num_questions: int
    total_facts_delivered: int
    learning_time_s: float
    questioning_time_s: float
    grading_time_s: float
    overall_score: float
    category_breakdown: list[CategoryBreakdown]
    results: list[EvalResult]
    memory_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "num_turns": self.num_turns,
            "num_questions": self.num_questions,
            "total_facts_delivered": self.total_facts_delivered,
            "learning_time_s": round(self.learning_time_s, 2),
            "questioning_time_s": round(self.questioning_time_s, 2),
            "grading_time_s": round(self.grading_time_s, 2),
            "overall_score": round(self.overall_score, 4),
            "category_breakdown": [
                {
                    "category": cb.category,
                    "num_questions": cb.num_questions,
                    "avg_score": round(cb.avg_score, 4),
                    "min_score": round(cb.min_score, 4),
                    "max_score": round(cb.max_score, 4),
                    "dimension_averages": {
                        k: round(v, 4) for k, v in cb.dimension_averages.items()
                    },
                }
                for cb in self.category_breakdown
            ],
            "results": [
                {
                    "question_id": r.question_id,
                    "question_text": r.question_text,
                    "category": r.category,
                    "expected_answer": r.expected_answer,
                    "actual_answer": r.actual_answer[:500],
                    "overall_score": round(r.overall_score, 4),
                    "dimensions": [
                        {
                            "dimension": d.dimension,
                            "score": round(d.score, 4),
                            "reasoning": d.reasoning[:200],
                        }
                        for d in r.dimensions
                    ],
                }
                for r in self.results
            ],
            "memory_stats": self.memory_stats,
        }


# Scoring dimensions
ALL_DIMENSIONS = [
    "factual_accuracy",
    "specificity",
    "temporal_awareness",
    "source_attribution",
    "confidence_calibration",
]


# Dimensions that can be graded deterministically when a rubric is present
_DETERMINISTIC_DIMENSIONS = {"factual_accuracy", "specificity"}

# Dimensions that always require LLM judgment
_LLM_ONLY_DIMENSIONS = {"confidence_calibration", "source_attribution", "temporal_awareness"}


def _deterministic_grade(
    rubric: GradingRubric,
    actual_answer: str,
    dimensions: list[str],
) -> dict[str, DimensionScore]:
    """Grade deterministic dimensions using regex/string matching against rubric.

    Returns a dict mapping dimension name -> DimensionScore for every dimension
    in `dimensions` that can be scored deterministically. Dimensions not in
    _DETERMINISTIC_DIMENSIONS are skipped (returned dict won't contain them).

    Scoring logic:
    - Start with keyword match ratio (required_keywords found / total required)
    - Bonus +0.1 for each acceptable_paraphrase found (capped at 1.0)
    - Score 0.0 if any incorrect_pattern matches
    """
    answer_lower = actual_answer.lower()
    scores: dict[str, DimensionScore] = {}

    for dim in dimensions:
        if dim not in _DETERMINISTIC_DIMENSIONS:
            continue

        # Check incorrect patterns first -- instant 0
        if rubric.incorrect_patterns:
            found_incorrect = any(
                re.search(re.escape(pat.lower()), answer_lower) for pat in rubric.incorrect_patterns
            )
            if found_incorrect:
                scores[dim] = DimensionScore(
                    dimension=dim,
                    score=0.0,
                    reasoning="Answer contains incorrect pattern from rubric",
                )
                continue

        # Keyword matching
        matched = 0
        if rubric.required_keywords:
            matched = sum(
                1
                for kw in rubric.required_keywords
                if re.search(re.escape(kw.lower()), answer_lower)
            )
            ratio = matched / len(rubric.required_keywords)
        else:
            ratio = 0.5  # No keywords = neutral

        # Paraphrase bonus
        paraphrase_hits = 0
        if rubric.acceptable_paraphrases:
            paraphrase_hits = sum(
                1
                for p in rubric.acceptable_paraphrases
                if re.search(re.escape(p.lower()), answer_lower)
            )
            ratio = min(1.0, ratio + paraphrase_hits * 0.1)

        reasoning_parts = []
        if rubric.required_keywords:
            reasoning_parts.append(
                f"Matched {matched}/{len(rubric.required_keywords)} required keywords"
            )
        if rubric.acceptable_paraphrases and paraphrase_hits:
            reasoning_parts.append(f"+{paraphrase_hits} paraphrase bonus")

        scores[dim] = DimensionScore(
            dimension=dim,
            score=round(ratio, 4),
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Deterministic score",
        )

    return scores


def _grade_hybrid(
    question: Question,
    actual_answer: str,
    dimensions: list[str],
    grader_model: str = "",
) -> list[DimensionScore]:
    """Hybrid grading: deterministic for rubric-compatible dimensions, LLM for the rest.

    If the question has a rubric, factual_accuracy and specificity are scored
    deterministically. Remaining dimensions (temporal_awareness,
    source_attribution, confidence_calibration) are sent to the LLM.
    If no rubric exists, all dimensions go to LLM (backward compatible).
    """
    if not question.rubric:
        return _grade_with_llm(question, actual_answer, dimensions, grader_model)

    # Score deterministic dimensions
    det_scores = _deterministic_grade(question.rubric, actual_answer, dimensions)

    # Remaining dimensions need LLM
    remaining = [d for d in dimensions if d not in det_scores]

    if remaining:
        llm_scores = _grade_with_llm(question, actual_answer, remaining, grader_model)
        llm_map = {s.dimension: s for s in llm_scores}
    else:
        llm_map = {}

    # Merge in original order
    result: list[DimensionScore] = []
    for dim in dimensions:
        if dim in det_scores:
            result.append(det_scores[dim])
        elif dim in llm_map:
            result.append(llm_map[dim])
        else:
            result.append(DimensionScore(dimension=dim, score=0.0, reasoning="Not graded"))

    # Apply dimension weights from rubric if provided
    if question.rubric.dimension_weights:
        for ds in result:
            if ds.dimension in question.rubric.dimension_weights:
                w = question.rubric.dimension_weights[ds.dimension]
                ds.score = round(ds.score * w, 4)
                ds.reasoning += f" (weight={w})"

    return result


def _grade_multi_vote(
    question: Question,
    actual_answer: str,
    dimensions: list[str],
    grader_model: str = "",
    num_votes: int = 3,
) -> list[DimensionScore]:
    """Grade with multiple votes and take median score per dimension.

    Calls _grade_hybrid N times concurrently and returns the median score per
    dimension. For N=1, this is equivalent to a single call (no overhead).
    When N>1, votes are submitted to a ThreadPoolExecutor for parallel execution.
    """
    if num_votes <= 1:
        return _grade_hybrid(question, actual_answer, dimensions, grader_model)

    # Collect all vote results using parallel execution
    all_votes: dict[str, list[float]] = {d: [] for d in dimensions}
    all_reasoning: dict[str, list[str]] = {d: [] for d in dimensions}

    def _do_vote() -> list[DimensionScore]:
        return _grade_hybrid(question, actual_answer, dimensions, grader_model)

    with ThreadPoolExecutor(max_workers=num_votes) as executor:
        futures = [executor.submit(_do_vote) for _ in range(num_votes)]
        for future in futures:
            try:
                scores = future.result()
                for ds in scores:
                    all_votes[ds.dimension].append(ds.score)
                    all_reasoning[ds.dimension].append(ds.reasoning)
            except Exception as e:
                logger.warning("Grading vote failed: %s", e)
                # Add zero scores for failed votes so median is still computable
                for dim in dimensions:
                    all_votes[dim].append(0.0)
                    all_reasoning[dim].append(f"Vote failed: {e}")

    # Take median per dimension
    result: list[DimensionScore] = []
    for dim in dimensions:
        votes = all_votes[dim]
        median_score = statistics.median(votes) if votes else 0.0
        # Pick reasoning from the vote closest to median
        best_idx = min(range(len(votes)), key=lambda i: abs(votes[i] - median_score))
        reasoning = all_reasoning[dim][best_idx]
        reasoning += f" [median of {len(votes)} votes: {', '.join(f'{v:.2f}' for v in votes)}]"

        result.append(
            DimensionScore(
                dimension=dim,
                score=round(median_score, 4),
                reasoning=reasoning,
            )
        )

    return result


def _grade_with_llm(
    question: Question,
    actual_answer: str,
    dimensions: list[str],
    grader_model: str = "",
) -> list[DimensionScore]:
    """Grade an answer on multiple dimensions using LLM.

    Args:
        question: The question with expected answer
        actual_answer: Agent's actual answer
        dimensions: Which dimensions to score
        grader_model: Model to use for grading

    Returns:
        List of DimensionScore for each requested dimension
    """
    import anthropic  # type: ignore[import-untyped]

    if not grader_model:
        grader_model = os.environ.get("GRADER_MODEL", "claude-sonnet-4-5-20250929")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Return zero scores if no API key
        return [DimensionScore(dimension=d, score=0.0, reasoning="No API key") for d in dimensions]

    client = anthropic.Anthropic(api_key=api_key)

    dimension_descriptions = {
        "factual_accuracy": "Is the answer factually correct? Does it match the expected answer on key facts?",
        "specificity": "Does the answer include specific details (names, numbers, dates)?",
        "temporal_awareness": "Does the answer correctly distinguish current vs historical values?",
        "source_attribution": "Does the answer correctly attribute information to its source?",
        "confidence_calibration": "Does the answer express appropriate confidence/uncertainty?",
    }

    dims_text = "\n".join(
        f"- {d}: {dimension_descriptions.get(d, 'General quality')}" for d in dimensions
    )

    prompt = f"""Grade this answer on the following dimensions (0.0 to 1.0 each):

{dims_text}

Question: {question.text}
Category: {question.category}

Expected Answer: {question.expected_answer}

Actual Answer: {actual_answer}

Return ONLY a JSON object mapping each dimension to a score and reasoning:
{{
  "scores": {{
    "factual_accuracy": {{"score": 0.85, "reasoning": "..."}},
    ...
  }}
}}

Scoring guide:
- 1.0: Perfect or semantically equivalent
- 0.8-0.9: Correct main points, minor differences
- 0.5-0.7: Partially correct, missing key details
- 0.2-0.4: Some relevant content, significant gaps
- 0.0-0.1: Incorrect or irrelevant
"""

    try:
        message = client.messages.create(
            model=grader_model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text.strip()

        # Parse JSON from response
        result = _extract_json(response_text)
        scores_dict = result.get("scores", result)

        dimension_scores = []
        for dim in dimensions:
            if dim in scores_dict:
                entry = scores_dict[dim]
                if isinstance(entry, dict):
                    dimension_scores.append(
                        DimensionScore(
                            dimension=dim,
                            score=float(entry.get("score", 0.0)),
                            reasoning=str(entry.get("reasoning", "")),
                        )
                    )
                elif isinstance(entry, (int, float)):
                    dimension_scores.append(
                        DimensionScore(
                            dimension=dim,
                            score=float(entry),
                            reasoning="",
                        )
                    )
                else:
                    dimension_scores.append(
                        DimensionScore(
                            dimension=dim,
                            score=0.0,
                            reasoning="Parse error",
                        )
                    )
            else:
                dimension_scores.append(
                    DimensionScore(
                        dimension=dim,
                        score=0.0,
                        reasoning="Not graded",
                    )
                )

        return dimension_scores

    except Exception as e:
        logger.warning("Grading failed for %s: %s", question.question_id, e)
        return [DimensionScore(dimension=d, score=0.0, reasoning=f"Error: {e}") for d in dimensions]


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response text."""
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


class EvalRunner:
    """Long-horizon dialogue memory stress test.

    Generates structured dialogue content, feeds it to an agent's learn method,
    then quizzes the agent on details from various points in the conversation.

    Grading uses a hybrid approach:
    - Deterministic: factual_accuracy and specificity scored via rubric keywords
    - LLM: temporal_awareness, source_attribution, confidence_calibration
    - Multi-vote: each question graded N times, median score per dimension

    Args:
        num_turns: Number of dialogue turns (default 1000)
        num_questions: Number of quiz questions (default 100)
        seed: Random seed for reproducibility (default 42)
        grader_votes: Number of grading votes per question (default 3)
        parallel_workers: Number of parallel workers for question answering
            and grading. Set to 1 for sequential execution (default 10, max 20).

    Example::

        from amplihack_eval import EvalRunner, AgentAdapter

        class MyAgent(AgentAdapter):
            ...

        agent = MyAgent()
        runner = EvalRunner(num_turns=100, num_questions=20, parallel_workers=10)
        report = runner.run(agent)
        print(f"Overall score: {report.overall_score:.2%}")
    """

    def __init__(
        self,
        num_turns: int = 1000,
        num_questions: int = 100,
        seed: int = 42,
        grader_votes: int = 3,
        parallel_workers: int = 10,
    ):
        self.num_turns = num_turns
        self.num_questions = num_questions
        self.seed = seed
        self.grader_votes = max(1, grader_votes)
        self.parallel_workers = max(1, min(20, parallel_workers))
        self.ground_truth: GroundTruth | None = None
        self.questions: list[Question] = []

    def generate(self) -> tuple[GroundTruth, list[Question]]:
        """Generate dialogue and questions.

        Returns:
            Tuple of (GroundTruth, list[Question])
        """
        self.ground_truth = generate_dialogue(num_turns=self.num_turns, seed=self.seed)
        self.questions = generate_questions(self.ground_truth, num_questions=self.num_questions)
        return self.ground_truth, self.questions

    def run_dialogue(self, agent: AgentAdapter, ground_truth: GroundTruth | None = None) -> float:
        """Feed all turns to the agent's learning method.

        Args:
            agent: Agent implementing AgentAdapter interface
            ground_truth: Override ground truth (uses self.ground_truth if None)

        Returns:
            Time taken in seconds
        """
        gt = ground_truth or self.ground_truth
        if gt is None:
            raise ValueError("Must call generate() first or pass ground_truth")

        start = time.time()
        total = len(gt.turns)

        for i, turn in enumerate(gt.turns):
            if not turn.content or not turn.content.strip():
                continue

            try:
                agent.learn(turn.content)
            except Exception as e:
                logger.warning("Failed to learn turn %d: %s", i, e)

            if (i + 1) % 50 == 0 or i == total - 1:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "Turn %d/%d (%.1f turns/s) - block: %s",
                    i + 1,
                    total,
                    rate,
                    turn.block_name,
                )

        elapsed = time.time() - start
        logger.info("Dialogue complete: %d turns in %.1fs", total, elapsed)
        return elapsed

    def evaluate(
        self,
        agent: AgentAdapter,
        questions: list[Question] | None = None,
        grader_model: str = "",
    ) -> EvalReport:
        """Ask questions and grade responses.

        When parallel_workers > 1, questions are answered and graded concurrently
        using a ThreadPoolExecutor. Results are collected in original question order
        to ensure deterministic output regardless of worker count.

        Args:
            agent: Agent implementing AgentAdapter interface
            questions: Override questions (uses self.questions if None)
            grader_model: Model for LLM grading

        Returns:
            EvalReport with all results
        """
        qs = questions or self.questions
        if not qs:
            raise ValueError("Must call generate() first or pass questions")

        q_start = time.time()

        if self.parallel_workers <= 1:
            results = self._evaluate_sequential(qs, agent, grader_model)
        else:
            results = self._evaluate_parallel(qs, agent, grader_model)

        q_elapsed = time.time() - q_start
        grade_total = sum(r.grading_time_s for r in results)

        # Build category breakdown
        categories: dict[str, list[EvalResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        breakdown = []
        for cat, cat_results in sorted(categories.items()):
            scores = [r.overall_score for r in cat_results]
            dim_avgs: dict[str, list[float]] = {}
            for r in cat_results:
                for d in r.dimensions:
                    dim_avgs.setdefault(d.dimension, []).append(d.score)

            breakdown.append(
                CategoryBreakdown(
                    category=cat,
                    num_questions=len(cat_results),
                    avg_score=sum(scores) / len(scores),
                    min_score=min(scores),
                    max_score=max(scores),
                    dimension_averages={k: sum(v) / len(v) for k, v in dim_avgs.items()},
                )
            )

        # Get memory stats if available
        mem_stats: dict[str, Any] = {}

        # Count facts delivered
        total_facts = sum(
            len(t.facts) for t in (self.ground_truth.turns if self.ground_truth else [])
        )

        overall_score = sum(r.overall_score for r in results) / len(results) if results else 0.0

        return EvalReport(
            num_turns=self.num_turns,
            num_questions=len(results),
            total_facts_delivered=total_facts,
            learning_time_s=0.0,  # Set by caller
            questioning_time_s=q_elapsed,
            grading_time_s=grade_total,
            overall_score=overall_score,
            category_breakdown=breakdown,
            results=results,
            memory_stats=mem_stats,
        )

    def _evaluate_sequential(
        self,
        qs: list[Question],
        agent: AgentAdapter,
        grader_model: str,
    ) -> list[EvalResult]:
        """Evaluate questions sequentially (parallel_workers=1)."""
        results: list[EvalResult] = []

        for i, q in enumerate(qs):
            logger.info("Question %d/%d: %s", i + 1, len(qs), q.text[:60])
            result = self._answer_and_grade_one(i, q, agent, grader_model, len(qs))
            results.append(result)

        return results

    def _evaluate_parallel(
        self,
        qs: list[Question],
        agent: AgentAdapter,
        grader_model: str,
    ) -> list[EvalResult]:
        """Evaluate questions in parallel using ThreadPoolExecutor.

        Each worker answers one question and grades the response. Results are
        collected into an ordered list matching the original question order.
        A threading lock protects the progress counter for clean log output.
        """
        total = len(qs)
        # Pre-allocate result slots to preserve order
        results: list[EvalResult | None] = [None] * total
        completed = [0]  # mutable counter for progress reporting
        lock = threading.Lock()

        logger.info(
            "Starting parallel evaluation: %d questions, %d workers",
            total,
            self.parallel_workers,
        )

        def _worker(idx: int, q: Question) -> None:
            result = self._answer_and_grade_one(idx, q, agent, grader_model, total)
            results[idx] = result
            with lock:
                completed[0] += 1
                logger.info(
                    "Completed %d/%d questions (%.0f%%)",
                    completed[0],
                    total,
                    completed[0] / total * 100,
                )

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(_worker, i, q): i for i, q in enumerate(qs)
            }
            # Wait for all futures and propagate exceptions from workers
            for future in as_completed(futures):
                exc = future.exception()
                if exc is not None:
                    idx = futures[future]
                    logger.warning("Worker for question %d failed: %s", idx + 1, exc)
                    # Create a zero-score result for the failed question
                    q = qs[idx]
                    results[idx] = EvalResult(
                        question_id=q.question_id,
                        question_text=q.text,
                        category=q.category,
                        expected_answer=q.expected_answer,
                        actual_answer=f"Error: {exc}",
                        dimensions=[
                            DimensionScore(dimension=d, score=0.0, reasoning=f"Worker error: {exc}")
                            for d in (q.scoring_dimensions or ["factual_accuracy"])
                        ],
                        overall_score=0.0,
                        grading_time_s=0.0,
                    )

        # Filter out any remaining None slots (should not happen)
        return [r for r in results if r is not None]

    def _answer_and_grade_one(
        self,
        idx: int,
        q: Question,
        agent: AgentAdapter,
        grader_model: str,
        total: int,
    ) -> EvalResult:
        """Answer a single question and grade the response.

        Thread-safe: uses only the agent's answer() method and stateless grading
        functions. The agent implementation must be thread-safe for parallel use.
        """
        logger.info("Question %d/%d: %s", idx + 1, total, q.text[:60])

        # Get agent's answer
        try:
            response = agent.answer(q.text)
            answer = response.answer
        except Exception as e:
            logger.warning("Agent failed to answer: %s", e)
            answer = f"Error: {e}"

        # Grade the answer (hybrid deterministic + LLM, with multi-vote)
        grade_start = time.time()
        dimensions = q.scoring_dimensions or ["factual_accuracy"]
        dim_scores = _grade_multi_vote(
            q, answer, dimensions, grader_model, num_votes=self.grader_votes
        )
        grade_time = time.time() - grade_start

        # Compute overall score as average of dimension scores
        overall = sum(d.score for d in dim_scores) / len(dim_scores) if dim_scores else 0.0

        result = EvalResult(
            question_id=q.question_id,
            question_text=q.text,
            category=q.category,
            expected_answer=q.expected_answer,
            actual_answer=answer if isinstance(answer, str) else str(answer),
            dimensions=dim_scores,
            overall_score=overall,
            grading_time_s=grade_time,
        )

        logger.info(
            "  Score: %.2f | Answer: %s",
            overall,
            (answer[:80] if isinstance(answer, str) else str(answer)[:80]) + "...",
        )

        return result

    def run(self, agent: AgentAdapter, grader_model: str = "") -> EvalReport:
        """Run the complete evaluation: generate, learn, quiz, grade.

        Args:
            agent: Agent implementing AgentAdapter interface
            grader_model: Model for LLM grading

        Returns:
            Complete EvalReport
        """
        logger.info(
            "Starting long-horizon memory eval: %d turns, %d questions",
            self.num_turns,
            self.num_questions,
        )

        # Step 1: Generate data
        self.generate()
        logger.info(
            "Generated %d turns, %d questions",
            len(self.ground_truth.turns) if self.ground_truth else 0,
            len(self.questions),
        )

        # Step 2: Feed dialogue to agent
        learning_time = self.run_dialogue(agent)

        # Step 3: Quiz and grade
        report = self.evaluate(agent, grader_model=grader_model)
        report.learning_time_s = learning_time

        logger.info(
            "Evaluation complete: overall=%.2f%%, learning=%.1fs, grading=%.1fs",
            report.overall_score * 100,
            report.learning_time_s,
            report.grading_time_s,
        )

        return report

    def run_skip_learning(
        self,
        agent: AgentAdapter,
        load_db_path: str,
        grader_model: str = "",
    ) -> EvalReport:
        """Run evaluation with a pre-built memory DB, skipping the learning phase.

        The agent must already have its memory loaded from load_db_path (done by
        the adapter's storage_path parameter). This method only generates questions
        and runs the quiz/grading phase.

        Args:
            agent: Agent with pre-loaded memory DB
            load_db_path: Path to the pre-built DB (for logging)
            grader_model: Model for LLM grading

        Returns:
            Complete EvalReport with learning_time_s = 0.0
        """
        logger.info(
            "Starting skip-learning eval: %d turns (questions only), %d questions, db=%s",
            self.num_turns,
            self.num_questions,
            load_db_path,
        )

        # Generate questions (needs ground truth for expected answers)
        self.generate()
        logger.info(
            "Generated %d questions (learning skipped, using pre-built DB)",
            len(self.questions),
        )

        # Quiz and grade (no learning phase)
        report = self.evaluate(agent, grader_model=grader_model)
        report.learning_time_s = 0.0

        logger.info(
            "Skip-learning eval complete: overall=%.2f%%, grading=%.1fs",
            report.overall_score * 100,
            report.grading_time_s,
        )

        return report


def print_report(report: EvalReport) -> None:
    """Print a human-readable summary of the evaluation report."""
    print("\n" + "=" * 70)
    print("LONG-HORIZON MEMORY EVALUATION REPORT")
    print("=" * 70)
    print(f"Turns: {report.num_turns} | Questions: {report.num_questions}")
    print(f"Facts delivered: {report.total_facts_delivered}")
    print(f"Learning time: {report.learning_time_s:.1f}s")
    print(f"Question+Grading time: {report.questioning_time_s:.1f}s")
    print(f"\nOVERALL SCORE: {report.overall_score:.2%}")
    print()

    print("CATEGORY BREAKDOWN:")
    print("-" * 70)
    print(f"{'Category':<25} {'Avg':>8} {'Min':>8} {'Max':>8} {'Count':>6}")
    print("-" * 70)
    for cb in report.category_breakdown:
        print(
            f"{cb.category:<25} {cb.avg_score:>7.2%} {cb.min_score:>7.2%} "
            f"{cb.max_score:>7.2%} {cb.num_questions:>6}"
        )
    print("-" * 70)

    print("\nDIMENSION AVERAGES BY CATEGORY:")
    for cb in report.category_breakdown:
        if cb.dimension_averages:
            dims = ", ".join(f"{k}: {v:.2%}" for k, v in sorted(cb.dimension_averages.items()))
            print(f"  {cb.category}: {dims}")

    print("\nMEMORY STATS:")
    for k, v in report.memory_stats.items():
        print(f"  {k}: {v}")

    # Show worst-performing questions
    print("\nWORST 5 QUESTIONS:")
    sorted_results = sorted(report.results, key=lambda r: r.overall_score)
    for r in sorted_results[:5]:
        print(f"  [{r.overall_score:.2%}] {r.question_text[:60]}")
        print(f"    Expected: {r.expected_answer[:80]}")
        print(f"    Got: {r.actual_answer[:80]}")
        print()


# ---------------------------------------------------------------------------
# YAML-driven level runner
# ---------------------------------------------------------------------------


@dataclass
class LevelResult:
    """Result of running a single YAML-defined level."""

    level_id: str
    level_name: str
    passed: bool
    overall_score: float
    pass_threshold: float
    results: list[EvalResult]
    category_breakdown: list[CategoryBreakdown]


@dataclass
class SuiteResult:
    """Result of running multiple YAML-defined levels."""

    level_results: list[LevelResult]
    overall_score: float
    passed_count: int
    total_count: int
    skipped: list[str]  # level IDs skipped due to failed prerequisites


def run_level(
    level_id: str,
    agent: AgentAdapter,
    grader_model: str = "",
) -> LevelResult:
    """Run a single YAML-defined evaluation level.

    Loads the level from YAML, feeds articles to the agent, asks questions,
    and grades the answers using the level's scoring configuration.

    Args:
        level_id: Level identifier (e.g. "L01")
        agent: Agent implementing AgentAdapter interface
        grader_model: Model for LLM grading (empty = env default)

    Returns:
        LevelResult with per-question scores and pass/fail
    """
    from ..levels.loader import load_level

    level = load_level(level_id)
    logger.info("Running level %s: %s", level.id, level.name)

    # Feed level data to agent via its learn() method.
    # We use the matching Python-defined TestLevel articles if available.
    from ..data.progressive_levels import get_level_by_id as get_python_level

    py_level = get_python_level(level_id.replace("L0", "L").replace("L", "L", 1))
    # Normalize: L01 -> L1, L12 -> L12
    normalized_id = "L" + str(int(level_id.replace("L", "")))
    py_level = get_python_level(normalized_id)

    if py_level:
        for article in py_level.articles:
            agent.learn(article.content)
    else:
        logger.warning("No Python-defined level data for %s, skipping article feeding", level_id)

    # Ask each question and grade
    results: list[EvalResult] = []
    for q in level.questions:
        logger.info("  Question %s: %s", q.id, q.text[:60])

        try:
            response = agent.answer(q.text)
            answer = response.answer
        except Exception as e:
            logger.warning("Agent failed to answer %s: %s", q.id, e)
            answer = f"Error: {e}"

        # Determine dimensions from question or level scoring config
        dimensions = q.scoring_dimensions or level.scoring.dimensions

        # Build a minimal Question-like object for the grading functions
        grade_start = time.time()
        dim_scores = _grade_level_question(
            question_text=q.text,
            expected_answer=q.expected_answer or "",
            actual_answer=answer,
            dimensions=dimensions,
            grading_mode=level.grading_mode,
            grader_model=grader_model,
            num_votes=level.scoring.grader_votes,
        )
        grade_time = time.time() - grade_start

        overall = sum(d.score for d in dim_scores) / len(dim_scores) if dim_scores else 0.0

        results.append(
            EvalResult(
                question_id=q.id,
                question_text=q.text,
                category=q.category,
                expected_answer=q.expected_answer or "",
                actual_answer=answer if isinstance(answer, str) else str(answer),
                dimensions=dim_scores,
                overall_score=overall,
                grading_time_s=grade_time,
            )
        )

    # Build category breakdown
    categories: dict[str, list[EvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    breakdown = []
    for cat, cat_results in sorted(categories.items()):
        scores = [r.overall_score for r in cat_results]
        dim_avgs: dict[str, list[float]] = {}
        for r in cat_results:
            for d in r.dimensions:
                dim_avgs.setdefault(d.dimension, []).append(d.score)

        breakdown.append(
            CategoryBreakdown(
                category=cat,
                num_questions=len(cat_results),
                avg_score=sum(scores) / len(scores),
                min_score=min(scores),
                max_score=max(scores),
                dimension_averages={k: sum(v) / len(v) for k, v in dim_avgs.items()},
            )
        )

    overall_score = sum(r.overall_score for r in results) / len(results) if results else 0.0
    passed = overall_score >= level.scoring.pass_threshold

    logger.info(
        "Level %s: score=%.2f%% threshold=%.2f%% %s",
        level.id,
        overall_score * 100,
        level.scoring.pass_threshold * 100,
        "PASSED" if passed else "FAILED",
    )

    return LevelResult(
        level_id=level.id,
        level_name=level.name,
        passed=passed,
        overall_score=overall_score,
        pass_threshold=level.scoring.pass_threshold,
        results=results,
        category_breakdown=breakdown,
    )


def run_suite(
    level_ids: list[str],
    agent: AgentAdapter,
    grader_model: str = "",
    check_prerequisites: bool = True,
) -> SuiteResult:
    """Run multiple YAML-defined evaluation levels in sequence.

    Args:
        level_ids: Level identifiers to run (e.g. ["L01", "L02", "L03"])
        agent: Agent implementing AgentAdapter interface
        grader_model: Model for LLM grading
        check_prerequisites: If True, skip levels whose prerequisites failed

    Returns:
        SuiteResult with per-level results and aggregate score
    """
    from ..levels.loader import load_level

    level_results: list[LevelResult] = []
    skipped: list[str] = []
    passed_ids: set[str] = set()

    for level_id in level_ids:
        # Check prerequisites
        if check_prerequisites:
            try:
                level_def = load_level(level_id)
            except FileNotFoundError:
                logger.warning("Level %s not found, skipping", level_id)
                skipped.append(level_id)
                continue

            missing_prereqs = [p for p in level_def.prerequisites if p not in passed_ids]
            if missing_prereqs:
                logger.info(
                    "Skipping %s: prerequisites not met: %s",
                    level_id,
                    missing_prereqs,
                )
                skipped.append(level_id)
                continue

        # Reset agent between levels
        try:
            agent.reset()
        except Exception as e:
            logger.warning("Agent reset failed before %s: %s", level_id, e)

        result = run_level(level_id, agent, grader_model=grader_model)
        level_results.append(result)

        if result.passed:
            passed_ids.add(level_id)

    overall = (
        sum(r.overall_score for r in level_results) / len(level_results)
        if level_results
        else 0.0
    )

    return SuiteResult(
        level_results=level_results,
        overall_score=overall,
        passed_count=sum(1 for r in level_results if r.passed),
        total_count=len(level_results),
        skipped=skipped,
    )


def _grade_level_question(
    question_text: str,
    expected_answer: str,
    actual_answer: str,
    dimensions: list[str],
    grading_mode: str,
    grader_model: str,
    num_votes: int,
) -> list[DimensionScore]:
    """Grade a single question from a YAML-defined level.

    Builds a lightweight Question-compatible object and delegates to
    the existing grading infrastructure.
    """
    from ..data.long_horizon import GradingRubric, Question

    # Build a minimal Question for the existing grading functions
    q = Question(
        question_id="yaml_q",
        text=question_text,
        expected_answer=expected_answer,
        category="yaml_level",
        relevant_turns=[],
        scoring_dimensions=dimensions,
        rubric=GradingRubric(
            required_keywords=expected_answer.split()[:5] if expected_answer else [],
        ),
    )

    if grading_mode == "deterministic":
        # Only use deterministic grading
        det_scores = _deterministic_grade(q.rubric, actual_answer, dimensions)
        result: list[DimensionScore] = []
        for dim in dimensions:
            if dim in det_scores:
                result.append(det_scores[dim])
            else:
                result.append(
                    DimensionScore(
                        dimension=dim,
                        score=0.0,
                        reasoning="Not gradable deterministically",
                    )
                )
        return result
    elif grading_mode == "llm":
        return _grade_multi_vote(q, actual_answer, dimensions, grader_model, num_votes)
    else:
        # hybrid (default)
        return _grade_multi_vote(q, actual_answer, dimensions, grader_model, num_votes)


# Backward compatibility alias
LongHorizonMemoryEval = EvalRunner


__all__ = [
    "EvalRunner",
    "LongHorizonMemoryEval",
    "EvalResult",
    "EvalReport",
    "CategoryBreakdown",
    "DimensionScore",
    "LevelResult",
    "SuiteResult",
    "print_report",
    "run_level",
    "run_suite",
    "_deterministic_grade",
    "_grade_hybrid",
    "_grade_multi_vote",
]
