"""Continuous evaluation runner comparing single-agent, flat-hive, and federated-hive.

Runs the SAME security analyst scenario through three conditions using
the SAME L1-L12 questions, producing a ContinuousEvalReport with per-condition
scores and confidence intervals.

Philosophy:
- Same data, same questions, different agent topologies
- Deterministic data generation, reproducible comparisons
- Reuses EvalRunner for grading consistency
- Structured report with confidence intervals for rigorous comparison

Public API:
    ContinuousEvalReport: Comparison report across conditions
    ConditionResult: Per-condition evaluation result
    run_continuous_eval: Main entry point
"""

from __future__ import annotations

import logging
import math
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..adapters.base import AgentAdapter, AgentResponse
from ..core.runner import EvalReport, EvalRunner
from ..data.security_analyst_scenario import (
    generate_dialogue,
    generate_questions,
)

logger = logging.getLogger(__name__)


@dataclass
class ConditionResult:
    """Result for a single evaluation condition."""

    condition: str  # "single", "flat", "federated"
    num_agents: int
    num_groups: int
    report: EvalReport
    elapsed_s: float
    hive_facts: int = 0
    per_level_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ContinuousEvalReport:
    """Comparison report across all conditions."""

    config: dict[str, Any]
    conditions: list[ConditionResult]
    comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "config": self.config,
            "conditions": [
                {
                    "condition": c.condition,
                    "num_agents": c.num_agents,
                    "num_groups": c.num_groups,
                    "overall_score": round(c.report.overall_score, 4),
                    "elapsed_s": round(c.elapsed_s, 1),
                    "hive_facts": c.hive_facts,
                    "per_level_scores": {
                        k: round(v, 4) for k, v in c.per_level_scores.items()
                    },
                    "category_breakdown": [
                        {
                            "category": cb.category,
                            "num_questions": cb.num_questions,
                            "avg_score": round(cb.avg_score, 4),
                        }
                        for cb in c.report.category_breakdown
                    ],
                    "learning_time_s": round(c.report.learning_time_s, 1),
                    "questioning_time_s": round(c.report.questioning_time_s, 1),
                    "grading_time_s": round(c.report.grading_time_s, 1),
                }
                for c in self.conditions
            ],
            "comparison": self.comparison,
        }


def _compute_ci(scores: list[float]) -> tuple[float, float, float]:
    """Compute mean and confidence interval for a list of scores.

    Returns (mean, ci_lower, ci_upper).
    """
    if not scores:
        return 0.0, 0.0, 0.0
    n = len(scores)
    mean = sum(scores) / n
    if n < 2:
        return mean, mean, mean
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    std_err = math.sqrt(variance / n)
    # Use z=1.96 for 95% CI (good enough for n >= 5)
    z = 1.96
    return mean, mean - z * std_err, mean + z * std_err


def _compute_per_level_scores(report: EvalReport) -> dict[str, float]:
    """Extract per-level scores from an EvalReport (L1-L12 categories)."""
    level_scores: dict[str, list[float]] = {}
    for result in report.results:
        # Category format: "L1_direct_recall", "L2_multi_source_synthesis", etc.
        level = result.category.split("_")[0] if "_" in result.category else result.category
        if level not in level_scores:
            level_scores[level] = []
        level_scores[level].append(result.overall_score)

    return {
        level: sum(scores) / len(scores) if scores else 0.0
        for level, scores in sorted(level_scores.items())
    }


# ---------------------------------------------------------------------------
# Adapter wrappers (mirror run_learning_agent_hive_eval.py patterns)
# ---------------------------------------------------------------------------


class _SingleAgentAdapter(AgentAdapter):
    """Wraps a LearningAgent for single-agent evaluation."""

    def __init__(self, agent: Any, model: str):
        self._agent = agent
        self._model = model

    def learn(self, content: str) -> None:
        self._agent.learn_from_content(content)

    def answer(self, question: str) -> AgentResponse:
        try:
            result = self._agent.answer_question(question)
            text = result[0] if isinstance(result, tuple) else str(result)
            return AgentResponse(answer=text, metadata={"model": self._model})
        except Exception as e:
            logger.warning("Answer failed: %s", e)
            return AgentResponse(answer=f"Error: {e}")

    def reset(self) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self._agent, "close"):
            self._agent.close()

    @property
    def name(self) -> str:
        return "SingleAgent"


class _MultiAgentAdapter(AgentAdapter):
    """N agents sharing a hive, presented as a single AgentAdapter.

    Learning: turns distributed round-robin. Supports parallel learning
    via learn_parallel() for multi-agent speedup.
    Answering: queries agents in parallel, returns longest non-error answer.
    """

    def __init__(self, agents: list[Any], model: str, parallel_workers: int = 10):
        self._agents = agents
        self._model = model
        self._turn_idx = 0
        self._parallel_workers = parallel_workers

    def learn(self, content: str) -> None:
        """Sequential learning (one turn at a time, round-robin)."""
        agent = self._agents[self._turn_idx % len(self._agents)]
        agent.learn_from_content(content)
        self._turn_idx += 1

    def learn_parallel(self, turns: list[Any]) -> float:
        """Parallel learning: pre-assign turns to agents, learn concurrently.

        Distributes turns round-robin across agents, then each agent
        learns its batch in parallel with other agents.

        Returns learning time in seconds.
        """
        # Pre-assign turns to agents (round-robin)
        agent_batches: dict[int, list[str]] = defaultdict(list)
        for i, turn in enumerate(turns):
            agent_idx = i % len(self._agents)
            content = turn.content if hasattr(turn, "content") else str(turn)
            agent_batches[agent_idx].append(content)

        num_agents = len(agent_batches)
        logger.info(
            "Parallel learning: %d turns across %d agents (%d workers)",
            len(turns),
            num_agents,
            self._parallel_workers,
        )

        learn_t0 = time.time()
        completed = 0
        lock = threading.Lock()

        def _learn_batch(agent_idx: int, contents: list[str]) -> int:
            nonlocal completed
            agent = self._agents[agent_idx]
            name = getattr(agent, "agent_name", f"agent_{agent_idx}")
            count = 0
            for content in contents:
                try:
                    agent.learn_from_content(content)
                    count += 1
                except Exception as e:
                    logger.warning("Agent %s learn failed: %s", name, e)
            with lock:
                completed += count
                if completed % 100 == 0:
                    elapsed = time.time() - learn_t0
                    logger.info(
                        "Learning progress: %d/%d turns (%.0fs)",
                        completed,
                        len(turns),
                        elapsed,
                    )
            return count

        with ThreadPoolExecutor(max_workers=self._parallel_workers) as pool:
            futures = {
                pool.submit(_learn_batch, idx, batch): idx
                for idx, batch in agent_batches.items()
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning("Batch learning failed: %s", e)

        learn_time = time.time() - learn_t0
        logger.info(
            "Parallel learning complete: %d turns in %.1fs (%.1f turns/s)",
            completed,
            learn_time,
            completed / max(0.1, learn_time),
        )
        return learn_time

    def answer(self, question: str) -> AgentResponse:
        """Query agents in parallel, return longest non-error answer."""
        best_answer = ""
        lock = threading.Lock()

        def _ask_agent(agent: Any) -> str | None:
            try:
                result = agent.answer_question(question)
                text = result[0] if isinstance(result, tuple) else str(result)
                if not text.startswith("Error:"):
                    return text
            except Exception as e:
                logger.debug(
                    "Agent %s failed: %s",
                    getattr(agent, "agent_name", "?"),
                    e,
                )
            return None

        # Query all agents in parallel
        with ThreadPoolExecutor(max_workers=self._parallel_workers) as pool:
            futures = {pool.submit(_ask_agent, agent): agent for agent in self._agents}
            for future in as_completed(futures):
                try:
                    text = future.result()
                    if text and len(text) > len(best_answer):
                        with lock:
                            if len(text) > len(best_answer):
                                best_answer = text
                except Exception:
                    pass

        if not best_answer:
            best_answer = "No agent could answer"
        return AgentResponse(answer=best_answer, metadata={"model": self._model})

    def reset(self) -> None:
        self.close()

    def close(self) -> None:
        for agent in self._agents:
            if hasattr(agent, "close"):
                agent.close()

    @property
    def name(self) -> str:
        return f"MultiAgent({len(self._agents)})"


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------


def _run_single(
    model: str,
    num_turns: int,
    num_questions: int,
    seed: int,
    tmpdir: str,
    parallel_workers: int,
    prompt_variant: int | None,
) -> ConditionResult:
    """Run SINGLE condition: 1 LearningAgent, no hive."""
    from amplihack.agents.goal_seeking.learning_agent import LearningAgent  # type: ignore[import-untyped]

    logger.info("=== SINGLE: 1 agent, no hive ===")
    t0 = time.time()

    kwargs: dict[str, Any] = {}
    if prompt_variant is not None:
        kwargs["prompt_variant"] = prompt_variant

    agent = LearningAgent(
        agent_name="single_agent",
        model=model,
        storage_path=Path(tmpdir) / "single",
        use_hierarchical=True,
        **kwargs,
    )
    adapter = _SingleAgentAdapter(agent, model)

    # Use security scenario
    ground_truth = generate_dialogue(num_turns=num_turns, seed=seed)
    questions = generate_questions(ground_truth, num_questions=num_questions)

    runner = EvalRunner(
        num_turns=num_turns,
        num_questions=num_questions,
        seed=seed,
        parallel_workers=parallel_workers,
    )

    # Feed dialogue turns
    learn_t0 = time.time()
    for turn in ground_truth.turns:
        adapter.learn(turn.content)
    learn_time = time.time() - learn_t0

    # Evaluate
    report = runner.evaluate(adapter, questions=questions, grader_model=model)
    report.learning_time_s = learn_time
    report.num_turns = num_turns
    report.total_facts_delivered = sum(len(t.facts) for t in ground_truth.turns)

    elapsed = time.time() - t0
    adapter.close()

    per_level = _compute_per_level_scores(report)
    logger.info("SINGLE done: %.2f%% in %.1fs", report.overall_score * 100, elapsed)

    return ConditionResult(
        condition="single",
        num_agents=1,
        num_groups=0,
        report=report,
        elapsed_s=elapsed,
        per_level_scores=per_level,
    )


def _run_flat(
    model: str,
    num_agents: int,
    num_turns: int,
    num_questions: int,
    seed: int,
    tmpdir: str,
    parallel_workers: int,
    prompt_variant: int | None,
) -> ConditionResult:
    """Run HIVE_FLAT: N agents sharing a single InMemoryHiveGraph."""
    from amplihack.agents.goal_seeking.hive_mind.hive_graph import InMemoryHiveGraph  # type: ignore[import-untyped]
    from amplihack.agents.goal_seeking.learning_agent import LearningAgent  # type: ignore[import-untyped]

    logger.info("=== FLAT: %d agents, shared hive ===", num_agents)
    t0 = time.time()

    try:
        from amplihack.agents.goal_seeking.hive_mind.embeddings import EmbeddingGenerator  # type: ignore[import-untyped]

        embedder = EmbeddingGenerator()
        if not embedder.available:
            embedder = None
    except Exception as e:
        logger.warning("Embedding generator unavailable: %s", e)
        embedder = None

    hive = InMemoryHiveGraph(
        "flat-hive",
        embedding_generator=embedder,
        enable_gossip=True,
        enable_ttl=True,
    )

    kwargs: dict[str, Any] = {}
    if prompt_variant is not None:
        kwargs["prompt_variant"] = prompt_variant

    agents = []
    for i in range(num_agents):
        name = f"flat_agent_{i}"
        hive.register_agent(name)
        agent = LearningAgent(
            agent_name=name,
            model=model,
            storage_path=Path(tmpdir) / f"flat_{i}",
            use_hierarchical=True,
            hive_store=hive,
            **kwargs,
        )
        agents.append(agent)

    adapter = _MultiAgentAdapter(agents, model, parallel_workers=parallel_workers)

    ground_truth = generate_dialogue(num_turns=num_turns, seed=seed)
    questions = generate_questions(ground_truth, num_questions=num_questions)

    runner = EvalRunner(
        num_turns=num_turns,
        num_questions=num_questions,
        seed=seed,
        parallel_workers=parallel_workers,
    )

    learn_time = adapter.learn_parallel(ground_truth.turns)

    report = runner.evaluate(adapter, questions=questions, grader_model=model)
    report.learning_time_s = learn_time
    report.num_turns = num_turns
    report.total_facts_delivered = sum(len(t.facts) for t in ground_truth.turns)

    elapsed = time.time() - t0
    hive_facts = hive.get_stats().get("fact_count", 0)
    adapter.close()

    per_level = _compute_per_level_scores(report)
    logger.info("FLAT done: %.2f%% in %.1fs (hive facts: %d)", report.overall_score * 100, elapsed, hive_facts)

    return ConditionResult(
        condition="flat",
        num_agents=num_agents,
        num_groups=1,
        report=report,
        elapsed_s=elapsed,
        hive_facts=hive_facts,
        per_level_scores=per_level,
    )


def _run_federated(
    model: str,
    num_agents: int,
    num_groups: int,
    num_turns: int,
    num_questions: int,
    seed: int,
    tmpdir: str,
    parallel_workers: int,
    prompt_variant: int | None,
) -> ConditionResult:
    """Run FEDERATED: N agents in M groups with federation tree.

    Uses DistributedHiveGraph (DHT-sharded) instead of InMemoryHiveGraph
    to avoid Kuzu mmap exhaustion with 100+ concurrent agents.
    Falls back to InMemoryHiveGraph if DistributedHiveGraph unavailable.
    """
    from amplihack.agents.goal_seeking.learning_agent import LearningAgent  # type: ignore[import-untyped]

    logger.info("=== FEDERATED: %d agents, %d groups ===", num_agents, num_groups)
    t0 = time.time()

    try:
        from amplihack.agents.goal_seeking.hive_mind.embeddings import EmbeddingGenerator  # type: ignore[import-untyped]

        embedder = EmbeddingGenerator()
        if not embedder.available:
            embedder = None
    except Exception as e:
        logger.warning("Embedding generator unavailable: %s", e)
        embedder = None

    # Use DistributedHiveGraph (DHT-sharded) for large agent counts
    try:
        from amplihack.agents.goal_seeking.hive_mind.distributed_hive_graph import DistributedHiveGraph  # type: ignore[import-untyped]

        HiveGraphClass = DistributedHiveGraph
        logger.info("Using DistributedHiveGraph (DHT-sharded) for %d agents", num_agents)
    except ImportError:
        from amplihack.agents.goal_seeking.hive_mind.hive_graph import InMemoryHiveGraph  # type: ignore[import-untyped]

        HiveGraphClass = InMemoryHiveGraph
        logger.warning("DistributedHiveGraph unavailable, falling back to InMemoryHiveGraph")

    root_hive = HiveGraphClass(
        "root-hive",
        embedding_generator=embedder,
        enable_gossip=True,
        enable_ttl=True,
    )
    group_hives = []
    for g in range(num_groups):
        group_hive = HiveGraphClass(
            f"group-{g}",
            embedding_generator=embedder,
            enable_gossip=True,
            enable_ttl=True,
        )
        group_hive.set_parent(root_hive)
        root_hive.add_child(group_hive)
        group_hives.append(group_hive)

    kwargs: dict[str, Any] = {}
    if prompt_variant is not None:
        kwargs["prompt_variant"] = prompt_variant

    agents_per_group = max(1, num_agents // num_groups)
    agents = []
    agent_idx = 0

    for g, group_hive in enumerate(group_hives):
        n = agents_per_group if g < num_groups - 1 else num_agents - agent_idx
        for _ in range(n):
            name = f"fed_agent_{agent_idx}"
            group_hive.register_agent(name)
            agent = LearningAgent(
                agent_name=name,
                model=model,
                storage_path=Path(tmpdir) / f"fed_{agent_idx}",
                use_hierarchical=True,
                hive_store=group_hive,
                **kwargs,
            )
            agents.append(agent)
            agent_idx += 1

    adapter = _MultiAgentAdapter(agents, model, parallel_workers=parallel_workers)

    ground_truth = generate_dialogue(num_turns=num_turns, seed=seed)
    questions = generate_questions(ground_truth, num_questions=num_questions)

    runner = EvalRunner(
        num_turns=num_turns,
        num_questions=num_questions,
        seed=seed,
        parallel_workers=parallel_workers,
    )

    learn_time = adapter.learn_parallel(ground_truth.turns)

    report = runner.evaluate(adapter, questions=questions, grader_model=model)
    report.learning_time_s = learn_time
    report.num_turns = num_turns
    report.total_facts_delivered = sum(len(t.facts) for t in ground_truth.turns)

    elapsed = time.time() - t0
    total_hive_facts = 0
    for hive in [root_hive, *group_hives]:
        total_hive_facts += hive.get_stats().get("fact_count", 0)

    adapter.close()

    per_level = _compute_per_level_scores(report)
    logger.info(
        "FEDERATED done: %.2f%% in %.1fs (hive facts: %d)",
        report.overall_score * 100, elapsed, total_hive_facts,
    )

    return ConditionResult(
        condition="federated",
        num_agents=num_agents,
        num_groups=num_groups,
        report=report,
        elapsed_s=elapsed,
        hive_facts=total_hive_facts,
        per_level_scores=per_level,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_continuous_eval(
    num_turns: int = 100,
    num_questions: int = 50,
    num_agents: int = 5,
    num_groups: int = 2,
    seed: int = 42,
    model: str = "",
    parallel_workers: int = 5,
    prompt_variant: int | None = None,
    conditions: list[str] | None = None,
) -> ContinuousEvalReport:
    """Run continuous evaluation across single, flat, and federated conditions.

    Uses the security analyst scenario with L1-L12 questions for all conditions.

    Args:
        num_turns: Number of dialogue turns
        num_questions: Number of quiz questions
        num_agents: Number of agents for hive conditions
        num_groups: Number of groups for federated condition
        seed: Random seed for reproducibility
        model: LLM model for agents and grading
        parallel_workers: Parallel workers for Q&A grading
        prompt_variant: Optional prompt variant (1-5)
        conditions: Which conditions to run (default: all three)

    Returns:
        ContinuousEvalReport comparing all conditions
    """
    import os

    if not model:
        model = os.environ.get("EVAL_MODEL", "claude-sonnet-4-5-20250929")

    if conditions is None:
        conditions = ["single", "flat", "federated"]

    config = {
        "num_turns": num_turns,
        "num_questions": num_questions,
        "num_agents": num_agents,
        "num_groups": num_groups,
        "seed": seed,
        "model": model,
        "prompt_variant": prompt_variant,
        "conditions": conditions,
    }

    logger.info("=" * 60)
    logger.info("Continuous Evaluation — Security Analyst Scenario")
    logger.info(
        "Turns=%d, Questions=%d, Agents=%d, Groups=%d, Variant=%s",
        num_turns, num_questions, num_agents, num_groups, prompt_variant,
    )
    logger.info("Conditions: %s", conditions)
    logger.info("=" * 60)

    results: list[ConditionResult] = []

    with tempfile.TemporaryDirectory(prefix="continuous_eval_") as tmpdir:
        if "single" in conditions:
            r = _run_single(
                model, num_turns, num_questions, seed,
                tmpdir, parallel_workers, prompt_variant,
            )
            results.append(r)

        if "flat" in conditions:
            r = _run_flat(
                model, num_agents, num_turns, num_questions, seed,
                tmpdir, parallel_workers, prompt_variant,
            )
            results.append(r)

        if "federated" in conditions:
            r = _run_federated(
                model, num_agents, num_groups, num_turns, num_questions, seed,
                tmpdir, parallel_workers, prompt_variant,
            )
            results.append(r)

    # Build comparison
    comparison: dict[str, Any] = {}
    if len(results) >= 2:
        scores = {r.condition: r.report.overall_score for r in results}
        comparison["scores"] = scores

        # Per-level comparison
        all_levels = set()
        for r in results:
            all_levels.update(r.per_level_scores.keys())

        level_comparison = {}
        for level in sorted(all_levels):
            level_comparison[level] = {
                r.condition: round(r.per_level_scores.get(level, 0.0), 4)
                for r in results
            }
        comparison["per_level"] = level_comparison

        # Delta vs baseline (single)
        baseline = scores.get("single")
        if baseline is not None:
            comparison["delta_vs_single"] = {
                cond: round(score - baseline, 4)
                for cond, score in scores.items()
                if cond != "single"
            }

        # CIs per condition (using per-question scores)
        cis = {}
        for r in results:
            q_scores = [res.overall_score for res in r.report.results]
            mean, ci_low, ci_high = _compute_ci(q_scores)
            cis[r.condition] = {
                "mean": round(mean, 4),
                "ci_95_lower": round(ci_low, 4),
                "ci_95_upper": round(ci_high, 4),
                "n": len(q_scores),
            }
        comparison["confidence_intervals"] = cis

    return ContinuousEvalReport(
        config=config,
        conditions=results,
        comparison=comparison,
    )


def print_continuous_report(report: ContinuousEvalReport) -> None:
    """Print a formatted summary of the continuous eval report."""
    print("\n" + "=" * 75)
    print("CONTINUOUS EVALUATION RESULTS — Security Analyst Scenario")
    print("=" * 75)

    cfg = report.config
    print(f"Turns: {cfg['num_turns']}, Questions: {cfg['num_questions']}, "
          f"Agents: {cfg['num_agents']}, Groups: {cfg['num_groups']}")
    if cfg.get("prompt_variant"):
        print(f"Prompt Variant: {cfg['prompt_variant']}")
    print()

    # Overall scores
    print(f"{'Condition':<15} {'Agents':>7} {'Score':>8} {'Time':>8} {'Hive Facts':>11}")
    print("-" * 55)
    for c in report.conditions:
        print(
            f"{c.condition:<15} {c.num_agents:>7} "
            f"{c.report.overall_score:>7.1%} "
            f"{c.elapsed_s:>7.1f}s "
            f"{c.hive_facts:>11}"
        )

    # Per-level comparison
    if report.comparison.get("per_level"):
        print(f"\n{'Level':<8}", end="")
        conds = [c.condition for c in report.conditions]
        for cond in conds:
            print(f" {cond:>12}", end="")
        print()
        print("-" * (8 + 13 * len(conds)))
        for level, scores in sorted(report.comparison["per_level"].items()):
            print(f"{level:<8}", end="")
            for cond in conds:
                score = scores.get(cond, 0.0)
                print(f" {score:>11.1%}", end="")
            print()

    # Confidence intervals
    if report.comparison.get("confidence_intervals"):
        print(f"\n{'Condition':<15} {'Mean':>8} {'95% CI':>20} {'N':>5}")
        print("-" * 50)
        for cond, ci in report.comparison["confidence_intervals"].items():
            ci_str = f"[{ci['ci_95_lower']:.3f}, {ci['ci_95_upper']:.3f}]"
            print(f"{cond:<15} {ci['mean']:>7.1%} {ci_str:>20} {ci['n']:>5}")

    # Delta vs baseline
    if report.comparison.get("delta_vs_single"):
        print("\nDelta vs Single Agent:")
        for cond, delta in report.comparison["delta_vs_single"].items():
            direction = "+" if delta >= 0 else ""
            print(f"  {cond}: {direction}{delta:.1%}")

    print("=" * 75)


__all__ = [
    "ConditionResult",
    "ContinuousEvalReport",
    "run_continuous_eval",
    "print_continuous_report",
]
