"""Distributed eval — uses the EXACT same eval harness as single-agent.

Creates a RemoteAgentAdapter that forwards learn_from_content() and
answer_question() to deployed Azure Container Apps agents via Event Hubs.
Passes it to LongHorizonMemoryEval.run() — identical code path, grading,
and report format as single-agent eval.

The agent's OODA loop processes all inputs normally. The adapter is pure DI.

Usage:
    python deploy/azure_hive/eval_distributed.py \
        --connection-string "$EH_CONN" \
        --input-hub hive-events-amplihiveeval \
        --response-hub eval-responses-amplihiveeval \
        --turns 5000 --questions 50 \
        --agents 100 \
        --grader-model claude-haiku-4-5-20251001 \
        --output results.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from pathlib import Path

try:
    from amplihack.observability import configure_otel, start_span
except ImportError:  # pragma: no cover

    def configure_otel(  # type: ignore[misc]
        service_name: str, *, component: str = "", attributes: object = None
    ) -> bool:
        return False

    def start_span(  # type: ignore[misc]
        name: str, *, tracer_name: str, attributes: object = None
    ) -> object:
        return contextlib.nullcontext()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval_distributed")


def _default_agent_count() -> int:
    raw = os.environ.get("AMPLIHACK_AGENT_COUNT") or os.environ.get("HIVE_AGENT_COUNT")
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("Ignoring invalid agent count override: %s", raw)

    return 10 if os.environ.get("HIVE_DEPLOYMENT_PROFILE", "").strip() == "smoke-10" else 100


def _default_agents_per_app() -> int:
    raw = os.environ.get("AMPLIHACK_AGENTS_PER_APP") or os.environ.get("HIVE_AGENTS_PER_APP")
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("Ignoring invalid agents-per-app override: %s", raw)

    return 1 if os.environ.get("HIVE_DEPLOYMENT_PROFILE", "").strip() == "smoke-10" else 5


def _default_parallel_workers(agent_count: int) -> int:
    if agent_count >= 100:
        return 1
    if agent_count >= 50:
        return 2
    return 10


def _default_question_failover_retries(agent_count: int) -> int:
    if agent_count >= 100:
        return 2
    if agent_count >= 50:
        return 1
    # <50 agents: 1 retry to compensate for the 120s answer_timeout default
    return 1


def _default_answer_timeout(agent_count: int) -> int:
    if agent_count >= 100:
        return 0
    return 120


def main() -> int:
    p = argparse.ArgumentParser(
        description="Distributed eval — same harness as single-agent, remote agents via Event Hubs"
    )
    p.add_argument("--connection-string", required=True, help="Event Hubs namespace connection string")
    p.add_argument("--input-hub", default="hive-events", help="Agent input Event Hub name")
    p.add_argument("--response-hub", default="eval-responses", help="Eval response Event Hub name")
    p.add_argument("--turns", type=int, default=300, help="Dialogue turns")
    p.add_argument("--questions", type=int, default=50, help="Number of questions")
    p.add_argument("--agents", type=int, default=_default_agent_count(), help="Number of deployed agents")
    p.add_argument(
        "--agents-per-app",
        type=int,
        default=_default_agents_per_app(),
        help="Number of agents packed into each Azure Container App failure domain",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--question-set",
        choices=("standard", "holdout"),
        default="standard",
        help="Deterministic question subset to use",
    )
    p.add_argument(
        "--grader-model",
        default="claude-haiku-4-5-20251001",
        help="Grading model override for this Azure distributed runner (default: claude-haiku-4-5-20251001)",
    )
    p.add_argument("--resource-group", default="", help="Azure resource group (optional, unused)")
    p.add_argument(
        "--answer-timeout",
        type=int,
        default=None,
        help=("Seconds to wait per answer before failover (default: scale-aware; 0 for 100+ agents, 120 otherwise)"),
    )
    p.add_argument("--output", default="", help="Output JSON path")
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Parallel question-answer workers (default: scale-aware; 10 up to 49 agents, 2 for 50-99, 1 for 100+)",
    )
    p.add_argument(
        "--replicate-learn-to-all-agents",
        action="store_true",
        default=False,
        help="Replicate each learn_from_content call to ALL agents (not just one per round-robin)",
    )
    p.add_argument(
        "--question-failover-retries",
        type=int,
        default=None,
        help=(
            "Number of failover retries for unanswered questions "
            "(default: scale-aware; 1 up to 49 agents, 1 for 50-99, 2 for 100+)"
        ),
    )
    args = p.parse_args()
    if args.parallel_workers is None:
        args.parallel_workers = _default_parallel_workers(args.agents)
    if args.question_failover_retries is None:
        args.question_failover_retries = _default_question_failover_retries(args.agents)
    if args.answer_timeout is None:
        args.answer_timeout = _default_answer_timeout(args.agents)
    args.agents_per_app = max(1, min(args.agents, args.agents_per_app))

    configure_otel(
        service_name=os.environ.get("OTEL_SERVICE_NAME", "").strip() or "amplihack.azure-eval-harness",
        component="eval-distributed",
        attributes={
            "amplihack.agent_count": args.agents,
            "amplihack.agents_per_app": args.agents_per_app,
            "amplihack.turns": args.turns,
            "amplihack.questions": args.questions,
            "amplihack.question_set": args.question_set,
            "amplihack.parallel_workers": args.parallel_workers,
            "amplihack.question_failover_retries": args.question_failover_retries,
            "amplihack.answer_timeout": args.answer_timeout,
        },
    )

    try:
        from amplihack.eval.long_horizon_memory import LongHorizonMemoryEval, _print_report
    except ImportError as exc:
        print(
            "Error: python -m amplihack_eval.azure.eval_distributed requires the sibling "
            "amplihack package to be installed because it reuses amplihack's long-horizon harness.",
            file=sys.stderr,
        )
        print(f"Detail: {exc}", file=sys.stderr)
        return 1

    from amplihack_eval.adapters.remote_agent_adapter import RemoteAgentAdapter

    # Create the remote adapter — same interface as LearningAgent
    adapter = RemoteAgentAdapter(
        connection_string=args.connection_string,
        input_hub=args.input_hub,
        response_hub=args.response_hub,
        agent_count=args.agents,
        agents_per_app=args.agents_per_app,
        resource_group=args.resource_group,
        answer_timeout=args.answer_timeout,
        replicate_learning_to_all_agents=args.replicate_learn_to_all_agents,
        question_failover_retries=args.question_failover_retries,
    )
    logger.info(
        "Distributed eval config: agents=%d agents_per_app=%d parallel_workers=%d "
        "answer_timeout=%d failover_retries=%d replicate_learn=%s question_set=%s",
        args.agents,
        args.agents_per_app,
        args.parallel_workers,
        args.answer_timeout,
        args.question_failover_retries,
        args.replicate_learn_to_all_agents,
        args.question_set,
    )

    # Create the eval harness — IDENTICAL to single-agent
    eval_harness = LongHorizonMemoryEval(
        num_turns=args.turns,
        num_questions=args.questions,
        seed=args.seed,
        parallel_workers=args.parallel_workers,
        question_set=args.question_set,
    )

    # Run — same code path as: python -m amplihack.eval.long_horizon_memory
    try:
        with start_span(
            "azure_eval.run_long_horizon",
            tracer_name=__name__,
            attributes={
                "amplihack.agent_count": args.agents,
                "amplihack.agents_per_app": args.agents_per_app,
                "amplihack.turns": args.turns,
                "amplihack.questions": args.questions,
                "amplihack.question_set": args.question_set,
                "amplihack.parallel_workers": args.parallel_workers,
                "amplihack.question_failover_retries": args.question_failover_retries,
                "amplihack.answer_timeout": args.answer_timeout,
            },
        ):
            report = eval_harness.run(adapter, grader_model=args.grader_model)
    finally:
        adapter.close()

    # Print report (same format)
    _print_report(report)

    # Write output
    output_path = args.output or f"/tmp/distributed_eval_{args.seed}.json"
    report_dict = report.to_dict()
    report_dict["eval_type"] = "distributed"
    report_dict["agent_count"] = args.agents
    report_dict["agents_per_app"] = args.agents_per_app
    report_dict["question_set"] = args.question_set
    report_dict["input_hub"] = args.input_hub
    report_dict["response_hub"] = args.response_hub
    report_dict["parallel_workers"] = args.parallel_workers
    report_dict["question_failover_retries"] = args.question_failover_retries
    report_dict["replicate_learn_to_all_agents"] = args.replicate_learn_to_all_agents
    report_dict["answer_timeout"] = args.answer_timeout
    Path(output_path).write_text(json.dumps(report_dict, indent=2))
    logger.info("Report written to %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
