"""CLI entry point: amplihack-eval

Subcommands:
    run          Run a long-horizon memory evaluation
    compare      Multi-seed comparison across random seeds
    self-improve Run the automated self-improvement loop
    report       Print a saved evaluation report

Usage:
    amplihack-eval run --turns 100 --questions 20
    amplihack-eval compare --seeds 42,123,456,789
    amplihack-eval report /path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a long-horizon memory evaluation."""
    from .core.runner import EvalRunner, print_report

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create adapter based on --adapter flag
    adapter = _create_adapter(args)
    if adapter is None:
        print("Error: Could not create agent adapter. See --adapter options.", file=sys.stderr)
        return 1

    try:
        runner = EvalRunner(
            num_turns=args.turns,
            num_questions=args.questions,
            seed=args.seed,
            grader_votes=args.grader_votes,
        )

        report = runner.run(adapter, grader_model=args.grader_model)
        print_report(report)

        # Save JSON report
        report_path = output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to {report_path}")

        return 0

    finally:
        adapter.close()


def _cmd_compare(args: argparse.Namespace) -> int:
    """Multi-seed comparison."""
    from .core.multi_seed import print_multi_seed_report, run_multi_seed_eval

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    def agent_factory():
        return _create_adapter(args)

    report = run_multi_seed_eval(
        agent_factory=agent_factory,
        num_turns=args.turns,
        num_questions=args.questions,
        seeds=seeds,
        grader_model=args.grader_model,
        grader_votes=args.grader_votes,
    )

    print_multi_seed_report(report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "multi_seed_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nReport saved to {report_path}")

    return 0


def _cmd_self_improve(args: argparse.Namespace) -> int:
    """Run self-improvement loop."""
    from .self_improve.runner import SelfImproveConfig, run_self_improve

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = SelfImproveConfig(
        num_turns=args.turns,
        num_questions=args.questions,
        seed=args.seed,
        max_iterations=args.iterations,
        failure_threshold=args.threshold,
        regression_threshold=args.regression_threshold,
        output_dir=args.output_dir,
        grader_model=args.grader_model,
    )

    def agent_factory():
        return _create_adapter(args)

    result = run_self_improve(
        config=config,
        agent_factory=agent_factory,
    )

    if not result.iterations:
        print("\nNo iterations completed.")
        return 1

    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Print a saved evaluation report."""
    report_path = Path(args.report_file)
    if not report_path.exists():
        print(f"Error: Report file not found: {report_path}", file=sys.stderr)
        return 1

    with open(report_path) as f:
        data = json.load(f)

    print(json.dumps(data, indent=2))
    return 0


def _create_adapter(args: argparse.Namespace):
    """Create an agent adapter based on CLI args."""
    adapter_type = getattr(args, "adapter", "http")

    if adapter_type == "learning-agent":
        from .adapters.learning_agent import LearningAgentAdapter

        return LearningAgentAdapter(
            model=getattr(args, "model", ""),
        )

    elif adapter_type == "subprocess":
        from .adapters.subprocess_adapter import SubprocessAdapter

        cmd = getattr(args, "agent_command", "").split()
        if not cmd:
            print("Error: --agent-command required for subprocess adapter", file=sys.stderr)
            return None
        return SubprocessAdapter(command=cmd)

    elif adapter_type == "http":
        from .adapters.http_adapter import HttpAdapter

        base_url = getattr(args, "agent_url", "http://localhost:8000")
        return HttpAdapter(base_url=base_url)

    else:
        print(f"Error: Unknown adapter type: {adapter_type}", file=sys.stderr)
        return None


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="amplihack-eval",
        description="Evaluation framework for goal-seeking AI agents",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run a long-horizon memory evaluation")
    run_parser.add_argument("--turns", type=int, default=100, help="Dialogue turns (default: 100)")
    run_parser.add_argument(
        "--questions", type=int, default=20, help="Quiz questions (default: 20)"
    )
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument(
        "--grader-votes", type=int, default=3, help="Grading votes per question"
    )
    run_parser.add_argument("--grader-model", default="", help="Model for grading")
    run_parser.add_argument("--model", default="", help="Agent model")
    run_parser.add_argument(
        "--output-dir", default="/tmp/amplihack-eval", help="Output directory"
    )
    run_parser.add_argument(
        "--adapter",
        choices=["http", "subprocess", "learning-agent"],
        default="http",
        help="Agent adapter type",
    )
    run_parser.add_argument("--agent-url", default="http://localhost:8000", help="Agent HTTP URL")
    run_parser.add_argument("--agent-command", default="", help="Agent subprocess command")

    # --- compare ---
    cmp_parser = subparsers.add_parser("compare", help="Multi-seed comparison")
    cmp_parser.add_argument("--turns", type=int, default=100, help="Dialogue turns")
    cmp_parser.add_argument("--questions", type=int, default=20, help="Quiz questions")
    cmp_parser.add_argument("--seeds", default="42,123,456,789", help="Comma-separated seeds")
    cmp_parser.add_argument("--grader-votes", type=int, default=3, help="Grading votes")
    cmp_parser.add_argument("--grader-model", default="", help="Grader model")
    cmp_parser.add_argument("--model", default="", help="Agent model")
    cmp_parser.add_argument("--output-dir", default="/tmp/amplihack-eval-multi", help="Output dir")
    cmp_parser.add_argument(
        "--adapter",
        choices=["http", "subprocess", "learning-agent"],
        default="http",
        help="Agent adapter type",
    )
    cmp_parser.add_argument("--agent-url", default="http://localhost:8000", help="Agent HTTP URL")
    cmp_parser.add_argument("--agent-command", default="", help="Agent subprocess command")

    # --- self-improve ---
    si_parser = subparsers.add_parser("self-improve", help="Run self-improvement loop")
    si_parser.add_argument("--turns", type=int, default=100, help="Dialogue turns")
    si_parser.add_argument("--questions", type=int, default=20, help="Quiz questions")
    si_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    si_parser.add_argument("--iterations", type=int, default=3, help="Max iterations")
    si_parser.add_argument("--threshold", type=float, default=0.7, help="Failure threshold")
    si_parser.add_argument(
        "--regression-threshold", type=float, default=5.0, help="Regression threshold (pp)"
    )
    si_parser.add_argument("--grader-model", default="", help="Grader model")
    si_parser.add_argument("--model", default="", help="Agent model")
    si_parser.add_argument(
        "--output-dir", default="/tmp/amplihack-eval-improve", help="Output dir"
    )
    si_parser.add_argument(
        "--adapter",
        choices=["http", "subprocess", "learning-agent"],
        default="http",
        help="Agent adapter type",
    )
    si_parser.add_argument("--agent-url", default="http://localhost:8000", help="Agent HTTP URL")
    si_parser.add_argument("--agent-command", default="", help="Agent subprocess command")

    # --- report ---
    rpt_parser = subparsers.add_parser("report", help="Print a saved report")
    rpt_parser.add_argument("report_file", help="Path to report JSON file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "run": _cmd_run,
        "compare": _cmd_compare,
        "self-improve": _cmd_self_improve,
        "report": _cmd_report,
    }

    handler = handlers.get(args.command)
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
