"""CLI entry point: amplihack-eval

Subcommands:
    run              Run a long-horizon memory evaluation
    compare          Multi-seed comparison across random seeds
    self-improve     Run the automated self-improvement loop
    report           Print a saved evaluation report
    download-dataset Download a pre-built learning DB from GitHub Releases
    list-datasets    List available pre-built datasets

Usage:
    amplihack-eval run --turns 100 --questions 20
    amplihack-eval run --skip-learning --load-db datasets/5000t-seed42-v1.0/memory_db
    amplihack-eval compare --seeds 42,123,456,789
    amplihack-eval download-dataset 5000t-seed42-v1.0
    amplihack-eval list-datasets
    amplihack-eval report /path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

QUESTION_SET_CHOICES = ("standard", "holdout")


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

    skip_learning = getattr(args, "skip_learning", False)
    load_db = getattr(args, "load_db", "")

    if skip_learning and not load_db:
        print("Error: --skip-learning requires --load-db <path>", file=sys.stderr)
        return 1

    if load_db and not Path(load_db).exists():
        print(f"Error: --load-db path does not exist: {load_db}", file=sys.stderr)
        return 1

    # Create adapter based on --adapter flag
    adapter = _create_adapter(args)
    if adapter is None:
        print("Error: Could not create agent adapter. See --adapter options.", file=sys.stderr)
        return 1

    repeats = getattr(args, "repeats", 1)

    try:
        if repeats > 1:
            # Use multi-seed eval with single seed + repeats for intra-seed CI
            from .core.multi_seed import print_multi_seed_report, run_multi_seed_eval

            def agent_factory():
                a = _create_adapter(args)
                if a is None:
                    raise RuntimeError("Failed to create agent adapter")
                return a

            adapter.close()  # Close the one we created; factory will make fresh ones

            ms_report = run_multi_seed_eval(
                agent_factory=agent_factory,
                num_turns=args.turns,
                num_questions=args.questions,
                seeds=[args.seed],
                grader_model=args.grader_model,
                grader_votes=args.grader_votes,
                repeats_per_seed=repeats,
                question_set=args.question_set,
            )

            print_multi_seed_report(ms_report)

            report_path = output_dir / "report.json"
            with open(report_path, "w") as f:
                json.dump(ms_report.to_dict(), f, indent=2)
            print(f"\nReport saved to {report_path}")
            return 0

        runner = EvalRunner(
            num_turns=args.turns,
            num_questions=args.questions,
            seed=args.seed,
            grader_votes=args.grader_votes,
            parallel_workers=getattr(args, "parallel_workers", 10),
            question_set=args.question_set,
        )

        if skip_learning:
            # Skip learning phase, go straight to evaluation
            report = runner.run_skip_learning(adapter, load_db_path=load_db, grader_model=args.grader_model)
        else:
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
        a = _create_adapter(args)
        if a is None:
            raise RuntimeError("Failed to create agent adapter")
        return a

    report = run_multi_seed_eval(
        agent_factory=agent_factory,
        num_turns=args.turns,
        num_questions=args.questions,
        seeds=seeds,
        grader_model=args.grader_model,
        grader_votes=args.grader_votes,
        repeats_per_seed=getattr(args, "repeats", 1),
        question_set=args.question_set,
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
        a = _create_adapter(args)
        if a is None:
            raise RuntimeError("Failed to create agent adapter")
        return a

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


def _cmd_download_dataset(args: argparse.Namespace) -> int:
    """Download a pre-built learning DB dataset from GitHub Releases."""
    from .datasets.download import download_dataset

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    name = args.dataset_name
    output_dir = Path(args.output_dir) if args.output_dir else None
    force = args.force

    try:
        path = download_dataset(name, output_dir=output_dir, force=force)
        print(f"Dataset downloaded to: {path}")
        print("\nTo use it:")
        print("  amplihack-eval run --adapter learning-agent --skip-learning \\")
        print(f"    --load-db {path}/memory_db --turns 5000 --questions 100")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_list_datasets(args: argparse.Namespace) -> int:
    """List available pre-built datasets."""
    from .datasets.download import list_datasets

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    datasets = list_datasets(include_remote=not args.local_only)

    if not datasets:
        print("No datasets found.")
        print("Download one with: amplihack-eval download-dataset <name>")
        return 0

    print(f"{'Name':<30} {'Local':<8} {'Score':<10} {'Turns':<8} {'Facts':<8}")
    print("-" * 64)
    for ds in datasets:
        name = ds.get("name", "unknown")
        local = "yes" if ds.get("local") else "no"
        score = f"{ds['baseline_score']:.2%}" if "baseline_score" in ds else "-"
        turns = str(ds.get("turns", "-"))
        facts = str(ds.get("facts_delivered", "-"))
        print(f"{name:<30} {local:<8} {score:<10} {turns:<8} {facts:<8}")

    return 0


def _cmd_continuous(args: argparse.Namespace) -> int:
    """Run continuous evaluation across single/flat/federated conditions."""
    from .core.continuous_eval import print_continuous_report, run_continuous_eval

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args.conditions.split(",")]

    report = run_continuous_eval(
        num_turns=args.turns,
        num_questions=args.questions,
        num_agents=args.agents,
        num_groups=args.groups,
        seed=args.seed,
        model=getattr(args, "model", ""),
        parallel_workers=getattr(args, "parallel_workers", 5),
        prompt_variant=getattr(args, "prompt_variant", None),
        conditions=conditions,
        repeats=getattr(args, "repeats", 3),
    )

    print_continuous_report(report)

    report_path = output_dir / "continuous_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nReport saved to {report_path}")

    return 0


def _create_adapter(args: argparse.Namespace):
    """Create an agent adapter based on CLI args."""
    adapter_type = getattr(args, "adapter", "http")
    load_db = getattr(args, "load_db", "")

    if adapter_type == "learning-agent":
        from .adapters.learning_agent import LearningAgentAdapter

        storage_path = load_db if load_db else "/tmp/eval_memory_db"
        prompt_variant = getattr(args, "prompt_variant", None)
        kwargs = {}
        if prompt_variant is not None:
            kwargs["prompt_variant"] = prompt_variant
        return LearningAgentAdapter(
            model=getattr(args, "model", ""),
            storage_path=storage_path,
            **kwargs,
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

    elif adapter_type == "distributed-hive":
        from .adapters.distributed_hive_adapter import DistributedHiveAdapter

        conn_str = getattr(args, "connection_string", "") or os.environ.get("AMPLIHACK_EH_CONNECTION_STRING", "")
        if not conn_str:
            print(
                "Error: --connection-string or AMPLIHACK_EH_CONNECTION_STRING required for distributed-hive adapter",
                file=sys.stderr,
            )
            return None
        return DistributedHiveAdapter(
            connection_string=conn_str,
            input_hub=getattr(args, "input_hub", "hive-events"),
            response_hub=getattr(args, "response_hub", "eval-responses"),
            agent_count=getattr(args, "agents", 5),
            answer_timeout=getattr(args, "answer_timeout", 0),
        )

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
    run_parser.add_argument("--questions", type=int, default=20, help="Quiz questions (default: 20)")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument(
        "--question-set",
        choices=QUESTION_SET_CHOICES,
        default="standard",
        help="Deterministic question subset to use (default: standard)",
    )
    run_parser.add_argument("--grader-votes", type=int, default=3, help="Grading votes per question")
    run_parser.add_argument("--grader-model", default="", help="Model for grading")
    run_parser.add_argument("--model", default="", help="Agent model")
    run_parser.add_argument("--output-dir", default="/tmp/amplihack-eval", help="Output directory")
    run_parser.add_argument(
        "--adapter",
        choices=["http", "subprocess", "learning-agent", "distributed-hive"],
        default="http",
        help="Agent adapter type",
    )
    run_parser.add_argument("--agent-url", default="http://localhost:8000", help="Agent HTTP URL")
    run_parser.add_argument("--agent-command", default="", help="Agent subprocess command")
    # Distributed hive adapter args
    run_parser.add_argument("--connection-string", default="", help="Event Hubs connection string")
    run_parser.add_argument("--input-hub", default="hive-events", help="Input Event Hub name")
    run_parser.add_argument("--response-hub", default="eval-responses", help="Response Event Hub name")
    run_parser.add_argument("--agents", type=int, default=5, help="Number of deployed agents")
    run_parser.add_argument("--answer-timeout", type=int, default=0, help="Answer timeout (0=none)")
    run_parser.add_argument(
        "--parallel-workers",
        type=int,
        default=10,
        help="Number of parallel workers for question answering/grading (1=sequential, max 20)",
    )
    run_parser.add_argument(
        "--skip-learning",
        action="store_true",
        help="Skip learning phase (requires --load-db)",
    )
    run_parser.add_argument(
        "--load-db",
        default="",
        help="Path to a pre-built memory DB to load instead of learning",
    )
    run_parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeats per seed for intra-seed variance (default: 1)",
    )
    run_parser.add_argument(
        "--prompt-variant",
        type=int,
        default=None,
        help="Prompt variant number (1-5) for testing different system prompts",
    )

    # --- compare ---
    cmp_parser = subparsers.add_parser("compare", help="Multi-seed comparison")
    cmp_parser.add_argument("--turns", type=int, default=100, help="Dialogue turns")
    cmp_parser.add_argument("--questions", type=int, default=20, help="Quiz questions")
    cmp_parser.add_argument("--seeds", default="42,123,456,789", help="Comma-separated seeds")
    cmp_parser.add_argument(
        "--question-set",
        choices=QUESTION_SET_CHOICES,
        default="standard",
        help="Deterministic question subset to use (default: standard)",
    )
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
    cmp_parser.add_argument(
        "--parallel-workers",
        type=int,
        default=10,
        help="Number of parallel workers for question answering/grading (1=sequential, max 20)",
    )
    cmp_parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeats per seed for intra-seed variance (default: 1)",
    )

    # --- self-improve ---
    si_parser = subparsers.add_parser("self-improve", help="Run self-improvement loop")
    si_parser.add_argument("--turns", type=int, default=100, help="Dialogue turns")
    si_parser.add_argument("--questions", type=int, default=20, help="Quiz questions")
    si_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    si_parser.add_argument("--iterations", type=int, default=3, help="Max iterations")
    si_parser.add_argument("--threshold", type=float, default=0.7, help="Failure threshold")
    si_parser.add_argument("--regression-threshold", type=float, default=5.0, help="Regression threshold (pp)")
    si_parser.add_argument("--grader-model", default="", help="Grader model")
    si_parser.add_argument("--model", default="", help="Agent model")
    si_parser.add_argument("--output-dir", default="/tmp/amplihack-eval-improve", help="Output dir")
    si_parser.add_argument(
        "--adapter",
        choices=["http", "subprocess", "learning-agent"],
        default="http",
        help="Agent adapter type",
    )
    si_parser.add_argument("--agent-url", default="http://localhost:8000", help="Agent HTTP URL")
    si_parser.add_argument("--agent-command", default="", help="Agent subprocess command")
    si_parser.add_argument(
        "--parallel-workers",
        type=int,
        default=10,
        help="Number of parallel workers for question answering/grading (1=sequential, max 20)",
    )

    # --- report ---
    rpt_parser = subparsers.add_parser("report", help="Print a saved report")
    rpt_parser.add_argument("report_file", help="Path to report JSON file")

    # --- download-dataset ---
    dl_parser = subparsers.add_parser("download-dataset", help="Download a pre-built learning DB dataset")
    dl_parser.add_argument("dataset_name", help="Dataset name (e.g., 5000t-seed42-v1.0)")
    dl_parser.add_argument("--output-dir", default="", help="Output directory (default: datasets/ in repo root)")
    dl_parser.add_argument("--force", action="store_true", help="Re-download even if already exists")

    # --- list-datasets ---
    ls_parser = subparsers.add_parser("list-datasets", help="List available datasets")
    ls_parser.add_argument("--local-only", action="store_true", help="Only show locally available datasets")

    # --- continuous ---
    cont_parser = subparsers.add_parser(
        "continuous",
        help="Run continuous eval comparing single/flat/federated hive conditions",
    )
    cont_parser.add_argument("--turns", type=int, default=100, help="Dialogue turns (default: 100)")
    cont_parser.add_argument("--questions", type=int, default=50, help="Quiz questions (default: 50)")
    cont_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    cont_parser.add_argument("--agents", type=int, default=5, help="Number of agents for hive conditions")
    cont_parser.add_argument("--groups", type=int, default=2, help="Number of groups for federated")
    cont_parser.add_argument("--model", default="", help="Agent/grader model")
    cont_parser.add_argument("--output-dir", default="/tmp/amplihack-eval-continuous", help="Output dir")
    cont_parser.add_argument(
        "--prompt-variant",
        type=int,
        default=None,
        help="Prompt variant number (1-5) for testing different system prompts",
    )
    cont_parser.add_argument(
        "--conditions",
        default="single,flat,federated",
        help="Comma-separated conditions to run (default: single,flat,federated)",
    )
    cont_parser.add_argument(
        "--parallel-workers",
        type=int,
        default=5,
        help="Parallel workers for Q&A grading (default: 5)",
    )
    cont_parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of evaluation repeats per condition; median taken as primary score (default: 3)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "run": _cmd_run,
        "compare": _cmd_compare,
        "self-improve": _cmd_self_improve,
        "report": _cmd_report,
        "download-dataset": _cmd_download_dataset,
        "list-datasets": _cmd_list_datasets,
        "continuous": _cmd_continuous,
    }

    handler = handlers.get(args.command)
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
