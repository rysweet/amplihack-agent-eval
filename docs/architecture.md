# Architecture

## Overview

`amplihack-agent-eval` is a modular evaluation framework for goal-seeking AI agents. It tests memory recall, tool use, planning, and reasoning across progressive difficulty levels (L1-L12).

The package follows a layered architecture with clear separation of concerns:

```
                  +------------------+
                  |      CLI         |  amplihack_eval.cli
                  +--------+---------+
                           |
              +------------+------------+
              |                         |
     +--------v---------+    +---------v----------+
     |   EvalRunner      |    |  SelfImproveRunner |
     |   (core/runner)   |    |  (self_improve/)   |
     +--------+----------+    +---------+----------+
              |                         |
     +--------v---------+              |
     |   Grader          |<-------------+
     |   (core/grader)   |
     +--------+----------+
              |
     +--------v---------+
     |   Data Generation |
     |   (data/)         |
     +-------------------+

     +-------------------+
     |   AgentAdapter    |  adapters/base.py (interface)
     +--------+----------+
              |
     +--------+----------+----------+
     |        |          |          |
   HTTP   Subprocess  Learning   Custom
   Adapter  Adapter   Agent      (yours)
```

## Package Layout

```
src/amplihack_eval/
    __init__.py              # Public API exports + __version__
    py.typed                 # PEP 561 type checking marker
    cli.py                   # CLI entry point (amplihack-eval command)

    adapters/                # Agent adapter layer
        base.py              # AgentAdapter ABC, AgentResponse, ToolCall
        http_adapter.py      # REST API adapter (POST /learn, /answer, /reset)
        subprocess_adapter.py # CLI subprocess adapter
        learning_agent.py    # amplihack LearningAgent adapter

    core/                    # Evaluation engine
        runner.py            # EvalRunner: long-horizon memory stress test
        grader.py            # Hybrid deterministic + LLM semantic grading
        multi_seed.py        # Multi-seed holdout evaluation

    data/                    # Test data generation
        long_horizon.py      # 5000-turn dialogue generator
        progressive_levels.py # L1-L12 level definitions (articles + questions)

    levels/                  # Convenience re-exports of level definitions
        __init__.py          # Re-exports ALL_LEVELS, get_level_by_id, etc.

    self_improve/            # Automated self-improvement loop
        runner.py            # SelfImproveRunner: EVAL->ANALYZE->PROPOSE->VOTE->APPLY
        patch_proposer.py    # LLM-powered patch generation
        reviewer_voting.py   # 3-reviewer A/B voting system

    multi_agent_eval/        # Multi-agent scenarios (future)
        __init__.py
```

## Core Concepts

### AgentAdapter (adapters/base.py)

The central abstraction. Any agent that implements `learn()`, `answer()`, `reset()`, and `close()` can be evaluated. Adapters capture tool call trajectories and reasoning traces for deeper analysis.

### EvalRunner (core/runner.py)

Generates a long-horizon dialogue (100-5000 turns), feeds it to the agent via `learn()`, then quizzes the agent with generated questions. Each answer is graded on multiple dimensions using the hybrid grader.

Key types:
- `EvalResult`: Per-question result with dimension scores
- `EvalReport`: Aggregate report with category breakdown
- `DimensionScore`: Score on a single grading dimension
- `CategoryBreakdown`: Average scores per question category

### Grader (core/grader.py)

Two-stage grading:
1. **Deterministic**: Required keywords, acceptable paraphrases, incorrect pattern detection
2. **LLM Semantic**: When deterministic grading is insufficient, uses LLM to evaluate semantic correctness

Multi-vote stability: Grade N times, take median to reduce LLM noise.

### Data Generation (data/)

- `long_horizon.py`: Generates synthetic dialogues with embedded facts across 12 topic blocks. Facts are tracked in a `GroundTruth` structure for question generation.
- `progressive_levels.py`: Hand-crafted test levels L1-L12 with articles and questions of increasing cognitive complexity.

### Self-Improvement Loop (self_improve/)

Automated improvement cycle:
1. **EVAL**: Run evaluation, get per-category scores
2. **ANALYZE**: Identify worst-performing category
3. **PROPOSE**: LLM generates a patch hypothesis
4. **CHALLENGE**: Devil's advocate reviews the proposal
5. **VOTE**: 3 reviewers vote accept/reject
6. **APPLY**: If accepted, apply the patch
7. **RE-EVAL**: Run evaluation again to measure impact
8. **DECIDE**: If regression, auto-revert; if improvement, keep

### Multi-Seed Evaluation (core/multi_seed.py)

Runs the same evaluation across multiple random seeds (default: 42, 123, 456, 789) to measure inter-seed variance. Questions with >10 percentage point variance are flagged as noisy.

## Design Principles

1. **Agent-agnostic**: The `AgentAdapter` interface makes any agent evaluable
2. **Deterministic data**: Same seed produces identical dialogues and questions
3. **Hybrid grading**: Deterministic rubrics + LLM semantic judgment
4. **Progressive difficulty**: L1 (simple recall) through L12 (far transfer)
5. **Safety**: Self-improvement never modifies grader, test data, or safety constraints
6. **Reproducibility**: All results are logged with full configuration
