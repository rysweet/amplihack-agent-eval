# Architecture

## Overview

`amplihack-agent-eval` is a modular evaluation framework for goal-seeking AI agents. It measures memory recall, tool use, planning, and reasoning across progressive difficulty levels (L1--L16) and long-horizon memory stress tests (up to 5000 dialogue turns). The package is designed to be agent-agnostic: any system that can learn content and answer questions is evaluable through the `AgentAdapter` interface.

## High-Level Architecture

```
                        +---------------------+
                        |        CLI          |
                        |  amplihack_eval.cli |
                        +--------+------------+
                                 |
            +--------------------+--------------------+
            |                    |                    |
   +--------v----------+ +------v-----------+ +------v-----------+
   |    EvalRunner      | | SelfImproveRunner| | MultiSeedRunner  |
   |   (core/runner)    | | (self_improve/)  | | (core/multi_seed)|
   +--------+-----------+ +------+-----------+ +---------+--------+
            |                    |                       |
            +--------------------+-----------+-----------+
                                 |           |
                        +--------v--------+  |
                        |     Grader      |  |
                        | (core/grader +  |<-+
                        |  runner inline) |
                        +--------+--------+
                                 |
                        +--------v--------+
                        | Data Generation |
                        |    (data/)      |
                        +-----------------+

   +---------------------+    +-----------------------+
   |   AgentAdapter       |    |  Multi-Agent Eval     |
   |  (adapters/base.py)  |    | (multi_agent_eval/)   |
   +-----+-----+-----+---+    | Coordinator, Grader,  |
         |     |     |        | Adversary, Analyst,   |
       HTTP  Subproc LearningAgent | Pipeline          |
       Adapter Adapter  Adapter    +-----------------------+
```

## Package Layout

```
src/amplihack_eval/
    __init__.py                  # Public API: EvalRunner, AgentAdapter, AgentResponse, ...
    cli.py                       # CLI entry point (amplihack-eval command)

    adapters/                    # Agent adapter layer
        __init__.py
        base.py                  # AgentAdapter ABC, AgentResponse, ToolCall
        http_adapter.py          # REST API adapter (POST /learn, /answer, /reset)
        subprocess_adapter.py    # CLI subprocess adapter (stdin/stdout)
        learning_agent.py        # amplihack LearningAgent wrapper

    core/                        # Evaluation engine
        __init__.py
        runner.py                # EvalRunner, LevelResult, SuiteResult, run_level, run_suite
        grader.py                # Standalone grade_answer() with multi-vote support
        multi_seed.py            # Multi-seed holdout evaluation with variance analysis

    data/                        # Test data generation
        __init__.py              # Re-exports all data modules
        long_horizon.py          # 5000-turn dialogue generator (12 blocks, 15 categories)
        progressive_levels.py    # L1-L12 Python-defined levels (articles + questions)
        tool_use_scenarios.py    # L13 tool selection scenarios
        forgetting_scenarios.py  # L14 selective forgetting scenarios
        adversarial_scenarios.py # L15 adversarial recall scenarios
        decision_scenarios.py    # L16 decision-from-memory scenarios

    levels/                      # Level schema, loader, and scoring modules
        __init__.py              # Convenience re-exports
        schema.py                # LevelDefinition, QuestionTemplate, ScoringConfig
        loader.py                # YAML-driven level loader
        L13_tool_selection.py    # Tool selection scoring logic
        L14_selective_forgetting.py  # Selective forgetting scoring logic
        L15_adversarial_recall.py    # Adversarial recall scoring logic
        L16_decision_from_memory.py  # Decision-from-memory scoring logic

    self_improve/                # Automated self-improvement loop
        __init__.py
        runner.py                # 8-phase self-improvement orchestrator
        patch_proposer.py        # LLM-powered patch generation with history
        reviewer_voting.py       # Devil's advocate + 3-reviewer A/B voting

    multi_agent_eval/            # Multi-agent evaluation pipeline
        __init__.py
        coordinator.py           # EvalCoordinator
        grader_agent.py          # GraderAgent with perspective-based grading
        adversary_agent.py       # AdversaryAgent for hard question generation
        analyst_agent.py         # AnalystAgent for failure analysis
        pipeline.py              # MultiAgentEvalPipeline end-to-end orchestrator

tests/
    test_adapters.py             # AgentAdapter interface + concrete adapter tests
    test_data_generation.py      # Data generator + progressive level tests

recipes/                         # YAML recipes (future, currently .gitkeep)
```

## Core Components

### 1. AgentAdapter (`adapters/base.py`)

The central abstraction. Four methods define the complete contract:

```
+------------------+
|   AgentAdapter   |  (abstract base class)
+------------------+
| + learn(content) |  Feed content for memorization
| + answer(question)| Ask a question, get AgentResponse
| + reset()        |  Reset state between runs
| + close()        |  Clean up resources
+------------------+
| capabilities     |  set[str] - what the agent can do
| name             |  str - human-readable name
+------------------+
```

`AgentResponse` captures more than just the answer text. It includes:
- `tool_calls: list[ToolCall]` -- the agent's tool use trajectory
- `reasoning_trace: str` -- chain-of-thought or reasoning log
- `confidence: float` -- self-reported confidence
- `metadata: dict` -- arbitrary key-value pairs (e.g., latency, model name)

**Built-in adapters:**

| Adapter               | Communication   | Use case                              |
|------------------------|-----------------|---------------------------------------|
| `HttpAdapter`          | REST API        | Any agent with HTTP endpoints         |
| `SubprocessAdapter`    | stdin/stdout    | Any CLI-invokable agent               |
| `LearningAgentAdapter` | Direct import   | amplihack's LearningAgent             |

### 2. EvalRunner (`core/runner.py`)

Orchestrates the full evaluation pipeline: generate data, feed to agent, quiz, grade.

```
EvalRunner
    |
    |-- generate()        -> (GroundTruth, list[Question])
    |-- run_dialogue()    -> feeds all turns to agent.learn()
    |-- evaluate()        -> quizzes agent, grades answers
    |-- run()             -> all three steps in sequence
```

The runner also supports YAML-driven level evaluation through `run_level()` and `run_suite()`, which load level definitions, feed articles to the agent, and grade with the level's scoring configuration.

**Key data flow:**

```
generate_dialogue(num_turns, seed)
         |
         v
    GroundTruth            # Turns with embedded facts, entity tracking
         |
         v
generate_questions(gt, num_questions)
         |
         v
    list[Question]         # Questions with expected answers, rubrics, categories
         |
         v
    agent.learn(turn.content)   # Feed each turn
         |
         v
    agent.answer(question.text) # Quiz the agent
         |
         v
    _grade_multi_vote()    # Hybrid grading with multi-vote stability
         |
         v
    EvalReport             # Aggregate scores by category and dimension
```

### 3. Grading System (`core/runner.py` inline + `core/grader.py`)

Two complementary grading subsystems:

**Runner-integrated grading** (in `runner.py`):
- `_deterministic_grade()` -- regex/keyword matching against rubrics
- `_grade_with_llm()` -- LLM semantic evaluation on multiple dimensions
- `_grade_hybrid()` -- deterministic for rubric-compatible dimensions, LLM for the rest
- `_grade_multi_vote()` -- runs hybrid grading N times, takes median per dimension

**Standalone grader** (in `grader.py`):
- `grade_answer()` -- independent grading function with level-specific criteria
- Level-aware grading prompts (L3 temporal, L5 contradiction, L9 causal, etc.)
- Multi-vote support with median aggregation

**Grading dimensions:**

| Dimension               | Deterministic? | Description                              |
|--------------------------|---------------|------------------------------------------|
| `factual_accuracy`       | Yes           | Does the answer match key facts?         |
| `specificity`            | Yes           | Does it include names, numbers, dates?   |
| `temporal_awareness`     | LLM only      | Current vs. historical value distinction |
| `source_attribution`     | LLM only      | Correct source labeling                  |
| `confidence_calibration` | LLM only      | Appropriate uncertainty expression       |

### 4. Data Generation (`data/`)

**Long-horizon generator** (`long_horizon.py`):
- Produces 100--5000 turns of structured dialogue
- 12 information blocks (people, projects, technical, evolving stories, numerical, contradictions, callbacks, distractors, security logs, incidents, infrastructure, problem-solving)
- Deterministic: same seed produces identical output
- Ground truth tracking: every fact is recorded with its delivery turn

**Progressive levels** (`progressive_levels.py`):
- Hand-crafted L1--L12 with curated articles and questions
- Each level is a `TestLevel` dataclass with articles, questions, and metadata

**Extended scenarios** (L13--L16):
- `tool_use_scenarios.py` -- tool selection/chaining scenarios with expected tool sequences
- `forgetting_scenarios.py` -- fact update scenarios testing stale data handling
- `adversarial_scenarios.py` -- plausible-but-wrong questions testing hallucination resistance
- `decision_scenarios.py` -- fact recall + reasoning + decision scenarios

### 5. Self-Improvement Loop (`self_improve/`)

An 8-phase automated improvement cycle:

```
EVAL -> ANALYZE -> PROPOSE -> CHALLENGE -> VOTE -> APPLY -> RE-EVAL -> DECIDE
  |                                                                      |
  +----------------------------------------------------------------------+
                         (iterate up to N times)
```

Three cooperating modules:
- **runner.py** -- orchestrates the 8 phases, manages iteration state, detects regression
- **patch_proposer.py** -- LLM-powered analysis of failures, generates unified diffs
- **reviewer_voting.py** -- devil's advocate challenge + 3-reviewer (quality, regression, simplicity) voting

### 6. Multi-Seed Evaluation (`core/multi_seed.py`)

Runs the same eval across multiple random seeds (default: 42, 123, 456, 789) to:
- Measure inter-seed variance for each category
- Flag noisy questions (>10 percentage point variance)
- Compute confidence intervals (mean +/- stddev)

Each seed gets a fresh agent instance to avoid cross-contamination.

### 7. Multi-Agent Evaluation Pipeline (`multi_agent_eval/`)

An advanced evaluation pipeline using specialized agent roles:
- **GraderAgent** -- grades from specific perspectives (quality, regression, simplicity)
- **AdversaryAgent** -- generates difficult questions targeting known weaknesses
- **AnalystAgent** -- analyzes results and proposes improvements
- **EvalCoordinator** -- orchestrates the multi-agent pipeline
- **MultiAgentEvalPipeline** -- end-to-end pipeline with adversarial rounds

## Design Principles

1. **Agent-agnostic**: The `AgentAdapter` interface makes any agent evaluable. No assumptions about the agent's internal architecture.

2. **Deterministic data generation**: Same seed always produces identical dialogues and questions. No LLM needed for data generation -- all content is template-based.

3. **Hybrid grading**: Deterministic rubrics for fast, cheap, reproducible scoring of factual accuracy and specificity. LLM semantic judgment for nuanced dimensions (temporal awareness, source attribution, confidence calibration).

4. **Multi-vote stability**: Grading N times and taking the median reduces LLM noise. For deterministic dimensions, multi-vote has zero overhead (same result every time).

5. **Progressive difficulty**: L1 (simple recall) through L16 (decision-from-memory). Each level isolates a specific cognitive capability.

6. **Safety-gated self-improvement**: The self-improvement loop never modifies the grader, test data, or safety constraints. Auto-revert on regression protects existing quality.

7. **Reproducibility**: Full configuration logging, JSON report output, and seeded generation ensure any result can be reproduced.

8. **Zero external dependencies for core**: The core package has no required dependencies. LLM grading requires `anthropic` (optional). The LearningAgent adapter requires `amplihack` (optional).

## Data Flow Summary

```
                    +-----------+
                    |   User    |
                    +-----+-----+
                          |
              CLI (amplihack-eval run)
              or Python API (EvalRunner)
                          |
                    +-----v-----+
                    | EvalRunner |
                    +-----+-----+
                          |
            +-------------+-------------+
            |                           |
    +-------v--------+        +--------v--------+
    | generate_dialogue|       | generate_questions|
    | (long_horizon)  |       | (long_horizon)   |
    +-------+--------+        +--------+--------+
            |                          |
    GroundTruth                 list[Question]
    (turns with facts)          (with rubrics)
            |                          |
    +-------v--------+                |
    | agent.learn()  |                |
    | (N turns)      |                |
    +-------+--------+                |
            |                         |
    +-------v-------------------------v--------+
    |          agent.answer(question)           |
    +-------+----------------------------------+
            |
    +-------v--------+
    | _grade_hybrid() |  deterministic + LLM
    | _grade_multi_vote() | N votes, median
    +-------+--------+
            |
    +-------v--------+
    |   EvalReport   |
    | (scores by     |
    |  category and  |
    |  dimension)    |
    +----------------+
```

## Environment Variables

| Variable            | Purpose                                  | Default                      |
|---------------------|------------------------------------------|------------------------------|
| `ANTHROPIC_API_KEY` | Required for LLM grading                 | (none -- grading disabled)   |
| `GRADER_MODEL`      | LLM model for grading                    | `claude-sonnet-4-5-20250929` |
| `EVAL_MODEL`        | LLM model for LearningAgent adapter      | `claude-sonnet-4-5-20250929` |
