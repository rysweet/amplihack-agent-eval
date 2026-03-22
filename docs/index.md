# amplihack-agent-eval

Evaluation framework for goal-seeking AI agents. It generates long-horizon datasets, runs local and distributed evals, and grades answers with deterministic and LLM-backed scoring.

## Key Features

- **Long-horizon memory stress tests** - Generates long conversations with embedded facts, then quizzes the agent on details from different points in the conversation
- **Hybrid grading** - Deterministic rubric checks plus LLM semantic judgment with multi-vote stability
- **Deterministic question sets** - Supports `standard` and `holdout` question slices for anti-overfitting checks
- **Agent-agnostic adapters** - Works with HTTP agents, subprocess agents, in-process learning agents, and distributed hives
- **Azure distributed eval** - Runs the same harness against Azure Container Apps through Event Hubs

## Running Evals

| Guide | What It Covers |
|-------|---------------|
| **[Running Evals Quick Start](running-evals.md)** | Current command surface for local runs, comparisons, and distributed Azure evals |
| **[Distributed Hive Eval on Azure](distributed-hive-eval.md)** | Event Hubs + Azure Container Apps wrapper and direct runner |
| [Azure Hive Q&A Eval Quick Reference](azure-hive-qa-eval.md) | Minimal Azure commands when you just need to run the job |
| [Long-Horizon Memory Eval](LONG_HORIZON_EVAL.md) | Single-agent eval design, categories, and grading details |
| [Hive Mind Eval Strategy](hive-mind-eval.md) | Multi-agent topology and scoring methodology |

## Framework Documentation

| Guide | Description |
|-------|-------------|
| [Architecture](architecture.md) | Package layout, core concepts, and design principles |
| [Evaluation Levels](levels.md) | Progressive difficulty levels (L1-L12) |
| [Writing Adapters](adapters.md) | How to implement a custom `AgentAdapter` |
| [Self-Improvement Loop](self-improvement.md) | Automated improvement cycle with safety gates |
| [Multi-Agent Eval](multi-agent-eval.md) | Multi-agent evaluation scenarios |

## Installation

```bash
# Basic installation (data generation and adapters, no LLM grading)
pip install amplihack-agent-eval

# Development install
pip install -e ".[dev]"
```

## Quick Start

### Run a local eval

```bash
amplihack-eval run   --turns 100   --questions 20   --adapter learning-agent   --question-set standard   --output-dir /tmp/eval-run
```

### Compare seeds or question sets

```bash
amplihack-eval compare   --seeds 42,123,456   --turns 100   --questions 20   --question-set holdout   --output-dir /tmp/eval-compare
```

### Run the distributed Azure runner directly

```bash
python -m amplihack_eval.azure.eval_distributed   --connection-string "<event-hubs-connection-string>"   --input-hub "hive-events-amplihive3175e"   --response-hub "eval-responses-amplihive3175e"   --agents 100   --turns 5000   --questions 50   --question-set standard   --parallel-workers 1   --question-failover-retries 2   --answer-timeout 0   --output /tmp/eval_report.json
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Required for grading and for the default learning-agent adapter | unset |
| `GRADER_MODEL` | Model used for grading | `claude-sonnet-4-5-20250929` |
| `EVAL_MODEL` | Model used by the learning-agent adapter | `claude-sonnet-4-5-20250929` |

## Contributing

```bash
git clone https://github.com/rysweet/amplihack-agent-eval.git
cd amplihack-agent-eval
pip install -e ".[dev]"
pytest tests/ -q
ruff check src/ tests/
ruff format --check src/ tests/
```

## License

MIT
