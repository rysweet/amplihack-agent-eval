# amplihack-agent-eval

![CI](https://github.com/rysweet/amplihack-agent-eval/actions/workflows/ci.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Evaluation framework for goal-seeking AI agents. Tests memory recall, tool use, planning, and reasoning across progressive difficulty levels (L1-L12).

## What it does

- **Long-horizon memory stress tests**: Generates 1000+ turn dialogues with embedded facts, then quizzes the agent on details from various points in the conversation
- **Hybrid grading**: Deterministic (rubric keywords) + LLM (semantic judgment) with multi-vote stability
- **Progressive difficulty levels**: L1 (simple recall) through L12 (far transfer reasoning)
- **Agent-agnostic**: Works with any agent through the `AgentAdapter` interface
- **Self-improvement loop**: Automated EVAL -> ANALYZE -> PROPOSE -> CHALLENGE -> VOTE -> APPLY -> RE-EVAL cycle
- **Multi-seed holdout**: Run across multiple random seeds to measure inter-seed variance

## Installation

```bash
# Basic installation (data generation and adapters, no LLM grading)
pip install amplihack-agent-eval

# With Anthropic grading support
pip install amplihack-agent-eval[anthropic]

# Development
pip install amplihack-agent-eval[dev]

# Everything
pip install amplihack-agent-eval[all,dev]
```

## Quick Start

### 1. Implement the AgentAdapter interface

```python
from amplihack_eval import AgentAdapter, AgentResponse

class MyMemoryAgent(AgentAdapter):
    def __init__(self):
        self.memory = []

    def learn(self, content: str) -> None:
        self.memory.append(content)

    def answer(self, question: str) -> AgentResponse:
        # Your agent's retrieval + reasoning logic here
        relevant = [m for m in self.memory if any(w in m.lower() for w in question.lower().split())]
        return AgentResponse(answer=" ".join(relevant[:3]) if relevant else "I don't know")

    def reset(self) -> None:
        self.memory.clear()

    def close(self) -> None:
        pass
```

### 2. Run an evaluation

```python
from amplihack_eval import EvalRunner

agent = MyMemoryAgent()
runner = EvalRunner(num_turns=100, num_questions=20, grader_votes=3)
report = runner.run(agent)

print(f"Overall score: {report.overall_score:.2%}")
for cb in report.category_breakdown:
    print(f"  {cb.category}: {cb.avg_score:.2%}")
```

### 3. CLI usage

```bash
# Run eval against an HTTP agent
amplihack-eval run --turns 100 --questions 20 --adapter http --agent-url http://localhost:8000

# Run eval with amplihack's LearningAgent
amplihack-eval run --turns 100 --questions 20 --adapter learning-agent

# Multi-seed comparison
amplihack-eval compare --seeds 42,123,456,789 --turns 100

# Self-improvement loop
amplihack-eval self-improve --iterations 5 --turns 100
```

## Agent Adapter Interface

The `AgentAdapter` is the core abstraction. Implement these four methods to make any agent evaluable:

| Method | Purpose |
|--------|---------|
| `learn(content: str)` | Feed content to the agent for learning/memorization |
| `answer(question: str) -> AgentResponse` | Ask the agent a question |
| `reset()` | Reset agent state between eval runs |
| `close()` | Clean up resources |

Optional properties:
- `capabilities -> set[str]`: Declare what the agent can do (default: `{"memory"}`)
- `name -> str`: Human-readable name (default: class name)

### Built-in Adapters

| Adapter | Use case |
|---------|----------|
| `HttpAdapter` | Any agent with REST API (`POST /learn`, `POST /answer`, `POST /reset`) |
| `SubprocessAdapter` | Any agent invokable via CLI subprocess |
| `LearningAgentAdapter` | amplihack's LearningAgent (requires `amplihack` package) |

See [docs/adapters.md](docs/adapters.md) for the complete adapter writing guide.

## Available Eval Levels

### Core Levels (L1-L6)
| Level | Name | Tests |
|-------|------|-------|
| L1 | Single Source Direct Recall | Basic fact retrieval from a single source |
| L2 | Multi-Source Synthesis | Combining information across multiple sources |
| L3 | Temporal Reasoning | Understanding changes over time, computing differences |
| L4 | Procedural Learning | Learning and applying step-by-step procedures |
| L5 | Contradiction Handling | Detecting and reasoning about conflicting information |
| L6 | Incremental Learning | Updating knowledge when new information arrives |

### Teacher-Student (L7)
| Level | Name | Tests |
|-------|------|-------|
| L7 | Teaching Session | Agent learns, then teaches; graded on teaching accuracy |

### Advanced Levels (L8-L10)
| Level | Name | Tests |
|-------|------|-------|
| L8 | Confidence Calibration | Knowing what you know vs. don't know |
| L9 | Causal Reasoning | Identifying causal chains and root causes |
| L10 | Counterfactual Reasoning | "What if X didn't happen?" reasoning |

### Novel Skills (L11-L12)
| Level | Name | Tests |
|-------|------|-------|
| L11 | Novel Skill Acquisition | Learning genuinely new skills from documentation |
| L12 | Far Transfer | Applying learned reasoning patterns to new domains |

See [docs/levels.md](docs/levels.md) for detailed descriptions of each level.

## API Reference

### Core Classes

#### `EvalRunner`

Main evaluation runner for long-horizon memory stress tests.

```python
runner = EvalRunner(
    num_turns=100,       # Number of dialogue turns to generate
    num_questions=20,    # Number of questions to ask
    grader_votes=3,      # Multi-vote grading (take median)
    seed=42,             # Random seed for reproducibility
)
report = runner.run(agent)  # Returns EvalReport
```

#### `EvalReport`

Aggregate evaluation results.

```python
report.overall_score        # float: 0.0 to 1.0
report.results              # list[EvalResult]: per-question results
report.category_breakdown   # list[CategoryBreakdown]: per-category averages
report.metadata             # dict: run configuration
```

#### `EvalResult`

Per-question evaluation result.

```python
result.question_id          # str: unique question identifier
result.question_text        # str: the question asked
result.expected_answer      # str: ground truth answer
result.actual_answer        # str: agent's answer
result.overall_score        # float: 0.0 to 1.0
result.dimensions           # list[DimensionScore]: per-dimension scores
result.category             # str: question category
```

#### `GradeResult`

Result from the hybrid grader.

```python
grade = grade_answer(question, expected, actual, votes=3)
grade.score                 # float: 0.0 to 1.0
grade.reasoning             # str: explanation of the grade
grade.vote_scores           # list[float] | None: individual vote scores
```

### Data Generation

```python
from amplihack_eval.data import generate_dialogue, generate_questions

# Generate a reproducible dialogue
ground_truth = generate_dialogue(num_turns=100, seed=42)

# Generate questions from the dialogue
questions = generate_questions(ground_truth, num_questions=20)
```

### Self-Improvement

```python
from amplihack_eval.self_improve.runner import SelfImproveConfig, SelfImproveRunner

config = SelfImproveConfig(max_iterations=5, num_turns=100)
runner = SelfImproveRunner(config)
result = runner.run(agent_factory=lambda: MyAgent())
```

See [docs/self-improvement.md](docs/self-improvement.md) for the complete self-improvement guide.

## How to Write a Custom Adapter

```python
from amplihack_eval import AgentAdapter, AgentResponse, ToolCall

class MyCustomAgent(AgentAdapter):
    """Adapter for my custom agent."""

    def __init__(self, config):
        self.config = config
        self.client = MyAgentClient(config)

    def learn(self, content: str) -> None:
        self.client.ingest(content)

    def answer(self, question: str) -> AgentResponse:
        result = self.client.query(question)
        return AgentResponse(
            answer=result.text,
            tool_calls=[
                ToolCall(
                    tool_name=tc.name,
                    arguments=tc.args,
                    result=tc.output,
                )
                for tc in result.tool_calls
            ],
            reasoning_trace=result.chain_of_thought,
            confidence=result.confidence,
        )

    def reset(self) -> None:
        self.client.clear_memory()

    def close(self) -> None:
        self.client.shutdown()

    @property
    def capabilities(self) -> set[str]:
        return {"memory", "tool_use", "planning"}

    @property
    def name(self) -> str:
        return f"MyAgent(v{self.config.version})"
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Required for LLM grading | - |
| `GRADER_MODEL` | Model for grading | `claude-sonnet-4-5-20250929` |
| `EVAL_MODEL` | Model for LearningAgent adapter | `claude-sonnet-4-5-20250929` |

## Project Structure

```
src/amplihack_eval/
    __init__.py          # Public API exports
    cli.py               # CLI entry point (amplihack-eval)
    py.typed             # PEP 561 type checking marker
    adapters/
        base.py          # AgentAdapter interface + ToolCall + AgentResponse
        http_adapter.py  # HTTP REST adapter
        subprocess_adapter.py  # CLI subprocess adapter
        learning_agent.py     # amplihack LearningAgent adapter
    core/
        runner.py        # EvalRunner (long-horizon memory eval)
        grader.py        # Hybrid deterministic + LLM grading
        multi_seed.py    # Multi-seed holdout evaluation
    data/
        long_horizon.py  # 5000-turn dialogue generator
        progressive_levels.py  # L1-L12 level definitions
    self_improve/
        runner.py        # Self-improvement loop orchestrator
        patch_proposer.py    # LLM-powered patch generation
        reviewer_voting.py   # 3-reviewer A/B voting
    levels/              # Convenience re-export of level definitions
    multi_agent_eval/    # Multi-agent scenarios (future)
docs/
    index.md             # Documentation landing page (GitHub Pages)
    architecture.md      # Package architecture overview
    adapters.md          # How to write custom AgentAdapters
    levels.md            # Complete guide to all eval levels
    self-improvement.md  # How the self-improvement loop works
    multi-agent-eval.md  # Multi-agent eval architecture
tests/
    test_adapters.py     # Adapter interface tests
    test_data_generation.py  # Data generator tests
```

## Contributing

### Development Setup

```bash
# Clone the repository
git clone https://github.com/rysweet/amplihack-agent-eval.git
cd amplihack-agent-eval

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -q

# Run linting
ruff check src/ tests/
ruff format --check src/ tests/
```

### Guidelines

- All code must pass `ruff check` and `ruff format` checks
- New features require tests in `tests/`
- Follow existing code patterns (dataclasses, type hints, docstrings)
- The `AgentAdapter` interface is the public contract -- changes require careful consideration
- Use `from __future__ import annotations` in all modules (Python 3.10 compatibility)

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure CI passes (lint + format + tests across Python 3.10-3.12)
4. Open a PR with a clear description of changes

## License

MIT
