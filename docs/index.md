# amplihack-agent-eval

Evaluation framework for goal-seeking AI agents. Tests memory recall, tool use, planning, and reasoning across progressive difficulty levels (L1-L12).

## Key Features

- **Long-horizon memory stress tests** -- Generates 1000+ turn dialogues with embedded facts, then quizzes the agent on details from various points in the conversation
- **Hybrid grading** -- Deterministic (rubric keywords) + LLM (semantic judgment) with multi-vote stability
- **Progressive difficulty levels** -- L1 (simple recall) through L12 (far transfer reasoning)
- **Agent-agnostic** -- Works with any agent through the `AgentAdapter` interface
- **Self-improvement loop** -- Automated EVAL -> ANALYZE -> PROPOSE -> CHALLENGE -> VOTE -> APPLY -> RE-EVAL cycle
- **Multi-seed holdout** -- Run across multiple random seeds to measure inter-seed variance

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
        relevant = [m for m in self.memory
                    if any(w in m.lower() for w in question.lower().split())]
        return AgentResponse(
            answer=" ".join(relevant[:3]) if relevant else "I don't know"
        )

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

## Documentation

| Guide | Description |
|-------|-------------|
| [Architecture](architecture.md) | Package layout, core concepts, and design principles |
| [Evaluation Levels](levels.md) | Complete guide to all 12 progressive difficulty levels (L1-L12) |
| [Writing Adapters](adapters.md) | How to write custom `AgentAdapter` implementations |
| [Self-Improvement Loop](self-improvement.md) | Automated improvement cycle with safety gates |
| [Multi-Agent Eval](multi-agent-eval.md) | Planned multi-agent evaluation scenarios |

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Required for LLM grading | -- |
| `GRADER_MODEL` | Model for grading | `claude-sonnet-4-5-20250929` |
| `EVAL_MODEL` | Model for LearningAgent adapter | `claude-sonnet-4-5-20250929` |

## Contributing

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

## License

MIT
