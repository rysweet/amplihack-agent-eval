# amplihack-agent-eval

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
tests/
    test_adapters.py     # Adapter interface tests
    test_data_generation.py  # Data generator tests
```

## License

MIT
