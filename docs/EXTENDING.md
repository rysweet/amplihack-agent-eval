# Extending the Eval Framework

This guide explains how to extend `amplihack-agent-eval` with custom agent adapters, new question categories, custom data generators, and new grading dimensions.

## Writing a Custom AgentAdapter

The `AgentAdapter` is the interface between the eval framework and your agent. To evaluate any agent, implement this abstract base class.

### Minimal Implementation

```python
from amplihack_eval.adapters.base import AgentAdapter, AgentResponse

class MyAgent(AgentAdapter):
    """Adapter for my custom agent."""

    def __init__(self, model_name: str = "my-model"):
        self._model = model_name
        self._memory: list[str] = []

    def learn(self, content: str) -> None:
        """Feed content to the agent for memorization.

        Called once per dialogue turn. The agent should store this content
        in whatever memory system it uses.
        """
        self._memory.append(content)

    def answer(self, question: str) -> AgentResponse:
        """Ask the agent a question and return its response.

        Must return an AgentResponse, which wraps the answer text along
        with optional tool calls, reasoning trace, confidence, and metadata.
        """
        # Your agent's actual answering logic here
        answer_text = self._my_query_logic(question)
        return AgentResponse(
            answer=answer_text,
            confidence=0.8,
            metadata={"model": self._model},
        )

    def reset(self) -> None:
        """Reset the agent's state between evaluation runs.

        Called by multi-seed evaluation and self-improvement loop to
        ensure each run starts from a clean state.
        """
        self._memory.clear()

    def close(self) -> None:
        """Clean up resources (connections, temp files, etc.).

        Called when evaluation is complete.
        """
        pass

    def _my_query_logic(self, question: str) -> str:
        # Implement your actual query logic
        return "I don't know"
```

### Adding Tool Call Tracking

If your agent uses tools, capture the trajectory in `AgentResponse`:

```python
from amplihack_eval.adapters.base import AgentResponse, ToolCall

def answer(self, question: str) -> AgentResponse:
    # Agent does tool calls
    tool_calls = []

    # Example: agent calls a search tool
    search_result = self._search(question)
    tool_calls.append(ToolCall(
        tool_name="search",
        arguments={"query": question},
        result=search_result,
        timestamp=time.time(),
    ))

    answer_text = self._generate_answer(question, search_result)

    return AgentResponse(
        answer=answer_text,
        tool_calls=tool_calls,
        reasoning_trace="Searched memory, found relevant fact, composed answer",
        confidence=0.9,
    )
```

The L13 tool selection evaluation uses `tool_calls` to score tool selection accuracy, efficiency, and chain correctness.

### Adding Custom Capabilities

Override the `capabilities` property to declare what your agent can do:

```python
@property
def capabilities(self) -> set[str]:
    return {"memory", "tool_use", "planning"}

@property
def name(self) -> str:
    return f"MyAgent({self._model})"
```

Capabilities are used by the evaluation framework to skip levels that require capabilities the agent does not have.

### Using Built-In Adapters

Three adapters are included for common integration patterns:

**HttpAdapter** -- for agents with REST API endpoints:

```python
from amplihack_eval.adapters.http_adapter import HttpAdapter

adapter = HttpAdapter(
    base_url="http://localhost:8000",
    learn_endpoint="/learn",      # POST, body: {"content": "..."}
    answer_endpoint="/answer",    # POST, body: {"question": "..."}
    reset_endpoint="/reset",      # POST, empty body
    timeout=30.0,
)
```

**SubprocessAdapter** -- for CLI-based agents:

```python
from amplihack_eval.adapters.subprocess_adapter import SubprocessAdapter

adapter = SubprocessAdapter(
    command=["python", "my_agent.py"],
    learn_flag="--learn",     # Passes content via stdin after this flag
    answer_flag="--answer",   # Passes question via stdin after this flag
    timeout=30.0,
)
```

**LearningAgentAdapter** -- for amplihack's built-in LearningAgent:

```python
from amplihack_eval.adapters.learning_agent import LearningAgentAdapter

adapter = LearningAgentAdapter(model="claude-sonnet-4-5-20250929")
```

## Adding New Question Categories

### For Progressive Levels (Python-defined)

Add a new level to `src/amplihack_eval/data/progressive_levels.py`:

```python
from amplihack_eval.data.progressive_levels import TestLevel, TestArticle, TestQuestion

LEVEL_17 = TestLevel(
    level_id="L17",
    level_name="Analogical Reasoning",
    description="Drawing analogies between structurally similar scenarios in different domains",
    articles=[
        TestArticle(
            title="How Ant Colonies Solve Optimization Problems",
            content=(
                "Ant colonies use pheromone trails to find shortest paths to food. "
                "When a shorter path exists, more ants use it (stronger pheromone), "
                "creating a positive feedback loop. This is analogous to ..."
            ),
            url="https://example.com/ant-colonies",
            published="2026-01-01T00:00:00Z",
        ),
        # More articles...
    ],
    questions=[
        TestQuestion(
            question="How is the ant colony optimization similar to network routing?",
            expected_answer=(
                "Both use local decisions (ants follow pheromones, packets follow routing tables) "
                "that collectively find globally optimal paths through positive feedback loops."
            ),
            level="L17",
            reasoning_type="analogical_mapping",
        ),
        # More questions...
    ],
)
```

Then add the level to the appropriate group list and re-export from `levels/__init__.py`.

### For YAML-Defined Levels

Create a YAML file in `src/amplihack_eval/levels/`:

```yaml
# L17_analogical_reasoning.yaml
id: "L17"
name: "Analogical Reasoning"
description: "Drawing analogies between structurally similar scenarios"
category: "reasoning"
difficulty: 4
data_source: "progressive_levels"
min_turns: 50
grading_mode: "hybrid"

scoring:
  pass_threshold: 0.6
  dimensions:
    - factual_accuracy
    - reasoning_quality
  weights:
    factual_accuracy: 0.4
    reasoning_quality: 0.6
  grader_votes: 3

questions:
  - id: "L17_Q01"
    text: "How is the ant colony optimization similar to network routing?"
    category: "analogical_mapping"
    expected_answer: "Both use local decisions with positive feedback loops..."
    scoring_dimensions:
      - factual_accuracy
      - reasoning_quality
    rubric:
      required_keywords: ["pheromone", "routing", "feedback"]
      acceptable_paraphrases: ["positive feedback", "reinforcement"]
```

YAML levels are automatically discovered by `load_all_levels()`.

### For Long-Horizon Categories

To add a new question category to the long-horizon evaluation, modify `generate_questions()` in `src/amplihack_eval/data/long_horizon.py`:

```python
def generate_questions(ground_truth: GroundTruth, num_questions: int = 100) -> list[Question]:
    questions = []

    # ... existing categories ...

    # Add your new category
    # Example: "entity_count" - questions about how many entities of a type exist
    for entity_type in ["people", "projects", "incidents"]:
        entity_facts = [
            f for f in ground_truth.facts_by_entity.items()
            if any(entity_type in str(v) for v in f[1])
        ]
        if entity_facts:
            questions.append(Question(
                question_id=f"entity_count_{entity_type}_001",
                text=f"How many distinct {entity_type} have been discussed?",
                expected_answer=str(len(entity_facts)),
                category="entity_count",
                relevant_turns=[],  # Meta-question, all turns relevant
                scoring_dimensions=["factual_accuracy"],
                rubric=GradingRubric(
                    required_keywords=[str(len(entity_facts))],
                ),
            ))

    return questions[:num_questions]
```

## Creating Custom Data Generators

### Template-Based Generator (Recommended)

Follow the pattern used by `long_horizon.py` -- template-based, deterministic, seeded:

```python
import random
from dataclasses import dataclass, field

@dataclass
class MyScenario:
    """A single evaluation scenario."""
    scenario_id: str
    context: str          # Content for agent.learn()
    question: str         # Question for agent.answer()
    expected_answer: str  # Ground truth
    category: str
    metadata: dict = field(default_factory=dict)


def generate_my_scenarios(num_scenarios: int = 50, seed: int = 42) -> list[MyScenario]:
    """Generate evaluation scenarios deterministically.

    Same seed = same output. No LLM needed.
    """
    rng = random.Random(seed)
    scenarios = []

    # Define templates
    templates = [
        {
            "context_template": "The {entity} was created on {date} by {creator}.",
            "question_template": "When was {entity} created?",
            "answer_template": "{date}",
            "category": "date_recall",
        },
        # More templates...
    ]

    entities = ["Project Alpha", "System Beta", "Service Gamma"]
    dates = ["2024-01-15", "2024-03-22", "2024-07-01"]
    creators = ["Alice", "Bob", "Carol"]

    for i in range(num_scenarios):
        template = rng.choice(templates)
        entity = rng.choice(entities)
        date = rng.choice(dates)
        creator = rng.choice(creators)

        scenarios.append(MyScenario(
            scenario_id=f"custom_{i:04d}",
            context=template["context_template"].format(
                entity=entity, date=date, creator=creator
            ),
            question=template["question_template"].format(entity=entity),
            expected_answer=template["answer_template"].format(date=date),
            category=template["category"],
        ))

    return scenarios
```

### Integrating with the EvalRunner

To use custom scenarios with the existing runner:

```python
from amplihack_eval import EvalRunner

# Generate your scenarios
scenarios = generate_my_scenarios(num_scenarios=50, seed=42)

# Create agent
agent = MyAgent()

# Feed scenarios as dialogue turns
for scenario in scenarios:
    agent.learn(scenario.context)

# Quiz and grade
results = []
for scenario in scenarios:
    response = agent.answer(scenario.question)
    # Use the built-in grader or your own
    score = grade_answer(
        response.answer,
        scenario.expected_answer,
        level=scenario.category,
    )
    results.append((scenario.scenario_id, score))
```

## Adding New Grading Dimensions

### Adding to the Hybrid Grader

The hybrid grader in `core/runner.py` supports arbitrary dimension names. To add a new dimension:

1. Add the dimension name to your question's `scoring_dimensions`:

```python
Question(
    ...,
    scoring_dimensions=["factual_accuracy", "specificity", "my_new_dimension"],
)
```

2. For deterministic grading, add logic in `_deterministic_grade()`:

```python
def _deterministic_grade(
    answer: str, expected: str, rubric: GradingRubric, dimension: str
) -> float | None:
    if dimension == "my_new_dimension":
        # Your deterministic scoring logic
        if "important_keyword" in answer.lower():
            return 1.0
        return 0.0
    # ... existing dimensions ...
```

3. For LLM grading, the dimension name is automatically included in the grading prompt. The LLM is instructed to score on a 0.0--1.0 scale. You can enhance the prompt with dimension-specific instructions:

```python
# In the grading prompt builder
DIMENSION_DESCRIPTIONS = {
    "factual_accuracy": "Does the answer contain the correct facts?",
    "specificity": "Does it include specific names, numbers, dates?",
    "my_new_dimension": "Does the answer demonstrate deep analytical thinking?",
}
```

### Adding a Level-Specific Scorer (L13--L16 Pattern)

For new levels that need custom scoring logic, create a scorer module:

```python
# src/amplihack_eval/levels/L17_analogical_reasoning.py

from dataclasses import dataclass

@dataclass
class AnalogyScore:
    """Score for a single analogical reasoning scenario."""
    scenario_id: str
    structural_mapping: float   # 0.0-1.0: correct structure mapped
    domain_bridging: float      # 0.0-1.0: connected domains correctly
    novel_inference: float      # 0.0-1.0: generated new insights from analogy
    overall: float              # Weighted average

def score_analogy(
    expected_mapping: dict[str, str],
    actual_answer: str,
    source_domain: str,
    target_domain: str,
) -> AnalogyScore:
    """Score an analogical reasoning response."""
    # Your scoring logic here
    ...
```

## Testing Your Extensions

Follow the testing patterns in `tests/`:

```python
# tests/test_my_extension.py
import pytest
from amplihack_eval.adapters.base import AgentAdapter, AgentResponse

class TestMyAdapter:
    def test_learn_and_answer(self):
        agent = MyAgent()
        agent.learn("The sky is blue.")
        resp = agent.answer("What color is the sky?")
        assert isinstance(resp, AgentResponse)
        assert "blue" in resp.answer.lower()

    def test_reset_clears_state(self):
        agent = MyAgent()
        agent.learn("fact 1")
        agent.reset()
        resp = agent.answer("Tell me fact 1")
        assert "don't know" in resp.answer.lower() or resp.answer == ""

    def test_capabilities(self):
        agent = MyAgent()
        assert "memory" in agent.capabilities

class TestMyScenarios:
    def test_all_scenarios_valid(self):
        for s in ALL_MY_SCENARIOS:
            assert s.scenario_id
            assert s.question
            assert s.expected_answer
            assert 1 <= s.difficulty <= 5

    def test_get_by_id(self):
        s = get_my_scenario_by_id("my_001")
        assert s is not None
        assert s.domain == "engineering"
```

Run tests:

```bash
uv run pytest tests/ -v
```

## Best Practices

1. **Deterministic data**: Always use seeded random for data generation. No LLM in data generation.

2. **Clear ground truth**: Every question must have an unambiguous expected answer.

3. **Isolated levels**: Each level should test one specific capability. Do not combine too many skills in one level.

4. **Dimension alignment**: Use `factual_accuracy` for objective facts, `specificity` for detail level, custom dimensions for nuanced skills.

5. **Test at multiple scales**: Run with 20, 100, and 500 questions to verify your category works at different scales.

6. **Document the category**: Add an entry to `CATEGORIES.md` explaining what it tests, why it matters, and example questions.
