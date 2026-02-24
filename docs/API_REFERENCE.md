# API Reference

Complete reference for all public classes and functions in `amplihack-agent-eval`.

## Top-Level Exports (`amplihack_eval`)

```python
from amplihack_eval import (
    AgentAdapter,       # Abstract base class for agents
    AgentResponse,      # Response wrapper with tool calls and metadata
    ToolCall,           # Single tool call record
    EvalRunner,         # Long-horizon memory evaluation runner
    EvalResult,         # Per-question evaluation result
    EvalReport,         # Aggregate evaluation report
    CategoryBreakdown,  # Per-category score summary
    DimensionScore,     # Score on a single grading dimension
    LevelResult,        # Result of a single YAML level
    SuiteResult,        # Result of multiple YAML levels
    GradeResult,        # Standalone grading result
    grade_answer,       # Standalone grading function
    run_level,          # Run a single YAML level
    run_suite,          # Run a suite of YAML levels
)
```

---

## Adapters (`amplihack_eval.adapters`)

### `AgentAdapter` (abstract base class)

The interface that all agents must implement to be evaluable.

```python
from amplihack_eval.adapters.base import AgentAdapter

class AgentAdapter(ABC):
    def learn(self, content: str) -> None: ...
    def answer(self, question: str) -> AgentResponse: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...

    @property
    def capabilities(self) -> set[str]: ...   # Default: {"memory"}
    @property
    def name(self) -> str: ...                # Default: class name
```

**Methods:**

| Method | Description |
|--------|-------------|
| `learn(content: str)` | Feed content for memorization. Called once per dialogue turn. |
| `answer(question: str) -> AgentResponse` | Ask a question and return a structured response. |
| `reset()` | Reset state between evaluation runs. |
| `close()` | Clean up resources (connections, temp files). |

**Properties:**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `capabilities` | `set[str]` | `{"memory"}` | Declared capabilities. Possible: `"memory"`, `"tool_use"`, `"planning"`. |
| `name` | `str` | Class name | Human-readable identifier. |

---

### `AgentResponse`

Wraps an agent's response with optional tool calls, reasoning, and metadata.

```python
@dataclass
class AgentResponse:
    answer: str                            # The answer text
    tool_calls: list[ToolCall] = []        # Tool call trajectory
    reasoning_trace: str = ""              # Chain-of-thought log
    confidence: float = 0.0                # Self-reported confidence (0.0-1.0)
    metadata: dict[str, Any] = {}          # Arbitrary key-value pairs
```

---

### `ToolCall`

Records a single tool invocation within an agent's trajectory.

```python
@dataclass
class ToolCall:
    tool_name: str                 # Name of the tool called
    arguments: dict[str, Any]      # Arguments passed to the tool
    result: str                    # Tool return value
    timestamp: float = 0.0        # Unix timestamp of the call
```

---

### `HttpAdapter`

Adapter for agents with REST API endpoints.

```python
from amplihack_eval.adapters.http_adapter import HttpAdapter

adapter = HttpAdapter(
    base_url: str,                         # Base URL (trailing slash stripped)
    learn_endpoint: str = "/learn",        # POST endpoint for learning
    answer_endpoint: str = "/answer",      # POST endpoint for answering
    reset_endpoint: str = "/reset",        # POST endpoint for reset
    timeout: float = 30.0,                 # Request timeout in seconds
)
```

**Endpoint contracts:**
- `POST /learn` -- Body: `{"content": "..."}` -- Response: any (ignored)
- `POST /answer` -- Body: `{"question": "..."}` -- Response: `{"answer": "...", "confidence": 0.8, ...}`
- `POST /reset` -- Body: empty -- Response: any (ignored)

---

### `SubprocessAdapter`

Adapter for CLI-based agents invoked via subprocess.

```python
from amplihack_eval.adapters.subprocess_adapter import SubprocessAdapter

adapter = SubprocessAdapter(
    command: list[str],                    # Base command (e.g., ["python", "agent.py"])
    learn_flag: str = "--learn",           # Flag for learn mode
    answer_flag: str = "--answer",         # Flag for answer mode
    timeout: float = 30.0,                 # Per-call timeout in seconds
)
```

**Protocol:** Content/question is passed via stdin. Response is captured from stdout.

---

### `LearningAgentAdapter`

Adapter for amplihack's built-in LearningAgent.

```python
from amplihack_eval.adapters.learning_agent import LearningAgentAdapter

adapter = LearningAgentAdapter(
    model: str = "",                       # LLM model (uses EVAL_MODEL env if empty)
    memory_backend: str = "sqlite",        # Memory backend type
)
```

---

## Core (`amplihack_eval.core`)

### `EvalRunner`

Main evaluation runner. Generates dialogue, feeds to agent, quizzes, grades.

```python
runner = EvalRunner(
    num_turns: int = 1000,         # Number of dialogue turns
    num_questions: int = 100,      # Number of quiz questions
    seed: int = 42,                # Random seed
    grader_votes: int = 3,         # Number of grading votes per question
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `generate()` | `(GroundTruth, list[Question])` | Generate dialogue and questions |
| `run_dialogue(agent, ground_truth=None)` | `float` | Feed turns to agent, return learning time (seconds) |
| `run(agent, grader_model="")` | `EvalReport` | Complete pipeline: generate + learn + quiz + grade |

---

### `EvalResult`

Result for a single question.

```python
@dataclass
class EvalResult:
    question_id: str                   # Unique question identifier
    question_text: str                 # The question asked
    category: str                      # Question category
    expected_answer: str               # Ground truth answer
    actual_answer: str                 # Agent's answer
    dimensions: list[DimensionScore]   # Per-dimension scores
    overall_score: float               # Average of dimension scores (0.0-1.0)
    grading_time_s: float = 0.0        # Time spent grading this question
```

---

### `DimensionScore`

```python
@dataclass
class DimensionScore:
    dimension: str         # Dimension name (e.g., "factual_accuracy")
    score: float           # 0.0 to 1.0
    reasoning: str = ""    # Grader's reasoning for the score
```

---

### `CategoryBreakdown`

```python
@dataclass
class CategoryBreakdown:
    category: str                              # Category name
    num_questions: int                         # Number of questions
    avg_score: float                           # Average score
    min_score: float                           # Minimum score
    max_score: float                           # Maximum score
    dimension_averages: dict[str, float] = {}  # Per-dimension averages
```

---

### `EvalReport`

```python
@dataclass
class EvalReport:
    num_turns: int
    num_questions: int
    total_facts_delivered: int
    learning_time_s: float
    questioning_time_s: float
    grading_time_s: float
    overall_score: float                           # 0.0-1.0
    category_breakdown: list[CategoryBreakdown]
    results: list[EvalResult]
    memory_stats: dict[str, Any] = {}
```

**Methods:** `to_dict() -> dict[str, Any]` -- JSON-serializable dictionary.

---

### `LevelResult`

```python
@dataclass
class LevelResult:
    level_id: str
    level_name: str
    passed: bool
    overall_score: float
    pass_threshold: float
    results: list[EvalResult]
    category_breakdown: list[CategoryBreakdown]
```

---

### `SuiteResult`

```python
@dataclass
class SuiteResult:
    level_results: list[LevelResult]
    overall_score: float
    passed_count: int
    total_count: int
    skipped: list[str]     # Levels skipped due to failed prerequisites
```

---

### `run_level()`

```python
def run_level(
    level_id: str,             # e.g., "L01"
    agent: AgentAdapter,
    grader_model: str = "",
) -> LevelResult
```

---

### `run_suite()`

```python
def run_suite(
    level_ids: list[str],
    agent: AgentAdapter,
    grader_model: str = "",
) -> SuiteResult
```

---

### `grade_answer()`

```python
def grade_answer(
    actual_answer: str,
    expected_answer: str,
    level: str = "",
    grading_model: str = "",
) -> GradeResult
```

---

## Data Generation (`amplihack_eval.data`)

### `generate_dialogue()`

```python
def generate_dialogue(num_turns: int = 1000, seed: int = 42) -> GroundTruth
```

### `generate_questions()`

```python
def generate_questions(ground_truth: GroundTruth, num_questions: int = 100) -> list[Question]
```

### Key Dataclasses

**`GroundTruth`**: `turns`, `facts_by_entity`, `current_values`, `superseded_values`

**`Turn`**: `turn_number`, `content`, `block`, `block_name`, `facts`

**`Question`**: `question_id`, `text`, `expected_answer`, `category`, `relevant_turns`, `scoring_dimensions`, `chain_length`, `rubric`

**`GradingRubric`**: `required_keywords`, `acceptable_paraphrases`, `incorrect_patterns`, `dimension_weights`

### Progressive Levels

```python
from amplihack_eval.data.progressive_levels import (
    ALL_LEVELS,             # L1-L6
    TEACHER_STUDENT_LEVELS, # [L7]
    ADVANCED_LEVELS,        # [L8, L9, L10]
    NOVEL_SKILL_LEVELS,     # [L11]
    TRANSFER_LEVELS,        # [L12]
    get_level_by_id,        # (str) -> TestLevel | None
)
```

### Extended Scenarios

**L13**: `amplihack_eval.data.tool_use_scenarios` -- `ToolDefinition`, `ToolUseScenario`, `ALL_TOOL_USE_SCENARIOS`

**L14**: `amplihack_eval.data.forgetting_scenarios` -- `FactUpdate`, `ForgettingScenario`, `ALL_FORGETTING_SCENARIOS`

**L15**: `amplihack_eval.data.adversarial_scenarios` -- `KnowledgeBaseFact`, `AdversarialScenario`, `ALL_ADVERSARIAL_SCENARIOS`

**L16**: `amplihack_eval.data.decision_scenarios` -- `ContextFact`, `DecisionScenario`, `ALL_DECISION_SCENARIOS`

---

## Levels (`amplihack_eval.levels`)

### YAML Schema

**`LevelDefinition`**: `id`, `name`, `description`, `category`, `difficulty`, `questions`, `scoring`, `prerequisites`, `data_source`, `min_turns`, `grading_mode`

**`ScoringConfig`**: `pass_threshold`, `dimensions`, `weights`, `grader_votes`

**`QuestionTemplate`**: `id`, `text`, `category`, `scoring_dimensions`, `expected_answer`, `rubric`

### Loader

```python
from amplihack_eval.levels.loader import load_level, load_all_levels, validate_level
```

### Level-Specific Scorers

- `L13_tool_selection`: `ToolTrajectory`, `ToolSelectionScore`
- `L14_selective_forgetting`: `ForgettingResult`
- `L15_adversarial_recall`: `AdversarialRecallScore`
- `L16_decision_from_memory`: `DecisionScore`

---

## Self-Improvement (`amplihack_eval.self_improve`)

### `run_self_improve()`

```python
def run_self_improve(
    config: SelfImproveConfig,
    agent_factory: Callable[[], AgentAdapter],
    llm_call: Callable[[str], str] | None = None,
    project_root: Path | None = None,
) -> RunnerResult
```

### Key Classes

**`SelfImproveConfig`**: `num_turns`, `num_questions`, `seed`, `max_iterations`, `failure_threshold`, `regression_threshold`, `output_dir`, `grader_model`

**`RunnerResult`**: `config`, `iterations`, `score_progression`, `category_progression`, `total_duration_seconds`

**`IterationResult`**: `iteration`, `report`, `category_analyses`, `improvements_applied`, `patch_proposal`, `review_result`, `post_scores`, `reverted`, `revert_reason`

**`CategoryAnalysis`**: `category`, `avg_score`, `num_questions`, `failed_questions`, `bottleneck`, `suggested_fix`

### Patch Proposer

**`PatchProposal`**: `target_file`, `hypothesis`, `description`, `diff`, `expected_impact`, `risk_assessment`, `confidence`

**`PatchHistory`**: `applied_patches`, `reverted_patches`, `rejected_patches`

### Reviewer Voting

**`ReviewVote`**: `reviewer_id`, `vote`, `rationale`, `concerns`, `suggested_modifications`

**`ChallengeResponse`**: `challenge_arguments`, `proposer_response`, `concerns_addressed`, `remaining_concerns`

**`ReviewResult`**: `proposal`, `challenge`, `votes`, `decision`, `consensus_rationale`

---

## Multi-Agent Evaluation (`amplihack_eval.multi_agent_eval`)

### `EvalCoordinator`

```python
coordinator = EvalCoordinator(grader_agents=3, enable_adversary=True)
report = coordinator.run_eval(agent, EvalConfig(num_turns=100))
```

### `MultiAgentEvalPipeline`

```python
pipeline = MultiAgentEvalPipeline()
report = pipeline.run(PipelineConfig(adversarial_rounds=2))
```

### Agent Types

- **`GraderAgent`** -- perspectives: `"factual"`, `"reasoning"`, `"completeness"`
- **`AdversaryAgent`** -- generates targeted hard questions
- **`AnalystAgent`** -- produces `AnalysisReport`, `ComparisonReport`, `Improvement`
