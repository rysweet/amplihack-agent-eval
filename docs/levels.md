# Evaluation Levels (L1-L12)

## Overview

The evaluation framework uses 12 progressive difficulty levels to test different cognitive capabilities of AI agents. Each level builds on previous ones, requiring increasingly sophisticated reasoning.

## Level Categories

### Core Levels (L1-L6)

These test fundamental memory and reasoning capabilities.

#### L1: Single Source Direct Recall

- **Tests**: Basic fact retrieval from a single source
- **Reasoning type**: `direct_recall`
- **Example**: "How many total medals does Norway have?" (answer is directly stated in one article)
- **What it measures**: Can the agent store and retrieve individual facts accurately?

#### L2: Multi-Source Synthesis

- **Tests**: Combining information across multiple independent sources
- **Reasoning type**: `cross_source_synthesis`, `specific_source_attribution`
- **Example**: "How many total goals were scored across all matches?" (requires adding numbers from multiple articles)
- **What it measures**: Can the agent integrate information from different sources?

#### L3: Temporal Reasoning

- **Tests**: Understanding changes over time, computing differences
- **Reasoning type**: `temporal_comparison`, `temporal_computation`
- **Requires**: `temporal_ordering = True`
- **Example**: "How much did the stock price change between the two reports?" (requires computing a delta)
- **What it measures**: Can the agent track temporal sequences and reason about changes?

#### L4: Procedural Learning

- **Tests**: Learning and applying step-by-step procedures
- **Reasoning type**: `procedure_recall`, `conditional_procedure`
- **Example**: "What are the steps for the basic sourdough method?" (requires recalling a sequence)
- **What it measures**: Can the agent learn and reproduce procedures?

#### L5: Contradiction Handling

- **Tests**: Detecting and reasoning about conflicting information
- **Reasoning type**: `contradiction_identification`, `contradiction_resolution`
- **Example**: "What are the conflicting claims about the vaccine efficacy?" (two sources disagree)
- **What it measures**: Can the agent detect contradictions and reason about them?

#### L6: Incremental Learning

- **Tests**: Updating knowledge when new information arrives
- **Reasoning type**: `knowledge_update`, `history_awareness`
- **Requires**: `update_handling = True`
- **Example**: "What is the current CEO after the leadership change?" (requires updating a stored fact)
- **What it measures**: Can the agent update beliefs when new information supersedes old?

### Teacher-Student (L7)

#### L7: Teaching Session

- **Tests**: Agent learns material, then teaches it; graded on teaching accuracy
- **Reasoning type**: `teaching_accuracy`, `self_explanation`
- **Example**: Agent learns about photosynthesis, then must explain it clearly enough for a student to understand
- **What it measures**: Depth of understanding (the best test of knowledge is teaching it)

### Advanced Levels (L8-L10)

These test metacognitive and causal reasoning.

#### L8: Confidence Calibration

- **Tests**: Knowing what you know vs. what you do not know
- **Reasoning type**: `confidence_calibration`, `uncertainty_detection`
- **Example**: "How confident are you about X?" (agent should be uncertain when info is ambiguous)
- **What it measures**: Is the agent's confidence well-calibrated to its actual accuracy?

#### L9: Causal Reasoning

- **Tests**: Identifying causal chains and root causes
- **Reasoning type**: `causal_chain`, `root_cause`
- **Example**: "What caused the server outage?" (requires tracing a chain of events)
- **What it measures**: Can the agent identify cause-and-effect relationships?

#### L10: Counterfactual Reasoning

- **Tests**: "What if X didn't happen?" reasoning
- **Reasoning type**: `counterfactual`, `alternative_outcome`
- **Example**: "What would have happened if the backup system had worked?" (requires hypothetical reasoning)
- **What it measures**: Can the agent reason about alternative scenarios?

### Novel Skills (L11-L12)

These test the agent's ability to learn genuinely new capabilities.

#### L11: Novel Skill Acquisition

- **Tests**: Learning genuinely new skills from documentation
- **Reasoning type**: `skill_application`, `novel_syntax`
- **Example**: Learn a made-up programming syntax, then write code in it
- **What it measures**: Can the agent learn and apply truly novel skills?

#### L12: Far Transfer

- **Tests**: Applying learned reasoning patterns to new domains
- **Reasoning type**: `cross_domain_transfer`, `analogical_reasoning`
- **Example**: Learn a pattern in domain A, apply it to solve a problem in domain B
- **What it measures**: Can the agent abstract and transfer knowledge across domains?

## Data Structure

Each level is defined as a `TestLevel` dataclass:

```python
@dataclass
class TestLevel:
    level_id: str                       # "L1", "L2", etc.
    level_name: str                     # Human-readable name
    description: str                    # What the level tests
    articles: list[TestArticle]         # Source content
    questions: list[TestQuestion]       # Evaluation questions
    requires_temporal_ordering: bool    # Does order matter?
    requires_update_handling: bool      # Does info get superseded?
```

## Accessing Levels Programmatically

```python
from amplihack_eval.levels import ALL_LEVELS, get_level_by_id

# Get all core levels (L1-L6)
for level in ALL_LEVELS:
    print(f"{level.level_id}: {level.level_name} ({len(level.questions)} questions)")

# Get a specific level
l3 = get_level_by_id("L3")
print(f"L3 has {len(l3.articles)} articles and {len(l3.questions)} questions")

# Access level groups
from amplihack_eval.levels import (
    TEACHER_STUDENT_LEVELS,  # [L7]
    ADVANCED_LEVELS,         # [L8, L9, L10]
    NOVEL_SKILL_LEVELS,      # [L11]
    TRANSFER_LEVELS,         # [L12]
)
```

## Adding New Levels

To add a new evaluation level (e.g., L13):

1. Define the level in `src/amplihack_eval/data/progressive_levels.py`:

```python
LEVEL_13 = TestLevel(
    level_id="L13",
    level_name="Your Level Name",
    description="What this level tests",
    articles=[
        TestArticle(
            title="Source Article",
            content="Article content with facts...",
            url="https://example.com/article",
            published="2026-01-01T00:00:00Z",
        )
    ],
    questions=[
        TestQuestion(
            question="A question about the content",
            expected_answer="The expected answer",
            level="L13",
            reasoning_type="your_reasoning_type",
        )
    ],
)
```

2. Add it to the appropriate level group list
3. Re-export it from `src/amplihack_eval/levels/__init__.py`
4. Add tests in `tests/test_data_generation.py`
