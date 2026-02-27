# Long-Horizon Memory Evaluation

## What It Tests

The long-horizon memory evaluation is a stress test for AI agent memory systems. It generates a structured dialogue of up to 5000 turns, feeds each turn to the agent's `learn()` method, then quizzes the agent on details from various points in the conversation. The goal is to measure how well an agent retains, organizes, and retrieves information at scale -- far beyond what fits in a single context window.

Key capabilities tested:
- **Needle-in-haystack retrieval**: Finding specific facts buried among thousands of turns
- **Temporal evolution tracking**: Understanding how values change over time
- **Numerical precision**: Exact recall of numbers, percentages, and metrics
- **Source attribution**: Correctly attributing facts to their original sources
- **Cross-referencing**: Connecting facts across different information blocks
- **Distractor resistance**: Ignoring irrelevant information when answering
- **Meta-memory**: Reasoning about what the agent knows (and does not know)
- **Security log analysis**: Detecting patterns and anomalies in structured logs
- **Incident tracking**: Following evolving incident timelines
- **Infrastructure knowledge**: Recalling configuration details
- **Problem solving**: Applying stored knowledge to solve new problems
- **Multi-hop reasoning**: Chaining multiple facts to answer complex questions

## How Dialogue Generation Works

### Deterministic Generation

All dialogue content is template-based. No LLM is needed for data generation. The same seed always produces identical output:

```python
from amplihack_eval.data.long_horizon import generate_dialogue, generate_questions

gt = generate_dialogue(num_turns=1000, seed=42)   # Deterministic
questions = generate_questions(gt, num_questions=50)
```

### The 12 Information Blocks

The dialogue is divided into 12 thematic blocks, each allocated a proportional range of the total turns. For a 5000-turn dialogue:

```
Block  1: People              (turns    1-  250,  5%)  -- personal details, preferences
Block  2: Projects            (turns  251-  750, 10%)  -- project updates with changes
Block  3: Technical           (turns  751- 1250, 10%)  -- technical facts across 9 domains
Block  4: Evolving Story      (turns 1251- 2000, 15%)  -- story with corrections/updates
Block  5: Numerical           (turns 2001- 2500, 10%)  -- precise metrics and KPIs
Block  6: Contradictory       (turns 2501- 2900,  8%)  -- conflicting reports from sources
Block  7: Callbacks           (turns 2901- 3200,  6%)  -- references back to earlier blocks
Block  8: Distractors         (turns 3201- 3500,  6%)  -- irrelevant fun facts
Block  9: Security Logs       (turns 3501- 4000, 10%)  -- structured security events
Block 10: Incidents           (turns 4001- 4400,  8%)  -- incident reports with status updates
Block 11: Infrastructure      (turns 4401- 4750,  7%)  -- server/network inventory
Block 12: Problem Solving     (turns 4751- 5000,  5%)  -- problem descriptions with solutions
```

Turn counts scale linearly. A 100-turn dialogue uses 1/50th of each range.

### Block Details

**Block 1: People** -- 10 team members with detailed personal profiles (name, birthday, allergy, hobby, role, team, pet, hometown, favorite food, degree). Each person's facts are delivered across multiple turns with natural-language context.

**Block 2: Projects** -- 5 projects (Atlas, Beacon, Cascade, Delta, Echo) each with initial descriptions and a series of updates that change deadlines, budgets, team sizes, and project leads at specific turn offsets. This tests temporal evolution tracking.

**Block 3: Technical** -- Facts from 9 technical domains: programming, security, databases, cloud, ML/AI, DevOps, architecture, frontend. Each fact is a standalone technical statement (e.g., "PostgreSQL 16 improved parallel query performance by 40%").

**Block 4: Evolving Story** -- A multi-chapter narrative about a startup's journey with deliberate corrections and updates. Tests the agent's ability to track the most current version of facts.

**Block 5: Numerical** -- 30 precise metrics (Q1 revenue, server uptime, test coverage, API response times, etc.) with specific values and context details. Tests numerical precision.

**Block 6: Contradictory** -- 8 topics where 2-3 different sources provide conflicting claims (e.g., Q3 revenue: Finance says $5.2M, Auditor says $4.8M, Board says $5.0M). Tests the agent's ability to acknowledge and reason about contradictions.

**Block 7: Callbacks** -- References back to facts from earlier blocks, creating cross-references that require connecting information across different topics.

**Block 8: Distractors** -- 30 irrelevant fun facts (e.g., "Octopuses have three hearts and blue blood") designed to test whether the agent can filter relevant from irrelevant information.

**Block 9: Security Logs** -- Structured security events with timestamps, source IPs, event types, users, and severity levels. Includes attack patterns (brute force SSH, SQL injection, data exfiltration, C2 communication) that require pattern recognition.

**Block 10: Incidents** -- Incident reports with evolving status updates (open -> investigating -> identified -> resolved). Each incident has a timeline of events that tests temporal tracking.

**Block 11: Infrastructure** -- Server and network inventory with detailed specifications (CPU, RAM, storage, OS, location, uptime).

**Block 12: Problem Solving** -- Problem descriptions paired with solutions, testing the agent's ability to recall and apply stored problem-solving knowledge.

### Ground Truth Tracking

Every fact delivered to the agent is tracked in a `GroundTruth` structure:

```python
@dataclass
class GroundTruth:
    turns: list[Turn]                           # All dialogue turns
    facts_by_entity: dict[str, list[dict]]      # Facts indexed by entity name
    current_values: dict[str, Any]              # Latest value for each entity
    superseded_values: dict[str, list[dict]]    # Historical values with timestamps
```

Each `Turn` records:
- `turn_number` -- position in the dialogue
- `content` -- the text delivered to the agent
- `block` -- which block (1-12) this turn belongs to
- `block_name` -- human-readable block name
- `facts` -- list of ground truth facts delivered in this turn

## The 15 Question Categories

Questions are generated from the ground truth data. Each question has an expected answer, relevant turn numbers, scoring dimensions, and an optional deterministic grading rubric.

### 1. `needle_in_haystack`

**What it tests**: Direct recall of specific facts from a single source among many turns.

**Example**: "What is Sarah Chen's allergy?" (Expected: "shellfish")

**Scoring dimensions**: `factual_accuracy`, `specificity`

**Why it matters**: The most fundamental memory capability. If an agent cannot find a specific fact in its memory, nothing else works.

### 2. `temporal_evolution`

**What it tests**: Tracking how values change over time, including computing differences.

**Example**: "What is the current deadline for Project Atlas, and how many times has it changed?" (Expected: "September 20, changed twice: June 15 -> August 3 -> September 20")

**Scoring dimensions**: `factual_accuracy`, `temporal_awareness`

**Why it matters**: Real-world information evolves constantly. Agents must track the current state while being aware of historical changes.

### 3. `numerical_precision`

**What it tests**: Exact recall of numbers, percentages, and metrics.

**Example**: "What is the Q1 revenue and how does it compare to the forecast?" (Expected: "$4.7M, 12% above forecast of $4.2M")

**Scoring dimensions**: `factual_accuracy`, `specificity`

**Why it matters**: Approximate recall is insufficient for financial data, metrics, and technical specifications.

### 4. `source_attribution`

**What it tests**: Correctly attributing claims to their original sources, especially when multiple sources discuss the same topic.

**Example**: "According to the Finance Department, what is Q3 revenue?" (Expected: "$5.2M, includes deferred revenue")

**Scoring dimensions**: `factual_accuracy`, `source_attribution`

**Why it matters**: In environments with multiple information sources, knowing who said what is as important as knowing the facts themselves.

### 5. `cross_reference`

**What it tests**: Connecting facts across different information blocks.

**Example**: "Sarah Chen leads Project Atlas. What is her educational background?" (requires connecting Block 2 project data with Block 1 people data)

**Scoring dimensions**: `factual_accuracy`, `specificity`

**Why it matters**: Real-world knowledge is interconnected. Agents must link related facts across different contexts.

### 6. `distractor_resistance`

**What it tests**: Answering questions accurately while ignoring irrelevant information.

**Example**: "What is the company's Q1 revenue? Note: do not confuse with the fact that honey never spoils." (Expected: "$4.7M")

**Scoring dimensions**: `factual_accuracy`, `confidence_calibration`

**Why it matters**: Memory systems that retrieve by similarity can be fooled by distractors. This tests precision of retrieval.

### 7. `meta_memory`

**What it tests**: The agent's awareness of what it knows and does not know.

**Example**: "How many distinct people have you been told about?" (Expected: count of people from Block 1)

**Scoring dimensions**: `factual_accuracy`, `confidence_calibration`

**Why it matters**: Agents that can reason about their own knowledge are more reliable and trustworthy.

### 8. `security_log_analysis`

**What it tests**: Pattern recognition in structured security event data.

**Example**: "What IP address was involved in the brute force SSH attack?" (Expected: "192.168.1.45")

**Scoring dimensions**: `factual_accuracy`, `specificity`

**Why it matters**: Security analysis requires precise recall of structured data including IPs, timestamps, and event sequences.

### 9. `incident_tracking`

**What it tests**: Following incident timelines and status evolution.

**Example**: "What is the current status of Incident INC-001, and when was the root cause identified?"

**Scoring dimensions**: `factual_accuracy`, `temporal_awareness`

**Why it matters**: Incident management requires tracking evolving states across multiple updates.

### 10. `infrastructure_knowledge`

**What it tests**: Recall of technical infrastructure details.

**Example**: "What is the CPU specification of server web-prod-01?" (Expected: specific CPU model and specs)

**Scoring dimensions**: `factual_accuracy`, `specificity`

**Why it matters**: Infrastructure queries require exact recall of configuration details.

### 11. `problem_solving`

**What it tests**: Recalling stored problem-solution pairs and applying them.

**Example**: "A user reports their application is running slowly after a deployment. What was the recommended diagnostic approach?"

**Scoring dimensions**: `factual_accuracy`, `specificity`

**Why it matters**: The ultimate purpose of stored knowledge is application to solve problems.

### 12. `multi_hop_reasoning`

**What it tests**: Chaining multiple facts to answer a question that requires 2+ retrieval steps.

**Example**: "The lead of Project Echo was recently changed. What team is the new lead originally from?" (requires: Echo's new lead -> that person's team)

**Scoring dimensions**: `factual_accuracy`, `specificity`

**Why it matters**: Complex questions rarely have answers in a single fact. Multi-hop reasoning tests the agent's ability to compose knowledge.

### 13-15. Additional Categories (generated from blocks)

The question generator also produces additional questions that combine categories -- for example, temporal + numerical ("What was the budget change for Project Atlas and when did it happen?") or cross-reference + security ("Which team member from the Security team had events logged in the security block?").

## The Grading System

### Hybrid Deterministic + LLM Grading

Each question is graded on multiple dimensions using a two-tier approach:

**Tier 1 -- Deterministic grading** (instant, free, reproducible):
- Checks `required_keywords` against the answer (case-insensitive)
- Awards bonus for `acceptable_paraphrases` found
- Scores 0.0 if any `incorrect_patterns` match
- Applies to `factual_accuracy` and `specificity` dimensions

**Tier 2 -- LLM semantic grading** (slower, costs API calls, nuanced):
- Used for `temporal_awareness`, `source_attribution`, `confidence_calibration`
- Also used when no deterministic rubric is available
- Prompt includes scoring guide and dimension descriptions
- Returns structured JSON with per-dimension scores and reasoning

### Multi-Vote Stability

To reduce LLM grading noise, each question is graded N times (configurable, default 3). For each dimension, the median score is taken as the final grade. The reasoning from the vote closest to the median is preserved.

```
Vote 1: factual_accuracy = 0.85
Vote 2: factual_accuracy = 0.90    -> Median: 0.85
Vote 3: factual_accuracy = 0.80
```

For deterministic dimensions, multi-vote has zero overhead since the score is identical every time.

### Dimension Weights

Questions can specify custom dimension weights via their rubric:

```python
GradingRubric(
    required_keywords=["Paris"],
    dimension_weights={"factual_accuracy": 1.5, "specificity": 0.5}
)
```

### Scoring Scale

| Score | Meaning                                       |
|-------|-----------------------------------------------|
| 1.0   | Perfect or semantically equivalent             |
| 0.8-0.9 | Correct main points, minor differences      |
| 0.5-0.7 | Partially correct, missing key details      |
| 0.2-0.4 | Some relevant content, significant gaps     |
| 0.0-0.1 | Incorrect or irrelevant                     |

## How to Interpret Results

### The EvalReport

A completed evaluation produces an `EvalReport` containing:

```python
@dataclass
class EvalReport:
    num_turns: int                          # Dialogue length
    num_questions: int                      # Questions asked
    total_facts_delivered: int              # Total facts in ground truth
    learning_time_s: float                  # Time to feed all turns
    questioning_time_s: float               # Time to ask + grade all questions
    grading_time_s: float                   # Time spent on grading only
    overall_score: float                    # Average of all question scores
    category_breakdown: list[CategoryBreakdown]  # Per-category averages
    results: list[EvalResult]              # Per-question details
    memory_stats: dict                     # Agent-reported memory statistics
```

### Reading the Category Breakdown

```
CATEGORY BREAKDOWN:
-----------------------------------------------------------------------
Category                     Avg      Min      Max   Count
-----------------------------------------------------------------------
cross_reference            85.00%   70.00%   95.00%      10
distractor_resistance      92.00%   80.00%  100.00%       8
meta_memory                78.00%   60.00%   90.00%       5
needle_in_haystack         88.00%   65.00%  100.00%      20
numerical_precision        82.00%   55.00%   95.00%      15
security_log_analysis      90.00%   80.00%  100.00%       8
source_attribution         75.00%   50.00%   90.00%      10
temporal_evolution          70.00%   40.00%   90.00%      15
```

**Focus on the weakest categories.** Categories below 70% indicate systematic weaknesses in the agent's memory system. The min score reveals worst-case performance.

### Dimension Averages

```
DIMENSION AVERAGES BY CATEGORY:
  needle_in_haystack: factual_accuracy: 90%, specificity: 86%
  temporal_evolution: factual_accuracy: 75%, temporal_awareness: 65%
  source_attribution: factual_accuracy: 80%, source_attribution: 70%
```

If `factual_accuracy` is high but `temporal_awareness` is low, the agent can recall facts but struggles with temporal ordering. If `source_attribution` is low, the agent retrieves facts but loses track of their origins.

### The Worst 5 Questions

The report highlights the 5 lowest-scoring questions. These are the best starting points for debugging:

```
WORST 5 QUESTIONS:
  [25.00%] What was the budget change for Project Atlas over time?
    Expected: $2.1M -> $2.5M (turn 45, additional cloud credits needed)
    Got: The budget is $2.5M.
```

This example shows the agent knows the current value but lost the change history -- a temporal awareness issue.

## Performance Characteristics

### Scaling

| Turns | Questions | Typical Facts | Generation Time | Learning Time* |
|-------|-----------|---------------|-----------------|---------------|
| 100   | 20        | ~80           | < 0.1s          | Depends on agent |
| 500   | 50        | ~400          | < 0.5s          | Depends on agent |
| 1000  | 100       | ~800          | < 1s            | Depends on agent |
| 5000  | 200       | ~4000         | < 5s            | Depends on agent |

*Learning time depends entirely on the agent implementation -- a simple list-based agent will be much faster than one using LLM-powered ingestion.

### Grading Costs

- **Deterministic grading**: Free, instant (no API calls)
- **LLM grading**: 1 API call per dimension per vote per question
- **Example**: 100 questions * 3 LLM dimensions * 3 votes = 900 API calls
- **Typical cost**: ~$0.50 for 100 questions with 3-vote grading (Sonnet model)

### Reproducibility

Same seed + same agent = same scores (within LLM grading variance). Multi-vote and multi-seed evaluation reduce this variance. For fully deterministic results, set all question rubrics and use `grading_mode="deterministic"`.

## Pre-built Datasets

### Skip the 4+ Hour Learning Phase

The 5000-turn learning phase takes 4+ hours and requires 10,000+ LLM API calls. Pre-built datasets let you skip this entirely and jump straight to evaluation.

### Available Datasets

| Name | Turns | Seed | Facts | Baseline Score | Size |
|------|-------|------|-------|---------------|------|
| `5000t-seed42-v1.0` | 5,000 | 42 | 762 | 90.47% | 1.3 MB |

### Download and Use

```bash
# List available datasets
amplihack-eval list-datasets

# Download a pre-built dataset
amplihack-eval download-dataset 5000t-seed42-v1.0

# Run evaluation using the pre-built DB (skip 4+ hour learning phase)
amplihack-eval run \
  --adapter learning-agent \
  --skip-learning \
  --load-db datasets/5000t-seed42-v1.0/memory_db \
  --turns 5000 \
  --questions 100
```

### Programmatic Usage

```python
from amplihack_eval.datasets import download_dataset, list_datasets

# List available datasets
datasets = list_datasets()
for ds in datasets:
    print(f"{ds['name']}: {'local' if ds.get('local') else 'remote'}")

# Download a dataset
path = download_dataset("5000t-seed42-v1.0")

# Use with evaluation
from amplihack_eval.adapters.learning_agent import LearningAgentAdapter
adapter = LearningAgentAdapter(storage_path=path / "memory_db")
```

### Dataset Structure

Each dataset contains:
- `metadata.json` -- Configuration and provenance (turns, seed, facts, baseline score, code version)
- `baseline_results.json` -- Full evaluation scores at time of creation
- `memory_db/` -- Kuzu graph database (the pre-built learning DB)

Datasets are distributed via [GitHub Releases](https://github.com/rysweet/amplihack-agent-eval/releases) to keep the repository lightweight.

## CLI Usage

```bash
# Basic evaluation (100 turns, 20 questions)
amplihack-eval run --turns 100 --questions 20 --adapter http --agent-url http://localhost:8000

# Large-scale stress test
amplihack-eval run --turns 5000 --questions 200 --grader-votes 5 --seed 42

# Skip learning with pre-built DB
amplihack-eval run --adapter learning-agent --skip-learning \
  --load-db datasets/5000t-seed42-v1.0/memory_db --turns 5000 --questions 100

# Multi-seed comparison
amplihack-eval compare --seeds 42,123,456,789 --turns 100 --questions 20

# With LearningAgent
amplihack-eval run --turns 100 --adapter learning-agent --model claude-sonnet-4-5-20250929

# Dataset management
amplihack-eval list-datasets
amplihack-eval download-dataset 5000t-seed42-v1.0
```

## Programmatic Usage

```python
from amplihack_eval import EvalRunner

# Create your agent (must implement AgentAdapter)
agent = MyAgent()

# Run evaluation
runner = EvalRunner(num_turns=1000, num_questions=100, seed=42, grader_votes=3)
report = runner.run(agent, grader_model="claude-sonnet-4-5-20250929")

# Inspect results
print(f"Overall: {report.overall_score:.2%}")
for cb in report.category_breakdown:
    print(f"  {cb.category}: {cb.avg_score:.2%} (n={cb.num_questions})")

# Save report
import json
with open("report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```
