# Self-Improvement Loop

## Overview

The self-improvement loop automates the process of identifying evaluation weaknesses and iteratively improving agent performance. It follows a disciplined 8-phase cycle with safety gates at every step, inspired by the principle: **measure first, change second**.

The loop consists of three cooperating modules:

- **`runner.py`** -- Orchestrates the 8 phases, manages iteration state, detects regression
- **`patch_proposer.py`** -- LLM-powered analysis of failures, generates hypotheses and unified diffs
- **`reviewer_voting.py`** -- Devil's advocate challenge + 3-reviewer voting (quality, regression, simplicity)

## The 8-Phase Improvement Cycle

```
EVAL -> ANALYZE -> PROPOSE -> CHALLENGE -> VOTE -> APPLY -> RE-EVAL -> DECIDE
  |                                                                      |
  +----------------------------------------------------------------------+
                           (iterate up to N times)
```

### Phase 1: EVAL

Run the full long-horizon evaluation to get per-category scores.

```python
evaluator = EvalRunner(num_turns=config.num_turns, num_questions=config.num_questions, seed=config.seed)
report = evaluator.run(agent, grader_model=config.grader_model)
```

A fresh agent is created from the `agent_factory` callable at each iteration to ensure clean state. The `EvalReport` provides `overall_score`, `category_breakdown` (per-category averages, min, max, dimension averages), and individual `EvalResult` entries for every question.

### Phase 2: ANALYZE

Identify the worst-performing category and diagnose the bottleneck.

The `_analyze_categories()` function groups all questions by category, identifies questions scoring below `failure_threshold`, and calls `_diagnose_bottleneck()` to map the failure to a specific system component.

**Bottleneck Diagnosis Mapping:**

The framework maps categories to system components using a two-level approach: category-specific mapping first, then fallback to worst-dimension mapping.

| Category | Bottleneck Component | Suggested Fix Direction |
|----------|---------------------|------------------------|
| `needle_in_haystack` | `retrieval:keyword_search` | Entity-centric indexing for people/projects |
| `meta_memory` | `retrieval:aggregation` | Route "how many" / "list all" to COUNT/DISTINCT queries |
| `source_attribution` | `retrieval:source_tracking` | Ensure source_label is included in retrieval results |
| `temporal_evolution` | `retrieval:temporal_ordering` | Ensure temporal_index metadata for chronological sorting |
| `cross_reference` | `retrieval:graph_traversal` | Expand hop depth to connect facts across blocks |
| `numerical_precision` | `synthesis:arithmetic` | Ensure calculate tool is used for all math operations |
| `distractor_resistance` | `retrieval:confidence_weighting` | Deprioritize distractor blocks with lower confidence |

**Dimension-based fallback** (when no category-specific mapping exists):

| Worst Dimension | Bottleneck Component | Suggested Fix |
|----------------|---------------------|---------------|
| `factual_accuracy` | `retrieval:coverage` | Increase retrieval coverage |
| `specificity` | `retrieval:precision` | Improve retrieval precision |
| `temporal_awareness` | `retrieval:temporal` | Add temporal metadata |
| `source_attribution` | `retrieval:provenance` | Improve source tracking |
| `confidence_calibration` | `synthesis:calibration` | Improve confidence expression |

Categories are sorted by average score (worst first). If no categories are below `failure_threshold`, the loop stops early with a "all categories above threshold" message.

### Phase 3: PROPOSE

Use LLM to generate a patch hypothesis targeting the worst category's bottleneck.

The `propose_patch()` function in `patch_proposer.py`:

1. **Reads the target file** based on the bottleneck component identifier. A `component_file_map` can be provided to map component prefixes (e.g., `"retrieval:"`) to file paths.

2. **Builds a structured prompt** containing:
   - The failing category name, score, bottleneck, and suggested fix direction
   - Up to 5 failed questions with their expected vs. actual answers, scores, and per-dimension breakdowns
   - Previously reverted patches (with revert reasons) -- to prevent repeating failed fixes
   - Previously rejected patches (with rejection reasons) -- to force different approaches
   - The target file's source code (up to 3000 characters)

3. **Calls the LLM** to generate a structured JSON response containing:
   - `hypothesis` -- why the category is failing
   - `description` -- what the patch does
   - `diff` -- unified diff format of the proposed change
   - `expected_impact` -- mapping of category name to expected score delta in percentage points
   - `risk_assessment` -- what could go wrong
   - `confidence` -- 0.0 to 1.0

4. **Returns a `PatchProposal`** dataclass. If no LLM callable is provided, a stub proposal with 10% confidence is returned.

**Patch rules enforced by the prompt:**

- The diff must be valid unified diff format
- Focus on the SMALLEST change that addresses the root cause
- Do NOT change test infrastructure, graders, or eval harness
- Prefer prompt/instruction changes over algorithmic changes
- Be honest about confidence -- lower is better than overconfident

### Phase 4: CHALLENGE

A devil's advocate reviews the proposal and raises concerns.

The `challenge_proposal()` function in `reviewer_voting.py` runs a two-step process:

**Step 1 -- Devil's Advocate Attack:**

An LLM call with the `DEVIL_ADVOCATE_PROMPT` argues aggressively against the proposal:
- What assumptions could be wrong?
- What could this break?
- Is there a simpler alternative?
- Is the hypothesis even correct?

The response includes a list of arguments against and a worst-case scenario.

**Step 2 -- Proposer Defense:**

A second LLM call receives the challenge arguments and the original proposal, then responds with:
- `defense` -- why the patch should still be applied
- `concerns_acknowledged` -- valid concerns the proposer acknowledges
- `concerns_refuted` -- concerns the proposer has addressed

**Adequacy Check:**

Concerns are considered adequately addressed if the proposer responds to at least 50% of the challenge arguments (acknowledged + refuted >= 50% of total challenges). If concerns are NOT adequately addressed, the proposal is logged as rejected and the iteration skips to the next cycle.

### Phase 5: VOTE

Three independent reviewers each vote on the proposal. Each reviewer has a distinct perspective:

**Quality Reviewer** -- evaluates:
- Engineering best practices (clean code, no side effects)
- Proper error handling
- Appropriate use of abstractions
- Consistency with existing code style
- Whether the change is testable

**Regression Reviewer** -- evaluates:
- Could this change break OTHER categories that currently pass?
- Does it modify shared code paths affecting unrelated functionality?
- Are there edge cases causing unexpected failures?
- Is the change scoped narrowly enough to avoid collateral damage?

**Simplicity Reviewer** -- evaluates:
- Is this the SIMPLEST possible fix?
- Could a smaller change achieve the same result?
- Does it add unnecessary complexity or abstraction?
- Could a prompt-only change work instead of a code change?

Each reviewer receives the formatted proposal (including the challenge phase results if available) and responds with a structured JSON vote: `accept`, `reject`, or `modify`, along with a rationale and specific concerns.

**Vote Tallying:**

```
>50% accept  -> "accepted"
>50% reject  -> "rejected"
otherwise    -> "modified" (mixed signals)
```

With 3 reviewers, this means at least 2 must accept for the proposal to pass.

If the proposal is rejected, it is logged in `PatchHistory.rejected_patches` and the iteration continues to the next cycle.

### Phase 6: APPLY

If the majority votes accept, the patch is applied. The proposal is logged in `PatchHistory.applied_patches` with its target file, description, hypothesis, and confidence.

### Phase 7: RE-EVAL

Run the evaluation again with a fresh agent to measure the patch's impact. The post-evaluation scores are compared against the baseline scores from Phase 1.

### Phase 8: DECIDE

The `detect_regression()` function checks whether any individual category regressed beyond the configured threshold:

```python
def detect_regression(
    baseline_scores: dict[str, float],
    post_scores: dict[str, float],
    threshold: float = 5.0,
) -> tuple[bool, str, float]:
```

**Regression Detection Logic:**

1. Compute the delta (in percentage points) for each category: `baseline - post`
2. Find the maximum regression across all categories
3. Find the maximum gain across all categories
4. Regression is detected if: `max_regression > threshold AND max_gain < threshold`

This means a large gain in one category can compensate for a small regression in another, but a large regression without corresponding gains triggers a revert.

**If regression detected:** The patch is automatically reverted and logged in `PatchHistory.reverted_patches` with the revert reason.

**If no regression:** The patch is kept and the loop continues.

## Configuration

```python
from amplihack_eval.self_improve.runner import SelfImproveConfig

config = SelfImproveConfig(
    num_turns=100,              # Dialogue turns per evaluation
    num_questions=20,           # Questions per evaluation
    seed=42,                    # Random seed for reproducibility
    max_iterations=3,           # Maximum improvement iterations
    failure_threshold=0.7,      # Scores below this are "failures"
    regression_threshold=5.0,   # Max regression (pp) before auto-revert
    output_dir="/tmp/self-improve",  # Where to write logs and reports
    grader_model="",            # LLM model for grading (default if empty)
)
```

**Parameter Tuning Guidelines:**

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| `failure_threshold` | 0.8 | 0.7 | 0.5 |
| `regression_threshold` | 2.0 | 5.0 | 10.0 |
| `max_iterations` | 3 | 5 | 10 |
| `num_turns` | 100 | 500 | 1000 |
| `num_questions` | 20 | 50 | 100 |

- **Conservative**: Strict quality gates, fewer iterations, catches only severe failures
- **Moderate**: Balanced -- good default for most use cases
- **Aggressive**: Allows more latitude, more iterations, may find more improvements but risks more churn

## Key Data Structures

### PatchProposal

```python
@dataclass
class PatchProposal:
    target_file: str                       # Path to the file to modify
    hypothesis: str                        # Why the category is failing
    description: str                       # What the patch does
    diff: str                              # Unified diff format
    expected_impact: dict[str, float] = {} # category -> expected score delta (pp)
    risk_assessment: str = ""              # What could go wrong
    confidence: float = 0.0               # 0.0 to 1.0
```

### PatchHistory

```python
@dataclass
class PatchHistory:
    applied_patches: list[dict] = []     # Successfully applied patches
    reverted_patches: list[dict] = []    # Patches that caused regression
    rejected_patches: list[dict] = []    # Patches rejected by reviewers
```

The history is passed to the patch proposer at each iteration. Previously reverted and rejected patches are included in the LLM prompt with their failure reasons, preventing the system from proposing the same fix twice.

### CategoryAnalysis

```python
@dataclass
class CategoryAnalysis:
    category: str                          # Category name
    avg_score: float                       # Average score
    num_questions: int                     # Number of questions
    failed_questions: list[dict] = []      # Details of failed questions
    bottleneck: str = ""                   # Identified system component
    suggested_fix: str = ""                # Suggested improvement direction
```

### ReviewVote

```python
@dataclass
class ReviewVote:
    reviewer_id: str                       # "quality", "regression", or "simplicity"
    vote: str                              # "accept", "reject", or "modify"
    rationale: str                         # Why this vote was cast
    concerns: list[str] = []              # Specific concerns
    suggested_modifications: str | None = None
```

### ChallengeResponse

```python
@dataclass
class ChallengeResponse:
    challenge_arguments: list[str]         # Arguments against the patch
    proposer_response: str                 # The proposer's defense
    concerns_addressed: bool               # Were >= 50% of concerns addressed?
    remaining_concerns: list[str] = []     # Unaddressed concerns
```

### ReviewResult

```python
@dataclass
class ReviewResult:
    proposal: PatchProposal                # The proposal being reviewed
    challenge: ChallengeResponse | None    # Challenge phase results
    votes: list[ReviewVote]                # Individual reviewer votes
    decision: str                          # "accepted", "rejected", or "modified"
    consensus_rationale: str               # Summary rationale
```

### RunnerResult

```python
@dataclass
class RunnerResult:
    config: dict[str, Any]                 # Configuration used
    iterations: list[IterationResult]      # Per-iteration results
    score_progression: list[float]         # Overall score at each iteration
    category_progression: dict[str, list[float]]  # Per-category scores over time
    total_duration_seconds: float
```

### IterationResult

```python
@dataclass
class IterationResult:
    iteration: int                         # Iteration number (1-based)
    report: dict[str, Any]                 # Full EvalReport as dict
    category_analyses: list[dict]          # Per-category failure analysis
    improvements_applied: list[str]        # Descriptions of applied patches
    patch_proposal: dict | None = None     # Proposal details
    review_result: dict | None = None      # Review details
    post_scores: dict[str, float] | None = None  # Post-eval scores
    reverted: bool = False                 # Was this patch reverted?
    revert_reason: str = ""                # Why it was reverted
    duration_seconds: float = 0.0          # Iteration wall clock time
```

## Safety Constraints

The self-improvement loop enforces strict safety rules:

1. **Never modifies the grader**: The grading system is the source of truth. The LLM prompt explicitly forbids changes to test infrastructure, graders, or eval harness.

2. **Never modifies test data**: Test levels and questions are fixed. Data generation is deterministic and untouchable.

3. **Never weakens safety constraints**: The challenge and voting phases cannot be circumvented. Even if an LLM proposes removing safety gates, reviewers would reject it.

4. **Auto-revert on regression**: If the overall score or any category drops by more than `regression_threshold` percentage points, the patch is automatically reverted. This protects existing quality from well-intentioned but harmful changes.

5. **Full history prevents repetition**: All attempted patches (applied, reverted, rejected) are tracked in `PatchHistory` and included in future proposal prompts, preventing the system from proposing the same failed fix repeatedly.

6. **Majority vote required**: No patch is applied without at least 2/3 reviewers voting to accept.

7. **Devil's advocate gate**: Proposals that fail the challenge phase (less than 50% of concerns addressed) are rejected before reaching the voting stage.

## CLI Usage

```bash
# Run 5 iterations of self-improvement
amplihack-eval self-improve --iterations 5 --turns 100 --questions 20

# Large-scale improvement with custom output
amplihack-eval self-improve --iterations 10 --turns 500 --questions 50 --output-dir ./logs

# Conservative settings (strict regression threshold)
amplihack-eval self-improve --iterations 3 --regression-threshold 2.0
```

## Programmatic Usage

```python
from amplihack_eval.self_improve.runner import SelfImproveConfig, run_self_improve
from amplihack_eval.adapters.learning_agent import LearningAgentAdapter

config = SelfImproveConfig(
    num_turns=100,
    num_questions=20,
    max_iterations=5,
    failure_threshold=0.7,
    regression_threshold=5.0,
    output_dir="/tmp/self-improve",
)

# Agent factory creates a fresh agent for each iteration
def make_agent():
    return LearningAgentAdapter(model="claude-sonnet-4-5-20250929")

# Optional: provide an LLM callable for patch proposal and review
import anthropic
client = anthropic.Anthropic()

def llm_call(prompt: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

# Run the loop
result = run_self_improve(
    config=config,
    agent_factory=make_agent,
    llm_call=llm_call,
    project_root=Path("/path/to/agent/code"),
)

# Inspect results
print(f"Iterations: {len(result.iterations)}")
print(f"Score progression: {' -> '.join(f'{s:.2%}' for s in result.score_progression)}")
print(f"Duration: {result.total_duration_seconds:.1f}s")

for it in result.iterations:
    status = "REVERTED" if it.reverted else "KEPT"
    print(f"  Iteration {it.iteration}: {status}")
    if it.patch_proposal:
        print(f"    Patch: {it.patch_proposal.get('description', '')[:80]}")
    if it.review_result:
        print(f"    Review: {it.review_result.get('decision', '')}")
```

## Output Structure

Each run produces files in the output directory:

```
/tmp/self-improve/
    self_improve_summary.json       # High-level summary
    iteration_1/
        report.json                 # Full EvalReport from Phase 1
    iteration_2/
        report.json
    ...
```

### Summary JSON

```json
{
    "config": {
        "num_turns": 100,
        "num_questions": 20,
        "max_iterations": 5,
        "failure_threshold": 0.7,
        "regression_threshold": 5.0
    },
    "score_progression": [0.72, 0.78, 0.82, 0.85],
    "category_progression": {
        "needle_in_haystack": [0.88, 0.90, 0.92, 0.92],
        "temporal_evolution": [0.55, 0.68, 0.72, 0.78],
        "numerical_precision": [0.62, 0.65, 0.70, 0.75]
    },
    "total_duration_seconds": 342.5,
    "iterations_run": 4,
    "patches_applied": 3,
    "patches_reverted": 1,
    "patches_rejected": 0
}
```

## How It Works End-to-End

A typical 5-iteration run:

```
Iteration 1:
  EVAL:      Overall 72%, temporal_evolution at 55% (worst)
  ANALYZE:   Bottleneck: retrieval:temporal_ordering
  PROPOSE:   "Add temporal_index metadata to all time-series facts"
  CHALLENGE: 3 arguments raised, 2 addressed, 1 remaining -> PASS
  VOTE:      quality=accept, regression=accept, simplicity=modify -> ACCEPTED
  APPLY:     Patch applied
  RE-EVAL:   temporal_evolution 55% -> 68%, overall 72% -> 78%
  DECIDE:    No regression -> KEEP

Iteration 2:
  EVAL:      Overall 78%, numerical_precision at 62% (worst)
  ANALYZE:   Bottleneck: synthesis:arithmetic
  PROPOSE:   "Route math questions to calculate tool"
  CHALLENGE: 2 arguments raised, 2 addressed -> PASS
  VOTE:      quality=accept, regression=reject, simplicity=accept -> ACCEPTED
  APPLY:     Patch applied
  RE-EVAL:   numerical_precision 62% -> 70%, but source_attribution 80% -> 72%
  DECIDE:    source_attribution regressed 8pp > 5pp threshold -> REVERT

Iteration 3:
  EVAL:      Overall 78% (reverted to pre-iteration-2 state)
  ANALYZE:   numerical_precision at 62% (same as before, different approach needed)
  PROPOSE:   "Add arithmetic verification step to synthesis prompt"
             (History shows previous approach was reverted due to regression)
  CHALLENGE: 2 arguments raised, 2 addressed -> PASS
  VOTE:      quality=accept, regression=accept, simplicity=accept -> ACCEPTED
  APPLY:     Patch applied
  RE-EVAL:   numerical_precision 62% -> 73%, no regressions
  DECIDE:    No regression -> KEEP
```

## Real-World Results

From the amplihack development history (5-loop improvement cycle):

- **Starting score**: 83.2%
- **Final score**: 96.6% (+13.4 percentage points)
- **Biggest single improvement**: Source-specific fact filtering, +53.3% on L2 (multi-source synthesis)
- **Key insight**: Retrieval threshold of 50 was too low; increasing to 150 prevented cascading failures as the knowledge base grew

## Integration with Multi-Seed Evaluation

For more robust improvement decisions, combine self-improvement with multi-seed evaluation:

```python
from amplihack_eval.core.multi_seed import run_multi_seed_eval

# After each improvement iteration, validate across multiple seeds
seeds = [42, 123, 456, 789]
multi_result = run_multi_seed_eval(agent, seeds=seeds, num_turns=100, num_questions=20)

# Check if improvement is consistent across seeds (not just lucky on seed=42)
if multi_result.inter_seed_variance["temporal_evolution"] > 0.10:
    print("WARNING: High variance -- improvement may be seed-dependent")
```

This guards against improvements that only work with specific random seeds.
