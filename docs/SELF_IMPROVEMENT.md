# Self-Improvement Loop

## Overview

The self-improvement loop automates the process of identifying evaluation weaknesses and iteratively improving agent performance. It follows a disciplined cycle with safety gates at every step.

## The Improvement Cycle

```
EVAL -> ANALYZE -> PROPOSE -> CHALLENGE -> VOTE -> APPLY -> RE-EVAL -> DECIDE
  |                                                                      |
  +----------------------------------------------------------------------+
                           (iterate N times)
```

### Step-by-Step

1. **EVAL**: Run the full long-horizon evaluation to get per-category scores
2. **ANALYZE**: Identify the worst-performing category and diagnose the bottleneck
3. **PROPOSE**: Use LLM to generate a patch hypothesis targeting the bottleneck
4. **CHALLENGE**: A devil's advocate reviews the proposal and raises concerns
5. **VOTE**: 3 independent reviewers vote to accept or reject the proposal
6. **APPLY**: If majority votes accept, apply the patch (git commit)
7. **RE-EVAL**: Run evaluation again to measure the patch's impact
8. **DECIDE**: If regression exceeds threshold, auto-revert; if improvement, keep

## Configuration

```python
from amplihack_eval.self_improve.runner import SelfImproveConfig, SelfImproveRunner

config = SelfImproveConfig(
    num_turns=100,              # Dialogue turns per eval
    num_questions=20,           # Questions per eval
    seed=42,                    # Random seed for reproducibility
    max_iterations=3,           # Max improvement iterations
    failure_threshold=0.7,      # Scores below this are "failures"
    regression_threshold=5.0,   # Max regression (pp) before auto-revert
    output_dir="/tmp/self-improve",
    grader_model="",            # LLM model for grading (uses default if empty)
)
```

## Key Components

### CategoryAnalysis

After each eval run, failures are grouped by question category:

```python
@dataclass
class CategoryAnalysis:
    category: str                        # e.g., "needle_in_haystack"
    avg_score: float                     # Average score for this category
    num_questions: int                   # Number of questions
    failed_questions: list[dict]         # Details of failed questions
    bottleneck: str                      # Identified system component
    suggested_fix: str                   # Suggested improvement
```

### PatchProposal (patch_proposer.py)

The LLM-powered patch generator receives:
- The worst category's failure analysis
- The full patch history (what was tried before)
- System component descriptions

It produces a structured proposal with:
- Target component
- Proposed changes
- Expected impact
- Risk assessment

### Reviewer Voting (reviewer_voting.py)

Three-stage review process:

1. **Challenge**: A devil's advocate reviews the proposal and raises concerns about correctness, side effects, and safety
2. **Vote**: 3 independent reviewers each vote accept/reject with reasoning
3. **Decision**: Majority rules; at least 2 of 3 must accept

## Safety Constraints

The self-improvement loop enforces strict safety rules:

- **Never modifies the grader**: The grading system is the source of truth
- **Never modifies test data**: Test levels and questions are fixed
- **Never modifies safety constraints**: Safety gates cannot be weakened
- **Auto-revert on regression**: If overall score drops by more than `regression_threshold` percentage points, the patch is automatically reverted
- **Full history**: All attempted patches (successful and failed) are logged to prevent repeating failed fixes

## CLI Usage

```bash
# Run 5 iterations of self-improvement
amplihack-eval self-improve --iterations 5 --turns 100 --questions 20

# With custom output directory
amplihack-eval self-improve --iterations 3 --output-dir ./improvement-logs
```

## Programmatic Usage

```python
from amplihack_eval.self_improve.runner import SelfImproveConfig, SelfImproveRunner

config = SelfImproveConfig(max_iterations=5, num_turns=100)
runner = SelfImproveRunner(config)

# You need to provide an agent factory since the agent may be modified between iterations
result = runner.run(agent_factory=lambda: MyAgent())

print(f"Score progression: {result.score_progression}")
print(f"Total iterations: {len(result.iterations)}")
for it in result.iterations:
    print(f"  Iteration {it.iteration}: reverted={it.reverted}")
```

## Output Structure

Each run produces a JSON log in the output directory:

```json
{
    "config": { ... },
    "iterations": [
        {
            "iteration": 0,
            "report": { "overall_score": 0.72, ... },
            "category_analyses": [ ... ],
            "patch_proposal": { ... },
            "review_result": { "accepted": true, "votes": [...] },
            "post_scores": { "overall": 0.78 },
            "reverted": false
        }
    ],
    "score_progression": [0.72, 0.78, 0.82],
    "total_duration_seconds": 342.5
}
```
