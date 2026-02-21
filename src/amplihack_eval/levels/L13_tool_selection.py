"""L13: Tool Use Selection scoring.

Scores an agent's ability to select the right tools and chain them correctly.

Metrics:
- tool_selection_accuracy: Did the agent pick the right tools?
- tool_efficiency: Were there unnecessary tool calls?
- tool_chain_correctness: Was the ordering/chaining correct?

Philosophy: Evaluate the trajectory, not just the final answer. An agent that
uses the right tools in the right order is demonstrating genuine planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ToolTrajectory:
    """Record of an agent's actual tool calls for a scenario."""

    scenario_id: str
    tool_calls: list[str]  # Ordered list of tool names the agent actually called
    total_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolSelectionScore:
    """Composite score for tool selection evaluation."""

    scenario_id: str
    selection_accuracy: float  # 0.0-1.0: right tools selected
    efficiency: float  # 0.0-1.0: no wasted calls (1.0 = perfectly efficient)
    chain_correctness: float  # 0.0-1.0: correct ordering
    overall: float  # Weighted average of all three
    details: str = ""  # Human-readable breakdown


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _longest_common_subsequence_length(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    """Length of the longest common subsequence between two sequences."""
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return 0

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def score_tool_selection(
    expected_sequence: list[str],
    actual_trajectory: ToolTrajectory,
) -> float:
    """Score whether the agent selected the right tools (ignoring order).

    Uses Jaccard similarity between the set of expected tools and the set of
    tools the agent actually used.

    Args:
        expected_sequence: Expected tool names (from scenario definition)
        actual_trajectory: Agent's actual tool call trajectory

    Returns:
        Score from 0.0 to 1.0
    """
    expected_set = set(expected_sequence)
    actual_set = set(actual_trajectory.tool_calls)
    return _jaccard_similarity(expected_set, actual_set)


def score_tool_efficiency(
    expected_sequence: list[str],
    actual_trajectory: ToolTrajectory,
) -> float:
    """Score tool call efficiency. Penalizes unnecessary calls.

    Efficiency = min(expected_count, actual_count) / max(expected_count, actual_count)
    A perfectly efficient agent uses exactly the expected number of calls.
    Extra calls reduce the score; fewer calls also reduce it (means skipped steps).

    Args:
        expected_sequence: Expected tool names (from scenario definition)
        actual_trajectory: Agent's actual tool call trajectory

    Returns:
        Score from 0.0 to 1.0
    """
    expected_count = len(expected_sequence)
    actual_count = len(actual_trajectory.tool_calls)

    if expected_count == 0 and actual_count == 0:
        return 1.0
    if expected_count == 0 or actual_count == 0:
        return 0.0

    return min(expected_count, actual_count) / max(expected_count, actual_count)


def score_tool_chain(
    expected_sequence: list[str],
    actual_trajectory: ToolTrajectory,
) -> float:
    """Score the correctness of the tool call ordering/chain.

    Uses Longest Common Subsequence (LCS) normalized by the expected sequence
    length. This measures how well the agent preserved the required ordering,
    allowing for extra calls in between.

    Args:
        expected_sequence: Expected ordered tool sequence
        actual_trajectory: Agent's actual tool call trajectory

    Returns:
        Score from 0.0 to 1.0
    """
    if not expected_sequence and not actual_trajectory.tool_calls:
        return 1.0
    if not expected_sequence or not actual_trajectory.tool_calls:
        return 0.0

    lcs_len = _longest_common_subsequence_length(expected_sequence, actual_trajectory.tool_calls)
    return lcs_len / len(expected_sequence)


def score_scenario(
    expected_sequence: list[str],
    actual_trajectory: ToolTrajectory,
    weight_selection: float = 0.4,
    weight_efficiency: float = 0.2,
    weight_chain: float = 0.4,
) -> ToolSelectionScore:
    """Compute composite tool selection score for a scenario.

    Args:
        expected_sequence: Expected ordered tool sequence
        actual_trajectory: Agent's actual tool call trajectory
        weight_selection: Weight for tool selection accuracy (default 0.4)
        weight_efficiency: Weight for efficiency (default 0.2)
        weight_chain: Weight for chain correctness (default 0.4)

    Returns:
        ToolSelectionScore with all sub-scores and overall
    """
    selection = score_tool_selection(expected_sequence, actual_trajectory)
    efficiency = score_tool_efficiency(expected_sequence, actual_trajectory)
    chain = score_tool_chain(expected_sequence, actual_trajectory)

    overall = (
        selection * weight_selection
        + efficiency * weight_efficiency
        + chain * weight_chain
    )

    details_parts = [
        f"Selection: {selection:.2f} (expected {set(expected_sequence)}, got {set(actual_trajectory.tool_calls)})",
        f"Efficiency: {efficiency:.2f} (expected {len(expected_sequence)} calls, got {len(actual_trajectory.tool_calls)})",
        f"Chain: {chain:.2f} (LCS-based ordering score)",
    ]

    return ToolSelectionScore(
        scenario_id=actual_trajectory.scenario_id,
        selection_accuracy=round(selection, 4),
        efficiency=round(efficiency, 4),
        chain_correctness=round(chain, 4),
        overall=round(overall, 4),
        details="; ".join(details_parts),
    )


def score_batch(
    scenarios: list[tuple[list[str], ToolTrajectory]],
) -> list[ToolSelectionScore]:
    """Score a batch of scenarios.

    Args:
        scenarios: List of (expected_sequence, actual_trajectory) tuples

    Returns:
        List of ToolSelectionScore, one per scenario
    """
    return [score_scenario(expected, trajectory) for expected, trajectory in scenarios]


def aggregate_scores(scores: list[ToolSelectionScore]) -> dict[str, float]:
    """Compute aggregate statistics across multiple scenario scores.

    Returns:
        Dict with avg_selection, avg_efficiency, avg_chain, avg_overall
    """
    if not scores:
        return {
            "avg_selection": 0.0,
            "avg_efficiency": 0.0,
            "avg_chain": 0.0,
            "avg_overall": 0.0,
        }

    n = len(scores)
    return {
        "avg_selection": round(sum(s.selection_accuracy for s in scores) / n, 4),
        "avg_efficiency": round(sum(s.efficiency for s in scores) / n, 4),
        "avg_chain": round(sum(s.chain_correctness for s in scores) / n, 4),
        "avg_overall": round(sum(s.overall for s in scores) / n, 4),
    }


__all__ = [
    "ToolTrajectory",
    "ToolSelectionScore",
    "score_tool_selection",
    "score_tool_efficiency",
    "score_tool_chain",
    "score_scenario",
    "score_batch",
    "aggregate_scores",
]
