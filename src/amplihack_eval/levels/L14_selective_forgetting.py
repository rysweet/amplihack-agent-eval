"""L14: Selective Forgetting scoring.

Scores an agent's ability to return current values and not be confused by
superseded (old) values.

Metrics:
- current_value_accuracy: Does the agent return the current value?
- stale_data_penalty: Does the agent present old values as if they were current?
- update_awareness: Does the agent acknowledge that the value was updated?

Philosophy: An agent that remembers everything indiscriminately is dangerous.
The ability to forget (or deprioritize) superseded information is as important
as the ability to recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ForgettingResult:
    """Score for a single selective forgetting scenario."""

    scenario_id: str
    current_value_accuracy: float  # 0.0-1.0: did agent return current value?
    stale_data_penalty: float  # 0.0-1.0: 1.0 = no stale data mentioned as current
    update_awareness: float  # 0.0-1.0: did agent note the update?
    overall: float  # Weighted average
    details: str = ""


def _normalize_value(value: str) -> str:
    """Normalize a value for comparison by lowering case and stripping whitespace."""
    return value.strip().lower()


def _value_present(value: str, answer: str) -> bool:
    """Check if a value is present in the answer (case-insensitive substring match)."""
    return _normalize_value(value) in _normalize_value(answer)


def score_current_value_accuracy(
    expected_current_value: str,
    actual_answer: str,
) -> float:
    """Score whether the agent returned the correct current value.

    Uses substring matching with normalization. A score of 1.0 means the
    current value appears in the answer.

    Args:
        expected_current_value: The correct current value
        actual_answer: Agent's response

    Returns:
        Score from 0.0 to 1.0
    """
    if not actual_answer or not actual_answer.strip():
        return 0.0

    if _value_present(expected_current_value, actual_answer):
        return 1.0

    # Partial credit: check if key parts of the value are present
    # Split on common separators and check individual tokens
    tokens = expected_current_value.replace(",", " ").replace("(", " ").replace(")", " ").split()
    significant_tokens = [t for t in tokens if len(t) > 2]  # Skip short words

    if not significant_tokens:
        return 0.0

    matched = sum(1 for t in significant_tokens if _value_present(t, actual_answer))
    return round(matched / len(significant_tokens), 4)


def score_stale_data_penalty(
    superseded_values: list[str],
    actual_answer: str,
) -> float:
    """Score whether old values are incorrectly presented as current.

    Returns 1.0 if no superseded values appear or if they are clearly marked
    as historical. Returns lower scores if stale data appears to be presented
    as current.

    Note: Mentioning old values in context (e.g., "changed from X to Y") is
    acceptable and should not be penalized. Only presenting old values AS the
    current answer is penalized.

    Args:
        superseded_values: List of old values that should not appear as current
        actual_answer: Agent's response

    Returns:
        Score from 0.0 to 1.0 (higher = better, 1.0 = no stale data)
    """
    if not actual_answer or not actual_answer.strip():
        return 1.0  # No answer = no stale data presented

    if not superseded_values:
        return 1.0

    answer_lower = _normalize_value(actual_answer)

    # Historical context indicators - if the answer uses these phrases around
    # old values, it's acceptable (agent is describing the change)
    historical_indicators = [
        "previously", "was previously", "changed from", "moved from",
        "used to be", "formerly", "before", "old value", "updated from",
        "earlier", "prior", "original", "superseded", "replaced",
    ]

    stale_count = 0
    for old_val in superseded_values:
        old_val_lower = _normalize_value(old_val)
        if old_val_lower not in answer_lower:
            continue  # Old value not mentioned at all - good

        # Check if old value appears in historical context
        has_historical_context = any(
            indicator in answer_lower for indicator in historical_indicators
        )

        if not has_historical_context:
            stale_count += 1

    if stale_count == 0:
        return 1.0

    # Penalty proportional to how many stale values are presented without context
    penalty = stale_count / len(superseded_values)
    return round(max(0.0, 1.0 - penalty), 4)


def score_update_awareness(
    superseded_values: list[str],
    expected_current_value: str,
    actual_answer: str,
) -> float:
    """Score whether the agent demonstrates awareness that the value was updated.

    An agent that says "The deadline was moved from June 15 to September 1"
    demonstrates update awareness. An agent that just says "September 1"
    is correct but shows less awareness.

    Args:
        superseded_values: List of old values
        expected_current_value: The current value
        actual_answer: Agent's response

    Returns:
        Score from 0.0 to 1.0
    """
    if not actual_answer or not actual_answer.strip():
        return 0.0

    answer_lower = _normalize_value(actual_answer)

    # Check if current value is present (prerequisite for update awareness)
    has_current = _value_present(expected_current_value, actual_answer)
    if not has_current:
        return 0.0

    # Check for update language
    update_indicators = [
        "changed", "updated", "moved", "increased", "decreased",
        "promoted", "transferred", "upgraded", "migrated", "switched",
        "raised", "lowered", "adjusted", "extended", "pushed",
        "was previously", "previously was", "changed from", "moved from",
        "now", "currently", "as of", "effective",
    ]

    has_update_language = any(indicator in answer_lower for indicator in update_indicators)

    # Check if any old values are mentioned in historical context
    mentions_old = any(
        _value_present(old_val, actual_answer) for old_val in superseded_values
    )

    if has_update_language and mentions_old:
        return 1.0  # Full awareness: mentions both old and new with update language
    elif has_update_language:
        return 0.7  # Partial: uses update language but doesn't reference old values
    elif mentions_old:
        return 0.5  # Mentions old values but no clear update narrative
    else:
        return 0.3  # Just returns current value with no update context


def score_forgetting_scenario(
    expected_current_value: str,
    superseded_values: list[str],
    actual_answer: str,
    scenario_id: str = "",
    weight_accuracy: float = 0.5,
    weight_stale: float = 0.3,
    weight_awareness: float = 0.2,
) -> ForgettingResult:
    """Compute composite forgetting score for a scenario.

    Args:
        expected_current_value: What the agent should answer
        superseded_values: Old values that should not be presented as current
        actual_answer: Agent's response
        scenario_id: Identifier for the scenario
        weight_accuracy: Weight for current value accuracy
        weight_stale: Weight for stale data penalty
        weight_awareness: Weight for update awareness

    Returns:
        ForgettingResult with all sub-scores and overall
    """
    accuracy = score_current_value_accuracy(expected_current_value, actual_answer)
    stale = score_stale_data_penalty(superseded_values, actual_answer)
    awareness = score_update_awareness(superseded_values, expected_current_value, actual_answer)

    overall = (
        accuracy * weight_accuracy
        + stale * weight_stale
        + awareness * weight_awareness
    )

    details = (
        f"Accuracy: {accuracy:.2f} (expected '{expected_current_value}'), "
        f"Stale penalty: {stale:.2f} ({len(superseded_values)} old values), "
        f"Update awareness: {awareness:.2f}"
    )

    return ForgettingResult(
        scenario_id=scenario_id,
        current_value_accuracy=round(accuracy, 4),
        stale_data_penalty=round(stale, 4),
        update_awareness=round(awareness, 4),
        overall=round(overall, 4),
        details=details,
    )


def aggregate_forgetting_scores(results: list[ForgettingResult]) -> dict[str, float]:
    """Compute aggregate statistics across multiple forgetting scenario results.

    Returns:
        Dict with averages for each metric
    """
    if not results:
        return {
            "avg_current_value_accuracy": 0.0,
            "avg_stale_data_penalty": 0.0,
            "avg_update_awareness": 0.0,
            "avg_overall": 0.0,
        }

    n = len(results)
    return {
        "avg_current_value_accuracy": round(sum(r.current_value_accuracy for r in results) / n, 4),
        "avg_stale_data_penalty": round(sum(r.stale_data_penalty for r in results) / n, 4),
        "avg_update_awareness": round(sum(r.update_awareness for r in results) / n, 4),
        "avg_overall": round(sum(r.overall for r in results) / n, 4),
    }


__all__ = [
    "ForgettingResult",
    "score_current_value_accuracy",
    "score_stale_data_penalty",
    "score_update_awareness",
    "score_forgetting_scenario",
    "aggregate_forgetting_scores",
]
