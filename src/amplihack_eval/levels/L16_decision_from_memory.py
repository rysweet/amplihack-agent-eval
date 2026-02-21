"""L16: Decision From Memory scoring.

Scores an agent's ability to recall facts, analyze them, and make correct
decisions. This is the highest cognitive level: recall -> reason -> decide.

Metrics:
- decision_quality: Is the decision correct given the available facts?
- reasoning_quality: Does the explanation reference the correct facts?
- fact_usage: Were the right facts used to support the decision?

Philosophy: The ultimate test of memory is not just retrieval but APPLICATION.
An agent that can recall facts and use them to make good decisions demonstrates
genuine understanding.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecisionScore:
    """Score for a single decision-from-memory scenario."""

    scenario_id: str
    decision_quality: float  # 0.0-1.0: correct decision given the facts
    reasoning_quality: float  # 0.0-1.0: explanation references correct facts
    fact_usage: float  # 0.0-1.0: used the right facts to support decision
    overall: float  # Weighted average
    details: str = ""


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    return text.strip().lower()


def _extract_key_phrases(text: str) -> set[str]:
    """Extract significant phrases (3+ chars, not stop words) from text."""
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "his", "how",
        "its", "may", "new", "now", "old", "see", "way", "who", "did",
        "get", "let", "say", "she", "too", "use", "with", "from",
        "that", "this", "will", "than", "them", "then", "they", "been",
        "have", "each", "make", "like", "into", "over", "such", "should",
        "would", "could", "about", "which", "their", "there", "these",
        "those", "being", "other",
    }
    words = set(_normalize(text).split())
    return {w for w in words if len(w) > 2 and w not in stop_words}


def score_decision_quality(
    expected_decision: str,
    alternative_decisions: list[str],
    actual_answer: str,
) -> float:
    """Score whether the agent made the correct decision.

    Checks the agent's answer against the expected decision and acceptable
    alternatives. Uses key phrase overlap to determine alignment.

    Args:
        expected_decision: The primary correct decision
        alternative_decisions: Other acceptable decisions
        actual_answer: Agent's response

    Returns:
        Score from 0.0 to 1.0
    """
    if not actual_answer or not actual_answer.strip():
        return 0.0

    answer_lower = _normalize(actual_answer)
    answer_phrases = _extract_key_phrases(actual_answer)

    # Check alignment with expected decision
    expected_phrases = _extract_key_phrases(expected_decision)
    if expected_phrases:
        expected_overlap = len(answer_phrases & expected_phrases) / len(expected_phrases)
    else:
        expected_overlap = 0.0

    # Check alignment with alternatives
    best_alt_overlap = 0.0
    for alt in alternative_decisions:
        alt_phrases = _extract_key_phrases(alt)
        if alt_phrases:
            overlap = len(answer_phrases & alt_phrases) / len(alt_phrases)
            best_alt_overlap = max(best_alt_overlap, overlap)

    # Best match across expected and alternatives
    best_match = max(expected_overlap, best_alt_overlap)

    # Scale to meaningful range
    if best_match >= 0.5:
        return min(1.0, 0.6 + best_match * 0.4)  # 0.8-1.0 range
    elif best_match >= 0.3:
        return 0.4 + best_match  # 0.7 range
    elif best_match >= 0.1:
        return 0.2 + best_match  # 0.3-0.5 range
    else:
        return best_match  # 0.0-0.1 range


def score_reasoning_quality(
    reasoning_chain: str,
    actual_answer: str,
) -> float:
    """Score the quality of the agent's reasoning/explanation.

    Checks if the agent's reasoning follows a logical chain and references
    the expected reasoning steps.

    Args:
        reasoning_chain: Expected chain of reasoning
        actual_answer: Agent's response

    Returns:
        Score from 0.0 to 1.0
    """
    if not actual_answer or not actual_answer.strip():
        return 0.0

    answer_lower = _normalize(actual_answer)

    # Extract reasoning steps from the chain
    # Reasoning chains are numbered: "1. First step. 2. Second step."
    steps = [s.strip() for s in reasoning_chain.split(". ") if s.strip()]

    if not steps:
        return 0.5  # No expected reasoning to compare against

    # Check how many reasoning steps are reflected in the answer
    step_matches = 0
    for step in steps:
        step_phrases = _extract_key_phrases(step)
        if not step_phrases:
            continue
        # Check if key phrases from this step appear in the answer
        overlap = len(step_phrases & _extract_key_phrases(actual_answer))
        if overlap >= min(2, len(step_phrases)):  # At least 2 matching phrases (or all if < 2)
            step_matches += 1

    step_ratio = step_matches / len(steps) if steps else 0.0

    # Check for reasoning structure (numbered steps, causal language, etc.)
    reasoning_indicators = [
        "because", "therefore", "since", "given that", "as a result",
        "this means", "which means", "so", "thus", "consequently",
        "first", "second", "then", "finally", "additionally",
        "1.", "2.", "3.", "step",
    ]
    has_structure = any(indicator in answer_lower for indicator in reasoning_indicators)

    if step_ratio >= 0.6 and has_structure:
        return min(1.0, step_ratio + 0.1)
    elif step_ratio >= 0.4:
        return step_ratio + (0.1 if has_structure else 0.0)
    elif has_structure:
        return max(0.3, step_ratio + 0.15)
    else:
        return step_ratio


def score_fact_usage(
    required_facts: list[str],
    actual_answer: str,
) -> float:
    """Score whether the agent used the right facts to support its decision.

    Checks if the agent's response references the facts that are required
    for making the correct decision.

    Args:
        required_facts: Key facts that must be referenced for a good decision
        actual_answer: Agent's response

    Returns:
        Score from 0.0 to 1.0
    """
    if not actual_answer or not actual_answer.strip():
        return 0.0

    if not required_facts:
        return 1.0  # No required facts = automatic full marks

    answer_lower = _normalize(actual_answer)
    answer_phrases = _extract_key_phrases(actual_answer)

    facts_referenced = 0
    for fact_desc in required_facts:
        fact_phrases = _extract_key_phrases(fact_desc)
        if not fact_phrases:
            continue

        # Check if key phrases from this fact appear in the answer
        overlap = len(fact_phrases & answer_phrases)
        # A fact is "referenced" if at least half its key phrases appear
        threshold = max(1, len(fact_phrases) // 2)
        if overlap >= threshold:
            facts_referenced += 1

    return round(facts_referenced / len(required_facts), 4)


def score_decision_scenario(
    expected_decision: str,
    alternative_decisions: list[str],
    reasoning_chain: str,
    required_facts: list[str],
    actual_answer: str,
    scenario_id: str = "",
    weight_decision: float = 0.4,
    weight_reasoning: float = 0.3,
    weight_facts: float = 0.3,
) -> DecisionScore:
    """Compute composite decision score for a scenario.

    Args:
        expected_decision: The primary correct decision
        alternative_decisions: Other acceptable decisions
        reasoning_chain: Expected chain of reasoning
        required_facts: Key facts that must be referenced
        actual_answer: Agent's response
        scenario_id: Identifier for the scenario
        weight_decision: Weight for decision quality
        weight_reasoning: Weight for reasoning quality
        weight_facts: Weight for fact usage

    Returns:
        DecisionScore with all sub-scores and overall
    """
    decision = score_decision_quality(expected_decision, alternative_decisions, actual_answer)
    reasoning = score_reasoning_quality(reasoning_chain, actual_answer)
    facts = score_fact_usage(required_facts, actual_answer)

    overall = (
        decision * weight_decision
        + reasoning * weight_reasoning
        + facts * weight_facts
    )

    details = (
        f"Decision: {decision:.2f}, "
        f"Reasoning: {reasoning:.2f}, "
        f"Fact usage: {facts:.2f} ({len(required_facts)} required facts)"
    )

    return DecisionScore(
        scenario_id=scenario_id,
        decision_quality=round(decision, 4),
        reasoning_quality=round(reasoning, 4),
        fact_usage=round(facts, 4),
        overall=round(overall, 4),
        details=details,
    )


def aggregate_decision_scores(results: list[DecisionScore]) -> dict[str, float]:
    """Compute aggregate statistics across multiple decision scenario results.

    Returns:
        Dict with averages for each metric
    """
    if not results:
        return {
            "avg_decision_quality": 0.0,
            "avg_reasoning_quality": 0.0,
            "avg_fact_usage": 0.0,
            "avg_overall": 0.0,
        }

    n = len(results)
    return {
        "avg_decision_quality": round(sum(r.decision_quality for r in results) / n, 4),
        "avg_reasoning_quality": round(sum(r.reasoning_quality for r in results) / n, 4),
        "avg_fact_usage": round(sum(r.fact_usage for r in results) / n, 4),
        "avg_overall": round(sum(r.overall for r in results) / n, 4),
    }


__all__ = [
    "DecisionScore",
    "score_decision_quality",
    "score_reasoning_quality",
    "score_fact_usage",
    "score_decision_scenario",
    "aggregate_decision_scores",
]
