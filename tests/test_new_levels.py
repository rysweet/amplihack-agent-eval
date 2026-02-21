"""Comprehensive tests for L13-L16 evaluation levels.

Tests cover:
- Data generation and scenario integrity for each level
- Scoring functions with known inputs and expected outputs
- Edge cases (empty inputs, missing data, boundary conditions)
- Aggregate scoring
"""

from __future__ import annotations

import pytest

# ══════════════════════════════════════════════════════════════════════
# L13: Tool Use Selection
# ══════════════════════════════════════════════════════════════════════

from amplihack_eval.data.tool_use_scenarios import (
    ALL_TOOL_USE_SCENARIOS,
    ALL_TOOLS,
    ToolDefinition,
    ToolUseScenario,
    get_scenario_by_id,
    get_scenarios_by_difficulty,
    get_scenarios_by_domain,
)
from amplihack_eval.levels.L13_tool_selection import (
    ToolSelectionScore,
    ToolTrajectory,
    aggregate_scores,
    score_batch,
    score_scenario,
    score_tool_chain,
    score_tool_efficiency,
    score_tool_selection,
)


class TestL13Data:
    """Tests for tool use scenario data."""

    def test_scenario_count(self):
        """At least 20 scenarios defined."""
        assert len(ALL_TOOL_USE_SCENARIOS) >= 20

    def test_all_tools_defined(self):
        """All standard tools have name and description."""
        assert len(ALL_TOOLS) >= 8
        for tool in ALL_TOOLS:
            assert tool.name != ""
            assert tool.description != ""

    def test_scenario_structure(self):
        """Each scenario has all required fields."""
        for s in ALL_TOOL_USE_SCENARIOS:
            assert s.scenario_id != ""
            assert s.task_description != ""
            assert len(s.available_tools) > 0
            assert len(s.expected_tool_sequence) > 0
            assert len(s.incorrect_alternatives) > 0
            assert s.domain in ("memory_search", "fact_storage", "explanation",
                                "gap_finding", "verification")
            assert s.difficulty in ("single", "chain", "complex_chain")
            assert s.rationale != ""

    def test_get_scenario_by_id(self):
        """Lookup by ID works."""
        s = get_scenario_by_id("TU01")
        assert s is not None
        assert s.scenario_id == "TU01"

    def test_get_scenario_by_id_not_found(self):
        """Lookup for nonexistent ID returns None."""
        assert get_scenario_by_id("NONEXISTENT") is None

    def test_get_scenarios_by_domain(self):
        """Filter by domain returns correct scenarios."""
        security = get_scenarios_by_domain("verification")
        assert len(security) > 0
        assert all(s.domain == "verification" for s in security)

    def test_get_scenarios_by_difficulty(self):
        """Filter by difficulty returns correct scenarios."""
        singles = get_scenarios_by_difficulty("single")
        assert len(singles) > 0
        assert all(s.difficulty == "single" for s in singles)

    def test_expected_tools_are_valid(self):
        """Expected tool sequences use tools from the available set."""
        tool_names = {t.name for t in ALL_TOOLS}
        for s in ALL_TOOL_USE_SCENARIOS:
            for tool in s.expected_tool_sequence:
                assert tool in tool_names, f"Scenario {s.scenario_id}: unknown tool '{tool}'"

    def test_incorrect_alternatives_differ(self):
        """Incorrect alternatives differ from expected sequence."""
        for s in ALL_TOOL_USE_SCENARIOS:
            for alt in s.incorrect_alternatives:
                assert alt != s.expected_tool_sequence, (
                    f"Scenario {s.scenario_id}: alternative matches expected"
                )


class TestL13Scoring:
    """Tests for tool selection scoring functions."""

    def test_perfect_selection(self):
        """Perfect match scores 1.0."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=["memory_search", "fact_verify"])
        score = score_tool_selection(["memory_search", "fact_verify"], traj)
        assert score == 1.0

    def test_partial_selection(self):
        """Partial match scores between 0 and 1."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=["memory_search"])
        score = score_tool_selection(["memory_search", "fact_verify"], traj)
        assert 0.0 < score < 1.0

    def test_wrong_selection(self):
        """Completely wrong tools score 0."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=["classify", "alert"])
        score = score_tool_selection(["memory_search", "fact_verify"], traj)
        assert score == 0.0

    def test_empty_trajectory(self):
        """Empty trajectory scores 0."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=[])
        score = score_tool_selection(["memory_search"], traj)
        assert score == 0.0

    def test_empty_expected(self):
        """Empty expected with empty actual scores 1."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=[])
        score = score_tool_selection([], traj)
        assert score == 1.0

    def test_perfect_efficiency(self):
        """Same count = perfect efficiency."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=["a", "b"])
        score = score_tool_efficiency(["x", "y"], traj)
        assert score == 1.0

    def test_extra_calls_penalized(self):
        """Extra calls reduce efficiency."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=["a", "b", "c", "d"])
        score = score_tool_efficiency(["a", "b"], traj)
        assert score == 0.5  # 2/4

    def test_fewer_calls_penalized(self):
        """Fewer calls also reduce efficiency."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=["a"])
        score = score_tool_efficiency(["a", "b", "c"], traj)
        assert abs(score - 1 / 3) < 0.01

    def test_both_empty_efficiency(self):
        """Both empty = perfect efficiency."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=[])
        assert score_tool_efficiency([], traj) == 1.0

    def test_perfect_chain(self):
        """Exact sequence match scores 1.0."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=["memory_search", "fact_verify"])
        score = score_tool_chain(["memory_search", "fact_verify"], traj)
        assert score == 1.0

    def test_reversed_chain(self):
        """Reversed sequence scores lower (LCS = 1)."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=["fact_verify", "memory_search"])
        score = score_tool_chain(["memory_search", "fact_verify"], traj)
        assert score == 0.5  # LCS=1 out of 2

    def test_chain_with_extras(self):
        """Extra calls in between don't break chain score."""
        traj = ToolTrajectory(
            scenario_id="test",
            tool_calls=["memory_search", "classify", "fact_verify"],
        )
        score = score_tool_chain(["memory_search", "fact_verify"], traj)
        assert score == 1.0  # LCS preserves subsequence

    def test_empty_chain(self):
        """Empty trajectory vs non-empty expected = 0."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=[])
        assert score_tool_chain(["memory_search"], traj) == 0.0

    def test_both_empty_chain(self):
        """Both empty = 1.0."""
        traj = ToolTrajectory(scenario_id="test", tool_calls=[])
        assert score_tool_chain([], traj) == 1.0

    def test_score_scenario_composite(self):
        """Composite score produces valid ToolSelectionScore."""
        traj = ToolTrajectory(scenario_id="TU01", tool_calls=["memory_search"])
        result = score_scenario(["memory_search"], traj)
        assert isinstance(result, ToolSelectionScore)
        assert result.scenario_id == "TU01"
        assert result.selection_accuracy == 1.0
        assert result.efficiency == 1.0
        assert result.chain_correctness == 1.0
        assert result.overall == 1.0

    def test_score_batch(self):
        """Batch scoring returns one result per scenario."""
        batch = [
            (["memory_search"], ToolTrajectory(scenario_id="s1", tool_calls=["memory_search"])),
            (["fact_store"], ToolTrajectory(scenario_id="s2", tool_calls=["fact_store"])),
        ]
        results = score_batch(batch)
        assert len(results) == 2

    def test_aggregate_scores(self):
        """Aggregate produces valid averages."""
        scores = [
            ToolSelectionScore("s1", 1.0, 1.0, 1.0, 1.0),
            ToolSelectionScore("s2", 0.5, 0.5, 0.5, 0.5),
        ]
        agg = aggregate_scores(scores)
        assert agg["avg_selection"] == 0.75
        assert agg["avg_efficiency"] == 0.75
        assert agg["avg_overall"] == 0.75

    def test_aggregate_empty(self):
        """Aggregate with empty list returns zeros."""
        agg = aggregate_scores([])
        assert agg["avg_overall"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# L14: Selective Forgetting
# ══════════════════════════════════════════════════════════════════════

from amplihack_eval.data.forgetting_scenarios import (
    ALL_FORGETTING_SCENARIOS,
    FactUpdate,
    ForgettingScenario,
    get_forgetting_scenario_by_id,
    get_forgetting_scenarios_by_domain,
)
from amplihack_eval.levels.L14_selective_forgetting import (
    ForgettingResult,
    aggregate_forgetting_scores,
    score_current_value_accuracy,
    score_forgetting_scenario,
    score_stale_data_penalty,
    score_update_awareness,
)


class TestL14Data:
    """Tests for forgetting scenario data."""

    def test_scenario_count(self):
        """At least 15 scenarios defined."""
        assert len(ALL_FORGETTING_SCENARIOS) >= 15

    def test_scenario_structure(self):
        """Each scenario has all required fields."""
        for s in ALL_FORGETTING_SCENARIOS:
            assert s.scenario_id != ""
            assert s.domain in ("people", "projects", "infrastructure")
            assert s.entity != ""
            assert s.attribute != ""
            assert len(s.updates) >= 2
            assert s.question != ""
            assert s.expected_current_value != ""
            assert len(s.superseded_values) >= 1
            assert s.rationale != ""

    def test_last_update_is_current(self):
        """The last update in each scenario is marked as current."""
        for s in ALL_FORGETTING_SCENARIOS:
            assert s.updates[-1].is_current, f"Scenario {s.scenario_id}: last update not marked current"

    def test_superseded_values_match_updates(self):
        """Superseded values correspond to non-current updates."""
        for s in ALL_FORGETTING_SCENARIOS:
            non_current_values = [u.value for u in s.updates if not u.is_current]
            for sv in s.superseded_values:
                assert sv in non_current_values, (
                    f"Scenario {s.scenario_id}: superseded value '{sv}' not in non-current updates"
                )

    def test_get_scenario_by_id(self):
        """Lookup by ID works."""
        s = get_forgetting_scenario_by_id("F01")
        assert s is not None
        assert s.entity == "Sarah Chen"

    def test_get_scenario_by_id_not_found(self):
        assert get_forgetting_scenario_by_id("NONEXISTENT") is None

    def test_get_scenarios_by_domain(self):
        """Filter by domain works."""
        people = get_forgetting_scenarios_by_domain("people")
        assert len(people) > 0
        assert all(s.domain == "people" for s in people)

    def test_three_domains_covered(self):
        """All three domains have scenarios."""
        for domain in ("people", "projects", "infrastructure"):
            assert len(get_forgetting_scenarios_by_domain(domain)) > 0


class TestL14Scoring:
    """Tests for selective forgetting scoring functions."""

    def test_perfect_current_value(self):
        """Agent returns exact current value -> 1.0."""
        score = score_current_value_accuracy("Engineering Director", "Sarah Chen is now Engineering Director.")
        assert score == 1.0

    def test_missing_current_value(self):
        """Agent returns wrong value -> 0.0."""
        score = score_current_value_accuracy("Engineering Director", "Sarah Chen is a Junior Engineer.")
        assert score < 0.5

    def test_empty_answer_accuracy(self):
        """Empty answer -> 0.0."""
        assert score_current_value_accuracy("Engineering Director", "") == 0.0

    def test_no_stale_data_mentioned(self):
        """No old values mentioned -> 1.0."""
        score = score_stale_data_penalty(
            ["Junior Engineer", "Senior Engineer"],
            "Sarah is the Engineering Director.",
        )
        assert score == 1.0

    def test_stale_data_without_context(self):
        """Old values mentioned without historical context -> penalty."""
        score = score_stale_data_penalty(
            ["Junior Engineer", "Senior Engineer"],
            "Sarah is a Senior Engineer and Engineering Director.",
        )
        assert score < 1.0

    def test_stale_data_with_context(self):
        """Old values mentioned with historical context -> OK."""
        score = score_stale_data_penalty(
            ["Junior Engineer"],
            "Sarah was previously a Junior Engineer but is now the Engineering Director.",
        )
        assert score == 1.0

    def test_empty_answer_stale(self):
        """Empty answer -> no stale data = 1.0."""
        assert score_stale_data_penalty(["old"], "") == 1.0

    def test_no_superseded_values(self):
        """No superseded values -> 1.0."""
        assert score_stale_data_penalty([], "any answer") == 1.0

    def test_full_update_awareness(self):
        """Agent mentions old + new with update language -> 1.0."""
        score = score_update_awareness(
            ["Senior Engineer"],
            "Engineering Director",
            "Sarah was promoted from Senior Engineer to Engineering Director.",
        )
        assert score == 1.0

    def test_partial_update_awareness(self):
        """Agent mentions current with update language but no old value."""
        score = score_update_awareness(
            ["Senior Engineer"],
            "Engineering Director",
            "Sarah is now the Engineering Director.",
        )
        assert 0.5 < score < 1.0

    def test_minimal_update_awareness(self):
        """Agent just says current value with no context."""
        score = score_update_awareness(
            ["Senior Engineer"],
            "Engineering Director",
            "Engineering Director.",
        )
        assert score <= 0.5

    def test_empty_answer_awareness(self):
        """Empty answer -> 0.0 awareness."""
        assert score_update_awareness(["old"], "current", "") == 0.0

    def test_composite_forgetting_score(self):
        """Composite score produces valid ForgettingResult."""
        result = score_forgetting_scenario(
            expected_current_value="Engineering Director",
            superseded_values=["Junior Engineer", "Senior Engineer"],
            actual_answer="Sarah was previously a Senior Engineer but was promoted to Engineering Director.",
            scenario_id="F01",
        )
        assert isinstance(result, ForgettingResult)
        assert result.scenario_id == "F01"
        assert 0.0 <= result.overall <= 1.0

    def test_aggregate_forgetting(self):
        """Aggregate produces valid averages."""
        results = [
            ForgettingResult("F01", 1.0, 1.0, 1.0, 1.0),
            ForgettingResult("F02", 0.5, 0.5, 0.5, 0.5),
        ]
        agg = aggregate_forgetting_scores(results)
        assert agg["avg_current_value_accuracy"] == 0.75
        assert agg["avg_overall"] == 0.75

    def test_aggregate_empty(self):
        agg = aggregate_forgetting_scores([])
        assert agg["avg_overall"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# L15: Adversarial Recall
# ══════════════════════════════════════════════════════════════════════

from amplihack_eval.data.adversarial_scenarios import (
    ALL_ADVERSARIAL_SCENARIOS,
    COMMON_KB,
    AdversarialScenario,
    KnowledgeBaseFact,
    get_adversarial_scenario_by_id,
    get_adversarial_scenarios_by_category,
    get_adversarial_scenarios_by_difficulty,
)
from amplihack_eval.levels.L15_adversarial_recall import (
    AdversarialRecallScore,
    aggregate_adversarial_scores,
    score_adversarial_scenario,
    score_confidence_calibration,
    score_fact_boundary_awareness,
    score_hallucination_resistance,
)


class TestL15Data:
    """Tests for adversarial scenario data."""

    def test_scenario_count(self):
        """At least 20 scenarios defined."""
        assert len(ALL_ADVERSARIAL_SCENARIOS) >= 20

    def test_scenario_structure(self):
        """Each scenario has all required fields."""
        for s in ALL_ADVERSARIAL_SCENARIOS:
            assert s.scenario_id != ""
            assert s.category in (
                "never_mentioned", "nonexistent_entity",
                "mixed_fabrication", "plausible_inference",
            )
            assert len(s.kb_facts) > 0
            assert s.adversarial_question != ""
            assert s.why_adversarial != ""
            assert s.correct_behavior != ""
            assert s.hallucination_trap != ""
            assert s.difficulty in ("easy", "medium", "hard")

    def test_common_kb_has_facts(self):
        """Common KB has 8 facts."""
        assert len(COMMON_KB) == 8

    def test_get_scenario_by_id(self):
        s = get_adversarial_scenario_by_id("A01")
        assert s is not None
        assert "birthday" in s.adversarial_question.lower()

    def test_get_scenario_by_id_not_found(self):
        assert get_adversarial_scenario_by_id("NONEXISTENT") is None

    def test_get_by_category(self):
        never_mentioned = get_adversarial_scenarios_by_category("never_mentioned")
        assert len(never_mentioned) > 0
        assert all(s.category == "never_mentioned" for s in never_mentioned)

    def test_get_by_difficulty(self):
        easy = get_adversarial_scenarios_by_difficulty("easy")
        assert len(easy) > 0
        assert all(s.difficulty == "easy" for s in easy)

    def test_all_categories_covered(self):
        """All four categories have scenarios."""
        for cat in ("never_mentioned", "nonexistent_entity", "mixed_fabrication", "plausible_inference"):
            scenarios = get_adversarial_scenarios_by_category(cat)
            assert len(scenarios) >= 3, f"Category '{cat}' has fewer than 3 scenarios"


class TestL15Scoring:
    """Tests for adversarial recall scoring functions."""

    def test_perfect_resistance(self):
        """Agent says 'I don't know' -> high score."""
        score = score_hallucination_resistance(
            "I don't know Sarah's birthday - that information is not in my records.",
            "Making up a date like 'March 15, 1990'.",
        )
        assert score >= 0.7

    def test_hallucinated_answer(self):
        """Agent makes up an answer matching the trap -> low score."""
        score = score_hallucination_resistance(
            "Sarah Chen's birthday is March 15, 1990.",
            "Making up a date like 'March 15, 1990'.",
        )
        assert score <= 0.3

    def test_empty_answer_resistance(self):
        """Empty answer is neutral."""
        score = score_hallucination_resistance("", "trap")
        assert score == 0.5

    def test_good_boundary_awareness(self):
        """Agent references known facts and acknowledges unknown -> high score."""
        score = score_fact_boundary_awareness(
            "I know Sarah Chen is the Engineering Director, but her birthday is not available.",
            ["Sarah Chen is the Engineering Director. She joined in January 2025."],
            "What is Sarah Chen's birthday?",
        )
        assert score >= 0.6

    def test_no_boundary_awareness(self):
        """Agent doesn't acknowledge known or unknown -> low score."""
        score = score_fact_boundary_awareness(
            "The birthday is April 5th.",
            ["Sarah Chen is the Engineering Director."],
            "What is Sarah Chen's birthday?",
        )
        assert score <= 0.3

    def test_empty_answer_boundary(self):
        assert score_fact_boundary_awareness("", [], "question") == 0.0

    def test_appropriately_uncertain(self):
        """Uncertain language -> high calibration score."""
        score = score_confidence_calibration(
            "I'm not sure about that - the information may not be available.",
            "Say that the information is not available.",
        )
        assert score >= 0.7

    def test_falsely_confident(self):
        """Confident fabrication language -> low calibration score."""
        score = score_confidence_calibration(
            "The answer is definitely March 15. I can confirm that.",
            "Say that the information is not available.",
        )
        assert score <= 0.3

    def test_composite_adversarial_score(self):
        """Composite scoring produces valid AdversarialRecallScore."""
        result = score_adversarial_scenario(
            actual_answer="I don't know Sarah's birthday - only her role as Engineering Director is recorded.",
            hallucination_trap="Making up a date",
            correct_behavior="Say birthday is not available",
            kb_facts_content=["Sarah Chen is the Engineering Director."],
            adversarial_question="What is Sarah Chen's birthday?",
            scenario_id="A01",
        )
        assert isinstance(result, AdversarialRecallScore)
        assert result.scenario_id == "A01"
        assert 0.0 <= result.overall <= 1.0

    def test_aggregate_adversarial(self):
        results = [
            AdversarialRecallScore("A01", 1.0, 1.0, 1.0, 1.0),
            AdversarialRecallScore("A02", 0.5, 0.5, 0.5, 0.5),
        ]
        agg = aggregate_adversarial_scores(results)
        assert agg["avg_hallucination_resistance"] == 0.75

    def test_aggregate_empty(self):
        agg = aggregate_adversarial_scores([])
        assert agg["avg_overall"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# L16: Decision From Memory
# ══════════════════════════════════════════════════════════════════════

from amplihack_eval.data.decision_scenarios import (
    ALL_DECISION_SCENARIOS,
    ContextFact,
    DecisionScenario,
    get_decision_scenario_by_id,
    get_decision_scenarios_by_difficulty,
    get_decision_scenarios_by_domain,
)
from amplihack_eval.levels.L16_decision_from_memory import (
    DecisionScore,
    aggregate_decision_scores,
    score_decision_quality,
    score_decision_scenario,
    score_fact_usage,
    score_reasoning_quality,
)


class TestL16Data:
    """Tests for decision scenario data."""

    def test_scenario_count(self):
        """At least 15 scenarios defined."""
        assert len(ALL_DECISION_SCENARIOS) >= 15

    def test_scenario_structure(self):
        """Each scenario has all required fields."""
        for s in ALL_DECISION_SCENARIOS:
            assert s.scenario_id != ""
            assert s.domain in (
                "security", "project_management", "infrastructure",
                "hiring", "resource_allocation",
            )
            assert len(s.context_facts) >= 2
            assert s.decision_question != ""
            assert s.expected_decision != ""
            assert len(s.required_facts_for_decision) >= 2
            assert s.reasoning_chain != ""
            assert len(s.alternative_acceptable_decisions) >= 1
            assert s.difficulty in ("moderate", "hard", "very_hard")

    def test_get_scenario_by_id(self):
        s = get_decision_scenario_by_id("D01")
        assert s is not None
        assert "firewall" in s.decision_question.lower()

    def test_get_scenario_by_id_not_found(self):
        assert get_decision_scenario_by_id("NONEXISTENT") is None

    def test_get_by_domain(self):
        security = get_decision_scenarios_by_domain("security")
        assert len(security) > 0
        assert all(s.domain == "security" for s in security)

    def test_get_by_difficulty(self):
        moderate = get_decision_scenarios_by_difficulty("moderate")
        assert len(moderate) > 0
        assert all(s.difficulty == "moderate" for s in moderate)

    def test_multiple_domains_covered(self):
        """At least 3 domains have scenarios."""
        domains = {s.domain for s in ALL_DECISION_SCENARIOS}
        assert len(domains) >= 3

    def test_context_facts_have_content(self):
        """All context facts have content and relevance."""
        for s in ALL_DECISION_SCENARIOS:
            for cf in s.context_facts:
                assert cf.content != ""
                assert cf.relevance != ""


class TestL16Scoring:
    """Tests for decision-from-memory scoring functions."""

    def test_good_decision(self):
        """Answer matching expected decision scores well."""
        score = score_decision_quality(
            expected_decision="Block 192.168.1.45 on port 22 and add rate limiting.",
            alternative_decisions=["Block the entire IP"],
            actual_answer="We should block 192.168.1.45 specifically on port 22 SSH and add rate limiting to prevent further brute force attacks.",
        )
        assert score >= 0.5

    def test_alternative_decision(self):
        """Answer matching an alternative scores well."""
        score = score_decision_quality(
            expected_decision="Deploy emergency patch immediately.",
            alternative_decisions=["Deploy emergency patch with WAF rules as interim"],
            actual_answer="We should deploy the emergency patch now and add WAF rules as an interim mitigation measure.",
        )
        assert score >= 0.4

    def test_wrong_decision(self):
        """Completely wrong answer scores low."""
        score = score_decision_quality(
            expected_decision="Block the IP and add rate limiting.",
            alternative_decisions=["Block the subnet"],
            actual_answer="The weather is nice today, let's go for a walk.",
        )
        assert score <= 0.3

    def test_empty_answer_decision(self):
        assert score_decision_quality("expected", [], "") == 0.0

    def test_good_reasoning(self):
        """Answer with structured reasoning scores well."""
        score = score_reasoning_quality(
            reasoning_chain="1. Attack from specific IP. 2. Firewall allows subnet. 3. Block IP on SSH port.",
            actual_answer="Because the attack came from 192.168.1.45 and our firewall currently allows the entire subnet, we should block this specific IP on the SSH port to prevent disruption to other services.",
        )
        assert score >= 0.3

    def test_no_reasoning(self):
        """Answer with no reasoning structure scores low."""
        score = score_reasoning_quality(
            reasoning_chain="1. Step one. 2. Step two.",
            actual_answer="Block it.",
        )
        assert score <= 0.5

    def test_empty_answer_reasoning(self):
        assert score_reasoning_quality("chain", "") == 0.0

    def test_good_fact_usage(self):
        """Answer referencing required facts scores well."""
        score = score_fact_usage(
            required_facts=["attack from 192.168.1.45", "firewall allows subnet", "SSH port 22"],
            actual_answer="Given the brute force attack from 192.168.1.45 targeting SSH on port 22, and that our firewall allows the entire subnet, we should add a specific block rule.",
        )
        assert score >= 0.5

    def test_missing_fact_usage(self):
        """Answer missing required facts scores lower."""
        score = score_fact_usage(
            required_facts=["attack from 192.168.1.45", "firewall allows subnet", "SSH port 22"],
            actual_answer="We should update the firewall rules.",
        )
        assert score < 0.5

    def test_empty_facts(self):
        """No required facts -> 1.0."""
        assert score_fact_usage([], "any answer") == 1.0

    def test_empty_answer_facts(self):
        assert score_fact_usage(["fact1"], "") == 0.0

    def test_composite_decision_score(self):
        """Composite scoring produces valid DecisionScore."""
        result = score_decision_scenario(
            expected_decision="Block IP on SSH and add rate limiting",
            alternative_decisions=["Block entire IP"],
            reasoning_chain="1. Attack from IP. 2. Firewall allows subnet. 3. Block specifically.",
            required_facts=["attack from 192.168.1.45", "SSH on port 22"],
            actual_answer="We should block 192.168.1.45 on SSH port 22 and implement rate limiting because the attack targeted SSH and our firewall currently allows the subnet.",
            scenario_id="D01",
        )
        assert isinstance(result, DecisionScore)
        assert result.scenario_id == "D01"
        assert 0.0 <= result.overall <= 1.0

    def test_aggregate_decision(self):
        results = [
            DecisionScore("D01", 1.0, 1.0, 1.0, 1.0),
            DecisionScore("D02", 0.5, 0.5, 0.5, 0.5),
        ]
        agg = aggregate_decision_scores(results)
        assert agg["avg_decision_quality"] == 0.75

    def test_aggregate_empty(self):
        agg = aggregate_decision_scores([])
        assert agg["avg_overall"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# Cross-level import tests
# ══════════════════════════════════════════════════════════════════════


class TestCrossLevelImports:
    """Verify all new modules are importable from expected locations."""

    def test_data_module_imports(self):
        """All new data modules importable from amplihack_eval.data."""
        from amplihack_eval.data import (
            ALL_ADVERSARIAL_SCENARIOS,
            ALL_DECISION_SCENARIOS,
            ALL_FORGETTING_SCENARIOS,
            ALL_TOOL_USE_SCENARIOS,
        )
        assert len(ALL_TOOL_USE_SCENARIOS) >= 20
        assert len(ALL_FORGETTING_SCENARIOS) >= 15
        assert len(ALL_ADVERSARIAL_SCENARIOS) >= 20
        assert len(ALL_DECISION_SCENARIOS) >= 15

    def test_levels_module_imports(self):
        """All new scoring modules importable from amplihack_eval.levels."""
        from amplihack_eval.levels import (
            AdversarialRecallScore,
            DecisionScore,
            ForgettingResult,
            ToolSelectionScore,
            ToolTrajectory,
            aggregate_adversarial_scores,
            aggregate_decision_scores,
            aggregate_forgetting_scores,
            aggregate_tool_scores,
            score_adversarial_scenario,
            score_decision_scenario,
            score_forgetting_scenario,
            score_tool_scenario,
        )
        assert ToolTrajectory is not None
        assert ForgettingResult is not None
        assert AdversarialRecallScore is not None
        assert DecisionScore is not None
