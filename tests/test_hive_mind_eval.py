"""Tests for the hive mind evaluation system.

Tests cover:
- Scenario data generation (all 5 scenarios well-formed)
- Scoring logic (mock responses, verify score calculation)
- Adapter basics (learn_distributed, ask_agent, ask_all)
- InMemorySharedStore operations
- PropagationResult serialization
- Coverage statistics
- Edge cases (empty responses, unknown agents)
"""

from __future__ import annotations

import json

import pytest

from amplihack_eval.adapters.base import AgentAdapter, AgentResponse
from amplihack_eval.adapters.hive_mind_adapter import (
    HiveMindGroupAdapter,
    InMemorySharedStore,
    PropagationResult,
)
from amplihack_eval.data.hive_mind_scenarios import (
    ALL_HIVE_MIND_SCENARIOS,
    HiveMindQuestion,
    HiveMindScenario,
    SCENARIO_ADVERSARIAL,
    SCENARIO_ARCH,
    SCENARIO_INCIDENT,
    SCENARIO_INFRA,
    SCENARIO_RESEARCH,
    get_questions_by_difficulty,
    get_scenario_by_id,
    get_scenarios_by_difficulty,
)
from amplihack_eval.levels.hive_mind_scoring import (
    HiveMindDimensionScore,
    HiveMindEvalReport,
    HiveMindQuestionResult,
    score_hive_mind_scenario,
    score_single_response,
)


# --- Test helpers ---


class MockHiveAgent(AgentAdapter):
    """Simple mock agent for hive mind testing."""

    def __init__(self, agent_id: str = "mock", answers: dict[str, str] | None = None):
        self.agent_id = agent_id
        self.learned: list[str] = []
        self.answers_map = answers or {}
        self.closed = False

    def learn(self, content: str) -> None:
        self.learned.append(content)

    def answer(self, question: str) -> AgentResponse:
        # Check answers map first
        for key, val in self.answers_map.items():
            if key.lower() in question.lower():
                return AgentResponse(answer=val)
        # Fall back to echoing learned content
        if self.learned:
            return AgentResponse(answer=" ".join(self.learned[-3:]))
        return AgentResponse(answer="I don't know")

    def reset(self) -> None:
        self.learned.clear()

    def close(self) -> None:
        self.closed = True

    @property
    def name(self) -> str:
        return f"MockHiveAgent({self.agent_id})"


def _make_mock_agents(num: int = 3) -> dict[str, AgentAdapter]:
    """Create a dict of mock agents."""
    return {f"agent_{i}": MockHiveAgent(f"agent_{i}") for i in range(num)}


def _make_responses(
    scenario: HiveMindScenario,
    score_level: float = 0.5,
) -> dict[str, dict[str, AgentResponse]]:
    """Create mock responses for a scenario at a given score level.

    If score_level >= 0.5, includes some expected keywords in answers.
    """
    responses: dict[str, dict[str, AgentResponse]] = {}
    agent_ids = list(scenario.agent_domains.keys())

    for agent_id in agent_ids:
        responses[agent_id] = {}
        for question in scenario.questions:
            if score_level >= 0.5:
                # Include some keywords in the answer
                n_keywords = max(1, int(len(question.expected_keywords) * score_level))
                keywords = question.expected_keywords[:n_keywords]
                answer_text = f"Based on available information: {', '.join(keywords)}."
            else:
                answer_text = "I don't have enough information to answer."
            responses[agent_id][question.question_id] = AgentResponse(answer=answer_text)

    return responses


# ====================================================================
# Scenario data generation tests
# ====================================================================


class TestScenarioDataGeneration:
    def test_all_scenarios_exist(self):
        """All 5 scenarios are defined."""
        assert len(ALL_HIVE_MIND_SCENARIOS) == 5

    def test_each_scenario_has_5_agents(self):
        """Each scenario has exactly 5 agent domains."""
        for scenario in ALL_HIVE_MIND_SCENARIOS:
            assert scenario.num_agents == 5, f"{scenario.scenario_id} has {scenario.num_agents} agents"
            assert len(scenario.agent_domains) == 5, (
                f"{scenario.scenario_id} has {len(scenario.agent_domains)} domains"
            )

    def test_each_agent_has_20_facts(self):
        """Each agent domain has exactly 20 facts."""
        for scenario in ALL_HIVE_MIND_SCENARIOS:
            for domain, facts in scenario.agent_domains.items():
                assert len(facts) == 20, (
                    f"{scenario.scenario_id}/{domain} has {len(facts)} facts (expected 20)"
                )

    def test_each_scenario_has_15_questions(self):
        """Each scenario has exactly 15 questions."""
        for scenario in ALL_HIVE_MIND_SCENARIOS:
            assert len(scenario.questions) == 15, (
                f"{scenario.scenario_id} has {len(scenario.questions)} questions (expected 15)"
            )

    def test_question_difficulty_distribution(self):
        """Each scenario has 5 single-domain, 5 cross-domain, 5 synthesis questions."""
        for scenario in ALL_HIVE_MIND_SCENARIOS:
            by_difficulty: dict[str, int] = {}
            for q in scenario.questions:
                by_difficulty[q.difficulty] = by_difficulty.get(q.difficulty, 0) + 1

            assert by_difficulty.get("single_domain", 0) == 5, (
                f"{scenario.scenario_id}: {by_difficulty.get('single_domain', 0)} single_domain"
            )
            assert by_difficulty.get("cross_domain", 0) == 5, (
                f"{scenario.scenario_id}: {by_difficulty.get('cross_domain', 0)} cross_domain"
            )
            assert by_difficulty.get("synthesis", 0) == 5, (
                f"{scenario.scenario_id}: {by_difficulty.get('synthesis', 0)} synthesis"
            )

    def test_question_ids_unique_within_scenario(self):
        """All question IDs are unique within each scenario."""
        for scenario in ALL_HIVE_MIND_SCENARIOS:
            ids = [q.question_id for q in scenario.questions]
            assert len(ids) == len(set(ids)), (
                f"{scenario.scenario_id} has duplicate question IDs"
            )

    def test_questions_have_expected_keywords(self):
        """Every question has at least one expected keyword."""
        for scenario in ALL_HIVE_MIND_SCENARIOS:
            for question in scenario.questions:
                assert len(question.expected_keywords) >= 1, (
                    f"{question.question_id} has no expected keywords"
                )

    def test_questions_have_required_domains(self):
        """Every question references valid agent domains."""
        for scenario in ALL_HIVE_MIND_SCENARIOS:
            valid_domains = set(scenario.agent_domains.keys())
            for question in scenario.questions:
                assert len(question.required_domains) >= 1, (
                    f"{question.question_id} has no required domains"
                )
                for domain in question.required_domains:
                    assert domain in valid_domains, (
                        f"{question.question_id} references invalid domain '{domain}'"
                    )

    def test_adversarial_scenario_has_misleading_agent(self):
        """The adversarial scenario has a 'misleading' agent domain."""
        assert "misleading" in SCENARIO_ADVERSARIAL.agent_domains

    def test_scenario_ids_unique(self):
        """All scenario IDs are unique across scenarios."""
        ids = [s.scenario_id for s in ALL_HIVE_MIND_SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_facts_are_nonempty_strings(self):
        """All facts are non-empty strings."""
        for scenario in ALL_HIVE_MIND_SCENARIOS:
            for domain, facts in scenario.agent_domains.items():
                for i, fact in enumerate(facts):
                    assert isinstance(fact, str), (
                        f"{scenario.scenario_id}/{domain}[{i}] is not a string"
                    )
                    assert len(fact.strip()) > 0, (
                        f"{scenario.scenario_id}/{domain}[{i}] is empty"
                    )


# ====================================================================
# Lookup function tests
# ====================================================================


class TestScenarioLookup:
    def test_get_scenario_by_id_found(self):
        """get_scenario_by_id returns correct scenario."""
        s = get_scenario_by_id("hive_infra")
        assert s is not None
        assert s.scenario_id == "hive_infra"

    def test_get_scenario_by_id_not_found(self):
        """get_scenario_by_id returns None for unknown ID."""
        assert get_scenario_by_id("nonexistent") is None

    def test_get_scenarios_by_difficulty(self):
        """get_scenarios_by_difficulty returns relevant scenarios."""
        # All scenarios have all difficulties
        results = get_scenarios_by_difficulty("synthesis")
        assert len(results) == 5

    def test_get_questions_by_difficulty(self):
        """get_questions_by_difficulty filters correctly."""
        single = get_questions_by_difficulty(SCENARIO_INFRA, "single_domain")
        assert len(single) == 5
        assert all(q.difficulty == "single_domain" for q in single)


# ====================================================================
# Scoring logic tests
# ====================================================================


class TestScoringLogic:
    def test_score_single_response_full_match(self):
        """All keywords present gives score 1.0."""
        score = score_single_response(
            "The F5 BIG-IP runs firmware v17.1.1",
            ["F5", "BIG-IP", "17.1.1"],
        )
        assert score == 1.0

    def test_score_single_response_partial_match(self):
        """Some keywords present gives proportional score."""
        score = score_single_response(
            "The F5 runs some firmware",
            ["F5", "BIG-IP", "17.1.1"],
        )
        assert 0.3 <= score <= 0.4  # 1 of 3

    def test_score_single_response_no_match(self):
        """No keywords present gives score 0.0."""
        score = score_single_response(
            "I have no information about this.",
            ["F5", "BIG-IP", "17.1.1"],
        )
        assert score == 0.0

    def test_score_single_response_empty_answer(self):
        """Empty answer gives score 0.0."""
        assert score_single_response("", ["keyword"]) == 0.0

    def test_score_single_response_empty_keywords(self):
        """Empty keywords with non-empty answer gives 1.0."""
        assert score_single_response("Some answer", []) == 1.0

    def test_score_single_response_case_insensitive(self):
        """Keyword matching is case-insensitive."""
        score = score_single_response("the f5 big-ip", ["F5", "BIG-IP"])
        assert score == 1.0

    def test_score_hive_mind_scenario_basic(self):
        """Full scoring pipeline produces valid report."""
        scenario = SCENARIO_INFRA
        hive_responses = _make_responses(scenario, score_level=0.8)
        baseline_responses = _make_responses(scenario, score_level=0.3)

        report = score_hive_mind_scenario(
            scenario=scenario,
            responses=hive_responses,
            baseline_responses=baseline_responses,
            coverage_stats={
                "total_facts_in_hive": 100,
                "num_agents": 5,
                "per_agent": {
                    "networking": {"own_facts": 20, "total_known": 80, "coverage_pct": 80.0},
                    "storage": {"own_facts": 20, "total_known": 80, "coverage_pct": 80.0},
                    "compute": {"own_facts": 20, "total_known": 80, "coverage_pct": 80.0},
                    "security": {"own_facts": 20, "total_known": 80, "coverage_pct": 80.0},
                    "monitoring": {"own_facts": 20, "total_known": 80, "coverage_pct": 80.0},
                },
            },
            propagation_rounds=2,
            max_propagation_rounds=3,
            total_facts_propagated=80,
        )

        assert isinstance(report, HiveMindEvalReport)
        assert report.scenario_id == "hive_infra"
        assert len(report.dimensions) == 5
        assert len(report.question_results) == 15
        assert 0.0 <= report.overall_score <= 1.0

    def test_score_hive_vs_baseline_delta(self):
        """Hive should score higher than baseline when hive has better answers."""
        scenario = SCENARIO_INFRA
        hive_responses = _make_responses(scenario, score_level=0.9)
        baseline_responses = _make_responses(scenario, score_level=0.2)

        report = score_hive_mind_scenario(
            scenario=scenario,
            responses=hive_responses,
            baseline_responses=baseline_responses,
        )

        assert report.hive_vs_baseline_delta > 0

    def test_score_adversarial_scenario(self):
        """Adversarial scenario has adversarial_resilience dimension."""
        scenario = SCENARIO_ADVERSARIAL
        responses = _make_responses(scenario, score_level=0.7)
        baseline = _make_responses(scenario, score_level=0.3)

        report = score_hive_mind_scenario(
            scenario=scenario,
            responses=responses,
            baseline_responses=baseline,
        )

        dim_names = [d.dimension for d in report.dimensions]
        assert "adversarial_resilience" in dim_names

        # Find the adversarial dimension
        adv_dim = next(d for d in report.dimensions if d.dimension == "adversarial_resilience")
        assert 0.0 <= adv_dim.score <= 1.0

    def test_per_difficulty_scores(self):
        """Report includes per-difficulty breakdown."""
        scenario = SCENARIO_INFRA
        responses = _make_responses(scenario, score_level=0.6)
        baseline = _make_responses(scenario, score_level=0.3)

        report = score_hive_mind_scenario(
            scenario=scenario,
            responses=responses,
            baseline_responses=baseline,
        )

        assert "single_domain" in report.per_difficulty_scores
        assert "cross_domain" in report.per_difficulty_scores
        assert "synthesis" in report.per_difficulty_scores


# ====================================================================
# Adapter tests
# ====================================================================


class TestHiveMindGroupAdapter:
    def test_creation(self):
        """HiveMindGroupAdapter creates with agents."""
        agents = _make_mock_agents(3)
        hive = HiveMindGroupAdapter(agents=agents)
        assert hive.num_agents == 3
        assert len(hive.agent_ids) == 3

    def test_learn_distributed(self):
        """Each agent learns its assigned facts."""
        agents = _make_mock_agents(2)
        hive = HiveMindGroupAdapter(agents=agents)

        facts_learned = hive.learn_distributed({
            "agent_0": ["Fact A1", "Fact A2"],
            "agent_1": ["Fact B1", "Fact B2", "Fact B3"],
        })

        assert facts_learned["agent_0"] == 2
        assert facts_learned["agent_1"] == 3

        # Verify facts stored in shared store
        all_facts = hive.shared_store.get_all_facts()
        assert len(all_facts) == 5

    def test_learn_distributed_unknown_agent_raises(self):
        """Learning for unknown agent raises ValueError."""
        agents = _make_mock_agents(1)
        hive = HiveMindGroupAdapter(agents=agents)

        with pytest.raises(ValueError, match="not found in hive"):
            hive.learn_distributed({"nonexistent": ["fact"]})

    def test_ask_agent(self):
        """ask_agent returns a response from the specified agent."""
        agents = {"net": MockHiveAgent("net", answers={"load balancer": "F5 BIG-IP v17"})}
        hive = HiveMindGroupAdapter(agents=agents)
        hive.learn_distributed({"net": ["The load balancer is F5 BIG-IP."]})

        response = hive.ask_agent("net", "What is the load balancer?")
        assert isinstance(response, AgentResponse)
        assert "F5" in response.answer

    def test_ask_agent_unknown_raises(self):
        """Asking unknown agent raises ValueError."""
        agents = _make_mock_agents(1)
        hive = HiveMindGroupAdapter(agents=agents)

        with pytest.raises(ValueError, match="not found in hive"):
            hive.ask_agent("nonexistent", "question")

    def test_ask_all(self):
        """ask_all returns responses from all agents."""
        agents = _make_mock_agents(3)
        hive = HiveMindGroupAdapter(agents=agents)
        hive.learn_distributed({
            "agent_0": ["Fact 0"],
            "agent_1": ["Fact 1"],
            "agent_2": ["Fact 2"],
        })

        responses = hive.ask_all("What do you know?")
        assert len(responses) == 3
        assert all(isinstance(r, AgentResponse) for r in responses.values())

    def test_propagate_knowledge(self):
        """propagate_knowledge spreads facts to all agents."""
        agents = _make_mock_agents(2)
        hive = HiveMindGroupAdapter(agents=agents, propagation_rounds=3)
        hive.learn_distributed({
            "agent_0": ["Unique fact from agent 0"],
            "agent_1": ["Unique fact from agent 1"],
        })

        result = hive.propagate_knowledge()
        assert isinstance(result, PropagationResult)
        assert result.facts_propagated >= 2  # Each agent learns the other's fact
        assert result.agents_reached >= 1

    def test_reset(self):
        """reset clears all agent state and shared store."""
        agents = _make_mock_agents(2)
        hive = HiveMindGroupAdapter(agents=agents)
        hive.learn_distributed({
            "agent_0": ["Fact A"],
            "agent_1": ["Fact B"],
        })

        hive.reset()
        assert len(hive.shared_store.get_all_facts()) == 0

    def test_close(self):
        """close calls close on all agents."""
        agents = {"a": MockHiveAgent("a"), "b": MockHiveAgent("b")}
        hive = HiveMindGroupAdapter(agents=agents)
        hive.close()

        for agent in agents.values():
            assert agent.closed

    def test_coverage_stats(self):
        """get_coverage_stats returns valid coverage data."""
        agents = _make_mock_agents(2)
        hive = HiveMindGroupAdapter(agents=agents)
        hive.learn_distributed({
            "agent_0": ["Fact A1", "Fact A2"],
            "agent_1": ["Fact B1"],
        })

        stats = hive.get_coverage_stats()
        assert stats["total_facts_in_hive"] == 3
        assert stats["num_agents"] == 2
        assert "agent_0" in stats["per_agent"]
        assert stats["per_agent"]["agent_0"]["own_facts"] == 2


# ====================================================================
# InMemorySharedStore tests
# ====================================================================


class TestInMemorySharedStore:
    def test_store_and_get(self):
        """store and get_all_facts work correctly."""
        store = InMemorySharedStore()
        store.store("a", "fact1")
        store.store("a", "fact2")
        store.store("b", "fact3")

        assert len(store.get_all_facts()) == 3
        assert len(store.get_all_facts("a")) == 2
        assert len(store.get_all_facts("b")) == 1

    def test_query(self):
        """query returns facts matching question keywords."""
        store = InMemorySharedStore()
        store.store("a", "The load balancer runs F5 firmware.")
        store.store("b", "Database uses PostgreSQL 16.")

        results = store.query("What firmware does the load balancer use?")
        assert any("F5" in r for r in results)

    def test_query_empty_store(self):
        """query on empty store returns empty list."""
        store = InMemorySharedStore()
        assert store.query("anything") == []

    def test_get_agent_ids(self):
        """get_agent_ids returns all agents with stored facts."""
        store = InMemorySharedStore()
        store.store("alpha", "fact")
        store.store("beta", "fact")
        assert set(store.get_agent_ids()) == {"alpha", "beta"}

    def test_clear(self):
        """clear removes all facts."""
        store = InMemorySharedStore()
        store.store("a", "fact")
        store.clear()
        assert len(store.get_all_facts()) == 0
        assert len(store.get_agent_ids()) == 0


# ====================================================================
# Serialization tests
# ====================================================================


class TestSerialization:
    def test_propagation_result_to_dict(self):
        """PropagationResult serializes to dict."""
        result = PropagationResult(rounds_executed=3, facts_propagated=50, agents_reached=4)
        d = result.to_dict()
        assert d["rounds_executed"] == 3
        assert d["facts_propagated"] == 50
        json.dumps(d)  # Should not raise

    def test_dimension_score_to_dict(self):
        """HiveMindDimensionScore serializes to dict."""
        score = HiveMindDimensionScore(dimension="test", score=0.85, details="good")
        d = score.to_dict()
        assert d["dimension"] == "test"
        assert d["score"] == 0.85
        json.dumps(d)

    def test_question_result_to_dict(self):
        """HiveMindQuestionResult serializes to dict."""
        qr = HiveMindQuestionResult(
            question_id="q1",
            question_text="Test?",
            difficulty="single_domain",
            required_domains=["net"],
            hive_answer="Answer",
            baseline_answer="Baseline",
            hive_score=0.8,
            baseline_score=0.3,
            keywords_found=["key1"],
            keywords_missing=["key2"],
        )
        d = qr.to_dict()
        assert d["improvement"] == 0.5
        json.dumps(d)

    def test_eval_report_to_dict(self):
        """HiveMindEvalReport fully serializes to JSON."""
        report = HiveMindEvalReport(
            scenario_id="test",
            dimensions=[
                HiveMindDimensionScore(dimension="d1", score=0.9, details="ok"),
            ],
            question_results=[],
            overall_score=0.85,
            hive_vs_baseline_delta=0.3,
            per_difficulty_scores={"single_domain": 0.9},
        )
        d = report.to_dict()
        json_str = json.dumps(d)
        assert "test" in json_str
        assert d["overall_score"] == 0.85


# ====================================================================
# Import verification
# ====================================================================


class TestImports:
    def test_scenario_imports(self):
        """All scenario exports are importable."""
        from amplihack_eval.data.hive_mind_scenarios import (
            ALL_HIVE_MIND_SCENARIOS,
            HiveMindQuestion,
            HiveMindScenario,
            SCENARIO_ADVERSARIAL,
            SCENARIO_ARCH,
            SCENARIO_INCIDENT,
            SCENARIO_INFRA,
            SCENARIO_RESEARCH,
            get_questions_by_difficulty,
            get_scenario_by_id,
            get_scenarios_by_difficulty,
        )
        assert HiveMindScenario is not None
        assert len(ALL_HIVE_MIND_SCENARIOS) == 5

    def test_adapter_imports(self):
        """All adapter exports are importable."""
        from amplihack_eval.adapters.hive_mind_adapter import (
            HiveMindGroupAdapter,
            InMemorySharedStore,
            PropagationResult,
            SharedMemoryStore,
        )
        assert HiveMindGroupAdapter is not None
        assert InMemorySharedStore is not None

    def test_scoring_imports(self):
        """All scoring exports are importable."""
        from amplihack_eval.levels.hive_mind_scoring import (
            HiveMindDimensionScore,
            HiveMindEvalReport,
            HiveMindQuestionResult,
            score_hive_mind_scenario,
            score_single_response,
        )
        assert score_hive_mind_scenario is not None
        assert score_single_response is not None
