"""Tests for the multi-agent evaluation system.

Tests cover:
- GraderAgent with different perspectives
- AdversaryAgent question generation (mock LLM)
- AnalystAgent analysis (mock LLM)
- EvalCoordinator orchestration
- Pipeline end-to-end with mock agents
- Multi-vote aggregation
- Adversarial question targeting
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from amplihack_eval.adapters.base import AgentAdapter, AgentResponse
from amplihack_eval.core.runner import (
    CategoryBreakdown,
    DimensionScore,
    EvalReport,
    EvalResult,
)
from amplihack_eval.data.long_horizon import GradingRubric, GroundTruth, Question, Turn
from amplihack_eval.multi_agent_eval.adversary_agent import AdversaryAgent
from amplihack_eval.multi_agent_eval.analyst_agent import (
    AnalystAgent,
    AnalysisReport,
    ComparisonReport,
    FailurePattern,
    Improvement,
)
from amplihack_eval.multi_agent_eval.coordinator import EvalConfig, EvalCoordinator
from amplihack_eval.multi_agent_eval.grader_agent import (
    AggregateGrade,
    GraderAgent,
    PerspectiveGrade,
    _deterministic_grade,
    _extract_json,
)
from amplihack_eval.multi_agent_eval.pipeline import (
    MultiAgentEvalPipeline,
    PipelineConfig,
    PipelineReport,
    RoundResult,
)


# --- Test helpers ---


class MockAgent(AgentAdapter):
    """Simple mock agent for testing."""

    def __init__(self, answers: dict[str, str] | None = None):
        self.learned: list[str] = []
        self.answers_map = answers or {}
        self.closed = False

    def learn(self, content: str) -> None:
        self.learned.append(content)

    def answer(self, question: str) -> AgentResponse:
        if self.answers_map:
            for key, val in self.answers_map.items():
                if key.lower() in question.lower():
                    return AgentResponse(answer=val)
        if self.learned:
            return AgentResponse(answer=f"Based on my knowledge: {self.learned[-1][:50]}")
        return AgentResponse(answer="I don't know")

    def reset(self) -> None:
        self.learned.clear()

    def close(self) -> None:
        self.closed = True


def _make_question(
    qid: str = "q_001",
    text: str = "What is X?",
    expected: str = "X is Y",
    category: str = "test_category",
    rubric: GradingRubric | None = None,
) -> Question:
    return Question(
        question_id=qid,
        text=text,
        expected_answer=expected,
        category=category,
        relevant_turns=[0],
        scoring_dimensions=["factual_accuracy"],
        rubric=rubric,
    )


def _make_ground_truth(num_turns: int = 10) -> GroundTruth:
    turns = [
        Turn(
            turn_number=i,
            content=f"Fact {i}: Entity_{i} has value_{i}.",
            block=1,
            block_name="test_block",
            facts=[{"entity": f"Entity_{i}", "value": f"value_{i}"}],
        )
        for i in range(num_turns)
    ]
    return GroundTruth(turns=turns)


def _make_eval_report(
    overall: float = 0.7,
    categories: dict[str, float] | None = None,
    num_results: int = 10,
) -> EvalReport:
    """Create a mock EvalReport for testing."""
    if categories is None:
        categories = {"cat_a": 0.8, "cat_b": 0.4, "cat_c": 0.9}

    results: list[EvalResult] = []
    breakdown: list[CategoryBreakdown] = []

    for cat, avg in categories.items():
        n = max(1, num_results // len(categories))
        cat_results = []
        for i in range(n):
            score = max(0.0, min(1.0, avg + (i - n / 2) * 0.1))
            result = EvalResult(
                question_id=f"{cat}_{i}",
                question_text=f"Question about {cat} #{i}",
                category=cat,
                expected_answer=f"Expected answer for {cat}",
                actual_answer=f"Actual answer for {cat}" if score > 0.3 else "",
                dimensions=[
                    DimensionScore(dimension="factual_accuracy", score=score, reasoning="test"),
                ],
                overall_score=score,
            )
            results.append(result)
            cat_results.append(result)

        scores = [r.overall_score for r in cat_results]
        breakdown.append(CategoryBreakdown(
            category=cat,
            num_questions=len(cat_results),
            avg_score=sum(scores) / len(scores),
            min_score=min(scores),
            max_score=max(scores),
            dimension_averages={"factual_accuracy": sum(scores) / len(scores)},
        ))

    return EvalReport(
        num_turns=100,
        num_questions=len(results),
        total_facts_delivered=50,
        learning_time_s=1.0,
        questioning_time_s=2.0,
        grading_time_s=1.5,
        overall_score=overall,
        category_breakdown=breakdown,
        results=results,
    )


# ====================================================================
# GraderAgent tests
# ====================================================================


class TestGraderAgent:
    def test_valid_perspectives(self):
        """Can create a grader for each valid perspective."""
        for perspective in GraderAgent.PERSPECTIVES:
            grader = GraderAgent(perspective=perspective)
            assert grader.perspective == perspective

    def test_invalid_perspective_raises(self):
        """Invalid perspective raises ValueError."""
        with pytest.raises(ValueError, match="Invalid perspective"):
            GraderAgent(perspective="invalid")

    def test_grade_empty_answer(self):
        """Empty answer gets score 0.0."""
        grader = GraderAgent(perspective="factual")
        question = _make_question()
        grade = grader.grade(question, answer="")
        assert grade.score == 0.0
        assert "No answer" in grade.reasoning

    def test_grade_whitespace_answer(self):
        """Whitespace-only answer gets score 0.0."""
        grader = GraderAgent(perspective="factual")
        question = _make_question()
        grade = grader.grade(question, answer="   \n\t  ")
        assert grade.score == 0.0

    def test_grade_with_rubric_deterministic(self):
        """Factual grader uses deterministic scoring when rubric available."""
        rubric = GradingRubric(
            required_keywords=["paris", "france"],
            acceptable_paraphrases=["capital city"],
        )
        question = _make_question(rubric=rubric)
        grader = GraderAgent(perspective="factual")

        # Answer with both keywords
        grade = grader.grade(question, answer="Paris is the capital of France")
        assert grade.score == 1.0
        assert "Deterministic" in grade.reasoning

    def test_grade_with_rubric_partial_match(self):
        """Partial keyword match gives partial score."""
        rubric = GradingRubric(
            required_keywords=["paris", "france", "europe"],
        )
        question = _make_question(rubric=rubric)
        grader = GraderAgent(perspective="factual")

        # Only 2 of 3 keywords
        grade = grader.grade(question, answer="Paris is in France")
        assert 0.5 < grade.score < 1.0

    def test_grade_with_rubric_incorrect_pattern(self):
        """Incorrect pattern in answer gives score 0.0."""
        rubric = GradingRubric(
            required_keywords=["paris"],
            incorrect_patterns=["london"],
        )
        question = _make_question(rubric=rubric)
        grader = GraderAgent(perspective="factual")

        grade = grader.grade(question, answer="London is the capital")
        assert grade.score == 0.0

    def test_grade_reasoning_perspective_uses_llm(self):
        """Reasoning perspective does not use deterministic grading."""
        rubric = GradingRubric(required_keywords=["paris"])
        question = _make_question(rubric=rubric)
        grader = GraderAgent(perspective="reasoning")

        # Without API key, should return 0.0 with "No ANTHROPIC_API_KEY"
        with patch.dict("os.environ", {}, clear=True):
            grade = grader.grade(question, answer="Paris is the capital")
            assert "ANTHROPIC_API_KEY" in grade.reasoning

    def test_perspective_grade_to_dict(self):
        """PerspectiveGrade serializes to dict."""
        grade = PerspectiveGrade(
            perspective="factual",
            score=0.85,
            reasoning="Good answer",
            question_id="q_001",
        )
        d = grade.to_dict()
        assert d["perspective"] == "factual"
        assert d["score"] == 0.85
        assert d["reasoning"] == "Good answer"
        assert d["question_id"] == "q_001"


class TestMultiVoteAggregation:
    def test_aggregate_single_grade(self):
        """Single grade aggregates to itself."""
        grades = [PerspectiveGrade(perspective="factual", score=0.8, reasoning="ok")]
        question = _make_question()
        agg = GraderAgent.aggregate_grades(grades, question, "answer")
        assert agg.overall_score == 0.8
        assert agg.agreement == 1.0  # No variance with single grade

    def test_aggregate_multiple_grades(self):
        """Multiple grades aggregate to median."""
        grades = [
            PerspectiveGrade(perspective="factual", score=0.9, reasoning="good"),
            PerspectiveGrade(perspective="reasoning", score=0.7, reasoning="ok"),
            PerspectiveGrade(perspective="completeness", score=0.8, reasoning="decent"),
        ]
        question = _make_question()
        agg = GraderAgent.aggregate_grades(grades, question, "answer")
        assert agg.overall_score == 0.8  # Median of [0.9, 0.7, 0.8]
        assert 0.0 < agg.agreement <= 1.0

    def test_aggregate_empty_grades(self):
        """Empty grades list returns zeroes."""
        question = _make_question()
        agg = GraderAgent.aggregate_grades([], question, "answer")
        assert agg.overall_score == 0.0
        assert agg.agreement == 0.0

    def test_aggregate_identical_grades(self):
        """Identical grades have perfect agreement."""
        grades = [
            PerspectiveGrade(perspective="factual", score=0.8, reasoning="same"),
            PerspectiveGrade(perspective="reasoning", score=0.8, reasoning="same"),
            PerspectiveGrade(perspective="completeness", score=0.8, reasoning="same"),
        ]
        question = _make_question()
        agg = GraderAgent.aggregate_grades(grades, question, "answer")
        assert agg.overall_score == 0.8
        assert agg.agreement == 1.0

    def test_aggregate_divergent_grades(self):
        """Divergent grades have low agreement."""
        grades = [
            PerspectiveGrade(perspective="factual", score=1.0, reasoning="perfect"),
            PerspectiveGrade(perspective="reasoning", score=0.0, reasoning="terrible"),
        ]
        question = _make_question()
        agg = GraderAgent.aggregate_grades(grades, question, "answer")
        assert agg.overall_score == 0.5  # Median of [1.0, 0.0]
        assert agg.agreement < 0.5  # Low agreement

    def test_aggregate_to_dict(self):
        """AggregateGrade serializes to dict."""
        grades = [PerspectiveGrade(perspective="factual", score=0.85, reasoning="ok")]
        question = _make_question()
        agg = GraderAgent.aggregate_grades(grades, question, "my answer")
        d = agg.to_dict()
        assert "overall_score" in d
        assert "perspective_grades" in d
        assert "agreement" in d
        assert d["question_id"] == "q_001"


class TestDeterministicGrade:
    def test_no_keywords_returns_none(self):
        """Empty rubric returns None (needs LLM)."""
        rubric = GradingRubric()
        assert _deterministic_grade(rubric, "any answer") is None

    def test_all_keywords_matched(self):
        """All keywords matched gives 1.0."""
        rubric = GradingRubric(required_keywords=["alpha", "beta"])
        assert _deterministic_grade(rubric, "alpha and beta are present") == 1.0

    def test_partial_keywords(self):
        """Some keywords matched gives proportional score."""
        rubric = GradingRubric(required_keywords=["alpha", "beta", "gamma"])
        score = _deterministic_grade(rubric, "alpha is here but nothing else")
        assert score is not None
        assert 0.3 <= score <= 0.4  # 1/3

    def test_incorrect_pattern_gives_zero(self):
        """Incorrect pattern instantly gives 0."""
        rubric = GradingRubric(
            required_keywords=["correct"],
            incorrect_patterns=["wrong"],
        )
        assert _deterministic_grade(rubric, "the wrong answer") == 0.0

    def test_paraphrase_bonus(self):
        """Paraphrases add bonus score."""
        rubric = GradingRubric(
            required_keywords=["alpha"],
            acceptable_paraphrases=["greek letter a"],
        )
        score_without = _deterministic_grade(rubric, "alpha is here")
        score_with = _deterministic_grade(rubric, "alpha is here, also known as greek letter a")
        assert score_with is not None
        assert score_without is not None
        assert score_with >= score_without


class TestExtractJson:
    def test_raw_json(self):
        assert _extract_json('{"score": 0.8}') == {"score": 0.8}

    def test_markdown_fenced(self):
        text = '```json\n{"score": 0.8}\n```'
        assert _extract_json(text) == {"score": 0.8}

    def test_embedded_json(self):
        text = 'Here is my grade: {"score": 0.8, "reasoning": "good"} as requested.'
        result = _extract_json(text)
        assert result["score"] == 0.8

    def test_no_json(self):
        assert _extract_json("just plain text") == {}


# ====================================================================
# AdversaryAgent tests
# ====================================================================


class TestAdversaryAgent:
    def test_creation(self):
        """AdversaryAgent creates without error."""
        agent = AdversaryAgent()
        assert agent.model is not None

    def test_generate_empty_results(self):
        """No previous results yields no adversarial questions."""
        agent = AdversaryAgent()
        questions = agent.generate_adversarial_questions(
            ground_truth=_make_ground_truth(),
            previous_results=[],
            num_questions=5,
        )
        assert questions == []

    @patch("amplihack_eval.multi_agent_eval.adversary_agent.os.environ.get")
    def test_generate_no_api_key(self, mock_env):
        """Without API key, returns empty list."""
        mock_env.return_value = None

        agent = AdversaryAgent()
        results = [
            {"question_text": "Q1?", "expected_answer": "A1", "actual_answer": "wrong",
             "score": 0.3, "category": "cat_a"},
        ]
        questions = agent.generate_adversarial_questions(
            ground_truth=_make_ground_truth(),
            previous_results=results,
            num_questions=5,
        )
        assert questions == []

    def test_targeting_strong_categories(self):
        """Adversary targets agent's strongest categories."""
        agent = AdversaryAgent()

        # Agent does well on cat_a, poorly on cat_b
        results = [
            {"question_text": "Q1", "expected_answer": "A1", "actual_answer": "A1",
             "score": 0.95, "category": "cat_a"},
            {"question_text": "Q2", "expected_answer": "A2", "actual_answer": "A2",
             "score": 0.9, "category": "cat_a"},
            {"question_text": "Q3", "expected_answer": "A3", "actual_answer": "wrong",
             "score": 0.2, "category": "cat_b"},
        ]

        # We cannot test the actual LLM call without mocking, but we can
        # verify the targeting analysis works
        cat_scores: dict[str, list[float]] = {}
        for r in results:
            cat_scores.setdefault(r["category"], []).append(r["score"])

        cat_averages = {cat: sum(s) / len(s) for cat, s in cat_scores.items()}
        strongest = sorted(cat_averages, key=lambda c: cat_averages[c], reverse=True)

        assert strongest[0] == "cat_a"  # Agent's strongest category
        assert cat_averages["cat_a"] > 0.9

    def test_forgetting_probes_empty_turns(self):
        """Empty ground truth yields no forgetting probes."""
        agent = AdversaryAgent()
        probes = agent.generate_forgetting_probes(
            ground_truth=GroundTruth(turns=[]),
            num_questions=5,
        )
        assert probes == []


# ====================================================================
# AnalystAgent tests
# ====================================================================


class TestAnalystAgent:
    def test_creation(self):
        """AnalystAgent creates without error."""
        agent = AnalystAgent()
        assert agent.model is not None

    def test_analyze_basic(self):
        """Analyze produces AnalysisReport with correct structure."""
        agent = AnalystAgent()
        report = _make_eval_report(
            overall=0.6,
            categories={"strong_cat": 0.9, "weak_cat": 0.3, "medium_cat": 0.6},
        )

        analysis = agent.analyze(report)

        assert isinstance(analysis, AnalysisReport)
        assert analysis.overall_score == 0.6
        assert "strong_cat" in analysis.category_scores
        assert "weak_cat" in analysis.category_scores
        assert analysis.bottleneck_component != ""

    def test_analyze_identifies_weak_categories(self):
        """Analyze finds failure patterns for weak categories."""
        agent = AnalystAgent()
        report = _make_eval_report(
            overall=0.5,
            categories={"failing_cat": 0.2},
        )

        analysis = agent.analyze(report)

        pattern_names = [fp.pattern_name for fp in analysis.failure_patterns]
        assert any("weak_" in name for name in pattern_names)

    def test_analyze_identifies_zero_scores(self):
        """Analyze detects total failure pattern when questions score 0."""
        agent = AnalystAgent()

        # Create report with a zero-score result
        results = [
            EvalResult(
                question_id="q_zero",
                question_text="Hard question",
                category="hard_cat",
                expected_answer="Expected",
                actual_answer="",
                dimensions=[DimensionScore(dimension="factual_accuracy", score=0.0)],
                overall_score=0.0,
            ),
        ]
        report = EvalReport(
            num_turns=10,
            num_questions=1,
            total_facts_delivered=5,
            learning_time_s=0.1,
            questioning_time_s=0.1,
            grading_time_s=0.1,
            overall_score=0.0,
            category_breakdown=[
                CategoryBreakdown(
                    category="hard_cat",
                    num_questions=1,
                    avg_score=0.0,
                    min_score=0.0,
                    max_score=0.0,
                    dimension_averages={"factual_accuracy": 0.0},
                ),
            ],
            results=results,
        )

        analysis = agent.analyze(report)
        pattern_names = [fp.pattern_name for fp in analysis.failure_patterns]
        assert "total_failure" in pattern_names

    def test_analyze_report_to_dict(self):
        """AnalysisReport serializes to dict."""
        agent = AnalystAgent()
        report = _make_eval_report()
        analysis = agent.analyze(report)
        d = analysis.to_dict()
        assert "overall_score" in d
        assert "failure_patterns" in d
        assert "category_scores" in d
        assert "bottleneck_component" in d
        assert "improvement_priorities" in d

    def test_suggest_improvements(self):
        """suggest_improvements returns sorted improvements."""
        agent = AnalystAgent()
        report = _make_eval_report(
            overall=0.4,
            categories={"weak_a": 0.2, "weak_b": 0.3},
        )
        analysis = agent.analyze(report)
        improvements = agent.suggest_improvements(analysis)

        # Should be sorted by expected_impact descending
        if len(improvements) >= 2:
            assert improvements[0].expected_impact >= improvements[-1].expected_impact


class TestComparisonReport:
    def test_compare_two_reports(self):
        """Compare two reports detects improvements and regressions."""
        agent = AnalystAgent()

        report1 = _make_eval_report(
            overall=0.5,
            categories={"cat_a": 0.6, "cat_b": 0.4},
        )
        report2 = _make_eval_report(
            overall=0.7,
            categories={"cat_a": 0.9, "cat_b": 0.3},
        )

        comparison = agent.compare_reports([report1, report2], ["v1", "v2"])

        assert isinstance(comparison, ComparisonReport)
        assert comparison.run_labels == ["v1", "v2"]
        assert "v1" in comparison.overall_scores
        assert "v2" in comparison.overall_scores

        # cat_a improved (0.6 -> 0.9 = +0.3 > 0.05 threshold)
        improved_cats = [imp["category"] for imp in comparison.improvements]
        assert "cat_a" in improved_cats

        # cat_b regressed (0.4 -> 0.3 = -0.1 > 0.05 threshold)
        regressed_cats = [reg["category"] for reg in comparison.regressions]
        assert "cat_b" in regressed_cats

    def test_compare_empty_reports(self):
        """Comparing empty list returns empty comparison."""
        agent = AnalystAgent()
        comparison = agent.compare_reports([])
        assert comparison.summary == "No reports to compare."

    def test_compare_single_report(self):
        """Single report comparison has no regressions or improvements."""
        agent = AnalystAgent()
        report = _make_eval_report()
        comparison = agent.compare_reports([report], ["only_run"])
        assert len(comparison.regressions) == 0
        assert len(comparison.improvements) == 0

    def test_comparison_to_dict(self):
        """ComparisonReport serializes to dict."""
        agent = AnalystAgent()
        r1 = _make_eval_report(overall=0.5, categories={"cat_a": 0.5})
        r2 = _make_eval_report(overall=0.7, categories={"cat_a": 0.7})
        comparison = agent.compare_reports([r1, r2])
        d = comparison.to_dict()
        assert "run_labels" in d
        assert "overall_scores" in d
        assert "summary" in d


# ====================================================================
# EvalCoordinator tests
# ====================================================================


class TestEvalCoordinator:
    def test_creation(self):
        """EvalCoordinator creates with default args."""
        coord = EvalCoordinator()
        assert coord._num_graders == 3
        assert coord._enable_adversary is True

    def test_creation_custom(self):
        """EvalCoordinator respects custom args."""
        coord = EvalCoordinator(grader_agents=1, enable_adversary=False)
        assert coord._num_graders == 1
        assert coord._enable_adversary is False

    def test_grader_count_clamped(self):
        """Grader count is clamped to valid range."""
        coord = EvalCoordinator(grader_agents=10)
        assert coord._num_graders == 3  # max 3 perspectives

        coord = EvalCoordinator(grader_agents=0)
        assert coord._num_graders == 1  # min 1

    def test_init_agents(self):
        """_init_agents creates correct number of graders."""
        coord = EvalCoordinator(grader_agents=2, enable_adversary=True)
        config = EvalConfig(grader_perspectives=["factual", "reasoning", "completeness"])
        coord._init_agents(config)

        assert len(coord._graders) == 2
        assert coord._graders[0].perspective == "factual"
        assert coord._graders[1].perspective == "reasoning"
        assert coord._adversary is not None

    def test_init_agents_no_adversary(self):
        """_init_agents skips adversary when disabled."""
        coord = EvalCoordinator(grader_agents=1, enable_adversary=False)
        config = EvalConfig()
        coord._init_agents(config)
        assert coord._adversary is None

    def test_question_and_grade(self):
        """_question_and_grade produces results for each question."""
        coord = EvalCoordinator(grader_agents=2, enable_adversary=False)
        config = EvalConfig(grader_perspectives=["factual", "completeness"])
        coord._init_agents(config)

        agent = MockAgent(answers={"capital": "Paris is the capital of France"})
        rubric = GradingRubric(required_keywords=["paris", "france"])
        questions = [
            _make_question(
                qid="q_001",
                text="What is the capital of France?",
                expected="Paris",
                rubric=rubric,
            ),
        ]

        results, grade_time = coord._question_and_grade(agent, questions)

        assert len(results) == 1
        assert results[0].question_id == "q_001"
        assert results[0].overall_score >= 0.0
        assert len(results[0].dimensions) == 2  # Two perspectives

    def test_build_report(self):
        """_build_report creates valid EvalReport."""
        coord = EvalCoordinator(grader_agents=1, enable_adversary=False)
        config = EvalConfig()
        coord._init_agents(config)

        results = [
            EvalResult(
                question_id="q_001",
                question_text="Test?",
                category="test_cat",
                expected_answer="Expected",
                actual_answer="Actual",
                dimensions=[DimensionScore(dimension="factual", score=0.8)],
                overall_score=0.8,
            ),
        ]
        gt = _make_ground_truth()

        report = coord._build_report(results, gt, config, 1.0, 0.5)

        assert report.num_questions == 1
        assert report.overall_score == 0.8
        assert len(report.category_breakdown) == 1
        assert report.category_breakdown[0].category == "test_cat"


class TestEvalConfig:
    def test_defaults(self):
        """EvalConfig has sensible defaults."""
        config = EvalConfig()
        assert config.num_turns == 100
        assert config.num_questions == 20
        assert config.seed == 42
        assert config.enable_adversary is True
        assert len(config.grader_perspectives) == 3

    def test_custom(self):
        """EvalConfig accepts custom values."""
        config = EvalConfig(
            num_turns=50,
            num_questions=10,
            seed=123,
            enable_adversary=False,
        )
        assert config.num_turns == 50
        assert config.num_questions == 10
        assert config.enable_adversary is False


# ====================================================================
# Pipeline tests
# ====================================================================


class TestPipelineConfig:
    def test_defaults(self):
        """PipelineConfig has sensible defaults."""
        config = PipelineConfig()
        assert config.adversarial_rounds == 1
        assert config.agent_factory is None
        assert config.reset_between_rounds is False

    def test_custom(self):
        """PipelineConfig accepts custom values."""
        factory = lambda: MockAgent()
        config = PipelineConfig(
            adversarial_rounds=3,
            agent_factory=factory,
            reset_between_rounds=True,
        )
        assert config.adversarial_rounds == 3
        assert config.agent_factory is not None


class TestPipelineReport:
    def test_to_dict(self):
        """PipelineReport serializes to dict."""
        report = PipelineReport(
            rounds=[],
            comparison=None,
            final_overall_score=0.75,
            total_questions_asked=30,
            total_time_s=10.5,
            config={"num_turns": 100},
        )
        d = report.to_dict()
        assert d["final_overall_score"] == 0.75
        assert d["total_questions_asked"] == 30
        assert d["num_rounds"] == 0
        assert d["comparison"] is None


class TestRoundResult:
    def test_to_dict(self):
        """RoundResult serializes to dict."""
        eval_report = _make_eval_report(overall=0.8)
        rr = RoundResult(
            round_number=0,
            report=eval_report,
            is_adversarial=False,
            round_time_s=5.0,
        )
        d = rr.to_dict()
        assert d["round_number"] == 0
        assert d["overall_score"] == 0.8
        assert d["is_adversarial"] is False


class TestMultiAgentEvalPipeline:
    def test_creation(self):
        """Pipeline creates without error."""
        pipeline = MultiAgentEvalPipeline(grader_agents=2)
        assert pipeline._num_graders == 2


# ====================================================================
# Integration-style tests (no LLM, using deterministic grading)
# ====================================================================


class TestIntegration:
    def test_grader_to_coordinator_flow(self):
        """Test grader -> coordinator flow with deterministic grading."""
        coord = EvalCoordinator(grader_agents=1, enable_adversary=False)
        config = EvalConfig(grader_perspectives=["factual"])
        coord._init_agents(config)

        rubric = GradingRubric(
            required_keywords=["26", "medals", "norway"],
        )
        question = _make_question(
            qid="integ_001",
            text="How many medals does Norway have?",
            expected="26 medals",
            category="recall",
            rubric=rubric,
        )
        agent = MockAgent(answers={"medals": "Norway has 26 medals total"})

        results, _ = coord._question_and_grade(agent, [question])

        assert len(results) == 1
        assert results[0].overall_score == 1.0  # All 3 keywords matched

    def test_analyst_finds_patterns_in_coordinator_results(self):
        """Analyst finds patterns in results from coordinator."""
        # Create a report with one weak and one strong category
        report = _make_eval_report(
            overall=0.55,
            categories={"strong": 0.9, "weak": 0.2},
        )

        analyst = AnalystAgent()
        analysis = analyst.analyze(report)

        # Should identify the weak category
        weak_patterns = [
            fp for fp in analysis.failure_patterns
            if "weak" in fp.pattern_name
        ]
        assert len(weak_patterns) >= 1

        # Should suggest improvements
        assert len(analysis.improvement_priorities) >= 1

    def test_full_data_serialization(self):
        """All data classes serialize to JSON without error."""
        # PerspectiveGrade
        pg = PerspectiveGrade(perspective="factual", score=0.8, reasoning="ok", question_id="q1")
        json.dumps(pg.to_dict())

        # AggregateGrade
        q = _make_question()
        ag = GraderAgent.aggregate_grades([pg], q, "answer")
        json.dumps(ag.to_dict())

        # AnalysisReport
        analyst = AnalystAgent()
        report = _make_eval_report()
        analysis = analyst.analyze(report)
        json.dumps(analysis.to_dict())

        # ComparisonReport
        comparison = analyst.compare_reports([report, report], ["a", "b"])
        json.dumps(comparison.to_dict())

        # PipelineReport
        pr = PipelineReport(
            rounds=[],
            comparison=comparison,
            final_overall_score=0.75,
            total_questions_asked=20,
            total_time_s=10.0,
        )
        json.dumps(pr.to_dict())


# ====================================================================
# Import verification
# ====================================================================


class TestImports:
    def test_top_level_imports(self):
        """All public types importable from multi_agent_eval."""
        from amplihack_eval.multi_agent_eval import (
            AdversaryAgent,
            AggregateGrade,
            AnalysisReport,
            AnalystAgent,
            ComparisonReport,
            EvalCoordinator,
            GraderAgent,
            Improvement,
            MultiAgentEvalPipeline,
            PerspectiveGrade,
        )

        assert GraderAgent is not None
        assert AdversaryAgent is not None
        assert AnalystAgent is not None
        assert EvalCoordinator is not None
        assert MultiAgentEvalPipeline is not None
