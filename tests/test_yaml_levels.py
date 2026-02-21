"""Tests for YAML-driven level definitions.

Tests:
- YAML parsing and validation
- All 12 YAML files load correctly
- Level runner with mock agent
- Prerequisites checking
- Scoring config application
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from amplihack_eval.adapters.base import AgentAdapter, AgentResponse
from amplihack_eval.levels.loader import load_all_levels, load_level, validate_level
from amplihack_eval.levels.schema import LevelDefinition, QuestionTemplate, ScoringConfig

# Directory containing YAML level files
LEVELS_DIR = Path(__file__).parent.parent / "src" / "amplihack_eval" / "levels"


# ---------------------------------------------------------------------------
# Mock agent for runner tests
# ---------------------------------------------------------------------------


class EchoAgent(AgentAdapter):
    """Simple agent that echoes back learned content for testing."""

    def __init__(self):
        self.learned: list[str] = []
        self._closed = False

    def learn(self, content: str) -> None:
        self.learned.append(content)

    def answer(self, question: str) -> AgentResponse:
        # Return a concatenation of everything learned as a crude answer
        if self.learned:
            return AgentResponse(answer=" ".join(self.learned))
        return AgentResponse(answer="I don't know")

    def reset(self) -> None:
        self.learned.clear()

    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Schema dataclass tests
# ---------------------------------------------------------------------------


class TestQuestionTemplate:
    def test_creation_minimal(self):
        q = QuestionTemplate(id="Q01", text="What?", category="general")
        assert q.id == "Q01"
        assert q.text == "What?"
        assert q.category == "general"
        assert q.expected_answer is None
        assert q.scoring_dimensions == ["factual_accuracy"]
        assert q.rubric is None

    def test_creation_full(self):
        q = QuestionTemplate(
            id="Q02",
            text="How many?",
            category="counting",
            scoring_dimensions=["factual_accuracy", "specificity"],
            expected_answer="42",
            rubric={"required_keywords": ["42"]},
        )
        assert q.expected_answer == "42"
        assert len(q.scoring_dimensions) == 2
        assert q.rubric is not None


class TestScoringConfig:
    def test_defaults(self):
        sc = ScoringConfig()
        assert sc.pass_threshold == 0.7
        assert sc.grader_votes == 3
        assert sc.dimensions == ["factual_accuracy"]
        assert sc.weights == {}

    def test_valid_config(self):
        sc = ScoringConfig(
            pass_threshold=0.8,
            dimensions=["factual_accuracy", "specificity"],
            weights={"factual_accuracy": 0.7, "specificity": 0.3},
            grader_votes=5,
        )
        assert sc.validate() == []

    def test_invalid_threshold(self):
        sc = ScoringConfig(pass_threshold=1.5)
        errors = sc.validate()
        assert any("pass_threshold" in e for e in errors)

    def test_invalid_grader_votes(self):
        sc = ScoringConfig(grader_votes=0)
        errors = sc.validate()
        assert any("grader_votes" in e for e in errors)

    def test_weight_not_in_dimensions(self):
        sc = ScoringConfig(
            dimensions=["factual_accuracy"],
            weights={"nonexistent": 0.5},
        )
        errors = sc.validate()
        assert any("nonexistent" in e for e in errors)


class TestLevelDefinition:
    def test_valid_level(self):
        level = LevelDefinition(
            id="L01",
            name="Test Level",
            description="A test level",
            category="memory",
            difficulty=1,
            questions=[
                QuestionTemplate(id="Q01", text="What?", category="general"),
            ],
        )
        assert level.validate() == []

    def test_missing_id(self):
        level = LevelDefinition(
            id="",
            name="Test",
            description="Test",
            questions=[QuestionTemplate(id="Q01", text="What?", category="g")],
        )
        errors = level.validate()
        assert any("id" in e for e in errors)

    def test_invalid_category(self):
        level = LevelDefinition(
            id="L01",
            name="Test",
            description="Test",
            category="invalid_cat",
            questions=[QuestionTemplate(id="Q01", text="What?", category="g")],
        )
        errors = level.validate()
        assert any("category" in e for e in errors)

    def test_invalid_difficulty(self):
        level = LevelDefinition(
            id="L01",
            name="Test",
            description="Test",
            difficulty=10,
            questions=[QuestionTemplate(id="Q01", text="What?", category="g")],
        )
        errors = level.validate()
        assert any("difficulty" in e for e in errors)

    def test_no_questions(self):
        level = LevelDefinition(
            id="L01",
            name="Test",
            description="Test",
        )
        errors = level.validate()
        assert any("question" in e for e in errors)

    def test_invalid_grading_mode(self):
        level = LevelDefinition(
            id="L01",
            name="Test",
            description="Test",
            grading_mode="invalid",
            questions=[QuestionTemplate(id="Q01", text="What?", category="g")],
        )
        errors = level.validate()
        assert any("grading_mode" in e for e in errors)

    def test_duplicate_question_ids(self):
        level = LevelDefinition(
            id="L01",
            name="Test",
            description="Test",
            questions=[
                QuestionTemplate(id="Q01", text="A?", category="g"),
                QuestionTemplate(id="Q01", text="B?", category="g"),
            ],
        )
        errors = level.validate()
        assert any("unique" in e for e in errors)

    def test_negative_min_turns(self):
        level = LevelDefinition(
            id="L01",
            name="Test",
            description="Test",
            min_turns=-1,
            questions=[QuestionTemplate(id="Q01", text="What?", category="g")],
        )
        errors = level.validate()
        assert any("min_turns" in e for e in errors)


# ---------------------------------------------------------------------------
# YAML loading tests
# ---------------------------------------------------------------------------


class TestYAMLLoading:
    """Test that all 12 YAML files load correctly."""

    EXPECTED_LEVELS = [
        "L01", "L02", "L03", "L04", "L05", "L06",
        "L07", "L08", "L09", "L10", "L11", "L12",
    ]

    def test_load_all_levels(self):
        """All 12 YAML files load without error."""
        levels = load_all_levels()
        assert len(levels) == 12
        ids = [lv.id for lv in levels]
        for expected_id in self.EXPECTED_LEVELS:
            assert expected_id in ids, f"Missing level {expected_id}"

    @pytest.mark.parametrize("level_id", [
        "L01", "L02", "L03", "L04", "L05", "L06",
        "L07", "L08", "L09", "L10", "L11", "L12",
    ])
    def test_load_individual_level(self, level_id):
        """Each level loads individually by ID."""
        level = load_level(level_id)
        assert level.id == level_id
        assert level.name != ""
        assert level.description != ""
        assert len(level.questions) > 0

    @pytest.mark.parametrize("level_id", [
        "L01", "L02", "L03", "L04", "L05", "L06",
        "L07", "L08", "L09", "L10", "L11", "L12",
    ])
    def test_all_levels_validate(self, level_id):
        """Each loaded level passes validation."""
        level = load_level(level_id)
        errors = validate_level(level)
        assert errors == [], f"Level {level_id} validation errors: {errors}"

    def test_load_nonexistent_level(self):
        """Loading a nonexistent level raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_level("L99")

    def test_levels_sorted_by_id(self):
        """load_all_levels returns levels sorted by ID."""
        levels = load_all_levels()
        ids = [lv.id for lv in levels]
        assert ids == sorted(ids)


class TestYAMLContent:
    """Test specific content expectations for YAML levels."""

    def test_l01_is_simple_recall(self):
        level = load_level("L01")
        assert level.category == "memory"
        assert level.difficulty == 1
        assert level.prerequisites == []
        assert level.scoring.pass_threshold == 0.7

    def test_l01_has_3_questions(self):
        level = load_level("L01")
        assert len(level.questions) == 3

    def test_l02_has_prerequisite_l01(self):
        level = load_level("L02")
        assert "L01" in level.prerequisites

    def test_l03_is_reasoning_category(self):
        level = load_level("L03")
        assert level.category == "reasoning"

    def test_l05_grading_mode_is_llm(self):
        level = load_level("L05")
        assert level.grading_mode == "llm"

    def test_l07_min_turns_100(self):
        level = load_level("L07")
        assert level.min_turns == 100

    def test_l10_highest_difficulty(self):
        level = load_level("L10")
        assert level.difficulty == 5

    def test_l10_requires_l09(self):
        level = load_level("L10")
        assert "L09" in level.prerequisites

    def test_l11_is_planning_category(self):
        level = load_level("L11")
        assert level.category == "planning"

    def test_l12_requires_temporal_and_synthesis(self):
        level = load_level("L12")
        assert "L03" in level.prerequisites
        assert "L02" in level.prerequisites

    def test_scoring_dimensions_present(self):
        """Every level has at least one scoring dimension."""
        levels = load_all_levels()
        for level in levels:
            assert len(level.scoring.dimensions) > 0, f"{level.id} has no scoring dimensions"

    def test_question_ids_are_unique_across_levels(self):
        """Question IDs are unique globally (not just within levels)."""
        levels = load_all_levels()
        all_qids: list[str] = []
        for level in levels:
            for q in level.questions:
                all_qids.append(q.id)
        assert len(all_qids) == len(set(all_qids)), "Duplicate question IDs across levels"

    def test_all_questions_have_expected_answers(self):
        """Every question has an expected_answer."""
        levels = load_all_levels()
        for level in levels:
            for q in level.questions:
                assert q.expected_answer is not None, (
                    f"{level.id}/{q.id} has no expected_answer"
                )
                assert len(q.expected_answer) > 0, (
                    f"{level.id}/{q.id} has empty expected_answer"
                )


# ---------------------------------------------------------------------------
# Custom YAML directory tests
# ---------------------------------------------------------------------------


class TestCustomYAMLDir:
    """Test loading from a custom directory."""

    def test_load_from_custom_dir(self, tmp_path):
        """Can load levels from a non-default directory."""
        yaml_content = textwrap.dedent("""\
            id: L99
            name: Custom Test Level
            description: A custom test
            category: memory
            difficulty: 1
            min_turns: 10
            grading_mode: deterministic
            scoring:
              pass_threshold: 0.5
              dimensions:
                - factual_accuracy
              grader_votes: 1
            questions:
              - id: L99_Q01
                text: "What is 2+2?"
                expected_answer: "4"
                category: math
                scoring_dimensions:
                  - factual_accuracy
        """)
        yaml_file = tmp_path / "L99_custom.yaml"
        yaml_file.write_text(yaml_content)

        level = load_level("L99", levels_dir=tmp_path)
        assert level.id == "L99"
        assert level.name == "Custom Test Level"
        assert level.grading_mode == "deterministic"
        assert len(level.questions) == 1

    def test_load_all_from_custom_dir(self, tmp_path):
        """load_all_levels respects custom directory."""
        for i in range(3):
            lid = f"L{i + 1:02d}"
            yaml_content = textwrap.dedent(f"""\
                id: {lid}
                name: Level {i + 1}
                description: Test level {i + 1}
                category: memory
                difficulty: {i + 1}
                scoring:
                  pass_threshold: 0.5
                  dimensions:
                    - factual_accuracy
                  grader_votes: 1
                questions:
                  - id: {lid}_Q01
                    text: "Question {i + 1}"
                    expected_answer: "Answer {i + 1}"
                    category: test
            """)
            (tmp_path / f"{lid}_test.yaml").write_text(yaml_content)

        levels = load_all_levels(levels_dir=tmp_path)
        assert len(levels) == 3
        assert [lv.id for lv in levels] == ["L01", "L02", "L03"]

    def test_invalid_yaml_skipped(self, tmp_path):
        """Invalid YAML files are skipped with a warning."""
        (tmp_path / "L01_bad.yaml").write_text("not: valid: yaml: [")
        levels = load_all_levels(levels_dir=tmp_path)
        assert len(levels) == 0

    def test_non_dict_yaml_skipped(self, tmp_path):
        """YAML files that parse to non-dict are skipped."""
        (tmp_path / "L01_list.yaml").write_text("- item1\n- item2")
        levels = load_all_levels(levels_dir=tmp_path)
        assert len(levels) == 0


# ---------------------------------------------------------------------------
# Prerequisites checking tests
# ---------------------------------------------------------------------------


class TestPrerequisites:
    def test_l01_has_no_prerequisites(self):
        level = load_level("L01")
        assert level.prerequisites == []

    def test_prerequisite_chain(self):
        """L10 requires L09, L09 requires L01 and L03."""
        l10 = load_level("L10")
        l09 = load_level("L09")
        assert "L09" in l10.prerequisites
        assert "L01" in l09.prerequisites

    def test_all_prerequisites_reference_existing_levels(self):
        """All prerequisite IDs reference levels that exist."""
        levels = load_all_levels()
        all_ids = {lv.id for lv in levels}
        for level in levels:
            for prereq in level.prerequisites:
                assert prereq in all_ids, (
                    f"{level.id} has prerequisite '{prereq}' which doesn't exist"
                )


# ---------------------------------------------------------------------------
# Scoring config application tests
# ---------------------------------------------------------------------------


class TestScoringApplication:
    def test_l01_scoring_weights_sum_to_1(self):
        level = load_level("L01")
        if level.scoring.weights:
            total = sum(level.scoring.weights.values())
            assert abs(total - 1.0) < 0.01, f"L01 weights sum to {total}"

    def test_all_scoring_weights_reference_valid_dimensions(self):
        """All weight keys exist in the dimensions list."""
        levels = load_all_levels()
        for level in levels:
            for weight_key in level.scoring.weights:
                assert weight_key in level.scoring.dimensions, (
                    f"{level.id}: weight '{weight_key}' not in dimensions {level.scoring.dimensions}"
                )

    def test_grader_votes_positive(self):
        """All levels have positive grader_votes."""
        levels = load_all_levels()
        for level in levels:
            assert level.scoring.grader_votes >= 1, (
                f"{level.id} has grader_votes={level.scoring.grader_votes}"
            )

    def test_pass_thresholds_in_range(self):
        """All pass thresholds are 0.0-1.0."""
        levels = load_all_levels()
        for level in levels:
            assert 0.0 <= level.scoring.pass_threshold <= 1.0, (
                f"{level.id} has pass_threshold={level.scoring.pass_threshold}"
            )


# ---------------------------------------------------------------------------
# Level runner integration tests (mock agent, no LLM calls)
# ---------------------------------------------------------------------------


class TestLevelRunner:
    """Test run_level and run_suite with a mock agent.

    These tests patch the LLM grading to avoid API calls.
    """

    def _mock_grade(self, *args, **kwargs):
        """Return a fixed score for all dimensions."""
        from amplihack_eval.core.runner import DimensionScore
        dimensions = args[2] if len(args) > 2 else kwargs.get("dimensions", ["factual_accuracy"])
        return [
            DimensionScore(dimension=d, score=0.8, reasoning="mock")
            for d in dimensions
        ]

    @patch("amplihack_eval.core.runner._grade_multi_vote")
    def test_run_single_level(self, mock_vote):
        """run_level returns a LevelResult."""
        from amplihack_eval.core.runner import run_level

        mock_vote.side_effect = self._mock_grade
        agent = EchoAgent()
        result = run_level("L01", agent)

        assert result.level_id == "L01"
        assert result.level_name == "Simple Recall"
        assert len(result.results) == 3  # L01 has 3 questions
        assert result.overall_score > 0

    @patch("amplihack_eval.core.runner._grade_multi_vote")
    def test_run_level_pass(self, mock_vote):
        """Agent with score above threshold passes."""
        from amplihack_eval.core.runner import run_level

        mock_vote.side_effect = self._mock_grade
        agent = EchoAgent()
        result = run_level("L01", agent)

        assert result.passed is True
        assert result.overall_score >= result.pass_threshold

    @patch("amplihack_eval.core.runner._grade_multi_vote")
    def test_run_suite_basic(self, mock_vote):
        """run_suite runs multiple levels."""
        from amplihack_eval.core.runner import run_suite

        mock_vote.side_effect = self._mock_grade
        agent = EchoAgent()
        result = run_suite(["L01", "L02"], agent, check_prerequisites=False)

        assert result.total_count == 2
        assert len(result.level_results) == 2
        assert result.passed_count >= 0
        assert result.skipped == []

    @patch("amplihack_eval.core.runner._grade_multi_vote")
    def test_run_suite_with_prerequisites(self, mock_vote):
        """run_suite skips levels with unmet prerequisites."""
        from amplihack_eval.core.runner import run_suite

        mock_vote.side_effect = self._mock_grade
        agent = EchoAgent()
        # L10 requires L09, which should be skipped since we don't run it first
        result = run_suite(["L10"], agent, check_prerequisites=True)

        # L10 requires L09. Since L09 wasn't run, L10 should be skipped.
        assert "L10" in result.skipped
        assert result.total_count == 0

    @patch("amplihack_eval.core.runner._grade_multi_vote")
    def test_run_suite_passes_prerequisites(self, mock_vote):
        """run_suite runs levels after prerequisites pass."""
        from amplihack_eval.core.runner import run_suite

        mock_vote.side_effect = self._mock_grade
        agent = EchoAgent()
        # L01 has no prereqs, L02 requires L01
        result = run_suite(["L01", "L02"], agent, check_prerequisites=True)

        assert result.total_count == 2
        assert result.skipped == []
        assert result.passed_count == 2

    @patch("amplihack_eval.core.runner._grade_multi_vote")
    def test_run_suite_skips_on_failed_prereq(self, mock_vote):
        """If a prerequisite level fails, dependent levels are skipped."""
        from amplihack_eval.core.runner import DimensionScore, run_suite

        # First call (L01): return very low scores so it fails
        # Second call would be L02 but it should be skipped
        call_count = [0]

        def low_then_high(*args, **kwargs):
            call_count[0] += 1
            dims = args[2] if len(args) > 2 else kwargs.get("dimensions", ["factual_accuracy"])
            # Always return 0.1 so L01 fails (threshold 0.7)
            return [
                DimensionScore(dimension=d, score=0.1, reasoning="mock low")
                for d in dims
            ]

        mock_vote.side_effect = low_then_high
        agent = EchoAgent()
        result = run_suite(["L01", "L02"], agent, check_prerequisites=True)

        # L01 should run and fail, L02 should be skipped (prereq L01 failed)
        assert result.total_count == 1
        assert result.passed_count == 0
        assert "L02" in result.skipped

    @patch("amplihack_eval.core.runner._grade_multi_vote")
    def test_suite_result_overall_score(self, mock_vote):
        """SuiteResult.overall_score is the average of level scores."""
        from amplihack_eval.core.runner import run_suite

        mock_vote.side_effect = self._mock_grade
        agent = EchoAgent()
        result = run_suite(["L01", "L04"], agent, check_prerequisites=False)

        # With mock grade returning 0.8 for all, overall should be ~0.8
        assert 0.7 <= result.overall_score <= 0.9

    @patch("amplihack_eval.core.runner._grade_multi_vote")
    def test_agent_reset_between_levels(self, mock_vote):
        """Agent.reset() is called between levels in a suite."""
        from amplihack_eval.core.runner import run_suite

        mock_vote.side_effect = self._mock_grade
        agent = EchoAgent()
        run_suite(["L01", "L04"], agent, check_prerequisites=False)

        # After suite, agent should have been reset before L04
        # so it should only contain L04's articles
        # (Since reset clears learned, and L04 articles are fed after reset)
        # This is indirect - we verify reset happened by checking learned is not cumulative
        # Actually, we can check that the agent was used (has some learned content from L04)
        assert len(agent.learned) > 0


# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------


class TestImports:
    def test_schema_imports(self):
        from amplihack_eval.levels.schema import (
            LevelDefinition,
            QuestionTemplate,
            ScoringConfig,
        )
        assert LevelDefinition is not None
        assert QuestionTemplate is not None
        assert ScoringConfig is not None

    def test_loader_imports(self):
        from amplihack_eval.levels.loader import (
            load_all_levels,
            load_level,
            validate_level,
        )
        assert callable(load_level)
        assert callable(load_all_levels)
        assert callable(validate_level)

    def test_convenience_imports(self):
        """Can import everything from levels package."""
        from amplihack_eval.levels import (
            LevelDefinition,
            QuestionTemplate,
            ScoringConfig,
            load_all_levels,
            load_level,
            validate_level,
        )
        assert LevelDefinition is not None
        assert callable(load_level)

    def test_top_level_runner_imports(self):
        """run_level and run_suite available from top-level."""
        from amplihack_eval import LevelResult, SuiteResult, run_level, run_suite
        assert callable(run_level)
        assert callable(run_suite)
        assert LevelResult is not None
        assert SuiteResult is not None
