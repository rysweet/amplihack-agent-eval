"""Tests that the data generator works in the new package."""

from __future__ import annotations

from amplihack_eval.data.long_horizon import (
    GradingRubric,
    GroundTruth,
    Question,
    Turn,
    generate_dialogue,
    generate_questions,
)
from amplihack_eval.data.progressive_levels import (
    ADVANCED_LEVELS,
    ALL_LEVELS,
    NOVEL_SKILL_LEVELS,
    TEACHER_STUDENT_LEVELS,
    TRANSFER_LEVELS,
    TestLevel,
    get_level_by_id,
)

# --- Turn / Question / GroundTruth dataclass tests ---


class TestDataClasses:
    def test_turn_creation(self):
        turn = Turn(
            turn_number=1,
            content="Hello world",
            block=1,
            block_name="people",
            facts=[{"name": "Alice", "age": "30"}],
        )
        assert turn.turn_number == 1
        assert turn.block == 1
        assert len(turn.facts) == 1

    def test_question_creation(self):
        q = Question(
            question_id="q_001",
            text="What is Alice's age?",
            expected_answer="30",
            category="needle_in_haystack",
            relevant_turns=[1],
            scoring_dimensions=["factual_accuracy"],
        )
        assert q.question_id == "q_001"
        assert q.chain_length == 1  # default

    def test_grading_rubric_defaults(self):
        rubric = GradingRubric()
        assert rubric.required_keywords == []
        assert rubric.acceptable_paraphrases == []
        assert rubric.incorrect_patterns == []
        assert rubric.dimension_weights == {}

    def test_ground_truth_creation(self):
        gt = GroundTruth(turns=[])
        assert gt.turns == []
        assert gt.facts_by_entity == {}
        assert gt.current_values == {}
        assert gt.superseded_values == {}


# --- Dialogue generation tests ---


class TestDialogueGeneration:
    def test_generate_small_dialogue(self):
        """Generate a small dialogue and verify structure."""
        gt = generate_dialogue(num_turns=20, seed=42)
        assert isinstance(gt, GroundTruth)
        assert len(gt.turns) == 20

    def test_reproducible_with_same_seed(self):
        """Same seed produces identical output."""
        gt1 = generate_dialogue(num_turns=10, seed=123)
        gt2 = generate_dialogue(num_turns=10, seed=123)
        assert len(gt1.turns) == len(gt2.turns)
        for t1, t2 in zip(gt1.turns, gt2.turns):
            assert t1.content == t2.content
            assert t1.block == t2.block

    def test_different_seeds_produce_different_output(self):
        """Different seeds produce different dialogue."""
        gt1 = generate_dialogue(num_turns=10, seed=42)
        gt2 = generate_dialogue(num_turns=10, seed=999)
        # At least some turns should differ
        differences = sum(1 for t1, t2 in zip(gt1.turns, gt2.turns) if t1.content != t2.content)
        assert differences > 0

    def test_turns_have_content(self):
        """All turns have non-empty content."""
        gt = generate_dialogue(num_turns=50, seed=42)
        for turn in gt.turns:
            assert turn.content is not None
            assert len(turn.content.strip()) > 0

    def test_turns_have_block_info(self):
        """All turns have block number and name."""
        gt = generate_dialogue(num_turns=30, seed=42)
        for turn in gt.turns:
            assert 1 <= turn.block <= 12
            assert turn.block_name != ""

    def test_facts_tracked(self):
        """Some turns contain tracked facts."""
        gt = generate_dialogue(num_turns=100, seed=42)
        total_facts = sum(len(t.facts) for t in gt.turns)
        assert total_facts > 0, "Expected some facts to be tracked"


# --- Question generation tests ---


class TestQuestionGeneration:
    def test_generate_questions(self):
        """Generate questions from a dialogue."""
        gt = generate_dialogue(num_turns=50, seed=42)
        questions = generate_questions(gt, num_questions=10)
        assert len(questions) > 0
        assert len(questions) <= 10

    def test_questions_have_expected_fields(self):
        """Questions have all required fields."""
        gt = generate_dialogue(num_turns=50, seed=42)
        questions = generate_questions(gt, num_questions=5)
        for q in questions:
            assert q.question_id != ""
            assert q.text != ""
            assert q.expected_answer != ""
            assert q.category != ""
            assert len(q.scoring_dimensions) > 0

    def test_questions_reference_valid_turns(self):
        """Question relevant_turns reference valid turn numbers."""
        gt = generate_dialogue(num_turns=30, seed=42)
        questions = generate_questions(gt, num_questions=5)
        max_turn = max(t.turn_number for t in gt.turns)
        for q in questions:
            for t in q.relevant_turns:
                assert 0 <= t <= max_turn


# --- Progressive levels tests ---


class TestProgressiveLevels:
    def test_all_levels_defined(self):
        """L1-L6 are in ALL_LEVELS."""
        assert len(ALL_LEVELS) == 6

    def test_teacher_student_levels(self):
        assert len(TEACHER_STUDENT_LEVELS) == 1
        assert TEACHER_STUDENT_LEVELS[0].level_id == "L7"

    def test_advanced_levels(self):
        assert len(ADVANCED_LEVELS) == 3
        ids = [level.level_id for level in ADVANCED_LEVELS]
        assert "L8" in ids
        assert "L9" in ids
        assert "L10" in ids

    def test_novel_skill_levels(self):
        assert len(NOVEL_SKILL_LEVELS) == 1
        assert NOVEL_SKILL_LEVELS[0].level_id == "L11"

    def test_transfer_levels(self):
        assert len(TRANSFER_LEVELS) == 1
        assert TRANSFER_LEVELS[0].level_id == "L12"

    def test_level_structure(self):
        """Each level has articles and questions."""
        all_lvls = ALL_LEVELS + TEACHER_STUDENT_LEVELS + ADVANCED_LEVELS
        all_lvls += NOVEL_SKILL_LEVELS + TRANSFER_LEVELS
        for level in all_lvls:
            assert isinstance(level, TestLevel)
            assert level.level_id.startswith("L")
            assert len(level.articles) > 0
            assert len(level.questions) > 0

    def test_get_level_by_id(self):
        """get_level_by_id returns correct level."""
        level = get_level_by_id("L1")
        assert level is not None
        assert level.level_id == "L1"
        assert level.level_name == "Single Source Direct Recall"

    def test_get_level_by_id_not_found(self):
        """get_level_by_id returns None for unknown ID."""
        assert get_level_by_id("L99") is None

    def test_articles_have_content(self):
        """All articles have title, content, url, published."""
        for level in ALL_LEVELS:
            for article in level.articles:
                assert article.title != ""
                assert len(article.content) > 10
                assert article.url != ""
                assert article.published != ""

    def test_questions_have_fields(self):
        """All questions have required fields."""
        for level in ALL_LEVELS:
            for q in level.questions:
                assert q.question != ""
                assert q.expected_answer != ""
                assert q.level == level.level_id
                assert q.reasoning_type != ""


# --- Import verification ---


class TestImports:
    def test_top_level_import(self):
        """Package-level imports work."""
        from amplihack_eval import AgentAdapter, AgentResponse, EvalRunner, ToolCall

        assert AgentAdapter is not None
        assert AgentResponse is not None
        assert EvalRunner is not None
        assert ToolCall is not None

    def test_data_module_import(self):
        """Data module imports work."""
        from amplihack_eval.data import generate_dialogue, generate_questions

        assert callable(generate_dialogue)
        assert callable(generate_questions)

    def test_levels_module_import(self):
        """Levels convenience module works."""
        from amplihack_eval.levels import ALL_LEVELS, get_level_by_id

        assert len(ALL_LEVELS) == 6
        assert callable(get_level_by_id)
