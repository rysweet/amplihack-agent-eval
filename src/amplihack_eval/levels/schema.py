"""YAML-driven level definition schema.

Follows the EleutherAI lm-evaluation-harness pattern: each eval level
is defined in a standalone YAML file with questions, scoring config,
and metadata.  The Python dataclasses here mirror the YAML structure
so a round-trip parse-validate-use cycle is straightforward.

Public API:
    LevelDefinition: Complete level loaded from YAML
    QuestionTemplate: Single question within a level
    ScoringConfig: Scoring rules (threshold, dimensions, weights, votes)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuestionTemplate:
    """A single evaluation question within a level.

    Attributes:
        id: Unique question identifier (e.g. "L01_Q01")
        text: Question text, may contain {placeholders} for dynamic data
        expected_answer: Ground-truth answer, or None for dynamic answers
        category: Question category (e.g. "needle_in_haystack", "temporal_difference")
        scoring_dimensions: Which dimensions to grade on
        rubric: Optional grading rubric with keywords / paraphrases
    """

    id: str
    text: str
    category: str
    scoring_dimensions: list[str] = field(default_factory=lambda: ["factual_accuracy"])
    expected_answer: str | None = None
    rubric: dict | None = None


@dataclass
class ScoringConfig:
    """Scoring rules for a level.

    Attributes:
        pass_threshold: Minimum score to pass (0.0-1.0, default 0.7)
        dimensions: Scoring dimensions to evaluate
        weights: Dimension name -> weight mapping (must sum to 1.0 for weighted scoring)
        grader_votes: Number of grading votes (1 = deterministic, 3+ = LLM multi-vote)
    """

    pass_threshold: float = 0.7
    dimensions: list[str] = field(default_factory=lambda: ["factual_accuracy"])
    weights: dict[str, float] = field(default_factory=dict)
    grader_votes: int = 3

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []
        if not 0.0 <= self.pass_threshold <= 1.0:
            errors.append(f"pass_threshold must be 0.0-1.0, got {self.pass_threshold}")
        if self.grader_votes < 1:
            errors.append(f"grader_votes must be >= 1, got {self.grader_votes}")
        if self.weights:
            for dim in self.weights:
                if dim not in self.dimensions:
                    errors.append(f"weight key '{dim}' not in dimensions {self.dimensions}")
        return errors


@dataclass
class LevelDefinition:
    """Complete evaluation level loaded from a YAML file.

    Attributes:
        id: Short identifier (e.g. "L01")
        name: Human-readable name (e.g. "Simple Recall")
        description: What this level tests
        category: Top-level category ("memory", "tool_use", "planning", "reasoning")
        difficulty: 1-5 difficulty scale
        questions: List of question templates
        scoring: Scoring configuration
        prerequisites: Level IDs that must pass before this level
        data_source: Which data generator to use (e.g. "progressive_levels")
        min_turns: Minimum dialogue turns needed for this level
        grading_mode: "deterministic", "llm", or "hybrid"
    """

    id: str
    name: str
    description: str
    category: str = "memory"
    difficulty: int = 1
    questions: list[QuestionTemplate] = field(default_factory=list)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    prerequisites: list[str] = field(default_factory=list)
    data_source: str = "progressive_levels"
    min_turns: int = 50
    grading_mode: str = "hybrid"

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []
        if not self.id:
            errors.append("id is required")
        if not self.name:
            errors.append("name is required")
        if not self.description:
            errors.append("description is required")
        if self.category not in ("memory", "tool_use", "planning", "reasoning"):
            errors.append(
                f"category must be one of memory/tool_use/planning/reasoning, got '{self.category}'"
            )
        if not 1 <= self.difficulty <= 5:
            errors.append(f"difficulty must be 1-5, got {self.difficulty}")
        if not self.questions:
            errors.append("at least one question is required")
        if self.grading_mode not in ("deterministic", "llm", "hybrid"):
            errors.append(
                f"grading_mode must be deterministic/llm/hybrid, got '{self.grading_mode}'"
            )
        if self.min_turns < 0:
            errors.append(f"min_turns must be >= 0, got {self.min_turns}")
        errors.extend(self.scoring.validate())
        # Validate question IDs are unique
        qids = [q.id for q in self.questions]
        if len(qids) != len(set(qids)):
            errors.append("question IDs must be unique within a level")
        return errors


__all__ = [
    "LevelDefinition",
    "QuestionTemplate",
    "ScoringConfig",
]
