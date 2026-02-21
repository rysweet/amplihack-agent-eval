"""Adversary agent that generates hard questions targeting agent weaknesses.

The adversary observes what the agent gets right and generates harder
variants designed to expose failure modes. Strategies include:

1. Target strong categories with subtle twists
2. Near-miss distractors based on actual wrong answers
3. Temporal traps (asking about superseded facts)
4. Hallucination triggers (plausible but nonexistent facts)
5. Forgetting probes (selective memory testing)

Philosophy:
- Adversarial testing reveals weaknesses that standard questions miss
- Questions generated from actual agent behavior are more targeted
- All questions come with ground truth for fair grading
- JSON-serializable for logging and reproducibility

Public API:
    AdversaryAgent: Generates adversarial questions from eval results
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..data.long_horizon import GroundTruth, Question

logger = logging.getLogger(__name__)


def _extract_json_list(text: str) -> list[dict]:
    """Extract a JSON array from LLM response text."""
    stripped = text.strip()

    # Try direct parse
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "questions" in result:
            return result["questions"]
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fenced:
        try:
            result = json.loads(fenced.group(1).strip())
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "questions" in result:
                return result["questions"]
        except json.JSONDecodeError:
            pass

    # Find first [ ... ] block
    bracket_match = re.search(r"\[.*\]", stripped, re.DOTALL)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except json.JSONDecodeError:
            pass

    return []


@dataclass
class AdversarialResult:
    """Container for adversarial question generation results."""

    questions: list[Question]
    strategy_used: str
    targeting_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_questions": len(self.questions),
            "strategy_used": self.strategy_used,
            "targeting_info": self.targeting_info,
            "questions": [
                {
                    "question_id": q.question_id,
                    "text": q.text,
                    "expected_answer": q.expected_answer,
                    "category": q.category,
                }
                for q in self.questions
            ],
        }


class AdversaryAgent:
    """Generates hard questions targeting agent weaknesses.

    Observes eval results and generates adversarial questions that are
    specifically designed to challenge the agent's capabilities in areas
    where it appears strong (to find hidden weaknesses) and in areas
    where it appears weak (to confirm and map failure boundaries).

    Args:
        model: LLM model identifier (default: from GRADER_MODEL env var)

    Example::

        adversary = AdversaryAgent()
        questions = adversary.generate_adversarial_questions(
            ground_truth=gt,
            previous_results=results,
            num_questions=10,
        )
    """

    def __init__(self, model: str = ""):
        self.model = model or os.environ.get("GRADER_MODEL", "claude-sonnet-4-5-20250929")

    def generate_adversarial_questions(
        self,
        ground_truth: GroundTruth,
        previous_results: list[dict[str, Any]],
        num_questions: int = 10,
    ) -> list[Question]:
        """Generate adversarial questions based on previous eval results.

        Strategy:
        1. Find categories where the agent scored highest
        2. Generate questions in those categories with subtle twists
        3. Add near-miss distractors based on the agent's actual wrong answers
        4. Include temporal traps (asking about superseded facts)
        5. Include hallucination triggers (plausible but nonexistent facts)

        Args:
            ground_truth: The ground truth data used in the original eval
            previous_results: List of dicts with keys: question_text,
                expected_answer, actual_answer, score, category
            num_questions: Number of adversarial questions to generate

        Returns:
            List of Question objects with adversarial questions
        """
        if not previous_results:
            logger.warning("No previous results to base adversarial questions on")
            return []

        # Analyze results to find targeting opportunities
        category_scores: dict[str, list[float]] = {}
        wrong_answers: list[dict[str, Any]] = []

        for result in previous_results:
            cat = result.get("category", "unknown")
            score = result.get("score", 0.0)
            category_scores.setdefault(cat, []).append(score)
            if score < 0.7:
                wrong_answers.append(result)

        # Compute per-category averages
        cat_averages = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
            if scores
        }

        # Build the facts summary from ground truth
        facts_summary = self._summarize_ground_truth(ground_truth)

        # Generate via LLM
        targeting_info = {
            "category_averages": {k: round(v, 3) for k, v in cat_averages.items()},
            "num_wrong_answers": len(wrong_answers),
            "strongest_categories": sorted(
                cat_averages, key=lambda c: cat_averages[c], reverse=True
            )[:3],
            "weakest_categories": sorted(
                cat_averages, key=lambda c: cat_averages[c]
            )[:3],
        }

        questions = self._generate_via_llm(
            facts_summary=facts_summary,
            targeting_info=targeting_info,
            wrong_answers=wrong_answers[:5],  # Sample of wrong answers
            num_questions=num_questions,
        )

        return questions

    def generate_forgetting_probes(
        self,
        ground_truth: GroundTruth,
        num_questions: int = 5,
    ) -> list[Question]:
        """Generate questions testing selective forgetting.

        Targets facts that were delivered early in the dialogue and may
        have been forgotten or overwritten by later information.

        Args:
            ground_truth: Ground truth with all delivered facts
            num_questions: Number of forgetting probe questions

        Returns:
            List of Question objects probing memory retention
        """
        if not ground_truth.turns:
            return []

        # Collect facts from early turns (first 20%)
        total_turns = len(ground_truth.turns)
        early_cutoff = max(1, total_turns // 5)
        early_facts: list[dict[str, Any]] = []
        for turn in ground_truth.turns[:early_cutoff]:
            for fact in turn.facts:
                early_facts.append({
                    "turn": turn.turn_number,
                    "block": turn.block_name,
                    "fact": fact,
                })

        if not early_facts:
            return []

        # Check for superseded values
        superseded_facts: list[dict[str, Any]] = []
        if hasattr(ground_truth, "superseded_values") and ground_truth.superseded_values:
            for entity, old_vals in ground_truth.superseded_values.items():
                for key, values in old_vals.items() if isinstance(old_vals, dict) else []:
                    superseded_facts.append({
                        "entity": entity,
                        "key": key,
                        "old_values": values,
                    })

        return self._generate_forgetting_via_llm(
            early_facts=early_facts[:20],
            superseded_facts=superseded_facts[:10],
            num_questions=num_questions,
        )

    def _summarize_ground_truth(self, ground_truth: GroundTruth) -> str:
        """Build a compact summary of facts from ground truth."""
        lines: list[str] = []

        # Collect entities and current values
        if hasattr(ground_truth, "current_values") and ground_truth.current_values:
            for entity, values in list(ground_truth.current_values.items())[:20]:
                if isinstance(values, dict):
                    vals_str = ", ".join(f"{k}={v}" for k, v in list(values.items())[:5])
                    lines.append(f"- {entity}: {vals_str}")

        # If no current_values, sample from turns
        if not lines:
            for turn in ground_truth.turns[:30]:
                if turn.facts:
                    for fact in turn.facts[:2]:
                        lines.append(f"- Turn {turn.turn_number} ({turn.block_name}): {fact}")

        return "\n".join(lines[:30])

    def _generate_via_llm(
        self,
        facts_summary: str,
        targeting_info: dict[str, Any],
        wrong_answers: list[dict[str, Any]],
        num_questions: int,
    ) -> list[Question]:
        """Use LLM to generate adversarial questions."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("No ANTHROPIC_API_KEY; returning empty adversarial questions")
            return []

        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(api_key=api_key)

        wrong_answers_text = ""
        if wrong_answers:
            wrong_answers_text = "\nAgent's wrong answers (for near-miss distractors):\n"
            for wa in wrong_answers:
                wrong_answers_text += (
                    f"  Q: {wa.get('question_text', '')[:80]}\n"
                    f"  Expected: {wa.get('expected_answer', '')[:80]}\n"
                    f"  Got: {wa.get('actual_answer', '')[:80]}\n\n"
                )

        prompt = f"""Generate {num_questions} adversarial quiz questions to stress-test an AI agent's memory.

KNOWN FACTS (from the agent's training dialogue):
{facts_summary}

AGENT PERFORMANCE:
- Strongest categories: {targeting_info.get('strongest_categories', [])}
- Weakest categories: {targeting_info.get('weakest_categories', [])}
- Category averages: {targeting_info.get('category_averages', {{}})}
{wrong_answers_text}

Generate questions using these adversarial strategies:
1. SUBTLE TWIST: Take a fact the agent knows and add a subtle twist (e.g., ask about a detail adjacent to what was stated)
2. NEAR-MISS: Create questions where the correct answer is very close to a plausible wrong answer
3. TEMPORAL TRAP: Ask about facts that may have been updated (old vs new values)
4. HALLUCINATION TRIGGER: Ask about plausible-sounding but nonexistent facts
5. CROSS-REFERENCE: Require combining facts from different blocks/entities

Return a JSON array of objects with these fields:
- "text": the question text
- "expected_answer": the correct answer
- "category": one of "adversarial_twist", "adversarial_near_miss", "adversarial_temporal", "adversarial_hallucination", "adversarial_cross_ref"
- "strategy": which strategy was used (1-5)

Return ONLY the JSON array, no other text."""

        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = message.content[0].text
            items = _extract_json_list(raw)

            questions: list[Question] = []
            for i, item in enumerate(items[:num_questions]):
                q = Question(
                    question_id=f"adv_{i:03d}",
                    text=item.get("text", ""),
                    expected_answer=item.get("expected_answer", ""),
                    category=item.get("category", "adversarial"),
                    relevant_turns=[],
                    scoring_dimensions=["factual_accuracy"],
                )
                if q.text and q.expected_answer:
                    questions.append(q)

            logger.info("Generated %d adversarial questions", len(questions))
            return questions

        except Exception as e:
            logger.warning("Adversarial question generation failed: %s", e)
            return []

    def _generate_forgetting_via_llm(
        self,
        early_facts: list[dict[str, Any]],
        superseded_facts: list[dict[str, Any]],
        num_questions: int,
    ) -> list[Question]:
        """Use LLM to generate forgetting probe questions."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("No ANTHROPIC_API_KEY; returning empty forgetting probes")
            return []

        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(api_key=api_key)

        facts_text = "\n".join(
            f"  Turn {f['turn']} ({f['block']}): {f['fact']}" for f in early_facts
        )

        superseded_text = ""
        if superseded_facts:
            superseded_text = "\nSuperseded facts (old values replaced by new):\n"
            for sf in superseded_facts:
                superseded_text += f"  {sf['entity']}.{sf['key']}: old={sf.get('old_values')}\n"

        prompt = f"""Generate {num_questions} "forgetting probe" questions to test whether an AI agent still remembers early facts.

EARLY FACTS (delivered in the first 20% of dialogue):
{facts_text}
{superseded_text}

Generate questions that:
1. Ask about facts delivered EARLY that may have been forgotten
2. If superseded facts exist, ask about the OLD value specifically
3. Test whether the agent confuses early facts with later similar facts

Return a JSON array with: "text", "expected_answer", "category" (always "forgetting_probe").
Return ONLY the JSON array."""

        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = message.content[0].text
            items = _extract_json_list(raw)

            questions: list[Question] = []
            for i, item in enumerate(items[:num_questions]):
                q = Question(
                    question_id=f"forget_{i:03d}",
                    text=item.get("text", ""),
                    expected_answer=item.get("expected_answer", ""),
                    category="forgetting_probe",
                    relevant_turns=[],
                    scoring_dimensions=["factual_accuracy"],
                )
                if q.text and q.expected_answer:
                    questions.append(q)

            logger.info("Generated %d forgetting probes", len(questions))
            return questions

        except Exception as e:
            logger.warning("Forgetting probe generation failed: %s", e)
            return []


__all__ = ["AdversaryAgent", "AdversarialResult"]
