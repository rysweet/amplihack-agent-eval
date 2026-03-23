"""Tests for quality audit fixes: no silent fallbacks, thread safety, named constants.

Covers issues #23-#29 from the quality audit.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from amplihack_eval.adapters.base import AgentAdapter, AgentResponse
from amplihack_eval.adapters.hive_mind_adapter import (
    MAX_SHARED_CONTEXT_FACTS,
    HiveMindGroupAdapter,
)
from amplihack_eval.adapters.http_adapter import HttpAdapter
from amplihack_eval.adapters.subprocess_adapter import SubprocessAdapter
from amplihack_eval.core.grader import MAX_GRADER_VOTES
from amplihack_eval.core.runner import (
    MAX_ANSWER_LENGTH_IN_REPORT,
    MAX_PARALLEL_WORKERS,
    MAX_REASONING_LENGTH_IN_REPORT,
    EvalRunner,
    _extract_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubAgent(AgentAdapter):
    """Minimal agent for testing."""

    def __init__(self, fail_learn: bool = False, fail_answer: bool = False):
        self._fail_learn = fail_learn
        self._fail_answer = fail_answer
        self.learned: list[str] = []

    def learn(self, content: str) -> None:
        if self._fail_learn:
            raise ValueError("learn boom")
        self.learned.append(content)

    def answer(self, question: str) -> AgentResponse:
        if self._fail_answer:
            raise ValueError("answer boom")
        return AgentResponse(answer="stub answer")

    def reset(self) -> None:
        self.learned.clear()

    def close(self) -> None:
        pass


# ===========================================================================
# Issue #23 - LearningAgentAdapter: no silent fallbacks
# ===========================================================================


class TestLearningAgentAdapterNoSilentFallbacks:
    """LearningAgentAdapter must raise, not swallow, exceptions."""

    def test_learn_raises_on_failure(self):
        """learn() must raise RuntimeError when the underlying agent fails."""
        mock_agent = MagicMock()
        mock_agent.learn_from_content.side_effect = ValueError("learn boom")
        mock_agent_cls = MagicMock(return_value=mock_agent)

        with patch.dict(
            "sys.modules",
            {
                "amplihack": MagicMock(),
                "amplihack.agents": MagicMock(),
                "amplihack.agents.goal_seeking": MagicMock(),
                "amplihack.agents.goal_seeking.learning_agent": MagicMock(LearningAgent=mock_agent_cls),
            },
        ):
            from amplihack_eval.adapters.learning_agent import LearningAgentAdapter

            adapter = LearningAgentAdapter(model="test-model", storage_path="/tmp/test")

        with pytest.raises(RuntimeError, match="LearningAgent failed to learn content: learn boom"):
            adapter.learn("content")


# ===========================================================================
# Issue #24 - HttpAdapter: no silent fallbacks
# ===========================================================================


class TestHttpAdapterNoSilentFallbacks:
    """HttpAdapter must raise on network/parse errors."""

    def test_post_raises_on_network_error(self):
        adapter = HttpAdapter(base_url="http://localhost:99999")
        with pytest.raises(ConnectionError, match="HTTP request failed"):
            adapter.learn("test content")

    def test_learn_raises_on_connection_failure(self):
        adapter = HttpAdapter(base_url="http://localhost:99999")
        with pytest.raises(ConnectionError):
            adapter.learn("some content")

    def test_answer_raises_on_connection_failure(self):
        adapter = HttpAdapter(base_url="http://localhost:99999")
        with pytest.raises(ConnectionError):
            adapter.answer("some question")

    def test_reset_raises_on_connection_failure(self):
        adapter = HttpAdapter(base_url="http://localhost:99999")
        with pytest.raises(ConnectionError):
            adapter.reset()


# ===========================================================================
# Issue #24 - SubprocessAdapter: no silent fallbacks
# ===========================================================================


class TestSubprocessAdapterNoSilentFallbacks:
    """SubprocessAdapter must raise on subprocess errors."""

    def test_run_raises_on_command_not_found(self):
        adapter = SubprocessAdapter(command=["nonexistent_command_xyz"])
        with pytest.raises(FileNotFoundError, match="Command not found"):
            adapter.learn("test")

    def test_run_raises_on_timeout(self):
        adapter = SubprocessAdapter(
            command=["python3", "-c", "import time; time.sleep(60)"],
            learn_flag="",
            timeout=0.1,
        )
        with pytest.raises(TimeoutError, match="timed out"):
            adapter._run([], stdin_text="test")

    def test_run_raises_on_nonzero_exit(self):
        adapter = SubprocessAdapter(command=["python3", "-c", "import sys; sys.exit(1)"])
        with pytest.raises(Exception):
            adapter.learn("test")

    def test_answer_raises_on_bad_json_when_json_output(self):
        adapter = SubprocessAdapter(
            command=["echo", "not json"],
            json_output=True,
        )
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            adapter.answer("question")


# ===========================================================================
# Issue #25 - _grade_with_llm: no silent fallback on missing API key
# ===========================================================================


class TestGradeWithLlmNoSilentFallback:
    """_grade_with_llm must raise when API key is missing."""

    def test_missing_api_key_raises(self):
        pytest.importorskip("anthropic")
        from amplihack_eval.core.runner import _grade_with_llm
        from amplihack_eval.data.long_horizon import Question

        q = Question(
            question_id="test",
            text="test?",
            expected_answer="yes",
            category="test",
            relevant_turns=[1],
            scoring_dimensions=["factual_accuracy"],
        )

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(OSError, match="ANTHROPIC_API_KEY"):
                _grade_with_llm(q, "some answer", ["factual_accuracy"])


# ===========================================================================
# Issue #26 - _extract_json: no silent fallback
# ===========================================================================


class TestExtractJsonNoSilentFallback:
    """_extract_json must raise on unparseable input, not return {}."""

    def test_valid_json_parses(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced_json_parses(self):
        result = _extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_embedded_json_parses(self):
        result = _extract_json('Some text {"key": "value"} more text')
        assert result == {"key": "value"}

    def test_garbage_input_raises(self):
        with pytest.raises(json.JSONDecodeError, match="No valid JSON"):
            _extract_json("not json at all")

    def test_empty_input_raises(self):
        with pytest.raises(json.JSONDecodeError, match="No valid JSON"):
            _extract_json("")

    def test_grader_agent_extract_json_raises(self):
        from amplihack_eval.multi_agent_eval.grader_agent import _extract_json as ga_extract

        with pytest.raises(json.JSONDecodeError):
            ga_extract("no json here")

    def test_analyst_agent_extract_json_raises(self):
        from amplihack_eval.multi_agent_eval.analyst_agent import _extract_json as aa_extract

        with pytest.raises(json.JSONDecodeError):
            aa_extract("no json here")


# ===========================================================================
# Issue #27 - Thread safety: _MultiAgentAdapter._turn_idx
# ===========================================================================


class TestMultiAgentAdapterThreadSafety:
    """_MultiAgentAdapter._turn_idx must be protected by a lock."""

    def test_turn_lock_exists(self):
        """The _MultiAgentAdapter class should have a _turn_lock attribute."""
        from amplihack_eval.core.continuous_eval import _MultiAgentAdapter

        adapter = _MultiAgentAdapter(agents=[MagicMock(), MagicMock()], model="test")
        assert hasattr(adapter, "_turn_lock")
        assert isinstance(adapter._turn_lock, type(threading.Lock()))


# ===========================================================================
# Issue #28 - Named constants for hardcoded limits
# ===========================================================================


class TestNamedConstants:
    """Magic numbers should be replaced by named constants."""

    def test_max_shared_context_facts_exists(self):
        assert MAX_SHARED_CONTEXT_FACTS == 50

    def test_max_parallel_workers_exists(self):
        assert MAX_PARALLEL_WORKERS == 20

    def test_max_grader_votes_exists(self):
        assert MAX_GRADER_VOTES == 9

    def test_max_answer_length_in_report_exists(self):
        assert MAX_ANSWER_LENGTH_IN_REPORT == 500

    def test_max_reasoning_length_in_report_exists(self):
        assert MAX_REASONING_LENGTH_IN_REPORT == 200

    def test_eval_runner_respects_max_workers(self):
        runner = EvalRunner(parallel_workers=100)
        assert runner.parallel_workers == MAX_PARALLEL_WORKERS

    def test_eval_runner_min_workers(self):
        runner = EvalRunner(parallel_workers=0)
        assert runner.parallel_workers == 1


# ===========================================================================
# Issue #29 - HiveMindGroupAdapter.ask_all() must not swallow errors
# ===========================================================================


class TestAskAllNoSilentFallback:
    """ask_all() must raise when agents fail."""

    def test_ask_all_raises_when_agent_fails(self):
        agents = {
            "good": _StubAgent(),
            "bad": _StubAgent(fail_answer=True),
        }
        hive = HiveMindGroupAdapter(agents=agents)
        with pytest.raises(RuntimeError, match="1/2 agents failed"):
            hive.ask_all("test question")

    def test_ask_all_succeeds_when_all_agents_work(self):
        agents = {
            "a1": _StubAgent(),
            "a2": _StubAgent(),
        }
        hive = HiveMindGroupAdapter(agents=agents)
        responses = hive.ask_all("test question")
        assert len(responses) == 2
        assert all(r.answer == "stub answer" for r in responses.values())
