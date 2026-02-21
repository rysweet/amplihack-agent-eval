"""Tests for the AgentAdapter interface and concrete adapters."""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from amplihack_eval.adapters.base import AgentAdapter, AgentResponse, ToolCall


# --- ToolCall tests ---


class TestToolCall:
    def test_creation(self):
        tc = ToolCall(tool_name="search", arguments={"q": "hello"}, result="found it")
        assert tc.tool_name == "search"
        assert tc.arguments == {"q": "hello"}
        assert tc.result == "found it"
        assert tc.timestamp == 0.0

    def test_with_timestamp(self):
        tc = ToolCall(tool_name="fetch", arguments={}, result="ok", timestamp=123.45)
        assert tc.timestamp == 123.45


# --- AgentResponse tests ---


class TestAgentResponse:
    def test_minimal(self):
        resp = AgentResponse(answer="Paris")
        assert resp.answer == "Paris"
        assert resp.tool_calls == []
        assert resp.reasoning_trace == ""
        assert resp.confidence == 0.0
        assert resp.metadata == {}

    def test_full(self):
        tc = ToolCall(tool_name="search", arguments={"q": "capital"}, result="Paris")
        resp = AgentResponse(
            answer="Paris",
            tool_calls=[tc],
            reasoning_trace="Searched memory for capital of France",
            confidence=0.95,
            metadata={"model": "test"},
        )
        assert len(resp.tool_calls) == 1
        assert resp.confidence == 0.95


# --- AgentAdapter interface tests ---


class MockAgent(AgentAdapter):
    """Concrete test implementation of AgentAdapter."""

    def __init__(self):
        self.learned: list[str] = []
        self.closed = False

    def learn(self, content: str) -> None:
        self.learned.append(content)

    def answer(self, question: str) -> AgentResponse:
        # Simple echo agent
        if self.learned:
            return AgentResponse(answer=f"I know: {self.learned[-1]}")
        return AgentResponse(answer="I don't know")

    def reset(self) -> None:
        self.learned.clear()

    def close(self) -> None:
        self.closed = True


class TestAgentAdapter:
    def test_learn_and_answer(self):
        agent = MockAgent()
        agent.learn("The capital of France is Paris.")
        resp = agent.answer("What is the capital of France?")
        assert "Paris" in resp.answer

    def test_answer_without_learning(self):
        agent = MockAgent()
        resp = agent.answer("What is 2+2?")
        assert resp.answer == "I don't know"

    def test_reset(self):
        agent = MockAgent()
        agent.learn("fact 1")
        agent.learn("fact 2")
        assert len(agent.learned) == 2
        agent.reset()
        assert len(agent.learned) == 0

    def test_close(self):
        agent = MockAgent()
        assert not agent.closed
        agent.close()
        assert agent.closed

    def test_capabilities_default(self):
        agent = MockAgent()
        assert agent.capabilities == {"memory"}

    def test_name_default(self):
        agent = MockAgent()
        assert agent.name == "MockAgent"

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AgentAdapter()  # type: ignore[abstract]


# --- Custom capabilities ---


class CustomCapAgent(AgentAdapter):
    def learn(self, content: str) -> None:
        pass

    def answer(self, question: str) -> AgentResponse:
        return AgentResponse(answer="test")

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def capabilities(self) -> set[str]:
        return {"memory", "tool_use", "planning"}

    @property
    def name(self) -> str:
        return "CustomBot"


class TestCustomCapabilities:
    def test_custom_capabilities(self):
        agent = CustomCapAgent()
        assert "tool_use" in agent.capabilities
        assert "planning" in agent.capabilities

    def test_custom_name(self):
        agent = CustomCapAgent()
        assert agent.name == "CustomBot"


# --- HttpAdapter tests ---


class TestHttpAdapter:
    def test_creation(self):
        from amplihack_eval.adapters.http_adapter import HttpAdapter

        adapter = HttpAdapter(base_url="http://localhost:8000")
        assert adapter.name == "HTTP(http://localhost:8000)"

    def test_trailing_slash_stripped(self):
        from amplihack_eval.adapters.http_adapter import HttpAdapter

        adapter = HttpAdapter(base_url="http://localhost:8000/")
        assert adapter._base_url == "http://localhost:8000"


# --- SubprocessAdapter tests ---


class TestSubprocessAdapter:
    def test_creation(self):
        from amplihack_eval.adapters.subprocess_adapter import SubprocessAdapter

        adapter = SubprocessAdapter(command=["echo", "hello"])
        assert "echo" in adapter.name

    def test_learn_calls_subprocess(self):
        from amplihack_eval.adapters.subprocess_adapter import SubprocessAdapter

        adapter = SubprocessAdapter(
            command=[sys.executable, "-c", "import sys; print('ok')"],
            learn_flag="--learn",
        )
        # Should not raise
        adapter.learn("test content")

    def test_answer_returns_response(self):
        from amplihack_eval.adapters.subprocess_adapter import SubprocessAdapter

        adapter = SubprocessAdapter(
            command=[sys.executable, "-c", "import sys; print('42')"],
            answer_flag="--answer",
        )
        resp = adapter.answer("What is the answer?")
        assert isinstance(resp, AgentResponse)
        # The subprocess ignores stdin but prints 42
        assert "42" in resp.answer

    def test_close_is_noop(self):
        from amplihack_eval.adapters.subprocess_adapter import SubprocessAdapter

        adapter = SubprocessAdapter(command=["echo"])
        adapter.close()  # Should not raise
