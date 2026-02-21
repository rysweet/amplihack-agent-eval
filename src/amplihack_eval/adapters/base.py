"""Base adapter interface for evaluating any agent.

Philosophy:
- Agent-agnostic: any agent that can learn and answer questions is evaluable
- Trajectory capture: tool calls and reasoning traces enable deeper analysis
- Simple interface: just learn(), answer(), reset(), close()

Public API:
    ToolCall: Record of a single tool invocation
    AgentResponse: Response from an agent including trajectory
    AgentAdapter: Abstract interface for evaluable agents
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Record of a single tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    result: str
    timestamp: float = 0.0


@dataclass
class AgentResponse:
    """Response from an agent including trajectory."""

    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_trace: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(ABC):
    """Interface for evaluating any agent.

    Implement this to make your agent evaluable by the harness.

    Example::

        class MyAgent(AgentAdapter):
            def learn(self, content: str) -> None:
                self.memory.store(content)

            def answer(self, question: str) -> AgentResponse:
                result = self.memory.query(question)
                return AgentResponse(answer=result)

            def reset(self) -> None:
                self.memory.clear()

            def close(self) -> None:
                pass
    """

    @abstractmethod
    def learn(self, content: str) -> None:
        """Feed content to the agent for learning/memorization."""

    @abstractmethod
    def answer(self, question: str) -> AgentResponse:
        """Ask the agent a question. Returns answer + trajectory."""

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state between eval runs."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""

    @property
    def capabilities(self) -> set[str]:
        """What the agent can do. Override to declare capabilities."""
        return {"memory"}

    @property
    def name(self) -> str:
        """Human-readable agent name."""
        return self.__class__.__name__


__all__ = [
    "ToolCall",
    "AgentResponse",
    "AgentAdapter",
]
