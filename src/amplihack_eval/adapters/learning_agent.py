"""Adapter for amplihack's LearningAgent (direct import).

Wraps a LearningAgent instance to implement the AgentAdapter interface.
Requires amplihack to be installed: pip install amplihack

Usage::

    from amplihack_eval.adapters.learning_agent import LearningAgentAdapter

    adapter = LearningAgentAdapter(
        model="claude-sonnet-4-5-20250929",
        storage_path="/tmp/eval_db",
    )
    adapter.learn("The capital of France is Paris.")
    response = adapter.answer("What is the capital of France?")
    print(response.answer)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import AgentAdapter, AgentResponse

logger = logging.getLogger(__name__)


class LearningAgentAdapter(AgentAdapter):
    """Adapter wrapping amplihack's LearningAgent.

    This adapter requires the ``amplihack`` package to be installed.
    It creates a LearningAgent instance and delegates learn/answer
    calls through it.

    Args:
        agent_name: Name for the underlying agent instance
        model: LLM model identifier
        storage_path: Path for the agent's memory database
        use_hierarchical: Whether to use hierarchical memory
    """

    def __init__(
        self,
        agent_name: str = "eval_agent",
        model: str = "",
        storage_path: str | Path = "/tmp/eval_memory_db",
        use_hierarchical: bool = True,
    ):
        try:
            from amplihack.agents.goal_seeking.learning_agent import (  # type: ignore[import-untyped]
                LearningAgent,
            )
        except ImportError as e:
            raise ImportError(
                "amplihack package is required for LearningAgentAdapter. "
                "Install it with: pip install amplihack"
            ) from e

        import os

        self._model = model or os.environ.get("EVAL_MODEL", "claude-sonnet-4-5-20250929")
        self._agent = LearningAgent(
            agent_name=agent_name,
            model=self._model,
            storage_path=Path(storage_path),
            use_hierarchical=use_hierarchical,
        )

    def learn(self, content: str) -> None:
        """Feed content to the LearningAgent."""
        try:
            self._agent.learn_from_content(content)
        except Exception as e:
            logger.warning("Failed to learn content: %s", e)

    def answer(self, question: str) -> AgentResponse:
        """Ask the LearningAgent a question."""
        try:
            result = self._agent.answer_question(question)
            answer_text = result
            if isinstance(result, tuple):
                answer_text = result[0]
            return AgentResponse(
                answer=str(answer_text),
                metadata={"model": self._model},
            )
        except Exception as e:
            logger.warning("Failed to answer question: %s", e)
            return AgentResponse(answer=f"Error: {e}")

    def reset(self) -> None:
        """Reset the agent (close and recreate)."""
        self.close()

    def close(self) -> None:
        """Close the underlying agent."""
        if hasattr(self._agent, "close"):
            self._agent.close()

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics from the underlying agent."""
        try:
            return self._agent.get_memory_stats()
        except Exception:
            return {}

    @property
    def name(self) -> str:
        return f"LearningAgent({self._model})"


__all__ = ["LearningAgentAdapter"]
