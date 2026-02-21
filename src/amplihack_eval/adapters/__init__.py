"""Agent adapters for the evaluation harness.

Provides the AgentAdapter interface and concrete adapters for
various agent implementations.
"""

from __future__ import annotations

from .base import AgentAdapter, AgentResponse, ToolCall

__all__ = [
    "AgentAdapter",
    "AgentResponse",
    "ToolCall",
]
