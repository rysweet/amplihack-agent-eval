"""Agent adapters for the evaluation harness.

Provides the AgentAdapter interface and concrete adapters for
various agent implementations, including the HiveMindGroupAdapter
for multi-agent shared-memory evaluation.
"""

from __future__ import annotations

from .base import AgentAdapter, AgentResponse, ToolCall
from .hive_mind_adapter import (
    HiveMindGroupAdapter,
    InMemorySharedStore,
    PropagationResult,
    SharedMemoryStore,
)

__all__ = [
    "AgentAdapter",
    "AgentResponse",
    "ToolCall",
    # Hive mind
    "HiveMindGroupAdapter",
    "InMemorySharedStore",
    "PropagationResult",
    "SharedMemoryStore",
]
