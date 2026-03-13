"""Agent adapters for the evaluation harness.

Provides the AgentAdapter interface and concrete adapters for
various agent implementations, including the HiveMindGroupAdapter
for multi-agent shared-memory evaluation and DistributedHiveAdapter
for remote agents deployed on Azure Container Apps.
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

# Lazy import for DistributedHiveAdapter (requires azure-eventhub)
def __getattr__(name: str):
    if name == "DistributedHiveAdapter":
        from .distributed_hive_adapter import DistributedHiveAdapter
        return DistributedHiveAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
