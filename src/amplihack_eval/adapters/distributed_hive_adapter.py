"""Distributed hive adapter — wraps Event Hubs-based distributed agents.

Connects the eval harness to a fleet of remote agents deployed on
Azure Container Apps, communicating via Event Hubs. Uses the same
``RemoteAgentAdapter`` from amplihack that the ad-hoc eval script uses,
but packages it behind the standard ``AgentAdapter`` interface so that
the full eval harness (levels, grading, reports) works transparently.

Usage::

    amplihack-eval run --adapter distributed-hive \
        --connection-string "$EH_CONN" \
        --input-hub hive-events-myhive \
        --response-hub eval-responses-myhive \
        --agents 100 --turns 1000 --questions 20
"""

from __future__ import annotations

import logging
import os

from .base import AgentAdapter, AgentResponse

logger = logging.getLogger(__name__)


class DistributedHiveAdapter(AgentAdapter):
    """Adapter for distributed hive mind agents deployed on Azure.

    Wraps ``amplihack``'s ``RemoteAgentAdapter`` which handles:
    - Round-robin content distribution across N agents via Event Hubs
    - Question routing with correlation-based answer collection
    - Event Hubs consumer for eval responses

    Parameters
    ----------
    connection_string:
        Event Hubs namespace connection string.
    input_hub:
        Event Hub name for agent input (LEARN_CONTENT, INPUT events).
    response_hub:
        Event Hub name for eval responses (EVAL_ANSWER events).
    agent_count:
        Number of deployed agents.
    answer_timeout:
        Seconds to wait per answer (0 = no timeout).
    resource_group:
        Azure resource group (informational only).
    """

    def __init__(
        self,
        connection_string: str,
        input_hub: str = "hive-events",
        response_hub: str = "eval-responses",
        agent_count: int = 5,
        answer_timeout: int = 0,
        resource_group: str = "",
    ) -> None:
        # Import here to avoid hard dependency at module level
        try:
            from amplihack.eval.distributed_adapter import RemoteAgentAdapter
        except ImportError:
            # Fall back to the deploy-local copy
            try:
                import importlib.util
                import sys

                spec = importlib.util.spec_from_file_location(
                    "remote_agent_adapter",
                    os.path.join(
                        os.path.dirname(__file__),
                        "..", "..", "..", "..",
                        "deploy", "azure_hive", "remote_agent_adapter.py",
                    ),
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["remote_agent_adapter"] = mod
                    spec.loader.exec_module(mod)
                    RemoteAgentAdapter = mod.RemoteAgentAdapter  # type: ignore[assignment]
                else:
                    raise ImportError("Could not load remote_agent_adapter")
            except Exception as exc:
                raise ImportError(
                    "amplihack package with RemoteAgentAdapter required. "
                    "Install with: pip install amplihack"
                ) from exc

        self._adapter = RemoteAgentAdapter(
            connection_string=connection_string,
            input_hub=input_hub,
            response_hub=response_hub,
            agent_count=agent_count,
            answer_timeout=answer_timeout,
            resource_group=resource_group,
        )
        self._agent_count = agent_count
        self._learn_count = 0
        self._question_count = 0
        logger.info(
            "DistributedHiveAdapter initialized: %d agents, hubs=%s/%s",
            agent_count, input_hub, response_hub,
        )

    def learn(self, content: str) -> None:
        """Feed content to a remote agent (round-robin distribution)."""
        self._adapter.learn_from_content(content)
        self._learn_count += 1

    def answer(self, question: str) -> AgentResponse:
        """Ask a remote agent and return the response."""
        raw_answer = self._adapter.answer_question(question)
        self._question_count += 1
        return AgentResponse(
            answer=raw_answer,
            metadata={
                "agent_count": self._agent_count,
                "question_index": self._question_count,
            },
        )

    def reset(self) -> None:
        """Reset counters. Remote agents maintain their own state."""
        self._learn_count = 0
        self._question_count = 0

    def close(self) -> None:
        """Close the remote adapter and Event Hub connections."""
        self._adapter.close()

    @property
    def capabilities(self) -> set[str]:
        return {"memory", "distributed", "hive_mind"}

    @property
    def name(self) -> str:
        return f"DistributedHive({self._agent_count} agents)"


__all__ = ["DistributedHiveAdapter"]
