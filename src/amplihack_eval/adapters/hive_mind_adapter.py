"""Hive mind group adapter for multi-agent knowledge sharing evaluation.

Wraps multiple AgentAdapter instances that share knowledge through a
configurable shared memory system. Decoupled from specific hive mind
implementations -- works with any shared memory that supports
store/query operations.

Philosophy:
- Agent-agnostic: wraps any AgentAdapter implementation
- Hive-agnostic: works with any shared memory (in-memory dict, graph DB, etc.)
- Deterministic learning: each agent learns its assigned facts
- Testable: built-in in-memory shared store for testing without external deps

Public API:
    SharedMemoryStore: Protocol for pluggable shared memory backends
    InMemorySharedStore: Simple dict-based shared store for testing
    HiveMindGroupAdapter: Manages a group of agents connected via shared memory
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .base import AgentAdapter, AgentResponse

logger = logging.getLogger(__name__)

# Maximum number of shared facts to inject as context when answering a question
MAX_SHARED_CONTEXT_FACTS = 50


@runtime_checkable
class SharedMemoryStore(Protocol):
    """Protocol for shared memory backends.

    Any object implementing these methods can serve as the hive mind's
    shared store. This keeps the adapter decoupled from specific
    implementations (graph DBs, vector stores, etc.).
    """

    def store(self, agent_id: str, fact: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a fact attributed to a specific agent."""
        ...

    def query(self, question: str, requesting_agent: str | None = None) -> list[str]:
        """Query the shared store. Returns relevant facts."""
        ...

    def get_all_facts(self, agent_id: str | None = None) -> list[str]:
        """Get all facts, optionally filtered by source agent."""
        ...

    def get_agent_ids(self) -> list[str]:
        """Get all agent IDs that have contributed facts."""
        ...

    def clear(self) -> None:
        """Clear all stored facts."""
        ...


class InMemorySharedStore:
    """Simple in-memory shared store for testing.

    Stores facts in a dict keyed by agent_id. Query returns all facts
    that contain any word from the question (case-insensitive).
    Suitable for unit tests and small-scale evaluations.
    """

    def __init__(self) -> None:
        self._facts: dict[str, list[str]] = {}

    def store(self, agent_id: str, fact: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a fact attributed to a specific agent."""
        if agent_id not in self._facts:
            self._facts[agent_id] = []
        self._facts[agent_id].append(fact)

    def query(self, question: str, requesting_agent: str | None = None) -> list[str]:
        """Query with simple keyword matching across all agents' facts."""
        question_words = set(question.lower().split())
        # Remove common stop words for better matching
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "how",
            "does",
            "do",
            "and",
            "or",
            "of",
            "in",
            "to",
            "for",
            "with",
            "on",
            "at",
            "by",
            "from",
            "that",
            "this",
            "it",
            "be",
            "has",
            "have",
            "had",
            "not",
            "but",
            "if",
            "can",
            "which",
            "as",
        }
        query_terms = question_words - stop_words

        if not query_terms:
            # Fall back to all facts if only stop words
            return [f for facts in self._facts.values() for f in facts]

        results = []
        for agent_id, facts in self._facts.items():
            for fact in facts:
                fact_lower = fact.lower()
                if any(term in fact_lower for term in query_terms):
                    results.append(fact)

        return results

    def get_all_facts(self, agent_id: str | None = None) -> list[str]:
        """Get all facts, optionally filtered by source agent."""
        if agent_id is not None:
            return list(self._facts.get(agent_id, []))
        return [f for facts in self._facts.values() for f in facts]

    def get_agent_ids(self) -> list[str]:
        """Get all agent IDs that have contributed facts."""
        return list(self._facts.keys())

    def clear(self) -> None:
        """Clear all stored facts."""
        self._facts.clear()


@dataclass
class PropagationResult:
    """Result of knowledge propagation across the hive.

    Attributes:
        rounds_executed: Number of gossip/propagation rounds completed
        facts_propagated: Total facts propagated across all rounds
        agents_reached: Number of agents that received new knowledge
    """

    rounds_executed: int
    facts_propagated: int
    agents_reached: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds_executed": self.rounds_executed,
            "facts_propagated": self.facts_propagated,
            "agents_reached": self.agents_reached,
        }


class HiveMindGroupAdapter:
    """Manages a group of agents connected via a shared hive mind.

    Each agent can learn facts independently. When connected to a shared
    memory store, agents can answer questions using both their own
    knowledge and knowledge from other agents in the hive.

    The adapter is decoupled from specific hive mind implementations.
    Provide any object implementing the SharedMemoryStore protocol.

    Args:
        agents: Dict mapping agent_id to AgentAdapter instance
        shared_store: Shared memory backend (defaults to InMemorySharedStore)
        propagation_rounds: Number of gossip rounds for propagate_knowledge()

    Example::

        agents = {"net": NetworkAgent(), "sec": SecurityAgent()}
        hive = HiveMindGroupAdapter(agents=agents)
        hive.learn_distributed({
            "net": ["Fact 1", "Fact 2"],
            "sec": ["Fact 3", "Fact 4"],
        })
        response = hive.ask_agent("net", "What security measures exist?")
    """

    def __init__(
        self,
        agents: dict[str, AgentAdapter],
        shared_store: SharedMemoryStore | None = None,
        propagation_rounds: int = 3,
    ):
        self.agents = agents
        self.shared_store: SharedMemoryStore = shared_store or InMemorySharedStore()
        self.propagation_rounds = propagation_rounds
        self._agent_facts: dict[str, list[str]] = {aid: [] for aid in agents}

    @property
    def agent_ids(self) -> list[str]:
        """List of all agent IDs in the hive."""
        return list(self.agents.keys())

    @property
    def num_agents(self) -> int:
        """Number of agents in the hive."""
        return len(self.agents)

    def learn_distributed(self, agent_knowledge: dict[str, list[str]]) -> dict[str, int]:
        """Each agent learns its assigned facts.

        Facts are also stored in the shared memory so other agents can
        access them via queries.

        Args:
            agent_knowledge: Dict mapping agent_id to list of fact strings

        Returns:
            Dict mapping agent_id to number of facts learned

        Raises:
            ValueError: If agent_id not found in the hive
        """
        facts_learned: dict[str, int] = {}

        for agent_id, facts in agent_knowledge.items():
            if agent_id not in self.agents:
                raise ValueError(f"Agent '{agent_id}' not found in hive. Available: {list(self.agents.keys())}")

            agent = self.agents[agent_id]
            count = 0

            for fact in facts:
                # Agent learns the fact directly
                agent.learn(fact)
                # Also store in shared memory for cross-agent access
                self.shared_store.store(agent_id, fact, metadata={"source": agent_id})
                count += 1

            self._agent_facts[agent_id] = list(facts)
            facts_learned[agent_id] = count
            logger.info("Agent '%s' learned %d facts", agent_id, count)

        return facts_learned

    def propagate_knowledge(self) -> PropagationResult:
        """Run gossip/event propagation across agents.

        Each round, every agent learns facts from the shared store that
        it hasn't seen yet. The number of rounds is configurable via
        propagation_rounds.

        Returns:
            PropagationResult with stats on the propagation
        """
        total_propagated = 0
        agents_reached = set()

        for round_num in range(self.propagation_rounds):
            round_propagated = 0

            for agent_id, agent in self.agents.items():
                # Get all facts this agent hasn't learned yet
                all_facts = self.shared_store.get_all_facts()
                own_facts = set(self._agent_facts.get(agent_id, []))

                new_facts = [f for f in all_facts if f not in own_facts]

                for fact in new_facts:
                    agent.learn(fact)
                    round_propagated += 1
                    agents_reached.add(agent_id)

                # Update the agent's known facts
                self._agent_facts[agent_id] = list(own_facts | set(new_facts))

            total_propagated += round_propagated
            logger.info(
                "Propagation round %d: %d facts propagated",
                round_num + 1,
                round_propagated,
            )

            # Early exit if no new facts propagated (convergence)
            if round_propagated == 0:
                logger.info("Propagation converged after %d rounds", round_num + 1)
                break

        return PropagationResult(
            rounds_executed=round_num + 1 if self.propagation_rounds > 0 else 0,
            facts_propagated=total_propagated,
            agents_reached=len(agents_reached),
        )

    def ask_agent(self, agent_id: str, question: str) -> AgentResponse:
        """Ask a specific agent a question.

        The agent can access the shared store to augment its answer with
        knowledge from other agents in the hive.

        Args:
            agent_id: Which agent to ask
            question: The question text

        Returns:
            AgentResponse from the specified agent

        Raises:
            ValueError: If agent_id not found in the hive
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent '{agent_id}' not found in hive. Available: {list(self.agents.keys())}")

        # Query shared store for relevant context
        shared_context = self.shared_store.query(question, requesting_agent=agent_id)

        # Feed context to agent if available
        if shared_context:
            context_text = "Shared knowledge from the hive:\n" + "\n".join(
                f"- {fact}" for fact in shared_context[:MAX_SHARED_CONTEXT_FACTS]
            )
            self.agents[agent_id].learn(context_text)

        return self.agents[agent_id].answer(question)

    def ask_all(self, question: str) -> dict[str, AgentResponse]:
        """Ask all agents the same question and collect responses.

        Useful for comparing answers across agents and detecting
        consensus or disagreement.

        Args:
            question: The question text

        Returns:
            Dict mapping agent_id to AgentResponse

        Raises:
            RuntimeError: If any agent fails to answer. The error message
                includes which agent(s) failed and why.
        """
        responses: dict[str, AgentResponse] = {}
        errors: dict[str, Exception] = {}

        for agent_id in self.agents:
            try:
                responses[agent_id] = self.ask_agent(agent_id, question)
            except Exception as e:
                errors[agent_id] = e

        if errors:
            error_details = "; ".join(f"{aid}: {type(e).__name__}: {e}" for aid, e in errors.items())
            raise RuntimeError(f"{len(errors)}/{len(self.agents)} agents failed to answer: {error_details}")

        return responses

    def reset(self) -> None:
        """Reset all agents and the shared store."""
        for agent in self.agents.values():
            agent.reset()
        self.shared_store.clear()
        self._agent_facts = {aid: [] for aid in self.agents}

    def close(self) -> None:
        """Close all agents and clean up resources."""
        for agent in self.agents.values():
            agent.close()

    def get_coverage_stats(self) -> dict[str, Any]:
        """Get knowledge coverage statistics for the hive.

        Returns:
            Dict with coverage metrics per agent and totals
        """
        all_facts = set(self.shared_store.get_all_facts())
        total_facts = len(all_facts)

        per_agent: dict[str, dict[str, Any]] = {}
        for agent_id in self.agents:
            agent_facts = set(self._agent_facts.get(agent_id, []))
            per_agent[agent_id] = {
                "own_facts": len(self.shared_store.get_all_facts(agent_id)),
                "total_known": len(agent_facts),
                "coverage_pct": (len(agent_facts) / total_facts * 100 if total_facts > 0 else 0.0),
            }

        return {
            "total_facts_in_hive": total_facts,
            "num_agents": len(self.agents),
            "per_agent": per_agent,
        }


__all__ = [
    "SharedMemoryStore",
    "InMemorySharedStore",
    "PropagationResult",
    "HiveMindGroupAdapter",
]
