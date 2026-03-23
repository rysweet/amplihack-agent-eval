"""RemoteAgentAdapter — makes a deployed Azure Container Apps agent look like a local agent.

Implements the same interface as LearningAgent (learn_from_content, answer_question)
so it can be passed directly to LongHorizonMemoryEval.run(). The eval harness
uses the exact same code path for local and distributed agents.

Transport: Azure Event Hubs (CBS-free AMQP — works reliably in Container Apps).
  - learn_from_content() sends LEARN_CONTENT events via EH producer,
    routed to the target agent's deterministic partition.
  - answer_question() sends INPUT events via EH producer,
    waits for EVAL_ANSWER on the eval-responses Event Hub.
  - learn_from_content() first pings all agents with ONLINE_CHECK so feed
    content is not published before every target agent is actually listening.
  - _wait_for_agents_idle() sends FEED_COMPLETE to all agents and waits
    for N AGENT_READY events on the eval-responses hub.
  - learn_from_content() also waits for an initial AGENT_PROGRESS event from
    every target agent before trusting feed progress logs.

Content is partitioned round-robin across agents (each agent learns N/agent_count turns).
Questions are targeted to specific agents via target_agent field.
Answers are collected from the eval-responses Event Hub, correlated by event_id.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from amplihack.observability import configure_otel, start_span
except ImportError:  # pragma: no cover
    import contextlib as _contextlib

    def configure_otel(  # type: ignore[misc]
        service_name: str, *, component: str = "", attributes: Any = None
    ) -> bool:
        return False

    def start_span(  # type: ignore[misc]
        name: str, *, tracer_name: str, attributes: Any = None
    ) -> Any:
        return _contextlib.nullcontext()


logger = logging.getLogger(__name__)
logging.getLogger("azure.eventhub").setLevel(logging.WARNING)
logging.getLogger("azure.eventhub._pyamqp").setLevel(logging.WARNING)
logging.getLogger("azure.eventhub._pyamqp.cbs").setLevel(logging.WARNING)
logging.getLogger("uamqp").setLevel(logging.WARNING)

_FAILOVER_ANSWER_PREFIXES = (
    "no answer received",
    "the provided facts do not contain",
    "the provided facts don't contain",
    "no information available",
    "not enough context",
    "i don't have enough",
    "i don't have information",
    "i cannot answer",
    "cannot answer this",
)


class RemoteAgentAdapter:
    """Adapter that forwards learn/answer calls to deployed agents via Event Hubs."""

    def __init__(
        self,
        connection_string: str,
        input_hub: str,
        response_hub: str,
        agent_count: int = 100,
        agents_per_app: int = 1,
        resource_group: str = "",
        answer_timeout: int = 0,
        replicate_learning_to_all_agents: bool = False,
        question_failover_retries: int = 0,
    ) -> None:
        self._connection_string = connection_string
        self._input_hub = input_hub
        self._response_hub = response_hub
        self._resource_group = resource_group
        self._agent_count = agent_count
        self._agents_per_app = max(1, min(agent_count, agents_per_app))

        self._learn_count = 0
        self._learn_turn_counts = [0 for _ in range(agent_count)]
        self._question_count = 0
        self._answer_timeout = answer_timeout
        self._replicate_learning_to_all_agents = replicate_learning_to_all_agents
        self._question_failover_retries = max(0, question_failover_retries)
        self._shutdown = threading.Event()
        self._startup_wait_done = threading.Event()
        self._idle_wait_done = threading.Event()

        # Thread safety for counters and answer dict
        self._counter_lock = threading.Lock()
        self._answer_lock = threading.Lock()
        self._producer_lock = threading.Lock()
        self._extractor_lock = threading.Lock()

        # Pending answers: event_id -> answer text
        self._pending_answers: dict[str, str] = {}
        self._answer_events: dict[str, threading.Event] = {}
        self._answer_targets: dict[str, str] = {}
        self._pending_question_meta: dict[str, dict[str, Any]] = {}
        self._agent_boot_ids: dict[str, str] = {}
        self._producer: Any | None = None
        self._fact_batch_extractor: Any | None = None
        self._fact_batch_extractor_dir: Path | None = None

        # AGENT_READY tracking for _wait_for_agents_idle
        self._ready_agents: set[str] = set()
        self._ready_boot_ids: dict[str, str] = {}
        self._question_ineligible_agents: set[str] = set()
        self._ready_lock = threading.Lock()
        self._all_agents_ready = threading.Event()

        # AGENT_ONLINE tracking for pre-feed startup synchronization
        self._online_agents: set[str] = set()
        self._online_lock = threading.Lock()
        self._all_agents_online = threading.Event()

        # AGENT_PROGRESS tracking for early feed telemetry verification
        self._progress_agents: set[str] = set()
        self._progress_counts: dict[str, int] = {}
        self._progress_lock = threading.Lock()
        self._feed_telemetry_wait_done = threading.Event()
        self._feed_telemetry_monitor_thread: threading.Thread | None = None

        # Unique run_id to filter stale events from previous eval runs
        self._run_id = uuid.uuid4().hex[:12]
        self._num_partitions: int | None = None

        # Listener liveness flag — fail fast if listener can't connect
        self._listener_alive = threading.Event()
        self._listener_restart_count = 0
        self._last_listener_event_at = 0.0
        self._last_listener_event_type = ""
        self._last_listener_partition = ""

        self._listener_thread = threading.Thread(target=self._listen_for_answers, daemon=True)
        self._listener_thread.start()

        # Wait up to 30s for listener to connect
        if not self._listener_alive.wait(timeout=30):
            raise RuntimeError(
                f"Failed to connect to response hub '{response_hub}'. "
                "Check that the Event Hub and consumer group 'eval-reader' exist."
            )

        logger.info(
            "RemoteAgentAdapter: input=%s response=%s agents=%d agents_per_app=%d run_id=%s",
            input_hub,
            response_hub,
            agent_count,
            self._agents_per_app,
            self._run_id,
        )
        configure_otel(
            service_name=os.environ.get("OTEL_SERVICE_NAME", "").strip() or "amplihack.azure-eval-harness",
            component="remote-agent-adapter",
            attributes=self._span_attributes(),
        )

    @staticmethod
    def _agent_index(agent_id: str) -> int:
        """Extract numeric index from ``agent-N`` names."""
        try:
            return int(agent_id.rsplit("-", 1)[-1])
        except (ValueError, IndexError):
            digest = hashlib.sha256(agent_id.encode("utf-8")).digest()
            return int.from_bytes(digest[:8], "big", signed=False)

    def _get_num_partitions(self) -> int:
        """Return the input hub partition count, caching the first result."""
        if self._num_partitions is not None:
            return self._num_partitions
        consumer = None
        try:
            from azure.eventhub import EventHubConsumerClient  # type: ignore[import-unresolved]

            consumer = EventHubConsumerClient.from_connection_string(
                self._connection_string,
                consumer_group="$Default",
                eventhub_name=self._input_hub,
            )
            self._num_partitions = len(consumer.get_partition_ids())
        except Exception as exc:
            logger.warning(
                "RemoteAgentAdapter: failed to query partition count for hub=%s; "
                "falling back to 32 partitions (%s: %s)",
                self._input_hub,
                type(exc).__name__,
                exc,
            )
            self._num_partitions = 32
        finally:
            if consumer is not None:
                consumer.close()
        return self._num_partitions

    def _target_partition(self, agent_id: str) -> str:
        """Deterministic partition for an agent: agent_index % num_partitions."""
        return str(self._agent_index(agent_id) % self._get_num_partitions())

    @staticmethod
    def _agent_name(agent_index: int) -> str:
        """Return the canonical agent name for an index."""
        return f"agent-{agent_index}"

    def _span_attributes(self, **extra: Any) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            "amplihack.input_hub": self._input_hub,
            "amplihack.response_hub": self._response_hub,
            "amplihack.agent_count": self._agent_count,
            "amplihack.agents_per_app": self._agents_per_app,
            "amplihack.run_id": self._run_id,
            "amplihack.replicate_learning": self._replicate_learning_to_all_agents,
            "amplihack.question_failover_retries": self._question_failover_retries,
        }
        attributes.update({key: value for key, value in extra.items() if value is not None})
        return attributes

    def _publish_event(self, payload: dict, partition_key: str) -> None:
        """Publish a single JSON event to the input Event Hub."""
        from azure.eventhub import EventData  # type: ignore[import-unresolved]

        payload["run_id"] = self._run_id
        route_partition_id: str | None = None
        if partition_key.startswith("agent-"):
            route_partition_id = self._target_partition(partition_key)

        producer = self._get_producer()
        try:
            kwargs: dict[str, str] = {}
            if route_partition_id is not None:
                kwargs["partition_id"] = route_partition_id
            else:
                kwargs["partition_key"] = partition_key
            batch = producer.create_batch(**kwargs)
            batch.add(EventData(json.dumps(payload)))
            producer.send_batch(batch)
        except Exception:
            logger.warning("EH publish failed, retrying once", exc_info=True)
            self._reset_producer()
            producer2 = self._get_producer()
            try:
                kwargs: dict[str, str] = {}
                if route_partition_id is not None:
                    kwargs["partition_id"] = route_partition_id
                else:
                    kwargs["partition_key"] = partition_key
                batch = producer2.create_batch(**kwargs)
                batch.add(EventData(json.dumps(payload)))
                producer2.send_batch(batch)
            except Exception:
                logger.error(
                    "EH publish failed after retry (event_type=%s)",
                    payload.get("event_type", "?"),
                    exc_info=True,
                )
                raise

    def _get_producer(self) -> Any:
        """Return a cached Event Hub producer, creating it lazily when needed."""
        with self._producer_lock:
            if self._producer is None:
                from azure.eventhub import EventHubProducerClient  # type: ignore[import-unresolved]

                self._producer = EventHubProducerClient.from_connection_string(
                    self._connection_string,
                    eventhub_name=self._input_hub,
                )
            return self._producer

    def _reset_producer(self) -> None:
        """Close and clear the cached Event Hub producer."""
        with self._producer_lock:
            producer = self._producer
            self._producer = None
        if producer is not None:
            try:
                producer.close()
            except Exception:
                logger.debug("Failed to close cached producer cleanly", exc_info=True)

    def _get_fact_batch_extractor(self) -> Any:
        """Return a reusable local extractor for replicated fact batches."""
        with self._extractor_lock:
            if self._fact_batch_extractor is None:
                from amplihack.agents.goal_seeking.runtime_factory import create_goal_agent_runtime

                extractor_dir = Path(tempfile.mkdtemp(prefix="amplihack-azure-fact-batch-"))
                self._fact_batch_extractor_dir = extractor_dir
                self._fact_batch_extractor = create_goal_agent_runtime(
                    agent_name="eval-harness-fact-extractor",
                    storage_path=extractor_dir,
                    use_hierarchical=True,
                    runtime_kind="goal",
                    bind_answer_mode=False,
                )
            return self._fact_batch_extractor

    def _prepare_fact_batch(self, content: str) -> dict[str, Any]:
        """Prepare a direct-storage fact batch for replicated learning."""
        extractor = self._get_fact_batch_extractor()
        return extractor.prepare_fact_batch(content, include_summary=False)

    def _wait_for_agents_online(self) -> None:
        """Wait until every target agent acknowledges ONLINE_CHECK.

        This prevents the eval feed from starting while some agents are still
        booting and not yet consuming their assigned Event Hubs partitions.
        """
        with start_span(
            "azure_eval.wait_for_agents_online",
            tracer_name=__name__,
            attributes=self._span_attributes(),
        ):
            logger.info(
                "Sending ONLINE_CHECK to all %d agents before feed phase...",
                self._agent_count,
            )

            with self._online_lock:
                self._online_agents.clear()
                self._all_agents_online.clear()

            poll_interval = 10
            while True:
                with self._online_lock:
                    missing_agents = [
                        f"agent-{i}" for i in range(self._agent_count) if f"agent-{i}" not in self._online_agents
                    ]
                    online_count = self._agent_count - len(missing_agents)

                if not missing_agents:
                    logger.info("All %d agents online. Starting feed phase.", self._agent_count)
                    return

                for target_name in missing_agents:
                    self._publish_event(
                        {
                            "event_type": "ONLINE_CHECK",
                            "event_id": uuid.uuid4().hex[:12],
                            "target_agent": target_name,
                            "source_agent": "eval-harness",
                            "payload": {"target_agent": target_name},
                        },
                        partition_key=target_name,
                    )

                logger.info(
                    "  %d/%d agents online, pinging missing agents: %s",
                    online_count,
                    self._agent_count,
                    ", ".join(missing_agents),
                )
                time.sleep(poll_interval)

    def _wait_for_feed_telemetry(self, expected_agents: set[str]) -> None:
        """Wait until each expected agent reports real feed processing."""
        if not expected_agents:
            return

        with start_span(
            "azure_eval.wait_for_feed_telemetry",
            tracer_name=__name__,
            attributes=self._span_attributes(expected_agents=len(expected_agents)),
        ):
            logger.info(
                "Waiting for initial AGENT_PROGRESS from %d agents before trusting feed telemetry...",
                len(expected_agents),
            )
            poll_interval = 5
            while not self._shutdown.is_set():
                with self._progress_lock:
                    missing_agents = sorted(expected_agents - self._progress_agents)

                if not missing_agents:
                    logger.info(
                        "Received AGENT_PROGRESS from all %d agents. Feed telemetry confirmed.",
                        len(expected_agents),
                    )
                    return

                logger.info("  waiting for AGENT_PROGRESS from: %s", ", ".join(missing_agents))
                time.sleep(poll_interval)

            logger.warning(
                "Feed telemetry monitor stopped before all agents reported progress; still missing: %s",
                ", ".join(sorted(expected_agents - self._progress_agents)),
            )

    def _start_feed_telemetry_monitor(self, expected_agents: set[str]) -> None:
        """Start a background monitor for initial AGENT_PROGRESS coverage."""
        if not expected_agents or self._feed_telemetry_wait_done.is_set():
            return

        self._feed_telemetry_wait_done.set()
        monitor = threading.Thread(
            target=self._wait_for_feed_telemetry,
            args=(set(expected_agents),),
            daemon=True,
            name="feed-telemetry-monitor",
        )
        self._feed_telemetry_monitor_thread = monitor
        monitor.start()

    def learn_from_content(self, content: str) -> dict[str, Any]:
        """Send content to one agent or to all agents when replication is enabled.

        5000 turns / N agents = ~(5000/N) turns each. Each agent learns its
        partition locally. The hive mind shares knowledge between agents
        so any agent can answer questions about any content.
        """
        with start_span(
            "azure_eval.learn_from_content",
            tracer_name=__name__,
            attributes=self._span_attributes(content_length=len(content)),
        ):
            if not self._startup_wait_done.is_set():
                with self._counter_lock:
                    if not self._startup_wait_done.is_set():
                        self._wait_for_agents_online()
                        self._startup_wait_done.set()

            event_id = uuid.uuid4().hex[:12]
            with self._counter_lock:
                target_agent = self._learn_count % self._agent_count
                self._learn_count += 1
                learn_count = self._learn_count
                if self._replicate_learning_to_all_agents:
                    target_agents = list(range(self._agent_count))
                    for agent_index in target_agents:
                        self._learn_turn_counts[agent_index] += 1
                else:
                    target_agents = [target_agent]
                    self._learn_turn_counts[target_agent] += 1

            fact_batch: dict[str, Any] | None = None
            if self._replicate_learning_to_all_agents:
                fact_batch = self._prepare_fact_batch(content)

            for agent_index in target_agents:
                target_name = self._agent_name(agent_index)
                if self._replicate_learning_to_all_agents:
                    payload = {
                        "event_type": "STORE_FACT_BATCH",
                        "event_id": event_id,
                        "target_agent": target_name,
                        "source_agent": "eval-harness",
                        "payload": {
                            "fact_batch": fact_batch,
                            "target_agent": target_name,
                        },
                    }
                else:
                    payload = {
                        "event_type": "LEARN_CONTENT",
                        "event_id": event_id,
                        "target_agent": target_name,
                        "source_agent": "eval-harness",
                        "payload": {
                            "content": content,
                            "target_agent": target_name,
                        },
                    }
                self._publish_event(
                    payload,
                    partition_key=target_name,
                )

            if not self._feed_telemetry_wait_done.is_set():
                expected_progress_agents: set[str] | None = None
                if (self._replicate_learning_to_all_agents and learn_count >= 1) or (
                    not self._replicate_learning_to_all_agents and learn_count >= self._agent_count
                ):
                    expected_progress_agents = {self._agent_name(i) for i in range(self._agent_count)}

                if expected_progress_agents is not None:
                    self._start_feed_telemetry_monitor(expected_progress_agents)

            log_every = 50 if self._replicate_learning_to_all_agents else 500
            if learn_count % log_every == 0:
                if self._replicate_learning_to_all_agents:
                    logger.info(
                        "RemoteAgentAdapter: sent %d content turns (STORE_FACT_BATCH replicated to all %d agents)",
                        learn_count,
                        self._agent_count,
                    )
                else:
                    logger.info(
                        "RemoteAgentAdapter: sent %d content turns (%d per agent)",
                        learn_count,
                        learn_count // max(1, self._agent_count),
                    )

            return {
                "facts_stored": len((fact_batch or {}).get("facts", [])) if fact_batch else 1,
                "event_id": event_id,
                "replicated_to": len(target_agents),
            }

    def _send_question_to_agent(self, question: str, target_agent: int) -> str:
        """Send one question attempt to one agent and wait for its answer."""
        target_name = self._agent_name(target_agent)
        target_partition = self._target_partition(target_name)
        with start_span(
            "azure_eval.send_question",
            tracer_name=__name__,
            attributes=self._span_attributes(
                question_length=len(question),
                target_agent=target_name,
            ),
        ):
            event_id = uuid.uuid4().hex[:12]
            question_id = f"q_{target_agent}_{event_id}"
            sent_at = time.monotonic()

            answer_event = threading.Event()
            with self._answer_lock:
                self._answer_events[event_id] = answer_event
                self._answer_targets[event_id] = target_name
                self._pending_question_meta[event_id] = {
                    "question_id": question_id,
                    "partition_id": target_partition,
                    "question_preview": question[:120],
                    "sent_at_monotonic": sent_at,
                }

            self._publish_event(
                {
                    "event_type": "INPUT",
                    "event_id": event_id,
                    "target_agent": target_name,
                    "source_agent": "eval-harness",
                    "payload": {
                        "question": question,
                        "question_id": question_id,
                        "target_agent": target_name,
                    },
                },
                partition_key=target_name,
            )

            logger.info(
                "RemoteAgentAdapter: sent question to %s (event_id=%s question_id=%s partition=%s run_id=%s): %s",
                target_name,
                event_id,
                question_id,
                target_partition,
                self._run_id,
                question[:60],
            )

            got_answer = self._wait_for_answer_event(event_id, answer_event)
            if not got_answer:
                logger.warning(
                    "answer_question: timeout after %ds waiting for event_id=%s",
                    self._answer_timeout,
                    event_id,
                )
                self._log_pending_question_state(event_id, reason="wait ended without answer")

            with self._answer_lock:
                answer = self._pending_answers.pop(event_id, "No answer received")
                self._answer_events.pop(event_id, None)
                self._answer_targets.pop(event_id, None)
                self._pending_question_meta.pop(event_id, None)

            logger.info(
                "RemoteAgentAdapter: completed question "
                "event_id=%s question_id=%s target=%s elapsed=%.3fs answer_chars=%d",
                event_id,
                question_id,
                target_name,
                time.monotonic() - sent_at,
                len(answer),
            )

            return answer

    def _release_pending_answers_for_agent(self, agent_id: str, *, reason: str, detail: str = "") -> None:
        released_event_ids: list[str] = []
        with self._answer_lock:
            for event_id, target_name in list(self._answer_targets.items()):
                if target_name != agent_id:
                    continue
                answer_event = self._answer_events.get(event_id)
                if answer_event is None:
                    continue
                self._pending_answers.setdefault(event_id, "No answer received")
                answer_event.set()
                released_event_ids.append(event_id)

        if released_event_ids:
            logger.warning(
                "RemoteAgentAdapter: released %d pending question(s) for %s after %s (%s)",
                len(released_event_ids),
                agent_id,
                reason,
                detail or "no detail",
            )
            for event_id in released_event_ids:
                self._log_pending_question_state(
                    event_id,
                    reason=f"released after {reason}: {detail or 'no detail'}",
                )

    def _handle_agent_online(self, agent_id: str, boot_id: str = "") -> None:
        if not agent_id:
            return
        with self._online_lock:
            already_online = agent_id in self._online_agents
            previous_boot_id = self._agent_boot_ids.get(agent_id, "")
            self._online_agents.add(agent_id)
            if boot_id:
                self._agent_boot_ids[agent_id] = boot_id
            online_count = len(self._online_agents)
        logger.info(
            "RemoteAgentAdapter: AGENT_ONLINE from %s boot_id=%s (%d/%d)",
            agent_id,
            boot_id or "unknown",
            online_count,
            self._agent_count,
        )
        boot_changed = bool(boot_id and previous_boot_id and previous_boot_id != boot_id)
        if self._idle_wait_done.is_set() and already_online and (boot_changed or not boot_id):
            with self._ready_lock:
                was_ready = agent_id in self._ready_agents
                ready_boot_id = self._ready_boot_ids.get(agent_id, "")
                self._ready_agents.discard(agent_id)
                self._question_ineligible_agents.add(agent_id)
            logger.warning(
                "RemoteAgentAdapter: excluding %s from question targets after post-ready restart "
                "(was_ready=%s ready_boot_id=%s previous_boot_id=%s new_boot_id=%s)",
                agent_id,
                was_ready,
                ready_boot_id or "unknown",
                previous_boot_id or "unknown",
                boot_id or "unknown",
            )
        if already_online and (boot_changed or not boot_id):
            self._release_pending_answers_for_agent(
                agent_id,
                reason="agent restart",
                detail=(f"duplicate AGENT_ONLINE with boot_id={boot_id or 'unknown'} while question was pending"),
            )

    def _handle_agent_shutdown(self, agent_id: str, reason: str = "", detail: str = "") -> None:
        if not agent_id:
            return
        with self._online_lock:
            self._online_agents.discard(agent_id)
            self._agent_boot_ids.pop(agent_id, None)
        with self._ready_lock:
            self._ready_agents.discard(agent_id)
            self._ready_boot_ids.pop(agent_id, None)
            if self._idle_wait_done.is_set():
                self._question_ineligible_agents.add(agent_id)
        logger.warning(
            "RemoteAgentAdapter: AGENT_SHUTDOWN from %s reason=%s detail=%s",
            agent_id,
            reason,
            detail,
        )
        self._release_pending_answers_for_agent(
            agent_id,
            reason=f"agent shutdown:{reason or 'unknown'}",
            detail=detail,
        )

    def _question_attempt_targets(self, base_target_agent: int, max_attempts: int) -> list[int]:
        if max_attempts <= 1 or self._agents_per_app <= 1:
            return [(base_target_agent + attempt) % self._agent_count for attempt in range(max_attempts)]

        app_groups = [
            list(range(start, min(start + self._agents_per_app, self._agent_count)))
            for start in range(0, self._agent_count, self._agents_per_app)
        ]
        if not app_groups:
            return []

        app_count = len(app_groups)
        base_app = min(app_count - 1, base_target_agent // self._agents_per_app)
        base_slot = base_target_agent % self._agents_per_app
        ordered: list[int] = []
        seen: set[int] = set()

        def add_candidate(candidate: int) -> bool:
            if candidate in seen:
                return False
            seen.add(candidate)
            ordered.append(candidate)
            return len(ordered) >= max_attempts

        for app_offset in range(app_count):
            group = app_groups[(base_app + app_offset) % app_count]
            for slot_offset in range(self._agents_per_app):
                slot = (base_slot + slot_offset) % self._agents_per_app
                if slot >= len(group):
                    continue
                if add_candidate(group[slot]):
                    return ordered

        return ordered

    def _is_question_target_eligible(self, target_agent: int) -> bool:
        target_name = self._agent_name(target_agent)
        with self._ready_lock:
            if target_name in self._question_ineligible_agents:
                return False
            ready_agents = set(self._ready_agents)
        return not ready_agents or target_name in ready_agents

    def _dispatch_question_targets(self, base_target_agent: int, max_attempts: int) -> list[int]:
        ordered_candidates = self._question_attempt_targets(base_target_agent, self._agent_count)
        eligible_candidates = [
            candidate for candidate in ordered_candidates if self._is_question_target_eligible(candidate)
        ]
        if eligible_candidates:
            skipped = len(ordered_candidates) - len(eligible_candidates)
            if skipped > 0:
                logger.warning(
                    "RemoteAgentAdapter: filtered %d ineligible question target(s) after restart",
                    skipped,
                )
            return eligible_candidates[:max_attempts]
        return ordered_candidates[:max_attempts]

    def _record_listener_event(self, event_type: str, partition_id: str = "") -> None:
        self._last_listener_event_at = time.monotonic()
        self._last_listener_event_type = event_type
        self._last_listener_partition = partition_id

    def _listener_thread_is_alive(self) -> bool:
        thread = getattr(self, "_listener_thread", None)
        is_alive = getattr(thread, "is_alive", None)
        if not callable(is_alive):
            return False
        try:
            return bool(is_alive())
        except Exception:
            return False

    def _log_pending_question_state(self, event_id: str, *, reason: str) -> None:
        with self._answer_lock:
            meta = dict(self._pending_question_meta.get(event_id, {}))
            target_name = self._answer_targets.get(event_id, "")

        if not meta and not target_name:
            return

        now = time.monotonic()
        sent_at = float(meta.get("sent_at_monotonic", now))
        question_age = max(0.0, now - sent_at)
        listener_age = max(0.0, now - self._last_listener_event_at) if self._last_listener_event_at else -1.0

        logger.warning(
            "RemoteAgentAdapter: pending question event_id=%s question_id=%s target=%s "
            "partition=%s age=%.1fs listener_alive=%s listener_thread_alive=%s "
            "listener_restarts=%d last_listener_event_type=%s last_listener_partition=%s "
            "last_listener_event_age=%.1fs target_boot_id=%s reason=%s preview=%s",
            event_id,
            str(meta.get("question_id", "")) or "unknown",
            target_name or "unknown",
            str(meta.get("partition_id", "")) or "unknown",
            question_age,
            self._listener_alive.is_set(),
            self._listener_thread_is_alive(),
            self._listener_restart_count,
            self._last_listener_event_type or "none",
            self._last_listener_partition or "unknown",
            listener_age,
            self._agent_boot_ids.get(target_name, "") or "unknown",
            reason,
            str(meta.get("question_preview", "")),
        )

    def _wait_for_answer_event(self, event_id: str, answer_event: threading.Event) -> bool:
        if self._answer_timeout > 0:
            deadline = time.monotonic() + self._answer_timeout
            log_interval = min(30.0, max(1.0, float(self._answer_timeout)))
            while not self._shutdown.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                if answer_event.wait(timeout=min(log_interval, remaining)):
                    return True
                self._log_pending_question_state(event_id, reason="still waiting")
            return False

        while not self._shutdown.is_set():
            if answer_event.wait(timeout=30.0):
                return True
            self._log_pending_question_state(event_id, reason="still waiting (no timeout)")
        return False

    def answer_question(self, question: str, target_agent: int | None = None) -> str:
        """Send question to one agent, retrying on other agents when configured."""
        with start_span(
            "azure_eval.answer_question",
            tracer_name=__name__,
            attributes=self._span_attributes(
                question_length=len(question),
                target_agent=(self._agent_name(target_agent) if target_agent is not None else "auto-round-robin"),
            ),
        ):
            if self._learn_count > 0 and not self._idle_wait_done.is_set():
                with self._counter_lock:
                    if not self._idle_wait_done.is_set():
                        self._wait_for_agents_idle()
                        self._idle_wait_done.set()

            with self._counter_lock:
                if target_agent is None:
                    base_target_agent = self._question_count % self._agent_count
                else:
                    base_target_agent = target_agent % self._agent_count
                self._question_count += 1

            max_attempts = min(self._agent_count, 1 + self._question_failover_retries)
            attempt_targets = self._dispatch_question_targets(base_target_agent, max_attempts)
            last_answer = "No answer received"
            for attempt, attempt_target in enumerate(attempt_targets):
                if attempt > 0:
                    logger.info(
                        "RemoteAgentAdapter: retrying question on %s after previous timeout/no-answer",
                        self._agent_name(attempt_target),
                    )
                answer = self._send_question_to_agent(question, attempt_target)
                if not self._answer_requires_failover(answer):
                    return answer
                if attempt + 1 < len(attempt_targets):
                    logger.info(
                        "RemoteAgentAdapter: retrying question on %s after incomplete answer from %s: %s",
                        self._agent_name(attempt_targets[attempt + 1]),
                        self._agent_name(attempt_target),
                        answer[:120],
                    )
                last_answer = answer

            return last_answer

    @staticmethod
    def _answer_requires_failover(answer: str) -> bool:
        normalized = answer.strip().lower()
        return any(normalized.startswith(prefix) for prefix in _FAILOVER_ANSWER_PREFIXES)

    def _wait_for_agents_idle(self) -> None:
        """Wait for all agents to finish processing content.

        Sends FEED_COMPLETE to every agent, then waits for each to publish
        AGENT_READY on the eval-responses hub.  Event-driven — no polling.
        """
        with start_span(
            "azure_eval.wait_for_agents_idle",
            tracer_name=__name__,
            attributes=self._span_attributes(),
        ):
            min_turns = min(self._learn_turn_counts, default=0)
            max_turns = max(self._learn_turn_counts, default=0)
            if min_turns == max_turns:
                turns_summary = f"{max_turns} content turns each"
            else:
                turns_summary = f"{min_turns}-{max_turns} content turns per agent"
            logger.info(
                "Sending FEED_COMPLETE to all %d agents (%s)...",
                self._agent_count,
                turns_summary,
            )

            with self._ready_lock:
                self._ready_agents.clear()
                self._ready_boot_ids.clear()
                self._question_ineligible_agents.clear()
                self._all_agents_ready.clear()

            logger.info(
                "Waiting for %d AGENT_READY events on '%s'...",
                self._agent_count,
                self._response_hub,
            )

            poll_interval = 15
            while True:
                with self._ready_lock:
                    missing_agents = [
                        f"agent-{i}" for i in range(self._agent_count) if f"agent-{i}" not in self._ready_agents
                    ]
                    ready_count = self._agent_count - len(missing_agents)
                if ready_count >= self._agent_count:
                    logger.info("All %d agents ready. Starting question phase.", self._agent_count)
                    return

                for target_name in missing_agents:
                    target_agent = self._agent_index(target_name)
                    total_turns = (
                        self._learn_turn_counts[target_agent]
                        if 0 <= target_agent < len(self._learn_turn_counts)
                        else self._learn_count // max(1, self._agent_count)
                    )
                    if self._replicate_learning_to_all_agents:
                        feed_payload: dict[str, Any] = {
                            "total_turns": total_turns,
                            "target_agent": target_name,
                            "expected_fact_batches": total_turns,
                        }
                    else:
                        feed_payload = {
                            "total_turns": total_turns,
                            "target_agent": target_name,
                            "expected_learn_content": total_turns,
                        }
                    self._publish_event(
                        {
                            "event_type": "FEED_COMPLETE",
                            "event_id": uuid.uuid4().hex[:12],
                            "target_agent": target_name,
                            "source_agent": "eval-harness",
                            "payload": feed_payload,
                        },
                        partition_key=target_name,
                    )

                logger.info(
                    "  %d/%d agents ready, re-sent FEED_COMPLETE to: %s",
                    ready_count,
                    self._agent_count,
                    ", ".join(missing_agents),
                )
                time.sleep(poll_interval)

    def _listen_for_answers(self) -> None:
        """Background thread: collect eval lifecycle and answer events."""
        try:
            from azure.eventhub import EventHubConsumerClient  # type: ignore[import-unresolved]
        except ImportError:
            logger.error("azure-eventhub not installed — RemoteAgentAdapter cannot receive answers")
            return

        starting_position = "@latest"
        while not self._shutdown.is_set():
            consumer = EventHubConsumerClient.from_connection_string(
                self._connection_string,
                consumer_group="eval-reader",
                eventhub_name=self._response_hub,
            )
            try:
                expected_partitions = set(consumer.get_partition_ids())
            except Exception:
                logger.debug("Failed to enumerate response hub partitions", exc_info=True)
                expected_partitions = set()

            initialized_partitions: set[str] = set()
            initialized_lock = threading.Lock()

            def _on_event(partition_context: Any, event: Any) -> None:
                if event is None:
                    return
                partition_id = str(getattr(partition_context, "partition_id", ""))
                try:
                    body = json.loads(event.body_as_str())
                    event_type = body.get("event_type", "")
                    self._record_listener_event(event_type or "UNKNOWN", partition_id)

                    # Filter stale events from previous eval runs
                    run_id = body.get("run_id", "")
                    if run_id and run_id != self._run_id:
                        return

                    if event_type == "AGENT_ONLINE":
                        agent_id = body.get("agent_id", "")
                        boot_id = body.get("boot_id", "")
                        self._handle_agent_online(agent_id, boot_id=boot_id)
                        if hasattr(partition_context, "update_checkpoint"):
                            partition_context.update_checkpoint(event)
                        return

                    if event_type == "AGENT_READY":
                        agent_id = body.get("agent_id", "")
                        boot_id = body.get("boot_id", "")
                        if agent_id:
                            with self._ready_lock:
                                self._ready_agents.add(agent_id)
                                if boot_id:
                                    self._ready_boot_ids[agent_id] = boot_id
                                if not self._idle_wait_done.is_set():
                                    self._question_ineligible_agents.discard(agent_id)
                                ready_count = len(self._ready_agents)
                            logger.info(
                                "RemoteAgentAdapter: AGENT_READY from %s boot_id=%s partition=%s (%d/%d)",
                                agent_id,
                                boot_id or "unknown",
                                partition_id or "unknown",
                                ready_count,
                                self._agent_count,
                            )
                        if hasattr(partition_context, "update_checkpoint"):
                            partition_context.update_checkpoint(event)
                        return

                    if event_type == "AGENT_PROGRESS":
                        agent_id = body.get("agent_id", "")
                        phase = body.get("phase", "")
                        processed_count = int(body.get("processed_count", 0) or 0)
                        if agent_id:
                            with self._progress_lock:
                                self._progress_agents.add(agent_id)
                                self._progress_counts[agent_id] = max(
                                    processed_count,
                                    self._progress_counts.get(agent_id, 0),
                                )
                                progress_count = len(self._progress_agents)
                            logger.info(
                                "RemoteAgentAdapter: AGENT_PROGRESS from %s "
                                "phase=%s count=%d boot_id=%s partition=%s (%d/%d)",
                                agent_id,
                                phase or "unknown",
                                processed_count,
                                body.get("boot_id", "") or "unknown",
                                partition_id or "unknown",
                                progress_count,
                                self._agent_count,
                            )
                        if hasattr(partition_context, "update_checkpoint"):
                            partition_context.update_checkpoint(event)
                        return

                    if event_type == "EVAL_ANSWER":
                        event_id = body.get("event_id", "")
                        answer = body.get("answer", "")

                        with self._answer_lock:
                            if event_id in self._answer_events:
                                self._pending_answers[event_id] = answer
                                self._answer_events[event_id].set()
                                logger.info(
                                    "RemoteAgentAdapter: got answer event_id=%s "
                                    "question_id=%s from %s boot_id=%s partition=%s: %s",
                                    event_id,
                                    body.get("question_id", "") or "unknown",
                                    body.get("agent_id", "?"),
                                    body.get("boot_id", "") or "unknown",
                                    partition_id or "unknown",
                                    answer[:80] if answer else "(empty)",
                                )
                            else:
                                logger.warning(
                                    "RemoteAgentAdapter: answer for unknown "
                                    "event_id=%s question_id=%s partition=%s (stale?)",
                                    event_id,
                                    body.get("question_id", "") or "unknown",
                                    partition_id or "unknown",
                                )
                        if hasattr(partition_context, "update_checkpoint"):
                            partition_context.update_checkpoint(event)
                        return

                    if event_type == "AGENT_SHUTDOWN":
                        self._handle_agent_shutdown(
                            body.get("agent_id", ""),
                            reason=body.get("reason", ""),
                            detail=body.get("detail", ""),
                        )

                    if hasattr(partition_context, "update_checkpoint"):
                        partition_context.update_checkpoint(event)
                except Exception:
                    logger.debug("Failed to parse response message", exc_info=True)

            def _on_partition_initialize(partition_context: Any) -> None:
                partition_id = str(getattr(partition_context, "partition_id", ""))
                should_mark_alive = False
                with initialized_lock:
                    if partition_id:
                        initialized_partitions.add(partition_id)
                    if expected_partitions:
                        should_mark_alive = initialized_partitions >= expected_partitions
                    else:
                        should_mark_alive = True
                if should_mark_alive and not self._listener_alive.is_set():
                    self._listener_alive.set()
                    logger.info(
                        "RemoteAgentAdapter: listening on '%s' (eval-reader, partitions=%d, starting_position=%s)",
                        self._response_hub,
                        len(initialized_partitions),
                        starting_position,
                    )

            try:
                consumer.receive(
                    on_event=_on_event,
                    on_partition_initialize=_on_partition_initialize,
                    starting_position=starting_position,
                )
                if self._shutdown.is_set():
                    break
                self._listener_alive.clear()
                self._listener_restart_count += 1
                logger.warning(
                    "RemoteAgentAdapter: response listener returned unexpectedly; "
                    "reconnecting with backfill (restart=%d)",
                    self._listener_restart_count,
                )
            except Exception:
                if self._shutdown.is_set():
                    break
                self._listener_alive.clear()
                self._listener_restart_count += 1
                logger.warning(
                    "RemoteAgentAdapter: response listener failed; reconnecting with backfill (restart=%d)",
                    self._listener_restart_count,
                    exc_info=True,
                )
            finally:
                try:
                    consumer.close()
                except Exception:
                    pass

            with self._answer_lock:
                pending_event_ids = list(self._answer_targets.keys())
            for event_id in pending_event_ids:
                self._log_pending_question_state(
                    event_id,
                    reason="response listener reconnecting with backfill",
                )

            if self._shutdown.is_set():
                break
            starting_position = "-1"
            time.sleep(1.0)

    def get_memory_stats(self) -> dict[str, Any]:
        """Return adapter stats."""
        return {
            "adapter": "remote",
            "learn_count": self._learn_count,
            "question_count": self._question_count,
            "agent_count": self._agent_count,
        }

    def close(self) -> None:
        """Clean up Event Hubs connections."""
        self._shutdown.set()
        self._reset_producer()
        extractor = self._fact_batch_extractor
        extractor_dir = self._fact_batch_extractor_dir
        self._fact_batch_extractor = None
        self._fact_batch_extractor_dir = None
        if extractor is not None:
            try:
                extractor.close()
            except Exception:
                logger.debug("Failed to close fact batch extractor cleanly", exc_info=True)
        if extractor_dir is not None:
            shutil.rmtree(extractor_dir, ignore_errors=True)
        if self._listener_thread.is_alive():
            self._listener_thread.join(timeout=5)
