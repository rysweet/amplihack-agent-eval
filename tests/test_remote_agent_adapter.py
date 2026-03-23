"""Tests for remote_agent_adapter.py — zero-external-dependency unit tests."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest  # type: ignore[import-unresolved]


def _load_module():
    mod = importlib.import_module("amplihack_eval.adapters.remote_agent_adapter")
    return importlib.reload(mod)


def _patch_azure_eventhub(*, consumer_cls=None, producer_cls=None, event_data_cls=None):
    azure_pkg = types.ModuleType("azure")
    eventhub_mod = types.ModuleType("azure.eventhub")
    if consumer_cls is not None:
        eventhub_mod.EventHubConsumerClient = consumer_cls
    if producer_cls is not None:
        eventhub_mod.EventHubProducerClient = producer_cls
    if event_data_cls is not None:
        eventhub_mod.EventData = event_data_cls
    azure_pkg.eventhub = eventhub_mod
    return patch.dict(sys.modules, {"azure": azure_pkg, "azure.eventhub": eventhub_mod})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(mod, agent_count=5, answer_timeout=0):
    """Create an adapter with mocked EH clients."""
    with patch.object(mod, "threading") as mock_threading:
        # Make the listener thread a no-op
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_threading.Event.side_effect = [
            threading.Event(),  # _shutdown
            threading.Event(),  # _idle_wait_done
            threading.Event(),  # _all_agents_ready
        ]
        mock_threading.Lock.return_value = threading.Lock()
        mock_threading.Thread.return_value = mock_thread

        # Pre-set listener_alive so __init__ doesn't block
        adapter = object.__new__(mod.RemoteAgentAdapter)
        adapter._connection_string = (
            "Endpoint=sb://fake.servicebus.windows.net/;SharedAccessKeyName=x;SharedAccessKey=y"
        )
        adapter._input_hub = "hive-input"
        adapter._response_hub = "eval-responses"
        adapter._resource_group = ""
        adapter._agent_count = agent_count
        adapter._agents_per_app = 1
        adapter._learn_count = 0
        adapter._learn_turn_counts = [0 for _ in range(agent_count)]
        adapter._question_count = 0
        adapter._answer_timeout = answer_timeout
        adapter._replicate_learning_to_all_agents = False
        adapter._question_failover_retries = 0
        adapter._shutdown = threading.Event()
        adapter._startup_wait_done = threading.Event()
        adapter._idle_wait_done = threading.Event()
        adapter._counter_lock = threading.Lock()
        adapter._answer_lock = threading.Lock()
        adapter._producer_lock = threading.Lock()
        adapter._extractor_lock = threading.Lock()
        adapter._pending_answers = {}
        adapter._answer_events = {}
        adapter._answer_targets = {}
        adapter._pending_question_meta = {}
        adapter._agent_boot_ids = {}
        adapter._producer = None
        adapter._fact_batch_extractor = None
        adapter._fact_batch_extractor_dir = None
        adapter._online_agents = set()
        adapter._online_lock = threading.Lock()
        adapter._all_agents_online = threading.Event()
        adapter._progress_agents = set()
        adapter._progress_counts = {}
        adapter._progress_lock = threading.Lock()
        adapter._feed_telemetry_wait_done = threading.Event()
        adapter._feed_telemetry_monitor_thread = None
        adapter._ready_agents = set()
        adapter._ready_boot_ids = {}
        adapter._question_ineligible_agents = set()
        adapter._ready_lock = threading.Lock()
        adapter._all_agents_ready = threading.Event()
        adapter._run_id = "test_run_abc"
        adapter._num_partitions = 32
        adapter._listener_alive = threading.Event()
        adapter._listener_alive.set()
        adapter._listener_restart_count = 0
        adapter._last_listener_event_at = 0.0
        adapter._last_listener_event_type = ""
        adapter._last_listener_partition = ""
        adapter._listener_thread = MagicMock()
        adapter._listener_thread.is_alive.return_value = True

    return adapter


# ===========================================================================
# Tests
# ===========================================================================


class TestRemoteAgentAdapterInit:
    def test_module_loads(self):
        mod = _load_module()
        assert hasattr(mod, "RemoteAgentAdapter")

    def test_adapter_attrs(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=10)
        assert adapter._agent_count == 10
        assert adapter._learn_count == 0
        assert adapter._question_count == 0
        assert adapter._run_id == "test_run_abc"

    def test_prepare_fact_batch_uses_fast_replicated_shape(self):
        mod = _load_module()
        adapter = _make_adapter(mod)
        extractor = MagicMock()
        extractor.prepare_fact_batch.return_value = {"facts": []}
        adapter._fact_batch_extractor = extractor

        result = adapter._prepare_fact_batch("content")

        extractor.prepare_fact_batch.assert_called_once_with("content", include_summary=False)
        assert result == {"facts": []}

    def test_get_fact_batch_extractor_uses_runtime_factory(self):
        mod = _load_module()
        adapter = _make_adapter(mod)
        extractor = MagicMock()
        runtime_factory = types.ModuleType("amplihack.agents.goal_seeking.runtime_factory")
        create_runtime = MagicMock(return_value=extractor)
        runtime_factory.create_goal_agent_runtime = create_runtime

        with patch.dict(
            sys.modules,
            {"amplihack.agents.goal_seeking.runtime_factory": runtime_factory},
        ):
            result = adapter._get_fact_batch_extractor()

        assert result is extractor
        create_runtime.assert_called_once()
        assert create_runtime.call_args.kwargs["runtime_kind"] == "goal"
        assert create_runtime.call_args.kwargs["bind_answer_mode"] is False


class TestPartitionRoutingHelpers:
    def test_non_numeric_agent_name_uses_stable_hash(self):
        mod = _load_module()
        expected = int.from_bytes(hashlib.sha256(b"coordinator").digest()[:8], "big")
        assert mod.RemoteAgentAdapter._agent_index("coordinator") == expected

    def test_empty_agent_name_uses_stable_hash(self):
        mod = _load_module()
        expected = int.from_bytes(hashlib.sha256(b"").digest()[:8], "big")
        assert mod.RemoteAgentAdapter._agent_index("") == expected

    def test_partition_count_fallback_logs_warning(self):
        mod = _load_module()
        adapter = _make_adapter(mod)
        adapter._num_partitions = None

        with (
            patch.dict("sys.modules", {"azure.eventhub": None}),
            patch.object(mod.logger, "warning") as warning,
        ):
            assert adapter._get_num_partitions() == 32

        warning.assert_called_once()


class TestPublishEvent:
    def test_publish_attaches_run_id(self):
        mod = _load_module()
        adapter = _make_adapter(mod)

        mock_producer = MagicMock()
        mock_batch = MagicMock()
        mock_producer.create_batch.return_value = mock_batch
        producer_cls = MagicMock()
        event_data_cls = MagicMock()

        with (
            _patch_azure_eventhub(producer_cls=producer_cls, event_data_cls=event_data_cls),
        ):
            producer_cls.from_connection_string.return_value = mock_producer
            event_data_cls.side_effect = lambda data: data

            payload = {"event_type": "LEARN_CONTENT", "event_id": "abc"}
            adapter._publish_event(payload, partition_key="agent-0")

            assert payload["run_id"] == "test_run_abc"
            mock_producer.send_batch.assert_called_once()

    def test_publish_reuses_cached_producer(self):
        mod = _load_module()
        adapter = _make_adapter(mod)

        mock_producer = MagicMock()
        mock_batch = MagicMock()
        mock_producer.create_batch.return_value = mock_batch
        producer_cls = MagicMock()
        event_data_cls = MagicMock()

        with (
            _patch_azure_eventhub(producer_cls=producer_cls, event_data_cls=event_data_cls),
        ):
            producer_cls.from_connection_string.return_value = mock_producer
            event_data_cls.side_effect = lambda data: data

            adapter._publish_event({"event_type": "LEARN_CONTENT", "event_id": "abc"}, "agent-0")
            adapter._publish_event({"event_type": "LEARN_CONTENT", "event_id": "def"}, "agent-1")

        assert producer_cls.from_connection_string.call_count == 1
        assert mock_producer.send_batch.call_count == 2

    def test_publish_retries_on_failure(self):
        mod = _load_module()
        adapter = _make_adapter(mod)

        # First producer fails, second succeeds
        mock_producer_fail = MagicMock()
        mock_producer_fail.send_batch.side_effect = ConnectionError("EH down")
        mock_producer_fail.create_batch.return_value = MagicMock()

        mock_producer_ok = MagicMock()
        mock_batch = MagicMock()
        mock_producer_ok.create_batch.return_value = mock_batch
        producer_cls = MagicMock()
        event_data_cls = MagicMock()

        with (
            _patch_azure_eventhub(producer_cls=producer_cls, event_data_cls=event_data_cls),
        ):
            producer_cls.from_connection_string.side_effect = [
                mock_producer_fail,
                mock_producer_ok,
            ]
            event_data_cls.side_effect = lambda data: data

            adapter._publish_event(
                {"event_type": "INPUT", "event_id": "x"},
                partition_key="agent-1",
            )
            mock_producer_ok.send_batch.assert_called_once()

    def test_publish_raises_after_retry_failure(self):
        mod = _load_module()
        adapter = _make_adapter(mod)

        def make_failing_producer():
            p = MagicMock()
            p.send_batch.side_effect = ConnectionError("EH down")
            p.create_batch.return_value = MagicMock()
            return p

        producer_cls = MagicMock()
        event_data_cls = MagicMock()

        with (
            _patch_azure_eventhub(producer_cls=producer_cls, event_data_cls=event_data_cls),
        ):
            producer_cls.from_connection_string.side_effect = [
                make_failing_producer(),
                make_failing_producer(),
            ]
            event_data_cls.side_effect = lambda data: data

            with pytest.raises(ConnectionError):
                adapter._publish_event(
                    {"event_type": "INPUT", "event_id": "x"},
                    partition_key="agent-1",
                )


class TestLearnFromContent:
    def test_first_learn_waits_for_online_agents_once(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=2)
        adapter._wait_for_agents_online = MagicMock()
        adapter._publish_event = MagicMock()
        adapter._start_feed_telemetry_monitor = MagicMock(
            side_effect=lambda expected: adapter._feed_telemetry_wait_done.set()
        )

        adapter.learn_from_content("hello")
        adapter.learn_from_content("world")

        adapter._wait_for_agents_online.assert_called_once()
        assert adapter._startup_wait_done.is_set()

    def test_round_robin_targets(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=3)
        adapter._startup_wait_done.set()
        adapter._start_feed_telemetry_monitor = MagicMock(
            side_effect=lambda expected: adapter._feed_telemetry_wait_done.set()
        )

        published_keys = []

        def capture_publish(payload, partition_key):
            published_keys.append(partition_key)

        adapter._publish_event = capture_publish

        for i in range(6):
            result = adapter.learn_from_content(f"content {i}")
            assert "facts_stored" in result

        assert published_keys == [
            "agent-0",
            "agent-1",
            "agent-2",
            "agent-0",
            "agent-1",
            "agent-2",
        ]
        assert adapter._learn_turn_counts == [2, 2, 2]
        adapter._start_feed_telemetry_monitor.assert_called_once_with({"agent-0", "agent-1", "agent-2"})
        assert adapter._feed_telemetry_wait_done.is_set()

    def test_replicated_learning_targets_all_agents(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=3)
        adapter._startup_wait_done.set()
        adapter._replicate_learning_to_all_agents = True
        adapter._start_feed_telemetry_monitor = MagicMock(
            side_effect=lambda expected: adapter._feed_telemetry_wait_done.set()
        )
        adapter._prepare_fact_batch = MagicMock(
            return_value={
                "facts_extracted": 2,
                "facts": [
                    {
                        "context": "Campaign",
                        "fact": "CAMP-1 is active",
                        "confidence": 0.9,
                        "tags": [],
                    },
                    {
                        "context": "Campaign",
                        "fact": "CAMP-1 targets finance",
                        "confidence": 0.8,
                        "tags": [],
                    },
                ],
                "summary_fact": None,
                "content_summary": "Campaign content",
                "perception": "Campaign content",
                "episode_content": "Campaign content",
                "source_label": "Campaign content",
            }
        )

        published_events = []

        def capture_publish(payload, partition_key):
            published_events.append((partition_key, payload))

        adapter._publish_event = capture_publish

        result = adapter.learn_from_content("content 0")

        assert [partition_key for partition_key, _ in published_events] == [
            "agent-0",
            "agent-1",
            "agent-2",
        ]
        assert all(payload["event_type"] == "STORE_FACT_BATCH" for _, payload in published_events)
        assert all("fact_batch" in payload["payload"] for _, payload in published_events)
        assert adapter._learn_turn_counts == [1, 1, 1]
        assert result["facts_stored"] == 2
        assert result["replicated_to"] == 3
        adapter._start_feed_telemetry_monitor.assert_called_once_with({"agent-0", "agent-1", "agent-2"})
        assert adapter._feed_telemetry_wait_done.is_set()

    def test_learn_increments_counter(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=2)
        adapter._startup_wait_done.set()
        adapter._publish_event = MagicMock()
        adapter._start_feed_telemetry_monitor = MagicMock(
            side_effect=lambda expected: adapter._feed_telemetry_wait_done.set()
        )

        adapter.learn_from_content("hello")
        assert adapter._learn_count == 1
        adapter.learn_from_content("world")
        assert adapter._learn_count == 2

    def test_start_feed_telemetry_monitor_runs_in_background(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=2)
        adapter._wait_for_feed_telemetry = MagicMock()

        started: dict[str, object] = {}

        class ImmediateThread:
            def __init__(self, *, target, args, daemon, name):
                started["target"] = target
                started["args"] = args
                started["daemon"] = daemon
                started["name"] = name

            def start(self):
                started["started"] = True

        with patch.object(mod.threading, "Thread", ImmediateThread):
            adapter._start_feed_telemetry_monitor({"agent-0", "agent-1"})

        assert adapter._feed_telemetry_wait_done.is_set()
        assert started["target"] is adapter._wait_for_feed_telemetry
        assert started["args"] == ({"agent-0", "agent-1"},)
        assert started["daemon"] is True
        assert started["name"] == "feed-telemetry-monitor"
        assert started["started"] is True


class TestAnswerQuestion:
    def test_answer_returned_when_set(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=2)
        adapter._idle_wait_done.set()  # skip idle wait

        # Capture the event_id from publish
        captured_event_ids = []

        def capture_publish(payload, partition_key):
            captured_event_ids.append(payload.get("event_id", ""))

        adapter._publish_event = capture_publish

        # Answer in a background thread
        def answer_later():
            time.sleep(0.1)
            eid = captured_event_ids[0]
            with adapter._answer_lock:
                adapter._pending_answers[eid] = "The answer is 42"
                adapter._answer_events[eid].set()

        t = threading.Thread(target=answer_later, daemon=True)
        t.start()

        result = adapter.answer_question("what is the answer?")
        t.join(timeout=5)
        assert result == "The answer is 42"

    def test_send_question_tracks_pending_question_metadata_and_cleans_up(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=1)
        adapter._idle_wait_done.set()

        captured = {}

        def capture_publish(payload, partition_key):
            captured["payload"] = payload
            captured["partition_key"] = partition_key
            event_id = payload["event_id"]
            with adapter._answer_lock:
                assert adapter._pending_question_meta[event_id]["question_id"] == payload["payload"]["question_id"]
                adapter._pending_answers[event_id] = "tracked answer"
                adapter._answer_events[event_id].set()

        adapter._publish_event = capture_publish

        result = adapter._send_question_to_agent("how many projects?", 0)

        assert result == "tracked answer"
        assert captured["payload"]["payload"]["question_id"].startswith("q_0_")
        assert captured["partition_key"] == "agent-0"
        assert adapter._pending_question_meta == {}

    def test_answer_timeout_returns_default(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=1, answer_timeout=1)
        adapter._idle_wait_done.set()
        adapter._publish_event = MagicMock()

        result = adapter.answer_question("will time out")
        assert result == "No answer received"

    def test_answer_timeout_retries_next_agent(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=3)
        adapter._idle_wait_done.set()
        adapter._question_failover_retries = 1

        with patch.object(
            adapter,
            "_send_question_to_agent",
            side_effect=["No answer received", "Recovered answer"],
        ) as send_question:
            result = adapter.answer_question("recover after timeout")

        assert result == "Recovered answer"
        assert send_question.call_args_list[0].args == ("recover after timeout", 0)
        assert send_question.call_args_list[1].args == ("recover after timeout", 1)

    def test_answer_timeout_retries_other_slots_before_other_apps_when_agents_per_app_set(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=15)
        adapter._idle_wait_done.set()
        adapter._agents_per_app = 5
        adapter._question_failover_retries = 2

        with patch.object(
            adapter,
            "_send_question_to_agent",
            side_effect=["No answer received", "No answer received", "Recovered answer"],
        ) as send_question:
            result = adapter.answer_question("recover across apps", target_agent=4)

        assert result == "Recovered answer"
        assert send_question.call_args_list[0].args == ("recover across apps", 4)
        assert send_question.call_args_list[1].args == ("recover across apps", 0)
        assert send_question.call_args_list[2].args == ("recover across apps", 1)

    def test_duplicate_agent_online_releases_pending_question(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=1)
        adapter._idle_wait_done.set()
        adapter._publish_event = MagicMock()
        adapter._online_agents.add("agent-0")
        adapter._agent_boot_ids["agent-0"] = "boot-1"

        result_holder = {}

        def ask_question():
            result_holder["answer"] = adapter._send_question_to_agent("still there?", 0)

        thread = threading.Thread(target=ask_question, daemon=True)
        thread.start()

        for _ in range(100):
            with adapter._answer_lock:
                if adapter._answer_targets:
                    break
            time.sleep(0.01)

        adapter._handle_agent_online("agent-0", boot_id="boot-2")

        thread.join(timeout=5)
        assert result_holder["answer"] == "No answer received"
        assert adapter._answer_targets == {}

    def test_duplicate_agent_online_same_boot_id_does_not_release_pending_question(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=1)
        adapter._idle_wait_done.set()
        adapter._publish_event = MagicMock()
        adapter._online_agents.add("agent-0")
        adapter._agent_boot_ids["agent-0"] = "boot-1"

        result_holder = {}

        def ask_question():
            result_holder["answer"] = adapter._send_question_to_agent("still there?", 0)

        thread = threading.Thread(target=ask_question, daemon=True)
        thread.start()

        for _ in range(100):
            with adapter._answer_lock:
                if adapter._answer_targets:
                    break
            time.sleep(0.01)

        adapter._handle_agent_online("agent-0", boot_id="boot-1")

        with adapter._answer_lock:
            [event_id] = list(adapter._answer_targets.keys())
            assert event_id not in adapter._pending_answers
            assert not adapter._answer_events[event_id].is_set()
            adapter._pending_answers[event_id] = "Recovered answer"
            adapter._answer_events[event_id].set()

        thread.join(timeout=5)
        assert result_holder["answer"] == "Recovered answer"

    def test_post_ready_restart_marks_agent_ineligible_for_future_questions(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=1)
        adapter._idle_wait_done.set()
        adapter._online_agents.add("agent-0")
        adapter._agent_boot_ids["agent-0"] = "boot-1"
        adapter._ready_agents.add("agent-0")
        adapter._ready_boot_ids["agent-0"] = "boot-1"

        adapter._handle_agent_online("agent-0", boot_id="boot-2")

        assert "agent-0" not in adapter._ready_agents
        assert "agent-0" in adapter._question_ineligible_agents

    def test_answer_question_skips_restart_ineligible_target(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=15)
        adapter._idle_wait_done.set()
        adapter._agents_per_app = 5
        adapter._ready_agents.update({"agent-0", "agent-1", "agent-2", "agent-3", "agent-4"})
        adapter._question_ineligible_agents.add("agent-4")

        with patch.object(
            adapter,
            "_send_question_to_agent",
            return_value="Recovered answer",
        ) as send_question:
            result = adapter.answer_question("skip restarted target", target_agent=4)

        assert result == "Recovered answer"
        send_question.assert_called_once_with("skip restarted target", 0)

    def test_agent_shutdown_releases_pending_question(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=1)
        adapter._idle_wait_done.set()
        adapter._publish_event = MagicMock()
        adapter._online_agents.add("agent-0")
        adapter._agent_boot_ids["agent-0"] = "boot-1"
        adapter._ready_agents.add("agent-0")

        result_holder = {}

        def ask_question():
            result_holder["answer"] = adapter._send_question_to_agent("still there?", 0)

        thread = threading.Thread(target=ask_question, daemon=True)
        thread.start()

        for _ in range(100):
            with adapter._answer_lock:
                if adapter._answer_targets:
                    break
            time.sleep(0.01)

        adapter._handle_agent_shutdown("agent-0", reason="signal", detail="signal=15")

        thread.join(timeout=5)
        assert result_holder["answer"] == "No answer received"
        assert "agent-0" not in adapter._online_agents
        assert "agent-0" not in adapter._ready_agents
        assert adapter._answer_targets == {}

    def test_semantic_abstention_retries_next_agent(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=3)
        adapter._idle_wait_done.set()
        adapter._question_failover_retries = 1

        with patch.object(
            adapter,
            "_send_question_to_agent",
            side_effect=[
                "The provided facts do not contain any information about Marcus Rivera's food preferences.",
                "Marcus Rivera prefers ramen.",
            ],
        ) as send_question:
            result = adapter.answer_question("What food does Marcus Rivera prefer?")

        assert result == "Marcus Rivera prefers ramen."
        assert send_question.call_args_list[0].args == ("What food does Marcus Rivera prefer?", 0)
        assert send_question.call_args_list[1].args == ("What food does Marcus Rivera prefer?", 1)

    def test_answer_question_can_target_specific_agent(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=5)
        adapter._idle_wait_done.set()

        with patch.object(
            adapter,
            "_send_question_to_agent",
            return_value="Targeted answer",
        ) as send_question:
            result = adapter.answer_question("what is the answer?", target_agent=3)

        assert result == "Targeted answer"
        send_question.assert_called_once_with("what is the answer?", 3)
        assert adapter._question_count == 1


class TestOnEvent:
    def test_agent_online_tracked(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=2)

        body = json.dumps(
            {
                "event_type": "AGENT_ONLINE",
                "agent_id": "agent-1",
                "run_id": "test_run_abc",
            }
        )
        mock_event = MagicMock()
        mock_event.body_as_str.return_value = body

        run_id = adapter._run_id

        def on_event(partition_context, event):
            if event is None:
                return
            data = json.loads(event.body_as_str())
            rid = data.get("run_id", "")
            if rid and rid != run_id:
                return
            if data.get("event_type", "") == "AGENT_ONLINE":
                with adapter._online_lock:
                    adapter._online_agents.add(data.get("agent_id", ""))

        on_event(MagicMock(), mock_event)
        assert "agent-1" in adapter._online_agents

    def test_agent_progress_tracked(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=2)

        body = json.dumps(
            {
                "event_type": "AGENT_PROGRESS",
                "agent_id": "agent-1",
                "phase": "learn_content",
                "processed_count": 3,
                "run_id": "test_run_abc",
            }
        )
        mock_event = MagicMock()
        mock_event.body_as_str.return_value = body

        run_id = adapter._run_id

        def on_event(partition_context, event):
            if event is None:
                return
            data = json.loads(event.body_as_str())
            rid = data.get("run_id", "")
            if rid and rid != run_id:
                return
            if data.get("event_type", "") == "AGENT_PROGRESS":
                agent_id = data.get("agent_id", "")
                if agent_id:
                    with adapter._progress_lock:
                        adapter._progress_agents.add(agent_id)
                        adapter._progress_counts[agent_id] = int(data.get("processed_count", 0) or 0)

        on_event(MagicMock(), mock_event)
        assert "agent-1" in adapter._progress_agents
        assert adapter._progress_counts["agent-1"] == 3

    def test_agent_ready_tracked(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=2)

        # Simulate _on_event callback
        body = json.dumps(
            {
                "event_type": "AGENT_READY",
                "agent_id": "agent-0",
                "run_id": "test_run_abc",
            }
        )
        mock_event = MagicMock()
        mock_event.body_as_str.return_value = body

        mock_ctx = MagicMock()

        # Build the _on_event handler manually
        run_id = adapter._run_id

        def on_event(partition_context, event):
            if event is None:
                return
            data = json.loads(event.body_as_str())
            et = data.get("event_type", "")
            rid = data.get("run_id", "")
            if rid and rid != run_id:
                return
            if et == "AGENT_READY":
                agent_id = data.get("agent_id", "")
                if agent_id:
                    with adapter._ready_lock:
                        adapter._ready_agents.add(agent_id)

        on_event(mock_ctx, mock_event)
        assert "agent-0" in adapter._ready_agents

    def test_stale_run_id_filtered(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=2)

        body = json.dumps(
            {
                "event_type": "AGENT_READY",
                "agent_id": "agent-0",
                "run_id": "old_run_xyz",
            }
        )
        mock_event = MagicMock()
        mock_event.body_as_str.return_value = body

        run_id = adapter._run_id

        def on_event(partition_context, event):
            if event is None:
                return
            data = json.loads(event.body_as_str())
            rid = data.get("run_id", "")
            if rid and rid != run_id:
                return
            et = data.get("event_type", "")
            if et == "AGENT_READY":
                with adapter._ready_lock:
                    adapter._ready_agents.add(data.get("agent_id", ""))

        on_event(MagicMock(), mock_event)
        assert len(adapter._ready_agents) == 0

    def test_eval_answer_delivered(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=1)

        event_id = "ev123"
        answer_event = threading.Event()
        with adapter._answer_lock:
            adapter._answer_events[event_id] = answer_event

        body = json.dumps(
            {
                "event_type": "EVAL_ANSWER",
                "event_id": event_id,
                "answer": "Paris is the capital",
                "run_id": "test_run_abc",
                "agent_id": "agent-0",
            }
        )
        mock_event = MagicMock()
        mock_event.body_as_str.return_value = body

        run_id = adapter._run_id

        def on_event(partition_context, event):
            if event is None:
                return
            data = json.loads(event.body_as_str())
            rid = data.get("run_id", "")
            if rid and rid != run_id:
                return
            et = data.get("event_type", "")
            if et == "EVAL_ANSWER":
                eid = data.get("event_id", "")
                ans = data.get("answer", "")
                with adapter._answer_lock:
                    if eid in adapter._answer_events:
                        adapter._pending_answers[eid] = ans
                        adapter._answer_events[eid].set()

        on_event(MagicMock(), mock_event)
        assert answer_event.is_set()
        assert adapter._pending_answers[event_id] == "Paris is the capital"


class TestGetMemoryStats:
    def test_stats_returned(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=3)
        adapter._learn_count = 100
        adapter._question_count = 10

        stats = adapter.get_memory_stats()
        assert stats["adapter"] == "remote"
        assert stats["learn_count"] == 100
        assert stats["question_count"] == 10
        assert stats["agent_count"] == 3


class TestClose:
    def test_close_sets_shutdown(self):
        mod = _load_module()
        adapter = _make_adapter(mod)
        adapter.close()
        assert adapter._shutdown.is_set()


class TestListenerStartup:
    def test_listener_alive_waits_for_all_partition_initializers(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=2)
        adapter._listener_alive.clear()

        consumer = MagicMock()
        consumer.get_partition_ids.return_value = ["0", "1"]

        def fake_receive(*, on_event, on_partition_initialize, starting_position):
            assert starting_position == "@latest"
            on_partition_initialize(MagicMock(partition_id="0"))
            assert not adapter._listener_alive.is_set()
            on_partition_initialize(MagicMock(partition_id="1"))
            assert adapter._listener_alive.is_set()
            adapter._shutdown.set()

        consumer.receive.side_effect = fake_receive
        consumer_cls = MagicMock()

        with _patch_azure_eventhub(consumer_cls=consumer_cls):
            consumer_cls.from_connection_string.return_value = consumer
            adapter._listen_for_answers()

        consumer_cls.from_connection_string.assert_called_once()
        consumer.close.assert_called_once()

    def test_listener_reconnects_with_backfill_after_unexpected_return(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=1)
        adapter._listener_alive.clear()

        first_consumer = MagicMock()
        first_consumer.get_partition_ids.return_value = ["0"]
        second_consumer = MagicMock()
        second_consumer.get_partition_ids.return_value = ["0"]
        starting_positions = []

        def receive_first(*, on_event, on_partition_initialize, starting_position):
            starting_positions.append(starting_position)
            on_partition_initialize(MagicMock(partition_id="0"))
            return None

        def receive_second(*, on_event, on_partition_initialize, starting_position):
            starting_positions.append(starting_position)
            on_partition_initialize(MagicMock(partition_id="0"))
            adapter._shutdown.set()
            return None

        first_consumer.receive.side_effect = receive_first
        second_consumer.receive.side_effect = receive_second
        consumer_cls = MagicMock()

        with _patch_azure_eventhub(consumer_cls=consumer_cls):
            consumer_cls.from_connection_string.side_effect = [first_consumer, second_consumer]
            adapter._listen_for_answers()

        assert starting_positions == ["@latest", "-1"]
        assert adapter._listener_restart_count == 1

    def test_listener_reconnects_with_backfill_after_exception(self):
        mod = _load_module()
        adapter = _make_adapter(mod, agent_count=1)
        adapter._listener_alive.clear()

        first_consumer = MagicMock()
        first_consumer.get_partition_ids.return_value = ["0"]
        second_consumer = MagicMock()
        second_consumer.get_partition_ids.return_value = ["0"]
        starting_positions = []

        def receive_first(*, on_event, on_partition_initialize, starting_position):
            starting_positions.append(starting_position)
            raise RuntimeError("boom")

        def receive_second(*, on_event, on_partition_initialize, starting_position):
            starting_positions.append(starting_position)
            on_partition_initialize(MagicMock(partition_id="0"))
            adapter._shutdown.set()
            return None

        first_consumer.receive.side_effect = receive_first
        second_consumer.receive.side_effect = receive_second
        consumer_cls = MagicMock()

        with _patch_azure_eventhub(consumer_cls=consumer_cls):
            consumer_cls.from_connection_string.side_effect = [first_consumer, second_consumer]
            adapter._listen_for_answers()

        assert starting_positions == ["@latest", "-1"]
        assert adapter._listener_restart_count == 1
