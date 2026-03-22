# Eval `eval-3175b-20260319T215735Z` — Question-Phase Failure Diagnosis

**Date**: 2026-03-19
**Hive**: `amplihive3175b`
**Resource Group**: `hive-pr3175-rg`
**Run ID**: `2b72abaf3fb6`
**Config**: agents=100, parallel\_workers=1, answer\_timeout=0, failover\_retries=2
**Log**: `/tmp/eval-3175b-20260319T215735Z.wrapper.log`

---

## Timeline Summary

| Time (UTC) | Event |
|---|---|
| 21:57:51 | First AGENT\_ONLINE received |
| 21:58:36 | All 100 agents online |
| 22:03:05 | FEED\_COMPLETE sent |
| 22:11:51 | All agents AGENT\_READY; question phase begins |
| 22:41:35 | Question-39 sent to agent-39 |
| 22:40:03–22:40:50 | AMQP SocketErrors on producer (Connection reset by peer) |
| 22:42:23 | Duplicate AGENT\_ONLINE from agent-39 → question-39 released as "No answer received" |
| 22:42:23–22:47:36 | Mass AGENT\_ONLINE wave covering ~80+ agents (container restart cascade) |
| 22:56:40 | Question-39 scores 0.00 (exhausted all 3 failover attempts: agents 39→40→41) |
| 23:01:49 | Question-40 scores 0.00 |
| 22:41–23:15 | Restarted containers come back online (app-7 agents at 22:59, app-8 agents at 23:08) |

Questions 0–38 scored 0.95–1.00. Only 2 questions (39 and 40) scored 0.00.

---

## Azure CLI Evidence

```
app-2  agent-10: restartCount=1, started=2026-03-19T22:42:43Z
app-2  agent-11: restartCount=1, started=2026-03-19T22:42:43Z
app-2  agent-12: restartCount=1, started=2026-03-19T22:42:45Z
app-2  agent-13: restartCount=1, started=2026-03-19T22:42:48Z
app-7  agent-35..39: restartCount=0, started=2026-03-19T22:59:57–23:00:03Z
app-8  agent-40..44: restartCount=0, started=2026-03-19T23:08:23–23:08:27Z
app-0  agent-0..4:  restartCount=0, started=2026-03-19T23:02:04Z
(all other apps: restartCount=0, started 23:03–23:15 UTC)
```

The `restartCount=0` with late start times for 95+ of the 100 containers indicates
those containers were **previously restarted** and the current container instance is
already a fresh one (restartCount resets per instance). app-2 is the only surviving
evidence of the first-wave restarts (restartCount still at 1).

Container logs for agent-10 at 22:44:04–22:44:05 (after restart):
```
Publishing SHARD_QUERY to agent-0
Publishing SHARD_QUERY to agent-1
...
Publishing SHARD_QUERY to agent-99
```
99 Event Hub messages published in under 2 seconds — the N² fan-out pattern.

---

## Root Causes

### RC1 — N² SHARD\_QUERY Fan-Out Causes Event Hub Saturation (CRITICAL)

**What**: `DistributedCognitiveMemory` / `DistributedHiveGraph` answer a question by
publishing a `SHARD_QUERY` to **all 100 agents** simultaneously. With 100 agents each
potentially processing a question, this is O(N²) = up to 9,900 concurrent Event Hub
publishes per query round.

**Evidence**:
- agent-10 container logs at 22:44 show 99 sequential `SHARD_QUERY` publishes in <2s
- AMQP `SocketError: Connection reset by peer` on producer at 22:40:03, 22:40:50 —
  Event Hub connection reset under write burst
- Mass restart wave begins at 22:42:23 (2 minutes after SocketErrors): consistent with
  OOM or watchdog kill following AMQP error → retry storm → memory exhaustion

**Code path** (`DistributedHiveGraph.query_shards` → `EventHubsShardTransport.broadcast`):
```python
for agent_id in self._all_agent_ids:   # 100 iterations
    self._producer.send(SHARD_QUERY, partition_key=agent_id)
```

**Impact**: Single question triggers 99 EH messages; any backpressure causes
synchronous retry loops, thread saturation, and eventual OOM or AMQP disconnect.

---

### RC2 — Container OOM/Crash Restarts During Question Phase (CRITICAL)

**What**: ~80 containers restarted within 5 minutes (22:42–22:47 UTC), forming a
cascade. Container Apps restart policy respawns the container, but:
1. In-memory Event Hub checkpoint is lost
2. New container starts at `starting_position="@latest"`, missing all messages
   sent during the outage
3. ONLINE\_CHECK flood (100 agents × ONLINE\_CHECK) follows immediately

**Evidence**: app-2 restart timestamps match the AMQP SocketError window; app-7/8
container start times at 22:59–23:08 indicate multiple subsequent restarts (current
pod with restartCount=0 started long after the original deployment).

---

### RC3 — Duplicate AGENT\_ONLINE Heuristic Is Too Aggressive (HIGH)

**What**: `RemoteAgentAdapter._handle_agent_online()` immediately releases any pending
question the moment a **second** AGENT\_ONLINE is received from an agent, with no
grace period:

```python
def _handle_agent_online(self, agent_id: str) -> None:
    with self._online_lock:
        already_online = agent_id in self._online_agents
        self._online_agents.add(agent_id)
    if already_online:
        self._release_pending_answers_for_agent(
            agent_id,
            reason="agent restart",
            detail="duplicate AGENT_ONLINE while question was pending",
        )
```

**Evidence**: At 22:42:23, duplicate AGENT\_ONLINE from agent-39 immediately released
question-39 as "No answer received" — 48 seconds after the question was sent, before
any answer could arrive. The restarted agent-39 came back online at 22:59:57 and
would likely have answered if given the chance.

**Semantic issue**: The duplicate heuristic assumes "second AGENT\_ONLINE = restart",
but this is only a heuristic. Causes include:
- Container restart (correct interpretation)
- Stale ONLINE\_CHECK reply from a previous poll cycle
- Network partition with delayed delivery
- Empty `run_id` events (see RC5)

---

### RC4 — Sequential Failover Into Unstable Agents Exhausts All Retries (HIGH)

**What**: With `answer_timeout=0` (infinite wait), `parallel_workers=1`, and
`question_failover_retries=2`, each failed question blocks the entire question phase
until all 3 attempts complete. Failover goes to agents `base+1` and `base+2` — but
during the mass restart wave, these adjacent agents had also just restarted:

```
question-39: agent-39 → released by duplicate AGENT_ONLINE
  failover-1: agent-40 → released by duplicate AGENT_ONLINE (also restarting)
  failover-2: agent-41 → released by duplicate AGENT_ONLINE (also restarting)
  RESULT: 0.00

question-40: agent-40 → (also unstable)
  failover-1: agent-41 → also unstable
  failover-2: agent-42 → also unstable
  RESULT: 0.00
```

**Evidence**: Log shows questions 39 and 40 cascade through the exact same restart
window, with each failover agent also triggering AGENT\_ONLINE release.

**Root issue**: Failover agent selection is `(base_agent + attempt) % agent_count` —
deterministic neighbors that are physically co-located (same Container App) and thus
crash together.

---

### RC5 — `run_id` Filter Bypassed for Empty `run_id` (MEDIUM)

**What**: The filter in `RemoteAgentAdapter._dispatch_response()`:

```python
if run_id and run_id != self._run_id:
    return
```

Uses Python truthiness: an **empty string** `run_id` is **falsy**, so the `if` block
is never entered and the empty-`run_id` event is processed as a current-run event.

**When this triggers**:
1. Agent publishes AGENT\_ONLINE at startup (before receiving ONLINE\_CHECK) with
   `run_id=""` because `_current_run_id` has not been set yet
2. Agent's `_CorrelatingInputSource` calls `clear_context()` on reconnect, setting
   `run_id` to `""` for the next event
3. Stale events from a prior run with no `run_id` field

**Startup hazard detail**: The deployed `agent_entrypoint.py`
(`issue-3172-cognitive-memory-unified-graph-clean` worktree) only publishes
AGENT\_ONLINE **in response to ONLINE\_CHECK** (not at startup), so the startup hazard
is **not currently triggered** in normal flow. However, if `_current_run_id` is
unset when ONLINE\_CHECK arrives, `publish_agent_online(run_id=self._current_run_id)`
publishes with empty `run_id`, which then bypasses the filter on the adapter side.

---

### RC6 — No Persistent Event Hub Checkpoint Store (MEDIUM)

**What**: `EventHubsInputSource` uses in-memory checkpoints only. On OOM kill, the
checkpoint position is lost; the new container starts at `"@latest"`, missing all
messages delivered during the outage.

**Impact**: Questions sent to a crashing agent are **silently dropped** — the agent
never sees them. The adapter eventually releases via duplicate AGENT\_ONLINE or
`answer_timeout`, but the question score is 0.

**Code location**: `EventHubConsumerClient` is created without a `checkpoint_store`
argument, which defaults to in-memory.

---

## Prioritized Fix List

### P0 — Limit SHARD\_QUERY Fan-Out (fixes RC1)

Replace full broadcast with targeted or batched SHARD\_QUERY:
- Route queries only to agents likely to hold the relevant shard (using DHT key → agent mapping)
- Or cap fan-out at `sqrt(N)` with hierarchical aggregation
- Or use a single coordinator agent per query

This alone should eliminate the N²=9,900 EH burst that triggers AMQP saturation
and the downstream OOM cascade.

### P1 — Add Grace Period to Duplicate AGENT\_ONLINE Heuristic (fixes RC3)

Instead of immediately releasing pending questions on duplicate AGENT\_ONLINE, wait
for a configurable grace period (e.g., 30s) before releasing:

```python
# Pseudo-code
if already_online:
    schedule_delayed_release(agent_id, delay=self._restart_grace_period, reason="agent restart")
    # Cancel the delayed release if agent answers in time
```

This allows a restarted agent to reconnect and answer before the question is forfeited.

### P2 — Fix `run_id` Filter to Treat Empty `run_id` as Mismatch (fixes RC5)

```python
# Before (buggy):
if run_id and run_id != self._run_id:
    return

# After:
if run_id != self._run_id:   # empty run_id always mismatches non-empty self._run_id
    return
```

This prevents stale or empty-`run_id` AGENT\_ONLINE events from being mistakenly
processed as current-run events.

### P3 — Use Non-Adjacent Failover Agent Selection (fixes RC4)

Instead of `(base + attempt) % N`, choose failover agents that are **physically
separated** (different Container Apps):

```python
# Spread failovers across Container Apps
failover_agent = (base + attempt * agents_per_app) % agent_count
```

Or use random selection with exclusion of the same Container App:
```python
# Exclude agents in the same app as base_agent
app_id = base_agent // agents_per_app
candidates = [i for i in range(agent_count) if i // agents_per_app != app_id]
failover_agent = random.choice(candidates)
```

### P4 — Add Azure Blob Checkpoint Store for Event Hub Consumer (fixes RC6)

Pass a `BlobCheckpointStore` to `EventConsumerClient`:

```python
from azure.eventhub.extensions.checkpointstoreblobaio import BlobCheckpointStore

checkpoint_store = BlobCheckpointStore.from_connection_string(
    blob_conn_str, container_name="eh-checkpoints"
)
consumer_client = EventHubConsumerClient(
    ...,
    checkpoint_store=checkpoint_store,
)
```

This allows restarted containers to resume from last committed position rather than
`"@latest"`, preventing silent message loss.

### P5 — Normalize `run_id` in `publish_agent_online` (reduces RC5 exposure)

In `agent_entrypoint.py`, ensure `_current_run_id` is set before ONLINE\_CHECK
processing. If not set, log a warning and use a sentinel rather than `""`:

```python
def publish_agent_online(self, run_id: str = "") -> None:
    effective_run_id = run_id or self._current_run_id or "UNKNOWN"
    self._publish_to_eh({"event_type": "AGENT_ONLINE", ..., "run_id": effective_run_id})
```

And on the adapter side, reject `"UNKNOWN"` run\_ids explicitly.

---

## AGENT\_ONLINE Heuristic Characterization

The current duplicate-AGENT\_ONLINE heuristic works as follows:

1. **First AGENT\_ONLINE** from an agent → marks agent as online in `_online_agents`
2. **Second AGENT\_ONLINE** from an agent already in `_online_agents` → infers restart,
   immediately releases any pending question for that agent as "No answer received"

**Correct interpretation cases**:
- Container OOM kill and respawn: ONLINE\_CHECK is re-sent; agent replies; duplicate
  detected → correct signal

**Incorrect / unsafe interpretation cases**:
1. **Network delay**: Duplicate ONLINE\_CHECK reply from a slow-but-alive agent arrives
   late. Agent did not restart but is penalized.
2. **Stale run events**: A prior eval run's AGENT\_ONLINE event arrives late and
   bypasses the `run_id` filter (see RC5). Harness incorrectly infers restart.
3. **Empty run\_id bypass**: Agent publishes AGENT\_ONLINE with empty `run_id` (startup
   hazard or context-clear race). Adapter processes it as current-run, inferring restart.
4. **Adjacent-agent cascade**: Mass restart wave causes duplicate AGENT\_ONLINE for
   **all** 80+ restarting agents simultaneously. Questions that happen to be in-flight
   at that moment are released, even if those specific questions' target agents would
   have recovered within the answer window.

**Startup AGENT\_ONLINE hazard** (currently **not triggered** in deployed code):
- The deployed `agent_entrypoint.py` only calls `publish_agent_online()` in response
  to ONLINE\_CHECK, not at startup
- However, if a container restarts **after** the harness has moved past the
  `_wait_for_agents_online()` phase and sends no further ONLINE\_CHECK, the restarted
  agent will not publish AGENT\_ONLINE at all — the harness will never know about the
  restart and will eventually hang with `answer_timeout=0`
- The duplicate AGENT\_ONLINE heuristic only fires if a **new** ONLINE\_CHECK is sent
  after the restart; if it isn't sent, the hang is broken only by `answer_timeout`

---

## Key Code Locations

| File | Function | Issue |
|---|---|---|
| `amplihack_eval/adapters/remote_agent_adapter.py` | `_handle_agent_online()` | RC3 — no grace period |
| `amplihack_eval/adapters/remote_agent_adapter.py` | `_dispatch_response()` | RC5 — empty run\_id bypass |
| `amplihack_eval/adapters/remote_agent_adapter.py` | `answer_question()` failover loop | RC4 — adjacent failover selection |
| `amplihack/agents/cognitive_memory/distributed_hive_graph.py` | `query_shards()` / broadcast | RC1 — N² fan-out |
| `amplihack/agents/goal_seeking/input_source.py` | `EventHubsInputSource.__init__` | RC6 — no checkpoint store |
| `amplihack/deploy/azure_hive/agent_entrypoint.py` | `publish_agent_online()` | RC5 — empty run\_id on startup |
