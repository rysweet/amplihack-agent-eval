# Hive Mind Eval Strategy

## What Is the Hive Mind?

The hive mind is a distributed knowledge-sharing layer for AI agents. Instead of
each agent operating in isolation with its own knowledge graph, the hive mind
lets agents pool facts, propagate discoveries, and answer questions using
collective knowledge.

### Four-Layer Architecture

The hive mind is built on four layers:

| Layer | Responsibility | Implementations |
|-------|---------------|-----------------|
| **Storage** (HiveGraph) | Shared fact store | `InMemoryHiveGraph` (local, < 20 agents), `DistributedHiveGraph` (DHT-sharded, 20-1000+ agents) |
| **Transport** (EventBus) | Message delivery between agents | `LocalEventBus` (in-process), Azure Event Hubs (distributed) |
| **Discovery** (Gossip) | Peer discovery and fact propagation | Gossip protocol with configurable broadcast thresholds |
| **Query** (Dedup + Rerank) | Answer assembly from distributed shards | Fan-out to K shard owners, RRF merge, deduplication |

**HiveMindOrchestrator** is the unified coordination layer that wires the four
layers together. It manages agent lifecycle, fact promotion, query routing, and
consensus filtering through a single entry point.

### Storage Implementations

- **InMemoryHiveGraph** -- All facts in one shared dict. Suitable for local
  eval with < 20 agents (single-process, no network).
- **DistributedHiveGraph** -- DHT-sharded across agent shards. Each agent holds
  `O(F/N)` facts instead of `O(F)` total. Queries fan out to K shard owners
  (default 5) instead of all N agents. Designed for 20-1000+ agents.

### Transport Implementations

- **LocalEventBus** -- In-process event delivery for local evaluation. No
  network overhead.
- **Azure Event Hubs** -- Cloud-based transport for distributed deployments.
  Agents consume targeted input partitions and publish eval responses through dedicated Event Hubs.

## Why Evaluate the Hive Mind?

Shared knowledge introduces trade-offs that don't exist in single-agent mode:

- **Accuracy vs. noise**: More facts means better coverage, but also more
  irrelevant or conflicting information to filter.
- **Latency**: Federated queries traverse a tree of hives. Does the extra
  retrieval time pay for itself in answer quality?
- **Consensus filtering**: The hive can require multiple agents to confirm a
  fact before it's visible. Does this block legitimate facts or only junk?
- **Scale effects**: Do 20 agents sharing knowledge outperform 1 agent that
  sees everything?
- **Parallel learning**: Multiple agents learning in parallel speeds ingestion
  but introduces ordering effects on fact extraction.

The eval measures whether collective knowledge actually improves Q&A accuracy
compared to the single-agent baseline.

## The Four Topologies

### Single (Baseline)

One agent learns all dialogue turns. All facts live in one Kuzu DB. No hive
involved. This is the control group.

```mermaid
graph TD
    subgraph SingleAgent
        KB[(Kuzu DB)] --- QA[answer_question]
    end
    Turns[All dialogue turns] --> SingleAgent
```

### Flat (InMemoryHiveGraph)

N agents split the dialogue turns (round-robin). Each has its own Kuzu DB plus
a shared `InMemoryHiveGraph`. Every promoted fact is immediately visible to all
agents. Suitable for in-process testing with up to ~20 agents.

```mermaid
graph TD
    T[Dialogue turns<br/>round-robin split] --> A0 & A1 & A2 & A3 & A4

    A0[Agent 0<br/>Kuzu DB] --> Hive
    A1[Agent 1<br/>Kuzu DB] --> Hive
    A2[Agent 2<br/>Kuzu DB] --> Hive
    A3[Agent 3<br/>Kuzu DB] --> Hive
    A4[Agent 4<br/>Kuzu DB] --> Hive

    Hive["Shared InMemoryHiveGraph<br/>All facts in one pool<br/>O(F) memory — all agents hold all facts"]
```

### Distributed Single DHT (DistributedHiveGraph)

All N agents share one `DistributedHiveGraph`. Facts are partitioned via
consistent hashing (DHT): each agent owns a keyspace shard and holds only
`O(F/N)` facts. Queries fan out to K shard owners and merge via RRF. No
federation tree overhead. Designed for 20-1000+ agents.

```mermaid
graph TD
    T[Dialogue turns<br/>round-robin split] --> A0 & A1 & A2 & A3 & A4

    subgraph DHT["DistributedHiveGraph (consistent hash ring)"]
        A0[Agent 0<br/>Shard: facts 0..F/N]
        A1[Agent 1<br/>Shard: facts F/N..2F/N]
        A2[Agent 2<br/>Shard: facts 2F/N..3F/N]
        A3[Agent 3<br/>Shard: facts 3F/N..4F/N]
        A4[Agent 4<br/>Shard: facts 4F/N..5F/N]
    end

    Q[query] --> DHT
    DHT -->|"fan-out to K=5 shard owners<br/>RRF merge"| Result
```

**Key property:** Memory is `O(F/N)` per agent instead of `O(F)` total.

### Federated

N agents organized into M groups. Each group has its own hive (InMemoryHiveGraph
or DistributedHiveGraph). A root hive connects the groups. High-confidence facts
(>= 0.9) broadcast across groups via the root. Lower-confidence facts stay in
their group but are reachable through `query_federated()` tree traversal with
RRF merge.

```mermaid
graph TD
    Root["Root Hive<br/>broadcast copies only"]
    Root --> G0["Group 0 Hive"]
    Root --> G1["Group 1 Hive"]

    G0 --> A0[Agent 0<br/>Kuzu]
    G0 --> A1[Agent 1<br/>Kuzu]
    G0 --> A2[Agent 2<br/>Kuzu]

    G1 --> A3[Agent 3<br/>Kuzu]
    G1 --> A4[Agent 4<br/>Kuzu]

    T0[Group 0 turns] --> A0 & A1 & A2
    T1[Group 1 turns] --> A3 & A4
```

## How to Run the Eval

### Prerequisites

```bash
pip install -e /path/to/amplihack           # hive mind + learning agent
pip install -e /path/to/amplihack-agent-eval  # eval harness
export ANTHROPIC_API_KEY=...                # or OPENAI_API_KEY / AZURE_OPENAI_* vars
```

### Run All Four Topologies

```bash
# Single-agent baseline
python -m amplihack_eval.run \
  --scenario long_horizon \
  --topology single \
  --output-dir results/single

# Flat hive (5 agents, InMemoryHiveGraph)
python -m amplihack_eval.run \
  --scenario long_horizon \
  --topology flat \
  --num-agents 5 \
  --output-dir results/flat

# Distributed single DHT (20 agents, DistributedHiveGraph)
python -m amplihack_eval.run \
  --scenario long_horizon \
  --topology distributed \
  --num-agents 20 \
  --output-dir results/distributed

# Federated hive (10 agents, 2 groups)
python -m amplihack_eval.run \
  --scenario long_horizon \
  --topology federated \
  --num-agents 10 \
  --num-groups 2 \
  --output-dir results/federated
```

### Parallel Learning

Use `--parallel-workers` to run the learning phase with multiple agents in
parallel:

```bash
python -m amplihack_eval.run \
  --scenario long_horizon \
  --topology distributed \
  --num-agents 20 \
  --parallel-workers 10 \
  --output-dir results/distributed-parallel
```

Note: parallel learning introduces ordering effects -- agents see different
subsets of turns. Run with `--seed` for reproducibility.

### Skip Learning (Q&A Only)

If you already have a populated memory DB from a previous run:

```bash
python -m amplihack_eval.run \
  --scenario long_horizon \
  --topology flat \
  --skip-learning \
  --load-db results/flat/memory_db \
  --output-dir results/flat-qa-only
```

### Compare Results

```bash
python -m amplihack_eval.compare \
  results/single/scores.json \
  results/flat/scores.json \
  results/distributed/scores.json \
  results/federated/scores.json
```

## Scoring Methodology

### Per-Question Grading (Median-of-3)

Each question is graded on a 0.0-1.0 scale using **median-of-3 voting** to
reduce LLM grading noise: the grader runs 3 times per question, and the median
score per dimension is taken as the final grade. The reasoning from the vote
closest to the median is preserved for inspection.

The grader checks per dimension:

1. **Factual correctness**: Does the answer contain the right facts?
2. **Completeness**: Does it cover all required details (e.g., port numbers,
   replica counts)?
3. **Absence of hallucination**: Does it avoid stating things not in the
   knowledge base?
4. **Specificity**: Are the right entities named (server names, IPs, versions)?
5. **Temporal awareness**: Does the answer reflect the correct time ordering?
6. **Confidence calibration**: Is uncertainty expressed when facts are ambiguous?

### Aggregation

- **Category scores**: Mean score across all questions in a category (15 categories).
- **Topology score**: Mean across all categories, weighted equally.
- **Statistical rigor**: Run 3+ seeds, report median (more robust than mean).
- **Comparison**: Delta between topology score and single-agent baseline.

### Question Categories (15)

| Category | Description |
|---|---|
| `needle_in_haystack` | Finding specific facts in a large corpus |
| `temporal_evolution` | Understanding time-ordered changes |
| `temporal_trap` | Contradictory time-ordered facts with misleading cues |
| `numerical_precision` | Exact numbers, ports, versions |
| `numerical_reasoning` | Derived quantities (totals, averages) |
| `cross_reference` | Connecting facts from different domains |
| `meta_memory` | Self-awareness of what the agent has learned |
| `source_attribution` | Citing where facts originated |
| `infrastructure_knowledge` | Server/network/deployment facts |
| `security_log_analysis` | Security event interpretation |
| `distractor_resistance` | Ignoring irrelevant or false information |
| `adversarial_distractor` | Facts injected by an adversarial agent |
| `incident_tracking` | Following incident timelines |
| `multi_hop_reasoning` | Connecting 3+ facts for the answer |
| `problem_solving` | Applying facts to solve a scenario |

### What "Better" Means

A topology is better than the baseline if:

- Median score across 3 seeds is higher (primary criterion)
- No individual category regresses by more than 5 percentage points
- `adversarial_distractor` and `distractor_resistance` do not degrade (consensus
  should filter noise, not pass it through)
- `temporal_trap` score does not fall below baseline (DHT sharding should not
  confuse temporal ordering)

## References

- **Architecture doc**: `docs/hive_mind/ARCHITECTURE.md` in the amplihack repo
- **Tutorial**: `docs/tutorial_prompt_to_distributed_hive.md` in the amplihack repo
- **HiveMindOrchestrator**: `src/amplihack/agents/goal_seeking/hive_mind/orchestrator.py`
- **Dataset**: `datasets/5000t-seed42-v1.0/` -- pre-built 5000-turn baseline DB
