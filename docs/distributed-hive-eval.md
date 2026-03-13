# Distributed Hive Mind Evaluation

Run memory evaluations against a fleet of distributed AI agents deployed on Azure Container Apps.

## Quick Start

```bash
# Single command: deploy + eval + publish results
ANTHROPIC_API_KEY="..." ./run_distributed_eval.sh --agents 100 --turns 1000 --questions 20
```

## How It Works

1. **Deploy** — Creates N agents across Azure Container Apps with Event Hubs transport
2. **Learn** — Feeds T dialogue turns (round-robin across agents)
3. **Question** — Asks Q questions, agents search their local + remote knowledge
4. **Grade** — LLM grades each answer against ground truth
5. **Publish** — Results tagged as a GitHub release with full metadata

## CLI Usage

### Via amplihack-eval CLI

```bash
amplihack-eval run \
    --adapter distributed-hive \
    --connection-string "$EH_CONN" \
    --input-hub hive-events-myhive \
    --response-hub eval-responses-myhive \
    --agents 100 \
    --turns 1000 \
    --questions 20
```

### Via run_distributed_eval.sh

```bash
# Deploy 100 agents, run 1000-turn eval
./run_distributed_eval.sh --agents 100 --turns 1000

# Skip deployment (reuse existing agents)
SKIP_DEPLOY=1 ./run_distributed_eval.sh --agents 100 --turns 5000

# Custom seed and grader model
./run_distributed_eval.sh --agents 50 --turns 500 --seed 123 \
    --grader-model claude-haiku-4-5-20251001
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--agents` | 5 | Number of deployed agents |
| `--turns` | 100 | Dialogue turns to feed |
| `--questions` | 20 | Evaluation questions |
| `--seed` | 42 | Random seed for reproducibility |
| `--agents-per-app` | 5 | Agents per Azure Container App |
| `--grader-model` | claude-haiku-4-5-20251001 | Model for answer grading |
| `--answer-timeout` | 0 | Per-answer timeout (0 = none) |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key for agents |
| `HIVE_RESOURCE_GROUP` | No | Azure resource group (default: hive-mind-eval-rg) |
| `HIVE_LOCATION` | No | Azure region (default: eastus) |
| `SKIP_DEPLOY` | No | Set to 1 to skip deployment |
| `SKIP_CLEANUP` | No | Set to 1 to keep infra after eval |

## Result Format

Each eval run produces:

- **eval_report.json** — Full graded report with per-category scores
- **metadata.json** — Config, git SHA, duration, re-run command
- **eval.log** — Complete execution log

Results are published as GitHub releases with the tag format:
```
eval-{agents}agents-{turns}turns-{timestamp}
```

### Re-running a Past Evaluation

Each release includes the exact re-run command:

```bash
git checkout <sha>
ANTHROPIC_API_KEY="..." SKIP_DEPLOY=1 \
  ./run_distributed_eval.sh --agents 100 --turns 1000 --seed 42
```

## Architecture

```
┌─────────────┐     Event Hubs      ┌──────────────────────┐
│ Eval Harness │ ─── LEARN_CONTENT ──→│ Container App 0      │
│              │ ─── INPUT ──────────→│  agent-0 ... agent-9 │
│              │                     ├──────────────────────┤
│              │                     │ Container App 1      │
│              │                     │  agent-10..agent-19  │
│              │                     ├──────────────────────┤
│              │                     │ ...                  │
│              │                     ├──────────────────────┤
│              │ ← EVAL_ANSWER ──────│ Container App N      │
│              │                     │  agent-90..agent-99  │
└─────────────┘                     └──────────────────────┘
                                           │ ↕ │
                                     SHARD_QUERY / SHARD_RESPONSE
                                     (cross-agent knowledge search)
```

Each agent:
- Stores facts in a local Kuzu graph database
- Responds to cross-shard queries from other agents via Event Hubs
- Searches the full distributed graph when answering questions

## Scaling Notes

| Agents | Apps | CPU/Agent | Mem/Agent | EH Partitions | EH TUs |
|--------|------|-----------|-----------|---------------|--------|
| ≤5     | 1    | 0.75      | 1.5Gi     | N+4           | 1      |
| ≤20    | 4    | 0.75      | 1.5Gi     | N+4           | 1      |
| ≤50    | 10   | 0.25      | 0.5Gi     | 32            | 2      |
| 100    | 20   | 0.25      | 0.5Gi     | 32            | 4      |

Azure Container Apps Consumption plan limits: 4 CPU / 8Gi per app, 30 apps per environment.

## Eval Results (2026-03-13)

### 100 agents × 5000 turns: 99.29%

| Category | Score |
|---|---|
| needle_in_haystack (4q) | 100% |
| numerical_precision (3q) | 100% |
| temporal_evolution (3q) | 100% |
| cross_reference (2q) | 100% |
| infrastructure_knowledge (1q) | 100% |
| meta_memory (1q) | 100% |
| security_log_analysis (1q) | 100% |
| incident_tracking (1q) | 97.50% |
| distractor_resistance (2q) | 97.50% |
| source_attribution (2q) | 96.67% |
| **OVERALL** | **99.29%** |

**Infrastructure**: 20 Container Apps × 5 containers, 0.75 CPU / 1.5Gi each, EH Standard 32 partitions.

**Key requirement**: Explicit `partition_id` routing in EventHubsShardTransport. Do NOT use `partition_key` — it causes 80%+ event loss when multiple agents share a consumer group.

### Release Tag

Results are stored as a GitHub release tag in the amplihack repo:
`eval/hive-100agents-5000turns-99.29pct`

The release includes JSON result files and re-run instructions pinned to the exact commit.
