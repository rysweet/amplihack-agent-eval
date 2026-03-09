# Running Evals -- Quick Start

This page covers all evaluation types in the amplihack-agent-eval framework.
Choose the eval type that matches your scenario.

## End-to-End Eval Pipeline

All eval types follow the same four-stage pipeline: generate data, feed it to
agents, ask questions, and grade the answers.

```mermaid
flowchart TB
    subgraph "Stage 1: Data Generation"
        GD[generate_dialogue<br/>N turns, seed=S] --> GT[GroundTruth<br/>turns + facts + blocks]
        GT --> GQ[generate_questions<br/>M questions across 15 categories]
        GQ --> QS[Questions<br/>with rubrics + expected answers]
    end

    subgraph "Stage 2: Learning"
        direction TB
        GT --> |content turns| LA

        subgraph local ["Local Eval"]
            LA[LearningAgent]
            LA --> |LLM fact extraction| KG[Kuzu Graph DB]
        end

        subgraph distributed ["Distributed Eval (Azure)"]
            FC[feed_content.py] --> |Service Bus| SB[hive-events topic]
            SB --> A1[Agent 0..4<br/>Container App 0]
            SB --> A2[Agent 5..9<br/>Container App 1]
            SB --> AN[Agent N..N+4<br/>Container App K]
            A1 --> KG1[Kuzu DB]
            A2 --> KG2[Kuzu DB]
            AN --> KGN[Kuzu DB]
        end
    end

    subgraph "Stage 3: Question Answering"
        QS --> |questions| QA

        subgraph local_qa ["Local"]
            QA[agent.answer_question] --> |LLM synthesis| ANS[Answers]
        end

        subgraph dist_qa ["Distributed"]
            QH[query_hive.py<br/>--ooda-eval] --> |INPUT events via SB| AGENTS[Agents process<br/>via OODA loop]
            AGENTS --> |stdout| LOGS[Log Analytics]
            LOGS --> |poll for ANSWER lines| QH
            QH --> DANS[Answers]
        end
    end

    subgraph "Stage 4: Grading + Reporting"
        ANS --> GRADE
        DANS --> GRADE
        GRADE[Hybrid Grader] --> |deterministic rubric<br/>keywords + patterns| DET[Deterministic Score]
        GRADE --> |LLM semantic judgment<br/>multi-vote for stability| LLM_G[LLM Score]
        DET --> MERGE[Weighted Merge]
        LLM_G --> MERGE
        MERGE --> RPT[JSON Report<br/>overall + per-category + per-question]
    end

    style local fill:#e8f5e9
    style distributed fill:#e3f2fd
    style local_qa fill:#e8f5e9
    style dist_qa fill:#e3f2fd
```

### Pipeline Stages

| Stage | What Happens | Key Tools |
|-------|-------------|-----------|
| **Data Generation** | Deterministic dialogue with embedded facts across 12 blocks | `generate_dialogue()`, `generate_questions()` |
| **Learning** | Agents ingest content via LLM fact extraction (~3 LLM calls/turn) | `LearningAgent.learn_from_content()`, `feed_content.py` |
| **Question Answering** | Agents retrieve facts and synthesize answers (~2 LLM calls/question) | `LearningAgent.answer_question()`, `query_hive.py --ooda-eval` |
| **Grading + Reporting** | Hybrid scoring: deterministic rubric + LLM semantic judgment | `HybridGrader`, JSON report output |

### Report Structure

The JSON report contains:

```json
{
  "overall_score": 0.97,
  "num_turns": 300,
  "num_questions": 50,
  "learning_time_s": 2486.6,
  "questioning_time_s": 38.0,
  "category_breakdown": [
    {"category": "needle_in_haystack", "avg_score": 1.0, "count": 10},
    {"category": "temporal_evolution", "avg_score": 1.0, "count": 7}
  ],
  "results": [
    {
      "question_id": "seclog_01",
      "question_text": "How many failed SSH logins...",
      "expected_answer": "6 failed SSH logins...",
      "actual_answer": "**6 failed SSH logins**...",
      "score": 1.0,
      "reasoning": "Agent correctly identified..."
    }
  ]
}
```

## Eval Types at a Glance

| Eval Type | Agents | Infrastructure | Use Case |
|-----------|--------|---------------|----------|
| [Single-agent (local)](#single-agent-local) | 1 | None | Baseline memory recall testing |
| [20-agent comparative (local)](#20-agent-comparative-local) | 20+1 | None | Multi-agent topology comparison with adversarial testing |
| [Distributed (Azure)](#distributed-azure) | 5-1000+ | Azure Container Apps + Service Bus | Production-scale distributed eval |

## Single-Agent (Local)

Tests a single agent's memory system with up to 5000 dialogue turns and 200
questions across 15 categories.

### Quick Run

```bash
python -m amplihack.eval.long_horizon_memory \
  --turns 300 --questions 50 \
  --model claude-sonnet-4-6 \
  --grader-model claude-haiku-4-5-20251001 \
  --output-dir /tmp/eval-results
```

### Smoke Test (fast, cheap)

```bash
python -m amplihack.eval.long_horizon_memory \
  --turns 100 --questions 20
```

### Full Stress Test

```bash
python -m amplihack.eval.long_horizon_memory \
  --turns 5000 --questions 200 \
  --grader-votes 5 --seed 42 \
  --output-dir /tmp/eval-results
```

### Skip Learning (use pre-built dataset)

```bash
amplihack-eval run \
  --adapter learning-agent \
  --skip-learning \
  --load-db datasets/5000t-seed42-v1.0/memory_db \
  --turns 5000 --questions 100
```

**Details**: [Long-Horizon Memory Eval](LONG_HORIZON_EVAL.md)

## 20-Agent Comparative (Local)

Runs four conditions in a single script to compare topologies:

1. **SINGLE_AGENT** -- One agent learns all turns (ceiling)
2. **ISOLATED_20** -- 20 agents, no sharing
3. **FLAT_SHARED_20** -- 20 agents, all facts bulk-loaded
4. **HIVE_20** -- 20 agents with consensus hive + 1 adversarial agent

```bash
python experiments/hive_mind/run_20agent_eval.py
```

This runs from the amplihack repo (not amplihack-agent-eval). Ensure both
packages are installed:

```bash
pip install -e /path/to/amplihack
pip install -e /path/to/amplihack-agent-eval
export ANTHROPIC_API_KEY=...
```

**Details**: [Hive Mind Eval Strategy](hive-mind-eval.md)

## Distributed (Azure)

Deploys agents to Azure Container Apps and evaluates via Service Bus.

### Deploy, Feed, Eval, Cleanup

```bash
# 1. Deploy
HIVE_AGENT_COUNT=100 HIVE_AGENTS_PER_APP=5 \
HIVE_AGENT_MODEL=claude-sonnet-4-6 \
bash deploy/azure_hive/deploy.sh

# 2. Feed content
SB_CONN=$(az servicebus namespace authorization-rule keys list \
  -g hive-mind-rg --namespace-name <ns> \
  --name RootManageSharedAccessKey \
  --query primaryConnectionString -o tsv)

AMPLIHACK_MEMORY_CONNECTION_STRING="$SB_CONN" \
AMPLIHACK_TOPIC_NAME="hive-events" \
python deploy/azure_hive/feed_content.py --turns 5000

# 3. Run eval
LA_ID=$(az monitor log-analytics workspace list -g hive-mind-rg \
  --query "[0].customerId" -o tsv)

python experiments/hive_mind/query_hive.py \
  --ooda-eval --repeats 1 \
  --connection-string "$SB_CONN" \
  --workspace-id "$LA_ID" \
  --topic hive-events \
  --answer-wait 600 \
  --output results.json

# 4. Cleanup
bash deploy/azure_hive/deploy.sh --cleanup
```

**Details**: [Running Distributed Eval on Azure](azure-hive-qa-eval.md)

## Prerequisites (All Eval Types)

```bash
# Install packages
pip install -e /path/to/amplihack
pip install -e /path/to/amplihack-agent-eval

# Set API key
export ANTHROPIC_API_KEY=...

# For distributed eval only:
az login
pip install azure-servicebus
```

## Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | All | API key for LLM calls |
| `GRADER_MODEL` | Single-agent, 20-agent | Model for grading (default: `claude-sonnet-4-5-20250929`) |
| `EVAL_MODEL` | Single-agent, 20-agent | Model for LearningAgent (default: `claude-sonnet-4-5-20250929`) |
| `HIVE_AGENT_MODEL` | Distributed | Model for hive agents (default: `claude-sonnet-4-6`) |
| `HIVE_AGENT_COUNT` | Distributed | Number of agents to deploy |
| `OODA_ANSWER_WAIT` | Distributed | Seconds to wait for answers (default: `600`) |

## Comparing Results

### Multi-seed comparison (single-agent)

```bash
amplihack-eval compare --seeds 42,123,456,789 --turns 100 --questions 20
```

### Cross-topology comparison

```bash
python -m amplihack_eval.compare \
  results/single/scores.json \
  results/flat/scores.json \
  results/distributed/scores.json
```
