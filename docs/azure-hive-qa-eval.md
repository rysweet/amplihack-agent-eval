# Azure Hive Q&A Eval Tutorial

This tutorial explains how to run a Q&A evaluation against the **live Azure
Hive Mind** — 20 agents deployed as Azure Container Apps, sharing knowledge
via a distributed hash table (DHT) over Azure Service Bus.

## What This Eval Does

`query_hive.py` sends a `network_graph.search_query` event to the live hive
via the Service Bus topic `hive-graph`. Each running agent receives the query,
searches its local knowledge shard, and publishes a `network_graph.search_response`
back. The eval script collects these responses and scores them against expected
keywords.

```
query_hive.py ──publishes──► hive-graph topic
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              agent-0 sub      agent-1 sub   …  agent-19 sub
              (Container App)  (Container App)   (Container App)
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
                          eval-query-agent sub
                          (responses collected here)
```

## Prerequisites

```bash
# Install the azure-servicebus SDK
pip install azure-servicebus

# Clone the amplihack repo on the feat/distributed-hive-mind branch
git clone https://github.com/rysweet/amplihack5 /path/to/amplihack
cd /path/to/amplihack
git checkout feat/distributed-hive-mind
```

The live hive is already deployed in Azure resource group `hive-mind-rg`.
No Azure credentials are required to query it — the connection string is
embedded in `query_hive.py` as a default.

To use your own hive, set:

```bash
export HIVE_CONNECTION_STRING="Endpoint=sb://YOUR-NS.servicebus.windows.net/;..."
export HIVE_TOPIC="hive-graph"
export HIVE_SUBSCRIPTION="eval-query-agent"
export HIVE_TIMEOUT="10"
```

## Running the Built-In Q&A Eval

```bash
cd /path/to/amplihack

# Run all 15 questions, print results to stdout
python experiments/hive_mind/query_hive.py --run-eval

# Write results to JSON
python experiments/hive_mind/query_hive.py --run-eval --output results.json

# Adjust timeout if agents are slow to respond
python experiments/hive_mind/query_hive.py --run-eval --timeout 15
```

### Example Output

```
======================================================================
LIVE AZURE HIVE Q&A EVAL
Hive: hive-sb / topic: hive-graph
Questions: 15
Timeout per query: 10s
======================================================================

Domain               Hit   Results  | Question
----------------------------------------------------------------------
  biology            HIT   7 results | What are cells made of?
  biology            HIT   5 results | How does DNA store information?
  biology            MISS  0 results | What do enzymes do?
  chemistry          HIT   8 results | What is the structure of water?
  ...

======================================================================
RESULTS
======================================================================
  Overall:   11/15 (73.3%)

  By domain:
    biology             : 2/3 (67%)
    chemistry           : 3/3 (100%)
    computer_science    : 2/3 (67%)
    mathematics         : 2/3 (67%)
    physics             : 2/3 (67%)

  Total time: 38.42s
======================================================================
```

## Running a Single Query

```bash
python experiments/hive_mind/query_hive.py \
    --query "What is Newton's second law?" \
    --timeout 10

# Output:
# Querying live hive: "What is Newton's second law?"
# Results (3):
#   1. [0.97] F equals ma is Newton's second law
#        concept: mechanics
#        source:  agent-7
```

## Q&A Eval Dataset

The built-in eval covers 15 questions across 5 domains (3 per domain):

| Domain | Questions |
|---|---|
| biology | cells, DNA, enzymes |
| chemistry | water structure, covalent bonds, pH |
| physics | Newton's 2nd law, speed of light, E=mc² |
| mathematics | Pythagorean theorem, derivatives, Pi |
| computer_science | binary search complexity, ACID, CAP theorem |

Each question is scored by keyword matching: a hit requires **all expected
keywords** to appear in at least one result from the hive.

## Scoring

The eval uses a binary keyword-match scoring scheme:

- **Hit**: at least one returned fact contains all expected keywords
- **Miss**: no returned fact contains all expected keywords

Overall accuracy = `hits / total_questions`.

## Understanding the Results

### Why Do MISS Results Occur?

1. **Agent asleep**: Azure Container Apps scale to zero after inactivity. The
   Container App runtime needs time to wake up (cold start ~30s).
2. **Shard miss**: The query hit the wrong shard owners (DHT routing is based
   on keyword hashing, not semantic similarity).
3. **Knowledge gap**: The specific fact was not promoted to the shared hive
   (it may be in local Kuzu memory only).
4. **Timeout too short**: Increase `--timeout` if agents need more time.

### Waking Up Agents

If agents are cold-started (zero instances), send a warmup query first:

```bash
# Warmup query — agents will start responding within 30s
python experiments/hive_mind/query_hive.py --query "warmup" --timeout 30

# Then run the eval
python experiments/hive_mind/query_hive.py --run-eval --timeout 15
```

## Adding Custom Questions

Edit `QA_EVAL_DATASET` in `query_hive.py`:

```python
QA_EVAL_DATASET = [
    # (domain, question, expected_keywords)
    ("my_domain", "What is X?", ["keyword1", "keyword2"]),
    ...
]
```

Or pass a custom dataset programmatically:

```python
from experiments.hive_mind.query_hive import HiveQueryClient, _score_response

client = HiveQueryClient(timeout=10)
results = client.query("What is Newton's second law?")
hit = _score_response(results, ["F", "ma"])
client.close()
```

## Architecture Reference

| Component | Value |
|---|---|
| Azure resource group | `hive-mind-rg` |
| Service Bus namespace | See `HIVE_CONNECTION_STRING` env var |
| Topic | `hive-graph` |
| Eval subscription | `eval-query-agent` |
| Agent subscriptions | `agent-0` … `agent-19` |
| Agent containers | `amplihive-app-0` … `amplihive-app-19` |
| Knowledge shard backend | In-memory DHT (DistributedHiveGraph) |
| Query protocol | `network_graph.search_query` / `network_graph.search_response` |

## Related Documentation

- `docs/hive-mind-eval.md` — Hive Mind eval strategy and full topology guide
- `experiments/hive_mind/query_hive.py` — The query/eval script
- `experiments/hive_mind/deploy_azure_hive.sh` — Azure deployment script
- `src/amplihack/memory/network_store.py` — NetworkGraphStore protocol
- `src/amplihack/agents/goal_seeking/hive_mind/event_bus.py` — Event bus
- `docs/distributed_hive_mind.md` — Distributed hive architecture

## Troubleshooting

| Symptom | Fix |
|---|---|
| `0 results` on all queries | Agents are cold-started — send warmup first |
| `ImportError: azure-servicebus` | `pip install azure-servicebus` |
| Timeout errors in JSON output | Increase `--timeout` to 15-30s |
| Subscription not found | Re-run `az servicebus topic subscription create ...` |
| All MISS, even warmup | Check `az containerapp revision list --name amplihive-app-0 --resource-group hive-mind-rg` |
