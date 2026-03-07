# Running Repeats Evaluation with query_hive.py

## Overview

The `--repeats N` flag in `query_hive.py` runs the evaluation N times and reports
per-run scores together with the **median** and **standard deviation** across runs.
This reduces noise from LLM grading variance and provides a more reliable estimate
of hive mind performance.

## Prerequisites

- Python 3.11+
- `amplihack-agent-eval` installed (`pip install amplihack-agent-eval`)
- Azure Service Bus connection string (for live hive mode)
- Anthropic API key (for grading)

```bash
pip install amplihack-agent-eval anthropic azure-servicebus
export HIVE_CONNECTION_STRING="Endpoint=sb://hive-sb-dj2qo2w7vu5zi.servicebus.windows.net/;SharedAccessKeyName=...;SharedAccessKey=..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Running the Repeats Eval

### Live Azure Hive (requires Azure Service Bus)

```bash
# Run the security analyst eval 3 times against the live Azure hive
python experiments/hive_mind/query_hive.py --run-eval --repeats 3

# With output saved to JSON
python experiments/hive_mind/query_hive.py --run-eval --repeats 3 --output repeats_results.json

# Seed the hive first if not already seeded, then run 3 repeats
python experiments/hive_mind/query_hive.py --seed --run-eval --repeats 3 --output repeats_results.json
```

### Demo Mode (no Azure needed)

```bash
# Run demo eval (local DistributedHiveGraph) 3 times
python experiments/hive_mind/query_hive.py --demo --repeats 3

# With output
python experiments/hive_mind/query_hive.py --demo --repeats 3 --output demo_repeats.json
```

## Output Format

Each repeat prints its full eval output. After all repeats, a summary is printed:

```
=======================================================================
REPEATS SUMMARY (3 runs)
=======================================================================
  Run 1: avg_score=0.872
  Run 2: avg_score=0.841
  Run 3: avg_score=0.882
  Median: 0.872  StdDev: 0.022
=======================================================================
```

When `--output` is specified, the JSON file contains:

```json
{
  "mode": "live_repeats",
  "repeats": 3,
  "scores": [0.872, 0.841, 0.882],
  "median": 0.872,
  "stddev": 0.022,
  "runs": [ ... ]
}
```

## Benchmark Results

### Live Azure Hive — 3-Repeat Security Analyst Eval

| Metric | Value |
|--------|-------|
| **Median score** | **86.5%** |
| **Std deviation** | **10.1%** |
| Runs | 3 |
| Eval type | Security analyst Q&A (live Azure hive) |
| Branch | `feat/distributed-hive-mind` |

The 10.1% standard deviation indicates moderate run-to-run variance, driven by:
- LLM grading noise (semantic scoring varies per call)
- Hive routing non-determinism (queries may hit different shards)
- Service Bus message ordering effects

### Interpreting Results

| Stddev | Interpretation |
|--------|----------------|
| < 5% | High consistency — routing and grading are stable |
| 5–15% | Moderate variance — acceptable for semantic eval |
| > 15% | High variance — investigate routing or grading issues |

## Why Report Median Instead of Mean?

The median is more robust to outlier runs caused by:
- Transient Service Bus timeouts
- LLM API rate limits causing partial grading failures
- Network partitions during federated queries

For N=3, the median equals the middle value. For N≥5, it provides
a more representative central estimate than the mean.

## Recommended Practice

- Run at least **N=3** repeats for any reported benchmark
- Report both **median** and **stddev** (not just the single best run)
- Use `--output` to save all run data for reproducibility
- Re-seed the hive before long eval campaigns to ensure fresh state

## Related

- [`query_hive.py`](../experiments/hive_mind/query_hive.py) — the eval script
- [Hive Mind Eval Strategy](hive-mind-eval.md) — background on eval methodology
- [Azure Hive Q&A Eval](azure-hive-qa-eval.md) — live hive setup guide
