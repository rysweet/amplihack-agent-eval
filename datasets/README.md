# Pre-built Learning DB Datasets

Pre-built memory databases for the long-horizon evaluation, distributed via GitHub Releases.

## Why?

The 5000-turn learning phase takes **4+ hours** and requires 10,000+ LLM API calls.
These pre-built datasets let you skip learning and jump straight to evaluation.

## Available Datasets

| Name | Turns | Seed | Facts | Baseline Score | Date |
|------|-------|------|-------|---------------|------|
| `5000t-seed42-v1.0` | 5,000 | 42 | 762 | 90.47% | 2026-02-24 |

## Quick Start

```bash
# Download a dataset
amplihack-eval download-dataset 5000t-seed42-v1.0

# Run evaluation using the pre-built DB (skip 4+ hour learning phase)
amplihack-eval run \
  --adapter learning-agent \
  --skip-learning \
  --load-db datasets/5000t-seed42-v1.0/memory_db \
  --turns 5000 \
  --questions 100
```

## Programmatic Download

```python
from amplihack_eval.datasets.download import download_dataset, list_datasets

# List available datasets
datasets = list_datasets()

# Download a specific dataset
path = download_dataset("5000t-seed42-v1.0")
print(f"Downloaded to: {path}")
```

## Dataset Structure

Each dataset directory contains:

```
5000t-seed42-v1.0/
├── metadata.json          # Dataset configuration and provenance
├── baseline_results.json  # Evaluation scores at time of creation
└── memory_db/             # Kuzu graph database (the actual pre-built DB)
```

### metadata.json Schema

```json
{
  "name": "5000t-seed42-v1.0",
  "version": "1.0",
  "turns": 5000,
  "seed": 42,
  "facts_delivered": 762,
  "baseline_score": 0.9047,
  "code_version": "ff20586",
  "date": "2026-02-24",
  "agent_type": "LearningAgent (CognitiveMemory)",
  "num_questions_tested": 100,
  "num_categories": 15,
  "memory_stats": {
    "episodic": 5000,
    "semantic": 10854,
    "total": 15854
  }
}
```

## Creating New Datasets

After running a long learning phase, package the results:

```bash
# 1. Run the full learning phase
amplihack-eval run --adapter learning-agent --turns 5000 --questions 100 \
  --output-dir /tmp/my-eval

# 2. Package as a dataset (future: automated script)
# For now, copy memory_db/ and create metadata.json manually

# 3. Create a GitHub release
tar -czf 5000t-seed42-v1.0.tar.gz -C datasets/ 5000t-seed42-v1.0/
gh release create dataset-5000t-seed42-v1.0 \
  --title "5000-Turn Learning DB (seed 42, v1.0)" \
  --notes "Pre-built memory DB: 5000 turns, 762 facts, 90.47% baseline" \
  5000t-seed42-v1.0.tar.gz
```
