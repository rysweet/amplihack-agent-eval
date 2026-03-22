# Distributed Hive Eval on Azure

This is the authoritative guide for running the distributed long-horizon eval against Azure Container Apps through Event Hubs.

## What Lives Where

- `amplihack-agent-eval` owns the eval dataset, grading, report generation, and Azure-side distributed runner
- the sibling `amplihack` repo owns `deploy/azure_hive/`, the agent runtime, and the Azure deployment assets

Use this repo for the eval harness. Use the main `amplihack` repo when you need to change the agent or the Azure deployment shape.

## Prerequisites

- Azure CLI authenticated to the target subscription
- Python environment with `amplihack-agent-eval` installed
- access to the sibling `amplihack` repo
- `ANTHROPIC_API_KEY` set for grading and the default learning-agent runtime

## Fastest End-to-End Path

Run the wrapper from the eval repo:

```bash
cd /path/to/amplihack-agent-eval

export ANTHROPIC_API_KEY=...
export AMPLIHACK_SOURCE_ROOT=/path/to/amplihack

./run_distributed_eval.sh   --agents 100   --turns 5000   --questions 50   --question-set standard
```

This path:

1. deploys or refreshes the Azure hive
2. ensures the required Event Hubs exist
3. runs the distributed eval
4. writes a report bundle with the report, logs, and rerun metadata

## Reuse an Existing Deployment

If the hive is already deployed, skip the deploy step and point the wrapper at the existing resources:

```bash
cd /path/to/amplihack-agent-eval

export ANTHROPIC_API_KEY=...
export AMPLIHACK_SOURCE_ROOT=/path/to/amplihack
export SKIP_DEPLOY=1
export HIVE_NAME=amplihive3175e
export HIVE_RESOURCE_GROUP=hive-pr3175-rg

./run_distributed_eval.sh   --agents 100   --turns 5000   --questions 50   --question-set holdout
```

## Run the Distributed Runner Directly

Use the Python module when you already know the Event Hubs connection string and hub names.

```bash
python -m amplihack_eval.azure.eval_distributed   --connection-string "<event-hubs-connection-string>"   --input-hub "hive-events-amplihive3175e"   --response-hub "eval-responses-amplihive3175e"   --agents 100   --agents-per-app 5   --turns 5000   --questions 50   --seed 42   --question-set standard   --parallel-workers 1   --question-failover-retries 2   --answer-timeout 0   --output /tmp/eval_report.json
```

## Recommended Knobs

| Situation | Setting |
|-----------|---------|
| canonical benchmark run | `--question-set standard` |
| anti-overfitting check | `--question-set holdout` |
| very large distributed hive | `--parallel-workers 1` |
| cold or failure-prone fleet | `--question-failover-retries 2` |
| replicate learning to every agent | `--replicate-learn-to-all-agents` |
| 100+ agents | `--answer-timeout 0` |

The direct runner also supports `--agents-per-app` so the eval metadata reflects the real deployment failure domain.

## Event Hubs Topology

The distributed runner expects two Event Hubs:

- input hub: `hive-events-<hive-name>`
- response hub: `eval-responses-<hive-name>`

The Azure deployment path in the main `amplihack` repo provisions the required consumer groups for the agents and the eval readers. Do not swap this back to Service Bus; the current path is Event Hubs end to end.

## Question Sets

`standard` and `holdout` are deterministic question slices drawn from the same generated fact universe.

- `standard` keeps the canonical slice
- `holdout` uses a different deterministic slice so you can re-test the same runtime against different questions

This is meant for generalization checks. It is not a second synthetic world.

## What the Wrapper Produces

`run_distributed_eval.sh` writes a result directory containing artifacts such as:

- the final report JSON
- the wrapper log
- the direct runner log
- metadata for the deployment and eval parameters
- a rerun command you can reuse later

The wrapper is intentionally example-driven. It does not currently expose its own `--help` screen, so this page is the reference for that path.

## Troubleshooting

### The wrapper cannot find the main repo

Set:

```bash
export AMPLIHACK_SOURCE_ROOT=/path/to/amplihack
```

### You want to run against an already live hive

Set `SKIP_DEPLOY=1`, `HIVE_NAME`, and `HIVE_RESOURCE_GROUP`, then run the wrapper.

### You want to compare prompt or runtime variants without changing the fact set

Keep `--seed` the same and vary `--question-set` or your runtime settings. That lets you compare the same dialogue under a different deterministic question slice.

## Related Docs

- [Running Evals](./running-evals.md)
- [Azure Hive Q&A Eval Quick Reference](./azure-hive-qa-eval.md)
- [Long-Horizon Memory Eval](./LONG_HORIZON_EVAL.md)
