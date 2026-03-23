# Azure Hive Q&A Eval Quick Reference

Use this page when you want the shortest path to a distributed Azure run.

## Wrapper Command

```bash
cd /path/to/amplihack-agent-eval

export ANTHROPIC_API_KEY=...
export AMPLIHACK_SOURCE_ROOT=/path/to/amplihack

./run_distributed_eval.sh   --agents 100   --turns 5000   --questions 50   --question-set standard
```

`AMPLIHACK_SOURCE_ROOT` is enough when the deployment source and runtime venv come from the same checkout. Set `AMPLIHACK_ROOT` separately only if the wrapper should use a different `amplihack/.venv`.

Reuse an existing deployment:

```bash
SKIP_DEPLOY=1 HIVE_NAME=amplihive3175e HIVE_RESOURCE_GROUP=hive-pr3175-rg ./run_distributed_eval.sh   --agents 100   --turns 5000   --questions 50   --question-set holdout
```

## Direct Runner Command

This path requires the sibling `amplihack` package to be installed because it reuses the main repo's long-horizon harness.

```bash
python -m amplihack_eval.azure.eval_distributed   --connection-string "<event-hubs-connection-string>"   --input-hub "hive-events-amplihive3175e"   --response-hub "eval-responses-amplihive3175e"   --agents 100   --agents-per-app 5   --turns 5000   --questions 50   --question-set standard   --parallel-workers 1   --question-failover-retries 2   --answer-timeout 0   --output /tmp/eval_report.json
```

## Current Azure Facts

- transport is **Event Hubs**, not Service Bus
- the input hub is `hive-events-<hive-name>`
- the response hub is `eval-responses-<hive-name>`
- `standard` and `holdout` are supported on both the wrapper and direct runner

## When To Use Which Path

- use `run_distributed_eval.sh` when you want deploy + eval in one step
- use `python -m amplihack_eval.azure.eval_distributed` when Azure is already running and you only need the eval harness

## See Also

- [Distributed Hive Eval on Azure](./distributed-hive-eval.md)
- [Running Evals](./running-evals.md)
