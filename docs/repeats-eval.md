# Repeats Eval (Superseded)

This page used to document `experiments/hive_mind/query_hive.py` and a Service Bus-based Azure flow.

That is not the current eval surface.

## Use These Docs Instead

- [Running Evals](./running-evals.md)
- [Distributed Hive Eval on Azure](./distributed-hive-eval.md)
- [Azure Hive Q&A Eval Quick Reference](./azure-hive-qa-eval.md)

## Current Repeatable Paths

For local variance checks, use `compare` with multiple seeds or repeats:

```bash
amplihack-eval compare \
  --seeds 42,123,456,789 \
  --turns 100 \
  --questions 20 \
  --question-set holdout \
  --repeats 3 \
  --output-dir /tmp/eval-compare
```

For Azure distributed reruns, use the current Event Hubs runner or the wrapper script:

```bash
python -m amplihack_eval.azure.eval_distributed ...
./run_distributed_eval.sh ...
```

Do not use the older `query_hive.py` plus Service Bus instructions from the previous version of this page.
