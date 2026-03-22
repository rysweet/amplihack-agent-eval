# Running Evals

This page is the command-oriented quick start for the current eval surface. Use it when you want to know which entrypoint to run.

## Choose an Entry Point

| Goal | Command | When To Use |
|------|---------|-------------|
| Run one eval locally | `amplihack-eval run` | Single-agent or adapter-level work |
| Compare seeds or question sets | `amplihack-eval compare` | Variance and anti-overfitting checks |
| Run the Event Hubs distributed runner directly | `python -m amplihack_eval.azure.eval_distributed` | Azure hive already exists |
| Deploy and run end-to-end | `./run_distributed_eval.sh` | Fresh Azure distributed run with wrapper-managed artifacts |

## Local Runs

### LearningAgent adapter

```bash
amplihack-eval run   --turns 100   --questions 20   --adapter learning-agent   --question-set standard   --output-dir /tmp/eval-run
```

### HTTP adapter

```bash
amplihack-eval run   --turns 100   --questions 20   --adapter http   --agent-url http://localhost:8000   --question-set holdout   --output-dir /tmp/eval-http
```

### Compare multiple seeds

```bash
amplihack-eval compare   --seeds 42,123,456,789   --turns 100   --questions 20   --question-set holdout   --output-dir /tmp/eval-compare
```

## Distributed Azure Runs

The distributed path uses **Event Hubs** for agent input and eval responses. The older Service Bus instructions are obsolete.

### Direct runner

Use this when the Azure hive is already deployed and you have the Event Hubs namespace connection string.

```bash
python -m amplihack_eval.azure.eval_distributed   --connection-string "<event-hubs-connection-string>"   --input-hub "hive-events-amplihive3175e"   --response-hub "eval-responses-amplihive3175e"   --agents 100   --agents-per-app 5   --turns 5000   --questions 50   --question-set standard   --parallel-workers 1   --question-failover-retries 2   --answer-timeout 0   --output /tmp/eval_report.json
```

### Wrapper script

Use the wrapper when you want deploy + Event Hubs setup + eval + packaged artifacts in one flow.

```bash
./run_distributed_eval.sh   --agents 100   --turns 5000   --questions 50   --question-set standard
```

Reuse an existing deployment instead of redeploying:

```bash
SKIP_DEPLOY=1 HIVE_NAME=amplihive3175e HIVE_RESOURCE_GROUP=hive-pr3175-rg ./run_distributed_eval.sh   --agents 100   --turns 5000   --questions 50   --question-set holdout
```

The wrapper expects the sibling `amplihack` repo for Azure deployment assets. Set `AMPLIHACK_SOURCE_ROOT` if that repo is not checked out next to `amplihack-agent-eval`.

## Question Sets

| Value | Meaning |
|-------|---------|
| `standard` | Canonical deterministic question slice |
| `holdout` | Alternate deterministic slice for anti-overfitting validation |

Both slices are generated from the same fact universe. `holdout` changes which questions are asked, not how the dialogue is produced.

## Common Flags

| Flag | Commands | Meaning |
|------|----------|---------|
| `--question-set {standard,holdout}` | `run`, `compare`, `eval_distributed`, wrapper | Choose the deterministic question slice |
| `--parallel-workers` | `run`, `compare`, `eval_distributed` | Parallelize question answering and grading |
| `--answer-timeout` | `run`, `eval_distributed` | Per-question timeout; `0` means no timeout |
| `--question-failover-retries` | `eval_distributed` | Retry unanswered questions on other agents |
| `--replicate-learn-to-all-agents` | `eval_distributed` | Replicate every learn call to every remote agent |
| `--repeats` | `run`, `compare` | Repeat each seed multiple times |

## Outputs

Current entrypoints all produce an `eval_report.json`-style artifact with:

- overall score
- per-category breakdown
- per-question grading details
- run metadata such as seed, turn count, question count, and question set

The wrapper also writes helper artifacts such as a rerun command and deployment metadata next to the report.

## Next Steps

- For a fuller Azure walkthrough, see [Distributed Hive Eval on Azure](./distributed-hive-eval.md).
- For the short Azure command sheet, see [Azure Hive Q&A Eval Quick Reference](./azure-hive-qa-eval.md).
- For single-agent dataset details, see [Long-Horizon Memory Eval](./LONG_HORIZON_EVAL.md).
