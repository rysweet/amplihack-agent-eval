# Running Distributed Eval on Azure

This guide covers how to deploy a distributed hive mind to Azure, feed it
content, and run a Q&A evaluation against the live agents.

## Architecture Overview

The distributed eval uses a `RemoteAgentAdapter` that implements the same
interface as a local `LearningAgent`. The eval harness (`LongHorizonMemoryEval`)
uses the **exact same code path** for local and distributed evaluation — same
question generation, same OODA loop processing, same grading, same report.

```
eval_distributed.py
  └── LongHorizonMemoryEval.run(RemoteAgentAdapter)
        │
        ├── adapter.learn_from_content(text)
        │     └── Service Bus LEARN_CONTENT ──► all agents (broadcast)
        │              → agent OODA loop: decide("store") → learn
        │
        ├── adapter.answer_question(text)
        │     ├── Service Bus INPUT {event_id} ──► one agent (round-robin)
        │     │        → agent OODA loop: decide("answer") → answer
        │     │        → AnswerPublisher → eval-responses topic {event_id, answer}
        │     └── ◄── collect answer by event_id correlation
        │
        └── _grade_multi_vote(question, answer) ──► score
```

The agent code is identical in single-agent and distributed modes. All
distribution happens via dependency injection at the entrypoint layer.

## Prerequisites

- **Azure CLI** authenticated (`az login`)
- **ANTHROPIC_API_KEY** set in your environment
- **Docker** daemon running (for image build)
- **amplihack** repo cloned with `.venv` activated
- **azure-servicebus** Python package installed (`pip install azure-servicebus`)

## Step 1: Deploy Agents

```bash
export ANTHROPIC_API_KEY="$(cat ~/.msec-k)"  # or your key source

HIVE_NAME=amplihive \
HIVE_RESOURCE_GROUP=hive-mind-rg \
HIVE_LOCATION=westus2 \
HIVE_AGENT_COUNT=100 \
HIVE_AGENTS_PER_APP=5 \
HIVE_AGENT_MODEL=claude-sonnet-4-6 \
bash deploy/azure_hive/deploy.sh
```

This provisions:

- Resource group and Azure Container Registry (ACR)
- Azure Service Bus namespace, topic, and per-agent subscriptions
- Container Apps Environment
- N Container Apps (`ceil(HIVE_AGENT_COUNT / HIVE_AGENTS_PER_APP)` apps)

To check deployment status:

```bash
bash deploy/azure_hive/deploy.sh --status
```

## Step 2: Run Eval

The eval script handles both content feeding and question answering in a single
command — using the same `LongHorizonMemoryEval` harness as single-agent eval:

```bash
SB_CONN=$(az servicebus namespace authorization-rule keys list \
  -g hive-mind-rg \
  --namespace-name <YOUR-SB-NAMESPACE> \
  --name RootManageSharedAccessKey \
  --query primaryConnectionString -o tsv)

python deploy/azure_hive/eval_distributed.py \
  --connection-string "$SB_CONN" \
  --input-topic hive-events-<HIVE_NAME> \
  --response-topic eval-responses-<HIVE_NAME> \
  --turns 5000 --questions 50 \
  --agents 100 \
  --grader-model claude-haiku-4-5-20251001 \
  --output results.json
```

Replace `<YOUR-SB-NAMESPACE>` and `<HIVE_NAME>` with values from the deploy
output.

### What happens during the eval

1. **Generate** — creates deterministic dialogue (5000 turns) and questions (50)
2. **Learn** — sends each turn to all agents via `LEARN_CONTENT` events
3. **Quiz** — sends each question to one agent (round-robin) via `INPUT` events;
   waits for `EVAL_ANSWER` on the response topic
4. **Grade** — uses the same hybrid grader as single-agent (`_grade_multi_vote`)
5. **Report** — writes JSON report in the same format as single-agent eval

### Quick validation (smaller scale)

```bash
python deploy/azure_hive/eval_distributed.py \
  --connection-string "$SB_CONN" \
  --input-topic hive-events-<HIVE_NAME> \
  --response-topic eval-responses-<HIVE_NAME> \
  --turns 100 --questions 10 \
  --agents 100 \
  --output quick-validation.json
```

## Step 5: Cleanup

Remove all Azure resources when done:

```bash
bash deploy/azure_hive/deploy.sh --cleanup
```

Or delete the resource group directly:

```bash
az group delete -n hive-mind-rg --yes
```

## Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HIVE_NAME` | Hive deployment name | `amplihive` |
| `HIVE_RESOURCE_GROUP` | Azure resource group | `hive-mind-rg` |
| `HIVE_LOCATION` | Azure region | `westus2` |
| `HIVE_AGENT_COUNT` | Total number of agents | `5` |
| `HIVE_AGENTS_PER_APP` | Agents per Container App | `5` |
| `HIVE_AGENT_MODEL` | LLM model for agents | `claude-sonnet-4-6` |
| `HIVE_TRANSPORT` | Transport type | `azure_service_bus` |
| `AMPLIHACK_TOPIC_NAME` | Service Bus topic name | `hive-events` |
| `AMPLIHACK_MEMORY_CONNECTION_STRING` | Service Bus connection string | -- |
| `LOG_ANALYTICS_WORKSPACE_ID` | Log Analytics workspace GUID | -- |
| `OODA_ANSWER_WAIT` | Eval answer timeout in seconds | `600` |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Rate limits (429 errors) | Agents retry with exponential backoff (2-32s). If persistent, reduce `HIVE_AGENT_COUNT` or use a smaller model (e.g., Sonnet instead of Opus). |
| Service Bus auth errors | Ensure topic name matches agent config (`hive-events`). Verify connection string has `Send` and `Listen` claims. |
| Missing answers in eval | Check Log Analytics for ANSWER entries. Increase `--answer-wait` beyond 600s for large deployments. |
| `0 results` on all queries | Agents may be cold-started (Azure Container Apps scale to zero). Send a warmup query first or check `az containerapp revision list`. |
| Subscription not found | Verify the eval subscription exists: `az servicebus topic subscription list --resource-group hive-mind-rg --namespace-name <ns> --topic-name hive-events` |
| Deploy script fails | Ensure Docker daemon is running and `az login` is current. Check `deploy.sh --status` for partial deployments. |

## Deploy Script Options

```bash
bash deploy/azure_hive/deploy.sh                 # Deploy everything
bash deploy/azure_hive/deploy.sh --build-only    # Build + push image only
bash deploy/azure_hive/deploy.sh --infra-only    # Provision infra only
bash deploy/azure_hive/deploy.sh --cleanup       # Tear down resource group
bash deploy/azure_hive/deploy.sh --status        # Show deployment status
```

## Related Documentation

- [Hive Mind Eval Strategy](hive-mind-eval.md) -- Topology comparison and scoring methodology
- [Running Evals Quick Start](running-evals.md) -- All eval types in one page
- [Long-Horizon Memory Eval](LONG_HORIZON_EVAL.md) -- Single-agent eval details
