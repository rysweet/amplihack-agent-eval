# Running Distributed Eval on Azure

This guide covers how to deploy a distributed hive mind to Azure, feed it
content, and run a Q&A evaluation against the live agents.

## Architecture Overview

The distributed eval deploys N agents as Azure Container Apps. Each agent holds
a shard of the distributed knowledge graph (DistributedHiveGraph). Agents
communicate via Azure Service Bus topics. The eval script sends questions as
events and collects answers from Log Analytics.

```
feed_content.py ──publishes──► Service Bus topic (hive-events)
                                       │
                      ┌────────────────┼────────────────┐
                      ▼                ▼                ▼
                agent-0 sub      agent-1 sub   …  agent-N sub
                (Container App)  (Container App)   (Container App)
                      │                │                │
                      └────────────────┼────────────────┘
                                       ▼
                            query_hive.py collects
                            answers from Log Analytics
```

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

## Step 2: Feed Content

Retrieve the Service Bus connection string and feed dialogue turns to the hive:

```bash
SB_CONN=$(az servicebus namespace authorization-rule keys list \
  -g hive-mind-rg \
  --namespace-name <YOUR-SB-NAMESPACE> \
  --name RootManageSharedAccessKey \
  --query primaryConnectionString -o tsv)

AMPLIHACK_MEMORY_CONNECTION_STRING="$SB_CONN" \
AMPLIHACK_TOPIC_NAME="hive-events" \
python deploy/azure_hive/feed_content.py --turns 5000
```

Replace `<YOUR-SB-NAMESPACE>` with the Service Bus namespace created by the
deploy script (visible in `--status` output).

## Step 3: Wait for Processing

Poll Log Analytics to confirm agents are processing content:

```bash
LA_ID=$(az monitor log-analytics workspace list \
  -g hive-mind-rg \
  --query "[0].customerId" -o tsv)

az monitor log-analytics query -w "$LA_ID" \
  --analytics-query "ContainerAppConsoleLogs_CL | where TimeGenerated > ago(2m) | where Log_s has 'Completed Call' | count"
```

Wait until the count stops increasing, indicating agents have finished
processing all turns.

## Step 4: Run Eval

```bash
python experiments/hive_mind/query_hive.py \
  --ooda-eval \
  --repeats 1 \
  --connection-string "$SB_CONN" \
  --workspace-id "$LA_ID" \
  --topic hive-events \
  --answer-wait 600 \
  --output results.json
```

### Alternative: Run the keyword-match eval

For a faster sanity check using keyword matching instead of LLM grading:

```bash
python experiments/hive_mind/query_hive.py \
  --seed --run-eval \
  --connection-string "$SB_CONN" \
  --topic hive-events \
  --output results.json
```

### Run eval multiple times for statistical stability

```bash
python experiments/hive_mind/query_hive.py \
  --ooda-eval \
  --repeats 3 \
  --connection-string "$SB_CONN" \
  --workspace-id "$LA_ID" \
  --topic hive-events \
  --answer-wait 600 \
  --output results.json
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
