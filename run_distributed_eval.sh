#!/usr/bin/env bash
# run_distributed_eval.sh — Single-command distributed hive mind evaluation.
#
# Deploys N agents to Azure Container Apps, feeds T turns of content,
# asks Q questions, grades answers, and publishes results as a GitHub
# release tag with full metadata for reproducibility.
#
# Usage:
#   ./run_distributed_eval.sh                     # defaults: 5 agents, 100 turns, 20 questions
#   ./run_distributed_eval.sh --agents 100 --turns 1000
#   ./run_distributed_eval.sh --agents 100 --turns 5000 --questions 50
#
# Required env vars:
#   ANTHROPIC_API_KEY   — Claude API key for agents
#
# Optional env vars (with defaults):
#   HIVE_RESOURCE_GROUP — Azure resource group (default: hive-mind-eval-rg)
#   HIVE_LOCATION       — Azure region (default: eastus)
#   EVAL_TAG_PREFIX     — Release tag prefix (default: eval)
#   SKIP_DEPLOY         — Set to 1 to skip deployment (reuse existing agents)
#   SKIP_CLEANUP        — Set to 1 to skip post-eval cleanup

set -euo pipefail

# ============================================================
# Parse arguments
# ============================================================
AGENTS=5
TURNS=100
QUESTIONS=20
SEED=42
AGENTS_PER_APP=5
GRADER_MODEL="claude-haiku-4-5-20251001"
ANSWER_TIMEOUT=""
PARALLEL_WORKERS=""
QUESTION_FAILOVER_RETRIES=""
REPLICATE_LEARN_TO_ALL_AGENTS=false
MEMORY_QUERY_FANOUT="${HIVE_MEMORY_QUERY_FANOUT:-}"
SHARD_QUERY_TIMEOUT_SECONDS="${HIVE_SHARD_QUERY_TIMEOUT_SECONDS:-}"
DEPLOYMENT_PROFILE="${HIVE_DEPLOYMENT_PROFILE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agents)       AGENTS="$2"; shift 2 ;;
        --turns)        TURNS="$2"; shift 2 ;;
        --questions)    QUESTIONS="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --agents-per-app) AGENTS_PER_APP="$2"; shift 2 ;;
        --grader-model) GRADER_MODEL="$2"; shift 2 ;;
        --answer-timeout) ANSWER_TIMEOUT="$2"; shift 2 ;;
        --parallel-workers) PARALLEL_WORKERS="$2"; shift 2 ;;
        --question-failover-retries) QUESTION_FAILOVER_RETRIES="$2"; shift 2 ;;
        --replicate-learn-to-all-agents) REPLICATE_LEARN_TO_ALL_AGENTS=true; shift ;;
        --memory-query-fanout) MEMORY_QUERY_FANOUT="$2"; shift 2 ;;
        --shard-query-timeout) SHARD_QUERY_TIMEOUT_SECONDS="$2"; shift 2 ;;
        --deployment-profile) DEPLOYMENT_PROFILE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ============================================================
# Configuration
# ============================================================
RESOURCE_GROUP="${HIVE_RESOURCE_GROUP:-hive-mind-eval-rg}"
LOCATION="${HIVE_LOCATION:-eastus}"
TAG_PREFIX="${EVAL_TAG_PREFIX:-eval}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
AMPLIHACK_ROOT="${AMPLIHACK_ROOT:-$(cd "${REPO_ROOT}/../amplihack" && pwd)}"
AMPLIHACK_SOURCE_ROOT="${AMPLIHACK_SOURCE_ROOT:-${AMPLIHACK_ROOT}}"

HIVE_NAME="${HIVE_NAME:-eval${AGENTS}a${TURNS}t}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
TAG_NAME="${TAG_PREFIX}-${AGENTS}agents-${TURNS}turns-${TIMESTAMP}"
RESULTS_DIR="/tmp/eval-results-${TAG_NAME}"

if [[ -z "${PARALLEL_WORKERS}" ]]; then
    if (( AGENTS >= 100 )); then
        PARALLEL_WORKERS=1
    elif (( AGENTS >= 50 )); then
        PARALLEL_WORKERS=2
    else
        PARALLEL_WORKERS=10
    fi
fi

if [[ -z "${QUESTION_FAILOVER_RETRIES}" ]]; then
    if (( AGENTS >= 100 )); then
        QUESTION_FAILOVER_RETRIES=2
    elif (( AGENTS >= 50 )); then
        QUESTION_FAILOVER_RETRIES=1
    else
        # <50 agents: 1 retry to compensate for the 120s answer_timeout default
        QUESTION_FAILOVER_RETRIES=1
    fi
fi

if [[ -z "${ANSWER_TIMEOUT}" ]]; then
    if (( AGENTS >= 100 )); then
        ANSWER_TIMEOUT=0
    else
        ANSWER_TIMEOUT=120
    fi
fi

if [[ -z "${MEMORY_QUERY_FANOUT}" ]]; then
    MEMORY_QUERY_FANOUT="${AGENTS}"
fi

if [[ -z "${SHARD_QUERY_TIMEOUT_SECONDS}" ]]; then
    if (( AGENTS >= 100 )); then
        SHARD_QUERY_TIMEOUT_SECONDS=0
    else
        SHARD_QUERY_TIMEOUT_SECONDS=60
    fi
fi

# Resolve deployment profile: explicit arg/env > infer from agent count
if [[ -z "${DEPLOYMENT_PROFILE}" ]]; then
    if (( AGENTS == 100 && AGENTS_PER_APP == 5 )); then
        DEPLOYMENT_PROFILE="federated-100"
    elif (( AGENTS == 10 && AGENTS_PER_APP == 1 )); then
        DEPLOYMENT_PROFILE="smoke-10"
    else
        DEPLOYMENT_PROFILE="custom"
    fi
fi

mkdir -p "${RESULTS_DIR}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

RERUN_COMMAND="ANTHROPIC_API_KEY=\\\"...\\\" SKIP_DEPLOY=1 HIVE_NAME=${HIVE_NAME} HIVE_RESOURCE_GROUP=${RESOURCE_GROUP} HIVE_LOCATION=${LOCATION} AMPLIHACK_ROOT=${AMPLIHACK_ROOT} AMPLIHACK_SOURCE_ROOT=${AMPLIHACK_SOURCE_ROOT} HIVE_DEPLOYMENT_PROFILE=${DEPLOYMENT_PROFILE} ./run_distributed_eval.sh --agents ${AGENTS} --turns ${TURNS} --questions ${QUESTIONS} --seed ${SEED} --agents-per-app ${AGENTS_PER_APP} --grader-model ${GRADER_MODEL} --answer-timeout ${ANSWER_TIMEOUT} --parallel-workers ${PARALLEL_WORKERS} --question-failover-retries ${QUESTION_FAILOVER_RETRIES} --memory-query-fanout ${MEMORY_QUERY_FANOUT} --shard-query-timeout ${SHARD_QUERY_TIMEOUT_SECONDS} --deployment-profile ${DEPLOYMENT_PROFILE}"
if [[ "${REPLICATE_LEARN_TO_ALL_AGENTS}" == "true" ]]; then
    RERUN_COMMAND+=" --replicate-learn-to-all-agents"
fi

log "============================================"
log "Distributed Hive Mind Evaluation"
log "  Eval repo:   ${REPO_ROOT}"
log "  Code root:   ${AMPLIHACK_SOURCE_ROOT}"
log "  Agents:     ${AGENTS}"
log "  Turns:      ${TURNS}"
log "  Questions:  ${QUESTIONS}"
log "  Seed:       ${SEED}"
log "  Hive name:  ${HIVE_NAME}"
log "  Parallel:   ${PARALLEL_WORKERS} workers"
log "  Timeout:    ${ANSWER_TIMEOUT}s"
log "  Failover:   ${QUESTION_FAILOVER_RETRIES} retries"
log "  Fanout:     ${MEMORY_QUERY_FANOUT} shards"
log "  Shard wait: ${SHARD_QUERY_TIMEOUT_SECONDS}s"
log "  Profile:    ${DEPLOYMENT_PROFILE}"
log "  Tag:        ${TAG_NAME}"
log "============================================"

# ============================================================
# Step 1: Deploy (unless SKIP_DEPLOY=1)
# ============================================================
if [[ "${SKIP_DEPLOY:-0}" != "1" ]]; then
    log "Step 1/4: Deploying ${AGENTS} agents..."
    HIVE_NAME="${HIVE_NAME}" \
    HIVE_RESOURCE_GROUP="${RESOURCE_GROUP}" \
    HIVE_LOCATION="${LOCATION}" \
    HIVE_AGENT_COUNT="${AGENTS}" \
    HIVE_AGENTS_PER_APP="${AGENTS_PER_APP}" \
    HIVE_MEMORY_QUERY_FANOUT="${MEMORY_QUERY_FANOUT}" \
    HIVE_SHARD_QUERY_TIMEOUT_SECONDS="${SHARD_QUERY_TIMEOUT_SECONDS}" \
    HIVE_DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE}" \
    bash "${AMPLIHACK_SOURCE_ROOT}/deploy/azure_hive/deploy.sh" 2>&1 | tee "${RESULTS_DIR}/deploy.log"
    log "Deployment complete."
else
    log "Step 1/4: Skipping deployment (SKIP_DEPLOY=1)"
    log "  Reusing live deployment; ensure query fanout=${MEMORY_QUERY_FANOUT} and shard timeout=${SHARD_QUERY_TIMEOUT_SECONDS}s are already deployed."
fi

# ============================================================
# Step 2: Get Event Hubs connection string
# ============================================================
log "Step 2/4: Retrieving Event Hubs connection..."
EH_NS=$(az eventhubs namespace list -g "${RESOURCE_GROUP}" --query "[0].name" -o tsv)
EH_CONN=$(az eventhubs namespace authorization-rule keys list \
    --namespace-name "${EH_NS}" \
    -g "${RESOURCE_GROUP}" \
    --name RootManageSharedAccessKey \
    --query primaryConnectionString -o tsv)
log "  EH Namespace: ${EH_NS}"

# ============================================================
# Step 3: Run evaluation
# ============================================================
log "Step 3/4: Running evaluation (${TURNS} turns, ${QUESTIONS} questions)..."
EVAL_START=$(date +%s)

cd "${REPO_ROOT}"
eval_cmd=(
    "${AMPLIHACK_ROOT}/.venv/bin/python" -m amplihack_eval.azure.eval_distributed
    --turns "${TURNS}"
    --questions "${QUESTIONS}"
    --agents "${AGENTS}"
    --seed "${SEED}"
    --grader-model "${GRADER_MODEL}"
    --connection-string "${EH_CONN}"
    --input-hub "hive-events-${HIVE_NAME}"
    --response-hub "eval-responses-${HIVE_NAME}"
    --answer-timeout "${ANSWER_TIMEOUT}"
    --parallel-workers "${PARALLEL_WORKERS}"
    --question-failover-retries "${QUESTION_FAILOVER_RETRIES}"
    --output "${RESULTS_DIR}/eval_report.json"
)
if [[ "${REPLICATE_LEARN_TO_ALL_AGENTS}" == "true" ]]; then
    eval_cmd+=(--replicate-learn-to-all-agents)
fi
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
PYTHONPATH="${REPO_ROOT}/src:${AMPLIHACK_SOURCE_ROOT}/src" \
"${eval_cmd[@]}" 2>&1 | tee "${RESULTS_DIR}/eval.log"

EVAL_END=$(date +%s)
EVAL_DURATION=$((EVAL_END - EVAL_START))
log "Evaluation complete in ${EVAL_DURATION}s."

# ============================================================
# Step 4: Package results and create release tag
# ============================================================
log "Step 4/4: Packaging results and creating release..."

# Capture git state
GIT_SHA=$(cd "${AMPLIHACK_SOURCE_ROOT}" && git rev-parse HEAD)
GIT_BRANCH=$(cd "${AMPLIHACK_SOURCE_ROOT}" && git rev-parse --abbrev-ref HEAD)
if [[ -n "$(cd "${AMPLIHACK_SOURCE_ROOT}" && git status --porcelain)" ]]; then
    GIT_DIRTY=true
else
    GIT_DIRTY=false
fi
EVAL_REPO_GIT_SHA=$(cd "${REPO_ROOT}" && git rev-parse HEAD)
EVAL_REPO_GIT_BRANCH=$(cd "${REPO_ROOT}" && git rev-parse --abbrev-ref HEAD)
if [[ -n "$(cd "${REPO_ROOT}" && git status --porcelain)" ]]; then
    EVAL_REPO_GIT_DIRTY=true
else
    EVAL_REPO_GIT_DIRTY=false
fi

# Write metadata
cat > "${RESULTS_DIR}/metadata.json" << EOF
{
    "tag": "${TAG_NAME}",
    "timestamp": "${TIMESTAMP}",
    "git_sha": "${GIT_SHA}",
    "git_branch": "${GIT_BRANCH}",
    "git_dirty": ${GIT_DIRTY},
    "eval_repo_git_sha": "${EVAL_REPO_GIT_SHA}",
    "eval_repo_git_branch": "${EVAL_REPO_GIT_BRANCH}",
    "eval_repo_git_dirty": ${EVAL_REPO_GIT_DIRTY},
    "config": {
        "agents": ${AGENTS},
        "turns": ${TURNS},
        "questions": ${QUESTIONS},
        "seed": ${SEED},
        "agents_per_app": ${AGENTS_PER_APP},
        "grader_model": "${GRADER_MODEL}",
        "answer_timeout": ${ANSWER_TIMEOUT},
        "parallel_workers": ${PARALLEL_WORKERS},
        "question_failover_retries": ${QUESTION_FAILOVER_RETRIES},
        "replicate_learn_to_all_agents": ${REPLICATE_LEARN_TO_ALL_AGENTS},
        "memory_query_fanout": ${MEMORY_QUERY_FANOUT},
        "shard_query_timeout_seconds": ${SHARD_QUERY_TIMEOUT_SECONDS},
        "hive_name": "${HIVE_NAME}",
        "resource_group": "${RESOURCE_GROUP}",
        "location": "${LOCATION}",
        "amplihack_root": "${AMPLIHACK_ROOT}",
        "amplihack_source_root": "${AMPLIHACK_SOURCE_ROOT}",
        "deployment_profile": "${DEPLOYMENT_PROFILE}"
    },
    "duration_seconds": ${EVAL_DURATION},
    "rerun_command": "${RERUN_COMMAND}"
}
EOF

# Extract overall score from report
OVERALL_SCORE=$(python3 -c "
import json, sys
try:
    r = json.load(open('${RESULTS_DIR}/eval_report.json'))
    print(f\"{r.get('overall_score', r.get('score', 0)) * 100:.1f}%\")
except: print('N/A')
" 2>/dev/null || echo "N/A")

# Create release notes
RELEASE_BODY="## Distributed Hive Mind Eval: ${OVERALL_SCORE}

| Parameter | Value |
|-----------|-------|
| Agents | ${AGENTS} |
| Turns | ${TURNS} |
| Questions | ${QUESTIONS} |
| Seed | ${SEED} |
| Duration | ${EVAL_DURATION}s |
| Git SHA | \`${GIT_SHA:0:8}\` |
| Branch | \`${GIT_BRANCH}\` |

### Re-run Command
\`\`\`bash
cd ${AMPLIHACK_SOURCE_ROOT}
git checkout ${GIT_SHA}
cd ${REPO_ROOT}
${RERUN_COMMAND}
\`\`\`
"

# Create GitHub release in the eval repo
cd "${REPO_ROOT}"
git tag "${TAG_NAME}" -m "Eval: ${AGENTS} agents, ${TURNS} turns — ${OVERALL_SCORE}"
git push origin "${TAG_NAME}" 2>/dev/null || true

gh release create "${TAG_NAME}" \
    --title "Eval: ${AGENTS} agents, ${TURNS} turns — ${OVERALL_SCORE}" \
    --notes "${RELEASE_BODY}" \
    "${RESULTS_DIR}/eval_report.json" \
    "${RESULTS_DIR}/metadata.json" \
    "${RESULTS_DIR}/eval.log" \
    2>&1 || log "Warning: Failed to create GitHub release (may need auth)"

log "============================================"
log "EVAL COMPLETE"
log "  Score: ${OVERALL_SCORE}"
log "  Tag:   ${TAG_NAME}"
log "  Results: ${RESULTS_DIR}/"
log "============================================"
