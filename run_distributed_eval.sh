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
ANSWER_TIMEOUT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agents)       AGENTS="$2"; shift 2 ;;
        --turns)        TURNS="$2"; shift 2 ;;
        --questions)    QUESTIONS="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --agents-per-app) AGENTS_PER_APP="$2"; shift 2 ;;
        --grader-model) GRADER_MODEL="$2"; shift 2 ;;
        --answer-timeout) ANSWER_TIMEOUT="$2"; shift 2 ;;
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
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AMPLIHACK_ROOT="${AMPLIHACK_ROOT:-$(cd "${REPO_ROOT}/../amplihack" && pwd)}"

HIVE_NAME="eval${AGENTS}a${TURNS}t"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
TAG_NAME="${TAG_PREFIX}-${AGENTS}agents-${TURNS}turns-${TIMESTAMP}"
RESULTS_DIR="/tmp/eval-results-${TAG_NAME}"

mkdir -p "${RESULTS_DIR}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "============================================"
log "Distributed Hive Mind Evaluation"
log "  Agents:     ${AGENTS}"
log "  Turns:      ${TURNS}"
log "  Questions:  ${QUESTIONS}"
log "  Seed:       ${SEED}"
log "  Hive name:  ${HIVE_NAME}"
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
    bash "${AMPLIHACK_ROOT}/deploy/azure_hive/deploy.sh" 2>&1 | tee "${RESULTS_DIR}/deploy.log"
    log "Deployment complete."
else
    log "Step 1/4: Skipping deployment (SKIP_DEPLOY=1)"
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

cd "${AMPLIHACK_ROOT}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
PYTHONPATH=src \
.venv/bin/python deploy/azure_hive/eval_distributed.py \
    --turns "${TURNS}" \
    --questions "${QUESTIONS}" \
    --agents "${AGENTS}" \
    --seed "${SEED}" \
    --grader-model "${GRADER_MODEL}" \
    --connection-string "${EH_CONN}" \
    --input-hub "hive-events-${HIVE_NAME}" \
    --response-hub "eval-responses-${HIVE_NAME}" \
    --answer-timeout "${ANSWER_TIMEOUT}" \
    --output "${RESULTS_DIR}/eval_report.json" \
    2>&1 | tee "${RESULTS_DIR}/eval.log"

EVAL_END=$(date +%s)
EVAL_DURATION=$((EVAL_END - EVAL_START))
log "Evaluation complete in ${EVAL_DURATION}s."

# ============================================================
# Step 4: Package results and create release tag
# ============================================================
log "Step 4/4: Packaging results and creating release..."

# Capture git state
GIT_SHA=$(cd "${AMPLIHACK_ROOT}" && git rev-parse HEAD)
GIT_BRANCH=$(cd "${AMPLIHACK_ROOT}" && git rev-parse --abbrev-ref HEAD)

# Write metadata
cat > "${RESULTS_DIR}/metadata.json" << EOF
{
    "tag": "${TAG_NAME}",
    "timestamp": "${TIMESTAMP}",
    "git_sha": "${GIT_SHA}",
    "git_branch": "${GIT_BRANCH}",
    "config": {
        "agents": ${AGENTS},
        "turns": ${TURNS},
        "questions": ${QUESTIONS},
        "seed": ${SEED},
        "agents_per_app": ${AGENTS_PER_APP},
        "grader_model": "${GRADER_MODEL}",
        "answer_timeout": ${ANSWER_TIMEOUT},
        "hive_name": "${HIVE_NAME}",
        "resource_group": "${RESOURCE_GROUP}",
        "location": "${LOCATION}"
    },
    "duration_seconds": ${EVAL_DURATION},
    "rerun_command": "ANTHROPIC_API_KEY=\\\"...\\\" SKIP_DEPLOY=1 HIVE_RESOURCE_GROUP=${RESOURCE_GROUP} ./run_distributed_eval.sh --agents ${AGENTS} --turns ${TURNS} --questions ${QUESTIONS} --seed ${SEED}"
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
git checkout ${GIT_SHA}
ANTHROPIC_API_KEY=\"...\" SKIP_DEPLOY=1 \\
  ./run_distributed_eval.sh --agents ${AGENTS} --turns ${TURNS} --questions ${QUESTIONS} --seed ${SEED}
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
