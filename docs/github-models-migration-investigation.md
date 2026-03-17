# GitHub Models / gh LM API Migration Investigation

**Issue**: [#40](https://github.com/rysweet/amplihack-agent-eval/issues/40)
**Date**: 2026-03-17
**Status**: Investigation Complete — Implementation Pending
**Constraint**: The currently running 5000-turn distributed eval MUST NOT be interrupted.

---

## Executive Summary

Migrating the benchmark/eval grading workloads from the Anthropic Python SDK to GitHub Models (OpenAI-compatible endpoint at `https://models.inference.ai.azure.com`) requires **6 files** to change, targeting **7 distinct `client.messages.create()` call sites**. The subject under evaluation — `LearningAgent` — already uses `litellm` (provider-agnostic) so only the grader layer requires SDK changes.

The migration is viable with **one breaking compatibility gap** (`system=` top-level parameter in `grader_agent.py`) and **one quality risk** (no Claude model available on GitHub Models free tier — only `gpt-4o`, `gpt-4o-mini`, and open-source LLaMA/Mistral models are present). The switch can be performed without touching the running eval.

---

## 1. All Eval Entry Points and Their SDK Dependencies

### 1.1 CLI Entry Point

**File**: `src/amplihack_eval/cli.py:368–561`
**Command**: `amplihack-eval` (registered in `pyproject.toml:43`)

| Subcommand | Lines | Delegates To | Anthropic SDK? |
|---|---|---|---|
| `run` | 30–119 | `EvalRunner` | Indirect (via grader) |
| `compare` | 121–158 | `run_multi_seed_eval()` | Indirect (via grader) |
| `self-improve` | 161–197 | `run_self_improve()` | Indirect (via grader) |
| `report` | 200–211 | Print saved JSON | None |
| `download-dataset` | 214–237 | `DatasetDownloader` | None |
| `list-datasets` | 240–267 | `DatasetLister` | None |
| `continuous` | 270–305 | `run_continuous_eval()` | Indirect (via grader) |

### 1.2 Core Evaluation Files

| File | Lines | Anthropic SDK Usage | Notes |
|---|---|---|---|
| `src/amplihack_eval/core/grader.py` | 225–231, 178 | Direct — `anthropic.Anthropic()` + `client.messages.create()` | Main grader, used by all eval types |
| `src/amplihack_eval/core/runner.py` | 356–368, 408–412 | Direct — `anthropic.Anthropic()` + `client.messages.create()` | Dimension-based grading |
| `src/amplihack_eval/core/continuous_eval.py` | None direct | None — delegates to above | Calls `grade_answer()` from grader |
| `src/amplihack_eval/core/multi_seed.py` | None direct | None | Orchestrates EvalRunner |

### 1.3 Multi-Agent Evaluation Files

| File | Lines | Anthropic SDK Usage | Special Parameters |
|---|---|---|---|
| `src/amplihack_eval/multi_agent_eval/grader_agent.py` | 257–268, 280–285 | Direct — `anthropic.Anthropic()` + `client.messages.create()` | **Uses `system=self._system_prompt` top-level param** |
| `src/amplihack_eval/multi_agent_eval/adversary_agent.py` | 268–275, 314–318 (fn1) and 350–357, 382–387 (fn2) | Direct — two separate functions each create a client | No system param |
| `src/amplihack_eval/multi_agent_eval/analyst_agent.py` | 545–547 | Direct — `anthropic.Anthropic()` + `client.messages.create()` | No system param |
| `src/amplihack_eval/multi_agent_eval/pipeline.py` | None direct | None — orchestrates above agents | |
| `src/amplihack_eval/multi_agent_eval/coordinator.py` | None direct | None — orchestrates above agents | |

### 1.4 Agent Adapter

| File | Lines | Anthropic SDK Usage | Notes |
|---|---|---|---|
| `src/amplihack_eval/adapters/learning_agent.py` | 54–56 | None — imports `amplihack.agents.goal_seeking.learning_agent.LearningAgent` | Model name passed as string; agent uses litellm internally |
| `src/amplihack_eval/adapters/hive_mind_adapter.py` | — | None | |
| `src/amplihack_eval/adapters/http_adapter.py` | — | None | |
| `src/amplihack_eval/adapters/subprocess_adapter.py` | — | None | |
| `src/amplihack_eval/adapters/distributed_hive_adapter.py` | — | None | |

### 1.5 Currently Running Eval (DO NOT MODIFY)

The active 5000-turn eval runs **outside** `amplihack-agent-eval`:

```
/home/azureuser/src/amplihack/deploy/azure_hive/eval_distributed.py
  --connection-string Endpoint=sb://hive-eh-atmex3kma74gg.servicebus.windows.net/...
  --input-hub hive-events-amplihive3175
  --response-hub eval-responses-amplihive3175
  --turns 5000  --questions 50  --agents 10  --seed 42
  --grader-model claude-haiku-4-5-20251001
  --output /tmp/pr3175_eval_dist_5000turns_2d4257cf3_clean_lh5000_2d4257cf3_rerun_162705.json
```

**Grader model**: `claude-haiku-4-5-20251001` (Anthropic SDK, hardcoded in `eval_distributed.py:54`).
This eval uses the `amplihack` repo's own grader, not `amplihack-agent-eval`'s `core/grader.py`.
**Do not change `eval_distributed.py` or any files in `amplihack/deploy/` until this eval completes.**

---

## 2. GoalSeekingAgent Control Flow

### 2.1 LearningAgent Architecture

The subject under evaluation is `amplihack.agents.goal_seeking.learning_agent.LearningAgent` (not in this repo). Key properties:

- **SDK**: Uses `litellm.completion()` — **provider-agnostic**
- **Model format**: litellm model string passed as `model=` parameter
- **Entry point in this repo**: `src/amplihack_eval/adapters/learning_agent.py:73`

### 2.2 End-to-End Control Flow (Single Agent, Standard Eval)

```
amplihack-eval run \
    --adapter learning-agent \
    --model claude-sonnet-4-5-20250929 \
    --grader-model claude-sonnet-4-5-20250929
        │
        ▼
cli.py:_cmd_run()
    │  Creates adapter via _create_adapter()
    ▼
adapters/learning_agent.py:LearningAgentAdapter.__init__()
    │  Imports and instantiates LearningAgent(model="claude-sonnet-4-5-20250929")
    │  LearningAgent uses litellm.completion(model=...) internally
    ▼
core/runner.py:EvalRunner.run()
    │  For each (content_block, question):
    │    1. adapter.learn(content)        → LearningAgent.learn_from_content()
    │    2. adapter.answer(question)      → LearningAgent.answer_question()
    │    3. grade_answer(question, expected, actual, level)
    │         ▼
    │    core/grader.py:grade_answer()
    │         anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    │         client.messages.create(model=GRADER_MODEL, ...)
    │         message.content[0].text → parse JSON → GradeResult
    └── Return EvalReport
```

### 2.3 End-to-End Control Flow (Continuous / Multi-Agent)

```
amplihack-eval continuous
        │
        ▼
core/continuous_eval.py:run_continuous_eval()
    │  Conditions: single | flat | federated
    │
    ├── _run_single(): 1 LearningAgent instance
    ├── _run_flat():   N LearningAgent instances + InMemoryHiveGraph
    └── _run_federated(): N LearningAgent instances + DistributedHiveGraph
    │
    For each condition:
        adapt → run EvalRunner
                    ▼
                core/grader.py  (Anthropic SDK)
                core/runner.py  (Anthropic SDK — dimension grading)
```

### 2.4 Multi-Agent Eval Flow

```
amplihack-eval run --adapter learning-agent (with multi-agent pipeline enabled)
        │
        ▼
multi_agent_eval/coordinator.py:EvalCoordinator.run_eval()
    │
    ├── adversary_agent.py:AdversaryAgent.generate_adversarial_questions()
    │       Anthropic SDK — generates hard follow-up questions
    ├── adversary_agent.py:AdversaryAgent.generate_forgetting_probes()
    │       Anthropic SDK — generates memory-decay probe questions
    ├── grader_agent.py:GraderAgent._grade_with_llm()  [multiple perspectives]
    │       Anthropic SDK — perspective-aware grading
    │       *** Uses system= top-level param (Anthropic-specific) ***
    └── analyst_agent.py:AnalystAgent.analyze()
            Anthropic SDK — pattern analysis across results
```

---

## 3. GitHub Models / gh LM API Compatibility Gaps

GitHub Models exposes an **OpenAI-compatible** endpoint:

- **Base URL**: `https://models.inference.ai.azure.com`
- **Auth**: `Authorization: Bearer $GITHUB_TOKEN` (PAT or `gh auth token`)
- **SDK**: `openai` Python SDK (not `anthropic`)

### 3.1 Available Models (as of 2026-03-17)

Confirmed via `gh api https://models.inference.ai.azure.com/models`:

| Model | Task | Notes |
|---|---|---|
| `gpt-4o` | chat-completion | Best quality, higher cost |
| `gpt-4o-mini` | chat-completion | Cheaper, adequate for grading |
| `Meta-Llama-3.1-405B-Instruct` | chat-completion | Large open-source model |
| `Meta-Llama-3.1-70B-Instruct` | chat-completion | |
| `Meta-Llama-3.1-8B-Instruct` | chat-completion | |
| `Meta-Llama-3-70B-Instruct` | chat-completion | |
| `Meta-Llama-3-8B-Instruct` | chat-completion | |
| `Mistral-large-2407` | chat-completion | |
| `Mistral-Nemo` | chat-completion | |
| `AI21-Jamba-Instruct` | chat-completion | |

**No Claude model is available on GitHub Models.** The closest quality equivalent for grading tasks is `gpt-4o` or `gpt-4o-mini`.

### 3.2 API Incompatibility Matrix

| Aspect | Anthropic SDK | GitHub Models (OpenAI SDK) | Breaking? |
|---|---|---|---|
| **Package** | `anthropic>=0.30.0` | `openai>=1.0.0` | YES — different package |
| **Client init** | `anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)` | `openai.OpenAI(base_url="https://models.inference.ai.azure.com", api_key=GITHUB_TOKEN)` | YES |
| **Auth env var** | `ANTHROPIC_API_KEY` | `GITHUB_TOKEN` (or `GH_TOKEN`) | YES |
| **Request method** | `client.messages.create(...)` | `client.chat.completions.create(...)` | YES |
| **System prompt** | `system="..."` top-level param | `{"role": "system", "content": "..."}` in messages list | YES (affects `grader_agent.py:283`) |
| **Response access** | `message.content[0].text` | `response.choices[0].message.content` | YES |
| **Streaming** | Not used in eval pipeline | Not needed (no streaming in eval) | No change needed |
| **Tool use** | Not used in eval pipeline | Not needed | No change needed |
| **Model names** | `claude-sonnet-4-5-20250929` | `gpt-4o`, `gpt-4o-mini`, etc. | YES |
| **max_tokens param** | `max_tokens=...` | `max_tokens=...` | No change |
| **messages format** | `[{"role": "user", "content": "..."}]` | Same format | No change |

### 3.3 Grader-Specific `system=` Parameter Issue

**Location**: `src/amplihack_eval/multi_agent_eval/grader_agent.py:283`
**Current code**:
```python
message = client.messages.create(
    model=self.model,
    max_tokens=500,
    system=self._system_prompt,          # ← Anthropic top-level param
    messages=[{"role": "user", "content": prompt}],
)
```
**Required change** for OpenAI SDK:
```python
response = client.chat.completions.create(
    model=self.model,
    max_tokens=500,
    messages=[
        {"role": "system", "content": self._system_prompt},  # ← moved into messages
        {"role": "user", "content": prompt},
    ],
)
```

---

## 4. Minimal Viable Migration Plan (Grader Layer Only)

This plan changes **only the grader layer**. The subject under evaluation (LearningAgent via litellm) is unchanged.

### 4.1 Required Code Changes

#### Change 1: `src/amplihack_eval/core/grader.py` (lines 225–184)

**Before:**
```python
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise OSError("ANTHROPIC_API_KEY environment variable is required for grading")

import anthropic  # type: ignore[import-untyped]
client = anthropic.Anthropic(api_key=api_key)
grader_model = os.environ.get("GRADER_MODEL", "claude-sonnet-4-5-20250929")
# ...
message = client.messages.create(
    model=model,
    max_tokens=500,
    messages=[{"role": "user", "content": prompt}],
)
response_text = message.content[0].text
```

**After:**
```python
github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
if not github_token:
    raise OSError("GITHUB_TOKEN environment variable is required for grading")

import openai
client = openai.OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=github_token,
)
grader_model = os.environ.get("GRADER_MODEL", "gpt-4o-mini")
# ...
response = client.chat.completions.create(
    model=model,
    max_tokens=500,
    messages=[{"role": "user", "content": prompt}],
)
response_text = response.choices[0].message.content
```

#### Change 2: `src/amplihack_eval/core/runner.py` (lines 356–414)

Same pattern as Change 1 — replace Anthropic client init and API call with OpenAI/GitHub Models equivalent.

- Replace `anthropic.Anthropic(api_key=...)` → `openai.OpenAI(base_url=..., api_key=...)`
- Replace `client.messages.create(...)` → `client.chat.completions.create(...)`
- Replace `message.content[0].text` → `response.choices[0].message.content`
- Replace `ANTHROPIC_API_KEY` → `GITHUB_TOKEN`

#### Change 3: `src/amplihack_eval/multi_agent_eval/grader_agent.py` (lines 257–287)

**Additional change needed**: Move `system=self._system_prompt` into the messages list:

```python
messages=[
    {"role": "system", "content": self._system_prompt},
    {"role": "user", "content": prompt},
]
```

All other Anthropic → OpenAI substitutions apply.

#### Change 4: `src/amplihack_eval/multi_agent_eval/adversary_agent.py` (two functions, lines 268–320 and 350–390)

Two separate client-creation patterns, both follow same substitution. No `system=` parameter used.

#### Change 5: `src/amplihack_eval/multi_agent_eval/analyst_agent.py` (lines 545+)

Same substitution pattern. Verify exact `system=` usage before applying.

#### Change 6: `pyproject.toml` (optional dependencies)

**Before:**
```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.30.0"]
openai = ["openai>=1.0.0"]
all = ["anthropic>=0.30.0", "openai>=1.0.0"]
```

**After (GitHub Models only):**
```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.30.0"]
github-models = ["openai>=1.0.0"]
openai = ["openai>=1.0.0"]
all = ["anthropic>=0.30.0", "openai>=1.0.0"]
```

The `openai` extra already exists — no new dependency needed.

### 4.2 LearningAgent Model Configuration

For the **subject under test** (LearningAgent), the `--model` / `EVAL_MODEL` value must be changed to match the litellm provider prefix format:

| Before (Anthropic) | After (GitHub Models via litellm) |
|---|---|
| `claude-sonnet-4-5-20250929` | `openai/gpt-4o` |
| `claude-opus-4-6` | `openai/gpt-4o` |
| `claude-haiku-4-5-20251001` | `openai/gpt-4o-mini` |

And set environment variable:
```bash
export OPENAI_API_BASE="https://models.inference.ai.azure.com"
export OPENAI_API_KEY="$GITHUB_TOKEN"
```

litellm routes based on model prefix — `openai/gpt-4o` tells litellm to use OpenAI-compatible endpoint with the configured `OPENAI_API_BASE`.

### 4.3 Summary Change Table

| File | Change Type | LOC Changed | Risk |
|---|---|---|---|
| `core/grader.py` | Client init + API call pattern | ~8 lines | Low |
| `core/runner.py` | Client init + API call pattern | ~8 lines | Low |
| `multi_agent_eval/grader_agent.py` | Client init + API call + `system=` param fix | ~12 lines | Medium (system param) |
| `multi_agent_eval/adversary_agent.py` | Client init + API call × 2 | ~16 lines | Low |
| `multi_agent_eval/analyst_agent.py` | Client init + API call | ~8 lines | Low |
| `pyproject.toml` | Add `github-models` extra | ~2 lines | None |
| **Total** | | **~54 lines** | Medium |

---

## 5. Blockers and Severity

### Blocker 1 — No Claude Equivalent on GitHub Models (SEVERITY: HIGH)

**Description**: GitHub Models free tier does not include any Claude model. The benchmark was designed and tuned with Claude graders (`claude-sonnet-4-5-20250929`). Grading quality with `gpt-4o` may differ, making results non-comparable with historical runs.

**Impact**: Benchmark scores before and after migration are **not directly comparable** without re-running baseline evals.

**Mitigation**:
1. Run a parallel 100-turn smoke test with both graders to establish a calibration delta before switching.
2. Accept `gpt-4o` as the new grader baseline and document the model change in eval reports.

### Blocker 2 — `system=` Top-Level Parameter in `grader_agent.py` (SEVERITY: MEDIUM)

**Description**: `grader_agent.py:283` uses `system=self._system_prompt` as a top-level Anthropic-specific parameter. OpenAI SDK does not accept this; it must be moved into the messages list as `{"role": "system", ...}`.

**Impact**: Runtime `TypeError` if not fixed before switching.

**Mitigation**: Simple mechanical fix — move `system=` into `messages[0]`.

### Blocker 3 — `ANTHROPIC_API_KEY` Error Messages in Graders (SEVERITY: LOW)

**Description**: All 5 grader files raise `OSError("ANTHROPIC_API_KEY environment variable is required...")` when key is missing. After migration, this error message is misleading.

**Impact**: Confusing error messages, not a runtime failure if key is absent.

**Mitigation**: Update error message to `GITHUB_TOKEN` in all 5 files (trivial).

### Blocker 4 — litellm Provider Prefix for LearningAgent (SEVERITY: MEDIUM)

**Description**: The LearningAgent uses litellm with model names like `claude-sonnet-4-5-20250929`. litellm auto-detects Anthropic provider from model name. For GitHub Models, the model name must include the provider prefix `openai/` and `OPENAI_API_BASE` must be set.

**Impact**: Without the prefix, litellm will attempt to reach Anthropic's API with a GitHub model name and fail.

**Mitigation**:
```bash
export EVAL_MODEL="openai/gpt-4o"
export OPENAI_API_BASE="https://models.inference.ai.azure.com"
export OPENAI_API_KEY="$GITHUB_TOKEN"
```

### Blocker 5 — Rate Limits / Quota on GitHub Models (SEVERITY: MEDIUM)

**Description**: GitHub Models free tier has per-minute and per-day rate limits. A 5000-turn eval with 50 questions × 1 grading call each = 250,000 grading calls. This likely exceeds free tier quota.

**Impact**: Eval may fail partway through due to 429 rate-limit errors.

**Mitigation**:
1. Test with a 100-turn smoke run first.
2. The existing `_single_grade_call()` in `grader.py` already catches exceptions per-vote — add a retry with exponential backoff.
3. Consider GitHub Models paid tier or Azure AI inference endpoint for large evals.

---

## 6. Concrete Validation Steps

These steps confirm a successful migration **without touching the running 5000-turn eval**.

### Step 1: Verify GitHub Token and Models Access

```bash
# Confirm gh CLI is authenticated
gh auth status

# Confirm GitHub Models endpoint is reachable
curl -s -H "Authorization: Bearer $(gh auth token)" \
  https://models.inference.ai.azure.com/models | python3 -c "import json,sys; models=json.load(sys.stdin); print([m['name'] for m in models[:5]])"

# Quick smoke test — single API call
python3 -c "
import openai, os
client = openai.OpenAI(
    base_url='https://models.inference.ai.azure.com',
    api_key='$(gh auth token)',
)
r = client.chat.completions.create(
    model='gpt-4o-mini',
    max_tokens=50,
    messages=[{'role': 'user', 'content': 'Reply with just OK.'}],
)
print(r.choices[0].message.content)
"
```

### Step 2: Install openai Package

```bash
# In the amplihack-agent-eval venv
pip install 'amplihack-agent-eval[openai]'
# or directly:
pip install 'openai>=1.0.0'
```

### Step 3: Unit-Test Grader with GitHub Models

```bash
export GITHUB_TOKEN="$(gh auth token)"
export GRADER_MODEL="gpt-4o-mini"

# Run grader unit test with GitHub Models
python3 -c "
import os
os.environ['GITHUB_TOKEN'] = os.environ.get('GITHUB_TOKEN', '')
os.environ['GRADER_MODEL'] = 'gpt-4o-mini'

# After applying migration changes to core/grader.py:
from amplihack_eval.core.grader import grade_answer
result = grade_answer(
    question='What is the capital of France?',
    expected='Paris',
    actual='The capital of France is Paris.',
    level='L1',
)
print(f'Score: {result.score}, Reasoning: {result.reasoning}')
assert result.score >= 0.8, f'Expected score >= 0.8, got {result.score}'
print('PASS: grader works with GitHub Models')
"
```

### Step 4: 100-Turn Smoke Eval (Parallel, Non-Disruptive)

Run a small-scale eval alongside the running 5000-turn eval to validate end-to-end:

```bash
export GITHUB_TOKEN="$(gh auth token)"
export GRADER_MODEL="gpt-4o-mini"
export EVAL_MODEL="openai/gpt-4o-mini"
export OPENAI_API_BASE="https://models.inference.ai.azure.com"
export OPENAI_API_KEY="$GITHUB_TOKEN"

amplihack-eval run \
  --adapter learning-agent \
  --model "$EVAL_MODEL" \
  --grader-model "$GRADER_MODEL" \
  --turns 100 \
  --output /tmp/github-models-smoke-$(date +%s).json

# Compare scores to baseline (from a previous Anthropic-graded run)
python3 -c "
import json
with open('/tmp/eval-sonnet-5000-FINAL.json') as f:  # adjust path
    baseline = json.load(f)
# load new result and compare
print('Baseline overall score:', baseline.get('overall_score'))
"
```

### Step 5: Verify Running Eval Unaffected

```bash
# Check the running 5000-turn eval is still running
cat /tmp/current_eval_pid.txt 2>/dev/null
ps aux | grep eval_distributed | grep -v grep

# Check its output log for recent activity
tail -5 /tmp/pr3175_eval_dist_5000turns_2d4257cf3_clean_lh5000_2d4257cf3_rerun_162705.log 2>/dev/null || \
  echo "Check eval log directly via: ps aux | grep eval_distributed"
```

### Step 6: Calibration Run (Before Full Migration)

To quantify grading quality delta between Anthropic and GitHub Models:

```bash
# Run same dataset with Anthropic grader (existing)
GRADER_MODEL=claude-sonnet-4-5-20250929 \
  amplihack-eval run --adapter learning-agent --turns 300 \
  --output /tmp/calibration-anthropic.json

# Run same dataset with GitHub Models grader (migrated)
GRADER_MODEL=gpt-4o-mini \
  amplihack-eval run --adapter learning-agent --turns 300 \
  --output /tmp/calibration-github-models.json

# Compare
python3 -c "
import json
a = json.load(open('/tmp/calibration-anthropic.json'))
g = json.load(open('/tmp/calibration-github-models.json'))
print(f'Anthropic: {a.get(\"overall_score\", 0):.3f}')
print(f'GitHub Models: {g.get(\"overall_score\", 0):.3f}')
print(f'Delta: {abs(a.get(\"overall_score\", 0) - g.get(\"overall_score\", 0)):.3f}')
"
```

---

## 7. Is Provider-Only Swap Sufficient?

**Short answer: No.** The following additional changes are required beyond provider swap:

1. **`system=` parameter fix** in `grader_agent.py:283` — required for OpenAI SDK compatibility.
2. **litellm model prefix** — `EVAL_MODEL` must use `openai/gpt-4o` format, not bare model name.
3. **Model name defaults** — all 7 default `claude-sonnet-4-5-20250929` strings must be updated to a GitHub Models model name (e.g., `gpt-4o-mini`).
4. **Error message updates** — all `ANTHROPIC_API_KEY` error strings should reference `GITHUB_TOKEN`.
5. **Calibration baseline** — scores are not comparable across grader models without establishing a calibration delta first.

---

## 8. Files Not Requiring Changes

These files are unaffected:

- `src/amplihack_eval/adapters/learning_agent.py` — uses litellm, no Anthropic SDK
- `src/amplihack_eval/adapters/hive_mind_adapter.py` — no SDK usage
- `src/amplihack_eval/adapters/http_adapter.py` — no SDK usage
- `src/amplihack_eval/adapters/subprocess_adapter.py` — no SDK usage
- `src/amplihack_eval/adapters/distributed_hive_adapter.py` — no SDK usage
- `src/amplihack_eval/core/continuous_eval.py` — delegates to graders only
- `src/amplihack_eval/multi_agent_eval/coordinator.py` — orchestrates only
- `src/amplihack_eval/multi_agent_eval/pipeline.py` — orchestrates only
- All `src/amplihack_eval/data/*.py` — static data
- All `src/amplihack_eval/levels/*.py` — level definitions

---

## 9. Recommended Migration Order

1. Apply Change 3 first (`grader_agent.py` — the breaking `system=` fix)
2. Apply Changes 1 and 2 (`core/grader.py`, `core/runner.py`)
3. Apply Changes 4 and 5 (`adversary_agent.py`, `analyst_agent.py`)
4. Apply Change 6 (`pyproject.toml`)
5. Run Step 1–4 of validation
6. Update model defaults in all 7 locations
7. Run calibration (Step 6)
8. Decide on full rollout based on delta

---

## 10. References

- GitHub Models endpoint: `https://models.inference.ai.azure.com`
- GitHub Models documentation: [GitHub Marketplace Models](https://github.com/marketplace/models)
- litellm OpenAI-compatible providers: `openai/` prefix with `OPENAI_API_BASE`
- Issue: [#40](https://github.com/rysweet/amplihack-agent-eval/issues/40)
