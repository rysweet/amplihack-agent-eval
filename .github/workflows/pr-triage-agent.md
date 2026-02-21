---
description: Categorizes and reviews PRs for the agent evaluation framework
on:
  pull_request:
    types: [opened, synchronize]
permissions:
  contents: read
  pull-requests: read
engine: copilot
tools:
  github:
    lockdown: true
    toolsets: [pull_requests, repos]
safe-outputs:
  add-labels:
    max: 3
    allowed: [eval-levels, adapters, core, self-improve, ci, docs, breaking-change]
  add-comment:
    max: 1
timeout-minutes: 5
strict: true
---

# PR Triage Agent

**Description:** Automatically categorizes and provides review guidance for PRs.

## Objective

Analyze incoming pull requests and:
1. Add appropriate labels based on changed files
2. Flag breaking changes to the AgentAdapter interface
3. Provide a brief summary comment

## Label Rules

### File-Path Based Classification

- `src/amplihack_eval/levels/` or `src/amplihack_eval/data/progressive_levels.py` -> `eval-levels`
- `src/amplihack_eval/adapters/` -> `adapters`
- `src/amplihack_eval/core/` -> `core`
- `src/amplihack_eval/self_improve/` -> `self-improve`
- `.github/workflows/` -> `ci`
- `docs/` or `README.md` -> `docs`

### Breaking Change Detection

Flag as `breaking-change` if:
- `AgentAdapter` abstract methods are added or modified
- `AgentResponse` or `ToolCall` fields are removed or renamed
- `EvalRunner` constructor signature changes
- Public API exports in `__init__.py` are removed

## Summary Comment

Provide a 2-3 sentence summary of what the PR changes and which components are affected. Mention any potential risks or areas that need careful review.
