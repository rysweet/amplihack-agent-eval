# Eval Recipes

Canonical home for the eval recipes. Previously these lived in
`rysweet/amplihack` and `rysweet/amplihack-rs` (`amplifier-bundle/recipes/`)
which made the recipes co-tenant with non-eval code and forced the main
amplihack Python package to keep its `amplihack.eval.*` modules around.

## Recipes

- **`domain-agent-eval.yaml`** — Evaluates the 5 domain agents
  (CodeReview, MeetingSynthesizer, DocumentCreator, DataAnalysis, ProjectPlanning)
  on their L1–L4 scenarios plus a teaching-session evaluation, then writes
  combined scores.

- **`long-horizon-memory-eval.yaml`** — 1000-turn memory stress test with
  a self-improvement loop (eval → analyze → diagnose → fix → re-eval).

## Running

```bash
amplihack recipe run recipes/domain-agent-eval.yaml \
  -c repo_path="."

amplihack recipe run recipes/long-horizon-memory-eval.yaml \
  -c num_turns="1000" \
  -c repo_path="."
```

## Dependencies — current state

Both recipes are partially ported to the `amplihack_eval` package shipped
with this repo:

| Recipe symbol                                | Status     | Source                                       |
| -------------------------------------------- | ---------- | -------------------------------------------- |
| `amplihack_eval.data.long_horizon`           | ✅ here    | `src/amplihack_eval/data/long_horizon.py`    |
| `amplihack_eval.data.progressive_levels`     | ✅ here    | `src/amplihack_eval/data/progressive_levels.py` |
| `amplihack.eval.run_domain_evals`            | ⏳ upstream | needs port from `rysweet/amplihack` |
| `amplihack.eval.teaching_session`            | ⏳ upstream | needs port from `rysweet/amplihack` |
| `amplihack.eval.teaching_eval`               | ⏳ upstream | needs port from `rysweet/amplihack` |
| `amplihack.eval.progressive_test_suite`      | ⏳ upstream | needs port from `rysweet/amplihack` |
| `amplihack.eval.long_horizon_memory`         | ⏳ upstream | needs port from `rysweet/amplihack` |
| `amplihack.eval.long_horizon_self_improve`   | ⏳ upstream | needs port from `rysweet/amplihack` |

The upstream-pending modules also depend on `amplihack.agents.domain_agents.*`
and `amplihack.llm`, so a full port requires either bringing those packages
into this repo as a sub-dependency or factoring out a shared agent-runtime
crate. Tracked as separate porting issues.

## Migration history

Recipes relocated to this repo on 2026-04-22 in response to
`rysweet/amplihack-rs#284`. The original copies in
`rysweet/amplihack/amplifier-bundle/recipes/` and
`rysweet/amplihack-rs/amplifier-bundle/recipes/` are removed in favor of
this canonical location.
