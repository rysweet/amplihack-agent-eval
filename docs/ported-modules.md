# Ported Eval Modules (issue #48)

This document tracks modules ported from the upstream `amplihack.eval.*`
namespace into `amplihack_eval.eval.*` so this repo owns them directly.

## Status

| Upstream module                           | Local module                                 | Status     |
| ----------------------------------------- | -------------------------------------------- | ---------- |
| `amplihack.eval.progressive_test_suite`   | `amplihack_eval.eval.progressive_test_suite` | ✅ ported  |
| `amplihack.eval.long_horizon_memory`      | `amplihack_eval.eval.long_horizon_memory`    | ✅ ported  |
| `amplihack.eval.long_horizon_self_improve`| `amplihack_eval.eval.long_horizon_self_improve` | ✅ ported |
| `amplihack.eval.grader`                   | `amplihack_eval.eval.grader`                 | ✅ ported  |
| `amplihack.eval.llm_grader`               | `amplihack_eval.eval.llm_grader`             | ✅ ported  |
| `amplihack.eval.metacognition_grader`     | `amplihack_eval.eval.metacognition_grader`   | ✅ ported  |
| `amplihack.llm.completion`                | `amplihack_eval.llm.completion` (lazy shim)  | ✅ shimmed |
| `amplihack.eval.run_domain_evals`         | —                                            | ⏳ deferred |
| `amplihack.eval.teaching_session`         | —                                            | ⏳ deferred |
| `amplihack.eval.teaching_eval`            | —                                            | ⏳ deferred |
| `amplihack.eval.domain_eval_harness`      | —                                            | ⏳ deferred |
| `amplihack.eval.agent_subprocess`         | —                                            | ⏳ deferred |
| `amplihack.eval.teaching_subprocess`      | —                                            | ⏳ deferred |

## LLM shim contract (`amplihack_eval.llm`)

`amplihack_eval.llm` exposes a single name, `completion`, as a *lazy*
re-export of `amplihack.llm.completion`:

- Importing `amplihack_eval.llm` is **import-time safe** — it does not
  itself import `amplihack`.
- The first call to `await completion(...)` resolves `amplihack.llm.completion`
  and forwards arguments. If the upstream package is unavailable, an
  `ImportError` is raised **at call time** with an actionable install
  hint (`pip install amplihack`).

This keeps the ported eval modules importable in test environments that
do not have the upstream `amplihack` package installed.

## Grader disambiguation

Two distinct `Grader`-style modules coexist in this repo. They are
separate code paths with different upstream sources and different LLM
backends; the right one to use depends on context.

| Module                              | Class / functions          | LLM backend                       | Origin                       |
| ----------------------------------- | -------------------------- | --------------------------------- | ---------------------------- |
| `amplihack_eval.core.grader`        | `GradeResult`, `grade_answer` | Anthropic SDK (direct)         | Native to this repo          |
| `amplihack_eval.eval.grader`        | `GradeResult`, `Grader`, `grade_answer` | Routed via `amplihack_eval.llm.completion` (LLM router) | Ported from `amplihack.eval.grader` (issue #48) |

Both are exercised by code in this repo; they are not interchangeable.
Tests in `tests/test_ported_eval_modules.py` lock down the distinction.

## Deferred follow-up work

The following items are intentionally out of scope for issue #48 and
tracked separately:

- **`amplihack.agents.domain_agents.*`** — depended on by
  `run_domain_evals`, `teaching_session`, `teaching_eval`, and the
  subprocess targets. Likely needs a separate `amplihack-agents`
  package or in-repo port.
- **`amplihack.agents.goal_seeking.runtime_factory` / `learning_agent`
  / `sub_agents`** — referenced by lazy imports inside
  `long_horizon_memory.run_eval` and
  `long_horizon_self_improve.run_long_horizon_self_improve`. They are
  imported only when the eval is actually executed; import-only smoke
  tests are unaffected.
- **Subprocess targets `amplihack.eval.{agent_subprocess,
  teaching_subprocess}`** — invoked via `python -m ...` from the ported
  `progressive_test_suite`. The string literals still reference the
  upstream module paths; running the learning/testing/teaching
  subprocesses still requires the upstream `amplihack` package on
  `PATH`. Module-level commentary in
  `src/amplihack_eval/eval/progressive_test_suite.py` documents this.

## Verification

`tests/test_ported_eval_modules.py` provides:

- Import-smoke checks for all 7 new modules / shim
- Shim contract checks (callable `completion`, lazy import behavior)
- Grader disambiguation (distinct namespaces)
- Progressive level lookup (L1 retrievable from a public container)
- Long-horizon CLI surface (`main`, `--multi-agent` flag)
- Recipe YAML rewrite checks (no stale `amplihack.eval.*` refs for
  ported leaf modules in either recipe)
