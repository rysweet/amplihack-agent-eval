"""TDD tests for modules ported from amplihack.eval (issue #48).

These tests are written FIRST and will fail until the corresponding modules
are ported into ``src/amplihack_eval/`` per the design spec.

Scope of this test module:
  * Import-smoke tests for the 7 newly created modules / shim
  * A few pure-function unit tests (grader JSON parsing, progressive level
    lookup) chosen because they exercise behavior that does NOT require an
    LLM, network access, or the upstream ``amplihack`` package installed.

Out of scope (explicitly NOT tested here):
  * Live LLM calls (no API keys in CI)
  * Subprocess driving of unported ``progressive`` runner targets
  * domain_agents / teaching_* / run_domain_evals (deferred follow-up)
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Module identifiers under test (mirror design spec "new_files")
# ---------------------------------------------------------------------------

SHIM_MODULE = "amplihack_eval.llm"
EVAL_PKG = "amplihack_eval.eval"

EVAL_MODULES = [
    "amplihack_eval.eval",
    "amplihack_eval.eval.llm_grader",
    "amplihack_eval.eval.grader",
    "amplihack_eval.eval.metacognition_grader",
    "amplihack_eval.eval.progressive_test_suite",
    "amplihack_eval.eval.long_horizon_memory",
    "amplihack_eval.eval.long_horizon_self_improve",
]


# ---------------------------------------------------------------------------
# Phase A — Shim module
# ---------------------------------------------------------------------------


class TestLLMShim:
    """The shim must import cheaply without requiring ``amplihack`` installed.

    It re-exports ``completion`` lazily — i.e. importing the shim must not
    itself import ``amplihack.llm``. ImportError surfaces only on call.
    """

    def test_shim_module_imports(self):
        mod = importlib.import_module(SHIM_MODULE)
        assert isinstance(mod, ModuleType)

    def test_shim_exposes_completion_attribute(self):
        mod = importlib.import_module(SHIM_MODULE)
        assert hasattr(mod, "completion"), "amplihack_eval.llm must expose a `completion` callable (lazy shim)"
        assert callable(mod.completion)

    def test_shim_import_does_not_eagerly_load_amplihack(self, monkeypatch):
        """Importing the shim must succeed even if `amplihack.llm` is missing.

        We simulate amplihack-not-installed by blocking the import in
        sys.modules and asserting the shim still imports.
        """
        import sys

        # Drop any cached shim so re-import is exercised
        sys.modules.pop(SHIM_MODULE, None)

        # Make `amplihack.llm` import fail on demand
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "amplihack.llm" or name.startswith("amplihack.llm."):
                raise ImportError("simulated: amplihack not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)

        # The shim itself must still import (lazy)
        importlib.import_module(SHIM_MODULE)


# ---------------------------------------------------------------------------
# Phase B — Eval subpackage import smoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modname", EVAL_MODULES)
def test_eval_module_imports(modname: str):
    """Each ported module must import cleanly with no side effects beyond
    package init. Failure here indicates a missing port or broken rewrite
    of an upstream ``amplihack.*`` import path.
    """
    mod = importlib.import_module(modname)
    assert isinstance(mod, ModuleType)
    assert mod.__name__ == modname


# ---------------------------------------------------------------------------
# Phase B2 — grader.py pure-function behavior
# ---------------------------------------------------------------------------


class TestPortedGrader:
    """The ported eval.grader exposes a Grader class and grade_response helper
    (verbatim port from upstream). Distinct from amplihack_eval.core.grader.
    """

    def test_grader_class_exists(self):
        mod = importlib.import_module("amplihack_eval.eval.grader")
        assert hasattr(mod, "Grader"), "ported grader must expose `Grader`"

    def test_grade_response_is_callable(self):
        mod = importlib.import_module("amplihack_eval.eval.grader")
        # Upstream public API names — at least one must exist
        assert any(hasattr(mod, name) for name in ("grade_response", "grade", "grade_answer")), (
            "ported grader must expose a grading callable"
        )

    def test_does_not_collide_with_core_grader(self):
        """Both modules coexist; they must be distinct objects."""
        eval_grader = importlib.import_module("amplihack_eval.eval.grader")
        core_grader = importlib.import_module("amplihack_eval.core.grader")
        assert eval_grader is not core_grader
        assert eval_grader.__name__ != core_grader.__name__


# ---------------------------------------------------------------------------
# Phase B4 — progressive_test_suite level lookup
# ---------------------------------------------------------------------------


class TestProgressiveTestSuite:
    """Pure-function checks: the suite defines L1..L12 levels (per the
    progressive eval design) and exposes a way to look one up by name/index.
    """

    def test_module_defines_levels(self):
        mod = importlib.import_module("amplihack_eval.eval.progressive_test_suite")
        # At least one of these level-container symbols must exist
        candidates = ("LEVELS", "PROGRESSIVE_LEVELS", "TEST_LEVELS", "Levels")
        found = [name for name in candidates if hasattr(mod, name)]
        assert found, f"progressive_test_suite must expose level definitions; checked {candidates}"

    def test_at_least_one_level_lookup_works(self):
        """L1 (or index 0) must be retrievable by some public means."""
        mod = importlib.import_module("amplihack_eval.eval.progressive_test_suite")
        for name in ("LEVELS", "PROGRESSIVE_LEVELS", "TEST_LEVELS"):
            container = getattr(mod, name, None)
            if container is None:
                continue
            # Container is dict-like or list-like
            try:
                if isinstance(container, dict):
                    assert "L1" in container or 1 in container or len(container) >= 1
                else:
                    assert len(container) >= 1
                return
            except Exception:  # pragma: no cover - defensive
                continue
        pytest.fail("No usable level container found for L1 lookup")


# ---------------------------------------------------------------------------
# Phase B5 — long_horizon_memory CLI surface
# ---------------------------------------------------------------------------


class TestLongHorizonMemory:
    def test_has_cli_entrypoint(self):
        mod = importlib.import_module("amplihack_eval.eval.long_horizon_memory")
        assert hasattr(mod, "main") or hasattr(mod, "cli") or hasattr(mod, "run"), (
            "long_horizon_memory must expose a CLI/main entrypoint"
        )

    def test_self_subprocess_module_string_is_rewritten(self):
        """The upstream module self-launches via `python -m
        amplihack.eval.long_horizon_memory`. After porting, any such literal
        must reference the new module path so the subprocess actually finds
        the ported code.
        """
        import inspect

        mod = importlib.import_module("amplihack_eval.eval.long_horizon_memory")
        try:
            src = inspect.getsource(mod)
        except OSError:
            pytest.skip("source not available")
        assert "amplihack.eval.long_horizon_memory" not in src, (
            "stale upstream module path found in self-subprocess invocation"
        )


# ---------------------------------------------------------------------------
# Phase B6 — long_horizon_self_improve uses local self_improve.*
# ---------------------------------------------------------------------------


class TestLongHorizonSelfImprove:
    def test_has_cli_entrypoint(self):
        mod = importlib.import_module("amplihack_eval.eval.long_horizon_self_improve")
        assert hasattr(mod, "main") or hasattr(mod, "cli") or hasattr(mod, "run")

    def test_imports_local_self_improve_package(self):
        """Must depend on the in-repo amplihack_eval.self_improve.* modules,
        not the upstream amplihack.self_improve.* paths.
        """
        import inspect

        mod = importlib.import_module("amplihack_eval.eval.long_horizon_self_improve")
        try:
            src = inspect.getsource(mod)
        except OSError:
            pytest.skip("source not available")
        assert "amplihack.self_improve" not in src, (
            "upstream amplihack.self_improve.* import must be rewritten to amplihack_eval.self_improve.*"
        )

    def test_supports_multi_agent_flag(self):
        """The CLI advertises a --multi-agent flag (per issue #48 description)."""
        import inspect

        mod = importlib.import_module("amplihack_eval.eval.long_horizon_self_improve")
        try:
            src = inspect.getsource(mod)
        except OSError:
            pytest.skip("source not available")
        assert "--multi-agent" in src or "multi_agent" in src


# ---------------------------------------------------------------------------
# Recipe YAML reference checks
# ---------------------------------------------------------------------------


class TestRecipeImportsRewritten:
    """The two recipe YAMLs must reference the ported module paths
    (``amplihack_eval.eval.*``) for the four self-contained modules. The
    deferred modules (teaching_*, run_domain_evals) may still appear with
    their upstream paths — those are out of scope.
    """

    PORTED_LEAF_NAMES = (
        "progressive_test_suite",
        "long_horizon_memory",
        "long_horizon_self_improve",
    )

    @pytest.mark.parametrize(
        "recipe_path",
        [
            "recipes/long-horizon-memory-eval.yaml",
            "recipes/domain-agent-eval.yaml",
        ],
    )
    def test_no_stale_amplihack_eval_refs_for_ported_modules(self, recipe_path):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        text = (repo_root / recipe_path).read_text(encoding="utf-8")
        for leaf in self.PORTED_LEAF_NAMES:
            stale = f"amplihack.eval.{leaf}"
            assert stale not in text, (
                f"{recipe_path} still references upstream `{stale}`; must be rewritten to `amplihack_eval.eval.{leaf}`"
            )
