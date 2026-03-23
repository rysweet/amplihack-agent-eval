"""Tests for scale-aware distributed eval defaults."""

import builtins
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from amplihack_eval.adapters import remote_agent_adapter
from amplihack_eval.azure import eval_distributed


def _install_fake_long_horizon(monkeypatch, fake_report, fake_run):
    module = types.ModuleType("amplihack.eval.long_horizon_memory")

    class FakeLongHorizonMemoryEval:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, adapter, grader_model=""):
            return fake_run(self, adapter, grader_model=grader_model)

    module.LongHorizonMemoryEval = FakeLongHorizonMemoryEval
    module._print_report = lambda report: None

    eval_package = types.ModuleType("amplihack.eval")
    eval_package.long_horizon_memory = module
    amplihack_package = types.ModuleType("amplihack")
    amplihack_package.eval = eval_package

    monkeypatch.setitem(sys.modules, "amplihack", amplihack_package)
    monkeypatch.setitem(sys.modules, "amplihack.eval", eval_package)
    monkeypatch.setitem(sys.modules, "amplihack.eval.long_horizon_memory", module)


class TestScaleAwareDefaults:
    def test_agents_per_app_default_scales_by_profile(self, monkeypatch):
        monkeypatch.delenv("AMPLIHACK_AGENTS_PER_APP", raising=False)
        monkeypatch.delenv("HIVE_AGENTS_PER_APP", raising=False)
        monkeypatch.delenv("HIVE_DEPLOYMENT_PROFILE", raising=False)
        assert eval_distributed._default_agents_per_app() == 5

        monkeypatch.setenv("HIVE_DEPLOYMENT_PROFILE", "smoke-10")
        assert eval_distributed._default_agents_per_app() == 1

    def test_parallel_workers_default_scales_down(self):
        assert eval_distributed._default_parallel_workers(10) == 10
        assert eval_distributed._default_parallel_workers(50) == 2
        assert eval_distributed._default_parallel_workers(100) == 1

    def test_failover_retries_default_scales_up(self):
        assert eval_distributed._default_question_failover_retries(10) == 1
        assert eval_distributed._default_question_failover_retries(50) == 1
        assert eval_distributed._default_question_failover_retries(100) == 2

    def test_answer_timeout_default_disables_for_100_agent_runs(self):
        assert eval_distributed._default_answer_timeout(10) == 120
        assert eval_distributed._default_answer_timeout(50) == 120
        assert eval_distributed._default_answer_timeout(100) == 0

    def test_help_text_matches_small_cluster_failover_default(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["eval_distributed", "--help"])

        with pytest.raises(SystemExit):
            eval_distributed.main()

        captured = capsys.readouterr()
        assert "1 up to 49 agents" in captured.out
        assert "1 for 50-99" in captured.out
        assert "2 for 100+" in captured.out

    def test_main_accepts_question_set_and_writes_report(self, monkeypatch, tmp_path):
        output_path = tmp_path / "eval_report.json"
        fake_report = MagicMock()
        fake_report.to_dict.return_value = {"overall_score": 1.0}

        class FakeAdapter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def close(self):
                return None

        def fake_run(self, adapter, grader_model=""):
            return fake_report

        _install_fake_long_horizon(monkeypatch, fake_report, fake_run)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval_distributed",
                "--connection-string",
                "Endpoint=sb://example/;SharedAccessKeyName=test;SharedAccessKey=value",
                "--input-hub",
                "hive-events-test",
                "--response-hub",
                "eval-responses-test",
                "--question-set",
                "holdout",
                "--output",
                str(output_path),
            ],
        )
        monkeypatch.setattr(remote_agent_adapter, "RemoteAgentAdapter", FakeAdapter)

        assert eval_distributed.main() == 0

        report = json.loads(output_path.read_text())
        assert report["question_set"] == "holdout"

    def test_main_reports_missing_amplihack_dependency(self, monkeypatch, capsys):
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "amplihack.eval.long_horizon_memory":
                raise ImportError("No module named 'amplihack'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval_distributed",
                "--connection-string",
                "Endpoint=sb://example/;SharedAccessKeyName=test;SharedAccessKey=value",
            ],
        )
        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert eval_distributed.main() == 1
        captured = capsys.readouterr()
        assert "requires the sibling amplihack package" in captured.err
