from __future__ import annotations

from argparse import Namespace

import amplihack_eval.core.continuous_eval as continuous_eval
from amplihack_eval.cli import _cmd_continuous
from amplihack_eval.core.continuous_eval import _MultiAgentAdapter
from amplihack_eval.core.runner import MAX_PARALLEL_WORKERS


class TestContinuousEvalValidation:
    def test_cmd_continuous_rejects_empty_conditions(self, tmp_path, capsys):
        args = Namespace(
            verbose=False,
            output_dir=str(tmp_path),
            conditions="   ",
            turns=10,
            questions=5,
            agents=2,
            groups=1,
            seed=42,
            model="",
            parallel_workers=5,
            prompt_variant=None,
            repeats=1,
        )

        exit_code = _cmd_continuous(args)
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "--conditions cannot be empty" in captured.out

    def test_cmd_continuous_rejects_unknown_conditions(self, tmp_path, capsys):
        args = Namespace(
            verbose=False,
            output_dir=str(tmp_path),
            conditions="single,unknown",
            turns=10,
            questions=5,
            agents=2,
            groups=1,
            seed=42,
            model="",
            parallel_workers=5,
            prompt_variant=None,
            repeats=1,
        )

        exit_code = _cmd_continuous(args)
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Invalid --conditions value" in captured.out

    def test_cmd_continuous_reports_missing_amplihack_dependency(self, tmp_path, capsys, monkeypatch):
        args = Namespace(
            verbose=False,
            output_dir=str(tmp_path),
            conditions="single",
            turns=10,
            questions=5,
            agents=2,
            groups=1,
            seed=42,
            model="",
            parallel_workers=5,
            prompt_variant=None,
            repeats=1,
        )

        def _raise_import_error(**kwargs):
            raise ImportError("No module named 'amplihack'")

        monkeypatch.setattr(continuous_eval, "run_continuous_eval", _raise_import_error)

        exit_code = _cmd_continuous(args)
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "requires the sibling amplihack package" in captured.err


class TestMultiAgentAdapterWorkerClamp:
    def test_parallel_workers_clamped_to_minimum(self):
        adapter = _MultiAgentAdapter([object()], model="", parallel_workers=0)
        assert adapter._parallel_workers == 1

    def test_parallel_workers_clamped_to_maximum(self):
        adapter = _MultiAgentAdapter([object()], model="", parallel_workers=100)
        assert adapter._parallel_workers == MAX_PARALLEL_WORKERS
