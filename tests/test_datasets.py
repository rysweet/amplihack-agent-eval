"""Tests for datasets download and management.

Tests the download helper, metadata loading, and CLI integration.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from amplihack_eval.datasets.download import (
    _extract_tarball,
    _get_datasets_dir,
    list_datasets,
    load_metadata,
)


class TestLoadMetadata:
    """Test metadata loading from dataset directories."""

    def test_load_valid_metadata(self, tmp_path: Path):
        meta = {"name": "test-v1.0", "turns": 100, "seed": 42}
        (tmp_path / "metadata.json").write_text(json.dumps(meta))

        result = load_metadata(tmp_path)
        assert result["name"] == "test-v1.0"
        assert result["turns"] == 100
        assert result["seed"] == 42

    def test_load_missing_metadata(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_metadata(tmp_path)


class TestListDatasets:
    """Test listing local and remote datasets."""

    def test_list_local_datasets(self, tmp_path: Path):
        # Create a fake local dataset
        ds_dir = tmp_path / "test-v1.0"
        ds_dir.mkdir()
        meta = {"name": "test-v1.0", "turns": 100, "baseline_score": 0.85}
        (ds_dir / "metadata.json").write_text(json.dumps(meta))

        with patch("amplihack_eval.datasets.download._get_datasets_dir", return_value=tmp_path):
            datasets = list_datasets(include_remote=False)

        assert len(datasets) == 1
        assert datasets[0]["name"] == "test-v1.0"
        assert datasets[0]["local"] is True

    def test_list_empty_directory(self, tmp_path: Path):
        with patch("amplihack_eval.datasets.download._get_datasets_dir", return_value=tmp_path):
            datasets = list_datasets(include_remote=False)

        assert datasets == []


class TestExtractTarball:
    """Test tarball extraction with safety checks."""

    def test_extract_single_directory(self, tmp_path: Path):
        # Create a tarball with a single top-level directory
        src = tmp_path / "src"
        ds = src / "my-dataset"
        ds.mkdir(parents=True)
        (ds / "metadata.json").write_text('{"name": "my-dataset"}')
        (ds / "data.txt").write_text("test data")

        tarball = tmp_path / "test.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(ds, arcname="my-dataset")

        dest = tmp_path / "output" / "my-dataset"
        _extract_tarball(tarball, dest)

        assert dest.exists()
        assert (dest / "metadata.json").exists()
        assert (dest / "data.txt").read_text() == "test data"

    def test_extract_rejects_path_traversal(self, tmp_path: Path):
        # Create a tarball with path traversal
        tarball = tmp_path / "evil.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            # Add a file with path traversal in name
            import io

            info = tarfile.TarInfo(name="../../../etc/passwd")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"evil"))

        dest = tmp_path / "output"
        with pytest.raises(RuntimeError, match="Unsafe path"):
            _extract_tarball(tarball, dest)


class TestGetDatasetsDir:
    """Test datasets directory resolution."""

    def test_creates_directory(self, tmp_path: Path):
        with patch("amplihack_eval.datasets.download.DATASETS_DIR", tmp_path / "datasets"):
            result = _get_datasets_dir()
            assert result.exists()
            assert result.is_dir()


class TestCLIIntegration:
    """Test CLI commands for dataset management."""

    def test_list_datasets_command(self):
        """Verify list-datasets command is registered."""
        import subprocess

        result = subprocess.run(
            ["amplihack-eval", "list-datasets", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "local-only" in result.stdout

    def test_download_dataset_command(self):
        """Verify download-dataset command is registered."""
        import subprocess

        result = subprocess.run(
            ["amplihack-eval", "download-dataset", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "dataset_name" in result.stdout

    def test_run_skip_learning_flag(self):
        """Verify --skip-learning flag is registered."""
        import subprocess

        result = subprocess.run(
            ["amplihack-eval", "run", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--skip-learning" in result.stdout
        assert "--load-db" in result.stdout

    def test_skip_learning_requires_load_db(self):
        """--skip-learning without --load-db should error."""
        import subprocess

        result = subprocess.run(
            ["amplihack-eval", "run", "--skip-learning"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "--load-db" in result.stderr


class TestRunSkipLearning:
    """Test EvalRunner.run_skip_learning method."""

    def test_method_exists(self):
        from amplihack_eval.core.runner import EvalRunner

        runner = EvalRunner(num_turns=10, num_questions=2)
        assert hasattr(runner, "run_skip_learning")

    def test_generates_questions_without_learning(self):
        """run_skip_learning should generate questions but not feed dialogue."""
        from amplihack_eval.core.runner import EvalRunner

        runner = EvalRunner(num_turns=10, num_questions=2, seed=42)

        # Mock agent that tracks calls
        class MockAgent:
            def __init__(self):
                self.learn_calls = 0
                self.answer_calls = 0

            def learn(self, content):
                self.learn_calls += 1

            def answer(self, question):
                self.answer_calls += 1
                from amplihack_eval.adapters.base import AgentResponse

                return AgentResponse(answer="test answer")

            def reset(self):
                pass

            def close(self):
                pass

        agent = MockAgent()
        report = runner.run_skip_learning(agent, load_db_path="/tmp/fake_db")

        # Should NOT have called learn
        assert agent.learn_calls == 0
        # Should have called answer for each question
        assert agent.answer_calls == 2
        # Learning time should be 0
        assert report.learning_time_s == 0.0
        assert report.num_questions == 2
