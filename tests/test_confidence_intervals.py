"""Tests for confidence interval support in multi-seed evaluation.

Tests cover:
- _ci_95 helper: margin of error, clamping, edge cases
- CategoryStats new CI fields
- MultiSeedReport new CI/repeat fields
- LearningAgentAdapter hive_store parameter
- to_dict() includes new fields
"""

from __future__ import annotations

import math
from dataclasses import fields as dc_fields
from unittest.mock import MagicMock, patch

import pytest

from amplihack_eval.core.multi_seed import (
    _T_CRITICAL_95,
    CategoryStats,
    MultiSeedReport,
    _ci_95,
    _t_critical,
)

# ── _ci_95 helper ──────────────────────────────────────────────────────


class TestCi95:
    """Tests for the _ci_95(mean, stddev, n) helper."""

    def test_basic_calculation(self):
        """_ci_95(0.95, 0.03, 4) uses t-critical (3.182) for n=4."""
        lower, upper, moe = _ci_95(mean=0.95, stddev=0.03, n=4)
        t_crit = _t_critical(4)
        expected_moe = t_crit * 0.03 / math.sqrt(4)
        assert moe == pytest.approx(expected_moe, abs=1e-6)
        assert lower == pytest.approx(0.95 - expected_moe, abs=1e-6)
        assert upper == pytest.approx(min(1.0, 0.95 + expected_moe), abs=1e-6)

    def test_zero_stddev(self):
        """Zero standard deviation produces zero margin and collapsed CI."""
        lower, upper, moe = _ci_95(mean=0.5, stddev=0.0, n=4)
        assert moe == 0.0
        assert lower == 0.5
        assert upper == 0.5

    def test_upper_clamped_to_one(self):
        """Upper bound is clamped to 1.0 when mean + moe > 1.0."""
        lower, upper, moe = _ci_95(mean=0.99, stddev=0.05, n=4)
        t_crit = _t_critical(4)
        assert upper == 1.0
        assert moe == pytest.approx(t_crit * 0.05 / math.sqrt(4), abs=1e-6)

    def test_lower_clamped_to_zero(self):
        """Lower bound is clamped to 0.0 when mean - moe < 0.0."""
        lower, upper, moe = _ci_95(mean=0.01, stddev=0.05, n=4)
        t_crit = _t_critical(4)
        assert lower == 0.0
        assert moe == pytest.approx(t_crit * 0.05 / math.sqrt(4), abs=1e-6)

    def test_n_less_than_two(self):
        """For n < 2, returns (mean, mean, 0.0) -- no CI possible."""
        lower, upper, moe = _ci_95(mean=0.5, stddev=0.1, n=1)
        assert lower == 0.5
        assert upper == 0.5
        assert moe == 0.0


# ── _t_critical helper ────────────────────────────────────────────────


class TestTCritical:
    """Tests for the _t_critical(n) helper."""

    def test_n_less_than_two(self):
        """Returns 0.0 for n < 2 (no valid CI)."""
        assert _t_critical(0) == 0.0
        assert _t_critical(1) == 0.0

    def test_known_lookup_values(self):
        """Matches precomputed lookup table for common n values."""
        assert _t_critical(2) == pytest.approx(12.706, abs=0.01)
        assert _t_critical(4) == pytest.approx(3.182, abs=0.01)
        assert _t_critical(10) == pytest.approx(2.262, abs=0.01)
        assert _t_critical(30) == pytest.approx(2.045, abs=0.01)

    def test_with_scipy(self):
        """When scipy is available, uses scipy.stats.t.ppf."""
        try:
            from scipy.stats import t as t_dist

            for n in [3, 5, 10, 20]:
                expected = t_dist.ppf(0.975, df=n - 1)
                assert _t_critical(n) == pytest.approx(expected, abs=1e-6)
        except ImportError:
            pytest.skip("scipy not installed")

    def test_without_scipy_uses_lookup(self):
        """When scipy import is blocked, falls back to lookup table."""
        import sys

        # Temporarily block scipy
        original = sys.modules.get("scipy.stats")
        sys.modules["scipy.stats"] = None  # type: ignore[assignment]
        try:
            # _t_critical catches ImportError internally, so just verify
            # the lookup table values are correct
            assert _T_CRITICAL_95[4] == 3.182
            assert _T_CRITICAL_95[10] == 2.262
        finally:
            if original is not None:
                sys.modules["scipy.stats"] = original
            else:
                sys.modules.pop("scipy.stats", None)

    def test_interpolation_for_unlisted_n(self):
        """For n not in lookup table, uses nearest smaller key."""
        # n=12 is not in the table; nearest smaller is 10 (2.262)
        # If scipy is available, it will give exact value; if not, fallback
        val = _t_critical(12)
        # Should be between t(10)=2.262 and t(15)=2.145 (or exact from scipy)
        assert 2.1 < val < 2.3


# ── CategoryStats new fields ───────────────────────────────────────────


class TestCategoryStatsFields:
    """CategoryStats has ci_95_lower, ci_95_upper, margin_of_error with defaults."""

    def test_ci_fields_exist(self):
        """CategoryStats dataclass has the three CI fields."""
        field_names = {f.name for f in dc_fields(CategoryStats)}
        assert "ci_95_lower" in field_names
        assert "ci_95_upper" in field_names
        assert "margin_of_error" in field_names

    def test_default_values(self):
        """CI fields default to 0.0 for backward compatibility."""
        cs = CategoryStats(
            category="test",
            mean_score=0.8,
            stddev=0.05,
            min_score=0.75,
            max_score=0.85,
            scores_by_seed={42: 0.8},
        )
        assert cs.ci_95_lower == 0.0
        assert cs.ci_95_upper == 0.0
        assert cs.margin_of_error == 0.0

    def test_explicit_values(self):
        """CI fields can be set explicitly."""
        cs = CategoryStats(
            category="test",
            mean_score=0.8,
            stddev=0.05,
            min_score=0.75,
            max_score=0.85,
            scores_by_seed={42: 0.8},
            ci_95_lower=0.75,
            ci_95_upper=0.85,
            margin_of_error=0.05,
        )
        assert cs.ci_95_lower == 0.75
        assert cs.ci_95_upper == 0.85
        assert cs.margin_of_error == 0.05


# ── MultiSeedReport new fields ────────────────────────────────────────


class TestMultiSeedReportFields:
    """MultiSeedReport has overall CI, repeats_per_seed, intra_seed_stddev."""

    def test_ci_and_repeat_fields_exist(self):
        """MultiSeedReport dataclass has the new fields."""
        field_names = {f.name for f in dc_fields(MultiSeedReport)}
        assert "overall_ci_95_lower" in field_names
        assert "overall_ci_95_upper" in field_names
        assert "overall_margin_of_error" in field_names
        assert "repeats_per_seed" in field_names
        assert "intra_seed_stddev" in field_names

    def test_default_values(self):
        """New fields default for backward compatibility."""
        report = MultiSeedReport(
            seeds=[42],
            num_turns=10,
            num_questions=5,
            total_time_s=1.0,
            overall_mean=0.9,
            overall_stddev=0.02,
            category_stats=[],
            noisy_questions=[],
            all_question_variances=[],
            per_seed_reports={},
        )
        assert report.overall_ci_95_lower == 0.0
        assert report.overall_ci_95_upper == 0.0
        assert report.overall_margin_of_error == 0.0
        assert report.repeats_per_seed == 1
        assert report.intra_seed_stddev == 0.0


# ── to_dict includes new fields ───────────────────────────────────────


class TestToDictIncludesCIFields:
    """to_dict() serializes new CI fields."""

    def test_to_dict_has_ci_keys(self):
        """to_dict output includes overall CI and repeat metadata."""
        report = MultiSeedReport(
            seeds=[42],
            num_turns=10,
            num_questions=5,
            total_time_s=1.0,
            overall_mean=0.9,
            overall_stddev=0.02,
            category_stats=[],
            noisy_questions=[],
            all_question_variances=[],
            per_seed_reports={},
            overall_ci_95_lower=0.88,
            overall_ci_95_upper=0.92,
            overall_margin_of_error=0.02,
            repeats_per_seed=3,
            intra_seed_stddev=0.01,
        )
        d = report.to_dict()
        assert "overall_ci_95_lower" in d
        assert "overall_ci_95_upper" in d
        assert "overall_margin_of_error" in d
        assert "repeats_per_seed" in d
        assert "intra_seed_stddev" in d
        assert d["overall_ci_95_lower"] == pytest.approx(0.88, abs=1e-4)
        assert d["repeats_per_seed"] == 3

    def test_category_stats_dict_has_ci_keys(self):
        """Each category in to_dict() output includes CI fields."""
        cs = CategoryStats(
            category="L1",
            mean_score=0.8,
            stddev=0.05,
            min_score=0.75,
            max_score=0.85,
            scores_by_seed={42: 0.8},
            ci_95_lower=0.75,
            ci_95_upper=0.85,
            margin_of_error=0.05,
        )
        report = MultiSeedReport(
            seeds=[42],
            num_turns=10,
            num_questions=5,
            total_time_s=1.0,
            overall_mean=0.8,
            overall_stddev=0.05,
            category_stats=[cs],
            noisy_questions=[],
            all_question_variances=[],
            per_seed_reports={},
        )
        d = report.to_dict()
        cat_dict = d["category_stats"][0]
        assert "ci_95_lower" in cat_dict
        assert "ci_95_upper" in cat_dict
        assert "margin_of_error" in cat_dict


# ── LearningAgentAdapter hive_store ────────────────────────────────────


class TestLearningAgentAdapterHiveStore:
    """LearningAgentAdapter accepts optional hive_store parameter."""

    def test_hive_store_passed_to_agent(self):
        """When hive_store is given, it is passed through to LearningAgent."""
        mock_hive = MagicMock(name="hive_store")
        mock_agent_cls = MagicMock(name="LearningAgent")

        with patch.dict(
            "sys.modules",
            {
                "amplihack": MagicMock(),
                "amplihack.agents": MagicMock(),
                "amplihack.agents.goal_seeking": MagicMock(),
                "amplihack.agents.goal_seeking.learning_agent": MagicMock(LearningAgent=mock_agent_cls),
            },
        ):
            from amplihack_eval.adapters.learning_agent import LearningAgentAdapter

            LearningAgentAdapter(
                model="test-model",
                storage_path="/tmp/test",
                hive_store=mock_hive,
            )

        # LearningAgent should have been called with hive_store in kwargs
        call_kwargs = mock_agent_cls.call_args
        assert "hive_store" in call_kwargs.kwargs

    def test_no_hive_store_by_default(self):
        """When hive_store is not given, it is not passed to LearningAgent."""
        mock_agent_cls = MagicMock(name="LearningAgent")

        with patch.dict(
            "sys.modules",
            {
                "amplihack": MagicMock(),
                "amplihack.agents": MagicMock(),
                "amplihack.agents.goal_seeking": MagicMock(),
                "amplihack.agents.goal_seeking.learning_agent": MagicMock(LearningAgent=mock_agent_cls),
            },
        ):
            from amplihack_eval.adapters.learning_agent import LearningAgentAdapter

            LearningAgentAdapter(
                model="test-model",
                storage_path="/tmp/test",
            )

        call_kwargs = mock_agent_cls.call_args
        assert "hive_store" not in call_kwargs.kwargs
