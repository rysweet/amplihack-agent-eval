"""Tests for scale-aware distributed eval defaults."""

from amplihack_eval.azure import eval_distributed


class TestScaleAwareDefaults:
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
