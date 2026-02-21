"""Multi-agent evaluation pipeline.

Uses specialized grader, adversary, and analyst agents to produce
higher-quality evaluation results through parallel grading (multi-vote),
adversarial question generation, and systematic failure analysis.

Public API:
    EvalCoordinator: Orchestrates the multi-agent evaluation pipeline
    MultiAgentEvalPipeline: Full end-to-end pipeline with adversarial rounds
    GraderAgent: Specialized grading agent with a specific perspective
    AdversaryAgent: Generates hard questions targeting agent weaknesses
    AnalystAgent: Analyzes eval results and proposes improvements
"""

from __future__ import annotations

from .analyst_agent import AnalystAgent, AnalysisReport, ComparisonReport, Improvement
from .adversary_agent import AdversaryAgent
from .coordinator import EvalCoordinator
from .grader_agent import AggregateGrade, GraderAgent, PerspectiveGrade
from .pipeline import MultiAgentEvalPipeline

__all__ = [
    "EvalCoordinator",
    "MultiAgentEvalPipeline",
    "GraderAgent",
    "PerspectiveGrade",
    "AggregateGrade",
    "AdversaryAgent",
    "AnalystAgent",
    "AnalysisReport",
    "ComparisonReport",
    "Improvement",
]
