"""Self-improvement infrastructure for agent evaluation.

Analyzes eval failures, generates hypotheses about root causes,
proposes code/prompt patches, validates via reviewer voting,
and gates promotion through regression checks.

Philosophy:
- Measure -> Analyze -> Hypothesize -> Patch -> Challenge -> Vote -> Validate -> Promote
- Never modify grader, test data, or safety constraints
- All changes go through review (never direct application)
- Focus on prompt templates first (safest), code changes second
- Every patch must survive challenge and 3-reviewer vote
"""

from __future__ import annotations

from .patch_proposer import PatchHistory, PatchProposal, propose_patch
from .reviewer_voting import (
    ChallengeResponse,
    ReviewResult,
    ReviewVote,
    challenge_proposal,
    review_result_to_dict,
    vote_on_proposal,
)

__all__ = [
    # patch_proposer
    "PatchProposal",
    "PatchHistory",
    "propose_patch",
    # reviewer_voting
    "ReviewVote",
    "ChallengeResponse",
    "ReviewResult",
    "challenge_proposal",
    "vote_on_proposal",
    "review_result_to_dict",
]
