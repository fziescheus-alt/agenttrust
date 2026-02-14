"""Adversarial reviewer: score agent outputs against the 8-dimension rubric.

The reviewer's job is to find errors, not confirm quality. It operates
as an adversarial QA agent: "Find the error. Attack the output. Score it."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agenttrust.qa.rubric import DIMENSIONS, RubricScore, create_rubric_score


@dataclass
class ReviewResult:
    """Complete review of an agent output.

    Attributes:
        rubric_score: The scored rubric (0-16).
        issues: List of specific issues found.
        verdict: PASS, REVISE, or FAIL.
        tier: The tier threshold used (1, 2, or 3).
        notes: Free-form reviewer notes.
    """

    rubric_score: RubricScore
    issues: list[str] = field(default_factory=list)
    verdict: str = ""
    tier: int = 2
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.verdict:
            if self.rubric_score.passes(self.tier):
                self.verdict = "PASS"
            elif self.rubric_score.total >= (self.tier * 4):
                self.verdict = "REVISE"
            else:
                self.verdict = "FAIL"


def review(
    output: str,
    score_fn: Callable[[str, dict[str, str]], int] | None = None,
    tier: int = 2,
) -> ReviewResult:
    """Review an agent output against the 8-dimension rubric.

    If no ``score_fn`` is provided, uses a simple heuristic scorer.
    For production use, pass a function that calls an LLM to score each
    dimension (see ``integrations.openai_provider`` for helpers).

    Args:
        output: The agent output text to review.
        score_fn: Optional callable(output, dimension_dict) → int (0-2).
            If None, uses a basic heuristic that checks for source citations,
            confidence statements, and structural markers.
        tier: Quality tier (1=quick, 2=standard, 3=deep). Affects pass threshold.

    Returns:
        ReviewResult with rubric scores, issues, and verdict.
    """
    if score_fn is None:
        score_fn = _heuristic_scorer

    scores: dict[str, int] = {}
    issues: list[str] = []

    for dim in DIMENSIONS:
        dim_score = score_fn(output, dim)
        dim_score = max(0, min(2, dim_score))
        scores[dim["id"]] = dim_score
        if dim_score == 0:
            issues.append(f"{dim['name']}: {dim['score_0']}")

    rubric = create_rubric_score(scores)
    return ReviewResult(rubric_score=rubric, issues=issues, tier=tier)


def _heuristic_scorer(output: str, dimension: dict[str, str]) -> int:
    """Simple heuristic scorer for demo/testing purposes.

    Not a replacement for LLM-based review. Just checks for basic signals.
    """
    text = output.lower()
    dim_id = dimension["id"]

    if dim_id == "accuracy":
        # Can't verify accuracy heuristically — give benefit of doubt
        return 1

    elif dim_id == "completeness":
        if len(output) < 50:
            return 0
        elif len(output) < 200:
            return 1
        return 2

    elif dim_id == "sources":
        source_signals = ["http", "arxiv", "doi", "source:", "reference"]
        hits = sum(1 for s in source_signals if s in text)
        if hits >= 2:
            return 2
        elif hits >= 1:
            return 1
        return 0

    elif dim_id == "clarity":
        structure_signals = ["\n\n", "##", "- ", "1.", "key takeaway"]
        hits = sum(1 for s in structure_signals if s in text)
        return min(2, hits)

    elif dim_id == "honesty":
        honesty_signals = ["uncertain", "might", "unclear", "not sure", "assumption"]
        hits = sum(1 for s in honesty_signals if s in text)
        return min(2, hits)

    elif dim_id == "actionability":
        action_signals = ["recommend", "next step", "should", "action", "todo"]
        hits = sum(1 for s in action_signals if s in text)
        return min(2, hits)

    elif dim_id == "calibration":
        if "confidence:" in text or "% confident" in text:
            return 2 if "beipackzettel" in text else 1
        return 0

    elif dim_id == "risks":
        risk_signals = ["risk", "caveat", "limitation", "warning", "might fail"]
        hits = sum(1 for s in risk_signals if s in text)
        return min(2, hits)

    return 1  # default: partial
