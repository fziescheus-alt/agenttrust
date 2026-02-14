"""8-dimension quality rubric for agent output review.

Adapted from the Exec Research Factory evaluation system.
Each dimension is scored 0-2, for a maximum of 16 points.

Scoring:
    0 = Missing or fundamentally flawed
    1 = Present but incomplete or has issues
    2 = Solid, meets expectations

Pass thresholds:
    Tier 1 (quick lookups): ≥10/16
    Tier 2 (research briefs): ≥12/16
    Tier 3 (deep dives): ≥14/16
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Score(IntEnum):
    """Rubric score per dimension."""
    MISSING = 0
    PARTIAL = 1
    SOLID = 2


DIMENSIONS: list[dict[str, str]] = [
    {
        "id": "accuracy",
        "name": "Factual Accuracy",
        "description": "Claims are correct and verifiable. No hallucinations.",
        "score_0": "Contains factual errors or unverifiable claims",
        "score_1": "Mostly accurate, minor issues or unverified claims",
        "score_2": "All claims accurate and verifiable",
    },
    {
        "id": "completeness",
        "name": "Completeness",
        "description": "Covers the topic adequately. No major gaps.",
        "score_0": "Major aspects missing",
        "score_1": "Covers basics but misses important nuances",
        "score_2": "Comprehensive coverage appropriate for the task",
    },
    {
        "id": "sources",
        "name": "Source Quality",
        "description": "Sources are cited, relevant, and accessible.",
        "score_0": "No sources or irrelevant sources",
        "score_1": "Some sources but gaps in citation or relevance",
        "score_2": "Well-sourced with relevant, accessible references",
    },
    {
        "id": "clarity",
        "name": "Clarity & Structure",
        "description": "Well-organized, easy to follow, appropriate format.",
        "score_0": "Disorganized or hard to follow",
        "score_1": "Readable but could be better structured",
        "score_2": "Clear, well-structured, appropriate format",
    },
    {
        "id": "honesty",
        "name": "Epistemic Honesty",
        "description": "Distinguishes evidence from interpretation. Flags uncertainty.",
        "score_0": "Presents speculation as fact, no uncertainty flagged",
        "score_1": "Some distinction but blurs evidence and interpretation",
        "score_2": "Clear separation of evidence, interpretation, and judgment",
    },
    {
        "id": "actionability",
        "name": "Actionability",
        "description": "Output leads to clear next steps or decisions.",
        "score_0": "No actionable takeaways",
        "score_1": "Some actionable content but vague",
        "score_2": "Clear, specific, actionable recommendations",
    },
    {
        "id": "calibration",
        "name": "Confidence Calibration",
        "description": "Stated confidence matches actual quality. Beipackzettel present.",
        "score_0": "No confidence stated or wildly miscalibrated",
        "score_1": "Confidence stated but over/underconfident",
        "score_2": "Confidence well-calibrated, Beipackzettel complete",
    },
    {
        "id": "risks",
        "name": "Risk Awareness",
        "description": "Known risks, limitations, and failure modes are flagged.",
        "score_0": "No risks mentioned despite obvious ones",
        "score_1": "Some risks flagged but incomplete",
        "score_2": "Comprehensive risk awareness",
    },
]


@dataclass(frozen=True)
class RubricScore:
    """A scored rubric with per-dimension results.

    Attributes:
        scores: Dict mapping dimension id to score (0-2).
        total: Sum of all scores (0-16).
        max_total: Maximum possible score (16).
    """

    scores: dict[str, int]
    total: int
    max_total: int = 16

    def passes(self, tier: int = 2) -> bool:
        """Check if the score passes for a given tier.

        Args:
            tier: 1 (≥10), 2 (≥12), or 3 (≥14).
        """
        thresholds = {1: 10, 2: 12, 3: 14}
        return self.total >= thresholds.get(tier, 12)

    def weakest(self) -> list[str]:
        """Return dimension ids that scored 0."""
        return [dim_id for dim_id, score in self.scores.items() if score == 0]


def create_rubric_score(scores: dict[str, int]) -> RubricScore:
    """Create a RubricScore from a dimension→score mapping.

    Args:
        scores: Dict mapping dimension id to score (0-2).
            Valid ids: accuracy, completeness, sources, clarity,
            honesty, actionability, calibration, risks.

    Returns:
        RubricScore with validated scores and computed total.

    Raises:
        ValueError: If invalid dimension ids or scores outside 0-2.
    """
    valid_ids = {d["id"] for d in DIMENSIONS}
    for dim_id, score in scores.items():
        if dim_id not in valid_ids:
            raise ValueError(f"Unknown dimension: {dim_id!r}")
        if score not in (0, 1, 2):
            raise ValueError(f"Score must be 0-2, got {score} for {dim_id!r}")

    total = sum(scores.values())
    return RubricScore(scores=scores, total=total)
