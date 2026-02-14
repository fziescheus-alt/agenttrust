"""Trust scores for AI agents: track calibration over time.

An agent's trust score reflects how well-calibrated its confidence is.
Honest uncertainty is rewarded. Overconfidence is penalized. Over time,
this produces a "credit score" for AI agents.

Scoring rules (from ARCHITECTURE.md):
    - Agent says 85% confident + output was good → +1
    - Agent says 95% confident + output was bad → -3
    - Agent flags uncertainty + it was real → +2
    - Agent hides problem QA finds → -3

Trust levels:
    0-30:  QA reviews everything
    31-60: QA reviews flagged items only
    61-80: QA spot-checks 20%
    81+:   Direct delivery
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TrustLevel(Enum):
    """Autonomy level derived from cumulative trust score."""

    UNTRUSTED = "untrusted"       # 0-30: QA reviews everything
    SUPERVISED = "supervised"     # 31-60: QA reviews flagged items
    SPOT_CHECK = "spot_check"     # 61-80: QA spot-checks 20%
    AUTONOMOUS = "autonomous"     # 81+: direct delivery


@dataclass(frozen=True)
class TrustEvent:
    """A single trust-relevant event in an agent's history.

    Attributes:
        timestamp: Unix timestamp of the event.
        stated_confidence: What the agent claimed (0-100).
        outcome: Whether the output was good, bad, or flagged uncertain.
        delta: Points added or removed from trust score.
        reason: Human-readable explanation.
    """

    timestamp: float
    stated_confidence: float
    outcome: Literal["good", "bad", "flagged_real", "hidden_problem"]
    delta: int
    reason: str


class TrustScore:
    """Tracks an agent's trust score over time.

    The score starts at 0 and evolves based on the agent's calibration
    accuracy. Well-calibrated agents earn trust and autonomy. Overconfident
    agents lose trust and get more QA oversight.

    Args:
        agent_id: Unique identifier for the agent.
        initial_score: Starting score (default 0).
        min_score: Floor for the score (default 0).
        max_score: Ceiling for the score (default 100).

    Example::

        >>> ts = TrustScore("writer-agent")
        >>> ts.update(stated_confidence=85, outcome="good")
        >>> ts.score
        1
        >>> ts.trust_level
        <TrustLevel.UNTRUSTED: 'untrusted'>
        >>> ts.update(stated_confidence=60, outcome="flagged_real")
        >>> ts.score
        3
    """

    def __init__(
        self,
        agent_id: str,
        initial_score: int = 0,
        min_score: int = 0,
        max_score: int = 100,
    ) -> None:
        self.agent_id = agent_id
        self._score = initial_score
        self._min = min_score
        self._max = max_score
        self._history: list[TrustEvent] = []

    @property
    def score(self) -> int:
        """Current trust score."""
        return self._score

    @property
    def trust_level(self) -> TrustLevel:
        """Current autonomy level based on score."""
        if self._score <= 30:
            return TrustLevel.UNTRUSTED
        elif self._score <= 60:
            return TrustLevel.SUPERVISED
        elif self._score <= 80:
            return TrustLevel.SPOT_CHECK
        else:
            return TrustLevel.AUTONOMOUS

    @property
    def history(self) -> list[TrustEvent]:
        """Full history of trust events."""
        return list(self._history)

    def update(
        self,
        stated_confidence: float,
        outcome: Literal["good", "bad", "flagged_real", "hidden_problem"],
        reason: str = "",
        timestamp: float | None = None,
    ) -> TrustEvent:
        """Record a trust event and update the score.

        Scoring rules:
            - ``good`` + high confidence (≥70) → +1
            - ``good`` + low confidence (<70) → +1
            - ``bad`` + high confidence (≥80) → -3 (overconfident and wrong)
            - ``bad`` + low confidence (<80) → -1 (wrong but honest)
            - ``flagged_real`` → +2 (flagged uncertainty that was real)
            - ``hidden_problem`` → -3 (QA found something agent hid)

        Args:
            stated_confidence: What the agent claimed (0-100).
            outcome: The actual outcome category.
            reason: Optional explanation.
            timestamp: Optional unix timestamp (defaults to now).

        Returns:
            The TrustEvent that was recorded.
        """
        ts = timestamp or time.time()

        if outcome == "good":
            delta = 1
            if not reason:
                reason = f"Good output (stated {stated_confidence:.0f}%)"
        elif outcome == "bad":
            if stated_confidence >= 80:
                delta = -3
                if not reason:
                    reason = f"Bad output with high confidence ({stated_confidence:.0f}%) — overconfident"
            else:
                delta = -1
                if not reason:
                    reason = f"Bad output with low confidence ({stated_confidence:.0f}%) — at least honest"
        elif outcome == "flagged_real":
            delta = 2
            if not reason:
                reason = "Flagged uncertainty that was confirmed real"
        elif outcome == "hidden_problem":
            delta = -3
            if not reason:
                reason = "QA found a problem the agent didn't flag"
        else:
            raise ValueError(f"Unknown outcome: {outcome!r}")

        self._score = max(self._min, min(self._max, self._score + delta))

        event = TrustEvent(
            timestamp=ts,
            stated_confidence=stated_confidence,
            outcome=outcome,
            delta=delta,
            reason=reason,
        )
        self._history.append(event)
        return event

    def needs_qa(self) -> bool:
        """Whether this agent's outputs need QA review."""
        return self.trust_level in (TrustLevel.UNTRUSTED, TrustLevel.SUPERVISED)

    def qa_sample_rate(self) -> float:
        """What fraction of outputs should be QA-reviewed.

        Returns:
            1.0 for UNTRUSTED, 0.5 for SUPERVISED, 0.2 for SPOT_CHECK,
            0.0 for AUTONOMOUS.
        """
        return {
            TrustLevel.UNTRUSTED: 1.0,
            TrustLevel.SUPERVISED: 0.5,
            TrustLevel.SPOT_CHECK: 0.2,
            TrustLevel.AUTONOMOUS: 0.0,
        }[self.trust_level]

    def summary(self) -> dict[str, object]:
        """Return a summary dict suitable for logging or display."""
        return {
            "agent_id": self.agent_id,
            "score": self._score,
            "trust_level": self.trust_level.value,
            "total_events": len(self._history),
            "needs_qa": self.needs_qa(),
            "qa_sample_rate": self.qa_sample_rate(),
        }

    def __repr__(self) -> str:
        return (
            f"TrustScore(agent_id={self.agent_id!r}, "
            f"score={self._score}, "
            f"level={self.trust_level.value!r})"
        )
