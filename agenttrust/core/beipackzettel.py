"""Beipackzettel: mandatory metadata for every agent output.

German for "package insert" — like the safety information that comes with
medicine. Every AI agent output should ship with one: what's the confidence,
what sources were used, what's uncertain, and what could go wrong.

This is the "nutrition label for AI" that turns opaque outputs into
accountable ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Beipackzettel:
    """Mandatory metadata attached to every agent output.

    Attributes:
        confidence: Overall confidence percentage (0-100). Must be calibrated,
            not just self-reported. Use ``sample_consistency`` or
            ``verbalized_confidence`` to compute this.
        sources: List of sources consulted (files, URLs, searches, APIs).
            Empty list means "pure generation with no grounding" — a red flag.
        uncertainties: Specific things the agent is uncertain about.
            "I don't know what I don't know" doesn't count. Be specific.
        risks: Known risks, failure modes, or corrections that might apply.
            Think: what could go wrong if someone acts on this output?
        not_checked: Things that were assumed but not verified.
        model: Which model produced the output (e.g., "claude-sonnet-4-20250514").
        agent_id: Which agent produced the output.
        metadata: Additional key-value pairs for domain-specific info.

    Example::

        >>> bpz = Beipackzettel(
        ...     confidence=72.0,
        ...     sources=["https://arxiv.org/abs/2506.04133"],
        ...     uncertainties=["Publication date not independently verified"],
        ...     risks=["Paper may have been updated since last check"],
        ... )
        >>> bpz.is_grounded
        True
        >>> bpz.risk_level
        'medium'
    """

    confidence: float
    sources: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    not_checked: list[str] = field(default_factory=list)
    model: str = ""
    agent_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence range."""
        if not 0 <= self.confidence <= 100:
            raise ValueError(f"Confidence must be 0-100, got {self.confidence}")

    @property
    def is_grounded(self) -> bool:
        """Whether the output has at least one source."""
        return len(self.sources) > 0

    @property
    def risk_level(self) -> str:
        """Qualitative risk level based on confidence and flags.

        Returns:
            'low' if confidence ≥80 and no risks flagged.
            'high' if confidence <50 or ≥3 risks.
            'medium' otherwise.
        """
        if self.confidence < 50 or len(self.risks) >= 3:
            return "high"
        elif self.confidence >= 80 and len(self.risks) == 0:
            return "low"
        else:
            return "medium"

    @property
    def has_gaps(self) -> bool:
        """Whether there are known unknowns (uncertainties or unchecked items)."""
        return len(self.uncertainties) > 0 or len(self.not_checked) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON/logging."""
        return {
            "confidence": self.confidence,
            "sources": self.sources,
            "uncertainties": self.uncertainties,
            "risks": self.risks,
            "not_checked": self.not_checked,
            "model": self.model,
            "agent_id": self.agent_id,
            "risk_level": self.risk_level,
            "is_grounded": self.is_grounded,
            **self.metadata,
        }

    def __str__(self) -> str:
        """Human-readable Beipackzettel."""
        lines = [
            f"📋 Beipackzettel",
            f"   Confidence: {self.confidence:.0f}%",
            f"   Sources: {', '.join(self.sources) or 'none (ungrounded)'}",
        ]
        if self.uncertainties:
            lines.append(f"   Uncertain: {'; '.join(self.uncertainties)}")
        if self.risks:
            lines.append(f"   Risks: {'; '.join(self.risks)}")
        if self.not_checked:
            lines.append(f"   Not checked: {'; '.join(self.not_checked)}")
        lines.append(f"   Risk level: {self.risk_level}")
        return "\n".join(lines)
